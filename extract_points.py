"""Extract weighted representative points from paired CT/segmentation NIfTI files."""

import argparse
import contextlib
import csv
import hashlib
import io
import multiprocessing
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi
from tqdm import tqdm


CLASSES = ("fat", "bone", "muscle", "lung gas", "digestive gas",
           "digestive system", "unknown_anatomical")
CONNECTIVITY = ndi.generate_binary_structure(3, 1)


def read_rules(path):
    """Read one rule per line; tolerate the source CSV's unclosed quote."""
    rules = {}
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines()[1:], 2):
        if not line.strip():
            continue
        parts = line.split(",", 2)
        if len(parts) != 3:
            raise ValueError(f"{path}:{number}: expected class, weight, method")
        name, weight, method = (part.strip().strip('"') for part in parts)
        if name not in CLASSES or name in rules or not method:
            raise ValueError(f"{path}:{number}: unknown/duplicate class or empty method")
        weight = float(weight)
        if not np.isfinite(weight) or weight <= 0:
            raise ValueError(f"{path}:{number}: weight must be finite and positive")
        rules[name] = weight
    if set(rules) != set(CLASSES):
        raise ValueError(f"Rules must contain exactly: {', '.join(CLASSES)}")
    return rules


def read_label_groups(path):
    names = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        match = re.match(r'^\s*(\d+)\s+.*?"(.*)$', line)
        if match:
            # Definitions such as "vertebrae_L"5 deliberately have suffixes.
            names[int(match[1])] = match[2].replace('"', '').strip()
    groups = {
        "bone": [i for i, n in names.items() if n.startswith((
            "vertebrae_", "rib_", "humerus_", "scapula_", "clavicula_",
            "femur_", "hip_")) or n in {"sacrum", "skull", "sternum"}],
        "muscle": [i for i, n in names.items() if n.startswith((
            "gluteus_", "autochthon_", "iliopsoas_"))],
        "lung": [i for i, n in names.items() if n.startswith("lung_")],
        "digestive": [i for i, n in names.items() if n in {
            "stomach", "small_bowel", "duodenum", "colon"}],
    }
    if any(not ids for ids in groups.values()):
        raise ValueError("Label definition is missing an anatomical group")
    return names, groups


def interior_distance(mask, spacing=(1, 1, 1)):
    # Padding makes the volume edge an explicit background boundary.
    return ndi.distance_transform_edt(np.pad(mask, 1), sampling=spacing)[1:-1, 1:-1, 1:-1]


def fat_candidates(ct):
    components, count = ndi.label((ct >= -190) & (ct <= -30), CONNECTIVITY)
    if count == 0:
        return np.zeros(ct.shape, bool), None
    sizes = np.bincount(components.ravel())
    sizes[0] = 0
    region = components == sizes.argmax()
    del components
    holes = ndi.binary_fill_holes(region, structure=CONNECTIVITY) & ~region
    components, _ = ndi.label(holes, CONNECTIVITY)
    sizes = np.bincount(components.ravel())
    small = sizes <= 10
    small[0] = False
    region |= small[components]
    distance = interior_distance(region)
    candidates = region & (ct >= -150) & (ct <= -50)
    indices = np.flatnonzero(candidates)
    # Stable ranking removes exactly floor(N/2), including tied distances.
    order = np.argsort(distance.ravel()[indices], kind="stable")
    candidates.ravel()[indices[order[:len(indices) // 2]]] = False
    return candidates, distance


def build_candidates(ct, labels, groups, spacing):
    fat, distance = fat_candidates(ct)
    masks = {"fat": fat}
    for name in ("bone", "muscle"):
        region = np.isin(labels, groups[name])
        masks[name] = region & (interior_distance(region, spacing) >= 2.0)
    digestive = np.isin(labels, groups["digestive"])
    masks["lung gas"] = np.isin(labels, groups["lung"]) & (ct <= -900)
    masks["digestive gas"] = digestive & (ct <= -900)
    masks["digestive system"] = digestive & (ct >= -500)
    # Priority follows the CSV, so every voxel can appear in only one class.
    used = np.zeros(ct.shape, bool)
    for name in CLASSES[:-1]:
        masks[name] &= ~used
        used |= masks[name]
    masks["unknown_anatomical"] = (labels != 0) & ~used
    return masks, distance


def allocate_counts(total, weights, capacities):
    """Exclude classes with fewer than 3 candidates per allocated point."""
    active = capacities > 0
    while active.any():
        shares = total * weights * active / weights[active].sum()
        counts = np.floor(shares).astype(np.int64)
        remainder = total - int(counts.sum())
        order = np.argsort(-(shares - counts), kind="stable")
        order = order[active[order]]
        counts[order[:remainder]] += 1
        insufficient = active & (capacities < 3 * counts)
        if not insufficient.any():
            return counts
        active[insufficient] = False
    raise ValueError(f"No eligible classes: each class needs at least 3 candidates per allocated point ({total} requested)")


def load_pair(image_path, label_path, known_ids):
    image, segmentation = nib.load(image_path), nib.load(label_path)
    if len(image.shape) != 3 or image.shape != segmentation.shape:
        raise ValueError("Image and label must be matching 3D volumes")
    # Compare corner displacement: tolerate small NIfTI header rounding only.
    corners = np.array(np.meshgrid(*[(0, n - 1) for n in image.shape])).reshape(3, -1).T
    displacement = nib.affines.apply_affine(image.affine, corners) - nib.affines.apply_affine(segmentation.affine, corners)
    if np.max(np.linalg.norm(displacement, axis=1)) > 0.1:
        raise ValueError("Image/label grids differ by more than 0.1 mm; align them first")
    unit = image.header.get_xyzt_units()[0]
    if unit not in ("mm", "unknown"):
        raise ValueError(f"Expected mm units, got {unit}")
    if unit == "unknown":
        print(f"{image_path.name}: NIfTI spatial unit unknown; assuming mm", file=sys.stderr)
    axes = image.affine[:3, :3]
    spacing = np.linalg.norm(axes, axis=0)
    if np.any(spacing <= 0) or not np.allclose((axes / spacing).T @ (axes / spacing), np.eye(3), atol=1e-3):
        raise ValueError("Distance calculation requires an orthogonal, nondegenerate grid")
    ct = image.get_fdata(dtype=np.float32)
    raw_labels = np.asarray(segmentation.dataobj)
    if not np.isfinite(ct).all() or not np.isfinite(raw_labels).all():
        raise ValueError("Nonfinite image/label values")
    if not np.equal(raw_labels, np.floor(raw_labels)).all():
        raise ValueError("Labels must be integers")
    unknown = set(np.unique(raw_labels)) - {0} - set(known_ids)
    if unknown:
        raise ValueError(f"Undefined label IDs: {sorted(unknown)}")
    return ct, raw_labels, spacing, image.affine


class WeightTree:
    """Segment tree for direct inverse-CDF draws and local weight updates."""

    def __init__(self, weights):
        self.size = 1 << (max(1, len(weights)) - 1).bit_length()
        self.values = np.zeros(2 * self.size, dtype=np.float64)
        self.values[self.size:self.size + len(weights)] = weights
        width = self.size // 2
        while width:
            self.values[width:2 * width] = (self.values[2 * width:4 * width:2]
                                          + self.values[2 * width + 1:4 * width:2])
            width //= 2

    def draw(self, uniform):
        total = self.values[1]
        if total <= 0:
            raise ValueError("No positive sampling weights remain")
        target = min(uniform * total, np.nextafter(total, 0))
        node = 1
        while node < self.size:
            node *= 2
            if target >= self.values[node]:
                target -= self.values[node]
                node += 1
        return node - self.size

    def update(self, positions, weights):
        nodes = positions + self.size
        self.values[nodes] = weights
        nodes = np.unique(nodes // 2)
        while len(nodes) and nodes[-1] > 0:
            self.values[nodes] = self.values[2 * nodes] + self.values[2 * nodes + 1]
            nodes = np.unique(nodes // 2)


def distance_kernel(affine, scale):
    """Voxel offsets inside the physical radius, including oblique grids."""
    axes = affine[:3, :3]
    extent = np.ceil(scale * np.linalg.norm(np.linalg.inv(axes), axis=1)).astype(int)
    offsets = np.stack(np.meshgrid(*(np.arange(-n, n + 1) for n in extent),
                                  indexing="ij"), axis=-1).reshape(-1, 3)
    distances = np.linalg.norm(offsets @ axes.T, axis=1)
    inside = distances < scale
    return offsets[inside], distances[inside] / scale


class LocalWeightSampler:
    def __init__(self, indices, shape, affine, scale, factors, weights=None, kernel=None):
        self.indices = indices  # Sorted flat indices, as returned by flatnonzero.
        self.shape = shape
        self.factors = factors  # Shared across classes; initialized to ones.
        self.base = np.ones(len(indices)) if weights is None else np.asarray(weights, dtype=np.float64)
        if self.base.shape != indices.shape or not np.isfinite(self.base).all() or np.any(self.base <= 0):
            raise ValueError("Base weights must be finite and positive")
        self.base = self.base / self.base.max()
        self.tree = WeightTree(self.base * factors[indices])
        self.offsets, self.kernel_factors = distance_kernel(affine, scale) if kernel is None else kernel

    def draw(self, rng):
        position = self.tree.draw(rng.random())
        selected = self.indices[position]
        center = np.array(np.unravel_index(selected, self.shape))
        neighbors = center + self.offsets
        inside = ((neighbors >= 0) & (neighbors < self.shape)).all(axis=1)
        flat = np.ravel_multi_index(neighbors[inside].T, self.shape)
        updated = np.minimum(self.factors[flat], self.kernel_factors[inside])
        changed = updated < self.factors[flat]
        flat, updated = flat[changed], updated[changed]
        self.factors[flat] = updated
        # Update only affected candidates of the currently sampled class.
        positions = np.searchsorted(self.indices, flat)
        valid = positions < len(self.indices)
        positions, flat = positions[valid], flat[valid]
        valid = self.indices[positions] == flat
        positions, flat = positions[valid], flat[valid]
        self.tree.update(positions, self.base[positions] * self.factors[flat])
        return selected


def extract_case(image_path, label_path, output_path, rules, names, groups, args):
    ct, labels, spacing, affine = load_pair(image_path, label_path, names)
    masks, distance = build_candidates(ct, labels, groups, spacing)
    classes = list(rules)
    capacities = np.array([np.count_nonzero(masks[name]) for name in classes])
    counts = allocate_counts(args.num_points, np.array(list(rules.values())), capacities)
    case_seed = int.from_bytes(hashlib.sha256(image_path.name.encode()).digest()[:8], "little")
    rng = np.random.default_rng([args.seed, case_seed])
    rows = []
    factors = np.ones(ct.size, dtype=np.float64)
    kernel = distance_kernel(affine, args.distance_scale_mm)
    for name, count, capacity in zip(classes, counts, capacities):
        if not count:
            continue
        indices = np.flatnonzero(masks[name])
        weights = None
        if name == "fat":
            weights = np.square(distance.ravel()[indices])
        try:
            sampler = LocalWeightSampler(indices, ct.shape, affine, args.distance_scale_mm,
                                         factors, weights, kernel)
            selected = np.array([sampler.draw(rng) for _ in range(int(count))])
            del sampler
        except ValueError as error:
            raise ValueError(f"{name}: {error}") from error
        coordinates = np.column_stack(np.unravel_index(selected, ct.shape))
        if args.coordinates == "world":
            coordinates = nib.affines.apply_affine(affine, coordinates)
        rows.extend((*xyz, name) for xyz in coordinates.tolist())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents accidental loss of previous results.
    with output_path.open("w" if args.overwrite else "x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("x", "y", "z", "class"))
        writer.writerows(rows)
    print(f"{image_path.name}: {len(rows)} points -> {output_path}")
    for name, count, capacity in zip(classes, counts, capacities):
        print(f"  {name}: {count} / {capacity} candidates")


def run_case(job, rules, names, groups, args):
    """Return worker logs to the parent to keep the progress display intact."""
    log = io.StringIO()
    failed = False
    with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        try:
            extract_case(*job, rules, names, groups, args)
        except Exception as error:
            print(f"ERROR {job[0].name}: {type(error).__name__}: {error}")
            failed = True
    return failed, log.getvalue().rstrip()


def process_jobs(jobs, rules, names, groups, args):
    workers = min(args.workers, len(jobs))
    failed = 0
    with tqdm(total=len(jobs), desc="Extracting", unit="case") as progress:
        def report(result):
            nonlocal failed
            case_failed, log = result
            failed += int(case_failed)
            if log:
                tqdm.write(log, file=sys.stderr)
            progress.set_postfix(failed=failed, refresh=False)
            progress.update()

        if workers == 1:
            for job in jobs:
                report(run_case(job, rules, names, groups, args))
        else:
            with ProcessPoolExecutor(max_workers=workers,
                                     mp_context=multiprocessing.get_context("spawn")) as executor:
                futures = {executor.submit(run_case, job, rules, names, groups, args): job
                           for job in jobs}
                for future in as_completed(futures):
                    try:
                        result = future.result()
                    except Exception as error:
                        image = futures[future][0]
                        result = True, f"ERROR {image.name}: {type(error).__name__}: {error}"
                    report(result)
    return int(failed > 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    default_root = Path("data") if Path("data/image").is_dir() else Path("data/test")
    parser.add_argument("--image-dir", type=Path, default=default_root / "image")
    parser.add_argument("--label-dir", type=Path, default=default_root / "label")
    parser.add_argument("--rules", type=Path, default=Path(".csv"))
    parser.add_argument("--label-definition", type=Path, default=Path("ttsg_label_definition.label"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/points"))
    parser.add_argument("--num-points", type=int, required=True, help="Total points per case")
    parser.add_argument("--case", action="append", help="Case ID, e.g. 1139 (repeatable)")
    parser.add_argument("--coordinates", choices=("voxel", "world"), default="voxel")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--distance-scale-mm", "--min-distance-mm", dest="distance_scale_mm",
                        type=float, default=5.0,
                        help="Distance where the sampling factor reaches 1, in mm (default: 5; no hard minimum)")
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1),
                        help="Parallel case processes (default: up to 4; use 1 for sequential)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.num_points <= 0 or args.seed < 0:
        parser.error("--num-points must be positive and --seed must be nonnegative")
    if args.workers <= 0:
        parser.error("--workers must be positive")
    if not np.isfinite(args.distance_scale_mm) or args.distance_scale_mm <= 0:
        parser.error("--distance-scale-mm must be finite and positive")
    try:
        rules = read_rules(args.rules)
        names, groups = read_label_groups(args.label_definition)
        images = sorted(args.image_dir.glob("*.nii.gz")) + sorted(args.image_dir.glob("*.nii"))
        if args.case:
            requested = set(args.case)
            images = [p for p in images if p.name.removesuffix(".gz").removesuffix(".nii") in requested]
            found = {p.name.removesuffix(".gz").removesuffix(".nii") for p in images}
            if requested - found:
                raise ValueError(f"Cases not found: {sorted(requested - found)}")
        if not images:
            raise ValueError(f"No NIfTI images in {args.image_dir}")
        jobs = []
        for image in images:
            case = image.name.removesuffix(".gz").removesuffix(".nii")
            label = args.label_dir / image.name
            output = args.output_dir / f"{case}.csv"
            if not label.is_file():
                raise ValueError(f"Missing label: {label}")
            if output.exists() and not args.overwrite:
                raise ValueError(f"Output exists: {output}; use --overwrite")
            if any(output.resolve() == p.resolve() for p in (image, label, args.rules, args.label_definition)):
                raise ValueError("Output would overwrite an input file")
            jobs.append((image, label, output))
        if len({output.resolve() for _, _, output in jobs}) != len(jobs):
            raise ValueError("Duplicate case IDs in input images")
        return process_jobs(jobs, rules, names, groups, args)
    except (ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    sys.exit(main())
