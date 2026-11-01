"""Convert representative-point CSVs to NIfTI label images and optional previews."""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from tqdm import tqdm

from extract_points import read_rules


COLORS = ('#e6b422', '#777777', '#e65555', '#55bbee', '#aa66cc', '#44bb88', '#ff88bb')


def points_to_labels(csv_path, image, classes, coordinates):
    if len(image.shape) != 3:
        raise ValueError('Reference image must be 3D')
    with csv_path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if not {'x', 'y', 'z', 'class'} <= set(reader.fieldnames or ()):
            raise ValueError('Expected CSV columns: x,y,z,class')
        rows = list(reader)
    xyz = np.array([[float(row[k]) for k in ('x', 'y', 'z')] for row in rows], dtype=float).reshape(-1, 3)
    if not np.isfinite(xyz).all():
        raise ValueError('Coordinates must be finite')
    if coordinates == 'world':
        xyz = nib.affines.apply_affine(np.linalg.inv(image.affine), xyz)
    rounded = np.rint(xyz)
    if not np.allclose(xyz, rounded, atol=1e-3, rtol=0):
        raise ValueError('Points do not lie on voxel centers; check --coordinates')
    if np.any(rounded < 0) or np.any(rounded >= np.array(image.shape)):
        raise ValueError('Point outside reference image; check image and --coordinates')
    voxels = rounded.astype(np.int64)
    if len(np.unique(voxels, axis=0)) != len(voxels):
        raise ValueError('Multiple points map to the same voxel')
    unknown = {row['class'] for row in rows} - set(classes)
    if unknown:
        raise ValueError(f'Unknown classes: {sorted(unknown)}')
    ids = np.array([classes[row['class']] for row in rows], dtype=np.uint8)
    volume = np.zeros(image.shape, dtype=np.uint8)
    if len(voxels):
        volume[tuple(voxels.T)] = ids
    header = image.header.copy()
    header.set_data_dtype(np.uint8)
    header.set_intent('label')
    result = image.__class__(volume, image.affine, header=header)
    # Preserve both coordinate transforms, including unknown form codes.
    for form in ('qform', 'sform'):
        transform, code = getattr(image, f'get_{form}')(coded=True)
        getattr(result, f'set_{form}')(transform, int(code))
    result.header.set_slope_inter(1, 0)
    result.header['cal_min'] = 0
    result.header['cal_max'] = len(classes)
    return result, voxels, ids


def save_preview(path, image, voxels, ids, classes):
    os.environ.setdefault('MPLCONFIGDIR', str(Path('.mplconfig').resolve()))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    world = nib.affines.apply_affine(image.affine, voxels)
    figure = plt.figure(figsize=(10, 8))
    axis = figure.add_subplot(111, projection='3d')
    for (name, label), color in zip(classes.items(), COLORS):
        points = world[ids == label]
        axis.scatter(*points.T, s=6, alpha=.7, color=color,
                     label=f'{label}: {name} ({len(points)})')
    corners = np.array(np.meshgrid(*[(0, n - 1) for n in image.shape])).reshape(3, -1).T
    extent = nib.affines.apply_affine(image.affine, corners)
    lower, upper = extent.min(axis=0), extent.max(axis=0)
    axis.set_xlim(lower[0], upper[0])
    axis.set_ylim(lower[1], upper[1])
    axis.set_zlim(lower[2], upper[2])
    axis.set_box_aspect(np.maximum(upper - lower, 1))
    axis.set(xlabel='R / x (header units)', ylabel='A / y (header units)',
             zlabel='S / z (header units)', title=path.stem)
    axis.legend(loc='upper left', fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=Path('outputs/points'))
    root = Path('data') if Path('data/image').is_dir() else Path('data/test')
    parser.add_argument('--image-dir', type=Path, default=root / 'image')
    parser.add_argument('--output-dir', type=Path, default=Path('data/points'))
    parser.add_argument('--rules', type=Path, default=Path('.csv'))
    parser.add_argument('--coordinates', choices=('voxel', 'world'), default='voxel',
                        help='Must match the coordinate format used to extract the CSV')
    parser.add_argument('--case', action='append')
    parser.add_argument('--preview', action='store_true', help='Also save a 3D scatter PNG')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--overwrite', action='store_true')
    mode.add_argument('--skip-existing', action='store_true')
    args = parser.parse_args()
    try:
        classes = {name: number for number, name in enumerate(read_rules(args.rules), 1)}
        files = sorted(args.input_dir.glob('*.csv'))
        if args.case:
            requested = set(args.case)
            files = [path for path in files if path.stem in requested]
            missing = requested - {path.stem for path in files}
            if missing:
                raise ValueError(f'Cases not found: {sorted(missing)}')
        if not files:
            raise ValueError(f'No CSV files in {args.input_dir}')
        jobs = []
        for path in files:
            image = args.image_dir / f'{path.stem}.nii.gz'
            if not image.is_file():
                image = args.image_dir / f'{path.stem}.nii'
            if not image.is_file():
                raise ValueError(f'Missing reference image for {path.stem}')
            output = args.output_dir / f'{path.stem}.nii.gz'
            preview = args.output_dir / f'{path.stem}.png'
            if output.resolve() == image.resolve():
                raise ValueError('Output would overwrite reference image')
            if args.skip_existing and output.exists() and (not args.preview or preview.exists()):
                continue
            if not args.overwrite and not args.skip_existing and (output.exists() or (args.preview and preview.exists())):
                raise ValueError(f'Output exists for {path.stem}; use --overwrite or --skip-existing')
            jobs.append((path, image, output, preview))
        mapping = {'background': 0, 'classes': classes}
        mapping_path = args.output_dir / 'class_mapping.json'
        if mapping_path.exists() and json.loads(mapping_path.read_text(encoding='utf-8')) != mapping and not args.overwrite:
            raise ValueError('Existing class_mapping.json differs; use a new output directory')
        args.output_dir.mkdir(parents=True, exist_ok=True)
        if not mapping_path.exists() or args.overwrite:
            mapping_path.write_text(json.dumps(mapping, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        failed = 0
        for source, reference, output, preview in tqdm(jobs, desc='Label images', unit='case'):
            try:
                image = nib.load(reference)
                labels, voxels, ids = points_to_labels(source, image, classes, args.coordinates)
                if args.overwrite or not output.exists():
                    nib.save(labels, output)
                if args.preview and (args.overwrite or not preview.exists()):
                    save_preview(preview, image, voxels, ids, classes)
            except (ValueError, OSError) as error:
                tqdm.write(f'ERROR {source.name}: {error}', file=sys.stderr)
                failed += 1
        print(f'{len(jobs) - failed} converted, {failed} failed -> {args.output_dir}')
        return int(failed > 0)
    except (ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    sys.exit(main())
