import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

from extract_points import allocate_counts, build_candidates, fat_candidates, interior_distance, read_label_groups, read_rules, LocalWeightSampler, WeightTree


class ExtractionTests(unittest.TestCase):
    def test_parallel_cli_matches_sequential_and_reports_failures(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=root) as directory:
            work = Path(directory)
            for name in ('image', 'label'):
                (work / name).mkdir()
            for case in ('a', 'b', 'empty'):
                ct = np.zeros((40, 40, 40), dtype=np.float32)
                if case != 'empty':
                    ct[2:38, 2:38, 2:38] = -100
                image = nib.Nifti1Image(ct, np.eye(4))
                image.header.set_xyzt_units('mm')
                nib.save(image, work / 'image' / f'{case}.nii.gz')
                nib.save(nib.Nifti1Image(np.zeros(ct.shape, dtype=np.int16), np.eye(4)),
                         work / 'label' / f'{case}.nii.gz')
            command = [sys.executable, str(root / 'extract_points.py'),
                       '--image-dir', str(work / 'image'), '--label-dir', str(work / 'label'),
                       '--num-points', '20']
            for workers in (1, 2):
                result = subprocess.run(command + ['--workers', str(workers), '--output-dir',
                                                  str(work / str(workers))],
                                        cwd=root, capture_output=True, text=True, timeout=60)
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertIn('ERROR empty.nii.gz', result.stderr)
                self.assertIn('3/3', result.stderr)
                self.assertIn('failed=1', result.stderr)
                self.assertFalse((work / str(workers) / 'empty.csv').exists())
            for case in ('a', 'b'):
                self.assertEqual((work / '1' / f'{case}.csv').read_bytes(),
                                 (work / '2' / f'{case}.csv').read_bytes())

    def test_weight_allocation_and_class_exclusion(self):
        weights = np.array([2, 2, 2, 1, 1, 10, 2], dtype=float)
        np.testing.assert_array_equal(allocate_counts(200, weights, np.full(7, 1000)), weights * 10)
        counts = allocate_counts(200, weights, np.array([0, 1, 1000, 1000, 1000, 1000, 1000]))
        self.assertEqual(counts.sum(), 200)
        self.assertEqual(counts[0], 0)
        self.assertEqual(counts[1], 0)
        with self.assertRaises(ValueError):
            allocate_counts(8, np.ones(2), np.array([3, 4]))

    def test_threefold_boundary_and_cascading_exclusion(self):
        np.testing.assert_array_equal(allocate_counts(10, np.ones(2), np.array([15, 15])), [5, 5])
        np.testing.assert_array_equal(allocate_counts(10, np.ones(2), np.array([14, 30])), [0, 10])
        # Initial 10/10/10 excludes class 1; 15/15 then excludes class 2.
        np.testing.assert_array_equal(allocate_counts(30, np.ones(3), np.array([29, 44, 90])), [0, 0, 30])
        with self.assertRaises(ValueError):
            allocate_counts(10, np.ones(2), np.array([14, 29]))

    def test_tree_inverse_cdf_and_local_updates(self):
        tree = WeightTree(np.array([0., 2., 0., 3., 1.]))
        self.assertEqual(tree.draw(0), 1)
        self.assertEqual(tree.draw(.5), 3)
        self.assertEqual(tree.draw(.99), 4)
        tree.update(np.array([1, 4]), np.array([0., 0.]))
        self.assertEqual(tree.values[1], 3)
        self.assertEqual(tree.draw(0), 3)
        tree.update(np.array([3]), np.array([0.]))
        with self.assertRaises(ValueError):
            tree.draw(.5)

    def test_local_map_matches_full_recomputation_after_every_point(self):
        shape = (9, 8, 7)
        affine = np.array([[0, -2, 0, -20], [1.5, 0, 0, 10], [0, 0, 3, 0], [0, 0, 0, 1]])
        indices = np.arange(np.prod(shape))
        factors = np.ones(len(indices))
        world = nib.affines.apply_affine(affine, np.column_stack(np.unravel_index(indices, shape)))
        expected = factors.copy()
        base = np.linspace(1, 3, len(indices))
        rng = np.random.default_rng(42)
        chosen = []
        # Start a second class after 10 points to check the shared distance map.
        for _ in range(2):
            sampler = LocalWeightSampler(indices, shape, affine, 5, factors, base)
            for _ in range(10):
                selected = sampler.draw(rng)
                chosen.append(selected)
                expected = np.minimum(expected, np.minimum(np.linalg.norm(world - world[selected], axis=1) / 5, 1))
                np.testing.assert_allclose(factors, expected, atol=1e-14)
                leaves = sampler.tree.values[sampler.tree.size:sampler.tree.size + len(indices)]
                np.testing.assert_allclose(leaves, base / base.max() * expected, atol=1e-14)
                self.assertAlmostEqual(sampler.tree.values[1], leaves.sum())
        self.assertEqual(len(set(chosen)), 20)

    def test_soft_distance_sampling_and_weight_multiplication(self):
        rng = np.random.default_rng(42)
        # Existing point x=0; candidates x=1 and x=5 have factors .2 and 1.
        for base, expected in ((None, 1 / 6), (np.array([10., 1.]), 2 / 3)):
            near = 0
            for _ in range(600):
                factors = np.array([0., .2, .4, .6, .8, 1.])
                sampler = LocalWeightSampler(np.array([1, 5]), (6, 1, 1), np.eye(4),
                                             5, factors, weights=base)
                near += int(sampler.draw(rng) == 1)
            self.assertAlmostEqual(near / 600, expected, delta=.06)
        # Dense points remain selectable; existing points across calls are not repeated.
        affine = np.array([[0, -2, 0, -20], [1.5, 0, 0, 10], [0, 0, 3, 0], [0, 0, 0, 1]])
        factors = np.ones(8)
        indices = np.arange(8)
        chosen = []
        for _ in range(2):
            sampler = LocalWeightSampler(indices, (2, 2, 2), affine, 5, factors)
            chosen.extend(sampler.draw(rng) for _ in range(4))
        self.assertEqual(len(set(chosen)), 8)

    def test_physical_distance_and_volume_edge(self):
        mask = np.ones((5, 5, 5), bool)
        distance = interior_distance(mask, (1, 2, 3))
        self.assertEqual(distance[0, 2, 2], 1)
        self.assertEqual(distance[2, 2, 2], 3)

    def test_fat_largest_component_small_holes_and_half(self):
        ct = np.zeros((15, 15, 15), dtype=np.float32)
        ct[2:11, 2:11, 2:11] = -100
        ct[5, 5, 5] = 0
        ct[13, 13, 13] = -100
        mask, distance = fat_candidates(ct)
        self.assertFalse(mask[13, 13, 13])
        self.assertGreater(distance[5, 5, 5], 1)
        self.assertEqual(mask.sum(), (9 ** 3 - 1) // 2)

    def test_rules_and_anatomy_from_actual_definitions(self):
        rules = read_rules(Path('.csv'))
        self.assertEqual(sum(rules.values()), 20)
        names, groups = read_label_groups(Path('ttsg_label_definition.label'))
        self.assertEqual(names[27], 'vertebrae_L5')
        self.assertIn(27, groups['bone'])
        self.assertIn(115, groups['bone'])
        self.assertNotIn(117, groups['bone'])
        self.assertNotIn(15, groups['digestive'])
        self.assertNotIn(5, groups['digestive'])

    def test_gas_thresholds_and_unknown_complement(self):
        ct = np.zeros((5, 5, 5), dtype=np.float32)
        labels = np.zeros(ct.shape, dtype=int)
        labels[1, 1, 1], ct[1, 1, 1] = 10, -900
        labels[2, 2, 2], ct[2, 2, 2] = 6, -900
        labels[3, 3, 3], ct[3, 3, 3] = 6, -500
        labels[1, 2, 3], ct[1, 2, 3] = 6, -700
        _, groups = read_label_groups(Path('ttsg_label_definition.label'))
        masks, _ = build_candidates(ct, labels, groups, (1, 1, 1))
        self.assertTrue(masks['lung gas'][1, 1, 1])
        self.assertTrue(masks['digestive gas'][2, 2, 2])
        self.assertTrue(masks['digestive system'][3, 3, 3])
        self.assertTrue(masks['unknown_anatomical'][1, 2, 3])
        self.assertEqual(sum(m.sum() for m in masks.values()), 4)


if __name__ == '__main__':
    unittest.main()
