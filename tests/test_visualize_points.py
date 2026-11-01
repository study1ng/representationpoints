import csv
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

from visualize_points import points_to_labels


class LabelImageTests(unittest.TestCase):
    def test_voxel_and_world_roundtrip_preserves_geometry(self):
        affine = np.array([[0, -2, 0, 10], [1.5, 0, 0, -5], [0, 0, 3, 20], [0, 0, 0, 1]])
        image = nib.Nifti1Image(np.zeros((4, 5, 6)), affine)
        image.header.set_slope_inter(2, -1000)
        classes = {'fat': 1, 'bone': 2, 'muscle': 3}
        coords = np.array([[1, 2, 3], [2, 3, 4]])
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            for mode in ('voxel', 'world'):
                path = Path(directory) / 'points.csv'
                values = coords if mode == 'voxel' else nib.affines.apply_affine(affine, coords)
                with path.open('w', newline='') as stream:
                    writer = csv.writer(stream)
                    writer.writerow(['x', 'y', 'z', 'class'])
                    for xyz, name in zip(values, ('fat', 'muscle')):
                        writer.writerow([*xyz, name])
                labels, voxels, ids = points_to_labels(path, image, classes, mode)
                output = Path(directory) / 'labels.nii.gz'
                nib.save(labels, output)
                loaded = nib.load(output)
                np.testing.assert_array_equal(voxels, coords)
                np.testing.assert_array_equal(ids, [1, 3])
                np.testing.assert_allclose(loaded.affine, affine)
                volume = loaded.get_fdata()
                self.assertEqual(volume[1, 2, 3], 1)
                self.assertEqual(volume[2, 3, 4], 3)
                self.assertEqual(np.count_nonzero(volume), 2)
                self.assertEqual(loaded.header.get_intent()[0], 'label')

    def test_invalid_points_are_rejected(self):
        image = nib.Nifti1Image(np.zeros((3, 3, 3)), np.eye(4))
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            path = Path(directory) / 'points.csv'
            for body in ('3,0,0,fat\n', '.5,0,0,fat\n', '0,0,0,invalid\n',
                         '0,0,0,fat\n0,0,0,fat\n', 'nan,0,0,fat\n'):
                path.write_text('x,y,z,class\n' + body)
                with self.assertRaises(ValueError):
                    points_to_labels(path, image, {'fat': 1}, 'voxel')
