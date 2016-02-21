import numpy as np
from numpy.testing import assert_array_equal
from nose.tools import assert_raises
import nibabel

from procasl import _utils


def test_check_images():
    data = np.zeros((91, 91, 60))
    affine = np.eye(4)
    affine[:, 3] = np.ones(4)
    file1 = '/tmp/file1.nii'
    file2 = '/tmp/file2.nii'
    file3 = '/tmp/file3.nii'
    nibabel.Nifti1Image(data, affine).to_filename(file1)
    nibabel.Nifti1Image(data, affine * 3).to_filename(file2)
    nibabel.Nifti1Image(data + 1, affine * 3).to_filename(file3)
    _utils.check_images(file1, file1)
    assert_raises(ValueError, _utils.check_images, file1, file2)
    assert_raises(ValueError, _utils.check_images, file1, file3)


def test_get_voxel_dims():
    data = np.zeros((91, 91, 60))
    affine = 3 * np.eye(4)
    affine[:, 3] = np.ones(4)
    in_file = '/tmp/file1.nii'
    nibabel.Nifti1Image(data, affine).to_filename(in_file)
    assert_array_equal(_utils.get_voxel_dims(in_file), [3., 3., 3.])


def test_threshold():
    data = np.zeros((5, 5, 6))
    data[1, 4, 1] = 1.
    data[2, 3, 2] = -1.
    affine = 3 * np.eye(4)
    affine[:, 3] = np.ones(4)
    in_file = '/tmp/in_file.nii'
    nibabel.Nifti1Image(data, affine).to_filename(in_file)
    threshold_min = -.1
    threshold_max = .1
    out_file =_utils.threshold(in_file, threshold_min=threshold_min,
                               threshold_max=threshold_max)
    data[1, 4, 1] = threshold_max
    data[2, 3, 2] = threshold_min
    assert_array_equal(nibabel.load(out_file).get_data(), data)
    assert_array_equal(nibabel.load(out_file).get_affine(), affine)


def test_fill_nan():
    data = np.zeros((5, 5, 6))
    data[1, 4, 1] = np.nan
    data[2, 3, 2] = np.nan
    affine = 3 * np.eye(4)
    affine[:, 3] = np.ones(4)
    in_file = '/tmp/in_file.nii'
    nibabel.Nifti1Image(data, affine).to_filename(in_file)
    fill_value = 1.
    out_file = _utils.fill_nan(in_file, fill_value=fill_value)
    data[1, 4, 1] = fill_value
    data[2, 3, 2] = fill_value
    assert_array_equal(nibabel.load(out_file).get_data(), data)
