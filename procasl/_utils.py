import os
import warnings

import numpy as np
import nibabel


def check_images(file1, file2):
    """Check that 2 images have the same affines and data shapes.

    Parameters
    ----------
    file1 : str
        Path to the first nifti image

    file2 : str
        Path to the second nifti image
    """
    img = nibabel.load(file1)
    shape1 = np.shape(img.get_data())
    affine1 = img.get_affine()
    img = nibabel.load(file2)
    shape2 = np.shape(img.get_data())
    affine2 = img.get_affine()
    if shape1 != shape2:
        raise ValueError('Images got different shapes: {0} of shape {1}, {2} '
                         'of shape {3}'.format(file1, shape1, file2, shape2))

    if np.any(affine1 != affine2):
        raise ValueError('Images got different affines: {0} has affine {1}, '
                         '{2} has affine {3}'.format(file1, affine1,
                                                     file2, affine2))


def get_voxels_dim(in_file):
    """Return the voxels resolution of a nifti image.

    Parameters
    ----------
    in_file : str
        Path to the nifti image

    Returns
    -------
    list of 3 float
        Resolutions
    """
    img = nibabel.load(in_file)
    header = img.get_header()
    voxdims = header.get_zooms()
    return [float(voxdims[0]), float(voxdims[1]), float(voxdims[2])]


def threshold(in_file, threshold_min=-1e7, threshold_max=1e7, out_file=None):
    """Put to thresholds values outside given thresholds

    Parameters
    ----------
    in_file : str
        Path to the nifti image

    threshold_min : float or None, optional
        Values less than this threshold are set to it.

    threshold_max : float or None, optional
        Values greater than this threshold are set to it.

    out_file : str or None, optional
        Path to the thresholded image

    Returns
    -------
    out_file : str
        Path to the thresholded image
    """
    img = nibabel.load(in_file)
    data = img.get_data()
    if threshold_max is not None:
        data[data > threshold_max] = threshold_max

    if threshold_min is not None:
        data[data < threshold_min] = threshold_min

    img = nibabel.Nifti1Image(data, img.get_affine(), img.get_header())
    if out_file is None:
        out_file, _ = os.path.splitext(in_file)
        out_file += '_thresholded.nii'

    if os.path.isfile(out_file):
        warnings.warn('File {0} exits, overwriting.'.format(out_file))

    nibabel.save(img, out_file)
    return out_file


def fill_nan(in_file, fill_value=0.):
    """Replace nan values with a given value

    Parameters
    ----------
    in_file : str
        Path to image file

    fill_value : float, optional
        Value replacing nan

    Returns
    -------
    out_file : str
        Path to the filled file
    """
    img = nibabel.load(in_file)
    data = img.get_data()
    if np.any(np.isnan(data)):
        data[np.isnan(data)] = fill_value
    img = nibabel.Nifti1Image(data, img.get_affine(), img.get_header())
    out_file, _ = os.path.splitext(in_file)
    out_file += '_no_nan.nii'
    nibabel.save(img, out_file)
    return out_file
