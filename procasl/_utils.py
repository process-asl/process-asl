import os
import glob
import warnings

import numpy as np
import nibabel


def _single_glob(pattern):
    filenames = glob.glob(pattern)
    if not filenames:
        print('Warning: non exitant file with pattern {}'.format(pattern))
        return None

    if len(filenames) > 1:
        raise ValueError('Non unique file with pattern {}'.format(pattern))

    return filenames[0]


def _list_to_4d(input_files):
    """Form a 4D data from a list of 3d images.
    """
    data = []
    for f in input_files:
        image = nibabel.load(f)
        data.append(image.get_data())
    data = np.array(data)
    data = np.transpose(data, (1, 2, 3, 0))


def check_images(file1, file2):
    """Check that 2 images have the same affines and data shapes.
    """
    img = nibabel.load(file1)
    shape1 = np.shape(img.get_data())
    affine1 = img.get_affine()
    img = nibabel.load(file2)
    shape2 = np.shape(img.get_data())
    affine2 = img.get_affine()
    if shape1 != shape2:
        raise ValueError('{0} of shape {1}, {2} of shape {3}'.format(
            file1, shape1, file2, shape2))

    if np.any(affine1 != affine2):
        raise ValueError('affine for {0}: {1}, for {2}: {3}'
                         .format(file1, affine1, file2, affine2))


def get_vox_dims(in_file):
    if isinstance(in_file, list):
        in_file = in_file[0]
    img = nibabel.load(in_file)
    header = img.get_header()
    voxdims = header.get_zooms()
    return [float(voxdims[0]), float(voxdims[1]), float(voxdims[2])]


def threshold(in_file, threshold_min=-1e7, threshold_max=1e7):
    img = nibabel.load(in_file)
    data = img.get_data()
    data[data > threshold_max] = threshold_max
    data[data < threshold_min] = threshold_min
    img = nibabel.Nifti1Image(data, img.get_affine(), img.get_header())
    out_file, _ = os.path.splitext(in_file)
    out_file += '_thresholded.nii'
    if os.path.isfile(out_file):
        warnings.warn('File {} exits, overwriting.'.format(out_file))

    nibabel.save(img, out_file)
    return out_file


def remove_nan(in_file, fill_value=0.):
    img = nibabel.load(in_file)
    data = img.get_data()
    if np.any(np.isnan(data)):
        data[np.isnan(data)] = fill_value
    img = nibabel.Nifti1Image(data, img.get_affine(), img.get_header())
    out_file, _ = os.path.splitext(in_file)
    out_file += '_no_nan.nii'
    nibabel.save(img, out_file)
    return out_file
