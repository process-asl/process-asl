from math import pi, cos, sin

import numpy as np
from scipy.io import loadmat
import nipype.interfaces.matlab as matlab

from procasl.spm_internals import (params_to_affine, affine_to_params,
                                   spm_affine)


def spm_matrix(params):
    """Run the spm function spm_matrix through Matlab.

    Parameters
    ==========
    params : 1D numpy.ndarray, shape > 6
        Parameters to convert to the affine transfomation matrix.

    Returns
    =======
    affine : numpy.ndarray of shape (4, 4)
        The affine transformation matrix.
    """
    params_str = '[{0}, {1}, {2}, {3}, {4}, {5}]'.format(*params)
    mat_path = '/tmp/matrix.mat'

    # Generate the Matlab script
    mlab = matlab.MatlabCommand()
    mlab.inputs.script = "\naddpath '/i2bm/local/spm8-6313/' \n" + \
                         "matrix = spm_matrix(" + params_str + ");\n" + \
                         "save('" + mat_path + "', 'matrix')"

    # Run the script and load the output matrix
    mlab.run()
    mat_dict = loadmat(mat_path)
    affine = mat_dict['matrix']
    return affine


def spm_imatrix(affine):
    """Run the spm function spm_imatrix through Matlab.

    Parameters
    ==========
    affine : numpy.ndarray of shape (4, 4)
        The affine transformation matrix.

    Returns
    =======
    params : 1D numpy.ndarray
        The parameters of the affine transformation.
    """
    affine_str = "["
    for row in affine:
        affine_str += '{0}, {1}, {2}, {3};'.format(*row)

    affine_str = affine_str[:-1]
    affine_str += "]"

    # Generate the Matlab script
    mat_path = '/tmp/params.mat'
    mlab = matlab.MatlabCommand()
    mlab.inputs.script = "\naddpath '/i2bm/local/spm8-6313/' \n" +\
                         "parameters = spm_imatrix(" + affine_str + ");\n" +\
                         "save('" + mat_path + "', 'parameters')"

    # Run the script and load the output vector
    mlab.run()
    mat_dict = loadmat(mat_path)
    params = mat_dict['parameters']
    params = np.squeeze(params)
    return params


def spm_get_space(in_file):
    """Run the spm function spm_get_space through Matlab.

    Parameters
    ==========
    in_file : str
        Path to an existant nifti image.

    Returns
    =======
    affine : numpy.ndarray of shape (4, 4)
        The affine transformation matrix.
    """
    mat_path = '/tmp/space.mat'
    mlab = matlab.MatlabCommand()
    mlab.inputs.script = "\naddpath '/i2bm/local/spm8-6313/' \n" + \
                         "matrix = spm_get_space(" + in_file + ");\n" + \
                         "save('" + mat_path + "', 'matrix')"
    mlab.run()
    mat_dict = loadmat(mat_path)
    affine = mat_dict['matrix']
    return affine


def test_params_to_affine():
    # TODO: test with zooms and shears
    eps = np.finfo(np.float).eps
    params = np.array([.1, .5, -.6, .1, -4., 5.1,
                       1., 1., 1., 0., 0., 0.])
    affine = np.eye(4)
    affine[:3, 3] = params[:3]
    pitch = np.array([[1., 0., 0.],
                      [0., cos(.1), sin(.1)],
                      [0., -sin(.1), cos(.1)]])
    roll = np.array([[cos(-4.), 0., sin(-4.)],
                     [0., 1., 0.],
                     [-sin(-4.), 0., cos(-4.)]])
    yaw = np.array([[cos(5.1), sin(5.1), 0.],
                    [-sin(5.1), cos(5.1), 0.],
                    [0., 0., 1.]])
    affine[:3, :3] = pitch.dot(roll).dot(yaw)
    np.testing.assert_allclose(params_to_affine(params), affine)

    # Test params_to_affine is the inverse of affine_to_params
    affine2 = params_to_affine(affine_to_params(affine))
    np.testing.assert_allclose(affine2, affine)

    # Test same result as spm_matrix function of spm
    np.testing.assert_allclose(affine, spm_matrix(params), atol=eps)


def test_affine_to_params():
    # TODO: test with zooms and shears
    eps = np.finfo(np.float).eps
    affine = np.eye(4)
    affine[:3, 3] = [1.1, 0.55, -.3]
    pitch = np.array([[1., 0., 0.],
                      [0., 0., 1.],
                      [0., -1., 0.]])
    roll = np.array([[cos(-pi / 3.), 0., sin(-pi / 3.)],
                     [0., 1., 0.],
                     [-sin(-pi / 3.), 0., cos(-pi / 3.)]])
    yaw = np.array([[cos(pi / 4.), sin(pi / 4.), 0.],
                    [-sin(pi / 4.), cos(pi / 4.), 0.],
                    [0., 0., 1.]])
    affine[:3, :3] = pitch.dot(roll).dot(yaw)
    params = np.array([1.1, 0.55, -.3, pi / 2., -pi / 3., pi / 4.,
                       1., 1., 1., 0., 0., 0.])
    np.testing.assert_allclose(affine_to_params(affine), params, atol=eps)

    # Test affine_to_params is the inverse of params_to_affine
    params2 = affine_to_params(params_to_affine(params))
    np.testing.assert_allclose(params2, params, atol=eps)

    # Test same result as spm_matrix function of spm
    np.testing.assert_allclose(params, spm_imatrix(affine), atol=1e4 * eps)


def test_spm_affine():
    pass
#    in_file = '/tmp/img.nii'
#    np.testing.assert_allclose(spm_affine(in_file), spm_get_space(in_file))
