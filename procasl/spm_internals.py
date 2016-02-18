import numpy as np
from math import pi, sin, cos, asin, atan2
from scipy import linalg


def rotation_matrix(angle):
    """Returns (2, 2) rotation matrix for a given angle."""
    rotation = np.array([[cos(angle), sin(angle)],
                         [-sin(angle), cos(angle)]])
    return rotation


def insure_trigo(x):
    return np.minimum(np.maximum(x, -1), 1)


def params_to_affine(params):
    """Transforms parameters to an affine matrix transform.
    Mimics the spm function spm_matrix.

    Parameters
    ----------
    params : 1D numpy.ndarray of size 6 or 12
        Parameters of the transform, in the following order:
        Tx, Ty, Tz, pitch, roll, yaw
        and possibly 3 zooms and 3 shears

    Returns
    -------
    affine : numpy.ndarray of shape (4, 4)
        The affine transformation matrix.
    """
    # Check shape
    if params.shape == (6, ):
        params = np.hstack((params, [1., 1., 1., 0., 0., 0.]))

    if params.shape != (12, ):
        raise ValueError('Expected 1D array of size 12, got array '
                         'of shape {}'.format(params.shape))

    # Compute the (3, 3) rotation matrix
    pitch = np.eye(3)
    roll = np.eye(3)
    yaw = np.eye(3)
    pitch[1:, 1:] = rotation_matrix(params[3])
    roll[np.ix_([0, 2], [0, 2])] = rotation_matrix(params[4])
    yaw[:2, :2] = rotation_matrix(params[5])
    rotation = pitch.dot(roll).dot(yaw)

    # Compute the  (3, 3) zooms matrix
    zooms = np.diag(params[6:9])

    # Compute the  (3, 3) shears matrix
    shears = np.eye(3)
    shears[np.triu_indices(3, 1)] = params[9:]

    # Form the (4, 4) affine transform
    affine = np.eye(4)
    affine[:3, :3] = rotation.dot(zooms).dot(shears)
    affine[:3, 3] = params[:3]
    return affine


def affine_to_params(affine):
    """Transforms a (4, 4) affine matrix transform to an 1D array of 12
    parameters. Mimics the spm function spm_imatrix.

    Parameters
    ----------
    affine : numpy.ndarray of shape (4, 4)
        The affine transformation matrix.

    Returns
    -------
    params : numpy.ndarray of shape (12, )
        Parameters of the transform, in the following order:
        Tx, Ty, Tz, pitch, roll, yaw
        and possibly 3 zooms and 3 shears
    """
    if affine.shape != (4, 4):
        raise ValueError('Expects a (4, 4) array, got {}'.format(
            affine.shape))

    params = np.zeros((12,))

    # tranlations
    params[:3] = affine[:3, 3]

    # zooms
    rotation = affine[:3, :3]
    chol = linalg.cholesky(rotation.T.dot(rotation))
    params[6:9] = np.diag(chol)
    if np.linalg.det(rotation) < 0:
        params[6] *= -1

    # shears
    shears = np.linalg.inv(np.diag(np.diag(chol))).dot(chol)
    params[9:12] = shears[np.triu_indices(3, 1)]

    # rotations
    r = params_to_affine(np.hstack((np.zeros(6,), params[6:])))
    r = r[:3, :3]
    rotation = rotation.dot(np.linalg.inv(r))
    params[4] = asin(insure_trigo(rotation[0, 2]))
    if (abs(params[4]) - pi / 2.) ** 2 < 1e-9:
        params[3] = 0.
        params[5] = atan2(-insure_trigo(rotation[1, 0]),
                          insure_trigo((-rotation[2, 0] / rotation[0, 2])))
    else:
        c = cos(params[4])
        params[3] = atan2(insure_trigo(rotation[1, 2] / c),
                          insure_trigo(rotation[2, 2] / c))
        params[5] = atan2(insure_trigo(rotation[0, 1] / c),
                          insure_trigo(rotation[0, 0] / c))
    return params


def spm_affine(in_file):
    """Returns the affine transform of a nifti image.
    Mimics the spm function spm_get_space.

    Parameters
    ----------
    in_file : str
        Path to an existant nifti image.

    Returns
    -------
    affine : numpy.ndarray of shape (4, 4)
        The affine transformation matrix.

    Notes
    -----
    This function uses nibabel to read the affine transform and corrects the
    translation part to match the affine output of spm.
    """
    import nibabel
    img = nibabel.load(in_file)
    affine = img.get_affine()

    # Compute zooms
    rotation = affine[:3, :3]
    chol = linalg.cholesky(rotation.T.dot(rotation))
    zooms = np.diag(chol).copy()
    if np.linalg.det(rotation) < 0:
        zooms[0] *= -1

    affine[:3, 3] = affine[:3, 3] - zooms * np.ones((3, ))
    return affine
