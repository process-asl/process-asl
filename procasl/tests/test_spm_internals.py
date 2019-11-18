import os
from nose import with_setup
from math import pi, cos, sin, sqrt

import numpy as np
from scipy.io import loadmat
import nibabel
import nipype.interfaces.matlab as matlab
from nipype.interfaces.base import TraitedSpec, isdefined, File, traits, OutputMultiPath, InputMultiPath
from nipype.interfaces.spm.base import SPMCommandInputSpec, SPMCommand, scans_for_fnames, scans_for_fname
from nilearn.datasets.tests import test_utils as tst

from procasl import spm_internals


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
    # TODO write using _make_matlab_command to rely only on SPM
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


class GetSpaceInputSpec(SPMCommandInputSpec):
    in_file = File(
        exists=True, mandatory=True,
        desc="target for reading the voxel-to-world"
    )
    mat_file = File(mandatory=True,
                    desc="Matlab file to save the transform to")


class GetSpaceOutputSpec(TraitedSpec):
    mat_file = File(exists=True, desc="Matlab file holding transform")


class GetSpace(SPMCommand):
    """ Uses SPM (spm_get_space) to read the affine transform and save it
    to a matlab file

    Examples
    --------
    >>> import nipype.interfaces.spm.utils as spmu
    >>> get_space = spmu.GetSpace(matlab_cmd='matlab-spm8')
    >>> get_space.inputs.in_file = 'structural.nii'
    >>> get_space.inputs.mat = 'func_to_struct.mat'
    >>> get_space.run() # doctest: +SKIP
    """

    input_spec = GetSpaceInputSpec
    output_spec = GetSpaceOutputSpec

    def _make_matlab_command(self, _):
        """checks for SPM, generates script"""

        script = """
        matrix = spm_get_space('%s');
        save('%s' , 'matrix' );
        """ % (
            self.inputs.in_file,
            self.inputs.mat_file,
        )
        return script

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["mat_file"] = os.path.abspath(self.inputs.mat_file)
        return outputs


@with_setup(tst.setup_tmpdata, tst.teardown_tmpdata)
def test_spm_affine():
    data = np.zeros((3, 6, 3))
    affine = np.array([[-1.7, -0.9, -1.2, 16.2 ],
                       [-0.9, -3.1, -2.3, 4.9],
                       [-0.2, 1.8, 3.7, -2.1],
                       [0., 0., 0., 1]])
    img = nibabel.Nifti1Image(data, affine=affine)
    in_file = os.path.join(tst.tmpdir, 'anat.nii')
    mat_file = os.path.join(tst.tmpdir, 'anat_mapping.mat')
    img.to_filename(in_file)
    get_space = GetSpace().run
    out_get_space = get_space(in_file=in_file, mat_file=mat_file)
    mat_dict = loadmat(out_get_space.outputs.mat_file)
    affine_from_mat = mat_dict['matrix']
    np.testing.assert_allclose(spm_internals.spm_affine(in_file),
                               affine_from_mat)


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
    np.testing.assert_allclose(spm_internals.params_to_affine(params), affine)

    # Test params_to_affine is the inverse of affine_to_params
    affine2 = spm_internals.params_to_affine(
        spm_internals.affine_to_params(affine))
    np.testing.assert_allclose(affine2, affine)

    # Test same result as spm_matrix function of spm
#    np.testing.assert_allclose(affine, spm_matrix(params), atol=eps)


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
    np.testing.assert_allclose(spm_internals.affine_to_params(affine), params,
                               atol=eps)

    # Test affine_to_params is the inverse of params_to_affine
    params2 = spm_internals.affine_to_params(
        spm_internals.params_to_affine(params))
    np.testing.assert_allclose(params2, params, atol=eps)

    # Test same result as spm_matrix function of spm
#    np.testing.assert_allclose(params, spm_imatrix(affine), atol=1e4 * eps)


def test_spm_affine():
    pass
#    in_file = '/tmp/img.nii'
#    np.testing.assert_allclose(spm_affine(in_file), spm_get_space(in_file))
