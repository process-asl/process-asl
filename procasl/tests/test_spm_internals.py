import os
from nose import with_setup
from nose.tools import assert_true
from math import pi, cos, sin, sqrt

import numpy as np
from scipy.io import loadmat
import nibabel
import nipype.interfaces.matlab as matlab
from nipype.interfaces.base import TraitedSpec, isdefined, File, traits, OutputMultiPath, InputMultiPath
from nipype.interfaces.spm.base import SPMCommandInputSpec, SPMCommand, scans_for_fnames, scans_for_fname
from nilearn.datasets.tests import test_utils as tst

from procasl import spm_internals


class SPMMatrixInputSpec(SPMCommandInputSpec):
    params = traits.Either(
        traits.List(traits.Float, minlen=6, maxlen=6),
        traits.List(traits.Float, minlen=12, maxlen=12),
        desc="Parameters of the transform, in the following order:"
             "Tx, Ty, Tz, pitch, roll, yaw"
             "and possibly 3 zooms and 3 shears")
    mat_file = File(mandatory=True,
                    desc="Matlab file to save the transform to")


class SPMMatrixOutputSpec(TraitedSpec):
    mat_file = File(exists=True, desc="Matlab file holding transform")


class SPMMatrix(SPMCommand):
    """ Uses SPM (spm_matrix) to convert 6 or 12 parameters to an affine
    transfomation matrix and save it to a matlab file

    Examples
    --------
    >>> import nipype.interfaces.spm.utils as spmu
    >>> spm_matrix = spmu.SPMMatrix(matlab_cmd='matlab-spm8')
    >>> spm_matrix.inputs.params = [12, 3., 4., 0.1, 0.02, -.1]
    >>> spm_matrix.inputs.mat = 'rigid.mat'
    >>> spm_matrix.run() # doctest: +SKIP
    """

    input_spec = SPMMatrixInputSpec
    output_spec = SPMMatrixOutputSpec

    def _make_matlab_command(self, _):
        """checks for SPM, generates script"""

        script = """
        matrix = spm_matrix([ %s ]);
        save('%s' , 'matrix' );
        """ % (
            ', '.join(map(str, self.inputs.params)),
            self.inputs.mat_file,
        )
        return script

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["mat_file"] = os.path.abspath(self.inputs.mat_file)
        return outputs


@with_setup(tst.setup_tmpdata, tst.teardown_tmpdata)
def test_spm_matrix():
    mat_file = os.path.join(tst.tmpdir, 'anat_mapping.mat')
    spm_matrix = SPMMatrix().run

    # test with 6 parameters
    out_spm_matrix = spm_matrix(params=[12., 3., 4., 0., 0., 0.],
                                mat_file=mat_file)
    assert_true(os.path.isfile(out_spm_matrix.outputs.mat_file))
    mat_dict = loadmat(mat_file)
    affine = mat_dict['matrix']
    expected_affine = np.eye(4)
    expected_affine[:3, 3] = np.array([12., 3., 4.])
    np.testing.assert_allclose(affine, expected_affine)

    # test with 12 parameters
    out_spm_matrix = spm_matrix(params=[0., 0., 0., 0., 0., 0.,
                                        2., 1., 3., 0., 0., 0.],
                                mat_file=mat_file)
    assert_true(os.path.isfile(out_spm_matrix.outputs.mat_file))
    mat_dict = loadmat(mat_file)
    affine = mat_dict['matrix']
    expected_affine = np.eye(4)
    expected_affine[[0, 2], [0, 2]] = np.array([2., 3.])
    np.testing.assert_allclose(affine, expected_affine)


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


class SPMGetSpaceInputSpec(SPMCommandInputSpec):
    in_file = File(
        exists=True, mandatory=True,
        desc="target for reading the voxel-to-world"
    )
    mat_file = File(mandatory=True,
                    desc="Matlab file to save the transform to")


class SPMGetSpaceOutputSpec(TraitedSpec):
    mat_file = File(exists=True, desc="Matlab file holding transform")


class SPMGetSpace(SPMCommand):
    """ Uses SPM (spm_get_space) to read the affine transform and save it
    to a matlab file

    Examples
    --------
    >>> import nipype.interfaces.spm.utils as spmu
    >>> spm_get_space = spmu.SPMGetSpace(matlab_cmd='matlab-spm8')
    >>> spm_get_space.inputs.in_file = 'structural.nii'
    >>> spm_get_space.inputs.mat = 'func_to_struct.mat'
    >>> spm_get_space.run() # doctest: +SKIP
    """

    input_spec = SPMGetSpaceInputSpec
    output_spec = SPMGetSpaceOutputSpec

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
def test_spm_get_space():
    data = np.zeros((3, 6, 3))
    affine = np.array([[-1.7, -0.9, -1.2, 16.2 ],
                       [-0.9, -3.1, -2.3, 4.9],
                       [-0.2, 1.8, 3.7, -2.1],
                       [0., 0., 0., 1]])
    img = nibabel.Nifti1Image(data, affine=affine)
    in_file = os.path.join(tst.tmpdir, 'anat.nii')
    mat_file = os.path.join(tst.tmpdir, 'anat_mapping.mat')
    img.to_filename(in_file)
    spm_get_space = SPMGetSpace().run
    out_spm_get_space = spm_get_space(in_file=in_file, mat_file=mat_file)
    assert_true(os.path.isfile(out_spm_get_space.outputs.mat_file))


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
    spm_get_space = SPMGetSpace().run
    out_spm_get_space = spm_get_space(in_file=in_file, mat_file=mat_file)
    mat_dict = loadmat(out_spm_get_space.outputs.mat_file)
    affine_from_mat = mat_dict['matrix']
    np.testing.assert_allclose(spm_internals.spm_affine(in_file),
                               affine_from_mat)


@with_setup(tst.setup_tmpdata, tst.teardown_tmpdata)
def test_params_to_affine():
    # TODO: test with zooms and shears
    params = [.1, .5, -.6, .1, -4., 5.1, 1., 1., 1., 0., 0., 0.]
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
    mat_file = os.path.join(tst.tmpdir, 'affine.mat')
    spm_matrix = SPMMatrix().run
    out_spm_matrix = spm_matrix(params=params, mat_file=mat_file)
    mat_dict = loadmat(out_spm_matrix.outputs.mat_file)
    affine_from_spm_matrix = mat_dict['matrix']
    np.testing.assert_allclose(affine, affine_from_spm_matrix)


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
