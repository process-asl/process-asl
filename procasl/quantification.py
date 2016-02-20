import os

import nibabel
import numpy as np

from nipype.interfaces.base import (
    BaseInterface, BaseInterfaceInputSpec, traits, File, TraitedSpec)
from nipype.interfaces.base import OutputMultiPath
from nipype.utils.filemanip import split_filename

from _utils import check_images


def compute_perfusion(in_file, ctl_scans=None,
                      tag_scans=None):
    """Compute the mean perfusion image from a functional ASL sequence. The
    ASL sequence is assumed aligned.

    Parameters
    ----------
    in_file : str
        The filename of the input functional ASL 4D image.

    ctl_scans : list of int or None, optional
        Indexes of control volumes.

    tag_scans : list of int or None, optional
        Indexes of tagged volumes, same length as ctl_scans.


    Returns
    -------
    out_file : str
        The filename of the output perfusion image.
    """
    image = nibabel.load(in_file)
    data = image.get_data()
    n_scans = data.shape[-1]
    if ctl_scans is None:
        ctl_scans = range(0, n_scans, 2)

    if tag_scans is None:
        tag_scans = range(1, n_scans, 2)

    if len(ctl_scans) != len(tag_scans):
        raise ValueError('{0} control scans, {1} tagged scans.'.format(
                         ctl_scans, tag_scans))

    ctl_data = data[..., ctl_scans]
    tag_data = data[..., tag_scans]
    perfusion_data = ctl_data - tag_data
    perfusion_data = np.mean(perfusion_data, axis=-1)
    image = nibabel.Nifti1Image(perfusion_data, image.get_affine(),
                                image.get_header())
    out_file = os.path.join(os.path.dirname(in_file),
                            'perfusion_' + os.path.basename(in_file))
    nibabel.save(image, out_file)
    return out_file


class QuantifyCBFInputSpec(BaseInterfaceInputSpec):
    perfusion_file = File(
        exists=True,
        mandatory=True,
        copyfile=True,
        desc='perfusion image filename')
    m0_file = File(
        exists=True,
        mandatory=True,
        copyfile=True,
        desc='M0 image filename')
    tr = traits.Float(
        mandatory=True,
        usedefault=True,
        desc='TR, in ms')
    t1_gm = traits.Float(
        1331.,
        usedefault=True,
        desc='Gray matter T1 value, in ms')


class QuantifyCBFOutputSpec(TraitedSpec):
    cbf_file = OutputMultiPath(
        traits.Either(traits.List(File(exists=True)), File(exists=True)),
        desc='The CBF filename')


class QuantifyCBF(BaseInterface):
    input_spec = QuantifyCBFInputSpec
    output_spec = QuantifyCBFOutputSpec

    def _run_interface(self, runtime):
        """Basic PASL quantification for QUIPPS II.
        CBF (ml/100g/min) = 6000 * brain_blood_coef * perfusion *
                            exp(TI / T1b) * (1 - exp(-TR / T1_tissue)) /
                            (2 * label_efficiency * M0 * TI1)

        References
        ----------
        Recommended implementation of arterial spin-labeled perfusion MRI for
        clinical applications: A consensus of the ISMRM perfusion study group
        and the European consortium for ASL in dementia. David C. Alsop et al.
        (2015), MRM 73.
        """
        # Load images
        perfusion_image = nibabel.load(self.inputs.perfusion_file)
        perfusion_data = perfusion_image.get_data()
        perfusion_affine = perfusion_image.get_affine()
        m0_image = nibabel.load(self.inputs.m0_file)
        m0_data = m0_image.get_data()

        # Check shapes and affines
        check_images(perfusion_image, m0_image)

        # Compute the CBF
        m0_data = m0_data.astype(float)
        m0_data = m0_data / (1. - np.exp(- self.inputs.tr / self.inputs.t1_gm))
        non_zero_m0 = np.abs(m0_data) > 1e-4
        brain_blood_coef = 0.9  # brain blood partition coefficient, in mL/g
        unit_scaling = 6000.  # from ml/g/s to ml/100g/min
        cbf_data = perfusion_data * unit_scaling
        cbf_data[non_zero_m0] /= m0_data[non_zero_m0]
        cbf_data[np.logical_not(non_zero_m0)] = np.nan
        cbf_data *= brain_blood_coef

        # Save the CBF image
        image = nibabel.Nifti1Image(cbf_data, perfusion_affine,
                                    perfusion_image.get_header())
        fname = self.inputs.perfusion_file
        _, base, _ = split_filename(fname)
        cbf_file = os.path.abspath('basic_cbf_' + base + '.nii')
        nibabel.save(image, cbf_file)
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        fname = self.inputs.perfusion_file
        _, base, _ = split_filename(fname)
        outputs["cbf_file"] = os.path.abspath('basic_cbf_' + base + '.nii')
        return outputs
