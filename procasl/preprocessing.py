# TODO: keep only classes, move functions to _utils, rename to preprocessor
import os

import numpy as np
from scipy.io import savemat
import nibabel
import nipype.interfaces.spm as spm
from nipype.interfaces import fsl
from nipype.interfaces.base import BaseInterface, \
    BaseInterfaceInputSpec, traits, File, TraitedSpec, Directory
from nipype.utils.filemanip import split_filename
from nipype.interfaces.base import InputMultiPath, OutputMultiPath

from procasl.spm_internals import params_to_affine, spm_affine
from procasl._utils import check_images
fsl.FSLCommand.set_default_output_type('NIFTI')


def _add_prefix(prefix, in_file):
    """Adds a prefix to a filename

    Parameters
    ----------
    prefix : str
        Prefix to append to the filename

    in_file : str
        Input file name.

    Returns
    -------
    out_file : str
        Output file name
    """
    out_file = os.path.join(os.path.dirname(in_file),
                            prefix + os.path.basename(in_file))
    return out_file


def get_scans_number(in_file):
    """Return the number of scans for a 4D image.

    Parameters
    ----------
    in_file : str
        Input file name.

    Returns
    -------
    int
        Number of scans.
    """
    image = nibabel.load(in_file)
    data = image.get_data()
    if data.ndim != 4:
        raise ValueError("Expect a 4D image, got a {0}D".format(data.ndim))

    return data.shape[-1]


def select_scans(in_file, selected_scans, convert_3d=False, out_file=None):
    """Save selected scan volumes from a 4D image.

    Parameters
    ----------
    in_file : str
        Path to the 4D image

    selected_scans : list of int
        Scans to keep.

    convert_3d : bool, optional
        If True, the image is saved as a 3D nifti image.

    out_file : str or None, optional
        Path to the extracted image

    Returns
    -------
    out_file : str
        Path to the extracted image
    """
    image = nibabel.load(in_file)
    data = image.get_data()
    data = data[..., selected_scans]
    if convert_3d:
        data = np.squeeze(data, axis=(3,))

    image = nibabel.Nifti1Image(data, image.get_affine(), image.get_header())
    if out_file is None:
        out_file = _add_prefix('subscans_', in_file)

    nibabel.save(image, out_file)
    return out_file


def save_first_scan(in_file, out_file=None):
    """Save first volume from a 4D image.

    Parameters
    ----------
    in_file : str
        Path to the 4D image

    out_file : str or None, optional
        Path to the out image

    Returns
    -------
    out_file : str
        Path to the out image
    """
    if out_file is None:
        out_file = os.path.join(os.path.dirname(in_file),
                                'first_volume_' + os.path.basename(in_file))
    out_file = select_scans(in_file, [0], convert_3d=True,
                            out_file=out_file)
    return out_file


def compute_brain_mask(in_file, frac=0.5):
    """Computes binary brain mask using FSL BET.

    Parameters
    ----------
    in_file : str
        Path to the 3D image

    frac : float, optional
        Fractional intensity threshold, smaller values give larger brain
        outline estimates.

    Returns
    -------
    str
        Path to the brain mask image
    """
    btr = fsl.BET()
    btr.inputs.in_file = in_file
    btr.inputs.frac = frac
    btr.inputs.mask = True
    res = btr.run()
    return res.outputs.mask_file


def apply_mask(in_file, mask_file, mask_value=np.nan, out_file=None):
    """Masks input with a binary mask_file.
    Parameters
    ----------
    in_file : str
        Path to the 3D image

    mask_file : str
        Path to the binary mask image

    mask_value : float, optional
        Value to allocate to masked voxels.

    out_file : str or None, optional
        Path to the masked image

    Returns
    -------
    out_file : str
        Path to the masked image
    """
    # Load images
    image = nibabel.load(in_file)
    data = image.get_data()
    mask_image = nibabel.load(mask_file)
    mask_data = mask_image.get_data()

    # Check shapes and affines
    check_images(in_file, mask_file)

    # Compute the masked image
    data[mask_data == 0] = mask_value
    out_image = nibabel.Nifti1Image(data, image.get_affine(),
                                    image.get_header())
    if out_file is None:
        out_file = _add_prefix('masked_', in_file)

    nibabel.save(out_image, out_file)
    return out_file


class RescaleInputSpec(BaseInterfaceInputSpec):
    in_file = File(
        exists=True,
        mandatory=True,
        copyfile=True,
        desc='image filename to rescale')
    ss_tr = traits.Float(
        mandatory=True,
        desc='Single slice repetition time, in ms')
    t_i_1 = traits.Float(
        mandatory=True,
        desc='Bolus length, in ms')
    t_i_2 = traits.Float(
        mandatory=True,
        desc='Inversion time (time from the application of the labeling'
             'pulse to image acquisition), in ms')
    t1_blood = traits.Float(
        1650.,
        usedefault=True,
        desc='T1 of the blood in ms')
    label_efficiency = traits.Float(
        .98,
        usedefault=True,
        desc='labeling efficiency')


class RescaleOutputSpec(TraitedSpec):
    rescaled_file = File(exists=True,
                         desc='The rescaled image file')


class Rescale(BaseInterface):
    """Correct for T1 relaxation between different slices.

    PASL images are acquired in EPI single shot with slices from
    bottom to up of the brain.
    For PASL,
    CBF (ml/100g/min) = DeltaM / (2 * M0b * tao * exp(-TI / T1b) * qTI)
    and
    M0b = Rwm * M0WM * exp((1 / T2wm - 1 / T2b) * TE)
    or M0b=Rcsf * M0csf * exp((1 / T2csf-1 / T2b) * TE)
    or M0b = MPD / (1 - exp(-TR / T1_tissue)),
    TI is the inversion time for different slice;
    T1b is the constant relaxation time of arterial blood.
    tao is actually TI1 in QUIPPS II
    qTI is close to unit, and is set to 0.85 in Warmuth 05. In addition, we
    introduce the label efficiency in the calculation.
    Rwm  - proton density ratio between blood and WM1.06 in Wong 97. 1.19 in
    Cavosuglu 09; T2wm and T2b are 55 msec and 100 for 1.5T, 40 and 80 for 3T,
    30 and 60 for 4T;
    Rcsf - proton density ratio between blood and csf, 0.87 in Cavosuglu,
    T2csf is 74.9 ms for 3T.
    M0WM means the mean value in an homogenous white matter region, and it
    could be selected by drawing an ROI in the M0 image.
    T2wm and T2b at 3T were set to 44.7 and 43.6,
    T2csf if used was set to 74.9 according to Cavusoglu 09 MRI

    Notes
    -----
    This is a reimplementation of the rescaling method of
    correction_scalefactors_philips_2010.m from the GIN toolbox,
    courtesy of Jan Warnking.

    References
    ----------
    Buxton et al, 1998 MRM 40:383-96.
    Warmuth C., Gunther M. and Zimmer G. Radiology, 2003; 228:523-532.

    Examples
    --------
    from procasl import preprocessing
    rescale = preprocessing.Rescale
    rescale.inputs.in_file = 'raw_asl.nii'
    rescale.inputs.ss_tr = 35.
    rescale.inputs.t_i_1 = 800.
    rescale.inputs.t_i_2 = 1800.
    out_rescale = rescale.run()
    print(out_rescale.rescaled files)
    """
    input_spec = RescaleInputSpec
    output_spec = RescaleOutputSpec

    def _run_interface(self, runtime):
        img = nibabel.load(self.inputs.in_file)
        data = img.get_data()
        n_slices = data.shape[2]
        milli_second = 1000.  # 1s in ms
        scaling = np.exp((self.inputs.t_i_2 + self.inputs.ss_tr *
                         np.arange(0, n_slices)) / self.inputs.t1_blood) /\
            (2. * self.inputs.label_efficiency *
             self.inputs.t_i_1 / milli_second)
        scaling = scaling[np.newaxis, np.newaxis, :, np.newaxis]
        scaling = np.tile(scaling, (data.shape[0], data.shape[1], 1,
                                    data.shape[-1]))
        data = data * scaling
        img = nibabel.Nifti1Image(data, img.get_affine(), img.get_header())
        out_file = _add_prefix('sc_', self.inputs.in_file)
        nibabel.save(img, out_file)
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        fname = self.inputs.in_file
        _, base, _ = split_filename(fname)
        outputs["rescaled_file"] = os.path.abspath(
            'sc_' + base + '.nii')
        return outputs


class AverageInputSpec(BaseInterfaceInputSpec):
    in_file = File(
        exists=True,
        mandatory=True,
        copyfile=True,
        desc='list of images filenames to average')


class AverageOutputSpec(TraitedSpec):
    mean_image = File(exists=True,
                      desc='The average image file')


class Average(BaseInterface):
    """Compute average functional across time, keeping the affine of
    first scan.

    Notes
    -----
    This is a reimplementation of the averaging method of
    average_2010.m from the GIN toolbox.
    """
    input_spec = AverageInputSpec
    output_spec = AverageOutputSpec

    def _run_interface(self, runtime):
        # Compute and save the mean
        img = nibabel.load(self.inputs.in_file)
        data = img.get_data().mean(axis=-1)
        img = nibabel.Nifti1Image(data, img.get_affine(), img.get_header())
        fname = self.inputs.in_file
        _, base, _ = split_filename(fname)
        out_file = os.path.abspath('mean_' + base + '.nii')
        nibabel.save(img, out_file)
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        fname = self.inputs.in_file
        _, base, _ = split_filename(fname)
        outputs["mean_image"] = os.path.abspath(
            'mean_' + base + '.nii')
        return outputs


class RealignInputSpec(BaseInterfaceInputSpec):
    in_file = File(
        exists=True,
        mandatory=True,
        copyfile=True,
        desc='The filename of the input ASL 4D image.')
    paths = InputMultiPath(Directory(exists=True),
                           desc='Paths to add to matlabpath')
    register_to_mean = traits.Bool(
        True,
        mandatory=True,
        usedefault=True,
        desc='Indicate whether realignment is done to the mean image')
    correct_tagging = traits.Bool(
        False,
        usedefault=True,
        desc='True/False correct for tagging artifact by zeroing the mean'
             ' difference between control and tag.')
    control_scans = traits.List(
        [],
        traits.Int(),
        desc='control frames numbers')
    tag_scans = traits.List(
        [],
        traits.Int(),
        desc='tag frames numbers')


class RealignOutputSpec(TraitedSpec):
    realigned_files = File(exists=True,
                           desc='The resliced files')
    realignment_parameters = OutputMultiPath(
        File(exists=True),
        desc='Estimated translation and rotation parameters')


class Realign(BaseInterface):
    """Realign ASL scans. Default parameters are those of the GIN
    pipeline.

    Notes
    -----
    This is a reimplementation of the realignement method from
    myrealign_pasl_2010.m of GIN toolbox.

    Examples
    --------
    from procasl import preprocessing
    realign = preprocessing.Realign
    realign.inputs.in_file = 'functional.nii'
    realign.inputs.register_to_mean = False
    realign.inputs.correct_tagging = True
    out_realign = realign.run()
    print(out_realign.realigned files, out_realign.realignement_parameters)
    """
    input_spec = RealignInputSpec
    output_spec = RealignOutputSpec

    def _run_interface(self, runtime):
        # Set the realignement options
        realign = spm.Realign()
        realign.inputs.paths = self.inputs.paths
        realign.inputs.in_files = self.inputs.in_file
        realign.inputs.register_to_mean = self.inputs.register_to_mean
        realign.inputs.quality = 0.9
        realign.inputs.fwhm = 5.
        realign.inputs.separation = 4  # TODO: understand this parameter
        realign.inputs.interp = 2
        if self.inputs.correct_tagging:
            # Estimate the realignement parameters
            realign.inputs.jobtype = 'estimate'
            realign.run()
            parameters_file = realign.aggregate_outputs().get()[
                'realignment_parameters']
            rea_parameters = np.loadtxt(parameters_file)

            # Correct for tagging: equal means for control and tag scans
            n_scans = len(rea_parameters)
            if self.inputs.control_scans:
                control_scans = self.inputs.control_scans
            else:
                control_scans = range(0, n_scans, 2)

            if self.inputs.tag_scans:
                tag_scans = self.inputs.tag_scans
            else:
                tag_scans = range(1, n_scans, 2)

            gap = np.mean(rea_parameters[control_scans], axis=0) -\
                np.mean(rea_parameters[tag_scans], axis=0)
            rea_parameters[control_scans] -= gap / 2.
            rea_parameters[tag_scans] += gap / 2.

            # Save the corrected realignement parameters
            np.savetxt(parameters_file, rea_parameters, delimiter=' ')

            # Save the corrected transforms for each frame in spm compatible
            #  .mat. This .mat serves as header for spm in case of 4D files
            affine = spm_affine(self.inputs.in_file)
            rea_affines = np.zeros((4, 4, n_scans))
            for n_scan, param in enumerate(rea_parameters):
                rea_affines[..., n_scan] = params_to_affine(param).dot(affine)
            affines_file = os.path.splitext(self.inputs.in_file)[0] + '.mat'
            savemat(affines_file, dict(mat=rea_affines))
        else:
            realign.inputs.jobtype = 'estimate'
            realign.inputs.register_to_mean = self.inputs.register_to_mean
            realign.run()

        # Reslice and save the aligned volumes
        realign = spm.Realign()
        realign.inputs.paths = self.inputs.paths
        realign.inputs.in_files = self.inputs.in_file
        realign.inputs.jobtype = 'write'
        realign.run()
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        fname = self.inputs.in_file
        _, base, _ = split_filename(fname)
        outputs["realignment_parameters"] = os.path.abspath(
            'rp_' + base + '.txt')
        outputs["realigned_files"] = os.path.abspath(
            'r' + base + '.nii')
        return outputs


class GetFirstScanInputSpec(BaseInterfaceInputSpec):
    in_file = File(
        exists=True,
        mandatory=True,
        copyfile=True,
        desc='The input 4D ASL image filename')


class GetFirstScanOutputSpec(TraitedSpec):
    m0_file = File(
        exists=True,
        desc='The first scan image filename')


class GetFirstScan(BaseInterface):
    """Save the first scan from 4D image (M0).
    """
    input_spec = GetFirstScanInputSpec
    output_spec = GetFirstScanOutputSpec

    def _run_interface(self, runtime):
        # Save first scan
        image = nibabel.load(self.inputs.in_file)
        data = image.get_data()
        data = data[..., 0]
        image = nibabel.Nifti1Image(data, image.get_affine(),
                                    image.get_header())
        _, base, _ = split_filename(self.inputs.in_file)
        out_file = os.path.abspath('m0_' + base + '.nii')
        nibabel.save(image, out_file)
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        fname = self.inputs.in_file
        _, base, _ = split_filename(fname)
        outputs["m0_file"] = os.path.abspath('m0_' + base + '.nii')
        return outputs


class RemoveFirstScanInputSpec(BaseInterfaceInputSpec):
    in_file = File(
        exists=True,
        mandatory=True,
        copyfile=True,
        desc='The input 4D ASL image filename')


class RemoveFirstScanOutputSpec(TraitedSpec):
    tag_ctl_file = File(
        exists=True,
        desc='The tag/control sequence image filename')


class RemoveFirstScanControl(BaseInterface):
    """Save the tag/control sequence of a 4D ASL image (removes first scan).
    """
    input_spec = RemoveFirstScanInputSpec
    output_spec = RemoveFirstScanOutputSpec

    def _run_interface(self, runtime):
        # Remove first scan
        image = nibabel.load(self.inputs.in_file)
        data = image.get_data()
        data = data[..., 1:]
        image = nibabel.Nifti1Image(data, image.get_affine(),
                                    image.get_header())
        _, base, _ = split_filename(self.inputs.in_file)
        out_file = os.path.abspath('tag_ctl_' + base + '.nii')
        nibabel.save(image, out_file)
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        fname = self.inputs.in_file
        _, base, _ = split_filename(fname)
        outputs["tag_ctl_file"] = os.path.abspath('tag_ctl_' + base + '.nii')
        return outputs
