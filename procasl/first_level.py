import os
import numpy as np
from nistats_hemodynamic_models import compute_regressor
from nipype.interfaces import spm
from nipype.interfaces.base import (BaseInterface,
    BaseInterfaceInputSpec, traits, File, TraitedSpec, Directory, isdefined)
from nipype.interfaces.spm.base import scans_for_fnames
from nipype.utils.filemanip import filename_to_list
import nibabel


def _get_perfusion_baseline_regressor(n_frames):
    regressor = np.ones((n_frames, ))
    regressor[1::2] *= 0.5
    regressor[::2] *= -0.5
    return [regressor.tolist()], ['perfusion_baseline']


def _get_perfusion_activation_regressor(condition, condition_name,
                                        hrf_model, frame_times,
                                        oversampling=16,
                                        fir_delays=None, min_onset=-24):
    """
    Parameters
    ----------
    condition : 3-tuple of arrays
        (onsets, durations, amplitudes)

    condition_name : str
        name of the condition

    hrf_model : {'spm', 'spm + derivative', 'spm + derivative + dispersion',
        'glover', 'glover + derivative', 'fir'}
        Name of the hrf model to be used

    frame_times : array of shape (n_scans)
        the desired sampling times

    oversampling : int, optional
        oversampling factor to perform the convolution

    fir_delays : 1D-array-like, optional
        delays (in seconds) used in case of a finite impulse reponse model

    min_onset : float, optional
        minimal onset relative to frame_times[0] (in seconds)
        events that start before frame_times[0] + min_onset are not considered

    Returns
    -------
    computed_regressors: array of shape(n_scans, n_reg)
        computed regressors sampled at frame times

    reg_names: list of strings
        corresponding regressor names

    Notes
    -----
    The different hemodynamic models can be understood as follows:
    'spm': this is the hrf model used in SPM
    'spm + derivative': SPM model plus its time derivative (2 regressors)
    'spm + time + dispersion': idem, plus dispersion derivative (3 regressors)
    'glover': this one corresponds to the Glover hrf
    'glover + derivative': the Glover hrf + time derivative (2 regressors)
    'glover + derivative + dispersion': idem + dispersion derivative
                                        (3 regressors)
    'fir': finite impulse response basis, a set of delayed dirac models
           with arbitrary length. This one currently assumes regularly spaced
           frame times (i.e. fixed time of repetition).
    It is expected that spm standard and Glover model would not yield
    large differences in most cases.

    In case of glover and spm models, the derived regressors are
    orthogonalized wrt the main one.
    """
    computed_regressors, reg_names = compute_regressor(
        condition, hrf_model, frame_times, con_id=condition_name,
        oversampling=oversampling, fir_delays=fir_delays, min_onset=min_onset)
    computed_regressors[:, 1::2] *= .5
    computed_regressors[:, ::2] *= -.5
    reg_names = ['perfusion_' + reg_name for reg_name in reg_names]
    return computed_regressors.T.tolist(), reg_names


def compute_perfusion_regressors(conditions, condition_names,
                                 hrf_model, frame_times, oversampling=16,
                                 fir_delays=None, min_onset=-24):
    n_frames = frame_times.size
    perfusions_regressors, perfusion_regressors_names = \
        _get_perfusion_baseline_regressor(n_frames)
    for condition, condition_name in zip(conditions, condition_names):
        print condition
        activation_regressors, activation_regressor_names = \
            _get_perfusion_activation_regressor(
                condition, condition_name, hrf_model, frame_times,
                oversampling=oversampling, fir_delays=fir_delays,
                min_onset=min_onset)
        perfusions_regressors.extend(activation_regressors)
        perfusion_regressors_names.extend(activation_regressor_names)
    return perfusions_regressors, perfusion_regressors_names


def subject_info_from_csv(paradigm_file):
    paradigm = experimental_paradigm.paradigm_from_csv(paradigm_file)
    paradigm = paradigm.groupby(paradigm.name)
    conditions = paradigm.groups.keys()
    onsets = [paradigm.get_group(condition).onset.tolist()
              for condition in conditions]
    durations = [paradigm.get_group(condition).duration.tolist()
                 for condition in conditions]
    amplitudes = [paradigm.get_group(condition).modulation.tolist()
                  for condition in conditions]


class Level1DesignInputSpec(BaseInterfaceInputSpec):
    spm_mat_dir = Directory(
        exists=True, field='dir', desc='directory to store SPM.mat file (opt)')
    timing_units = traits.Enum(
        'secs', 'scans', field='timing.units',
        desc='units for specification of onsets', mandatory=True)
    interscan_interval = traits.Float(
        field='timing.RT', desc='Interscan interval in secs', mandatory=True)
    microtime_resolution = traits.Int(
        field='timing.fmri_t',
        desc='Number of time-bins per scan in secs (opt)')
    microtime_onset = traits.Float(
        field='timing.fmri_t0',
        desc='The onset/time-bin in seconds for alignment (opt)')
    session_info = traits.Any(
        field='sess', desc='Session specific information generated by '
                           '``modelgen.SpecifyModel``', mandatory=True)
    factor_info = traits.List(
        traits.Dict(traits.Enum('name', 'levels')), field='fact',
        desc='Factor specific information file (opt)')
    bases = traits.Dict(
        traits.Enum('hrf', 'fourier', 'fourier_han', 'gamma', 'fir'),
        field='bases', desc="""
            dict {'name':{'basesparam1':val,...}}
            name : string
                Name of basis function (hrf, fourier, fourier_han,
                gamma, fir)

                hrf :
                    derivs : 2-element list
                        Model  HRF  Derivatives. No derivatives: [0,0],
                        Time derivatives : [1,0], Time and Dispersion
                        derivatives: [1,1]
                fourier, fourier_han, gamma, fir:
                    length : int
                        Post-stimulus window length (in seconds)
                    order : int
                        Number of basis functions
                            """, mandatory=True)
    perfusion_bases = traits.Enum('bases', 'physio',
        field='perfusion bases', desc="""
        Name of the prf model to be used
                bases :
                    same as the basis function in bases
                physio:
                    linear transformation of the basis function.
                                     """, mandatory=False)
    volterra_expansion_order = traits.Enum(
        1, 2, field='volt', desc='Model interactions - yes:1, no:2')
    global_intensity_normalization = traits.Enum(
        'none', 'scaling', field='global',
        desc='Global intensity normalization - scaling or none')
    mask_image = File(
        exists=True, field='mask',
        desc='Image  for  explicitly  masking the analysis')
    mask_threshold = traits.Either(
        traits.Enum('-Inf'), traits.Float(),
        desc="Thresholding for the mask", default='-Inf', usedefault=True)
    model_serial_correlations = traits.Enum(
        'AR(1)', 'FAST', 'none',
        field='cvi',
        desc=('Model serial correlations AR(1), FAST or none. FAST '
              'is available in SPM12'))


class Level1DesignOutputSpec(TraitedSpec):
    spm_mat_file = File(exists=True, desc='SPM mat file')


class Level1Design(BaseInterface):
    """Generate an SPM design matrix possibly with perfusion regressors.
    Perfusion regressors consist of
        - a baseline blood flow reflecting the presence or absence of the
        inversion tag
        - BOLD regressors modulated with the baseline blood flow regressor
    as described in 'Estimation efficiency and statistical power in arterial
    spin labeling fMRI'. Mumford J.A. et al., 2006. Neuroimage 33,p. 103-114.

    http://www.fil.ion.ucl.ac.uk/spm/doc/manual.pdf#page=59

    Examples
    --------

    >>> level1design = Level1Design()
    >>> level1design.inputs.timing_units = 'secs'
    >>> level1design.inputs.interscan_interval = 2.5
    >>> level1design.inputs.bases = {'hrf':{'derivs': [0,0]}}
    >>> level1design.inputs.session_info = 'session_info.npz'
    >>> level1design.run() # doctest: +SKIP

    """
    input_spec = Level1DesignInputSpec
    output_spec = Level1DesignOutputSpec
    _jobtype = 'stats'
    _jobname = 'fmri_spec'

    def _format_arg(self, opt, spec, val):
        """Convert input to appropriate format for spm
        """
        if opt in ['spm_mat_dir', 'mask_image']:
            return np.array([str(val)], dtype=object)
        if opt in ['session_info']:  #, 'factor_info']:
            if isinstance(val, dict):
                return [val]
            else:
                return val
        return super(Level1Design, self)._format_arg(opt, spec, val)

    def _parse_inputs(self):
        """validate spm realign options if set to None ignore
        """
        einputs = super(Level1Design, self)._parse_inputs(
            skip=('mask_threshold'))
        for sessinfo in einputs[0]['sess']:
            sessinfo['scans'] = scans_for_fnames(filename_to_list(
                sessinfo['scans']), keep4d=False)
        if not isdefined(self.inputs.spm_mat_dir):
            einputs[0]['dir'] = np.array([str(os.getcwd())], dtype=object)
        return einputs

    def _run_interface(self, runtime):
        # Set the design parameters
        level1design = spm.Level1Design()
        level1design.inputs.spm_mat_dir = self.inputs.spm_mat_dir
        level1design.inputs.timing_units = self.inputs.timing_units
        level1design.inputs.interscan_interval = self.inputs.interscan_interval   
        level1design.inputs.microtime_resolution =\
            self.inputs.microtime_resolution
        level1design.inputs.microtime_onset = self.inputs.microtime_onset
        level1design.inputs.session_info = self.inputs.session_info
        level1design.inputs.factor_info = self.inputs.factor_info
        level1design.inputs.bases = self.inputs.bases
        level1design.inputs.volterra_expansion_order = \
            self.inputs.volterra_expansion_order
        level1design.inputs.global_intensity_normalization = \
            self.inputs.global_intensity_normalization
        level1design.inputs.mask_image = self.inputs.mask_image
        level1design.inputs.mask_threshold = self.inputs.mask_threshold
        level1design.inputs.model_serial_correlations = \
            self.inputs.model_serial_correlations

        if isdefined(self.inputs.perfusion_bases):
            # Compute perfusion regressors
            tr = self.inputs.interscan_interval
            # TODO: robustify (check session_info type is list of length 1)
            if isinstance(self.inputs.session_info, list):
                session_info = self.inputs.session_info[0]
            elif isinstance(self.inputs.session_info, str):
                session_info = self.inputs.session_info
            else:
                raise ValueError('session_info trait of Level1Design has type'
                ' {0}'.format(self.inputs.session_info))
                
            n_scans = nibabel.load(session_info['scans']).get_data().shape[-1]
            frametimes = np.arange(0, n_scans * tr, tr)
            if self.inputs.perfusion_bases == 'bases':
                hrf_model = 'spm'
                if self.inputs.bases['hrf']['derivs'] == [1, 0]:
                    hrf_model.extend(' + derivative')
                elif self.inputs.bases['hrf']['derivs'] == [1, 1]:
                    hrf_model.extend(' + derivative + dispersion')
            else:
                raise ValueError('physio PRF not implemented yet')
            condition_names = [c['name'] for c in session_info['cond']]  # robustify
            onsets = [c['onset'] for c in session_info['cond']]
            durations = [c['duration'] for c in session_info['cond']]
            if 'amplitude' in session_info['cond'][0].keys():
                amplitudes = [c['amplitude'] for c in session_info['cond']]
            else:
                amplitudes = [1 for c in session_info['cond']]

            conditions = zip(onsets, durations, amplitudes)
            perfusion_regressors, perfusion_regressor_names = \
                compute_perfusion_regressors(conditions, condition_names,
                                             hrf_model, frametimes)
            # Add perfusion regressors to model
            for n, (regressor, regressor_name) in enumerate(
                    zip(perfusion_regressors, perfusion_regressor_names)):
                session_info['regress'].insert(
                    n, {'val': regressor, 'name': regressor_name})
            level1design.inputs.session_info = [session_info]

        level1design.run()
        return runtime

    def _make_matlab_command(self, content):
        """validates spm options and generates job structure
        if mfile is True uses matlab .m file
        else generates a job structure and saves in .mat
        """
        if isdefined(self.inputs.mask_image):
            # SPM doesn't handle explicit masking properly, especially
            # when you want to use the entire mask image
            postscript = "load SPM;\n"
            postscript += "SPM.xM.VM = spm_vol('%s');\n" % list_to_filename(self.inputs.mask_image)
            postscript += "SPM.xM.I = 0;\n"
            postscript += "SPM.xM.T = [];\n"
            postscript += "SPM.xM.TH = ones(size(SPM.xM.TH))*(%s);\n" % self.inputs.mask_threshold
            postscript += "SPM.xM.xs = struct('Masking', 'explicit masking only');\n"
            postscript += "save SPM SPM;\n"
        else:
            postscript = None
        return super(Level1Design, self)._make_matlab_command(content, postscript=postscript)

    def _list_outputs(self):
        outputs = self._outputs().get()
        spm_mat_file = os.path.join(os.getcwd(), 'SPM.mat')
        outputs['spm_mat_file'] = spm_mat_file
        return outputs
