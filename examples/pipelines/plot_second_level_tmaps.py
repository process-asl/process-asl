"""
===================
Second level t-maps
===================

Second level pipeline for functional ASL.
Thresholded t-maps are plotted for visual condition and left-right
motor-auditory conditions, both for BOLD and perfusion regressors.
"""
###############
# Data loading#
###############
# Give the paths to smoothed normalized functional ASL images and confounds
import os
from procasl import datasets
subjects_parent_directory = os.path.expanduser(
        '~/CODE/process-asl/procasl_cache/heroes')
preprocessed_heroes = datasets.load_heroes_dataset(
    subjects_parent_directory=subjects_parent_directory,
    paths_patterns={'motion': 'nipype_mem/procasl*Realign/*/rp*vismot1*.txt',
                    'func ASL': 'nipype_mem/*Smooth/*/swrsc*vismot1*ASL*.nii',
                    'mean ASL': 'nipype_mem/*Smooth/*/swmean_rsc*vismot1*.nii'})

# Give the paths to paradigms
heroes = datasets.load_heroes_dataset(
    subjects_parent_directory=os.path.join(
        os.path.expanduser('~/procasl_data'), 'heroes'),
    paths_patterns={'paradigm': 'paradigms/acquisition1/*ASL*1b.csv'})

#######################
# First level analysis#
#######################
# Loop over subjects
import numpy as np
from scipy.io import loadmat
from nipype import caching
from nipype.interfaces import base, spm
from nipype.algorithms.modelgen import SpecifySPMModel
from procasl.first_level import Level1Design
from procasl.preprocessing import compute_brain_mask
current_directory = os.getcwd()
con_images = []
for (func_file, mean_func_file, realignment_parameters, paradigm_file) in zip(
        preprocessed_heroes['func ASL'], preprocessed_heroes['mean ASL'],
        preprocessed_heroes['motion'], heroes['paradigm']):
    # Create a memory context
    subject_directory = os.path.relpath(func_file, subjects_parent_directory)
    subject_directory = subject_directory.split(os.sep)[0]
    cache_directory = os.path.join(os.path.expanduser('~/CODE/process-asl'),
                                   'procasl_cache', 'heroes',
                                   subject_directory)
    if not os.path.exists(cache_directory):
        os.mkdir(cache_directory)

    os.chdir(cache_directory)  # nipype saves .m scripts into cwd
    mem = caching.Memory(cache_directory)

    # Read the paradigm
    paradigm = np.recfromcsv(paradigm_file)
    conditions = np.unique(paradigm['name']).tolist()
    onsets = [paradigm['onset'][paradigm['name'] == condition].tolist()
              for condition in conditions]
    durations = [paradigm['duration'][paradigm['name'] == condition].tolist()
                 for condition in conditions]

    #  Generate SPM-specific Model
    subject_info = base.Bunch(conditions=conditions, onsets=onsets,
                              durations=durations)
    tr = 2.5
    modelspec = mem.cache(SpecifySPMModel)
    out_modelspec = modelspec(
        input_units='secs',
        time_repetition=tr,
        high_pass_filter_cutoff=128,
        realignment_parameters=realignment_parameters,
        functional_runs=func_file,
        subject_info=subject_info)

    # Generate an SPM design matrix
    spm_mat = os.path.join(cache_directory, 'SPM.mat')
    if os.path.isfile(spm_mat):
        os.remove(spm_mat)  # design crashes if existant SPM.mat

    level1design = mem.cache(Level1Design)
    out_level1design = level1design(
        bases={'hrf': {'derivs': [0, 0]}},
        perfusion_bases='bases',
        timing_units='secs',
        interscan_interval=tr,
        session_info=out_modelspec.outputs.session_info,
        mask_image=compute_brain_mask(mean_func_file, frac=.2))

    # Estimate the parameters of the model
    level1estimate = mem.cache(spm.EstimateModel)
    out_level1estimate = level1estimate(
        estimation_method={'Classical': 1},
        spm_mat_file=out_level1design.outputs.spm_mat_file)

    # Read regressors names
    spm_mat_struct = loadmat(out_level1design.outputs.spm_mat_file,
                             struct_as_record=False, squeeze_me=True)['SPM']
    regressor_names = spm_mat_struct.xX.name.tolist()

    # Specify contrasts
    cont01 = ('[BOLD] ' + conditions[0] + ' > ' + conditions[1], 'T',
              regressor_names,
              [1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    cont02 = ('[BOLD] ' + conditions[2], 'T', regressor_names,
              [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    cont03 = ('[perfusion] ' + conditions[0] + ' > ' + conditions[1], 'T',
              regressor_names,
              [0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0])
    cont04 = ('[perfusion] ' + conditions[2], 'T', regressor_names,
              [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    contrast_list = [cont01, cont02, cont03, cont04]

    # Estimate contrasts
    conestimate = mem.cache(spm.EstimateContrast)
    out_conestimate = conestimate(
        spm_mat_file=out_level1estimate.outputs.spm_mat_file,
        beta_images=out_level1estimate.outputs.beta_images,
        residual_image=out_level1estimate.outputs.residual_image,
        contrasts=contrast_list,
        group_contrast=True)

    con_images.append(out_conestimate.outputs.con_images)

########################
# Second level pipeline#
########################
contrast_names = zip(*out_conestimate.inputs['contrasts'])[0]
from nipype.interfaces.spm import OneSampleTTestDesign
t_maps = []
for con_files in zip(*con_images):
    # Create one sample T-Test Design
    con_files = con_files[:5] + con_files[7:]
    onesamplettestdes = mem.cache(OneSampleTTestDesign)
    out_onesamplettestdes = onesamplettestdes(
        in_files=list(con_files))

    # Estimate the parameters of the model
    level2estimate = mem.cache(spm.EstimateModel)
    out_level2estimate = level2estimate(
        estimation_method={'Classical': 1},
        spm_mat_file=out_onesamplettestdes.outputs.spm_mat_file)

    # Estimate group contrast
    level2conestimate = mem.cache(spm.EstimateContrast)
    out_level2conestimate = level2conestimate(
        group_contrast=True,
        spm_mat_file=out_level2estimate.outputs.spm_mat_file,
        beta_images=out_level2estimate.outputs.beta_images,
        residual_image=out_level2estimate.outputs.residual_image,
        contrasts=['Group', 'T', ['mean'], [1]])
    t_maps.append(out_level2conestimate.outputs.spmT_images)

# Plot thresholded t-maps
from nilearn import plotting
for contrast_name, t_map in zip(contrast_names, t_maps):
    plotting.plot_glass_brain(t_map, threshold=5., title=contrast_name,
                              colorbar=True, plot_abs=False,
                              black_bg=True, display_mode='lyrz')
plotting.show()
