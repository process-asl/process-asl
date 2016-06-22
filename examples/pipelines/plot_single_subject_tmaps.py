"""
=====================
Single subject t-maps
=====================

This example is a basic first level pipeline for functinal ASL.
Conditions include a visual task and 2 motor-audio tasks.
T-maps are plotted for visual condition and motor audio left vs right.
"""
# Give the paths to the preprocessed functional ASL images and paradigm
import os
from procasl import datasets
subjects = (3,)
preprocessed_heroes = datasets.load_heroes_dataset(
    subjects=subjects,
    subjects_parent_directory=os.path.expanduser(
        '~/CODE/process-asl/procasl_cache/heroes'),
    paths_patterns={'motion': 'nipype_mem/procasl*Realign/*/rp*vismot1*.txt',
                    'func ASL': 'nipype_mem/*Realign/*/rsc*vismot1*ASL*.nii',
                    'mean ASL': 'nipype_mem/*Average/*/mean*rsc*vismot1*.nii'})
heroes = datasets.load_heroes_dataset(
    subjects=subjects,
    subjects_parent_directory=os.path.join(
        os.path.expanduser('~/procasl_data'), 'heroes'),
    paths_patterns={'paradigm': 'paradigms/acquisition1/*ASL*1b.csv'})
func_file = preprocessed_heroes['func ASL'][0]
mean_func_file = preprocessed_heroes['mean ASL'][0]
realignment_parameters = preprocessed_heroes['motion'][0]
paradigm_file = heroes['paradigm'][0]

# Read the paradigm
import numpy as np
paradigm = np.recfromcsv(paradigm_file)
conditions = np.unique(paradigm['name']).tolist()
onsets = [paradigm['onset'][paradigm['name'] == condition].tolist()
          for condition in conditions]
durations = [paradigm['duration'][paradigm['name'] == condition].tolist()
             for condition in conditions]

# Create a memory context
from nipype.caching import Memory
current_directory = os.getcwd()
cache_directory = '/tmp'
os.chdir(cache_directory)
mem = Memory(cache_directory)

#  Generate SPM-specific Model
from nipype.algorithms.modelgen import SpecifySPMModel
from nipype.interfaces.base import Bunch
subject_info = Bunch(conditions=conditions, onsets=onsets, durations=durations)
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
from procasl.first_level import Level1Design
from procasl.preprocessing import compute_brain_mask
spm_mat = os.path.join(cache_directory, 'SPM.mat')
if os.path.isfile(spm_mat):
    os.remove(spm_mat)  # design crashes if existant SPM.mat

level1design = mem.cache(Level1Design)
out_level1design = level1design(
    bases={'hrf': {'derivs': [0, 0]}},
    perfusion_bases='bases',
    timing_units='secs',
    interscan_interval=tr,
    model_serial_correlations='AR(1)',
    session_info=out_modelspec.outputs.session_info,
    mask_image=compute_brain_mask(mean_func_file, frac=.2))  # brain neck mask

# Estimate the parameters of the model
from nipype.interfaces.spm import EstimateModel
level1estimate = mem.cache(EstimateModel)
out_level1estimate = level1estimate(
    estimation_method={'Classical': 1},
    spm_mat_file=out_level1design.outputs.spm_mat_file)

# Specify contrasts
cont01 = (conditions[0] + ' > ' + conditions[1], 'T', conditions, [1, -1, 0])
cont02 = (conditions[2],   'T', conditions, [0, 0, 1])
contrast_list = [cont01, cont02]

# Estimate contrasts
from nipype.interfaces.spm import EstimateContrast
conestimate = mem.cache(EstimateContrast)
out_conestimate = conestimate(
    spm_mat_file=out_level1estimate.outputs.spm_mat_file,
    beta_images=out_level1estimate.outputs.beta_images,
    residual_image=out_level1estimate.outputs.residual_image,
    contrasts=contrast_list)
os.chdir(current_directory)

# Plot t-maps
from nilearn import plotting
contrast_names = zip(*out_conestimate.inputs['contrasts'])[0]
for contrast_name, t_image in zip(contrast_names,
                                    out_conestimate.outputs.spmT_images):
    plotting.plot_stat_map(t_image, threshold=3., title=contrast_name,
                           bg_img=mean_func_file)
plotting.show()
