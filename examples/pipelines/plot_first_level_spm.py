# Give the paths to the 4D ASL and anatomical images and paradigm
import os
from procasl import datasets
current_directory = os.getcwd()
subjects_parent_directory = os.path.expanduser(
        '~/CODE/process-asl/procasl_cache/heroes')
subjects = (0,)
preprocessed_heroes = datasets.load_heroes_dataset(
    subjects=subjects,
    subjects_parent_directory=os.path.expanduser(
        '~/CODE/process-asl/procasl_cache/heroes'),
    paths_patterns={'motion': 'nipype_mem/procasl*Realign/*/rp*vismot1*.txt',
                    'func ASL': 'nipype_mem/*Realign/*/rsc*vismot1*ASL*.nii',
                    'mean ASL':'nipype_mem/*Average/*/mean*rsc*vismot1*.nii'})
heroes = datasets.load_heroes_dataset(
    subjects=subjects,
    subjects_parent_directory=os.path.join(
        os.path.expanduser('~/procasl_data'), 'heroes'),
    paths_patterns={'paradigm': 'paradigms/acquisition1/*ASL*1b.csv'})
func_file = preprocessed_heroes['func ASL'][0]
realignment_parameters = preprocessed_heroes['motion'][0]
paradigm_file = heroes['paradigm'][0]
mean_func_file = preprocessed_heroes['mean ASL'][0]

# Define directory to save .m scripts and outputs to
subject_directory = os.path.relpath(func_file, subjects_parent_directory)
subject_directory = subject_directory.split(os.sep)[0]
working_directory = os.path.join('/tmp', subject_directory)
if not os.path.exists(working_directory):
    os.mkdir(working_directory)

os.chdir(working_directory)

#  Generate SPM-specific Model
from nipype.algorithms.modelgen import SpecifySPMModel
tr = 2.5
modelspec = SpecifySPMModel(input_units='secs',
                            time_repetition=tr,
                            high_pass_filter_cutoff=128)
modelspec.inputs.realignment_parameters = realignment_parameters
modelspec.inputs.functional_runs = func_file
# Read the conditions
import numpy as np
from nipype.interfaces.base import Bunch
paradigm = np.recfromcsv(paradigm_file)
conditions = np.unique(paradigm['name'])
onsets = [paradigm['onset'][paradigm['name'] == condition].tolist()
          for condition in conditions]
durations = [paradigm['duration'][paradigm['name'] == condition].tolist()
             for condition in conditions]
modelspec.inputs.subject_info = Bunch(conditions=conditions, onsets=onsets,
                                      durations=durations)
out_modelspec = modelspec.run()

# Generate an SPM design matrix
from procasl.first_level import Level1Design
from procasl.preprocessing import compute_brain_mask
spm_mat = os.path.join(os.getcwd(), 'SPM.mat')
if os.path.isfile(spm_mat):
    os.remove(spm_mat)  # design crashes if existant SPM.mat

level1design = Level1Design(bases={'hrf': {'derivs': [0, 0]}},
                            perfusion_bases='bases',
                            timing_units='secs',
                            interscan_interval=tr,
                            model_serial_correlations='AR(1)')
level1design.inputs.mask_image = compute_brain_mask(
    mean_func_file, frac=.2)  # Compute cut neck mask
level1design.inputs.session_info = out_modelspec.outputs.session_info
out_level1design = level1design.run()

# Estimate the parameters of the model
from nipype.interfaces.spm import EstimateModel
level1estimate = EstimateModel(estimation_method={'Classical': 1})
level1estimate.inputs.spm_mat_file = out_level1design.outputs.spm_mat_file
out_level1estimate = level1estimate.run()

# Specify contrasts
cont01 = [conditions[0],   'T', conditions, [1, 0, 0]]
cont02 = [conditions[1], 'T', conditions, [0, 1, 0]]
cont03 = [conditions[1], 'T', conditions, [0, 1, 0]]
cont04 = [conditions[1] + ' vs ' + conditions[2], 'T', conditions, [0, 1, -1]]
cont05 = [conditions[2] + ' vs ' + conditions[1], 'T', conditions, [0, -1, 1]]
cont06 = ['Cond vs zero', 'F', [cont01, cont02]]
cont07 = ['Diff vs zero', 'F', [cont04, cont05]]
contrast_list = [cont01, cont02, cont03, cont04, cont05, cont06, cont07]

# Estimate contrasts
from nipype.interfaces.spm import EstimateContrast
conestimate = EstimateContrast()
conestimate.inputs.spm_mat_file = out_level1estimate.outputs.spm_mat_file
conestimate.inputs.beta_images = out_level1estimate.outputs.beta_images
conestimate.inputs.residual_image = out_level1estimate.outputs.residual_image
conestimate.inputs.contrasts = contrast_list
conestimate = conestimate.run()
os.chdir(current_directory)

# Plot the contrast maps
print('Done. \nPlotting')
from nilearn import plotting
contrast_names = zip(*conestimate.inputs['contrasts'])[0]
for contrast_name, con_image in zip(contrast_names,
                                    conestimate.outputs.con_images):
    plotting.plot_stat_map(con_image, threshold=3., title=contrast_name)
plotting.show()