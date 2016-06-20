from __future__ import print_function


# Give the path to the 4D ASL and anatomical images
import os
from procasl import datasets
current_directory = os.getcwd()
subjects = (3,)
preprocessed_heroes = datasets.load_heroes_dataset(
    subjects=subjects,
    subjects_parent_directory=os.path.expanduser(
        '~/CODE/process-asl/procasl_cache/heroes'),
    paths_patterns={'func BOLD': 'nipype_mem/*spm*Realign/*/rvismot1*BOLD*.nii',
                    'motion': 'nipype_mem/*spm*Realign/*/rp*.txt'})
func_file = preprocessed_heroes['func BOLD'][0]
realignment_parameters = preprocessed_heroes['motion'][0]

#  Generate SPM-specific Model
from nipype.algorithms.modelgen import SpecifySPMModel
print('Generate SPM-specific Model ...', end='')
tr = 2.5
modelspec = SpecifySPMModel(input_units='secs',
                            time_repetition=tr,
                            high_pass_filter_cutoff=128)
modelspec.inputs.realignment_parameters = realignment_parameters
modelspec.inputs.functional_runs = func_file
# Read the paradigm
from nistats import experimental_paradigm
from nipype.interfaces.base import Bunch
heroes = datasets.load_heroes_dataset(
    subjects=subjects,
    subjects_parent_directory=os.path.join(
        os.path.expanduser('~/procasl_data'), 'heroes'),
    paths_patterns={'paradigm': 'paradigms/acquisition1/*BOLD*1b.csv'})
paradigm_file = heroes['paradigm'][0]
paradigm = experimental_paradigm.paradigm_from_csv(paradigm_file)
paradigm = paradigm.groupby(paradigm.name)
conditions = paradigm.groups.keys()
onsets = [paradigm.get_group(condition).onset.tolist()
          for condition in conditions]
durations = [paradigm.get_group(condition).duration.tolist()
             for condition in conditions]
modelspec.inputs.subject_info = Bunch(conditions=conditions,
                                      onsets=onsets,
                                      durations=durations)
out_modelspec = modelspec.run()
print('Done.')

# Remove old SPM.mat
spm_mat = os.path.join(os.getcwd(), 'SPM.mat')
if os.path.isfile(spm_mat):
    print('ok')
    os.remove(spm_mat)

# Generate an SPM design matrix
from nipype.interfaces.spm import Level1Design
level1design = Level1Design(bases={'hrf': {'derivs': [0, 0]}},
                            timing_units='secs',
                            interscan_interval=tr,
                            model_serial_correlations='AR(1)')

#level1design.inputs.mask_image = binarize.binary_file
level1design.inputs.session_info = out_modelspec.outputs.session_info
print('Generate an SPM design matrix ...')
out_level1design = level1design.run()
print('Done.')

# Plot the design matrix
import numpy as np
from scipy.io import loadmat
from nilearn.plotting import _set_mpl_backend  # importing it sets the backend
_set_mpl_backend
import matplotlib.pyplot as plt
spm_mat_struct = loadmat(out_level1design.outputs.spm_mat_file,
                         struct_as_record=False, squeeze_me=True)['SPM']
design_matrix = spm_mat_struct.xX.X
regressor_names = spm_mat_struct.xX.name
design_matrix /= np.maximum(1.e-12,  # normalize for better visu
                            np.sqrt(np.sum(design_matrix ** 2, 0)))
plt.imshow(design_matrix, interpolation='nearest', aspect='auto')
plt.xlabel('regressors')
plt.ylabel('scan number')
plt.xticks(range(len(regressor_names)),
           regressor_names, rotation=60, ha='right')
plt.tight_layout()

# Plot the conditions regressors
for n, (regressor, regressor_name) in enumerate(zip(design_matrix.T[:3],
                                                    regressor_names)):
    plt.plot(regressor, label=regressor_name)
plt.legend()
print('Done.')
# Estimate the parameters of the model
from nipype.interfaces.spm import EstimateModel
print('Estimate the parameters of the model ...', end='')
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