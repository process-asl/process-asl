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
                            perfusion_bases={'hrf': {'derivs': [0, 0]}},
                            timing_units='secs',
                            interscan_interval=tr,
                            model_serial_correlations='AR(1)')


level1design.inputs.mask_image = compute_brain_mask(
    mean_func_file, frac=.2)  # Compute cut neck mask
level1design.inputs.session_info = out_modelspec.outputs.session_info
out_level1design = level1design.run()
os.chdir(current_directory)

# Read the design matrix
from scipy.io import loadmat
spm_mat_struct = loadmat(out_level1design.outputs.spm_mat_file,
                         struct_as_record=False, squeeze_me=True)['SPM']
design_matrix = spm_mat_struct.xX.X
regressor_names = spm_mat_struct.xX.name

# Plot the activation perfusion regressors
import matplotlib.pyplot as plt
n_conditions = len(conditions)
for regressor, regressor_name in zip(
        design_matrix.T[n_conditions + 1:n_conditions + 4],
        regressor_names[n_conditions + 1:n_conditions + 4]):
    plt.plot(regressor, label=regressor_name)
plt.legend()

# Plot the design matrix
from nilearn.plotting import _set_mpl_backend  # set the backend
_set_mpl_backend
design_matrix /= np.maximum(1.e-12,  # normalize for better visu
                            np.sqrt(np.sum(design_matrix ** 2, 0)))
plt.figure()
plt.imshow(design_matrix, interpolation='nearest', aspect='auto')
plt.xlabel('regressors')
plt.ylabel('scan number')
plt.xticks(range(len(regressor_names)),
           regressor_names, rotation=60, ha='right')
plt.tight_layout()
plt.show()
