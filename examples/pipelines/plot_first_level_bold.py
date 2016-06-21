"""
======================
First level model demo
======================

A basic first level model for BOLD data.
"""

# Give the paths to normalized BOLD EPI and anatomical
import os
from procasl import datasets
subjects = (3,)
preprocessed_heroes = datasets.load_heroes_dataset(
    subjects=subjects,
    subjects_parent_directory=os.path.expanduser(
        '~/CODE/process-asl/procasl_cache/heroes'),
    paths_patterns={'anat': 'nipype_mem/*Normalize/*/wanat*.nii',
                    'func BOLD': 'nipype_mem/*Smooth/*/swrvismot1*.nii'})
func_file = preprocessed_heroes['func BOLD'][0]
anat_file = preprocessed_heroes['anat'][0]

# Give the path to the paradigm
heroes = datasets.load_heroes_dataset(
    subjects=subjects,
    subjects_parent_directory=os.path.join(
        os.path.expanduser('~/procasl_data'), 'heroes'),
    paths_patterns={'paradigm': 'paradigms/acquisition1/*BOLD*1b.csv'})
paradigm_file = heroes['paradigm'][0]

# Read the paradigm
from nistats import experimental_paradigm
paradigm = experimental_paradigm.paradigm_from_csv(
    paradigm_file)

# Create the design matrix
import numpy as np
import matplotlib.pyplot as plt
import nibabel
from nistats.design_matrix import make_design_matrix, plot_design_matrix
tr = 2.5
n_scans = nibabel.load(func_file).get_data().shape[-1]
frametimes = np.arange(0, n_scans * tr, tr)
design_matrix = make_design_matrix(frametimes, paradigm)
plot_design_matrix(design_matrix)
plt.tight_layout()

# Fit GLM
print('Fitting a GLM')
from nistats.first_level_model import FirstLevelModel
fmri_glm = FirstLevelModel(tr)
fmri_glm = fmri_glm.fit(func_file, design_matrices=design_matrix)

# Specify the contrasts
contrasts = {}
n_columns = len(design_matrix.columns)
for n, name in enumerate(design_matrix.columns[:3]):
    contrasts[name] = np.zeros((n_columns,))
    contrasts[name][n] = 1
contrasts['[motor audio] left - right'] = \
    contrasts['motor_audio_left'] - contrasts['motor_audio_right']

# Compute contrast maps
from nilearn import plotting
for contrast_id, contrast_val in contrasts.items():
    z_map = fmri_glm.compute_contrast(
        contrast_val, contrast_name=contrast_id, output_type='z_score')
    plotting.plot_stat_map(
        z_map, bg_img=anat_file, threshold=5., title=contrast_id)

plotting.show()
