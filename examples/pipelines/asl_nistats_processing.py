"""ASL processing with nistats
"""

import os
import os.path as op
import numpy as np
import nibabel as nb
import glob
import pandas as pd
import matplotlib.pyplot as plt
import nilearn
import scipy

from collections import OrderedDict
from nistats import experimental_paradigm, design_matrix, glm
from nistats.glm import FirstLevelGLM
from nistats.thresholding import map_threshold
from nilearn.input_data import NiftiMasker
from nilearn.plotting import plot_stat_map

from pyhrf.retreat.glm_tools import make_design_matrix_asl
from pypreprocess.reporting.glm_reporter import group_one_sample_t_test
from pypreprocess.external.nistats.glm import FMRILinearModel

from . import nistats_processing_tools as nt



archives = op.join('/home', 'aina', 'Data', 'HEROES_preprocessed')
mask_fn = op.join(archives, 'brain_mask_willard.nii')

t_r = 2.5
n_scans = 164
frametimes = np.linspace(0, (n_scans - 1) * t_r, n_scans)

drift_model = 'Cosine'
hrf_model = 'Canonical'
_duration = 10
epoch_duration = _duration * t_r
hfcut = 2 * 2 * epoch_duration

subject_names = ["AC150013", "CD110147", "KP140463", "MP130368", "SC120530",
                 "NS110383", "PB130006", "SB150062", "MB140004", "XM140202",
                 "RG130377", "SH140342", "AM140315", "LZ140352", "HB140194"]
subject_names = ["SH140342", "AM140315", "LZ140352", "HB140194"]

first_levels = []
mask_fn = op.join(archives, 'gm_mask_willard_asl.nii')
mask_img = nb.load(mask_fn)
masker = NiftiMasker(mask_img=mask_img).fit()
affine = mask_img.get_affine()

for subject in subject_names:
    print subject
    #mvt_file = glob.glob(op.join(archives, subject, 'Preprocessed', 'rp_sc_tag_ctl_vismot2.txt
    anat_fn = glob.glob(op.join(archives, subject, 'Preprocessed', 'wranat_*.nii'))[0]
    data_fn1 = glob.glob(op.join(archives, subject, 'Preprocessed', 'swrsc_tag_ctl_vismot1*.nii'))[0]
    data_fn2 = glob.glob(op.join(archives, subject, 'Preprocessed', 'swrsc_tag_ctl_vismot2*.nii'))[0]

    output_dir = op.join(archives, subject, 'results_glm_asl')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Build design matrix
    paradigm_fn = op.join(archives, subject, 'events', 'acquisition1', 'paradigm_' + subject + '_ASL_overlap_paradigm-version-1b.csv')
    paradigm=experimental_paradigm.paradigm_from_csv(paradigm_fn)

    mvt_file = [None, None]
    dm1 = nt.make_design_matrix_asl(paradigm, n_scans, t_r, hrf_length=32., oversampling=16, hrf_model='canonical', prf_model='physio', drift_model='Cosine', drift_order=7, hf_cut=hfcut, mvt_file=mvt_file[0])
    paradigm_fn = op.join(archives, subject, 'events', 'acquisition1', 'paradigm_' + subject + '_ASL_overlap_paradigm-version-2b.csv')
    paradigm=experimental_paradigm.paradigm_from_csv(paradigm_fn)
    dm2 =nt.make_design_matrix_asl(paradigm, n_scans, t_r, hrf_length=32., oversampling=16, hrf_model='canonical', prf_model='physio', drift_model='Cosine', drift_order=7, hf_cut=hfcut, mvt_file=mvt_file[1])
    names = dm1.columns.values

    # Contrasts
    contrasts = {}
    contrast_matrix = np.eye(len(names))
    for i in np.arange(0, len(names)):
        contrasts[names[i]] = contrast_matrix[i]
    contrasts = OrderedDict({'motor_audio_R-L': contrasts['motor_audio_right'] - contrasts['motor_audio_left'], 'motor_audio_L-R': contrasts['motor_audio_left'] - contrasts['motor_audio_right'], 'visual': contrasts['visual'], 'perfusion_motor_audio_R-L': contrasts['perfusion_motor_audio_right'] - contrasts['perfusion_motor_audio_left'], 'perfusion_motor_audio_L-R': contrasts['perfusion_motor_audio_left'] - contrasts['perfusion_motor_audio_right'], 'perfusion_visual': contrasts['perfusion_visual'], 'perfusion_baseline': contrasts['perfusion_baseline']})

    # First level GLM
    data_img1, data_img2, anat_img = nb.load(data_fn1), nb.load(data_fn2), nb.load(anat_fn)
    if subject[:2] in ("MB", "MP"):
        data_img1 = nb.Nifti1Image(data_img1.get_data(), affine=affine)
        data_img2 = nb.Nifti1Image(data_img2.get_data(), affine=affine)

    niimg1 = masker.inverse_transform(masker.transform(data_img1))
    subject_data = nt.run_glm_nistats_contrasts(subject, output_dir, niimg1, dm1, contrasts, mask_img, noise_model='ols', nsession=1)
    first_levels.append(subject_data)
    niimg2 = masker.inverse_transform(masker.transform(data_img2))
    subject_data = nt.run_glm_nistats_contrasts(subject, output_dir, niimg2, dm2, contrasts, mask_img, noise_model='ols', nsession=2)
    first_levels.append(subject_data)



# run second-level GLM

output_dir = op.join(archives, 'results_glm_asl')
if not op.exists(output_dir):
    os.makedirs(output_dir)

group_z_map = group_one_sample_t_test([subject_data["mask"] for subject_data in first_levels], [subject_data["effects_maps"] for subject_data in first_levels], first_levels[0]["contrasts"], output_dir)

map_type = 'z'
group_stats = []
for contrast_id, file_name in group_z_map.iteritems():
    zmap = nb.load(file_name)
    if "perfusion_baseline"==contrast_id:
        p = 0.0000001
        img = zmap
    elif 'perfusion' in contrast_id:
        p = 0.3
        img, zth = map_threshold(zmap, mask_img=mask_img, threshold=0.001, height_control='fdr', cluster_threshold=50)
    else:
        p = 0.3
        img, zth = map_threshold(zmap, mask_img=mask_img, threshold=0.001, height_control='fdr', cluster_threshold=50)
    plot_stat_map(zmap, title=contrast_id, display_mode='ortho', threshold=(zmap.get_data().max()*p))
    plt.savefig(op.join(output_dir, 'group_level_%s.png' % (contrast_id)))
    group_stats.append(np.ravel(masker.transform(zmap)))
    plot_stat_map(img, title=contrast_id, display_mode='ortho')
    plt.savefig(op.join(output_dir, 'group_level_th_%s.png' % (contrast_id)))


# Checking individual maps vs group maps

n_contrasts = len(contrasts)
n_subjects = len(first_levels)
group_stats = np.array(group_stats)
all_cross_correlation = np.zeros((n_subjects, n_contrasts, n_contrasts))
f, ax = plt.subplots(len(first_levels)/5, 5, sharex=True, sharey=True, figsize=(13, 0.5*len(first_levels)))
for i, subject_data in enumerate(first_levels):
    subject = subject_names[np.floor(i/2.).astype(int)]
    individual_stats = np.array([np.ravel(masker.transform(map_)) for c_id, map_ in subject_data["z_maps"].iteritems()])
    contrast_list = [c_id for c_id, map_ in subject_data["z_maps"].iteritems()]
    correlation = np.corrcoef(np.vstack((individual_stats, group_stats)))
    cross_correlation = correlation[:n_contrasts, n_contrasts:]
    all_cross_correlation[i] = cross_correlation
    im = plt.imshow(cross_correlation, interpolation='nearest', vmin=-1, vmax=1)
    ax[np.floor(i/5), i%5].imshow(cross_correlation, interpolation='nearest', vmin=-1, vmax=1)
    ax[np.floor(i/5), i%5].set_title(subject)
    ax[np.floor(i/5), i%5].set_xticks(np.arange(n_contrasts))
    ax[np.floor(i/5), i%5].set_yticks(np.arange(n_contrasts))
    ax[np.floor(i/5), i%5].set_xticklabels(contrast_list, rotation=60, ha='right')
    ax[np.floor(i/5), i%5].set_yticklabels(contrast_list)
cax = f.add_axes([0.95, 0.1, 0.03, 0.8]) # [left, bottom, width, height]
f.colorbar(im, cax=cax)
f.savefig(op.join(output_dir, 'similarity_between_subjects_and_group.png'))


average_cross_correlation = np.mean(all_cross_correlation, axis=0)

plt.figure(figsize=(6, 6))
plt.title("average across-subject cross-correlations")
im = plt.imshow(average_cross_correlation, interpolation='nearest', vmin=-1., vmax=1.)
plt.colorbar(im)
plt.xticks(np.arange(n_contrasts), contrast_list, rotation=60, ha='right')
plt.yticks(np.arange(n_contrasts), contrast_list)
plt.subplots_adjust(bottom=.45, right=1., top=1., left=.4)
plt.savefig(op.join(output_dir, 'average_similarity_between_subjects_and_group.png'))

pos = np.arange(all_cross_correlation.shape[0]) + .5
scores = [np.diag(matrix).mean() for matrix in all_cross_correlation]
plt.figure()
plt.barh(pos, scores, align="center")
plt.yticks(pos*2, subject_names)
plt.axvline(np.median(scores), linestyle="--", c="k", linewidth=2)
plt.axis('tight')
plt.title("average across-subject cross-correlations")
plt.savefig(op.join(output_dir,'mean_diagonal_corr.png'))
