import matplotlib.pyplot as plt

import os
import os.path as op
import numpy as np
import nibabel as nb
import glob
import pandas as pd

from nistats import experimental_paradigm, design_matrix, glm
from nistats.glm import FirstLevelGLM
from nistats.thresholding import map_threshold
from nilearn.input_data import NiftiMasker
from pypreprocess.reporting.glm_reporter import generate_subject_stats_report, group_one_sample_t_test

from . import nistats_processing_tools as nt


archives = op.join('/home', 'aina', 'Data', 'HEROES_preprocessed')

t_r = 2.5
n_scans = 165
frametimes = np.linspace(0, (n_scans - 1) * t_r, n_scans)
drift_model = 'Cosine'
hrf_model = 'Canonical'
epoch_duration = 10 * t_r
hfcut = 2 * 2 * epoch_duration

subject_names = ["AC150013", "CD110147", "KP140463", "MP130368", "SC120530",
                 "NS110383", "PB130006", "SB150062", "MB140004", "XM140202",
                 "RG130377", "SH140342", "AM140315", "HB140194", "LZ140352"]



from nilearn.plotting import plot_stat_map
import nilearn
import scipy

first_levels = []

mask_fn = op.join(archives, 'gm_mask_willard_bold.nii')
mask_img = nb.load(mask_fn)
affine = mask_img.get_affine()

for subject in subject_names:
    print subject
    #mvt_file = glob.glob(op.join(archives, subject, 'Preprocessed', 'rp_sc_tag_ctl_vismot2.txt'))
    data_fn1 = glob.glob(op.join(archives, subject, 'Preprocessed', 'swravismot1*_BOLD*.nii'))[0]
    data_fn2 = glob.glob(op.join(archives, subject, 'Preprocessed', 'swravismot2*_BOLD*.nii'))[0]
    anat_fn = glob.glob(op.join(archives, subject, 'Preprocessed', 'wranat_*.nii'))[0]

    output_dir = op.join(archives, subject, 'results_glm')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Build design matrix
    paradigm_fn = op.join(archives, subject, 'events', 'acquisition1', 'paradigm_' + subject + '_BOLD_overlap_paradigm-version-1b.csv')
    paradigm1=experimental_paradigm.paradigm_from_csv(paradigm_fn)
    dm1 = design_matrix.make_design_matrix(frametimes, paradigm=paradigm1, hrf_model=hrf_model, drift_model=drift_model, period_cut=hfcut)
    paradigm_fn = op.join(archives, subject, 'events', 'acquisition1', 'paradigm_' + subject + '_BOLD_overlap_paradigm-version-2b.csv')
    paradigm2=experimental_paradigm.paradigm_from_csv(paradigm_fn)
    dm2 = design_matrix.make_design_matrix(frametimes, paradigm=paradigm2, hrf_model=hrf_model, drift_model=drift_model, period_cut=hfcut)
    names = dm1.columns.values

    # Contrasts
    contrasts = {}
    contrast_matrix = np.eye(len(names))
    for i in range(len(names)):
        contrasts[names[i]] = contrast_matrix[i]
    contrasts = {'motor_audio_L-R': contrasts['motor_audio_left'] - contrasts['motor_audio_right'],
                 'motor_audio_R-L': contrasts['motor_audio_right'] - contrasts['motor_audio_left'],
                 'visual': contrasts['visual']}

    # First level GLM
    data_img1, data_img2, anat_img = nb.load(data_fn1), nb.load(data_fn2), nb.load(anat_fn)
    masker = NiftiMasker(mask_img=mask_fn).fit()
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

output_dir = op.join(archives, 'results_glm')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

group_z_map = group_one_sample_t_test([subject_data["mask"] for subject_data in first_levels], [subject_data["effects_maps"] for subject_data in first_levels], first_levels[0]["contrasts"], output_dir, t_r=t_r)

group_stats = []
map_type = 'z'
map_dir = os.path.join(output_dir, '%s_maps' % map_type)
for contrast_id, file_name in group_z_map.iteritems():
    map_path = os.path.join(map_dir, 'group_level_%s.nii.gz' % (contrast_id))
    tmap = nb.load(map_path)
    plot_stat_map(tmap, title=contrast_id, display_mode='ortho', threshold=(tmap.get_data().max()*0.3))
    plt.savefig(op.join(map_dir,'group_level_%s.png' % (contrast_id)))
    group_stats.append(np.ravel(masker.transform(tmap)))
    img, zth = map_threshold(tmap, mask_img=mask_img, threshold=0.00001, height_control='fdr', cluster_threshold=50)
    #threshold: 'fpr'|'fdr'|'bonferroni'|'none'
    plot_stat_map(img, title=contrast_id, display_mode='ortho')
    plt.savefig(op.join(map_dir,'group_level_th_%s.png' % (contrast_id)))



# Checking maps... individual vs group

n_contrasts = len(contrasts)
n_subjects = len(first_levels)
group_stats = np.array(group_stats)
all_cross_correlation = np.zeros((n_subjects, n_contrasts, n_contrasts))
f, ax = plt.subplots(6, 5, sharex=True, sharey=True, figsize=(13, 0.5 * len(first_levels)))
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


plt.figure()
pos = np.arange(all_cross_correlation.shape[0]) + .5
scores = [np.diag(matrix).mean() for matrix in all_cross_correlation]
plt.barh(pos, scores, align="center")
plt.yticks(pos*2, subject_names)
plt.axvline(np.median(scores), linestyle="--", c="k", linewidth=2)
plt.axis('tight')
plt.title("average across-subject cross-correlations")
plt.savefig(op.join(output_dir,'mean_diagonal_corr.png'))
