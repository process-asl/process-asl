"""Functions to use in notebooks
"""
import os
import os.path as op
import logging
import numpy as np
import nibabel
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd

from nilearn.plotting import plot_stat_map
from nilearn.input_data import NiftiMasker
from nistats import glm, experimental_paradigm, design_matrix
from nistats.hemodynamic_models import _gamma_difference_hrf, glover_hrf

from pyhrf import xndarray
import pyhrf.vbjde.vem_tools as vt
from pyhrf.paradigm import Paradigm
from pyhrf.retreat.glm_tools import make_design_matrix_asl


logger = logging.getLogger(__name__)


# DESIGN MATRIX

def make_control_tag_vector(length):
    w = np.ones((length)) * 0.5
    w[1::2] = - 0.5
    return w

from pyhrf.sandbox.physio_params import (linear_rf_operator,
                                         PHY_PARAMS_KHALIDOV11)
from pyhrf.retreat.glm_tools import compute_prf_regressor
from nistats import design_matrix
def make_design_matrix_asl(paradigm, n_scans, t_r, hrf_length=32., oversampling=16, hrf_model='canonical', prf_model='physio', drift_model='polynomial', drift_order=4, hf_cut=100, model = 'ols', mvt_file=None, use_pypreprocess=False):
    dt = t_r / oversampling
    frametimes = np.arange(0, n_scans * t_r, t_r)
    phy_params = PHY_PARAMS_KHALIDOV11
    phy_params['V0'] = 2.
    prf_matrix = linear_rf_operator(hrf_length / dt,  phy_params, dt, calculating_brf=False)

    # Activation BOLD and ASL regressors
    bold_regs = []
    add_regs = []
    add_reg_names = []
    for condition_name in np.unique(paradigm.name):
        onsets = paradigm.onset[paradigm.name == condition_name]
        values = paradigm.modulation[paradigm.name == condition_name]
        try:
            duration = paradigm.duration[paradigm.name == condition_name]
        except:
            duration = np.zeros_like(onsets)
        exp_condition = (onsets, duration, values)
        bold_regs.append(compute_prf_regressor(exp_condition, hrf_model, frametimes, prf_model=prf_model, prf_matrix=prf_matrix, con_id=condition_name, oversampling=16, normalize=True, plot=False)[0])
        reg, reg_name = compute_prf_regressor(exp_condition, hrf_model, frametimes, prf_model=prf_model, prf_matrix=prf_matrix, con_id=condition_name, oversampling=16, normalize=True)
        #reg[0] = 0.
        reg[::2] *= -0.5
        reg[1::2] *= 0.5
        add_regs.append(reg.squeeze())
        add_reg_names.append(reg_name[0] + '_' + condition_name)

    # Baseline ASL regressor
    reg = make_control_tag_vector(n_scans)
    reg = np.ones(n_scans)
    reg[::2] *= -0.5
    reg[1::2] *= 0.5
    add_regs.append(reg)
    add_reg_names.append('perfusion_baseline')

    #reg = np.zeros(n_scans)
    #reg[0] = 1.
    #add_regs.append(reg)
    #add_reg_names.append('m_0')

    bold_regs = np.array(bold_regs).squeeze(axis=-1)
    bold_regs = bold_regs.transpose()
    add_regs = np.array(add_regs).transpose()

    if mvt_file is not None:
        # Motion regressors
        #print np.genfromtxt(mvt_file).shape
        #print add_regs.shape
        add_regs = np.hstack((add_regs, np.genfromtxt(mvt_file, skip_header=0)))
        add_reg_names += ['translation x', 'translation y', 'translation z', 'pitch', 'roll', 'yaw']

    # Create the design matrix
    dm = design_matrix.make_design_matrix(frametimes, paradigm=paradigm,hrf_model=hrf_model, drift_model=drift_model, drift_order=drift_order, add_regs=add_regs, add_reg_names=add_reg_names, period_cut=hf_cut)
    if use_pypreprocess:
        from pypreprocess.external.nistats.design_matrix import make_design_matrix
        dm = make_design_matrix(frametimes, paradigm, hrf_model=hrf_model, drift_model=drift_model, drift_order=drift_order, period_cut=hf_cut, add_regs=add_regs, add_reg_names=add_reg_names)

    return dm



# GLM

def run_glm_subject(bold_fn, dm, noise_model, mask_fn=None, scale=False):
    """ Construct masker and run GLM """
    if scale:
        niimg = get_data_scaled(bold_fn)
    else:
        niimg = nibabel.load(bold_fn)
    if mask_fn is None:
        masker = NiftiMasker(mask_strategy='epi')
    else:
        mask_img = nibabel.load(mask_fn)
        masker = NiftiMasker(mask_img=mask_img)
    niimg_masked = masker.fit_transform(niimg)
    glm_results = glm.session_glm(niimg_masked, dm, noise_model=noise_model)
    labels = glm_results[0]
    reg_results = glm_results[1]
    return labels, reg_results, masker, niimg_masked


from pypreprocess.external.nistats.glm import FMRILinearModel
def run_glm_nistats_contrasts(subject, output_dir, niimg, dm, contrasts, mask_img, noise_model='ols', th=0.2, nsession=1):
    fmri_glm = FMRILinearModel([niimg], [dm], mask='compute')
    fmri_glm.fit(do_scaling=True, model=noise_model)
    #fmri_glm = FirstLevelGLM(t_r=t_r, noise_model='ols', mask=masker)
    #fmri_glm.fit(niimg1, dm1)

    # Compute contrasts
    z_maps = {}
    effects_maps = {}
    for contrast_id, contrast_val in contrasts.iteritems():
        z_map, t_map, eff_map, var_map = fmri_glm.contrast(contrasts[contrast_id], con_id=contrast_id, output_z=True, output_stat=True, output_effects=True, output_variance=True)
        #z_map, t_map, eff_map, var_map = fmri_glm.transform(contrasts[contrast_id], contrast_name=contrast_id,output_z=True, output_stat=True, output_effects=True, output_variance=True)
        if contrast_id=='perfusion_baseline':
            plot_stat_map(z_map, display_mode='z', title=subject+', '+contrast_id, cut_coords=[-5, 5, 15, 30, 50, 60, 70])
        else:
            plot_stat_map(z_map, display_mode='z', title=subject+', '+contrast_id, threshold=z_map.get_data().flatten().max()*th, cut_coords=[-5, 5, 15, 30, 50, 60, 70])
        plt.savefig(op.join(output_dir, subject + '_' + contrast_id + '_zmap' + str(nsession) + '.png'))

        # store stat maps to disk
        for dtype, out_map in zip(['z', 't', 'effects', 'variance'], [z_map, t_map, eff_map, var_map]):
            map_dir = os.path.join(output_dir, '%s_maps' % dtype)
            if not os.path.exists(map_dir):
                os.makedirs(map_dir)
            map_path = os.path.join(map_dir, '%s.nii.gz' % contrast_id)
            nibabel.save(out_map, map_path)

            if dtype == "z":
                z_maps[contrast_id] = map_path
            if dtype == "effects":
                effects_maps[contrast_id] = map_path

    subject_data = {"mask": mask_img, "effects_maps": effects_maps, "z_maps": z_maps, "contrasts": contrasts}

    return subject_data




# GROUP LEVEL STATISTICS
from nilearn.masking import intersect_masks
from nistats.glm import FirstLevelGLM

def group_one_sample_t_test(masks, effects_maps, contrasts, output_dir, t_r):
    """
    Runs a one-sample t-test procedure for group analysis. Here, we are
    for each experimental condition, only interested refuting the null
    hypothesis H0: "The average effect accross the subjects is zero!"
    Parameters
    ----------
    masks: list of strings or nibabel image objects
        subject masks, one per subject
    effects_maps: list of dicts of lists
        effects maps from subject-level GLM; each entry is a dictionary;
        each entry (indexed by condition id) of this dictionary is the
        filename (or correspinding nibabel image object) for the effects
        maps for that condition (aka contrast),for that subject
    contrasts: dictionary of array_likes
        contrasts vectors, indexed by condition id
    """

    # make output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    assert len(masks) == len(effects_maps), (len(masks), len(effects_maps))

    # compute group mask
    group_mask = intersect_masks(masks)

    # construct design matrix (only one covariate, namely the "mean effect")
    design_matrix = pd.DataFrame(np.ones(len(effects_maps)
                            )[:, np.newaxis])  # only the intercept

    group_level_z_maps = {}
    group_level_t_maps = {}
    for contrast_id in contrasts:
        print "\tcontrast id: %s" % contrast_id

        # effects maps will be the input to the second level GLM
        first_level_image = nibabel.concat_images(
            [x[contrast_id] for x in effects_maps])

        # fit 2nd level GLM for given contrast
        group_model = FirstLevelGLM(t_r=t_r, noise_model='ols')
        group_model.fit(first_level_image, design_matrix)

        # specify and estimate the contrast
        contrast_val = np.array(([[1.]])
                                )  # the only possible contrast !
        z_map, t_map = group_model.transform(
            contrast_val, contrast_name='one_sample %s' % contrast_id, output_z=True, output_stat=True)

        # save map
        for map_type, map_img in zip(["z", "t"], [z_map, t_map]):
            map_dir = os.path.join(output_dir, '%s_maps' % map_type)
            if not os.path.exists(map_dir):
                os.makedirs(map_dir)
            map_path = os.path.join(map_dir, 'group_level_%s.nii.gz' % (
                    contrast_id))
            print "\t\tWriting %s ..." % map_path
            nibabel.save(map_img, map_path)
            if map_type == "z":
                group_level_z_maps[contrast_id] = map_path
            elif map_type == "t":
                group_level_t_maps[contrast_id] = map_path

    return group_level_z_maps, group_level_t_maps

