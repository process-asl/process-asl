"""
==========================
Multiple subjects pipeline
==========================

A basic multiple subjects pipeline for ASL. CBF maps are normalized to
the reference MNI template.
"""
import os
import matplotlib.pylab as plt

import nipype.interfaces.spm as spm
from nipype.caching import Memory
from nilearn import plotting

from procasl import preprocessing, quantification, datasets, _utils

paths = ['/home/aina/Software/spm8']

# Load the dataset
subjects_parent_directory = os.path.join(os.path.expanduser('~/Data'),
                                         'HEROES_DB', 'Subjects')
heroes = datasets.load_heroes_dataset(
    subjects=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14),
    subjects_parent_directory=subjects_parent_directory,
    paths_patterns={'anat': 't1mri/acquisition1/anat*.nii',
                    'basal ASL': 'fMRI/acquisition1/basal_rawASL*.nii'})
current_directory = os.getcwd()

# Loop over subjects
for (func_file, anat_file) in zip(
        heroes['basal ASL'], heroes['anat']):
    # Create a memory context
    subject_directory = os.path.relpath(anat_file, subjects_parent_directory)
    subject_directory = subject_directory.split(os.sep)[0]
    cache_directory = os.path.join(os.path.expanduser('~/Data/HEROES_DB'),
                                   'procasl_cache', subject_directory)
    if not os.path.exists(cache_directory):
        os.mkdir(cache_directory)

    # nipype saves .m scripts into cwd
    os.chdir(cache_directory)
    mem = Memory(cache_directory)

    # Get Tag/Control sequence
    get_tag_ctl = mem.cache(preprocessing.RemoveFirstScanControl)
    out_get_tag_ctl = get_tag_ctl(in_file=func_file)

    # Rescale
    rescale = mem.cache(preprocessing.Rescale)
    out_rescale = rescale(in_file=out_get_tag_ctl.outputs.tag_ctl_file,
                          ss_tr=35.4, t_i_1=800., t_i_2=1800.)

    # Realign to first scan
    realign = mem.cache(preprocessing.ControlTagRealign)
    out_realign = realign(
        in_file=out_rescale.outputs.rescaled_file,
        register_to_mean=False,
        correct_tagging=True)

    # Compute mean ASL
    average = mem.cache(preprocessing.Average)
    out_average = average(in_file=out_realign.outputs.realigned_files)

    # Segment anat
    segment = mem.cache(spm.Segment)
    out_segment = segment(
        data=anat_file,
        gm_output_type=[True, False, True],
        wm_output_type=[True, False, True],
        csf_output_type=[True, False, True],
        save_bias_corrected=True,
        paths=paths)

    # Coregister anat to mean ASL
    coregister_anat = mem.cache(spm.Coregister)
    out_coregister_anat = coregister_anat(
        target=out_average.outputs.mean_image,
        source=anat_file,
        apply_to_files=[out_segment.outputs.native_gm_image,
                        out_segment.outputs.native_wm_image],
        write_interp=3,
        jobtype='estwrite',
        paths=paths)

    # Get M0
    get_m0 = mem.cache(preprocessing.GetFirstScan)
    out_get_m0 = get_m0(in_file=func_file)

    # Coregister M0 to mean ASL
    coregister_m0 = mem.cache(spm.Coregister)
    out_coregister_m0 = coregister_m0(
        target=out_average.outputs.mean_image,
        source=out_get_m0.outputs.m0_file,
        write_interp=3,
        jobtype='estwrite',
        paths=paths)

    # Smooth M0
    smooth_m0 = mem.cache(spm.Smooth)
    out_smooth_m0 = smooth_m0(
        in_files=out_coregister_m0.outputs.coregistered_source,
        fwhm=[5., 5., 5.],
        paths=paths)

    # Compute perfusion
    n_scans = preprocessing.get_scans_number(
        out_realign.outputs.realigned_files)
    ctl_scans = range(1, n_scans, 2)
    tag_scans = range(0, n_scans, 2)
    perfusion_file = quantification.compute_perfusion(
        out_realign.outputs.realigned_files,
        ctl_scans=ctl_scans,
        tag_scans=tag_scans)

    # Compute CBF
    quantify = mem.cache(quantification.QuantifyCBF)
    out_quantify = quantify(
        perfusion_file=perfusion_file,
        m0_file=out_smooth_m0.outputs.smoothed_files,
        tr=2500.,
        t1_gm=1331.)

    # Compute brain mask
    brain_mask_file = preprocessing.compute_brain_mask(
        out_coregister_anat.outputs.coregistered_source, frac=.2)

    # Normalize CBF
    normalize = mem.cache(spm.Normalize)
    out_normalize = normalize(
        parameter_file=out_segment.outputs.transformation_mat,
        apply_to_files=[out_quantify.outputs.cbf_file,
                        brain_mask_file],
        write_voxel_sizes=_utils.get_voxel_dims(func_file),
        write_interp=2,
        jobtype='write',
        paths=paths)

    # Mask CBF map with brain mask
    cbf_map = preprocessing.apply_mask(
        out_normalize.outputs.normalized_files[0],
        out_normalize.outputs.normalized_files[1])

    # Plot CBF map on top of MNI template
    plotting.plot_stat_map(
        cbf_map,
        bg_img='/usr/share/fsl/5.0/data/standard/MNI152_T1_2mm.nii.gz',
        threshold=.1, vmax=150.,
        display_mode='z')
    plt.show()
os.chdir(current_directory)
