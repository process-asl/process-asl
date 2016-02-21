"""
===============================
Multiple subjects pipeline demo
===============================

A basic multiple subjects pipeline for BOLD data.
"""
import os
import matplotlib.pylab as plt

import nipype.interfaces.spm as spm
from nipype.caching import Memory
from nilearn import plotting

from procasl import datasets, preprocessing, _utils


# Load the dataset
subjects_parent_directory = os.path.join(os.path.expanduser('~/procasl_data'),
                                         'heroes')
heroes = datasets.load_heroes_dataset(
    subjects=(0, 4, 9),
    subjects_parent_directory=subjects_parent_directory,
    dataset_pattern={'anat': 't1mri/acquisition1/anat*.nii',
                     'BOLD EPI': 'fMRI/acquisition1/vismot1_BOLDepi*.nii'})
current_directory = os.getcwd()

# Loop over subjects
for (func_file, anat_file) in zip(
        heroes['BOLD EPI'], heroes['anat']):
    # Create a memory context
    subject_directory = os.path.relpath(anat_file, subjects_parent_directory)
    subject_directory = subject_directory.split(os.sep)[0]
    cache_directory = os.path.join(os.path.expanduser('~/CODE/process-asl'),
                                   'procasl_cache', 'heroes',
                                   subject_directory)
    if not os.path.exists(cache_directory):
        os.mkdir(cache_directory)

    os.chdir(cache_directory)  # nipype saves .m scripts in current directory
    mem = Memory(cache_directory)

    # Realign EPIs
    realign = mem.cache(spm.Realign)
    out_realign = realign(
        in_files=func_file,
        register_to_mean=True)

    # Coregister anat to mean EPIs
    coregister = mem.cache(spm.Coregister)
    out_coregister = coregister(
        target=out_realign.outputs.mean_image,
        source=anat_file,
        write_interp=3,
        jobtype='estimate')

    # Segment anat
    segment = mem.cache(spm.Segment)
    out_segment = segment(
        data=anat_file,
        gm_output_type=[True, False, True],
        wm_output_type=[True, False, True],
        csf_output_type=[True, False, True],
        save_bias_corrected=True)

    # Normalize anat
    normalize_anat = mem.cache(spm.Normalize)
    out_normalize_anat = normalize_anat(
        parameter_file=out_segment.outputs.transformation_mat,
        apply_to_files=[out_coregister.outputs.coregistered_source],
        write_voxel_sizes=_utils.get_voxel_dims(anat_file),
        write_interp=1,
        jobtype='write')

    # Normalize EPIs
    normalize_func = mem.cache(spm.Normalize)
    out_normalize_func = normalize_func(
        parameter_file=out_segment.outputs.transformation_mat,
        apply_to_files=[out_realign.outputs.realigned_files,
                        out_segment.outputs.native_gm_image,
                        out_segment.outputs.native_wm_image,
                        out_segment.outputs.native_csf_image],
        write_voxel_sizes=_utils.get_voxel_dims(func_file),
        write_interp=1,
        jobtype='write')

    # Smooth EPIs
    smooth = mem.cache(spm.Smooth)
    out_smooth = smooth(
        in_files=out_normalize_func.outputs.normalized_files[0],
        fwhm=[5., 5., 5.])

    # Plot mean smoothed EPI
    average = preprocessing.Average()
    average.inputs.in_file = out_smooth.outputs.smoothed_files
    out_average = average.run()
    plotting.plot_epi(out_average.outputs.mean_image)
    plt.show()

os.chdir(current_directory)
