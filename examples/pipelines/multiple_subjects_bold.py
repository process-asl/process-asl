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

paths = ['/home/aina/Software/spm12']

# Load the dataset
subjects_parent_directory = os.path.join(os.path.expanduser('~/Data'),
                                         'HEROES_DB', 'Subjects')
heroes = datasets.load_heroes_dataset(
    #subjects=(2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 1), #13, 14, 0, 1 didnt work
    subjects=(1, 0), #13, 14, 0, 1 didnt work
    subjects_parent_directory=subjects_parent_directory,
    paths_patterns={'anat': 't1mri/acquisition1/anat*.nii',
                    'BOLD EPI': 'fMRI/acquisition1/vismot2_BOLDepi*.nii'})
current_directory = os.getcwd()

# Loop over subjects
for (func_file, anat_file) in zip(
        heroes['BOLD EPI'], heroes['anat']):
    # Create a memory context
    subject_directory = os.path.relpath(anat_file, subjects_parent_directory)
    subject_directory = subject_directory.split(os.sep)[0]
    #cache_directory = os.path.join(os.path.expanduser('~/CODE/process-asl'),
    #                               'procasl_cache', 'heroes',
    #                               subject_directory)
    cache_directory = os.path.join(os.path.expanduser('~/Data/HEROES_DB'),
                                   'procasl_cache', subject_directory)
    if not os.path.exists(cache_directory):
        os.mkdir(cache_directory)

    os.chdir(cache_directory)  # nipype saves .m scripts in current directory
    mem = Memory(cache_directory)


    # Slice timing correction
    TR = 2.5
    num_slices = 42
    print func_file
    list_slice = range(1, num_slices+1, 2) + range(2, num_slices+1, 2)
    st_correction = mem.cache(spm.SliceTiming)
    out_stcorr = st_correction(
        in_files=func_file,
        num_slices=num_slices,
        ref_slice=1,
        slice_order=list_slice,
        time_acquisition=TR-(TR/num_slices),
        time_repetition=TR,
        paths=paths)

    # Realign EPIs
    realign = mem.cache(spm.Realign)
    out_realign = realign(
        in_files=out_stcorr.outputs.timecorrected_files,
        register_to_mean=True,
        paths=paths)

    # Coregister anat to mean EPIs
    coregister = mem.cache(spm.Coregister)
    out_coregister = coregister(
        target=out_realign.outputs.mean_image,
        source=anat_file,
        write_interp=3,
        jobtype='estimate',
        paths=paths)

    # Segment anat
    segment = mem.cache(spm.Segment)
    out_segment = segment(
        data=anat_file,
        gm_output_type=[True, False, True],
        wm_output_type=[True, False, True],
        csf_output_type=[True, False, True],
        save_bias_corrected=True,
        paths=paths)

    # Normalize anat
    normalize_anat = mem.cache(spm.Normalize)
    out_normalize_anat = normalize_anat(
        parameter_file=out_segment.outputs.transformation_mat,
        apply_to_files=[out_coregister.outputs.coregistered_source],
        write_voxel_sizes=_utils.get_voxel_dims(anat_file),
        write_interp=1,
        jobtype='write',
        paths=paths)

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
        jobtype='write',
        paths=paths)

    # Smooth EPIs
    smooth = mem.cache(spm.Smooth)
    out_smooth = smooth(
        in_files=out_normalize_func.outputs.normalized_files[0],
        fwhm=[5., 5., 5.],
        paths=paths)

    # Plot mean smoothed EPI
    average = preprocessing.Average()
    average.inputs.in_file = out_smooth.outputs.smoothed_files
    out_average = average.run()
    plotting.plot_epi(out_average.outputs.mean_image)
    #plt.show()

os.chdir(current_directory)
