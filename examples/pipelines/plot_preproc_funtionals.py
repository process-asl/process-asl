"""
=========================
Preprocessing functionals
=========================

Standard preprocessing pipeline for functional ASL.
The mean functional in standard space is plotted for the last subject.
"""
# Load the dataset
import os
from procasl import datasets
subjects_parent_directory = os.path.join(os.path.expanduser('~/procasl_data'),
                                         'heroes')
heroes = datasets.load_heroes_dataset(
    subjects=(5, 6,),
    subjects_parent_directory=subjects_parent_directory,
    paths_patterns={'anat': 't1mri/acquisition1/anat*.nii',
                    'func ASL': 'fMRI/acquisition1/vismot1_rawASL*.nii'})

# Loop over subjects
import numpy as np
import nipype.interfaces.spm as spm
from nipype.caching import Memory
from procasl import preprocessing, _utils
current_directory = os.getcwd()
for (func_file, anat_file) in zip(
        heroes['func ASL'], heroes['anat']):
    # Create a memory context
    subject_directory = os.path.relpath(anat_file, subjects_parent_directory)
    subject_directory = subject_directory.split(os.sep)[0]
    cache_directory = os.path.join(os.path.expanduser('~/CODE/process-asl'),
                                   'procasl_cache', 'heroes',
                                   subject_directory)
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
        save_bias_corrected=True)

    # Coregister anat to mean ASL
    coregister_anat = mem.cache(spm.Coregister)
    out_coregister_anat = coregister_anat(
        target=out_average.outputs.mean_image,
        source=anat_file,
        apply_to_files=[out_segment.outputs.native_gm_image,
                        out_segment.outputs.native_wm_image],
        write_interp=3,
        jobtype='estwrite')

    # Normalize func
    normalize = mem.cache(spm.Normalize)
    write_voxel_sizes = np.round(_utils.get_voxel_dims(func_file), 2).tolist()
    out_normalize = normalize(
        parameter_file=out_segment.outputs.transformation_mat,
        apply_to_files=[out_realign.outputs.realigned_files,
                        out_average.outputs.mean_image],
        write_voxel_sizes=write_voxel_sizes,
        write_interp=2,
        jobtype='write')

    # Smooth func
    smooth = mem.cache(spm.Smooth)
    out_smooth = smooth(
        in_files=out_normalize.outputs.normalized_files,
        fwhm=[5., 5., 5.])

os.chdir(current_directory)

# Plot the mean func map for the last subject
from nilearn import plotting
plotting.plot_epi(out_smooth.outputs.smoothed_files[1])
plotting.show()
