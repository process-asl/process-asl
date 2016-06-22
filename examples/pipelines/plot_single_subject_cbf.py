"""
==================
Single subject CBF
==================

A basic single subject pipeline for computing CBF from basal ASL data in native
space.
The computed map is compared to the basal CBF map output of the scanner.
"""
# Give the path to the 4D ASL and anatomical images
import os
from procasl import datasets
heroes = datasets.load_heroes_dataset(
    subjects=(0,),
    subjects_parent_directory=os.path.join(
        os.path.expanduser('~/procasl_data'), 'heroes'),
    paths_patterns={'anat': 't1mri/acquisition1/anat*.nii',
                    'basal ASL': 'fMRI/acquisition1/basal_rawASL*.nii',
                    'basal CBF': 'fMRI/acquisition1/basal_relCBF*.nii'})
asl_file = heroes['basal ASL'][0]
anat_file = heroes['anat'][0]

# Create a memory context
from nipype.caching import Memory
current_directory = os.getcwd()
cache_directory = '/tmp'
os.chdir(cache_directory)
mem = Memory(cache_directory)

# Get Tag/Control sequence
from procasl import preprocessing
get_tag_ctl = mem.cache(preprocessing.RemoveFirstScanControl)
out_get_tag_ctl = get_tag_ctl(in_file=asl_file)

# Rescale
rescale = mem.cache(preprocessing.Rescale)
out_rescale = rescale(in_file=out_get_tag_ctl.outputs.tag_ctl_file,
                      ss_tr=35.4, t_i_1=800., t_i_2=1800.)

# Realign to first scan
realign = mem.cache(preprocessing.Realign)
out_realign = realign(
    in_file=out_rescale.outputs.rescaled_file,
    register_to_mean=False,
    correct_tagging=True)

# Compute mean ASL
average = mem.cache(preprocessing.Average)
out_average = average(in_file=out_realign.outputs.realigned_files)

# Segment anat
import nipype.interfaces.spm as spm
segment = mem.cache(spm.Segment)
out_segment = segment(
    data=anat_file,
    gm_output_type=[False, False, True],
    wm_output_type=[False, False, True],
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

# Get M0
get_m0 = mem.cache(preprocessing.GetFirstScan)
out_get_m0 = get_m0(in_file=asl_file)

# Coregister M0 to mean ASL
coregister_m0 = mem.cache(spm.Coregister)
out_coregister_m0 = coregister_m0(
    target=out_average.outputs.mean_image,
    source=out_get_m0.outputs.m0_file,
    write_interp=3,
    jobtype='estwrite')

# Smooth M0
smooth_m0 = mem.cache(spm.Smooth)
out_smooth_m0 = smooth_m0(
    in_files=out_coregister_m0.outputs.coregistered_source,
    fwhm=[5., 5., 5.])

# Compute perfusion
from procasl import preprocessing, quantification
n_scans = preprocessing.get_scans_number(out_realign.outputs.realigned_files)
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

# Mask CBF map with brain mask
brain_mask_file = preprocessing.compute_brain_mask(
    out_coregister_anat.outputs.coregistered_source, frac=.2)
cbf_map = preprocessing.apply_mask(out_quantify.outputs.cbf_file,
                                   brain_mask_file)
os.chdir(current_directory)

# Plot CBF map on top of anat
import matplotlib.pylab as plt
from nilearn import plotting
for map_to_plot, title, vmax, threshold in zip(
    [cbf_map, heroes['basal CBF'][0]], ['pipeline CBF', 'scanner CBF'],
    [150., 1500.], [1., 10.]):  # scanner CBF maps are scaled
    plotting.plot_stat_map(
        map_to_plot,
        bg_img=out_coregister_anat.outputs.coregistered_source,
        threshold=threshold, vmax=vmax, cut_coords=(-15, 0, 15, 45, 60, 75,),
        display_mode='z', title=title)

plt.show()
