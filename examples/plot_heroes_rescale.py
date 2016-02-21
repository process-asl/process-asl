"""
================
Rescaling demo
================

This example compares a volume before and after T1 correction.
"""
# Give the path to the 4D ASL image
import os
from procasl import datasets
heroes = datasets.load_heroes_dataset(
    subjects=(0,),
    subjects_parent_directory=os.path.join(
        os.path.expanduser('~/procasl_data'), 'heroes'),
    dataset_pattern={'raw ASL': 'fMRI/acquisition1/vismot1_rawASL*.nii'})
raw_asl_file = heroes['raw ASL'][0]

# Create a memory context
from nipype.caching import Memory
cache_directory = '/tmp'
mem = Memory('/tmp')
os.chdir(cache_directory)
# Rescale
from procasl import preprocessing
rescale = mem.cache(preprocessing.Rescale)
out_rescale = rescale(
    in_file=raw_asl_file, ss_tr=35.4, t_i_1=800., t_i_2=1800.)

# Plot the first volume before and after rescaling
from nilearn import plotting
import matplotlib.pylab as plt
for filename, title in zip(
        [raw_asl_file, out_rescale.outputs.rescaled_file],
        ['raw', 'rescaled']):
    figure = plt.figure(figsize=(5, 4))
    volume_file = preprocessing.save_first_scan(filename)
    plotting.plot_img(volume_file, figure=figure, display_mode='z',
                      cut_coords=(65,), title=title, colorbar=True)
plt.show()
