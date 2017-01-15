"""
================
Rescaling demo
================

This example compares a volume before and after T1 correction.
"""
# Load functional ASL image of KIRBY dataset first subject
import os
from procasl import datasets
kirby = datasets.fetch_kirby(subjects=[4])
raw_asl_file = kirby.asl[0]

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
    first_scan_file = preprocessing.save_first_scan(filename)
    plotting.plot_img(first_scan_file, figure=figure, display_mode='z',
                      cut_coords=(65,), title=title, colorbar=True)
plt.show()
