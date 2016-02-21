"""
================
Realignment demo
================

This example compares standard realignement to realignement with tagging
correction.
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

# Realign with and without tagging correction
from procasl import preprocessing
import numpy as np
realign = mem.cache(preprocessing.Realign)
x_translation = {}
for correct_tagging in [True, False]:
    out_realign = realign(in_file=raw_asl_file,
                          correct_tagging=correct_tagging)
    x_translation[correct_tagging] = np.loadtxt(
        out_realign.outputs.realignment_parameters)[:, 2]

# Plot x-translation parameters with and without tagging correction
import matplotlib.pylab as plt
plt.figure(figsize=(10, 5))
for correct_tagging, label, color in zip([True, False],
                                         ['corrected', 'uncorrected'], 'rb'):
    plt.plot(x_translation[correct_tagging], label=label, color=color)
plt.ylabel('translation in x [mm]')
plt.legend()
figure = plt.figure(figsize=(10, 5))
plt.plot(x_translation[True] - x_translation[False], color='g')
plt.ylabel('difference [mm]')
plt.xlabel('frame')
figure.suptitle('Impact of tagging correction')
plt.show()
