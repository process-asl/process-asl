"""
================
Realignment demo
================

This example compares standard realignement to realignement with tagging
correction.
"""
######################################################
# Load 4D ASL image of HEROES dataset first subject
import os
from procasl import datasets, preprocessing
heroes = datasets.load_heroes_dataset(
    subjects=(0,),
    subjects_parent_directory=os.path.join(
        os.path.expanduser('~/procasl_data'), 'heroes'),
    paths_patterns={'raw ASL': 'fMRI/acquisition1/vismot1_rawASL*.nii'})
raw_asl_file = heroes['raw ASL'][0]

# Create a memory context
from nipype.caching import Memory
cache_directory = '/tmp'
mem = Memory('/tmp')
os.chdir(cache_directory)

realign = mem.cache(preprocessing.ControlTagRealign)

######################################################
# Realign to first scan
out_realign = realign(in_file=raw_asl_file)
import numpy as np
import matplotlib.pylab as plt
plt.plot(np.loadtxt(out_realign.outputs.realignment_parameters))

######################################################
# Realign to first control scan
out_realign = realign(in_file=raw_asl_file,
                      realign_to_first_ctl=True)
plt.plot(np.loadtxt(out_realign.outputs.realignment_parameters))

######################################################
# Register to mean
out_realign = realign(in_file=raw_asl_file,
                      register_to_mean=True)
plt.plot(np.loadtxt(out_realign.outputs.realignment_parameters))

######################################################
# Do not correct for tagging
out_realign = realign(in_file=raw_asl_file,
                      correct_tagging=False)
plt.plot(np.loadtxt(out_realign.outputs.realignment_parameters))