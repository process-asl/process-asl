import os

import pytest

import nipype.interfaces.spm as spm
from nipype.interfaces.spm import no_spm
import nipype.interfaces.matlab as mlab

try:
    matlab_cmd = os.environ['MATLABCMD']
except:
    matlab_cmd = 'matlab'

mlab.MatlabCommand.set_default_matlab_cmd(matlab_cmd)


def test_slicetiming():
    assert spm.SliceTiming._jobtype == 'temporal'
    assert spm.SliceTiming._jobname == 'st'


def test_realign():
    assert spm.Realign._jobtype == 'spatial'
    assert spm.Realign._jobname == 'realign'
    assert spm.Realign().inputs.jobtype == 'estwrite'