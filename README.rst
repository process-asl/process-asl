.. -*- mode: rst -*-
process-asl
===========

Process-asl builds relevant **pipelines** for processing Arterial Spin Labeling data.
For the moment, it is dedicated to pulsed ASL.

Dependencies
============

The required dependencies to use the software are the python packages:

* Python 2.7
* setuptools
* Numpy >= 1.6.2
* SciPy >= 0.11
* Nibabel >= 2.0.1
* Nilearn >= 0.1.3
* Matplotlib >= 1.1.1
* Nipype 0.11.0
* NetworkX >= 1.7
* Enthought Traits >= 4.3
* Dateutil >= 1.5

as well as the interfaces:

* FSL >= 4.1.0
* SPM8/12

Installation
============

For the moment process-asl is available as a development version. You can download the source code with the command::

    git clone https://github.com/process-asl/process-asl
