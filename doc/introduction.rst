=====================
Introduction: procasl
=====================

.. contents:: **Contents**
    :local:
    :depth: 1


What is procasl: Preprocessing and quantification of ASL
========================================================

    Procasl builds relevant **pipelines** for processing Arterial Spin Labeling data.
    For the moment, it is dedicated to pulsed ASL.

ASL basics
==========

Weighting the MRI signal by cerebral blood flow
-----------------------------------------------
*"Aristotle taught that the brain exists merely to cool the blood and is not involved in the process of thinking. This is true only of certain persons." -Will Cuppy*

Blood flows from carotid arteries to capilleray bed. The volume of arterial blood (mL) delivered to 100 g of tissue per minute is called the Cerebral Blood Flow (CBF). In human brain, it has a typical value of 60 mL/100 g-minute. CBF is a fundamental physiological quantity, closely related to brain function. It is an important indicator of tissue health as well as neuronal activity.

Prior to ASL, the techniques used for determining cerebral blood flow were rather invasive and involved the 
use of exogenous contrast agents, such as the 15O H2O radiotracer in Positron Emission Tomography (PET).


Imaging principle: Labeling arterial blood water magnetically
-------------------------------------------------------------
Similar to these techniques, ASL is based on tracer kinetics but with no contrast agent: blood is used as a tracer by inverting its longitudinal relaxation.

Magnetization is tagged sequentially to capture the flowing of the arterial blood.

First, arterial blood water is magnetically labeled just below the region (slice) of interest by applying a 180 degree radiofrequency (RF) inversion pulse. The result of this pulse is inversion of the net magnetization of the blood water. After a period of time (called the transit time), this 'paramagnetic tracer’ flows into the slice of interest where it exchanges with tissue water. The inflowing inverted spins within the blood water alter total tissue magnetization, reducing it and, consequently, the MR signal & image intensity. During this time, an image is taken (called the tag image). 
The experiment is then repeated without labeling the arterial blood to create another image (called the control image). The control image and the tag image are subtracted to produce a perfusion image. This image will reflect the amount of arterial blood delivered to each voxel within the slice within the transit time.

Many modalities and many sequences exist, and ASL techniques is an active research field.


From ASL images to CBF: Quantification how tos
----------------------------------------------
CBF is proportional to the difference in magnetization between control and labeled images.

To obtain absolute perfusion (CBF) in ml/100ml/min,
the mean difference image is multiplied by constant factors.
This scaling is essential for “whole brain” diseases,
like large vessel occlusive disease, neurodegenerative
disease, sickle cell disease, etc . It plays an important role when
comparing patients to controls, to avoid misinterpreting a difference in
these factors (T1, labeling efficiency) as a difference in brain perfusion.

Some factors, such as transit time or labeling efficiency for
different arteries, vary within the ASL-image and therefore will
change relative perfusion. 

- How does disease affect blood flow distribution

- Relative perfusion of tumor compared to normal appearing GM/WM

- Comparison with contralateral hemisphere

Correction for regional differences is thus important and can be done through
more elaborate methods and additionnal acquisitions.
This model is simplified, but is recommended for its
robustness and simplicity, and because more complete
models require additional information that involves
more scan time, and often only reduces systematic errors
at the cost of SNR. Types of additional information
include ATT, water exchange rates and times between
blood and tissue, tissue T1 values, and tissue segmentation.
Ongoing active research aims to more fully understand the range and effects of these parameters, but the
complexity, uncertainty, and additional noise associated
with correcting for these factors was deemed to be counterproductive
as a default protocol at this stage of adoption of clinical ASL.


:General kinetic model:

:The parameters:


Applications
------------
The cerebral blood flow (CBF) is a fundamental physiological quantity, closely related to brain function.

:Diagnosis:

:An accurate hemodynamics estimation:

:Assessing the neural contribution in activation BOLD signal:


Whetting Your Appetite
----------------------
To complete.

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
* Pandas >= 0.13

as well as the interfaces:

* FSL >= 4.1.0
* SPM8/12

If you want to run the tests, you need nose >= 1.2.1

Interfaces configuration
========================
**Configuring FSL**: On an Ubuntu system, FSL is usually installed at :: /usr/share/fsl. You need to add this location to your .bashrc file. Edit this file by running the shell command::

    gedit ~/.bashrc

and add the following lines::

    # FSL
    FSLDIR=/usr/share/fsl
    . ${FSLDIR}/5.0/etc/fslconf/fsl.sh
    PATH=${FSLDIR}/5.0/bin:${PATH}
    export FSLDIR PATH

To test if FSL is correctly installed, open a new terminal and type in the shell command::

    fsl

You should see the FSL GUI with the version number in the header.

**Configuring SPM**: Add the following lines specifying the location of the spm folder to your .bashrc file::

    # SPM8
    export SPM_PATH=/i2bm/local/spm8-standalone/spm8_mcr/spm8

**Using SPM MCR**: If you don't have a matlab licence, specify the location of the Matlab Compiler Runtime and force the
use of the standalone MCR version of spm by appending the following lines to the .bashrc::

    # SPM MCR
    export SPMMCRCMD='/home/salma/Téléchargements/spm8/run_spm8.sh /home/salma/Téléchargements/MCR/v713 script'
    export FORCE_SPMMCR='True'

Installation
============
For the moment process-asl is available as a development version. To download the source code, run the shell command::

    git clone https://github.com/process-asl/process-asl

In the ``process-asl`` directory created by the previous step, run
(again, as a shell command)::

    python setup.py install --user
