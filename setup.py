import os
from setuptools import setup


# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="procasl",
    version="alpha",
    maintainer="Salma Bougacha",
    maintainer_email="salmabougacha@hotmail.com",
    description=("Arterial Spin Labeling image processing in python."),
    license="BSD",
    keywords="Arterial Spin Labeling",
    url="https://github.com/process-asl/process-asl",
    packages=['procasl', ],
    long_description=read('README.md'),
    classifiers=[
        "Topic :: Scientific/Engineering",
        "License :: OSI Approved :: BSD License",
    ],
)
