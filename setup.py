#Sample setup.py file is found here: https://github.com/pypa/sampleproject/blob/master/setup.py
#import sys
#if sys.version_info < (3,3):
#    print("This package only works with python version 3.3 or higher.")
#    sys.exit(1)
    
# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

setup(
  name='filterAndView',
  version='0.0.0',
  description="Interactive analysis of data from the commandline",
  url="https://github.com/Bernhard10/filterAndView",
  license="AGPL-3.0",
  author="Bernhard Thiel",
  author_email="thiel@tbi.univie.ac.at",
  packages=["fav"],
  install_requires=["pandas>=0.19", "numpy>=1.11"],
  extras_require={'plotting': ['matplotlib>1.5'], "learn":["sklearn"]}
)

