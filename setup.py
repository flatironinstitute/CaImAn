#!/usr/bin/env python

from setuptools import setup, find_packages
from os import path
import numpy as np
from Cython.Build import cythonize
from setuptools.extension import Extension
from distutils.command.build_ext import build_ext

"""
    Installation script for anaconda installers
"""

here = path.abspath(path.dirname(__file__))

with open('README.md', 'r') as rmf:
    readme = rmf.read()

# compile with:     python setup.py build_ext -i
# clean up with:    python setup.py clean --all
ext_modules = [Extension("caiman.source_extraction.cnmf.oasis",
                         sources=["caiman/source_extraction/cnmf/oasis.pyx"],
                         include_dirs=[np.get_include()],
                         language="c++")]

setup(
    name='caiman',
    version='1.0',
    author='Andrea Giovannucci, Eftychios Pnevmatikakis, Johannes Friedrich, Valentina Staneva, Ben Deverett, Erick Cobos, Jeremie Kalfon',
    author_email='agiovannucci@flatironinstitute.org',
    url='https://github.com/simonsfoundation/CaImAn',
    license='GPL-2',
    description='Advanced algorithms for ROI detection and deconvolution of Calcium Imaging datasets.',
    long_description=readme,
    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 4 - Beta',

        # Indicate who your project is intended for
        'Intended Audience :: Researchers',
        'Topic :: Calcium Imaging :: Analysis Tools',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: GPL-2 License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 2,3',
    ],
    keywords='fluorescence calcium ca imaging deconvolution ROI identification',
    packages=find_packages(exclude=['use_cases', 'use_cases.*']),
    data_files=[('', ['LICENSE.txt']),
                ('', ['README.md'])],
    install_requires=[''],
    ext_modules=cythonize(ext_modules),
    cmdclass={'build_ext': build_ext}
)
