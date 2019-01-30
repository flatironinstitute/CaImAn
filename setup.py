#!/usr/bin/env python

from setuptools import setup, find_packages
import os
from os import path
import sys
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

############
# This stanza asks for caiman datafiles (demos, movies, ...) to be stashed in "share/caiman", either
# in the system directory if this was installed with a system python, or inside the virtualenv/conda
# environment dir if this was installed with a venv/conda python. This ensures:
# 1) That they're present somewhere on the system if Caiman is installed this way, and
# 2) We can programmatically get at them to manage the user's conda data directory.
#
# We can access these by using sys.prefix as the base of the directory and constructing from there.
# Note that if python's packaging standards ever change the install base of data_files to be under the
# package that made them, we can switch to using the pkg_resources API.

binaries = ['caimanmanager.py']
extra_dirs = ['demos', 'docs', 'model']
data_files = [('share/caiman', ['LICENSE.txt', 'README.md', 'test_demos.sh']),
              ('share/caiman/example_movies', ['example_movies/data_endoscope.tif', 'example_movies/demoMovie.tif']),
              ('share/caiman/testdata', ['testdata/groundtruth.npz', 'testdata/example.npz'])
             ]
for part in extra_dirs:
	newpart = [("share/caiman/" + d, [os.path.join(d,f) for f in files]) for d, folders, files in os.walk(part)]
	for newcomponent in newpart:
		data_files.append(newcomponent)

data_files.append(['bin', binaries])
############

# compile with:     python setup.py build_ext -i
# clean up with:    python setup.py clean --all
if sys.platform == 'darwin':
	extra_compiler_args = ['-stdlib=libc++']
else:
	extra_compiler_args = []

ext_modules = [Extension("caiman.source_extraction.cnmf.oasis",
                         sources=["caiman/source_extraction/cnmf/oasis.pyx"],
                         include_dirs=[np.get_include()],
                         language="c++",
                         extra_compile_args = extra_compiler_args)]

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
    data_files=data_files,
    install_requires=[''],
    ext_modules=cythonize(ext_modules),
    cmdclass={'build_ext': build_ext}
)
