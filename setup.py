#!/usr/bin/env python

from setuptools import setup, find_packages
import numpy as np
import os
import sys
from Cython.Build import cythonize
from setuptools.extension import Extension
from distutils.command.build_ext import build_ext

"""
    Installation script for anaconda installers
"""

here = os.path.abspath(os.path.dirname(__file__))

with open('README.md', 'r') as rmf:
    readme = rmf.read()

with open('VERSION', 'r') as verfile:
    version = verfile.read().strip()

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

extra_dirs = ['bin', 'demos', 'docs', 'model']
data_files = [('share/caiman', ['LICENSE.txt', 'README.md', 'test_demos.sh', 'VERSION']),
              ('share/caiman/example_movies', ['example_movies/data_endoscope.tif', 'example_movies/demoMovie.tif', 'example_movies/avg_mask_fixed.png']),
              ('share/caiman/testdata', ['testdata/groundtruth.npz', 'testdata/example.npz', 'testdata/2d_sbx.mat', 'testdata/2d_sbx.sbx', 'testdata/3d_sbx_1.mat', 'testdata/3d_sbx_1.sbx', 'testdata/3d_sbx_2.mat', 'testdata/3d_sbx_2.sbx']),
             ]
for part in extra_dirs:
	newpart = [("share/caiman/" + d, [os.path.join(d,f) for f in files]) for d, folders, files in os.walk(part)]
	for newcomponent in newpart:
		data_files.append(newcomponent)

############

# compile with:     python setup.py build_ext -i
# clean up with:    python setup.py clean --all
if sys.platform == 'darwin':
        # see https://github.com/pandas-dev/pandas/issues/23424
	extra_compiler_args = ['-stdlib=libc++']  # not needed #, '-mmacosx-version-min=10.9']
else:
	extra_compiler_args = []

ext_modules = [Extension("caiman.source_extraction.cnmf.oasis",
                         sources=["caiman/source_extraction/cnmf/oasis.pyx"],
                         include_dirs=[np.get_include()],
                         language="c++",
                         extra_compile_args = extra_compiler_args,
                         extra_link_args = extra_compiler_args,
                         )]

setup(
    name='caiman',
    version=version,
    author='Andrea Giovannucci, Eftychios Pnevmatikakis, Johannes Friedrich, Valentina Staneva, Ben Deverett, Erick Cobos, Jeremie Kalfon',
    author_email='pgunn@flatironinstitute.org',
    url='https://github.com/flatironinstitute/CaImAn',
    license='GPL-2',
    description='Advanced algorithms for ROI detection and deconvolution of Calcium Imaging datasets.',
    long_description=readme,
    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 5 - Production/Stable',

        # Indicate who your project is intended for
        'Intended Audience :: Researchers',
        'Topic :: Calcium Imaging :: Analysis Tools',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: GPL-2 License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3',
    ],
    keywords='fluorescence calcium ca imaging deconvolution ROI identification',
    packages=find_packages(),
    entry_points = { 'console_scripts': ['caimanmanager = caiman.caimanmanager:main' ] },
    data_files=data_files,
    install_requires=[''],
    ext_modules=cythonize(ext_modules, language_level="3"),
    cmdclass={'build_ext': build_ext}
)
