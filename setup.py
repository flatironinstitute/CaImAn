from setuptools import setup
from os import path
#import os
import numpy as np
from Cython.Build import cythonize

"""
    Installation script for anaconda installers
"""

here = path.abspath(path.dirname(__file__))

with open('README.md', 'r') as rmf:
    readme = rmf.read()

#incdir = os.path.join(get_python_inc(plat_specific=1), 'Numerical')

setup(
    name='CaImAn',
    version='0.1',
    author='Andrea Giovannucci, Eftychios Pnevmatikakis, Johannes Friedrich, Valentina Staneva, Ben Deverett',
    author_email='agiovannucci@simonsfoundation.org',
    url='https://github.com/agiovann/Constrained_NMF',
    license='GPL-2',
    description='Advanced algorithms for ROI detection and deconvolution of Calcium Imaging datasets.',
    long_description=readme,
    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Testers',
        'Topic :: Calcium Imaging :: Analysis Tools',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: GPL-2 License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 2.7',
    ],
    keywords='fluorescence calcium ca imaging deconvolution ROI identification',
    packages=['caiman'],
    data_files=[	('', ['LICENSE.txt']),
                 ('', ['README.md'])],
    # 'matplotlib', 'scikit-learn', 'scikit-image', 'ipyparallel','scikit-learn','ipython','scipy','numpy'],#,'bokeh','jupyter','tifffile','cvxopt','picos', 'joblib>=0.8.4'],
    install_requires=['python==2.7.*'],
    include_dirs=[np.get_include()],
    # compile with:     python setup.py build_ext -i
    # clean up with:    python setup.py clean --all
    ext_modules=cythonize("caiman/source_extraction/cnmf/oasis.pyx")

)
