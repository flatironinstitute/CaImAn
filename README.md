Python translation of Constrained Non-negative Matrix Factorzation algorithm for source extraction from calcium imaging data. 

[![Join the chat at https://gitter.im/agiovann/SOURCE_EXTRACTION_PYTHON](https://badges.gitter.im/agiovann/SOURCE_EXTRACTION_PYTHON.svg)](https://gitter.im/agiovann/SOURCE_EXTRACTION_PYTHON?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

# Deconvolution and demixing of calcium imaging data

The code implements a method for simultaneous source extraction and spike inference from large scale calcium imaging movies. The code is suitable for the analysis of somatic imaging data. Implementation for the analysis of dendritic/axonal imaging data will be added in the future. 

The algorithm is presented in more detail in

Pnevmatikakis, E.A., Soudry, D., Gao, Y., Machado, T., Merel, J., ... & Paninski, L. (2016). Simultaneous denoising, deconvolution, and demixing of calcium imaging data. Neuron, in press, http://dx.doi.org/10.1016/j.neuron.2015.11.037

Pnevmatikakis, E.A., Gao, Y., Soudry, D., Pfau, D., Lacefield, C., ... & Paninski, L. (2014). A structured matrix factorization framework for large scale calcium imaging data analysis. arXiv preprint arXiv:1409.2903. http://arxiv.org/abs/1409.2903

# Contributors

Andrea Giovannucci @agiovann
Eftychios Pnevmatikakis @epnev

Center for Computational Biology, Simons Foundation, New York, NY


Code description and related packages
=======

This repository contains a MATLAB implementation of the spatio-temporal demixing, i.e., (source extraction) code for large scale calcium imaging data. Related code can be found in the following links:

## Python
- [Source extraction with CNMF (this package)](https://github.com/agiovann/SOURCE_EXTRACTION_PYTHON)
- [Group LASSO initialization and spatial CNMF](https://github.com/danielso/ROI_detect)

## Matlab 
- [Constrained deconvolution and source extraction with CNMF](https://github.com/epnev/ca_source_extraction)
- [MCMC spike inference](https://github.com/epnev/continuous_time_ca_sampler)
- [Group LASSO initialization and spatial CNMF](https://github.com/danielso/ROI_detect)

## Integration with other libraries
- [SIMA](http://www.losonczylab.org/sima/1.3/): The [constrained deconvolution](https://github.com/losonczylab/sima/blob/master/sima/spikes.py) method has been integrated with SIMA, a Python based library for calcium imaging data analysis.
- [Thunder](http://thunder-project.org/): The [group LASSO initialization and spatial CNMF](https://github.com/j-friedrich/thunder/tree/LocalNMF) method has been integrated with Thunder, a library for large scale neural data analysis with Spark.

Dependencies
========
The code uses the following libraries
- [NumPy](http://www.numpy.org/)
- [SciPy](http://www.scipy.org/)
- [Matplotlib](http://matplotlib.org/)
- [Scikit-Learn](http://scikit-learn.org/stable/)
- [Tifffile](https://pypi.python.org/pypi/tifffile) For reading tiff files. Other choices can work there too.

External Dependencies
============

The constrained deconvolution method (constrained_foopsi_python.py) can estimate with two different methods, each of which requires some additional packages:
1. 'spgl1': For this option, the [SPGL1](https://github.com/epnev/SPGL1_python_port) python implementation is required. Please use the "forked" repository linked here.
2. 'cvx': For this option, the following packages are needed:
 * [CVXOPT](http://cvxopt.org/) Required.
 * [PICOS](http://picos.zib.de/) Required.
 * [MOSEK](https://www.mosek.com/) Optional but strongly recommended for speed improvement, free for academic use.

In general 'spgl1' can be faster, but the python implementation is not as fast as in Matlab and not thoroughly tested.

Questions, comments, issues
=======
Please use the gitter chat room (use the button above) for questions and comments and create an issue for any bugs you might encounter.

Important note
======
The implementation of this package is based on the matlab implementation which can be found [here](https://github.com/epnev/ca_source_extraction). Some of the Matlab features are currently lacking, but will be included in future releases. 

License
=======

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
