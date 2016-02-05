

Python translation of Constrained Non-negative Matrix Factorization algorithm for source extraction from calcium imaging data. 

[![Join the chat at https://gitter.im/agiovann/SOURCE_EXTRACTION_PYTHON](https://badges.gitter.im/agiovann/SOURCE_EXTRACTION_PYTHON.svg)](https://gitter.im/agiovann/SOURCE_EXTRACTION_PYTHON?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

# Deconvolution and demixing of calcium imaging data

The code implements a method for simultaneous source extraction and spike inference from large scale calcium imaging movies. The code is suitable for the analysis of somatic imaging data. Implementation for the analysis of dendritic/axonal imaging data will be added in the future. 

The algorithm is presented in more detail in

Pnevmatikakis, E.A., Soudry, D., Gao, Y., Machado, T., Merel, J., ... & Paninski, L. (2016). Simultaneous denoising, deconvolution, and demixing of calcium imaging data. Neuron, in press, http://dx.doi.org/10.1016/j.neuron.2015.11.037

Pnevmatikakis, E.A., Gao, Y., Soudry, D., Pfau, D., Lacefield, C., ... & Paninski, L. (2014). A structured matrix factorization framework for large scale calcium imaging data analysis. arXiv preprint arXiv:1409.2903. http://arxiv.org/abs/1409.2903

# Contributors

Andrea Giovannucci and 
Eftychios Pnevmatikakis 

Center for Computational Biology, Simons Foundation, New York, NY


Code description and related packages
=======

This repository contains a Python implementation of the spatio-temporal demixing, i.e., (source extraction) code for large scale calcium imaging data. Related code can be found in the following links:

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

Installation
===================================================

Installation on MAC OS
----------------------

1. Download and install Anaconda (Python 2.7) <http://docs.continuum.io/anaconda/install> 

2. EASY WAY. type:
    ```
    conda create -n CNMF  ipython
    source activate CNMF
    conda install -c https://conda.anaconda.org/agiovann constrained_nmf
    pip install 'tifffile>=0.7'
    pip install picos
    pip install cvxpy
    ```

3. ADVANCED WAY (with access to source code).
    ```
    git clone --recursive https://github.com/agiovann/Constrained_NMF.git
    conda create -n CNMF ipython
    source activate CNMF
    conda install spyder numpy scipy ipyparallel matplotlib bokeh jupyter scikit-image scikit-learn joblib cvxopt      
    pip install 'tifffile>=0.7'
    pip install picos
    
    (
        if you get errors compiling scs when installing cvxpy you probably need to create a link to openblas or libgfortran         in  /usr/local/lib/, for instance:
        sudo ln -s  /Library/Frameworks/R.framework/Libraries/libgfortran.3.dylib  /usr/local/lib/libgfortran.2.dylib
    )
    
    pip install cvxpy
    
    ```
    This second option will not allow to import the package from every folder but only from within the Constrained_NMF folder. You can access it globally by setting the environment variable PYTHONPATH
    ```
    export PYTHONPATH="/path/to/Constrained_NMF:$PYTHONPATH"
    ```

Test the system
----------------------
In case you used installation af point 2 above you will need to download the test files from
<https://github.com/agiovann/Constrained_NMF/releases/download/0.1-beta/Demo.zip>


A. Using the Spyder (type `conda install spyder`) IDE. 
    
    1. Unzip the file Demo.zip (you do not need this step if you installed dusing method 3 above, just enter the Constrained_NMF folder and you will find all the required files there).
    2. Open the file demo.py with spyder
    3. Change the current folder of the console to the 'Demo' folder
    3. Run the cells one by one inspecting the output
    4. Remember to stop the cluster (last three lines of file). You can also stop it manually by typing in a terminal 
    'ipcluster stop'

B. Using notebook. 
    
    1. Unzip the file Demo.zip (you do not need this step if you installed dusing method 3 above, just enter the Constrained_NMF folder and you will find all the required files there).
    2. type `ipython notebook`
    3. open the notebook called demoCNMF.ipynb and run cell by cell inspecting the result
    4. Remember to stop the cluster (last three lines of file). You can also stop it manually by typing in a terminal 
    'ipcluster stop'
    
Documentation
========

Documentation of the code can be found [here](http://agiovann.github.io/Constrained_NMF)

Dependencies
========
The code uses the following libraries
- [NumPy](http://www.numpy.org/)
- [SciPy](http://www.scipy.org/)
- [Matplotlib](http://matplotlib.org/)
- [Scikit-Learn](http://scikit-learn.org/stable/)
- [Tifffile](https://pypi.python.org/pypi/tifffile) For reading tiff files. Other choices can work there too.
- [Joblib](https://pypi.python.org/pypi/joblib) for parallel processing
- [ipyparallel](http://ipyparallel.readthedocs.org/en/latest/) for parallel processing

External Dependencies
============

The constrained deconvolution method (constrained_foopsi_python.py) can estimate with two different methods, each of which requires some additional packages:

1. 'spgl1': For this option, the SPGL1 python implementation is required. It is by default imported as a submodule. The original implementation can be found at (https://github.com/mpf/spgl1).  
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
