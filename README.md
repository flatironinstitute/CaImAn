
Please refer to the following wiki [page](https://github.com/agiovann/Constrained_NMF/wiki/Processing-large-datasets) or read in the testing section below

ConstrainedNMF
==============

Python translation of Constrained Non-negative Matrix Factorization algorithm for source extraction from calcium imaging data.

[![Join the chat at https://gitter.im/agiovann/SOURCE_EXTRACTION_PYTHON](https://badges.gitter.im/agiovann/SOURCE_EXTRACTION_PYTHON.svg)](https://gitter.im/agiovann/SOURCE_EXTRACTION_PYTHON?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

<a href='https://travis-ci.org/agiovann/Constrained_NMF'><img src='https://secure.travis-ci.org/agiovann/Constrained_NMF.png?branch=testing'></a>

# Deconvolution and demixing of calcium imaging data

The code implements a method for simultaneous source extraction and spike inference from large scale calcium imaging movies. The code is suitable for the analysis of somatic imaging data. Implementation for the analysis of dendritic/axonal imaging data will be added in the future.

The algorithm is presented in more detail in

Pnevmatikakis, E.A., Soudry, D., Gao, Y., Machado, T., Merel, J., ... & Paninski, L. (2016). Simultaneous denoising, deconvolution, and demixing of calcium imaging data. Neuron 89(2):285-299, http://dx.doi.org/10.1016/j.neuron.2015.11.037

Pnevmatikakis, E.A., Gao, Y., Soudry, D., Pfau, D., Lacefield, C., ... & Paninski, L. (2014). A structured matrix factorization framework for large scale calcium imaging data analysis. arXiv preprint arXiv:1409.2903. http://arxiv.org/abs/1409.2903

# Contributors

Andrea Giovannucci, Eftychios Pnevmatikakis, 
Center for Computational Biology, Simons Foundation, New York, NY

Valentina Staneva
eScience Institute. University of Washinghton. Seattle, WA.

Johannes Friedrich
Columbia University, New York, NY. 

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

Download and install Anaconda (Python 2.7) <http://docs.continuum.io/anaconda/install>

    ```
    git clone --recursive https://github.com/agiovann/Constrained_NMF.git
    cd Constrained_NMF/CalBlitz/
    git checkout master
    git pull
    conda create -n CNMF ipython
    source activate CNMF
    conda install numpy scipy 
    conda install spyder 
    conda install ipyparallel
    conda install matplotlib bokeh jupyter
    conda install scikit-image
    conda install scikit-learn
    conda install cvxopt     
    conda install h5py
    pip install picos
    pip install cvxpy
    pip install pims
    conda install -c menpo opencv3=3.1.0
    pip install 'tifffile>=0.7'
    ```
    This second option will not allow to import the package from every folder but only from within the Constrained_NMF folder. You can access it globally by setting the environment variable PYTHONPATH
    ```
    export PYTHONPATH="/path/to/Constrained_NMF:$PYTHONPATH"
    ```



Troubleshooting
----------------
**SCS**:

if you get errors compiling scs when installing cvxpy you probably need to create a link to openblas or libgfortran in
/usr/local/lib/, for instance:

`sudo ln -s  /Library/Frameworks/R.framework/Libraries/libgfortran.3.dylib  /usr/local/lib/libgfortran.2.dylib`



**debian fortran compiler problems:**
if you get the error  gcc: error trying to exec 'cc1plus': execvp: No such file or directory in ubuntu run
or issues related to SCS type

 ```
 sudo apt-get install g++ libatlas-base-dev gfortran  libopenblas-dev
 conda install openblas atlas
 ```

 if still there are issues try

  `export LD_LIBRARY_PATH=/path_to_your_home/anaconda2/lib/`

 if more problems try 

 ```
 conda install  atlas (only Ubuntu)
 pip install 'tifffile>=0.7'
 conda install accelerate
 conda install openblas 
 ```
 
**CVXOPT**:

If you are on Windows and don't manage to install or compile cvxopt, a simple solution is to download the right binary [there](http://www.lfd.uci.edu/~gohlke/pythonlibs/#cvxopt) and install the library by typing:

```
pip install cvxopt-1.1.7-XXXX.whl
```

Test the system
----------------------

**SINGLE PATCH**

In case you used installation af point 1 above you will need to download the test files from
<https://github.com/agiovann/Constrained_NMF/releases/download/v0.3/Demo.zip>

A. Go into the cloned folder, type `python demo.py`

B. Using the Spyder (type `conda install spyder`) IDE.

    1. Unzip the file Demo.zip (you do not need this step if you installed dusing method 2 above, just enter the Constrained_NMF folder and you will find all the required files there).
    2. Open the file demo.py with spyder
    3. change the base_folder variable to point to the folder you just unzipped
    3. Run the cells one by one inspecting the output
    4. Remember to stop the cluster (last three lines of file). You can also stop it manually by typing in a terminal
    'ipcluster stop'

C. Using notebook.

    1. Unzip the file Demo.zip (you do not need this step if you installed dusing method 3 above, just enter the Constrained_NMF folder and you will find all the required files there).
    2. type `ipython notebook`
    3. open the notebook called demoCNMF.ipynb 
    4. change the base_folder variable to point to the folder you just unzipped
    5. and run cell by cell inspecting the result
    6. Remember to stop the cluster (last three lines of file). You can also stop it manually by typing in a terminal
    'ipcluster stop'


**MULTI PATCH**
+ Download the two demo movies [here](https://github.com/agiovann/Constrained_NMF/releases/download/v0.4-alpha/Patch_demo.zip) (courtesy of Dr. Sue Ann Koay from the Tank Lab, Princeton Neuroscience Institute, Princeton. NJ). Unzip the folder. Then in Spyder open the file demo_patches.py, and change the base_folder variable to point to the folder you just unzipped. 
+ Run one by one the cells (delimited by '#%%') 
+ Inspect the results. The demo will start a cluster and process pathes of the movie (more details [here](https://github.com/agiovann/Constrained_NMF/wiki/Processing-large-datasets)) in parallel (cse.map_reduce.run_CNMF_patches). Afterwards, it will merge the results back together and proceed to firstly merge potentially overlaping components (cse.merge_components) from different patches, secondly to update the spatial extent of the joined spatial components (cse.spatial.update_spatial_components), and finally denoising the traces (cse.temporal.update_temporal_components). THe final bit is used for visualization. 

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
- [cvxpy](http://www.cvxpy.org/) for solving optimization problems
- [ipyparallel](http://ipyparallel.readthedocs.org/en/latest/) for parallel processing

External Dependencies
============

For the constrained deconvolution method (```deconvolution.constrained_foopsi```)  various solvers can be used, each of which requires some additional packages:

1. ```'cvxpy'```: (default) For this option, the following packages are needed:
  * [CVXOPT](http://cvxopt.org/) Required.
  * [CVXPY](http://www.cvxpy.org/) Required.
2. ```'cvx'```: For this option, the following packages are needed:
  * [CVXOPT](http://cvxopt.org/) Required.
  * [PICOS](http://picos.zib.de/) Required.
  * [MOSEK](https://www.mosek.com/) Optional but strongly recommended for speed improvement, free for academic use.
3. ```'spgl1'```: For this option, the SPGL1 python implementation is required. It is by default imported as a submodule. The original implementation can be found at (https://github.com/mpf/spgl1).  

In general ```'cvxpy'``` can be faster, when using the 'ECOS' or 'SCS' sovlers, which are included with the CVXPY installation. ```'spgl1'``` can also be very fast but the python implementation is not as fast as in Matlab and not thoroughly tested.

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


Experimental
=============

The package comes with a toolbox to manipulate movies written in Python, Calblitz. If you want to give it a try use the demo_pipeline.py file. Before that you need to install some packages:

```
pip install pims
conda install opencv
conda install h5py
```
