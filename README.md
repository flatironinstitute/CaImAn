
CaImAn
======
<img src="https://github.com/simonsfoundation/CaImAn/blob/master/docs/LOGOS/Caiman_logo_FI.png" width="500" align="right">


[![Join the chat at https://gitter.im/agiovann/SOURCE_EXTRACTION_PYTHON](https://badges.gitter.im/agiovann/SOURCE_EXTRACTION_PYTHON.svg)](https://gitter.im/agiovann/SOURCE_EXTRACTION_PYTHON?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

<a href='https://travis-ci.org/simonsfoundation/CaImAn'><img src='https://secure.travis-ci.org/simonsfoundation/CaImAn.png?branch=master'></a>



A Computational toolbox for large scale **Ca**lcium **Im**aging **An**alysis* and behavioral analysis




Recent advances in calcium imaging acquisition techniques are creating datasets of the order of Terabytes/week. Memory and computationally efficient algorithms are required to analyze in reasonable amount of time terabytes of data. This projects implements a set of essential methods required in the calcium imaging movies analysis pipeline. Fast and scalable algorithms are implemented for motion correction, movie manipulation and source and spike extraction. CaImAn also contains some routine to the analyisi of behavior from video cameras. In summary, CaImAn provides a general purpose tool to handle large movies, with special emphasis tools for calcium imaging and behavioral datasets. 


### Features

* Handling of very large datasets

    * Memory mapping
    * Frame-by-frame online processing  (some functions)
    * opencv-based efficient movie playing and resizing

* Motion correction

    * Fast parallelizable open-cv and fft-based motion correction of large movies
    * Run also in online mode (i.e. one frame at a time)
    * non rigid motion correction

* Source extraction 

    * identification of pixles associated to each neuron/neuronal structure
    * deals with heavily overlaping and neuroopil contaminated movies 
    * separates different sources based on Nonnegative Matrix Factorization algorithms

* Denoising, deconvolution and spike extraction

    * spikes can be inferred from fluorescence traces
    * also works in online mode (i.e. one sample at a time)

* Behavioral Analysis ([paper](http://www.sciencedirect.com/science/article/pii/S0165027017302509))

    * unsupervised algorithms based on optical flow and NMF to automatically extract motor kinetics 
    * scales to large datasets since it exploits online dictionary learning  (Mairal et. Al, [link](http://spams-devel.gforge.inria.fr/))
    * we also developed a tool for acquiring movies at high speed with low cost equipment ([package](https://github.com/bensondaled/eyeblink)) 



### Installation for calcium imaging


* Installation on Mac 

   * Download and install Anaconda (Python 2.7 or Python 3.5) <http://docs.continuum.io/anaconda/install>

   ```bash
   
   git clone https://github.com/simonsfoundation/CaImAn
   cd CaImAn/
   git pull
   
   EITHER
   conda create -n CaImAn python=3.5 --file requirements_conda.txt   (For Python 3)
   OR
   conda create -n CaImAn ipython --file requirements_conda.txt   (For Python 2) 
   
   source activate CaImAn
   pip install -r requirements_pip.txt
   conda install -c menpo opencv3=3.1.0
   python setup.py build_ext -i
   conda install bokeh
   ```


* Installation on Linux 

   * Download and install Anaconda (Python 2.7 or Python 3.5) <http://docs.continuum.io/anaconda/install>

   ```bash
   
   git clone https://github.com/simonsfoundation/CaImAn
   cd CaImAn/
   git pull
   conda create -n CaImAn ipython --file requirements_conda.txt    
   source activate CaImAn
   pip install -r requirements_pip.txt
   conda install -c menpo opencv3=3.2.0
   python setup.py build_ext -i
   conda install bokeh
   ```

   * To make the package available from everywhere and have it working *efficiently* under any configuration ALWAYS run these lines before starting spyder:

   ```bash
   export PYTHONPATH="/path/to/caiman:$PYTHONPATH"
   export MKL_NUM_THREADS=1
   export OPENBLAS_NUM_THREADS=1
   ```

* Installation on posix (Windows)


   * Download and install Anaconda (Python 2.7) <http://docs.continuum.io/anaconda/install>, GIT (<https://git-scm.com/>) and Microsoft Visual C++ Compiler for Python 2.7 <https://www.microsoft.com/en-us/download/details.aspx?id=44266>

    ```bash
    
    git clone  https://github.com/simonsfoundation/CaImAn
    cd CaImAn
    git pull
    conda update conda
    conda create -n CaImAn ipython --file requirements_conda.txt    
    activate CaImAn
    pip install -r requirements_pip.txt
    conda install -c menpo opencv3=3.1.0
    python setup.py build_ext -i
    conda update --all
    
    ```
 
### Installation for behavioral analysis
* Installation on Linux (windows and mac os are problematic with anaconda at the moment)
   * create a new environment (suggested for safety) and follow the instructions for the calcium imaging installation
   * Install spams, as explained [here](http://spams-devel.gforge.inria.fr/). Installation is not straightforward and it might take some trials to get it right

 
# Example

  ### Demos

* Notebooks : The notebooks provide a simple and friendly way to get into CaImAn and understand its main characteristics. 

   * you can find them in directly in CaImAn folder and launch them from your ipython Notebook application:
   
   * to launch jupyter notebook :
   
       ```bash
    
        source activate CaImAn
        conda launch jupyter
    
       ```
* demo files are to be found also in the demos_detailed subfolder. We suggest to try demo_pipeline.py first since it contains most of the tasks required by calcium imaging. For behavior use demo_behavior.py
   
  * /!\ if you want to launch directly the python files, please be advised that your python console still needs to be in the CaImAn folder and not somewhere else. 


# Testing

* As of today, all of the commits needs to be previously tested before asking for a pull request. Call 'nosetests' program from inside of your CaImAn folder to look for errors. 

  ### general_test

   * This test will run the entire CaImAn program and look for differences against the original one. If your changes have made significant differences able to be recognise by this test.  
   
   
# Contributors:

* Giovannucci, Andrea. **Simons Foundation** 
* Pnevmatikakis, Eftychios. **Simons Foundation** 
* Friedrich, Johannes. **Columbia University and Janelia Farm**
* Cobos, Erick. **Baylor College of Medicine**
* Staneva, Valentina. **eScience Institute**
* Deverett, Ben. **Princeton University**
* Kalfon, Jérémie. **University of Kent** , **ECE paris** 


Please refer to the following wiki [page](https://github.com/simonsfoundation/CaImAn/wiki/Processing-large-datasets) or read in the testing section below.

# Deconvolution and demixing of calcium imaging data

The code implements simultaneous source extraction and spike inference from large scale calcium imaging movies. The code is suitable for the analysis of somatic imaging data. Implementation for the analysis of dendritic/axonal imaging data will be added in the near future. The following references provide the theoretical background and original code for the included methods. 

Pnevmatikakis, E.A., Soudry, D., Gao, Y., Machado, T., Merel, J., ... & Paninski, L. (2016). Simultaneous denoising, deconvolution, and demixing of calcium imaging data. Neuron 89(2):285-299, http://dx.doi.org/10.1016/j.neuron.2015.11.037. [Github repo](https://github.com/epnev/ca_source_extraction). 

Pnevmatikakis, E.A., Gao, Y., Soudry, D., Pfau, D., Lacefield, C., ... & Paninski, L. (2014). A structured matrix factorization framework for large scale calcium imaging data analysis. arXiv preprint arXiv:1409.2903. http://arxiv.org/abs/1409.2903. 

Friedrich J. and Paninski L. Fast active set methods for online spike inference from calcium imaging. NIPS, 29:1984-1992, 2016. [PDF](https://papers.nips.cc/paper/6505-fast-active-set-methods-for-online-spike-inference-from-calcium-imaging). [Github repository](https://github.com/j-friedrich/OASIS).

# Behavioral analysis

Giovannucci, A., Pnevmatikakis, E. A., Deverett, B., Pereira, T., Fondriest, J., Brady, M. J., ... & Masip, D. (2017). Automated gesture tracking in head-fixed mice. Journal of Neuroscience Methods.

Code description and related packages
=======

The implementation of this package is developed in parallel with a MATLAB toobox, which can be found [here](https://github.com/epnev/ca_source_extraction). 

Some tools that are currently available in Matlab and not in Python are at the following links

- [MCMC spike inference](https://github.com/epnev/continuous_time_ca_sampler) 
- [Group LASSO initialization and spatial CNMF](https://github.com/danielso/ROI_detect)




Troubleshooting
----------------

**Python 3 and spyder**
if spyder crashes on mac os run 
```
brew install --upgrade openssl
brew unlink openssl && brew link openssl --force
```

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
 
Test the system
----------------




Documentation
========

Documentation of the code can be found [here](http://simonsfoundation.github.io/CaImAn/)

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
- [opencv](http://opencv.org/) for efficient image manipulation and visualization
- [Spams](http://spams-devel.gforge.inria.fr/) for online dictionary learning

External Dependencies
============

For the constrained deconvolution method (```deconvolution.constrained_foopsi```)  various solvers can be used, some of which requires  additional packages:

1. ```'cvxpy'```: (default) For this option, the following packages are needed:
  * [CVXOPT](http://cvxopt.org/) optional.
  * [CVXPY](http://www.cvxpy.org/) optional.
2. ```'cvx'```: For this option, the following packages are needed:
  * [CVXOPT](http://cvxopt.org/) optional.
  * [PICOS](http://picos.zib.de/) optional.

In general ```'cvxpy'``` can be faster, when using the 'ECOS' or 'SCS' sovlers, which are included with the CVXPY installation.

Questions, comments, issues
=======
Please use the gitter chat room (use the button above) for questions and comments and create an issue for any bugs you might encounter.


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
```
