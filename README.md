
CaImAn
======
<img src="https://github.com/simonsfoundation/CaImAn/blob/master/docs/LOGOS/Caiman_logo_FI.png" width="500" align="right">


[![Join the chat at https://gitter.im/agiovann/SOURCE_EXTRACTION_PYTHON](https://badges.gitter.im/agiovann/SOURCE_EXTRACTION_PYTHON.svg)](https://gitter.im/agiovann/SOURCE_EXTRACTION_PYTHON?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

<a href='https://travis-ci.org/simonsfoundation/CaImAn'><img src='https://secure.travis-ci.org/simonsfoundation/CaImAn.png?branch=master'></a>


A Computational toolbox for large scale **Ca**lcium **Im**aging data **An**alysis and behavioral analysis.

Recent advances in calcium imaging acquisition techniques are creating datasets of the order of Terabytes/week. Memory and computationally efficient algorithms are required to analyze in reasonable amount of time terabytes of data. This projects implements a set of essential methods required in the calcium imaging movies analysis pipeline. Fast and scalable algorithms are implemented for motion correction, movie manipulation, and source and spike extraction. CaImAn also contains some routine to the analyisis of behavior from video cameras. In summary, CaImAn provides a general purpose tool to handle large movies, with special emphasis tools for two-photon and one-photon calcium imaging and behavioral datasets. 


## Features

* **Handling of very large datasets**

    * Memory mapping 
    * Parallel processing in patches
    * Frame-by-frame online processing [[5]](#onacid)
    * OpenCV-based efficient movie playing and resizing

* **Motion correction** [[6]](#normcorre)

    * Fast parallelizable OpenCV and FFT-based motion correction of large movies
    * Can be run also in online mode (i.e. one frame at a time)
    * Corrects for non-rigid artifacts due to raster scanning or non-uniform brain motion

* **Source extraction** 

    * Separates different sources based on constrained nonnegative matrix Factorization (CNMF) [[1-2]](#neuron)
    * Deals with heavily overlaping and neuroopil contaminated movies     
    * Suitable for both 2-photon [[1]](#neuron) and 1-photon [[3]](#cnmfe) calcium imaging data
    * Selection of inferred sources using a pre-trained convolutional neural network classifier
    * Online processing available [[5]](#onacid)

* **Denoising, deconvolution and spike extraction**

    * Inferres neural activity from fluorescence traces [[1]](#neuron)
    * Also works in online mode (i.e. one sample at a time) [[4]](#oasis)

* **Behavioral Analysis** [[7]](#behavior)

    * Unsupervised algorithms based on optical flow and NMF to automatically extract motor kinetics 
    * Scales to large datasets by exploiting online dictionary learning
    * We also developed a tool for acquiring movies at high speed with low cost equipment [[Github repository]](https://github.com/bensondaled/eyeblink). 


## New: Online analysis

We recently incorporated a Python implementation of the OnACID [[5]](#onacid) algorithm, that enables processing data in an online mode and in real time. Check the script ```demos_detailed/demo_OnACID_mesoscope.py``` or the notebook ```demo_OnACID_mesoscope.ipynb``` for an application on two-photon mesoscope data provided by the Tolias lab (Baylor College of Medicine).

## Installation for calcium imaging data analysis


* Installation on Mac 

   * Download and install Anaconda (Python 2.7 or Python 3.5) <http://docs.continuum.io/anaconda/install>

    ```bash
   
   git clone https://github.com/simonsfoundation/CaImAn
   cd CaImAn/
   conda env create -f environment_mac.yml -n caiman
   source activate caiman
   (ONLY FOR PYTHON 2) conda install numpy==1.12  
   (ONLY FOR PYTHON 2) conda install spyder=3.1
   conda install -c conda-forge tensorflow keras
   python setup.py build_ext -i   
   ```


* Installation on Linux 

   * Download and install Anaconda (Python 2.7 or Python 3.5) <http://docs.continuum.io/anaconda/install>

   ```bash
   
   git clone https://github.com/simonsfoundation/CaImAn
   cd CaImAn/
   conda env create -f environment.yml -n caiman
   source activate caiman   
   (ONLY FOR PYTHON 2) conda install spyder=3.1
   python setup.py build_ext -i   
   ```


   * To make the package available from everywhere and have it working *efficiently* under any configuration ALWAYS run these lines before starting spyder:

   ```bash
   export PYTHONPATH="/path/to/caiman:$PYTHONPATH"
   export MKL_NUM_THREADS=1
   export OPENBLAS_NUM_THREADS=1
   ```

* Installation on  Windows (OUTDATED, NEEDS TESTING)


   * Download and install Anaconda (Python 2.7) <http://docs.continuum.io/anaconda/install>, GIT (<https://git-scm.com/>) and Microsoft Visual C++ Compiler for Python 2.7 <https://www.microsoft.com/en-us/download/details.aspx?id=44266>

    ```bash
    
    git clone  https://github.com/simonsfoundation/CaImAn
    cd CaImAn
    git pull
    conda env create -f environment_mac.yml -n caiman
    source activate caiman   
    conda install -c conda-forge tensorflow keras
    python setup.py build_ext -i       
    ```

 
### Installation for behavioral analysis
* Installation on Linux (windows and mac os are problematic with anaconda at the moment)
   * create a new environment (suggested for safety) and follow the instructions for the calcium imaging installation
   * Install spams, as explained [here](http://spams-devel.gforge.inria.fr/). Installation is not straightforward and it might take some trials to get it right


## Demos

* Notebooks : The notebooks provide a simple and friendly way to get into CaImAn and understand its main characteristics. 

   * you can find them in directly in CaImAn folder and launch them from your ipython Notebook application:
   
   * to launch jupyter notebook :
   
       ```bash
    
        source activate CaImAn
        conda launch jupyter
        (if errors on plotting use this instead) jupyter notebook --NotebookApp.iopub_data_rate_limit=1.0e10
    
       ```
* demo files are to be found also in the demos_detailed subfolder. We suggest to try demo_pipeline.py first since it contains most of the tasks required by calcium imaging. For behavior use demo_behavior.py
   
  * /!\ if you want to launch directly the python files, please be advised that your python console still needs to be in the CaImAn folder and not somewhere else. 

## Testing

* All of the commits and pull requests need to be previously tested before asking for a pull request. Call 'nosetests' program from inside of your CaImAn folder to look for errors. 
   For python3 on mac OS x nosetests does not work properly. If you need to test, then type the following from within the CaImAn folder:
```bash
cd caiman/tests
ls test_*py | while read t; do nosetests --nologcapture ${t%%.py}; done;
```

  ### general_test

   * This test will run the entire CaImAn program and look for differences against the original one. If your changes have made significant differences able to be recognise by this test.  
   
   
# Contributors:

* Andrea Giovannucci, **Flatiron Institute, Simons Foundation**
* Eftychios A. Pnevmatikakis, **Flatiron Institute, Simons Foundation** 
* Johannes Friedrich, **Flatiron Institute, Simons Foundation**
* Erick, Cobos, **Baylor College of Medicine**
* Valentina Staneva, **University of Washington**
* Ben Deverett, **Princeton University**
* Jérémie Kalfon, **University of Kent, ECE paris** 

A complete list of contributors can be found [here](https://github.com/simonsfoundation/CaImAn/graphs/contributors).

# References

The following references provide the theoretical background and original code for the included methods. 

### Deconvolution and demixing of calcium imaging data

<a name="neuron"></a>[1] Pnevmatikakis, E.A., Soudry, D., Gao, Y., Machado, T., Merel, J., ... & Paninski, L. (2016). Simultaneous denoising, deconvolution, and demixing of calcium imaging data. Neuron 89(2):285-299, [[paper]](http://dx.doi.org/10.1016/j.neuron.2015.11.037), [[Github repository]](https://github.com/epnev/ca_source_extraction). 

<a name="struct"></a>[2] Pnevmatikakis, E.A., Gao, Y., Soudry, D., Pfau, D., Lacefield, C., ... & Paninski, L. (2014). A structured matrix factorization framework for large scale calcium imaging data analysis. arXiv preprint arXiv:1409.2903. [[paper]](http://arxiv.org/abs/1409.2903). 

<a name="cnmfe"></a>[3] Zhou, P., Resendez, S. L., Stuber, G. D., Kass, R. E., & Paninski, L. (2016). Efficient and accurate extraction of in vivo calcium signals from microendoscopic video data. arXiv preprint arXiv:1605.07266. [[paper]](https://arxiv.org/abs/1605.07266), [[Github repository]](https://github.com/zhoupc/CNMF_E).

<a name="oasis"></a>[4] Friedrich J. and Paninski L. Fast active set methods for online spike inference from calcium imaging. NIPS, 29:1984-1992, 2016. [[paper]](https://papers.nips.cc/paper/6505-fast-active-set-methods-for-online-spike-inference-from-calcium-imaging), [[Github repository]](https://github.com/j-friedrich/OASIS).

### Online Analysis

<a name="onacid"></a>[5] Giovannucci, A., Friedrich J., Kaufman M., Churchland A., Chklovskii D., Paninski L., & Pnevmatikakis E.A. (2017). OnACID: Online analysis of calcium imaging data in real data. NIPS 2017, to appear. [[paper]](https://www.biorxiv.org/content/early/2017/10/02/193383)

### Motion Correction

<a name="normcorre"></a>[6] Pnevmatikakis, E.A., and Giovannucci A. (2017). NoRMCorre: An online algorithm for piecewise rigid motion correction of calcium imaging data. Journal of Neuroscience Methods, 291:83-92 [[paper]](https://doi.org/10.1016/j.jneumeth.2017.07.031), [[Github repository]](https://github.com/simonsfoundation/normcorre).

### Behavioral analysis

<a name="behavior"></a>[7] Giovannucci, A., Pnevmatikakis, E. A., Deverett, B., Pereira, T., Fondriest, J., Brady, M. J., ... & Masip, D. (2017). Automated gesture tracking in head-fixed mice. Journal of Neuroscience Methods, in press. [[paper]](https://doi.org/10.1016/j.jneumeth.2017.07.014).

## Related packages

The implementation of this package is developed in parallel with a MATLAB toobox, which can be found [here](https://github.com/epnev/ca_source_extraction). 

Some tools that are currently available in Matlab but have been ported to CaImAn are

- [MCMC spike inference](https://github.com/epnev/continuous_time_ca_sampler) 
- [Group LASSO initialization and spatial CNMF](https://github.com/danielso/ROI_detect)


## Troubleshooting

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

## Dependencies

The code uses the following libraries
- [NumPy](http://www.numpy.org/)
- [SciPy](http://www.scipy.org/)
- [Matplotlib](http://matplotlib.org/)
- [Scikit-Learn](http://scikit-learn.org/stable/)
- [ipyparallel](http://ipyparallel.readthedocs.org/en/latest/) for parallel processing
- [opencv](http://opencv.org/) for efficient image manipulation and visualization
- [Tifffile](https://pypi.python.org/pypi/tifffile) For reading tiff files. Other choices can work there too.
- [cvxpy](http://www.cvxpy.org/) for solving optimization problems (for deconvolution, optional)
- [Spams](http://spams-devel.gforge.inria.fr/) for online dictionary learning (for behavioral analysis, optional)

For the constrained deconvolution method (```deconvolution.constrained_foopsi```)  various solvers can be used, some of which requires  additional packages:

1. ```'cvxpy'```: (default) For this option, the following packages are needed:
  * [CVXOPT](http://cvxopt.org/) optional.
  * [CVXPY](http://www.cvxpy.org/) optional.
2. ```'cvx'```: For this option, the following packages are needed:
  * [CVXOPT](http://cvxopt.org/) optional.
  * [PICOS](http://picos.zib.de/) optional.

In general ```'cvxpy'``` can be faster, when using the 'ECOS' or 'SCS' sovlers, which are included with the CVXPY installation. Note that these dependencies are circumvented by using the OASIS algoritm for deconvolution.


# Documentation & Wiki

Documentation of the code can be found [here](http://simonsfoundation.github.io/CaImAn/). 
Moreover, our [wiki page](https://github.com/simonsfoundation/CaImAn/wiki) covers some aspects of the code.

# Acknowledgements

Special thanks to the following people for letting us use their datasets for our various demo files:

* Weijian Yang, Darcy Peterka, Rafael Yuste, Columbia University
* Sue Ann Koay, David Tank, Princeton University
* Manolis Froudarakis, Jake Reimers, Andreas Tolias, Baylor College of Medicine

# Citation

If you use this code please cite the corresponding papers where original methods appeared (see References above), as well as the following abstract:

Giovannucci, A., Friedrich, J., Deverett, B., Staneva, V., Chklovskii, D., & Pnevmatikakis, E. (2017). CaImAn: An open source toolbox for large scale calcium imaging data analysis on standalone machines. Cosyne Abstracts.

# Questions, comments, issues

Please use the [gitter chat room](https://gitter.im/agiovann/Constrained_NMF) for questions and comments and create an issue for any bugs you might encounter.

# License

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
