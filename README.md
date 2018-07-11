
CaImAn
======
<img src="https://github.com/flatironinstitute/CaImAn/blob/master/docs/LOGOS/Caiman_logo_FI.png" width="500" align="right">


[![Join the chat at https://gitter.im/agiovann/SOURCE_EXTRACTION_PYTHON](https://badges.gitter.im/agiovann/SOURCE_EXTRACTION_PYTHON.svg)](https://gitter.im/agiovann/SOURCE_EXTRACTION_PYTHON?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)


A Computational toolbox for large scale **Ca**lcium **Im**aging data **An**alysis and behavioral analysis.

Recent advances in calcium imaging acquisition techniques are creating datasets of the order of Terabytes/week. Memory and computationally efficient algorithms are required to analyze in reasonable amount of time terabytes of data. This project implements a set of essential methods required in the calcium imaging movies analysis pipeline. Fast and scalable algorithms are implemented for motion correction, movie manipulation, and source and spike extraction. CaImAn also contains some routines for the analyisis of behavior from video cameras. In summary, CaImAn provides a general purpose tool to handle large movies, with special emphasis on tools for two-photon and one-photon calcium imaging and behavioral datasets. 

## Companion paper
A paper explaining most of the implementation details and benchmarking can be found at this [link](https://www.biorxiv.org/content/early/2018/06/05/339564)

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
    * FFTs can be computed on GPUs (experimental). Requires pycuda and skcuda to be installed.

* **Source extraction** 

    * Separates different sources based on constrained nonnegative matrix Factorization (CNMF) [[1-2]](#neuron)
    * Deals with heavily overlapping and neuropil contaminated movies     
    * Suitable for both 2-photon [[1]](#neuron) and 1-photon [[3]](#cnmfe) calcium imaging data
    * Selection of inferred sources using a pre-trained convolutional neural network classifier
    * Online processing available [[5]](#onacid)

* **Denoising, deconvolution and spike extraction**

    * Inferes neural activity from fluorescence traces [[1]](#neuron)
    * Also works in online mode (i.e. one sample at a time) [[4]](#oasis)

* **Behavioral Analysis** [[7]](#behavior)

    * Unsupervised algorithms based on optical flow and NMF to automatically extract motor kinetics 
    * Scales to large datasets by exploiting online dictionary learning
    * We also developed a tool for acquiring movies at high speed with low cost equipment [[Github repository]](https://github.com/bensondaled/eyeblink). 
    
* **Variance Stabilization** [[8]](#vst)
    * Noise parameters estimation under the Poisson-Gaussian noise model
    * Fast algorithm that scales to large datasets
    * A basic demo can be found at `CaImAn/demos/notebooks/demo_VST.ipynb` 

## New: Online analysis

We recently incorporated a Python implementation of the OnACID [[5]](#onacid) algorithm, that enables processing data in an online mode and in real time. Check the script ```demos/general/demo_OnACID_mesoscope.py``` or the notebook ```demos/notebooks/demo_OnACID_mesoscope.ipynb``` for an application on two-photon mesoscope data provided by the Tolias lab (Baylor College of Medicine).

## Installation for calcium imaging data analysis

### Installation Changes
In May 2018, the way CaImAn is installed changed; we now register the package with Python's package management facilities rather than rely on people working out of the source tree. If you have an older install, these are things you should be aware of:
* You should not set PYTHONPATH to the CaImAn source directory any more. If you did this before (in your dotfiles or elsewhere) you should remove that.
* Unless you're installing with `pip install -e` (documented below), you should no longer work out of your checkout directory. The new install mode expects you to use caimanmanager (also documented below) to manage the demos and the place in which you'll be running code. An installed version of caimanmanager will be added to your path and should not be run out of the checkout directory.

In July 2018, Python 2.x support was removed; Python 3.6 or higher is required for CaImAn.
### Upgrading CaImAn

If you want to upgrade CaImAn (and have already used the pip installer to install it) following the instructions given in the [wiki](https://github.com/flatironinstitute/CaImAn/wiki/Updating-CaImAn).


### Installation on Mac or Linux

   * Download and install Anaconda or Miniconda (Python 3.6 version recommended) <http://docs.continuum.io/anaconda/install>
     
   ```bash
   git clone https://github.com/flatironinstitute/CaImAn
   cd CaImAn/
   conda env create -f environment.yml -n caiman
   source activate caiman
   pip install .
   ```
   If you want to develop code then replace the last command with
   ```
   pip install -e .
   ```

**For Linux users:** To make the package available from everywhere and have it working *efficiently* under any configuration ALWAYS run these commands before starting spyder:

   ```bash
   export MKL_NUM_THREADS=1
   export OPENBLAS_NUM_THREADS=1
   ```

### Setting up caimanmanager

  Once CaImAn is installed, you may want to get a working directory with code samples and datasets; pip installed a caimanmanager.py command that manages this. If you have not installed Caiman before, you can do 
  ```
  caimanmanager.py install
  ```
  or 
  ```
  python caimanmanager.py install --inplace
  ```
  if you used "pip install -e ." 
  
This will place that directory under your home directory in a directory called caiman_data. If you have, some of the demos or datafiles may have changed since your last install, to follow API changes. You can check to see if they have by doing `caimanmanager.py check`. If they have not, you may keep using them. If they have, we recommend moving your old caiman data directory out of the way (or just remove them if you have no precious data) and doing a new data install as per above.

If you prefer to manage this information somewhere else, the `CAIMAN_DATA` environment variable can be set to customise it. The caimanmanager tool and other libraries will respect that.

### Installation on Windows
   * Increase the maximum size of your pagefile to 64G or more (http://www.tomshardware.com/faq/id-2864547/manage-virtual-memory-pagefile-windows.html ) - The Windows memmap interface is sensitive to the maximum setting and leaving it at the default can cause errors when processing larger datasets
   * Download and install Anaconda (Python 3.6 recommended) <http://docs.continuum.io/anaconda/install>. We recommend telling conda to modify your PATH variable (it is a checkbox during Anaconda install, off by default)
   * Use Conda to install git (With "conda install git") - use of another commandline git is acceptable, but may lead to issues depending on default settings
   * Microsoft Build Tools for Visual Studio 2017 <https://www.visualstudio.com/downloads/#build-tools-for-visual-studio-2017>. Check the "Build Tools" box, and in the detailed view on the right check the "C/C++ CLI Tools" component too. The specifics of this occasionally change as Microsoft changes its products and website; you may need to go off-script.

Use the following menu item to launch a anaconda-enabled command prompt: start>programs>anaconda3>anaconda prompt

    ```bash
    git clone  https://github.com/flatironinstitute/CaImAn
    cd CaImAn
    conda env create -f environment.yml -n caiman
    activate caiman
    pip install . (OR pip install -e . if you want to develop code)
    copy caimanmanager.py ..
    conda install numba
    cd ..
    ```
Then run ```caimanmanager``` as described above to make a data directory.

Alternative environments:
   * [Using experimental CUDA support](/README-cuda.md)

### Installation for behavioral analysis
* Installation on Linux (Windows and MacOS are problematic with anaconda at the moment)
   * create a new environment (suggested for safety) and follow the instructions for the calcium imaging installation
   * Install spams, as explained [here](http://spams-devel.gforge.inria.fr/). Installation is not straightforward and it might take some trials to get it right

## Demos

* Notebooks: The notebooks provide a simple and friendly way to get into CaImAn and understand its main characteristics. 
They are located in the `demos/notebooks`. To launch one of the jupyter notebooks:
        
	```bash
        source activate CaImAn
        jupyter notebook --NotebookApp.iopub_data_rate_limit=1.0e10
	```
	and select the notebook from within Jupyter's browser. The argument `--NotebookApp.iopub_data_rate_limit=1.0e10` will prevent any memory issues while plotting on a notebook.
   
* demo files are also found in the demos/general subfolder. We suggest trying demo_pipeline.py first as it contains most of the tasks required by calcium imaging. For behavior use demo_behavior.py
   
* If you want to directly launch the python files, your python console still must be in the CaImAn directory. 

## On Clustering
Please read [this link](CLUSTER.md) for information on your clustering options and how to avoid trouble with them.

## Testing

* All diffs must be tested before asking for a pull request. Call ```python caimanmanager.py test``` from outside of your CaImAn folder to look for errors (you need to pass the path to the caimanmanager.py file). 
     
# Contributors:

* Andrea Giovannucci, **Flatiron Institute, Simons Foundation**
* Eftychios A. Pnevmatikakis, **Flatiron Institute, Simons Foundation** 
* Johannes Friedrich, **Flatiron Institute, Simons Foundation**
* Mariano Tepper, **Flatiron Institute, Simons Foundation**
* Erick, Cobos, **Baylor College of Medicine**
* Valentina Staneva, **University of Washington**
* Ben Deverett, **Princeton University**
* Jérémie Kalfon, **University of Kent, ECE paris** 

A complete list of contributors can be found [here](https://github.com/flatironinstitute/CaImAn/graphs/contributors).

# References

The following references provide the theoretical background and original code for the included methods.

### Software package detailed description and benchmarking

If you use this code please cite the corresponding papers where original methods appeared (see References below), as well as: 

<a name="caiman"></a>[1] Giovannucci A., Friedrich J., Gunn P., Kalfon J., Koay S.A., Taxidis J., Najafi F., Gauthier J.L., Zhou P., Tank D.W., Chklovskii D.B., Pnevmatikakis E.A. (2018). CaImAn: An open source tool for scalable Calcium Imaging data Analysis. bioarXiv preprint. [[paper]](https://doi.org/10.1101/339564)

### Deconvolution and demixing of calcium imaging data

<a name="neuron"></a>[1] Pnevmatikakis, E.A., Soudry, D., Gao, Y., Machado, T., Merel, J., ... & Paninski, L. (2016). Simultaneous denoising, deconvolution, and demixing of calcium imaging data. Neuron 89(2):285-299, [[paper]](http://dx.doi.org/10.1016/j.neuron.2015.11.037), [[Github repository]](https://github.com/epnev/ca_source_extraction). 

<a name="struct"></a>[2] Pnevmatikakis, E.A., Gao, Y., Soudry, D., Pfau, D., Lacefield, C., ... & Paninski, L. (2014). A structured matrix factorization framework for large scale calcium imaging data analysis. arXiv preprint arXiv:1409.2903. [[paper]](http://arxiv.org/abs/1409.2903). 

<a name="cnmfe"></a>[3] Zhou, P., Resendez, S. L., Stuber, G. D., Kass, R. E., & Paninski, L. (2016). Efficient and accurate extraction of in vivo calcium signals from microendoscopic video data. arXiv preprint arXiv:1605.07266. [[paper]](https://arxiv.org/abs/1605.07266), [[Github repository]](https://github.com/zhoupc/CNMF_E).

<a name="oasis"></a>[4] Friedrich J. and Paninski L. Fast active set methods for online spike inference from calcium imaging. NIPS, 29:1984-1992, 2016. [[paper]](https://papers.nips.cc/paper/6505-fast-active-set-methods-for-online-spike-inference-from-calcium-imaging), [[Github repository]](https://github.com/j-friedrich/OASIS).

### Online Analysis

<a name="onacid"></a>[5] Giovannucci, A., Friedrich J., Kaufman M., Churchland A., Chklovskii D., Paninski L., & Pnevmatikakis E.A. (2017). OnACID: Online analysis of calcium imaging data in real data. NIPS 2017, pp. 2378-2388. [[paper]](http://papers.nips.cc/paper/6832-onacid-online-analysis-of-calcium-imaging-data-in-real-time)

### Motion Correction

<a name="normcorre"></a>[6] Pnevmatikakis, E.A., and Giovannucci A. (2017). NoRMCorre: An online algorithm for piecewise rigid motion correction of calcium imaging data. Journal of Neuroscience Methods, 291:83-92 [[paper]](https://doi.org/10.1016/j.jneumeth.2017.07.031), [[Github repository]](https://github.com/simonsfoundation/normcorre).

### Behavioral Analysis

<a name="behavior"></a>[7] Giovannucci, A., Pnevmatikakis, E. A., Deverett, B., Pereira, T., Fondriest, J., Brady, M. J., ... & Masip, D. (2017). Automated gesture tracking in head-fixed mice. Journal of Neuroscience Methods, 300:184-195. [[paper]](https://doi.org/10.1016/j.jneumeth.2017.07.014).



### Variance Stabilization

<a name="vst"></a>[8] Tepper, M., Giovannucci, A., and Pnevmatikakis, E (2018). Anscombe meets Hough: Noise variance stabilization via parametric model estimation. In ICASSP, 2018. [[paper]](https://marianotepper.github.io/papers/anscombe-meets-hough.pdf). [[Github repository]](https://github.com/marianotepper/hough-anscombe)


## Related packages

The implementation of this package is developed in parallel with a MATLAB toobox, which can be found [here](https://github.com/epnev/ca_source_extraction). 

Some tools that are currently available in Matlab but have been ported to CaImAn are

- [MCMC spike inference](https://github.com/epnev/continuous_time_ca_sampler) 
- [Group LASSO initialization and spatial CNMF](https://github.com/danielso/ROI_detect)


## Troubleshooting

A list of known issues can be found [here](https://github.com/flatironinstitute/CaImAn/wiki/Known-Issues). If you still encounter problems please open an issue.

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

For the constrained deconvolution method (```deconvolution.constrained_foopsi```) various solvers can be used, some of which require additional packages:

1. ```'cvxpy'```: (default) For this option, the following packages are needed:
  * [CVXOPT](http://cvxopt.org/) optional.
  * [CVXPY](http://www.cvxpy.org/) optional.
2. ```'cvx'```: For this option, the following packages are needed:
  * [CVXOPT](http://cvxopt.org/) optional.
  * [PICOS](http://picos.zib.de/) optional.

In general ```'cvxpy'``` can be faster, when using the 'ECOS' or 'SCS' solvers, which are included with the CVXPY installation. Note that these dependencies are circumvented by using the OASIS algoritm for deconvolution.


# Documentation & Wiki

Documentation of the code can be found [here](http://flatironinstitute.github.io/CaImAn/). 
Moreover, our [wiki page](https://github.com/flatironinstitute/CaImAn/wiki) covers some aspects of the code.

# Acknowledgements

Special thanks to the following people for letting us use their datasets for our various demo files:

* Weijian Yang, Darcy Peterka, Rafael Yuste, Columbia University
* Sue Ann Koay, David Tank, Princeton University
* Manolis Froudarakis, Jake Reimers, Andreas Tolias, Baylor College of Medicine


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
