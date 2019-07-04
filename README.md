Position available
======

The CaImAn team is hiring! We're looking for a data scientist/software engineer with a strong research component. For more information please follow [this link](https://simonsfoundation.wd1.myworkdayjobs.com/en-US/simonsfoundationcareers/job/162-Fifth-Avenue/Software-Engineer_R0000500).

CaImAn
======
<img src="https://github.com/flatironinstitute/CaImAn/blob/master/docs/LOGOS/Caiman_logo_FI.png" width="500" align="right">


[![Join the chat at https://gitter.im/agiovann/SOURCE_EXTRACTION_PYTHON](https://badges.gitter.im/agiovann/SOURCE_EXTRACTION_PYTHON.svg)](https://gitter.im/agiovann/SOURCE_EXTRACTION_PYTHON?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)


A Python toolbox for large scale **Ca**lcium **Im**aging data **An**alysis and behavioral analysis.

CaImAn implements a set of essential methods required in the analysis pipeline of large scale calcium imaging data. Fast and scalable algorithms are implemented for motion correction, source extraction, spike deconvolution, and component registration across multiple days. It is suitable for both two-photon and one-photon fluorescence microscopy data, and can be run in both batch and online modes. CaImAn also contains some routines for the analysis of behavior from video cameras. A list of features as well as relevant references can be found [here](https://github.com/flatironinstitute/CaImAn/wiki/CaImAn-features-and-references).

## Companion paper
A paper explaining most of the implementation details and benchmarking can be found [here](https://elifesciences.org/articles/38173).

```
@article{giovannucci2019caiman,
  title={CaImAn: An open source tool for scalable Calcium Imaging data Analysis},
  author={Giovannucci, Andrea and Friedrich, Johannes and Gunn, Pat and Kalfon, Jeremie and Brown, Brandon L and Koay, Sue Ann and Taxidis, Jiannis and Najafi, Farzaneh and Gauthier, Jeffrey L and Zhou, Pengcheng and Khakh, Baljit S and Tank, David W and Chklovskii, Dmitri B and Pnevmatikakis, Eftychios A},
  journal={eLife},
  volume={8},
  pages={e38173},
  year={2019},
  publisher={eLife Sciences Publications Limited}
}
```

All the results and figures of the paper can be regenerated using this package. For more information visit this [page](https://github.com/flatironinstitute/CaImAn/tree/master/use_cases/eLife_scripts).

## New: Code refactoring (October 2018)

We recently refactored the code to simplify the parameter setting and usage of the various algorithms. The code now is based around the following objects:

* `params`: A single object containing a set of dictionaries with the parameters used in all the algorithms. It can be set and changed easily and is passed into all the algorithms.
* `MotionCorrect`: An object for motion correction which can be used for both rigid and piece-wise rigid motion correction.
* `cnmf`: An object for running the CaImAn batch algorithm either in patches or not, suitable for both two-photon (CNMF) and one-photon (CNMF-E) data.
* `online_cnmf`: An object for running the CaImAn online (OnACID) algorithm on two-photon data with or without motion correction.
* `estimates`: A single object that stores the results of the algorithms (CaImAn batch, CaImAn online) in a unified way that also contains plotting methods. For an interpretation of the various entries of the `estimates` object see [here](https://github.com/flatironinstitute/CaImAn/wiki/Interpreting-Results).
   
To see examples of how these methods are used, please consult the demos. While the `cnmf` methods can also be called in the old way by passing all the parameters when initializing the `cnmf` object, we recommend using the `params` object. Similarly, to run the CaImAn online algorithm it is recommended to pass a `params` object inside the `online_cnmf` object. Older scripts should be usable with the latest version of the code except for online analysis where the `cnmf` object will need to be replaced with an `online_cnmf` object. The results should be read from `estimates`, i.e., `cnm.estimates.C` as opposed to `cnm.C`.

## New: Voltage Imaging (June 2019)

We recently added the code for analyzing voltage imaging data. The analysis is based on following objects:

* `volparams`: An object for setting parameters of voltage imaging. It can be set and changed easily and is passed into the algorithms.
* `MotionCorrect`: An object for motion correction which can be used for both rigid and piece-wise rigid motion correction.
* `VOLPY`: An object for running the spike detection algorithm and saving results.
   
To see examples of how these methods are used, please consult the demo_pipeline_voltage_imaging.py in demos. 
## Installation for calcium imaging data analysis

### Installation Changes
In May 2018, the way CaImAn is installed changed; we now register the package with Python's package management facilities rather than rely on people working out of the source tree. If you have an older install, these are things you should be aware of:
* You should not set PYTHONPATH to the CaImAn source directory any more. If you did this before (in your dotfiles or elsewhere) you should remove that.
* Unless you're installing with `pip install -e` (documented below), you should no longer work out of your checkout directory. The new install mode expects you to use caimanmanager (also documented below) to manage the demos and the place in which you'll be running code. An installed version of caimanmanager will be added to your path and should not be run out of the checkout directory.

In July 2018, Python 2.x support was removed; Python 3.6 or higher is required for CaImAn.

### Upgrading CaImAn

If you want to upgrade CaImAn (and have already used the pip installer to install it) follow the instructions given in the [wiki](https://github.com/flatironinstitute/CaImAn/wiki/Updating-CaImAn).

Also, if you want to install new packages into your conda environment for CaImAn, it is important that you not mix conda-forge and the defaults channel; we recommend only using conda-forge. To ensure you're not mixing channels, perform the install (inside your environment) as follows:
   ```bash
   conda install -c conda-forge --override-channels NEW_PACKAGE_NAME
   ```
You will notice that any packages installed this way will mention, in their listing, that they're from conda-forge, with none of them having a blank origin. If you fail to do this, differences between how packages are built in conda-forge versus the default conda channels may mean that some packages (e.g. OpenCV) stop working despite showing as installed.

### Installation on Windows
On Windows, please follow the install instructions [here](/INSTALL-windows.md) .

### Installation on Mac or Linux

   * Download and install Anaconda or Miniconda (Python 3.x version) <http://docs.continuum.io/anaconda/install>
     
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
   If any of these steps gives you errors do not proceed to the following step without resolving it

#### known issues
    
With OSX Mojave you will need to perform the following steps before your first install:
    
   ```
   xcode-select --install
   open /Library/Developer/CommandLineTools/Packages/
   ```

(install the package file you will find in the folder that pops up)

### Setting up environment variables 

To make the package work *efficiently* and eliminate "crosstalk" between different processes, run these commands before launching Python (this is for Linux and OSX):

   ```bash
   export MKL_NUM_THREADS=1
   export OPENBLAS_NUM_THREADS=1
   ```   
The commands should be run every time before launching python. It is recommended that you save these values inside your environment so you don't have to repeat this process every time. You can do this by following the instructions [here](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#saving-environment-variables).

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


Alternative environments:
   * [Using GPU](/README-GPU.md)
   * [Older Linux Distros](/README-Distros.md)
   
### Known Issues

A list of known issues can be found [here](https://github.com/flatironinstitute/CaImAn/wiki/Known-Issues). If you still encounter problems please open an issue.  

## Documentation & Wiki

Documentation of the code can be found [here](http://flatironinstitute.github.io/CaImAn/). 
Moreover, our [wiki page](https://github.com/flatironinstitute/CaImAn/wiki) covers some aspects of the code. 

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


## Testing

* All diffs must be tested before asking for a pull request. Call ```python caimanmanager.py test``` from outside of your CaImAn folder to look for errors (you need to pass the path to the caimanmanager.py file). 
     
# Contributors:

* Andrea Giovannucci, **Flatiron Institute, Simons Foundation**
* Eftychios A. Pnevmatikakis, **Flatiron Institute, Simons Foundation** 
* Johannes Friedrich, **Flatiron Institute, Simons Foundation**
* Pat Gunn, **Flatiron Institute, Simons Foundation**
* Erick, Cobos, **Baylor College of Medicine**
* Valentina Staneva, **University of Washington**
* Ben Deverett, **Princeton University**
* Jérémie Kalfon, **University of Kent, ECE paris** 
* Mike Schachter, **Inscopix**
* Brandon Brown, **UCSF**

A complete list of contributors can be found [here](https://github.com/flatironinstitute/CaImAn/graphs/contributors).

# References

The following references provide the theoretical background and original code for the included methods.

### Software package detailed description and benchmarking

If you use this code please cite the corresponding papers where original methods appeared (see References below), as well as: 

<a name="caiman"></a>[1] Giovannucci A., Friedrich J., Gunn P., Kalfon J., Brown, B., Koay S.A., Taxidis J., Najafi F., Gauthier J.L., Zhou P., Baljit, K.S., Tank D.W., Chklovskii D.B., Pnevmatikakis E.A. (2019). CaImAn: An open source tool for scalable Calcium Imaging data Analysis. eLife 8, e38173. [[paper]](https://elifesciences.org/articles/38173)

### Deconvolution and demixing of calcium imaging data

<a name="neuron"></a>[2] Pnevmatikakis, E.A., Soudry, D., Gao, Y., Machado, T., Merel, J., ... & Paninski, L. (2016). Simultaneous denoising, deconvolution, and demixing of calcium imaging data. Neuron 89(2):285-299, [[paper]](http://dx.doi.org/10.1016/j.neuron.2015.11.037), [[Github repository]](https://github.com/epnev/ca_source_extraction). 

<a name="struct"></a>[3] Pnevmatikakis, E.A., Gao, Y., Soudry, D., Pfau, D., Lacefield, C., ... & Paninski, L. (2014). A structured matrix factorization framework for large scale calcium imaging data analysis. arXiv preprint arXiv:1409.2903. [[paper]](http://arxiv.org/abs/1409.2903). 

<a name="cnmfe"></a>[4] Zhou, P., Resendez, S. L., Stuber, G. D., Kass, R. E., & Paninski, L. (2016). Efficient and accurate extraction of in vivo calcium signals from microendoscopic video data. arXiv preprint arXiv:1605.07266. [[paper]](https://arxiv.org/abs/1605.07266), [[Github repository]](https://github.com/zhoupc/CNMF_E).

<a name="oasis"></a>[5] Friedrich J. and Paninski L. Fast active set methods for online spike inference from calcium imaging. NIPS, 29:1984-1992, 2016. [[paper]](https://papers.nips.cc/paper/6505-fast-active-set-methods-for-online-spike-inference-from-calcium-imaging), [[Github repository]](https://github.com/j-friedrich/OASIS).

### Online Analysis

<a name="onacid"></a>[6] Giovannucci, A., Friedrich J., Kaufman M., Churchland A., Chklovskii D., Paninski L., & Pnevmatikakis E.A. (2017). OnACID: Online analysis of calcium imaging data in real data. NIPS 2017, pp. 2378-2388. [[paper]](http://papers.nips.cc/paper/6832-onacid-online-analysis-of-calcium-imaging-data-in-real-time)

### Motion Correction

<a name="normcorre"></a>[7] Pnevmatikakis, E.A., and Giovannucci A. (2017). NoRMCorre: An online algorithm for piecewise rigid motion correction of calcium imaging data. Journal of Neuroscience Methods, 291:83-92 [[paper]](https://doi.org/10.1016/j.jneumeth.2017.07.031), [[Github repository]](https://github.com/simonsfoundation/normcorre).

### Behavioral Analysis

<a name="behavior"></a>[8] Giovannucci, A., Pnevmatikakis, E. A., Deverett, B., Pereira, T., Fondriest, J., Brady, M. J., ... & Masip, D. (2017). Automated gesture tracking in head-fixed mice. Journal of Neuroscience Methods, 300:184-195. [[paper]](https://doi.org/10.1016/j.jneumeth.2017.07.014).

### Variance Stabilization

<a name="vst"></a>[9] Tepper, M., Giovannucci, A., and Pnevmatikakis, E (2018). Anscombe meets Hough: Noise variance stabilization via parametric model estimation. In ICASSP, 2018. [[paper]](https://marianotepper.github.io/papers/anscombe-meets-hough.pdf). [[Github repository]](https://github.com/marianotepper/hough-anscombe)


## Related packages

The implementation of this package is developed in parallel with a MATLAB toobox, which can be found [here](https://github.com/epnev/ca_source_extraction). 

Some tools that are currently available in Matlab but have been ported to CaImAn are

- [MCMC spike inference](https://github.com/epnev/continuous_time_ca_sampler) 
- [Group LASSO initialization and spatial CNMF](https://github.com/danielso/ROI_detect)

## Dependencies

A list of dependencies can be found in the [environment file](https://github.com/flatironinstitute/CaImAn/blob/master/environment.yml).



## Questions, comments, issues

Please use the [gitter chat room](https://gitter.im/agiovann/Constrained_NMF) for questions and comments and create an issue for any bugs you might encounter.

## Acknowledgements

Special thanks to the following people for letting us use their datasets for our various demo files:

* Weijian Yang, Darcy Peterka, Rafael Yuste, Columbia University
* Sue Ann Koay, David Tank, Princeton University
* Manolis Froudarakis, Jake Reimers, Andreas Tolias, Baylor College of Medicine


## License

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
