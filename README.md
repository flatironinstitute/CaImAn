Position available
======

The CaImAn team is hiring! We're looking for a data scientist/software engineer with a strong research component. For more information please follow [this link](https://simonsfoundation.wd1.myworkdayjobs.com/en-US/simonsfoundationcareers/job/162-Fifth-Avenue/Software-Engineer_R0000500).


CaImAn
======
<img src="https://github.com/flatironinstitute/CaImAn/blob/master/docs/LOGOS/Caiman_logo_FI.png" width="500" align="right">



[![Join the chat at https://gitter.im/agiovann/SOURCE_EXTRACTION_PYTHON](https://badges.gitter.im/agiovann/SOURCE_EXTRACTION_PYTHON.svg)](https://gitter.im/agiovann/SOURCE_EXTRACTION_PYTHON?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)


A Python toolbox for large scale **Ca**lcium **Im**aging data **An**alysis and behavioral analysis.

CaImAn implements a set of essential methods required in the analysis pipeline of large scale calcium imaging data. Fast and scalable algorithms are implemented for motion correction, source extraction, spike deconvolution, and component registration across multiple days. It is suitable for both two-photon and one-photon fluorescence microscopy data, and can be run in both batch and online modes. CaImAn also contains some routines for the analysis of behavior from video cameras. A list of features as well as relevant references can be found [here](https://github.com/flatironinstitute/CaImAn/wiki/CaImAn-features-and-references).

## Web-based Docs
Documentation for CaImAn (including install instructions) can be found online [here](https://caiman.readthedocs.io/en/master/Overview.html).

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

## New: Analysis of Voltage Imaging data (December 2019)

We recently added VolPy for analyzing voltage imaging data. The analysis is based on following objects:

* `MotionCorrect`: An object for motion correction which can be used for both rigid and piece-wise rigid motion correction.
* `volparams`: An object for setting parameters of voltage imaging. It can be set and changed easily and is passed into the algorithms.
* `VOLPY`: An object for running the spike detection algorithm and saving results.

If you utilize VolPy for neurons segmentation, please make sure model and relevant files are in your caiman_data folder.
   
To see examples of how these methods are used, please consult the demo_pipeline_voltage_imaging.py in demos/general folder. 

## New: Installation through conda-forge (August 2019)

Beginning in August 2019 we have an experimental binary release of the software in the conda-forge package repos. This is intended for people who can use CaImAn as a library, interacting with it as the demos do. It also does not need a compiler. It is not suitable for people intending to change the CaImAn codebase. Comfort with conda is still required. If you wish to use the binary package, you do not need the sources (including this repo) at all. Installation and updating instructions can be found [here](./docs/source/Installation.rst).

You will still need to use caimanmanager.py afterwards to create a data directory. If you install this way, do not follow any of the other install instructions below.

## New: Exporting results, GUI and NWB support (July 2019)

You can now use the `save` method included in both the `CNMF` and `OnACID` objects to export the results (and parameters used) of your analysis. The results are saved in an HDF5 file that you can then load in a graphical user interface for more inspection. The GUI will allow you to inspect the results and modify the selected components based on the various quality metrics. For more information click [here](docs/GUI.md)

The [Neurodata Without Borders (NWB)](https://www.nwb.org/) file format is now supported by CaImAn. You read and analyze NWB files and can save the results of the analysis (`Estimates` object) back to the original NWB file. Consult this [demo](use_cases/NWB/demo_pipeline_NWB.py) for an example on how to use this feature.

**To use CaImAn with these additional features you'll need to create a new environment following the usual instructions.**

## Documentation & Wiki

Documentation of the code can be found [here](https://caiman.readthedocs.io/en/master/). 

### Installation for behavioral analysis
* Installation on Linux (Windows and MacOS are problematic with anaconda at the moment)
   * create a new environment (suggested for safety) and follow the instructions for the calcium imaging installation
   * Install spams, as explained [here](http://spams-devel.gforge.inria.fr/). Installation is not straightforward and it might take some trials to get it right.

## Demos

* Notebooks: The notebooks provide a simple and friendly way to get into CaImAn and understand its main characteristics. 
They are located in the `demos/notebooks`. To launch one of the jupyter notebooks:
        
	```bash
        source activate CaImAn
        jupyter notebook --NotebookApp.iopub_data_rate_limit=1.0e10
	```
	and select the notebook from within Jupyter's browser. The argument `--NotebookApp.iopub_data_rate_limit=1.0e10` will prevent any memory issues while plotting on a notebook.
   
* demo files are also found in the demos/general subfolder. We suggest trying demo_pipeline.py first as it contains most of the tasks required by calcium imaging. For behavior use demo_behavior.py

* If you modify the demos to use them for your own data it is recommended that you save them in a different file to avoid file conflicts during updating.

* If you want to directly launch the python files, your python console still must be in the CaImAn directory. 

## Testing

* All diffs must be tested before asking for a pull request. Call ```python caimanmanager.py test``` from outside of your CaImAn folder to look for errors (you need to pass the path to the caimanmanager.py file). 
     
# Main developers:

* Eftychios A. Pnevmatikakis, **Flatiron Institute, Simons Foundation** 
* Andrea Giovannucci, **University of North Carolina, Chapel Hill**
* Johannes Friedrich, **Flatiron Institute, Simons Foundation**
* Pat Gunn, **Flatiron Institute, Simons Foundation**

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


## Other docs in this repo
* [Running CaImAn on a Cluster](docs/CLUSTER.md)
* [Install quirks on some Linux Distributions](docs/README-Distros.md)
* [How CaImAn can use your GPUs](docs/README-GPU.md)
* [The CaImAn GUI](docs/GUI.md)


## Related packages

The implementation of this package is developed in parallel with a MATLAB toobox, which can be found [here](https://github.com/epnev/ca_source_extraction). 

Some tools that are currently available in Matlab but have not been ported to CaImAn are

- [MCMC spike inference](https://github.com/epnev/continuous_time_ca_sampler) 

## Dependencies

A list of dependencies can be found in the [environment file](https://github.com/flatironinstitute/CaImAn/blob/master/environment.yml).


## Questions, comments, issues

Please use the [gitter chat room](https://gitter.im/agiovann/Constrained_NMF) for questions and comments and create an issue for any bugs you might encounter.

## Acknowledgements

Special thanks to the following people for letting us use their datasets for our various demo files:

* Weijian Yang, Darcy Peterka, Rafael Yuste, Columbia University
* Sue Ann Koay, David Tank, Princeton University
* Manolis Froudarakis, Jake Reimers, Andreas Tolias, Baylor College of Medicine
* Clay Lacefield, Randy Bruno, Columbia University
* Daniel Aharoni, Peyman Golshani, UCLA

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
