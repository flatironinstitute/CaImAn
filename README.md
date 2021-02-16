CaImAn
======
<img src="https://github.com/flatironinstitute/CaImAn/blob/master/docs/LOGOS/Caiman_logo_FI.png" width="500" align="right">



[![Join the chat at https://gitter.im/agiovann/SOURCE_EXTRACTION_PYTHON](https://badges.gitter.im/agiovann/SOURCE_EXTRACTION_PYTHON.svg)](https://gitter.im/agiovann/SOURCE_EXTRACTION_PYTHON?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)


A Python toolbox for large scale **Ca**lcium **Im**aging data **An**alysis and behavioral analysis.

CaImAn implements a set of essential methods required in the analysis pipeline of large scale calcium imaging data. Fast and scalable algorithms are implemented for motion correction, source extraction, spike deconvolution, and component registration across multiple days. It is suitable for both two-photon and one-photon fluorescence microscopy data, and can be run in both batch and online modes. CaImAn also contains some routines for the analysis of behavior from video cameras. A list of features as well as relevant references can be found [here](https://caiman.readthedocs.io/en/latest/CaImAn_features_and_references.html).

## Web-based Docs
Documentation for CaImAn (including install instructions) can be found online [here](https://caiman.readthedocs.io/en/master/Overview.html).

## Companion paper and related references
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

CaImAn implements a variety of algorithms for analyzing calcium (and voltage) imaging data. A list of references that provide the theoretical background and original code for the included methods can be found [here](https://caiman.readthedocs.io/en/latest/CaImAn_features_and_references.html). 
 
If you use this code please cite the corresponding papers where original methods appeared as well the companion paper.

## New: Real-time analysis of microendoscopic 1p data (January 2021)

Our online algorithms can be used for newly enabled real-time analysis of live-streaming data. An example for real-time analysis of microendoscopic 1p data is shown in the notebook `demos/notebooks/demo_realtime_cnmfE.ipynb`.
For more information about the approach check the [paper](https://doi.org/10.1371/journal.pcbi.1008565).

## New: Online analysis for microendoscopic 1p data (January 2020)

We developed two approaches for analyzing streaming microendoscopic 1p data with high speed and low memory requirements. 
The first approach (OnACID-E) features a direct implementation of the CNMF-E algorithm in an online setup. An example can be seen in the notebook `demos/notebooks/demo_online_cnmfE.ipynb`. The second approach first extracts the background by modeling it with a simple convolutional neural network (Ring-CNN) and proceeds with the analysis using the OnACID algorithm, see `demos/notebooks/demo_Ring_CNN.ipynb`.

## New: Analysis pipeline for Voltage Imaging data (December 2019)

We recently added VolPy, an analysis pipeline for voltage imaging data. The analysis is based on following objects:

* `MotionCorrect`: An object for motion correction which can be used for both rigid and piece-wise rigid motion correction.
* `volparams`: An object for setting parameters of voltage imaging. It can be set and changed easily and is passed into the algorithms.
* `VOLPY`: An object for running the spike detection algorithm and saving results.

In order to use VolPy, you must install Keras into your conda environment. You can do this by activating your environment, and then issuing the command "conda install -c conda-forge keras".

To see examples of how these methods are used, please consult the `demo_pipeline_voltage_imaging.py` script in the `demos/general` folder. For more information about the approach check the [preprint](https://www.biorxiv.org/content/10.1101/2020.01.02.892323v1).

## New: Installation through conda-forge (August 2019)

Beginning in August 2019 we have an experimental binary release of the software in the conda-forge package repos. This is intended for people who can use CaImAn as a library, interacting with it as the demos do. It also does not need a compiler. It is not suitable for people intending to change the CaImAn codebase. Comfort with conda is still required. If you wish to use the binary package, you do not need the sources (including this repo) at all. Installation and updating instructions can be found [here](./docs/source/Installation.rst).

You will still need to use caimanmanager.py afterwards to create a data directory. If you install this way, do not follow any of the other install instructions below.

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


## Other docs in this repo
* [Running CaImAn on a Cluster](docs/CLUSTER.md)
* [Install quirks on some Linux Distributions](docs/README-Distros.md)
* [How CaImAn can use your GPUs](docs/README-GPU.md)

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
