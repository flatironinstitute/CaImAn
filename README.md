<a href="https://colab.research.google.com/drive/1vkp-uPV8tKavmX12bcN2L-jYH8_MgmHL?usp=sharing"><img src="https://img.shields.io/badge/-Colab%20Demo-blue" /></a>

CaImAn
======
<img src="https://github.com/flatironinstitute/Caiman/blob/master/docs/LOGOS/Caiman_logo_FI.png" width="500" align="right">

A Python toolbox for large scale **Ca**lcium **Im**aging data **An**alysis.

CaImAn implements a set of essential methods required to analyze calcium and voltage imaging data. It provides fast and scalable algorithms for motion correction, source extraction, spike deconvolution, and component registration across multiple days. It is suitable for both two-photon and one-photon fluorescence microscopy data, and can be run in both offline and online modes. A full list of features as well as relevant references can be found [here](https://caiman.readthedocs.io/en/latest/Caiman_features_and_references.html).

# Quick start :rocket:
To get started quickly working on a demo notebook, just take the following three steps:

### Step 1: Install caiman
The following is all done in your anaconda prompt, starting in your base environment:
  
    conda install mamba -n base -c conda-forge   # install mamba in base environment
    mamba create -n caiman -c conda-forge caiman # install caiman
    conda activate caiman  # activate virtual environment

### Step 2: Download code samples and data sets
Here we will create a working directory with code samples and related data in a folder called `caiman_data`. Once this step is done, you will easily be able to run any of the demos provided with Caiman! While still in the caiman environment in conda from the previous step:  

    caimanmanager.py install

### Step 3: Try out a demo notebook
In Step 2, you created a `caiman_data` folder in your home directory. Here, we `cd` to the demo notebooks folder in `caiman_data`, and fire up a jupyter notebook:

    cd home/caiman_data/demos/notebooks  # go to demo notebooks
    jupyter notebook  # fire up a jupyter notebook

Jupyter will open: click on `demo_pipeline.ipynb` to get started with a demo! Note that what counts as `home` in the first command depends on your OS, so be sure to fill in your actual home directory.

## For more details
 There are more detailed installation and setup instructions [here](./docs/source/Installation.rst). For instance go there to learn how to build a development environment, or if you want to change where `caiman_data` is placed.

## Installation problem?
We want Caiman to install easily on Linux, Mac, and Windows. If you run into problems, please send us a [message on Gitter](https://app.gitter.im/#/room/#agiovann_Constrained_NMF:gitter.im).  


# Where to go next? :red_car:
Caiman provides demo notebooks to explain and demonstrate each of our main pipelines, and you can adapt these for your personal needs. The main use cases and notebooks are:

| Use case | Demo notebook | 
|:-------- |:------------- |
| Motion correction | demo_motion_correction.ipynb | 
| CNMF for 2p or low-noise 1p data | demo_pipeline.ipynb |  
| CNMFE for 1p data  | demo_pipeline.ipynb |  
| Volpy for voltage data | demo_pipeline_voltag_imaging.ipynb |  
| Volumetric (3D) CNMF | demo_caiman_cnmf_3D.ipynb |  
| CNMF for dendrites | data_dendritic.ipynb |  
| Online CNMF (OnACID) | demo_OnACID_mesoscope.ipynb |
| Online volumetric CNMF | demo_online_3D.ipynb |
| Online CNMFE (OnACID-E) | demo_realtime_cnmfE.ipynb |
| Register cells across sessions | demo_multisession_registration.ipynb |  

A comprehensive list of references where these methods were developed can be found [here](https://caiman.readthedocs.io/en/master/CaImAn_features_and_references.html#references). 


# Need help?
If you need help using Caiman, or hit an installation snag, we recommend asking about it in the [Gitter forum](https://app.gitter.im/#/room/#agiovann_Constrained_NMF:gitter.im). If you have found a bug, we recommend searching the [issues at github](https://github.com/flatironinstitute/CaImAn/issues) and opening a new one if you can't find it there. If there is a feature you would like to see, discuss at Gitter or open an issue at Github. 

# How to contribute :hammer:
 Caiman is an open-source project and improves because of people like you. If you found something about Caiman that you want to improve, then you are qualified to contribute! See the [contributing page](./CONTRIBUTING.md) for more details on how to participate in the project: as an open source project driven by community engagement, we are always looking for more contributors!

## Papers and data

### Main paper
A paper explaining most of the implementation details and benchmarking can be found [here](https://elifesciences.org/articles/38173).

```
@article{giovannucci2019caiman,
  title={Caiman: An open source tool for scalable Calcium Imaging data Analysis},
  author={Giovannucci, Andrea and Friedrich, Johannes and Gunn, Pat and Kalfon, Jeremie and Brown, Brandon L and Koay, Sue Ann and Taxidis, Jiannis and Najafi, Farzaneh and Gauthier, Jeffrey L and Zhou, Pengcheng and Khakh, Baljit S and Tank, David W and Chklovskii, Dmitri B and Pnevmatikakis, Eftychios A},
  journal={eLife},
  volume={8},
  pages={e38173},
  year={2019},
  publisher={eLife Sciences Publications Limited}
}
```

All the results and figures of the paper can be regenerated using this package. For more information visit this [page](https://github.com/flatironinstitute/caiman_use_cases/tree/master/use_cases/eLife_scripts).

Caiman implements a variety of algorithms for analyzing calcium (and voltage) imaging data. A list of references that provide the theoretical background and original code for the included methods can be found [here](https://caiman.readthedocs.io/en/latest/Caiman_features_and_references.html). 
 
If you use this code please cite the corresponding papers where original methods appeared as well the companion paper.

### Videos

These talks by Andrea Giovannucci from past Caiman workshops/events are an excellent start for newcomers. They go through NoRMCorre, CNMF(E) and VolPy.

Open Neuroscience talks, this is a good high-level introduction to Caiman:

https://www.youtube.com/watch?v=5APzPRbzUIA

Nemonic workshops, more in depth:

https://www.youtube.com/watch?v=KjHrjhvhRy0

https://www.youtube.com/watch?v=rUwIqU6gVvw

https://www.youtube.com/watch?v=NZZ6_zo0YIM

https://www.youtube.com/watch?v=z6TlH28MLRo


### Real-time analysis of microendoscopic 1p data

Our online algorithms can be used for real-time analysis of live-streaming data. An example for real-time analysis of microendoscopic 1p data is shown in the notebook `demos/notebooks/demo_realtime_cnmfE.ipynb`.
For more information about the approach check the [paper](https://doi.org/10.1371/journal.pcbi.1008565).

### Analysis pipeline for Voltage Imaging data

VolPy is an analysis pipeline for voltage imaging data. The analysis is based on following objects:

* `MotionCorrect`: An object for motion correction which can be used for both rigid and piece-wise rigid motion correction.
* `volparams`: An object for setting parameters of voltage imaging. It can be set and changed easily and is passed into the algorithms.
* `VOLPY`: An object for running the spike detection algorithm and saving results.

The object detection network Mask R-CNN in VolPy is now compatible with tensorflow 2.4 or above.

To see examples of how these methods are used, please consult the `demo_pipeline_voltage_imaging.py` script in the `demos/general` folder. For more information about the approach check the [general paper](https://journals.plos.org/ploscompbiol/article/comments?id=10.1371/journal.pcbi.1008806) on this pipeline.

## Documentation & Wiki

Documentation of the code can be found [here](https://caiman.readthedocs.io/en/master/). 

Other docs:
* [Running Caiman on a Cluster](docs/CLUSTER.md)
* [Install quirks on some Linux Distributions](docs/README-Distros.md)
* [How Caiman can use your GPUs](docs/README-GPU.md)

# Main developers:
* (emeritus) Eftychios A. Pnevmatikakis, **Flatiron Institute, Simons Foundation** 
* (emeritus) Andrea Giovannucci, **University of North Carolina, Chapel Hill**
* Johannes Friedrich, **Flatiron Institute, Simons Foundation**
* Changlia Cai, **University of North Carolina, Chapel Hill**
* Pat Gunn, **Flatiron Institute, Simons Foundation**


A complete list of contributors can be found [here](https://github.com/flatironinstitute/Caiman/graphs/contributors).

Currently Pat Gunn and Johannes Friedrich are the most active maintainers.

# Supplementary repos
* [use\_cases repo](https://github.com/flatironinstitute/caiman_use_cases) - Contains additional code (unmaintained) demonstrating how to use/extend Caiman

### Related packages
* [jnormcorre](https://github.com/apasarkar/jnormcorre) - [JAX](https://github.com/google/jax) implementation of NoRMCorre for motion correction using JAX acceleration
* [funimag](https://github.com/paninski-lab/funimag) - matrix decomposition for denoising and compression
* [mesmerize-core](https://github.com/nel-lab/mesmerize-core) - parameter optimization, data organization and visualizations with caiman
* [improv](https://github.com/project-improv/improv) - a platform for creating online analysis workflows

If you have questions about these related packages please reach out to them directly.

## Questions, comments, issues

For support, you can create a Github issue describing any bugs you wish to report, or any feature requests you may have.

You may also use the [gitter chat room](https://gitter.im/agiovann/Constrained_NMF) for discussion.

Finally, you may reach out via email to one of the primary maintainers (above).

## Acknowledgements

Special thanks to the following people for letting us use their datasets in demo files:

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
