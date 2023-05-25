# CaImAn 
<img src="docs/LOGOS/Caiman_logo_FI.png" width="400" align="right">

A Python toolbox for large-scale **Ca**lcium **Im**aging **An**alysis.    

CaImAn implements a set of essential methods required to analyze calcium and voltage imaging data. It provides fast and scalable algorithms for motion correction, source extraction, spike deconvolution, and registering neurons across multiple days. It is suitable for both two-photon and one-photon fluorescence microscopy data, and can be run in both offline and real-time online modes. Online documentation can be found [here](https://caiman.readthedocs.io/en/latest/). 

# Quick start :rocket:
To get started quickly working on a demo notebook, just take the following three steps:

### Step 1: Install caiman
The following is all done in your anaconda prompt, starting in your base environment:
  
    conda install mamba -n base -c conda-forge   # install mamba in base environment
    mamba create -n caiman -c conda-forge caiman # install caiman
    conda activate caiman  # activate virtual environment

### Step 2: Download code samples and data sets
Here we will create a working directory with code samples and related data in a folder called `caiman_data`. Once this step is done, you will easily be able to run any of the demos provided with Caiman. While still in the caiman virtual environment created in the previous step:  

    caimanmanager.py install

### Step 3: Try out a demo notebook
In Step 2, you created a `caiman_data` folder in your home directory. Here, we `cd` to the demo notebooks folder in `caiman_data`, and fire up a jupyter notebook:

    cd home/caiman_data/demos/notebooks  # go to demo notebooks
    jupyter notebook  # fire up a jupyter notebook

Jupyter will open: click on `demo_pipeline.ipynb` to get started with a demo! Note that what counts as `home` in the first command depends on your OS, so be sure to fill in your actual home directory.

## For more details
 There are much more detailed installation and setup instructions [here](./docs/source/Installation.rst). For instance go there if you want to change where `caiman_data` is placed.

## Problem setting up Caiman?
We want Caiman to install easily on Linux, Mac, and Windows. If you run into problems, please send us a [message on Gitter](https://app.gitter.im/#/room/#agiovann_Constrained_NMF:gitter.im).  


# Demo notebooks :page_with_curl:
Caiman provides demo notebooks to explain and demonstrate each of our main pipelines, and you can adapt these for your personal needs. You can find the main use cases and notebooks listed in the following table.

| Use case | Demo notebook | Paper |
|:-------- |:------------- | --------------------- |
| Motion correction | demo_motion_correction.ipynb | [Pnevmatikakis et al., 2017](https://pubmed.ncbi.nlm.nih.gov/28782629/) | 
| CNMF for 2p or low-noise 1p data | demo_pipeline.ipynb |  [Pnevmatikakis et al., 2016](https://pubmed.ncbi.nlm.nih.gov/26774160/) | 
| CNMFE for 1p data  | demo_pipeline.ipynb |  [Zhou et al., 2018](https://pubmed.ncbi.nlm.nih.gov/29469809/) | 
| Volpy for voltage data | demo_pipeline_voltag_imaging.ipynb |  [Cai et al., 2021](https://pubmed.ncbi.nlm.nih.gov/33852574/) | 
| Volumetric (3D) CNMF | demo_caiman_cnmf_3D.ipynb | [Mentioned in Giovannucci et al., 2019](https://pubmed.ncbi.nlm.nih.gov/30652683/) | 
| CNMF for dendrites | data_dendritic.ipynb | Developed by Eftychios Pnevmatikakis | 
| Online CNMF (OnACID) | demo_OnACID_mesoscope.ipynb |[Giovannucci et al., 2017](https://proceedings.neurips.cc/paper_files/paper/2017/hash/4edaa105d5f53590338791951e38c3ad-Abstract.html) | 
| Online volumetric CNMF | demo_online_3D.ipynb | Developed by Johannes Friedrich | 
| Online CNMFE (OnACID-E) | demo_realtime_cnmfE.ipynb |[Friedrich et al. 2020](https://pubmed.ncbi.nlm.nih.gov/33507937/) | 
| Register cells across sessions | demo_multisession_registration.ipynb | [Pnevmatikakis et al., 2016](https://pubmed.ncbi.nlm.nih.gov/26774160/) | 

A comprehensive list of references, where you can find detailed discussion of the methods and their development, can be found [here](https://caiman.readthedocs.io/en/master/CaImAn_features_and_references.html#references). 


# For questions, comments, and help :question:
If you need help using Caiman, or hit an installation snag, we recommend asking about it in the [Gitter forum](https://app.gitter.im/#/room/#agiovann_Constrained_NMF:gitter.im). If you have found a bug, we recommend searching the [issues at github](https://github.com/flatironinstitute/CaImAn/issues) and opening a new one if you can't find it there. If you have a general question about how things work or there is a feature you would like to see, feel free to come chat at Gitter or open an issue at Github. 

# How to contribute :hammer:
 Caiman is an open-source project and improves because of contributions from users all over the world. If there is something about Caiman that you would like to improve, then you are qualified to contribute! We are always looking for more contributors, so please come read the [contributors page](./CONTRIBUTING.md) for more details about how. 

# Videos :tv:
These talks by Andrea Giovannucci from past Caiman workshops/events are an excellent start for newcomers. They go through NoRMCorre, CNMF(E) and VolPy.

The following Open Neuroscience talk provides a good high-level introduction to Caiman:    
https://www.youtube.com/watch?v=5APzPRbzUIA

The following Nemonic workshops are more in depth:

* https://www.youtube.com/watch?v=KjHrjhvhRy0
* https://www.youtube.com/watch?v=rUwIqU6gVvw
* https://www.youtube.com/watch?v=NZZ6_zo0YIM
* https://www.youtube.com/watch?v=z6TlH28MLRo


# Related repositories :pushpin:
There are many repositories that make heavy use of Caiman, or will help make using Caiman easier. If you would like your software to be in this list, please ask at Gitter or open an issue. 

* [use\_cases repo](https://github.com/flatironinstitute/caiman_use_cases):  additional code (unmaintained) demonstrating how to reproduce results in some Caiman-related papers, and how to use/extend Caiman.
* [jnormcorre](https://github.com/apasarkar/jnormcorre): [JAX](https://github.com/google/jax) implementation of NoRMCorre for motion correction using JAX acceleration
* [funimag](https://github.com/paninski-lab/funimag): matrix decomposition for denoising and compression
* [mesmerize-core](https://github.com/nel-lab/mesmerize-core): parameter optimization, data organization and visualizations with Caiman
* [improv](https://github.com/project-improv/improv):  a platform for creating online analysis workflows that lets you use Caiman in real time (e.g., for all-optical experiments)

If you have questions about these related packages please reach out to their maintainers directly. 

# Caiman paper
If you use Caiman and end up publishing, we kindly ask that you [cite Giovannucci et al., 2019](https://elifesciences.org/articles/38173):
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

# Main developers
* (emeritus) Eftychios A. Pnevmatikakis, **Flatiron Institute, Simons Foundation** 
* (emeritus) Andrea Giovannucci, **University of North Carolina, Chapel Hill**
* Johannes Friedrich, **Allen Institute,Seattle Washington**
* Changlia Cai, **University of North Carolina, Chapel Hill**
* Pat Gunn, **Flatiron Institute, Simons Foundation**
* Eric Thomson, **Flatiron Institute, Simons Foundation**

A complete list of contributors can be found [here](https://github.com/flatironinstitute/Caiman/graphs/contributors). Currently Pat Gunn, Johannes Friedrich, and Eric Thomson are the most active contributors.


# Acknowledgements :clap:
Special thanks to the following people for letting us use their datasets in demo files:

* Weijian Yang, Darcy Peterka, Rafael Yuste, Columbia University
* Sue Ann Koay, David Tank, Princeton University
* Manolis Froudarakis, Jake Reimers, Andreas Tolias, Baylor College of Medicine
* Clay Lacefield, Randy Bruno, Columbia University
* Daniel Aharoni, Peyman Golshani, UCLA

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
