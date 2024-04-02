<a href="https://colab.research.google.com/drive/1vkp-uPV8tKavmX12bcN2L-jYH8_MgmHL?usp=sharing"><img src="https://img.shields.io/badge/-Colab%20Demo-blue" /></a>


CaImAn
======
<img src="https://github.com/flatironinstitute/CaImAn/blob/main/docs/LOGOS/Caiman_logo_2.png" width="400" align="right">

A Python toolbox for large-scale **Ca**lcium **Im**aging **An**alysis.    

CaImAn implements a set of essential methods required to analyze calcium and voltage imaging data. It provides fast and scalable algorithms for motion correction, source extraction, spike deconvolution, and registering neurons across multiple sessions. It is suitable for both two-photon and one-photon fluorescence microscopy data, and can be run in both offline and online modes. Documentation is [here](https://caiman.readthedocs.io/en/latest/). 

Caiman Central
--------------
- [Caiman Central](https://github.com/flatironinstitute/caiman_central) is the hub for sharing information about CaImAn. Information on quarterly community meetings, workshops, other events, and any other communications between the developers and the user community can be found there.

# Quick start :rocket:
Follow these three steps to get started quickly, from installation to working through a demo notebook. If you do not already have conda installed, [you can find it here](https://docs.conda.io/en/latest/miniconda.html). There is a video walkthrough of the following steps [here](https://youtu.be/b63zAmKihIY?si=m7WleTwdU0rJup_2).

### Step 1: Install caiman
The following is all done in your anaconda prompt, starting in your base environment:
  
    conda install -n base -c conda-forge mamba   # install mamba in base environment
    mamba create -n caiman -c conda-forge caiman # install caiman
    conda activate caiman  # activate virtual environment

### Step 2: Download code samples and data sets
Create a working directory called `caiman_data` that includes code samples and related data. Run the following command from the same virtual environment that you created in Step 1:  

    caimanmanager install

### Step 3: Try out a demo notebook
Go into the working directory you created in Step 2, and open a Jupyter notebook:

    cd <your home>/caiman_data/
    jupyter notebook 

Jupyter will open. Navigate to demos/notebooks/ and click on `demo_pipeline.ipynb` to get started with a demo.

> Note that what counts as `<your home>` in the first line depends on your OS/computer. Be sure to fill in your actual home directory. On Linux/Mac it is `~` while on Windows it will be something like `C:\Users\your_user_name\` 

## For installation help
Caiman should install easily on Linux, Mac, and Windows. If you run into problems, we have a dedicated [installation page](./docs/source/Installation.rst): the details there should help you troubleshoot. If you don't find what you need there, *please* [create an issue](https://github.com/flatironinstitute/CaImAn/issues) at GitHub, and we will help you get it sorted out. 

# Demo notebooks :page_with_curl:
Caiman provides demo notebooks to showcase each of our main features, from motion correction to online CNMF. We recommend starting with the CNMF notebook (`demo_pipeline.ipynb`), which contains more explanation and details than the other notebooks: it covers many concepts that will be used without explanation in the other notebooks. The CNMFE notebook (`demo_pipeline_cnmfE.ipynb`), is also more detailed. Once you've gotten things set up and worked through those "anchor" notebooks, the best way to get started is to work through the demo notebook that most closely matches your use case; you should be able to adapt it for your particular needs.

The main use cases and notebooks are listed in the following table:

| Use case | Demo notebook | Paper |
|:-------- |:------------- | --------------------- |
| CNMF for 2p or low-noise 1p data | [demo_pipeline.ipynb](./demos/notebooks/demo_pipeline.ipynb) |  [Pnevmatikakis et al., 2016](https://pubmed.ncbi.nlm.nih.gov/26774160/) | 
| CNMFE for 1p data  | [demo_pipeline_cnmfE.ipynb](./demos/notebooks/demo_pipeline_cnmfE.ipynb) |  [Zhou et al., 2018](https://pubmed.ncbi.nlm.nih.gov/29469809/) | 
| Volpy for voltage data | [demo_pipeline_voltage_imaging.ipynb](./demos/notebooks/demo_pipeline_voltage_imaging.ipynb) |  [Cai et al., 2021](https://pubmed.ncbi.nlm.nih.gov/33852574/) | 
| Volumetric (3D) CNMF | [demo_caiman_cnmf_3D.ipynb](./demos/notebooks/demo_caiman_cnmf_3D.ipynb) | [Mentioned in Giovannucci et al., 2019](https://pubmed.ncbi.nlm.nih.gov/30652683/) | 
| CNMF for dendrites | [demo_dendritic.ipynb](./demos/notebooks/demo_dendritic.ipynb) |  [Pnevmatikakis et al., 2016](https://pubmed.ncbi.nlm.nih.gov/26774160/) | 
| Online CNMF (OnACID) | [demo_OnACID_mesoscope.ipynb](./demos/notebooks/demo_OnACID_mesoscope.ipynb) |[Giovannucci et al., 2017](https://proceedings.neurips.cc/paper_files/paper/2017/hash/4edaa105d5f53590338791951e38c3ad-Abstract.html) | 
| Online volumetric CNMF | [demo_online_3D.ipynb](./demos/notebooks/demo_online_3D.ipynb) | Developed by Johannes Friedrich | 
| Online CNMFE (OnACID-E) | [demo_realtime_cnmfE.ipynb](./demos/notebooks/demo_realtime_cnmfE.ipynb) |[Friedrich et al. 2020](https://pubmed.ncbi.nlm.nih.gov/33507937/) | 
| Motion correction | [demo_motion_correction.ipynb](./demos/notebooks/demo_motion_correction.ipynb) | [Pnevmatikakis et al., 2017](https://pubmed.ncbi.nlm.nih.gov/28782629/) | 
| Seed CNMF with external masks | [demo_seeded_CNMF.ipynb](./demos/notebooks/demo_seeded_CNMF.ipynb) |  [Mentioned in Giovannucci et al., 2019](https://pubmed.ncbi.nlm.nih.gov/30652683/) | 
| Register cells across sessions | [demo_multisession_registration.ipynb](./demos/notebooks/demo_multisession_registration.ipynb) | [Pnevmatikakis et al., 2016](https://pubmed.ncbi.nlm.nih.gov/26774160/) | 

A comprehensive list of references, where you can find detailed discussion of the methods and their development, can be found [here](https://caiman.readthedocs.io/en/master/CaImAn_features_and_references.html#references). 


# How to get help
- [Online documentation](https://caiman.readthedocs.io/en/latest/) contains a lot of general information about Caiman, the parameters, how to interpret its outputs, and more.
- [GitHub Discussions](https://github.com/flatironinstitute/CaImAn/discussions) is our preferred venue for users to ask for help.
- The [Gitter forum](https://app.gitter.im/#/room/#agiovann_Constrained_NMF:gitter.im) is our old forum: we sometimes will ask people to join us there when something can best be solved in real time (e.g., installation problems).
- If you have found a bug, we recommend searching the [issues at github](https://github.com/flatironinstitute/CaImAn/issues) and opening a new issue if you can't find the solution there. 
- If there is a feature you would like to see implemented, feel free to come chat at the above forums or open an issue at Github.

# How to contribute :hammer:
 Caiman is an open-source project and improves because of contributions from users all over the world. If there is something about Caiman that you would like to work on, then please reach out. We are always looking for more contributors, so please come read the [contributors page](./CONTRIBUTING.md) for more details about how. 

# Videos 
There are multiple online videos by Andrea Giovannucci from past Caiman workshops/events that are an excellent start for newcomers.

The following talk provides a good high-level introduction to Caiman:    
https://www.youtube.com/watch?v=5APzPRbzUIA

The following talks are more in depth:

* https://www.youtube.com/watch?v=KjHrjhvhRy0
* https://www.youtube.com/watch?v=rUwIqU6gVvw
* https://www.youtube.com/watch?v=NZZ6_zo0YIM
* https://www.youtube.com/watch?v=z6TlH28MLRo


# Related repositories :pushpin:
There are many repositories that use Caiman, or help make using Caiman easier.

* [use\_cases repo](https://github.com/flatironinstitute/caiman_use_cases):  additional code (unmaintained) demonstrating how to reproduce results in some Caiman-related papers, and how to use/extend Caiman.
* [jnormcorre](https://github.com/apasarkar/jnormcorre): [JAX](https://github.com/google/jax) implementation of NoRMCorre for motion correction using JAX acceleration
* [funimag](https://github.com/paninski-lab/funimag): matrix decomposition for denoising and compression
* [mesmerize-core](https://github.com/nel-lab/mesmerize-core): parameter optimization, data organization and visualizations with Caiman
* [improv](https://github.com/project-improv/improv):  a platform for creating online analysis workflows that lets you use Caiman in real time (e.g., for all-optical experiments)

If you have questions about these related packages please reach out to their maintainers directly. If you would like your software to be in this list, please contact one of the developers or open an issue.

# Citing Caiman and related papers
If you publish a paper that relied on Caiman, we kindly ask that you [cite Giovannucci et al., 2019](https://elifesciences.org/articles/38173):
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
If possible, we'd also ask that you cite the papers where the original algorithms you use (such as CNMF) were developed. A list of such references can be found [here](https://caiman.readthedocs.io/en/master/CaImAn_features_and_references.html#references). 


# Main developers
* (emeritus) Eftychios A. Pnevmatikakis, **Flatiron Institute, Simons Foundation** 
* (emeritus) Andrea Giovannucci, **University of North Carolina, Chapel Hill**
* Johannes Friedrich, **Allen Institute, Seattle Washington**
* Changjia Cai, **University of North Carolina, Chapel Hill**
* Pat Gunn, **Flatiron Institute, Simons Foundation**
* Eric Thomson, **Flatiron Institute, Simons Foundation**

A complete list of contributors can be found [here](https://github.com/flatironinstitute/Caiman/graphs/contributors). Currently Pat Gunn, Johannes Friedrich, and Eric Thomson are the most active contributors.


# Acknowledgements 
Special thanks to the following people for letting us use their datasets in demo files:

* Weijian Yang, Darcy Peterka, Rafael Yuste, Columbia University
* Bernardo Sabatini, Harvard University
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
