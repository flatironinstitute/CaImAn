What's new?
===========

Online analysis for microendoscopic 1p data (January 2020)
--------------------------------------------------------

We developed two approaches for analyzing streaming microendoscopic 1p data with high speed and low memory requirements.
The first approach (OnACID-E) features a direct implementation of the CNMF-E algorithm in an online setup.
An example can be shown in the notebook `demos/notebooks/demo_online_CNMFE.ipynb`. The second approach first extracts
the background by modeling it with a simple convolutional neural network (Ring-CNN) and proceeds with the analysis using the OnACID algorithm.

Analysis pipeline for Voltage Imaging data (December 2019)
----------------------------------------------------------

We recently added VolPy an analysis pipeline for voltage imaging data. The analysis is based on following objects:

- `MotionCorrect`: An object for motion correction which can be used for both rigid and piece-wise rigid motion correction.
- `volparams`: An object for setting parameters of voltage imaging. It can be set and changed easily and is passed into the algorithms.
- `VOLPY`: An object for running the spike detection algorithm and saving results.

To see examples of how these methods are used, please consult the `demo_pipeline_voltage_imaging.py` script in the `demos/general` folder.
 For more information about the approach check the `preprint <https://www.biorxiv.org/content/10.1101/2020.01.02.892323v1>`_.

Installation through conda-forge (August 2019)
-----------------------------------------------

Beginning in August 2019 we have an experimental binary release of the software in the conda-forge package repos.
This is intended for people who can use CaImAn as a library, interacting with it as the demos do. It also does not need a compiler.
It is not suitable for people intending to change the CaImAn codebase. Comfort with conda is still required. If you wish to use the binary package,
you do not need the sources (including this repo) at all. Installation and updating instructions can be found `here <Installation.rst>`_.

You will still need to use caimanmanager.py afterwards to create a data directory. If you install this way, do not follow any of the other install instructions below.

Exporting results, GUI and NWB support (July 2019)
---------------------------------------------------

You can now use the `save` method included in both the `CNMF` and `OnACID` objects to export the results (and parameters used) of your analysis. The results are saved in an HDF5 file that you can then load in a graphical user interface for more inspection. The GUI will allow you to inspect the results and modify the selected components based on the various quality metrics. For more information click [here](docs/GUI.md)

The `Neurodata Without Borders (NWB) <https://www.nwb.org/>`_ file format is now supported by CaImAn.
You read and analyze NWB files and can save the results of the analysis (`Estimates` object) back to the original NWB file.
Consult the script `use_cases/NWB/demo_pipeline_NWB.py` for an example on how to use this feature.

**To use CaImAn with these additional features you'll need to create a new environment following the usual instructions.**
