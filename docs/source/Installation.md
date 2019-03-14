## Installation

Download and install Anaconda or Miniconda (Python 3.x version) <http://docs.continuum.io/anaconda/install>

<details>
  <summary> Installation on Windows </summary>
  
  The Windows installation process differs more widely from installation on Linux or MacOSX and has different issues you may run into.

### Process
   * Increase the maximum size of your pagefile to 64G or more (http://www.tomshardware.com/faq/id-2864547/manage-virtual-memory-pagefile-windows.html ) - The Windows memmap interface is sensitive to the maximum setting and leaving it at the default can cause errors when processing larger datasets
   * Download and install Anaconda (Python 3.x) <http://docs.continuum.io/anaconda/install>. Allow the installer to modify your PATH variable
   * Use Conda to install git (With "conda install git") - use of another commandline git is acceptable, but may lead to issues depending on default settings
   * Install Microsoft Build Tools for Visual Studio 2017 <https://www.visualstudio.com/downloads/#build-tools-for-visual-studio-2017>. Check the "Build Tools" box, and in the detailed view on the right check the "C/C++ CLI Tools" component too. The specifics of this occasionally change as Microsoft changes its products and website; you may need to go off-script.

Use the following menu item to launch a anaconda-enabled command prompt: start>programs>anaconda3>anaconda prompt
From that prompt. issue the following commands (if you wish to use the dev branch, you may switch branches after the clone):

   ```bash
   git clone https://github.com/flatironinstitute/CaImAn
   cd CaImAn
   conda env create -f environment.yml -n caiman
   conda install -n caiman vs2017_win-64
   ```

At this point you will want to remove a startup script that visual studio made for your conda environment that can cause conda to crash while entering the caiman environment. Use the Windows find-file utility (under the Start Menu) to look for vs2015_compiler_vars.bat and/or vs2015_compiler_vars.bat under your home directory. At least one copy should show up. Delete the version that has conda\envs\caiman as part of its location. You may also want to do a search for keras_activate.bat under your home directory, find the one in conda\envs\caiman, and edit it so KERAS_BACKEND is set to tensorflow rather than theano. You may then continue the installation.

   ```bash
   activate caiman
   pip install . (OR pip install -e . if you want to develop code)
   copy caimanmanager.py ..
   conda install numba
   cd ..
   ```

### Setting up a data directory with caimanmanager

Now that you have stepped out of the caiman source directory, you are ready to make a data directory with code samples and datasets. You will not use the source tree directory any more. 

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

### Setting up environment variables 

To make the package work *efficiently* and eliminate "crosstalk" between different processes, run these commands before launching Python:

   ```bash
   set MKL_NUM_THREADS=1
   set OPENBLAS_NUM_THREADS=1
   set KERAS_BACKEND=tensorflow
   ```   

 The commands should be run every time you enter the caiman conda environment. We recommend you save these values inside your environment so you do not have to repeat this process every time. You can do this by following the instructions [here](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#saving-environment-variables).

</details>

<details>
  <summary> Installation on MacOS and Linux </summary>
       
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
    
  If you recently upgraded to OSX Mojave you may need to perform the following steps before your first install:
    
   ```
   xcode-select --install
   open /Library/Developer/CommandLineTools/Packages/
   ```
  and install the package file you will find in the folder that pops up

  ### Setting up environment variables 

  To make the package work *efficiently* and eliminate "crosstalk" between different processes, run these commands before launching Python (this is for Linux and OSX):

   ```bash
   export MKL_NUM_THREADS=1
   export OPENBLAS_NUM_THREADS=1
   export KERAS_BACKEND=tensorflow
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
</details>

## Upgrading

If you already have CaImAn installed with the pip installer (May 2018 or later), but want to upgrade, please follow the procedure below. If you reinstall CaImAn frequently, you can try skip deleting and recreating your Conda environment. In this case you can do only steps 1, 5, and 7 below to update the code. However, if the environment file has changed since your last update this may lead to you not the latest version.

From the conda environment you used to install CaImAn:
1. `pip uninstall caiman`
2. remove or rename your ~/caiman_data directory
3. Remove your conda environment: `conda env remove -n NAME_OF_YOUR_ENVIRONMENT`
4. Close and reopen your shell (to clear out the old conda environment)
5. Do a `git pull` from inside your CaImAn folder.
6. Recreate and reenter your conda environment as you did in the [README](https://github.com/flatironinstitute/CaImAn)
7. Do a `pip install .` inside that code checkout
8. Run `caimanmanager.py install` to reinstall the data directory (use `--inplace` if you used the `pip install -e .` during your initial installation).

- If you used the `pip install -e .` option when installing, then you can try updating by simply doing a `git pull`. Again, this might not lead to the latest version of the code if the environment variables have changed.

- The same applies if you want to modify some internal function of CaImAn. If you used the `pip install -e .` option then you can directly modify it (that's why it's called developer mode). If you used the `pip install .` option then you will need to `pip uninstall caiman` followed by `pip install .` for your changes to take effect. Depending on the functions you're changing so you might be able to skip this step.

## Installing additional packages

CaImAn uses the conda-forge conda channel for installing its required packages. If you want to install new packages into your conda environment for CaImAn, it is important that you not mix conda-forge and the defaults channel; we recommend only using conda-forge. To ensure you're not mixing channels, perform the install (inside your environment) as follows:
   ```bash
   conda install -c conda-forge --override-channels NEW_PACKAGE_NAME
   ```
You will notice that any packages installed this way will mention, in their listing, that they're from conda-forge, with none of them having a blank origin. If you fail to do this, differences between how packages are built in conda-forge versus the default conda channels may mean that some packages (e.g. OpenCV) stop working despite showing as installed.
