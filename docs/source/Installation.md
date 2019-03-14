## Installation

Download and install Anaconda or Miniconda (Python 3.x version) <http://docs.continuum.io/anaconda/install>

<details>
  <summary> Installation on Windows </summary>
  blah blah
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
