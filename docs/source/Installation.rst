Installation and Updating
=========================

This page will give more detailed setup instructions for Caiman than the main readme in the source tree.
This is the place you should visit first if you run into problems and need to troubleshoot. It includes 
info on initial setup, as well as updating with new releases. There is a Table of Contents on the 
left-hand side of this page, so if you need help with a particular task, that should help you get oriented. 

If the information on this page doesn't help, please ask us for 
help at `Gitter <https://app.gitter.im/#/room/#agiovann_Constrained_NMF:gitter.im/>`_, 
or open an `issue <https://github.com/flatironinstitute/CaImAn/issues/>`_.

Caiman setup consists of two main steps:

1. Install Caiman
2. Use caimanmanager to download sample code and data to a ``caiman_data`` directory.

We will discuss each of these steps for different operating systems below. In a separate section, we will discuss how to 
upgrade once you've already installed. 

If you do not already have conda installed, first install a 3.x version for your platform `here <https://docs.conda.io/en/latest/miniconda.html>`_. 
When installing, allow conda to modify your PATH variable. If you are using an M1-based Mac, please ignore the ARM builds of conda; install an x86 version instead (ignore any warnings you get while doing so; 
it will work fine).

We recommend you familiarise yourself a **little** bit with Conda before going further,
if you are not already (https://docs.conda.io/projects/conda/en/latest/user-guide/index.html). 

Section 1: Install Caiman
-------------------------
There are two main ways to install Caiman: you can install using **conda** (package-based install), or 
you can install in **development-mode**. The first will be appropriate for almost all users -- those that
want to get started quickly using Caiman to analyze data. The second, development-mode installation, 
is for those who want to install an editable version of Caiman in order to tweak to the code base, 
and make contributions to Caiman. 


Section 1A. Install with conda
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
These are basically the Quickstart instructions on the main readme page at the repo: <https://github.com/flatironinstitute/CaImAn>, so if you follow those, you 
should be good to go. 

.. raw:: html

   <details>
   <summary>Details for conda install</summary>

This process is the same on every operating system, and is what most users will need who just want to use Caiman to 
get things running quickly. You will install mamba into your base environment, create a new environment with the 
caiman package from conda-forge, and then activate the new environment.

.. code:: bash

    conda install -n base -c conda-forge mamba
    mamba create -n caiman -c conda-forge caiman
    conda activate caiman

Note if you are installing on Windows, run the above commands in the anaconda prompt rather than powershell or the dos shell:

**Known issues**

If you are on Windows, have used Caiman before using our github repo and now want to use the conda-forge package,
you might encounter some errors with Python reading the files from the wrong directory. In this case, rename
(or remove) the caiman directory that contains the source of the repo and the ``caiman_data`` folder and then proceed
with setting up the ``caiman_data`` folder as explained below in Section 2.

.. raw:: html

   </details>


Section 1B. Development-mode install
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Dev mode install allows you to modify the source files of Caiman and makes it easier
to contribute to the project, fix bugs etc.

If you install in dev mode you will likely need to set some environment variables manually (it is 
done automatically when you do the conda install): this is discussed in Section 4C.

Dev-mode install on Windows
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. raw:: html

   <details>
   <summary>Dev Mode Installation on Windows</summary>

The Windows installation process differs quite a bit from installation
on Linux or MacOSX. If you can work on Linux, it will be easier. Everything 
below should be from a Conda-prompt rather than from Powershell or any other shell.

-  Install Microsoft Build Tools for Visual Studio 2019, which you can download 
   from (https://visualstudio.microsoft.com/vs/older-downloads/). Check the 
   “Build Tools” box, and in the detailed view on the right check the “C/C++ CLI 
   Tools” component too. The specifics of this occasionally change as Microsoft 
   changes its products and website; you may need to go off-script.
-  Remove any associations you may have made between .py files and an existing python
   interpreter or editor
-  If you have less than 64GB of RAM, increase the maximum size of your pagefile to 64G or more
   (http://www.tomshardware.com/faq/id-2864547/manage-virtual-memory-pagefile-windows.html).
   The Windows memmap interface is sensitive to the maximum setting
   and leaving it at the default can cause errors when processing larger
   datasets

At the conda prompt:

.. code:: bash

     git clone git@github.com:flatironinstitute/CaImAn.git
     cd CaImAn
     mamba env create -f environment.yml -n caiman
     conda activate caiman 
     mamba install -n caiman vs2019_win-64
     pip install -e . 

A couple of things to note:

-  If you don't want to develop code then replace the second-to-last command with
   ``pip install .`` 
-  If any of these steps gives you errors do not proceed to the following step without resolving it.
-  If the environment doesn't active properly, there may be ``bat`` files that 
   need to be removed. Use the Windows find-file utility
   (under the Start Menu) to look for ``vs2019_compiler_vars.bat`` under 
   your home directory. If a copy shows up, delete the version that has
   your ``caiman`` environment name as part of its location.
   You may then continue the installation.

.. raw:: html

   </details>

Dev Mode Install on MacOS and Linux
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. raw:: html

   <details>
   <summary>Installation on MacOS and Linux</summary>

.. code:: bash

     git clone git@github.com:flatironinstitute/CaImAn.git
     cd CaImAn/
     mamba env create -f environment.yml -n caiman
     source activate caiman
     pip install -e .

If you don't want to develop code then replace the last command with
``pip install .`` If any of these steps gives you errors do not
proceed to the following step without resolving it

**Known issues**

If you recently upgraded to OSX Mojave you may need to perform the
following steps before your first install:

.. code:: bash

     xcode-select --install
     open /Library/Developer/CommandLineTools/Packages/

and install the package file you will find in the folder that pops up

.. raw:: html

   </details>



Section 2: Set up demos with caimanmanager
------------------------------------------

Once Caiman is installed, you will likely want to set up a working directory with code samples and datasets. 
The installation step in Section 1 produced a command ``caimanmanager`` that handles this. caimanmanager will
place demos and data in a ``caiman_data`` folder in your home directory. Install using:

``caimanmanager install``

if you used the conda-forge package or the ``pip install .`` option.

If you installed using the developer-mode option (installing with ``pip install -e .``) then run caimanmanager with:

``caimanmanager install --inplace`` 

If you prefer to manage this information somewhere other than your home directory, the
``CAIMAN_DATA`` environment variable can be set to customise it. The caimanmanager tool 
and other libraries will respect that.


Section 3: Upgrading
--------------------

Upgrading can mean a couple of things. First, it typically means there has been a new release of Caiman, so you need 
to install the new version of Caiman. Second, it could mean you need to upgrade changes to the demos in ``caiman_data`` 
using ``caimanmanager``. We'll discuss both options.


Section 3A: Upgrade conda install
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details>
   <summary>Updating the conda-forge package</summary>

From within your caiman environment type ```conda update caiman -c conda-forge```. In most cases this should be enough.

If not, you may want to create a new environmrent from scratch. 

1. Remove your conda environment: ``conda env remove -n caiman`` (or whatever you called the conda environment you used)

2. Remove or rename your ~/caiman_data directory

3. Repeat the install instructions from above.

.. raw:: html

   </details>


Section 3B: Upgrade the dev-mode install
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details>
   <summary>Updating in development mode</summary>

If you already have Caiman installed with the pip installer (development mode),
but want to upgrade, please follow the procedure below. If you
reinstall Caiman frequently, you can try skip deleting and recreating
your Conda environment. In this case you can do only steps 1, 5, and 7
below to update the code. However, if the environment file has changed
since your last update this may lead to you not the latest version. None of this applies
to the conda-forge route (for which instructions are given above).

From the conda environment you used to install Caiman:

1. ``pip uninstall caiman``

2. Remove your conda environment: ``conda env remove -n caiman`` (or whatever you called the conda environment you used)

3. Close and reopen your shell (to clear out the old conda environment)

4. Do a ``git pull`` from inside your CaImAn folder.

5. Recreate and reenter your conda environment as you did in the installation instructions

6. Do a ``pip install .`` inside that code checkout

7. Run ``caimanmanager install`` to reinstall the data directory (use ``--inplace`` if you used the ``pip install -e .`` during your initial installation).

-  If you used the ``pip install -e .`` option when installing, then you
   can try updating by simply doing a ``git pull``. Again, this might
   not lead to the latest version of the code if the environment
   variables have changed.

-  The same applies if you want to modify some internal function of
   Caiman. If you used the ``pip install -e .`` option then you can
   directly modify it (that's why it's the editable developer mode). If you
   used the ``pip install .`` option then you will need to
   ``pip uninstall caiman`` followed by ``pip install .`` for your
   changes to take effect. Depending on the functions you're changing so
   you might be able to skip this step.

.. raw:: html

   </details>


Section 3C: Upgrade the demos with caimanmanager
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details>
   <summary>Upgrade the demos</summary>

When you upgrade Caiman, sometimes the underlying APIs change. When this happens and it impacts a demo (or otherwise 
requires changes to files in ``caiman_data``), we update the demo and data. This means that upgrading Caiman works 
best if you also replace the ``caiman_data`` directory with a new version.

To check if the demos or datafiles have changed since your last install, you can run ``caimanmanager check``. If they have not,
you may keep using them. If they have, we recommend moving your old caiman data directory out of the way (or just remove them if you have no
precious data in ``caiman_data``) and updating ``caiman_data`` as described below.

However, you may also have made your own changes to the demos (e.g. to work with your data). If you have done this, 
you may need to massage your changes into the new versions of the demos. For this reason, we recommend that if 
you modify the demos to operate on your own data to save them as a different file to avoid losing your work 
when updating the caiman_data directory.

To update ``caiman_data`` you can follow the following procedure:

- If there are no new demos or files in the new Caiman distribution, then you can leave it as is.

- If you have not modified anything in ``caiman_data`` but there have been changes in the new Caiman release, 
  then remove ``caiman_data`` directory before upgrading and have ``caimanmanager`` make a new one after the upgrade, by 
  running caimanmanager as discussed in Section 2.

- If you have extensively modified things in ``caiman_data``, rename your ``caiman_data`` directory, and have ``caimanmanager`` 
  make a new one after the upgrade, and then massage your changes back in. E.g., if you have extensively 
  modified ``demo_pipeline.ipynb`` for personal use, then change the name of this notebook before folding it back into ``caiman_data``.

.. raw:: html

   </details>

Section 4: Miscellaneous
-------------------------

Section 4A: System Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

32G RAM is required for a good experience, and depending on datasets, more may be necessary. Caiman is optimized for use by 
multiple CPUs, so workstations or clusters with multiple CPU cores are ideal (8+ logical cores). GPU computation is not used 
heavily by Caiman (though see Section 4D). 

Right now, Caiman works and is supported on the following platforms:

- Linux on 64-bit x86 CPUs
- MacOS on 64-bit x86 CPUs or ARM CPUs
- Windows on 64-bit x86 CPUs

Support for Linux on ARM (e.g. AWS Graviton) is not available (but it may work with the port of conda, 
if you compile Caiman yourself - we do not have binary packages and this is untested). If you care about this,
please let us know.


Section 4B: Installing additional packages
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Caiman installs through the conda-forge conda channel. Some packages are available on multiple conda channels, and in this 
case it is important that you use the conda-forge channel if possible. To do this, when installing new packages 
inside your environment, use the following command:

::

   mamba install -c conda-forge --override-channels NEW_PACKAGE_NAME

You will notice that any packages installed this way will mention, in their listing, 
that they are from conda-forge, with none of them having a blank origin. If you don't do this, 
differences between how packages are built in different channels could lead to some packages failing to work
(e.g., OpenCV). 

Section 4C: Setting up environment variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is only important for people who are doing the dev-mode install. If you 
installed using the conda packaged-based install, this is done automatically.

To make the package work *efficiently* and eliminate "crosstalk" between
different processes, some multithreading operations need to be turned off
This is for Linux and Windows and is not necessary in OSX. 

For **Linux** run these commands before launching Python:

.. code:: bash

     export MKL_NUM_THREADS=1
     export OPENBLAS_NUM_THREADS=1
     export VECLIB_MAXIMUM_THREADS=1

For **Windows** run the same commands, replacing the word ```export``` with the word ```set```.

The commands should be run *every time* before launching python. It is
recommended that you save these values inside your environment so you
don’t have to repeat this process every time. You can do this by
following the instructions
`here <https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#saving-environment-variables>`__.


Section 4D: Other topics
~~~~~~~~~~~~~~~~~~~~~~~~~
See also:

* :doc:`Our clustering doc <cluster>`
* :doc:`Caiman and GPUs <readme-gpu>`

