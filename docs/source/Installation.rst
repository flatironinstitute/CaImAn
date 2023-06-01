Installation and Updating
=========================

This page will give more detailed setup instructions for Caiman than the `main readme <../../README.md>`_. This is the place 
you should come first if you run into problems and need to troubleshoot. It includes info on initial setup, as well as 
updating with new releases. Note there is a Table of Contents on the left-hand side of this page: there are a lot of 
sections here, so if you get lost or need help with a particular task, that should help you get oriented. 

Caiman setup consists of two main steps:

1. Install Caiman
2. Use caimanmanager to download sample code and data to a ``caiman_data`` directory.

We will discuss each of these steps for different operating systems below. In a separate section, we will discuss how to 
upgrade once you've already installed. 

If you do not already have conda installed, first install a 3.x version for your platform `here <https://docs.conda.io/en/latest/miniconda.html>`_. 
We recommend you familiarise yourself with Conda before going further. If you are using an M1-based Mac, please ignore the 
ARM builds of conda; install an x86 version instead (ignore any warnings you get while doing so; it will work fine).


Section 1: Install Caiman
-------------------------

There are two main ways to install Caiman: you can use conda (package-based install), and a **development-mode** 
installation. The first will be appropriate for first-time users that want to get started quickly using 
Caiman to analyze data. The second, development-mode installation, is for those who want to install an 
editable version of Caiman in order to tweak to the code base, and make contributions to Caiman.


Section 1A. Pre-built conda install
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
These are basically the Quickstart instructions on the `main readme page <../../README.md>`_, so if you follow those, you 
should be good to go. 

.. raw:: html

   <details>
   <summary>Conda installer instructions</summary>

-  This process is the same on every operating system
-  Follow this process if you won't need to work with the CaImAn sources
   and instead wish to use it as a library (the demos still work this way, and
   you can use/modify them).
-  You do not need a compiler for this route.
-  You should not download the sources (with git or otherwise) for this route.
-  This route also sets environment variables for you (skip that section below)
-  Download and install Anaconda (Python 3.x)
   http://docs.continuum.io/anaconda/install. Allow the installer to
   modify your PATH variable
-  Install mamba into your base environment, with ``conda install -n base -c conda-forge mamba``
-  Create a new environment with the caiman package from conda-forge:
-  If you are installing on Windows, use the conda enabled shell (under "Anaconda" in your programs menu) rather than powershell or a generic prompt:

.. code:: bash

    mamba create -n caiman -c conda-forge caiman
    conda activate caiman

- The previous two steps installed caiman and activated the virtual environment. 
-  Skip ahead to the section on setting up a data directory with `caimanmanager`

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
to contribute to the project, fix bugs etc.. The general motivation for setting up
an editable development environment is described in detail in our `contributors page <../../CONTRIBUTING.md>`_.


Dev-mode install on Windows
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. raw:: html

   <details>
   <summary>Dev Mode Installation on Windows</summary>

The Windows installation process differs more widely from installation
on Linux or MacOSX and has different issues you may run into. Everything 
you do should be from the anaconda prompt rather than from Powershell 
or any other shell.

-  Remove any associations you may have made between .py files and an existing python
   interpreter or editor
-  If you have less than 64GB of RAM, increase the maximum size of your pagefile to 64G or more
   (http://www.tomshardware.com/faq/id-2864547/manage-virtual-memory-pagefile-windows.html).
   The Windows memmap interface is sensitive to the maximum setting
   and leaving it at the default can cause errors when processing larger
   datasets
Installing CaImAn from a package on Windows should be otherwise the same as any other OS for the
package-based process described above.

At the conda prompt:

.. code:: bash

     git clone https://github.com/your-username/CaImAn
     cd CaImAn
     mamba env create -f environment.yml -n caiman
     mamba install -n caiman vs2017_win-64


Note, as discussed at CONTRIBUTORS.md, you should clone from a fork of caiman at your own 
github repo. 

At this point you may need to remove a startup script that visual
studio made for your conda environment that can cause conda to crash
while entering the caiman environment. Use the Windows find-file utility
(under the Start Menu) to look for vs2015_compiler_vars.bat and/or
vs2017_compiler_vars.bat under your home directory. If a copy shows up, delete the version that has
conda:raw-latex:`\envs`:raw-latex:`\caiman` as part of its location.
You may then continue the installation.

.. code:: bash

     conda activate caiman
     pip install -e .  
     copy caimanmanager.py ..
     cd ..

.. raw:: html

   </details>

Dev Mode Install on MacOS and Linux
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. raw:: html

   <details>
   <summary>Installation on MacOS and Linux</summary>

.. code:: bash

     git clone https://github.com/your-username/CaImAn
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



Section 2: Set up demos 
-----------------------

Once Caiman is installed, you will likely want to set up a working directory with
code samples and datasets. The first installation step produced a command ``caimanmanager.py`` that
manages this. If you have not installed Caiman before, you can do

``caimanmanager.py install``

if you used the conda-forge package or the ``pip install .`` option

OR

``python caimanmanager.py install --inplace`` if you used the developer
mode with ``pip install -e .``

This will place that directory under your home directory in a directory
called caiman_data. If you have, some of the demos or datafiles may have
changed since your last install, to follow API changes. You can check to
see if they have by doing ``caimanmanager.py check``
(or ``python caimanmanager.py check``). If they have not,
you may keep using them. If they have, we recommend moving your old
caiman data directory out of the way (or just remove them if you have no
precious data) and doing a new data install as per above.

If you prefer to manage this information somewhere else, the
``CAIMAN_DATA`` environment variable can be set to customise it. The
caimanmanager tool and other libraries will respect that.


Section 3: Upgrading
--------------------

Upgrading can mean a couple of things. First, it typically means there has been a new release of Caiman, so you need 
to install the new version of Caiman. Second, it could mean you need
to upgrade changes to the demos in ``caiman_data`` using ``caimanmanager``. Here, we'll discuss how to upgrade 
Caiman depending on how you've installed, and also how to upgrade your demo code/data in ``caiman_data`` using 
``caimanmanager``. 


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

From the conda environment you used to install CaImAn:

1. ``pip uninstall caiman``

2. Remove your conda environment: ``conda env remove -n caiman`` (or whatever you called the conda environment you used)

3. Close and reopen your shell (to clear out the old conda environment)

4. Do a ``git pull`` from inside your CaImAn folder.

5. Recreate and reenter your conda environment as you did in the installation instructions

6. Do a ``pip install .`` inside that code checkout

7. Run ``caimanmanager.py install`` to reinstall the data directory (use ``--inplace`` if you used the ``pip install -e .`` during your initial installation).

-  If you used the ``pip install -e .`` option when installing, then you
   can try updating by simply doing a ``git pull``. Again, this might
   not lead to the latest version of the code if the environment
   variables have changed.

-  The same applies if you want to modify some internal function of
   CaImAn. If you used the ``pip install -e .`` option then you can
   directly modify it (that's why it's called developer mode). If you
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
requires changes to files in ``caiman_data``), we update the demo and data. This means that upgrading CaImAn works 
best if you also replace the ``caiman_data`` directory with a new version.

However, you may have made your own changes to the demos (e.g. to work with your data). If you have done this, 
you may need to massage your changes into the new versions of the demos. For this reason, we recommend that if 
you modify the demos to operate on your own data to save them as a different file to avoid losing your work 
when updating the caiman_data directory.

To update ``caiman_data`` you can follow the following procedure:

- If there are no new demos or files in the new CaImAn distribution, then you can leave it as is.

- If you have not modified anything in caiman_data but there have been changes in the new Caiman release, 
  then remove ``caiman_data`` directory before upgrading and have ``caimanmanager`` make a new one after the upgrade, by 
  running caimanmanager as discussed in Section 2.

- If you have extensively modified things in ``caiman_data``, rename your ``caiman_data`` directory, and have caimanmanager 
  make a new one after the upgrade, and then massage your changes back in. E.g., if you have extensively 
  modified ``demo_pipeline.ipynb`` for your personal use-case, then change the name of this notebook before putting it back into ``caiman_data``.

.. raw:: html

   </details>

Section 4: Miscellaneous
-------------------------

Section 4A: Setting up environment variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To make the package work *efficiently* and eliminate "crosstalk" between
different processes, some multithreading operations need to be turned off
This is for Linux and Windows and is not necessary is OSX. This process is
not needed if you installed using conda/mamba.

For **Linux (and OSX)** run these commands before launching Python:

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

**If you installed using the conda-forge package, this is done automatically for you.**


Section 4B: Requirements
~~~~~~~~~~~~~~~~~~~~~~~~

32G RAM is required for a good experience, and depending on datasets, more may be necessary. Caiman is optimized for use by 
multiple CPUs, so workstations or clusters with multiple CPU cores are ideal (8+ logical cores). GPU computation is not used 
heavily by Caiman (though see Section 4D). 

Right now, Caiman works and is supported on the following platforms:

- Linux on 64-bit x86 CPUs
- MacOS on 64-bit x86 CPUs
- Windows on 64-bit x86 CPUs

ARM-based versions of Apple hardware work (if on a 16G model), but currently happen under x86 emulation and we cannot 
support them as well.  Support for Linux on ARM (e.g. AWS Graviton) is not available (but it may work with the port of conda, 
if you compile Caiman yourself - we do not have binary packages and this is untested). If you care about this, please let us know.


Section 4C: Installing additional packages
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

Section 4D: Other topics
~~~~~~~~~~~~~~~~~~~~~~~~~

- `Running Caiman on a cluster <./Cluster.md>`_ 
- `Setting up Caiman to use your GPUs <./README-GPU.md>`_
- `Install quirks on some older Linux distributions <./README-Distros.md>`_ 