Installation and Updating
======================

Download and install Anaconda or Miniconda (Python 3.x version)
http://docs.continuum.io/anaconda/install

CaImAn installation consists of two steps:

1. Install the CaImAn package
2. Setting up the caimanmanager which will setup a directory with all the demos, test datasets etc.

Installing CaImAn
---------------------

There are two ways to install CaImAn. A package based installation and a development
mode installation.

Package-based Process
---------------------

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
-  Create a new environment with the caiman package from conda-forge:

.. code:: bash

    conda create -n caiman  # caiman here refers to the name of the environment (you can pick any name you want)
    conda activate caiman
    conda install caiman -c conda-forge

-  Skip ahead to the section on setting up a data directory with caimanmanager

Known issues
~~~~~~~~~~~~

If you are on Windows, have used CaImAn before using our github repo and now want to use the conda-forge package,
you might encounter some errors with Python reading the files from the wrong directory. In this case rename
(or remove) the caiman directory that contains the source of the repo and the caiman_data folder and then proceed
with setting up the caiman_data folder as explained below.

.. raw:: html

   </details>


Development mode Installation Process
------------------------------------------


This will allow you to modify the source files of CaImAn and will make it easier
to contribute to the CaImAn project, fix bugs etc.


Installation on Windows
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details>
   <summary>Installation on Windows</summary>

The Windows installation process differs more widely from installation
on Linux or MacOSX and has different issues you may run into.

-  Increase the maximum size of your pagefile to 64G or more
   (http://www.tomshardware.com/faq/id-2864547/manage-virtual-memory-pagefile-windows.html).
   The Windows memmap interface is sensitive to the maximum setting
   and leaving it at the default can cause errors when processing larger
   datasets
-  Use Conda to install git (With “conda install git”) - use of
   another commandline git is acceptable, but may lead to issues
   depending on default settings
-  Install Microsoft Build Tools for Visual Studio 2017
   https://www.visualstudio.com/downloads/#build-tools-for-visual-studio-2017.
   Check the “Build Tools” box, and in the detailed view on the right
   check the “C/C++ CLI Tools” component too. The specifics of this
   occasionally change as Microsoft changes its products and website;
   you may need to go off-script.

Use the following menu item to launch a anaconda-enabled command prompt:
start>programs>anaconda3>anaconda prompt From that prompt. issue the
following commands (if you wish to use the dev branch, you may switch
branches after the clone):

.. code:: bash

     git clone https://github.com/flatironinstitute/CaImAn
     cd CaImAn
     conda env create -f environment.yml -n caiman
     conda install -n caiman vs2017_win-64

At this point you will want to remove a startup script that visual
studio made for your conda environment that can cause conda to crash
while entering the caiman environment. Use the Windows find-file utility
(under the Start Menu) to look for vs2015_compiler_vars.bat and/or
vs2015_compiler_vars.bat under your home directory. At least one copy
should show up. Delete the version that has
conda:raw-latex:`\envs`:raw-latex:`\caiman` as part of its location.
You may then continue the installation.

.. code:: bash

     conda activate caiman
     pip install -e .  # OR `pip install .` if you don't want to develop code
     copy caimanmanager.py ..
     cd ..

.. raw:: html

   </details>

Installation on MacOS and Linux
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details>
   <summary>Installation on MacOS and Linux</summary>

.. code:: bash

     git clone https://github.com/flatironinstitute/CaImAn
     cd CaImAn/
     conda env create -f environment.yml -n caiman
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


Setting up environment variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details>
   <summary>Setting up environmental variables (all platforms)</summary>

To make the package work *efficiently* and eliminate “crosstalk” between
different processes, some multithreading operations need to be turned off
This is for Linux and Windows and is not necessary is OSX. This process is
not needed if you used the conda-forge installation process.

For **Linux (and OSX)** run these commands before launching Python:

.. code:: bash

     export MKL_NUM_THREADS=1
     export OPENBLAS_NUM_THREADS=1

For **Windows** run the same commands, replacing the word ```export``` with the word ```set```.

The commands should be run *every time* before launching python. It is
recommended that you save these values inside your environment so you
don’t have to repeat this process every time. You can do this by
following the instructions
`here <https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#saving-environment-variables>`__.

**If you installed using the conda-forge package, this is done automatically for you.**

.. raw:: html

    </details>


Setting up caimanmanager
-------------------------

Once CaImAn is installed, you may want to get a working directory with
code samples and datasets; pip installed a caimanmanager.py command that
manages this. If you have not installed Caiman before, you can do

``caimanmanager.py install``
if you used the conda-forge package or the `pip install .` option

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


Upgrading
--------------

To upgrade CaImAn you will need to upgrade both the package and the ``caiman_data`` directory through the ``caimanmanager``.


Upgrading the conda-forge package
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details>
   <summary>Updating the conda-forge package</summary>

From within your caiman environment type ```conda update caiman -c conda-forge```. In most cases this should be enough.
If not, you may want to create a new environmrent from scratch and (optionally) remove your existing environment. To do that:

1. Remove your conda environment: ``conda env remove -n caiman`` (or whatever you called the conda environment you used)

2. remove or rename your ~/caiman_data directory

3. Repeat the install instructions

.. raw:: html

   </details>


Upgrading and source-based installations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <details>
   <summary>Updating in development mode</summary>

If you already have CaImAn installed with the pip installer (development mode),
but want to upgrade, please follow the procedure below. If you
reinstall CaImAn frequently, you can try skip deleting and recreating
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
   directly modify it (that’s why it’s called developer mode). If you
   used the ``pip install .`` option then you will need to
   ``pip uninstall caiman`` followed by ``pip install .`` for your
   changes to take effect. Depending on the functions you’re changing so
   you might be able to skip this step.

.. raw:: html

   </details>


Upgrading and caiman_data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When you upgrade CaImAn, sometimes the underlying APIs change. When this happens and it impacts a demo (or otherwise requires changes to caiman_data), we update the demo and data. This means that upgrading CaImAn works best if you also replace the caiman_data directory with a new version.

However, you may have made your own changes to the demos (e.g. to work with your data). If you have done this, you may need to massage your changes into the new versions of the demos. For this reason, we recommend that if you modify the demos to operate
on your own data to save them as a different file to avoid losing your work when updating the caiman_data directory.

To update the caiman_data directory you can follow the following procedure:

- If there are no new demos or files in the new CaImAn distribution, then you can leave it as is.

- If you have not modified anything in caiman_data but there are upstream changes in the new CaImAn distribution, then remove caiman_data directory before upgrading and have caimanmanager make a new one after the upgrade.

- If you have extensively modified things in caiman_data, rename the caiman_manager directory, have caimanmanager make a new one after the upgrade, and then massage your changes back in.


Installing additional packages
---------------------------------

CaImAn uses the conda-forge conda channel for installing its required
packages. If you want to install new packages into your conda
environment for CaImAn, it is important that you not mix conda-forge and
the defaults channel; we recommend only using conda-forge. To ensure
you’re not mixing channels, perform the install (inside your
environment) as follows:

::

   conda install -c conda-forge --override-channels NEW_PACKAGE_NAME

You will notice that any packages installed this way will mention, in
their listing, that they’re from conda-forge, with none of them having a
blank origin. If you fail to do this, differences between how packages
are built in conda-forge versus the default conda channels may mean that
some packages (e.g. OpenCV) stop working despite showing as installed.
