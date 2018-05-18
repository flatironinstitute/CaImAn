Upgrading
=========

If you already have CaImAn installed, but want to upgrade, please follow this procedure:

From the conda environment you used to install CaImAn:
* pip uninstall caiman
* remove or rename your ~/caiman_data directory
* Remove your conda environment: "conda env remove"
* Close and reopen your shell (to clear out the old conda environment)
* Do a "git pull" inside your existing code checkout
* Recreate and reenter your conda environment as you did in the README
* Do a "pip install ." inside that code checkout
* Run "caimanmanager.py install" to reinstall the data directory

If you reinstall CaImAn frequently, you may sometimes be able to skip deleting and recreating your Conda environment, but this may lead to your not getting environment changes we made to fix bugs.

