If you have you not used the pip installation before (established in May
2018) follow the instructions in the
`README <https://github.com/flatironinstitute/CaImAn>`__ file.

If you already have CaImAn installed with the pip installer, but want to
upgrade, please follow the procedure below. If you reinstall CaImAn
frequently, you can try skip deleting and recreating your Conda
environment. In this case you can do only steps 1, 5, and 7 below to
update the code. However, if the environment file has changed since your
last update this may lead to you not the latest version.

From the conda environment you used to install CaImAn: 1.
``pip uninstall caiman`` 2. remove or rename your ~/caiman_data
directory 3. Remove your conda environment:
``conda env remove -n NAME_OF_YOUR_ENVIRONMENT`` 4. Close and reopen
your shell (to clear out the old conda environment) 5. Do a ``git pull``
from inside your CaImAn folder. 6. Recreate and reenter your conda
environment as you did in the
`README <https://github.com/flatironinstitute/CaImAn>`__ 7. Do a
``pip install .`` inside that code checkout 8. Run
``caimanmanager.py install`` to reinstall the data directory (use
``--inplace`` if you used the ``pip install -e .`` during your initial
installation).

-  If you used the ``pip install -e .`` option, then you can try
   updating by simply doing a ``git pull``. Again, this might not lead
   to the latest version of the code if the environment variables have
   changed.
