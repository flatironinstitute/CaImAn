Demos
=====

-  Notebooks: The notebooks provide a simple and friendly way to get
   into CaImAn and understand its main characteristics. They are located
   in the ``demos/notebooks``. To launch one of the jupyter notebooks:

   .. code:: bash

          source activate CaImAn
          jupyter notebook --NotebookApp.iopub_data_rate_limit=1.0e10

   and select the notebook from within Jupyter’s browser. The argument
   ``--NotebookApp.iopub_data_rate_limit=1.0e10`` will prevent any
   memory issues while plotting on a notebook.

-  demo files are also found in the demos/general subfolder. We suggest
   trying demo_pipeline.py first as it contains most of the tasks
   required by calcium imaging. For behavior use demo_behavior.py

-  If you want to directly launch the python files, your python console
   still must be in the CaImAn directory.

Basic Structure
===============

We recently refactored the code to simplify the parameter setting and
usage of the various algorithms. The code now is based revolves around
the following objects:

-  ``params``: A single object containing a set of dictionaries with the
   parameters used in all the algorithms. It can be set and changed
   easily and is passed into all the algorithms.
-  ``MotionCorrect``: An object for motion correction which can be used
   for both rigid and piece-wise rigid motion correction.
-  ``cnmf``: An object for running the CaImAn batch algorithm either in
   patches or not, suitable for both two-photon (CNMF) and one-photon
   (CNMF-E) data.
-  ``online_cnmf``: An object for running the CaImAn online (OnACID)
   algorithm on two-photon data with or without motion correction.
-  ``estimates``: A single object that stores the results of the
   algorithms (CaImAn batch, CaImAn online) in a unified way that also
   contains plotting methods.

To see examples of how these methods are used, please consult the demos.

Result Interpretation
=====================

As mentioned above, the results of the analysis are stored within the
``estimates`` objects. The basic entries are the following:

Result variables for 2p batch analysis
--------------------------------------

The results of CaImAn are saved in an ``estimates`` object. This is
stored inside the cnmf object, i.e. it can be accessed using
``cnmf.estimates``. The variables of interest are:

-  ``estimates.A``: Set of spatial components. Saved as a sparse column format matrix with
   dimensions (# of pixels X # of components). Each column corresponds to a
   spatial component.
-  ``estimates.C``: Set of temporal components. Saved as a numpy array with dimensions (# of components X # of timesteps).
   Each row corresponds to a background component denoised and deconvolved.
-  ``estimates.b``: Set of background spatial components (for 2p
   analysis): Saved as a numpy array with dimensions (# of pixels X # of
   components). Each column corresponds to a spatial background component.
-  ``estimates.f``: Set of temporal background components (for 2p
   analysis). Saved as a numpy array with dimensions (# of background
   components X # of timesteps). Each row corresponds to a temporal
   background component. 
-  ``estimates.S``: Deconvolved neural activity
   (spikes) for each component. Saved as a numpy array with dimensions (#
   of background components X # of timesteps). Each row corresponds to the
   deconvolved neural activity for the corresponding component. 
-  ``estimates.YrA``: Set or residual components. Saved as a numpy array
   with dimensions (# of components X # of timesteps). Each row corresponds
   to the residual signal after denoising the corresponding component in
   ``estimates.C``.
-  ``estimates.F_dff``: Set of DF/F normalized temporal
   components. Saved as a numpy array with dimensions (# of components X #
   of timesteps). Each row corresponds to the DF/F fluorescence for the
   corresponding component.

To view the spatial components, their corresponding vectors need first
to be reshaped into 2d images. For example if you want to view the i-th
component you can type

::

   import matplotlib.pyplot as plt
   plt.figure(); plt.imshow(np.reshape(estimates.A[:,i-1].toarray(), dims, order='F'))

where ``dims`` is a list or tuple that has the dimensions of the FOV.
Similarly if you want to plot the trace for the i-th component you can
simply type

::

   plt.figure(); plt.plot(estimates.V[i-1])

The methods ``estimates.plot_contours`` and
``estimates.view_components`` can be used to visualize all the
components.

Variables for component evaluation
----------------------------------

If you use post-screening to evaluate the quality of the components and
remove bad components the results are stored in the lists: -
``idx_components``: List containing the indexes of accepted components.
- ``idx_components_bad``: List containing the indexes of rejected
components.

These lists can be used to index the results. For example
``estimates.A[:,idx_components]`` or ``estimates.C[idx_components]``
will return the accepted spatial or temporal components, respectively.
If you want to view the first accepted component you can type

::

   plt.figure(); plt.imshow(np.reshape(estimates.A[:,idx_components[0]].toarray(), dims, order='F'))
   plt.figure(); plt.plot(cnm.estimates.C[idx_components[0]])

Variables for 1p processing (CNMF-E)
------------------------------------

The variables for one photon processing are the same, with an additional
variable ``estimates.W`` for the matrix that is used to compute the
background using the ring model, and ``estimates.b0`` for the baseline
value for each pixel.

Variables for online processing
-------------------------------

The same ``estimates`` object is also used for the results of online
processing, stored in ``onacid.estimates``.
