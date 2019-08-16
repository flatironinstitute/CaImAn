Getting Started with CaImAn
===========================

Demos
-----

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
---------------

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


Parameters
-----------

CaImAn gives you access to a lot of parameters and lets you adapt the analysis to your data. Parameters are stored in
the ``params`` object in a set of dictionaries, sorted by the part of the analysis they are used in:

-  ``data``: General params describing the dataset like dimensions, decay time, filename and framerate
-  ``init``: Parameters for component initialization like neuron size ``gSig``, patch size etc.
-  ``motion``: motion correction parameters (max shift size, patch size etc.)
-  ``online``: Parameters specific for the online OnACID algorithm
-  ``quality``: Parameters for component evaluation (spatial correlation, SNR and CNN)
-  ``spatial``: Parameters used in detection of spatial components
-  ``temporal``: Parameters used in extraction of temporal components and deconvolution

Of these parameters, most have a default value that usually does not have to be adjusted. However, some parameters are
crucial to be adapted to the specific dataset for proper analysis performance:

-  ``fnames``: List of paths to the file(s) to be analysed. Memmap and hdf5 result files will be saved in the same directory.
-  ``fr``: Imaging frame rate in frames per second.
-  ``decay_time``: Length of a typical transient in seconds. ``decay_time`` is an approximation of the time
   scale over which to expect a significant shift in the calcium signal during a transient. It defaults to ``0.4``, which is
   appropriate for fast indicators (GCaMP6f), slow indicators might use 1 or even more. However, `decay_time` does not have to 
   precisely fit the data, approximations are enough.
-  ``p``: Order of the autoregressive model. ``p = 0`` turns deconvolution off. If transients in your data rise
   instantaneously, set ``p = 1`` (occurs at low sample rate or slow indicator). If transients have visible rise time,
    set ``p = 2``. If the wrong order is chosen, spikes are extracted unreliably.
-  ``nb``: Number of global background components. This is a measure of the complexity of your background noise. Defaults
   to ``nb = 2``, assuming a relatively homogeneous background. ``nb = 3`` might fit for more complex noise, ``nb = 1``
   is usually too low. If ``nb`` is set too low, extracted traces appear too noisy, if ``nb`` is set too high, neuronal
   signal starts getting absorbed into the background reduction, resulting in reduced transients.
-  ``merge_thr``: Merging threshold of components after initialization. If two components are correlated more than this value
   (e.g. when during initialization a neuron was split in two components), they are merged and treated as one.
-  ``rf``: Half-size of the patches in pixels. Should be at least 3 to 4 times larger than the expected neuron size to
   capture the complete neuron and its local background. Larger patches lead to less parallelization.
-  ``stride``: Overlap between patches in pixels. This should be roughly the neuron diameter. Larger overlap increases
   computational load, but yields better results during reconstruction/denoising of the data.
-  ``K`` : Number of (expected) components per patch. Adapt to ``rf`` and estimated component density.
-  ``gSig``: Expected half-size of neurons in pixels [rows X columns]. CRUCIAL parameter for proper component detection.
-  ``method_init``: Initialization method, depends mainly on the recording method. Use ``greedy_roi`` for 2p data,
   ``corr_pnr`` for 1p data, and ``sparse_nmf`` for dendritic/axonal data.
-  ``ssub/tsub``: Spatial and temporal subsampling during initialization. Defaults to 1 (no compression). Can be set
   to 2 or even higher to save resources, but might impair detection/extraction quality.

Component evaluation
--------------------

The quality of detected components is evaluated with three parameters:

-  Spatial footprint consistency (``rval``): The spatial footprint of the component is compared with the
   frames where this component is active. Other component's signals are subtracted from these frames, and
   the resulting raw data is correlated against the spatial component. This ensures that the raw data at
   the spatial footprint aligns with the extracted trace.
-  Trace signal-noise-ratio (``SNR``): Peak SNR is calculated from strong calcium transients and the noise estimate.
-  CNN-based classifier (``cnn``): The shape of components is evaluated by a 4-layered convolutional neural network
   trained on a manually annotated dataset. The CNN assigns a value of 0-1 to each component depending on its
   resemblance to a neuronal soma.

Each parameter has a low threshold (``rval_lowest (default -1), SNR_lowest (default 0.5), cnn_lowest (default 0.1)``)
and high threshold (``rval_thr (default 0.8), min_SNR (default 2.5), min_cnn_thr (default 0.9)``). A component has
to exceed ALL low thresholds as well as ONE high threshold to be accepted.

Additionally, CNN evaluation can be turned off completely with the ``use_cnn`` boolean parameter. This might be useful
when working with manually annotated spatial components (seeded CNMF (link to notebook?)), where it can be assumed
that manually registered ROIs already have a neuron-like shape.


Result Interpretation
----------------------

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
-  ``estimates.YrA``: Set of residual components. Saved as a numpy array
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

where ``dims`` is a list or tuple that has the dimensions of the FOV. To get binary masks
from spatial components you can apply a threshold before reshaping:

::

    M = estimates.A > 0
    masks = [np.reshape(np.array(M[:,i]), dims, order=‘F') for i in range(M.shape[1])]

Similarly if you want to plot the trace for the i-th component you can
simply type

::

   plt.figure(); plt.plot(estimates.C[i-1])

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
