Core Functions
=======================

Functions that are required to operate the package at a basic level

.. autosummary::

   caiman.source_extraction.cnmf.CNMF
   caiman.source_extraction.cnmf.CNMF.fit
   caiman.source_extraction.cnmf.online_cnmf.OnACID
   caiman.source_extraction.cnmf.online_cnmf.OnACID.fit_online
   caiman.source_extraction.cnmf.params.CNMFParams
   caiman.source_extraction.cnmf.estimates.Estimates
   caiman.motion_correction.MotionCorrect
   caiman.motion_correction.MotionCorrect.motion_correct
   caiman.base.movies.load
   caiman.base.movies.movie.play
   caiman.base.rois.register_ROIs
   caiman.base.rois.register_multisession
   caiman.source_extraction.cnmf.utilities.detrend_df_f


Movie Handling
---------------

.. currentmodule:: caiman.base.movies

.. autoclass:: movie
.. automethod:: movie.play
.. automethod:: movie.resize
.. automethod:: movie.computeDFF
.. autofunction:: get_file_size
.. autofunction:: load
.. autofunction:: load_iter
.. autofunction:: load_movie_chain

Timeseries Handling
--------------------

.. currentmodule:: caiman.base.timeseries

.. autoclass:: timeseries
.. automethod:: timeseries.save


Motion Correction
------------------

.. currentmodule:: caiman.motion_correction

.. autoclass:: MotionCorrect
.. automethod:: MotionCorrect.motion_correct
.. automethod:: MotionCorrect.motion_correct_rigid
.. automethod:: MotionCorrect.motion_correct_pwrigid
.. automethod:: MotionCorrect.apply_shifts_movie
.. autofunction:: motion_correct_oneP_rigid
.. autofunction:: motion_correct_oneP_nonrigid


Estimates
---------------

.. currentmodule:: caiman.source_extraction.cnmf.estimates

.. autoclass:: Estimates
.. automethod:: Estimates.compute_residuals
.. automethod:: Estimates.deconvolve
.. automethod:: Estimates.detrend_df_f
.. automethod:: Estimates.evaluate_components
.. automethod:: Estimates.evaluate_components_CNN
.. automethod:: Estimates.filter_components
.. automethod:: Estimates.hv_view_components
.. automethod:: Estimates.nb_view_components
.. automethod:: Estimates.nb_view_components_3d
.. automethod:: Estimates.normalize_components
.. automethod:: Estimates.play_movie
.. automethod:: Estimates.plot_contours
.. automethod:: Estimates.plot_contours_nb
.. automethod:: Estimates.remove_duplicates
.. automethod:: Estimates.remove_small_large_neurons
.. automethod:: Estimates.select_components
.. automethod:: Estimates.restore_discarded_components
.. automethod:: Estimates.save_NWB
.. automethod:: Estimates.view_components

Deconvolution
---------------

.. currentmodule:: caiman.source_extraction.cnmf.deconvolution

.. autofunction:: constrained_foopsi
.. autofunction:: constrained_oasisAR2


Parameter Setting
-----------------

.. currentmodule:: caiman.source_extraction.cnmf.params

.. autoclass:: CNMFParams
.. automethod:: CNMFParams.__init__
.. automethod:: CNMFParams.set
.. automethod:: CNMFParams.get
.. automethod:: CNMFParams.get_group
.. automethod:: CNMFParams.change_params
.. automethod:: CNMFParams.to_dict


CNMF
---------------

.. currentmodule:: caiman.source_extraction.cnmf.cnmf

.. autoclass:: CNMF
.. automethod:: CNMF.fit
.. automethod:: CNMF.refit
.. automethod:: CNMF.fit_file
.. automethod:: CNMF.save
.. automethod:: CNMF.deconvolve
.. automethod:: CNMF.update_spatial
.. automethod:: CNMF.update_temporal
.. automethod:: CNMF.compute_residuals
.. automethod:: CNMF.remove_components
.. automethod:: CNMF.HALS4traces
.. automethod:: CNMF.HALS4footprints
.. automethod:: CNMF.merge_comps
.. automethod:: CNMF.initialize
.. automethod:: CNMF.preprocess
.. autofunction:: load_CNMF


Online CNMF (OnACID)
-------------------------

.. currentmodule:: caiman.source_extraction.cnmf.online_cnmf

.. autoclass:: OnACID
.. automethod:: OnACID.fit_online
.. automethod:: OnACID.fit_next
.. automethod:: OnACID.save
.. automethod:: OnACID.initialize_online
.. autofunction:: load_OnlineCNMF


Preprocessing
---------------
.. currentmodule:: caiman.source_extraction.cnmf.pre_processing

.. autofunction:: preprocess_data


Initialization
---------------
.. currentmodule:: caiman.source_extraction.cnmf.initialization

.. autofunction:: initialize_components
.. autofunction:: greedyROI
.. autofunction:: greedyROI_corr
.. autofunction:: graphNMF
.. autofunction:: sparseNMF


Spatial Components
-------------------
.. currentmodule:: caiman.source_extraction.cnmf.spatial

.. autofunction:: update_spatial_components


Temporal Components
-------------------
.. currentmodule:: caiman.source_extraction.cnmf.temporal

.. autofunction:: update_temporal_components


Merge components
----------------
.. currentmodule:: caiman.source_extraction.cnmf.merging

.. autofunction:: merge_components


Utilities
---------------
.. currentmodule:: caiman.source_extraction.cnmf.utilities

.. autofunction:: detrend_df_f
.. autofunction:: update_order


ROIs
---------------

.. currentmodule:: caiman.base.rois

.. autofunction:: register_ROIs
.. autofunction:: register_multisession
.. autofunction:: com
.. autofunction:: extract_binary_masks_from_structural_channel


Memory mapping
---------------

.. currentmodule:: caiman.mmapping

.. autofunction:: load_memmap
.. autofunction:: save_memmap_join
.. autofunction:: save_memmap


Image statistics
-----------------

.. currentmodule:: caiman.summary_images

.. autofunction:: local_correlations
.. autofunction:: max_correlation_image
.. autofunction:: correlation_pnr


Parallel Processing functions
-----------------------------

.. currentmodule:: caiman.cluster

.. autofunction:: setup_cluster
.. autofunction:: start_server
.. autofunction:: stop_server

Ring-CNN functions
------------------

.. currentmodule:: caiman.utils.nn_models

.. autoclass:: Masked_Conv2D
.. autoclass:: Hadamard
.. autoclass:: Additive
.. autofunction:: create_LN_model
.. autofunction:: fit_NL_model
.. autofunction:: quantile_loss

VolPy
-----

.. currentmodule:: caiman.source_extraction.volpy.volpy
.. autoclass:: VOLPY
.. automethod:: VOLPY.__init__
.. automethod:: VOLPY.fit

.. currentmodule:: caiman.source_extraction.volpy.volparams
.. autoclass:: volparams
.. automethod:: volparams.__init__
.. automethod:: volparams.set
.. automethod:: volparams.get
.. automethod:: volparams.get_group
.. automethod:: volparams.change_params

.. currentmodule:: caiman.source_extraction.volpy.spikepursuit
.. autofunction:: volspike

