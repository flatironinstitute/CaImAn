Core Functions
=======================

Functions that are required to operate the package at a basic level

.. autosummary::

   caiman.source_extraction.cnmf.CNMF

   caiman.source_extraction.cnmf.CNMF.fit

   caiman.source_extraction.online_cnmf.OnACID

   caiman.source_extraction.online_cnmf.OnACID.fit_online

   caiman.source_extraction.params.CNMFparams

   caiman.source_extraction.estimates.Estimates

   caiman.motion_correction.MotionCorrect

   caiman.motion_correction.MotionCorrect.motion_correct

   caiman.movies.movie.load
   
   caiman.movies.movie.play
   
   caiman.base.rois.register_ROIs

   caiman.base.rois.register_multisession


Movie Handling
---------------

.. currentmodule:: caiman.base.movies.py

.. autoclass:: movie
.. autofunction:: movie.load
.. autofunction:: movie.load_movie_chain
.. autofunction:: movie.play
.. autofunction:: movie.resize
.. autofunction:: movie.computeDFF

Timeseries Handling
---------------

.. currentmodule:: caiman.base.timeseries.py

.. autoclass:: timeseries
.. autofunction:: timeseries.save
.. autofunction:: timeseries.concatenate

ROIs 
---------------

.. currentmodule:: caiman.base.rois.py

.. autofunction:: com
.. autofunction:: extract_binary_masks_from_structural_channel
.. autofunction:: register_ROIs
.. autofunction:: register_multisession


Parallel Processing functions
---------------

.. currentmodule:: caiman.cluster

.. autofunction:: apply_to_patch
.. autofunction:: start_server
.. autofunction:: stop_server


Memory mapping
---------------

.. currentmodule:: caiman.mmaping

.. autofunction:: load_memmap
.. autofunction:: save_memmap_join
.. autofunction:: save_memmap


Image statistics
---------------

.. currentmodule:: caiman.summary_images

.. autofunction:: local_correlations
.. autofunction:: max_correlation_image
.. autofunction:: correlation_pnr


Motion Correction
---------------

.. currentmodule:: caiman.motion_correction

.. autoclass:: MotionCorrect
.. autofunction:: MotionCorrect.motion_correct
.. autofunction:: MotionCorrect.motion_correct_rigid
.. autofunction:: MotionCorrect.motion_correct_pwrigid
.. autofunction:: MotionCorrect.apply_shifts_movie
.. autofunction:: motion_correct_oneP_rigid
.. autofunction:: motion_correct_oneP_nonrigid


Estimates
---------------

.. currentmodule:: caiman.source_extraction.cnmf.utilities

.. autoclass:: Estimates
.. autofunction:: Estimates.compute_residuals
.. autofunction:: Estimates.detrend_df_f
.. autofunction:: Estimates.normalize_components
.. autofunction:: Estimates.select_components
.. autofunction:: Estimates.evaluate_components
.. autofunction:: Estimates.evaluate_components_CNN
.. autofunction:: Estimates.filter_components
.. autofunction:: Estimates.remove_duplicates
.. autofunction:: Estimates.plot_contours
.. autofunction:: Estimates.plot_contours_nb
.. autofunction:: Estimates.view_components
.. autofunction:: Estimates.nb_view_components
.. autofunction:: Estimates.nb_view_components_3d
.. autofunction:: Estimates.play_movie

Deconvolution
---------------

.. currentmodule:: caiman.source_extraction.cnmf.deconvolution

.. autofunction:: constrained_foopsi
.. autofunction:: constrained_oasisAR2


Parameter Setting
---------------

.. currentmodule:: caiman.source_extraction.cnmf.params

.. autoclass:: CNMFParams
.. autofunction:: CNMFParams.set
.. autofunction:: CNMFParams.get
.. autofunction:: CNMFParams.get_group
.. autofunction:: CNMFParams.change_params

CNMF
---------------

.. currentmodule:: caiman.source_extraction.cnmf.cnmf

.. autoclass:: CNMF
.. autofunction:: CNMF.fit
.. autofunction:: CNMF.refit
.. autofunction:: CNMF.fit_file
.. autofunction:: CNMF.save
.. autofunction:: CNMF.deconvolve
.. autofunction:: CNMF.update_spatial
.. autofunction:: CNMF.update_temporal
.. autofunction:: CNMF.HALS4traces
.. autofunction:: CNMF.HALS4footprints
.. autofunction:: CNMF.merge_comps
.. autofunction:: CNMF.initialize
.. autofunction:: CNMF.preprocess
.. autofunction:: CNMF.load_CNMF


Online CNMF (OnACID)
---------------

.. currentmodule:: caiman.source_extraction.cnmf.online_cnmf

.. autoclass:: OnACID
.. autofunction:: OnACID.fit_online
.. autofunction:: OnACID.fit_next
.. autofunction:: OnACID.save
.. autofunction:: OnACID.initialize_online
.. autofunction:: OnACID.load_OnlineCNMF

Preprocessing
---------------
.. currentmodule:: caiman.source_extraction.cnmf.pre_processing

.. autofunction:: preprocess_data


Initialization
---------------
.. currentmodule:: caiman.source_extraction.cnmf.initialization

.. autofunction:: initialize_components


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

.. autofunction:: detrend_df_f_auto
.. autofunction:: update_order
.. autofunction:: get_file_size
