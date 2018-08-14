Core Functions
=======================

Functions that are required to operate the package at a basic level

.. autosummary::

   source_extraction.cnmf.pre_processing.preprocess_data

   initialization.initialize_components

   spatial.update_spatial_components

   temporal.update_temporal_components

   merging.merge_components

   utilities.local_correlations

   utilities.plot_contours

   utilities.view_patches_bar

   utilities.order_components

   utilities.manually_refine_components


Movie Handling
---------------

.. currentmodule:: caiman.base.movies.py

.. autofunction:: load
.. autofunction:: load_movie_chain
.. autofunction:: play
.. autofunction:: resize
.. autofunction:: computeDFF

Timeseries Handling
---------------

.. currentmodule:: caiman.base.timeseries.py

.. autofunction:: save
.. autofunction:: concatenate

ROIs 
---------------

.. currentmodule:: caiman.base.rois.py

.. autofunction:: com
.. autofunction:: extract_binary_masks_from_structural_channel
.. autofunction:: register_ROIs
.. autofunction:: register_multisession
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

.. automodule:: motion_correct
.. automodule:: motion_correct_rigid
.. automodule:: motion_correct_pwrigid
.. automodule:: apply_shifts_movie
.. autofunction:: motion_correct_oneP_rigid
.. autofunction:: motion_correct_oneP_nonrigid


Estimates
---------------

.. currentmodule:: caiman.source_extraction.cnmf.utilities

.. autoclass:: Estimates
.. automodule:: compute_residuals
.. automodule:: detrend_df_f
.. automodule:: normalize_components
.. automodule:: select_components
.. automodule:: evaluate_components
.. automodule:: evaluate_components_CNN
.. automodule:: filter_components
.. automodule:: remove_duplicates
.. automodule:: plot_contours
.. automodule:: plot_contours_nb
.. automodule:: view_components
.. automodule:: nb_view_components
.. automodule:: nb_view_components_3d
.. automodule:: play_movie

Deconvolution
---------------

.. currentmodule:: caiman.source_extraction.cnmf.deconvolution

.. autofunction:: constrained_foopsi
.. autofunction:: constrained_oasisAR2


Parameter Setting
---------------

.. currentmodule:: caiman.source_extraction.cnmf.params

.. autoclass:: CNMFParams
.. automodule:: set
.. automodule:: get
.. automodule:: get_group
.. automodule:: change_params

