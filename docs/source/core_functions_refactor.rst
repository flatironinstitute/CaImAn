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





Utilities
---------------
.. currentmodule:: utilities

.. autofunction:: local_correlations
.. autofunction:: plot_contours
.. autofunction:: view_patches_bar
.. autofunction:: view_patches
.. autofunction:: manually_refine_components
.. autofunction:: nb_view_patches
.. autofunction:: nb_imshow
.. autofunction:: nb_plot_contour
.. autofunction:: start_server
.. autofunction:: stop_server
.. autofunction:: order_components		  
.. autofunction:: extract_DF_F
