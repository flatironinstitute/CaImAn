Motion Correction tips
----------------------

-  Non-rigid motion correction is not always necessary. Sometimes, rigid
   motion correction will be sufficient and it will lead to significant
   performance gains in terms of speed. Check your data to before/after
   rigid motion correction to decide what is best for you. The boolean
   parameter ``params.motion.pw_rigid`` can be used to alternate between
   rigid and non-rigid motion correction using the same function
   ``MotionCorrect.motion_correct``.

-  When using piecewise rigid motion correction, use parameters that
   physically make sense. For example, a typical patch size could be
   around 100um x 100um since motion can in many times be approximated
   as rigid for smaller patches (if the imaging is not too slow).
   Similarly, the maximum allowed shifts can in typical 2p recordings
   chosen to correspond to 10um. The patch size is given by the sum of
   the parameters ``params.motion.strides + params.motion.overlaps``.
   The maximum shifts parameter is ``params.motion.max_shifts``. These
   values corresponds to pixels so make sure you have a rough idea of
   the spatial resolution of your data. There is a parameter for that
   ``params.data.dxy``.

-  Motion correction works in parallel by splitting each file in
   multiple chunks and processing them in parallel. Make sure that the
   length of each chunk is not too small by setting the parameter
   ``params.motion.num_frames_split``.

CaImAn Online Processing tips
-----------------------------

-  Important parameters for online processing are the CNN threshold
   value ``params.online.thresh_CNN_noisy``, the trace SNR
   ``params.online.min_SNR`` and the number of candidate components to
   be considered at each timestep ``params.online.min_num_trial``. Lower
   values for the thresholds (e.g., 1 for ``params.online.min_SNR`` and
   0.5 for ``params.online.thresh_CNN_noisy``) and/or higher values for
   ``params.online.min_num_trial`` (e.g., 10) can lead to higher recall
   values, although potentially at the expense of lower precision. In
   general they are preferable for datasets that are relatively short
   (e.g., 10000 frames or less). On the other hand, higher threshold
   values (e.g., 1.5 for ``params.online.min_SNR`` and 0.7 for
   ``params.online.thresh_CNN_noisy``) and/or lower values for
   ``params.online.min_num_trial`` (e.g., 5) will lead to higher
   precision values, although potentially at the expense of lower
   recall. n general they are preferable for datasets that are longer
   (e.g., 10000 frames or more).

-  If your analysis setup allows it, multiple epochs over the data can
   be very beneficial, especially in the strict regime or high
   acceptance thresholds.

-  In general, ``bare`` initialization can be used most of the times, to
   capture the neuropil activity and a small number of neurons at an
   initial chunk. For a large FOV with lots of active neurons, e.g., a
   plane from a zebrafish dataset, ``bare`` initialization can be
   inadequate. In this case, a proper initialization with ``cnmf`` can
   lead to substantially better results.

-  Spatial downsampling can lead to significant speed gains, often at no
   expense in terms of accuracy. It can be set through the parameter
   ``ds_factor``.

-  When using the CNN for screening candidate components, the usage of a
   GPU can lead to significant computational gains.

CaImAn Batch processing tips
----------------------------

-  In order to optimize memory consumption and parallelize computing, it
   is suggested to adopt computing in patches (see companion paper). The
   user will inspect the correlation image and select an appropriate
   number of neurons per each patch. The
   ``params.patches['rf']' and``\ params.patches.stride’ parameters
   controls the size of patches and their overlap. Given the patch size
   and the correlation image the user can set an upper bound on the
   number of neurons per patches. We suggest to start exploring regions
   that contain 4-10 neurons.

-  Important parameters for selecting components based on quality are

-  the CNN lower bound and upper threshold ``params.quality.cnn_lowest``
   and ``params.quality.min_cnn_thr``

-  the trace SNR ``params.quality.min_SNR``

-  the footprint consistency threshold ``params.quality.rval_thr``

The user should explore these parameters around the default to optimize
for specific data sets.

1p processing tips
------------------

-  For microendoscopic 1p data use CNMF-E’s background model and
   initialization method by setting ``center_psf=True``,
   ``method_init='corr_pnr'`` and ``ring_size_factor`` to some value
   around 1.5. In this case the spatial and temporal components are
   updated during the initialization phase, hence use
   ``only_init_patch=True``.

-  Other important parameters for microendoscopic 1p data are ``gSig``,
   ``gSiz``, ``min_corr`` and ``min_pnr``. ``gSig`` specifies the
   gaussian width of a 2D gaussian kernel, which approximates a neuron
   and ``gSiz`` the average diameter of a neuron, in general
   ``4*gSig+1``. To pick the thresholds ``min_corr`` and ``min_pnr`` you
   can use ``caiman.utils.visualization.inspect_correlation_pnr`` and
   vary the slider values.

-  Because the background has no high spatial frequency components, it
   can be spatially downscaled to speed up processing without loss in
   accuracy, e.g. by a factor of 2 by setting ``ssub_B=2``.

-  The exact background can be returned as full rank matrix
   (``gnb=-1``), or more compactly as parameters of the ring model
   (``gnb=0``), or not at all (``gnb<-1``). Further the background can
   also be approximated as low rank matrix by setting ``gnb`` to the
   desired rank. ``gnb=0`` is usually the desired choice. If you have
   plenty of RAM and process in patches ``gnb=-1`` is a good and faster
   option.

-  The CNMF-E algorithm poses high demands on RAM. There is however a
   trade off between computing time and memory usage when processing in
   patches. The number of processes ``n_processes`` specifies how many
   patches are processed in parallel, thus a higher number decreases
   computing time but increases RAM usage. If you have insufficient RAM,
   use a smaller value for ``n_processes`` to reduce memory consumption,
   or don’t even use parallelization at all by setting ``dview=None``.

Deconvolution tips
------------------

-  Simultaneous deconvolution and source extraction can mostly offer
   benefits in particularly low SNR data. In most cases, running source
   extraction without deconvolution (``p=0``), followed by deconvolution
   will be sufficient.

-  It is generally better to perform some sort of de-trending on the
   extracted calcium traces prior to deconvolution to correct for
   baseline drifts that can results in wrongfully deconvolved neural
   activity. You can use the ``estimates.detrend_df_f`` methods for
   that.

-  For interpreting the deconvolved neural activity varible ``S``, see
   `here <https://github.com/flatironinstitute/CaImAn-MATLAB/wiki/Interpretation-of-spiking-variable-S>`__.
