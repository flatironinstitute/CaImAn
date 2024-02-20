Caiman Features
---------------

CaImAn includes a variety of scalable methods for the analysis of
calcium (and voltage) imaging data:

|
-  **Motion correction** [NoRMCorre]_

   -  Fast parallelizable OpenCV and FFT-based motion correction of
      large movies
   -  Can be run also in online mode (i.e. one frame at a time)
   -  Corrects for non-rigid artifacts due to raster scanning or
      non-uniform brain motion
   -  FFTs can be computed on GPUs (experimental). Requires pycuda and
      skcuda to be installed.
|
-  **Source extraction**

   -  Separates different sources based on constrained nonnegative
      matrix Factorization [CNMF]_, [SNMF]_, [CaImAn]_
   -  Deals with heavily overlapping and neuropil contaminated movies
   -  Suitable for both 2-photon [CNMF]_ and 1-photon [CNMF_E]_ calcium imaging data
   -  Selection of inferred sources using a pre-trained convolutional
      neural network classifier
   -  Online processing available for both 2-photon [OnACID]_ and 1-photon
      data streams [OnACID-E]_
|
-  **Denoising, deconvolution and spike extraction**

   -  Infers neural activity from fluorescence traces [CNMF]_
   -  Also works in online mode (i.e. one sample at a time) [OASIS]_
|
-  **Automatic ROI registration across multiple days** [CaImAn]_
|
-  **Handling of very large datasets**

   -  Utilizes memory mapping for efficient loading of large datasets.
   -  Parallel processing in patches.
   -  Frame-by-frame online processing.
   -  OpenCV-based efficient movie playing and resizing.
|
- **Pipeline for Voltage Imaging Analysis** [VolPY]_

   -  Uses a Mask R-CNN to identify neurons in the FOV
   -  Extracts spiking activity using adaptive template matching.
   -  Fully integrated with CaImAn, inherits all its capabilities.
|
-  **Behavioral Analysis** [Behavior]_

   -  Unsupervised algorithms based on optical flow and NMF to
      automatically extract motor kinetics
   -  Scales to large datasets by exploiting online dictionary learning
   -  We also developed a tool for acquiring movies at high speed with
      low cost equipment `[Github
      repository] <https://github.com/bensondaled/eyeblink>`__.
|
-  **Variance Stabilization** [VST]_

   -  Noise parameters estimation under the Poisson-Gaussian noise model
   -  Fast algorithm that scales to large datasets
   -  A basic demo can be found at
      ``CaImAn/demos/notebooks/demo_VST.ipynb``



References
==========

The following references provide the theoretical background and original
code for the included methods.

Software package detailed description and benchmarking
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you use this code please cite the corresponding papers where original
methods appeared (see References below), as well as:

.. [CaImAn]  Giovannucci A., Friedrich J., Gunn P., Kalfon J., Koay S.A., Taxidis
    J., Najafi F., Gauthier J.L., Zhou P., Tank D.W., Chklovskii D.B.,
    Pnevmatikakis E.A. (2018). CaImAn: An open source tool for scalable
    Calcium Imaging data Analysis. eLife 2019;8:e38173. `[paper] <https://elifesciences.org/articles/38173>`__

Deconvolution and demixing of calcium imaging data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. [CNMF]  Pnevmatikakis, E.A., Soudry, D., Gao, Y., Machado, T., Merel, J., …
    & Paninski, L. (2016). Simultaneous denoising, deconvolution, and
    demixing of calcium imaging data. Neuron 89(2):285-299,
    `[paper] <http://dx.doi.org/10.1016/j.neuron.2015.11.037>`__, `[Github
    repository] <https://github.com/epnev/ca_source_extraction>`__.

.. [SNMF]  Pnevmatikakis, E.A., Gao, Y., Soudry, D., Pfau, D., Lacefield, C., …
    & Paninski, L. (2014). A structured matrix factorization framework for
    large scale calcium imaging data analysis. arXiv preprint
    arXiv:1409.2903. `[paper] <http://arxiv.org/abs/1409.2903>`__.

.. [CNMF_E]  Zhou, P., Resendez, S. L., Stuber, G. D., Kass, R. E., & Paninski,
    L. (2016). Efficient and accurate extraction of in vivo calcium signals
    from microendoscopic video data. eLife 2018;7:e28728.
    `[paper] <https://elifesciences.org/articles/28728>`__, `[Github
    repository] <https://github.com/zhoupc/CNMF_E>`__.

.. [OASIS] Friedrich J. and Paninski L. Fast active set methods for online
    spike inference from calcium imaging. NIPS, 29:1984-1992, 2016.
    `[paper] <https://papers.nips.cc/paper/6505-fast-active-set-methods-for-online-spike-inference-from-calcium-imaging>`__,
    `[Github repository] <https://github.com/j-friedrich/OASIS>`__.

Online Analysis
~~~~~~~~~~~~~~~

.. [OnACID] Giovannucci, A., Friedrich J., Kaufman M., Churchland A., Chklovskii
    D., Paninski L., & Pnevmatikakis E.A. (2017). OnACID: Online analysis of
    calcium imaging data in real data. NIPS 2017, pp. 2378-2388.
    `[paper] <http://papers.nips.cc/paper/6832-onacid-online-analysis-of-calcium-imaging-data-in-real-time>`__

.. [OnACID-E] Friedrich J., Giovannucci A. & Pnevmatikakis E.A. (2020).
    Online analysis of microendoscopic 1-photon calcium imaging data streams. PLoS Comput Biol 17(1):e1008565. `[paper] <https://doi.org/10.1371/journal.pcbi.1008565>`__.

Motion Correction
~~~~~~~~~~~~~~~~~

.. [NoRMCorre] Pnevmatikakis, E.A., and Giovannucci A. (2017). NoRMCorre: An online
    algorithm for piecewise rigid motion correction of calcium imaging data.
    Journal of Neuroscience Methods, 291:83-92
    `[paper] <https://doi.org/10.1016/j.jneumeth.2017.07.031>`__, `[Github
    repository] <https://github.com/simonsfoundation/normcorre>`__.

Behavioral Analysis
~~~~~~~~~~~~~~~~~~~

.. [Behavior] Giovannucci, A., Pnevmatikakis, E. A., Deverett, B., Pereira, T.,
    Fondriest, J., Brady, M. J., … & Masip, D. (2017). Automated gesture
    tracking in head-fixed mice. Journal of Neuroscience Methods, 300:184-195.
    `[paper] <https://doi.org/10.1016/j.jneumeth.2017.07.014>`__.

Variance Stabilization
~~~~~~~~~~~~~~~~~~~~~~

.. [VST]  Tepper, M., Giovannucci, A., and Pnevmatikakis, E (2018). Anscombe
    meets Hough: Noise variance stabilization via parametric model
    estimation. In ICASSP, 2018.
    `[paper] <https://marianotepper.github.io/papers/anscombe-meets-hough.pdf>`__.
    `[Github repository] <https://github.com/marianotepper/hough-anscombe>`__

Voltage imaging
~~~~~~~~~~~~~~~~

.. [VolPY]  Cai, C. , Friedrich, J. , Pnevmatikakis, E. A. , Podgorski, K. , Giovannucci, A.(2020).
    VolPy: automated and scalable analysis pipelines for voltage imaging datasets.
    bioRxiv 2020.01.02.892323 `[paper] <https://www.biorxiv.org/content/10.1101/2020.01.02.892323v1>`__.
