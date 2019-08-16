Features
--------

CaImAn includes a variety of scalable methods for the analysis of
calcium imaging data:

-  **Handling of very large datasets**

   -  Memory mapping
   -  Parallel processing in patches
   -  Frame-by-frame online processing `[6] <#onacid>`__
   -  OpenCV-based efficient movie playing and resizing

-  **Motion correction** `[7] <#normcorre>`__

   -  Fast parallelizable OpenCV and FFT-based motion correction of
      large movies
   -  Can be run also in online mode (i.e. one frame at a time)
   -  Corrects for non-rigid artifacts due to raster scanning or
      non-uniform brain motion
   -  FFTs can be computed on GPUs (experimental). Requires pycuda and
      skcuda to be installed.

-  **Source extraction**

   -  Separates different sources based on constrained nonnegative
      matrix Factorization (CNMF) `[1-3] <#caiman>`__
   -  Deals with heavily overlapping and neuropil contaminated movies
   -  Suitable for both 2-photon `[2] <#neuron>`__ and 1-photon
      `[4] <#cnmfe>`__ calcium imaging data
   -  Selection of inferred sources using a pre-trained convolutional
      neural network classifier
   -  Online processing available `[6] <#onacid>`__

-  **Denoising, deconvolution and spike extraction**

   -  Infers neural activity from fluorescence traces `[2] <#neuron>`__
   -  Also works in online mode (i.e. one sample at a time)
      `[5] <#oasis>`__

-  **Automatic ROI registration across multiple days** `[1] <#caiman>`__

-  **Behavioral Analysis** `[8] <#behavior>`__

   -  Unsupervised algorithms based on optical flow and NMF to
      automatically extract motor kinetics
   -  Scales to large datasets by exploiting online dictionary learning
   -  We also developed a tool for acquiring movies at high speed with
      low cost equipment `[Github
      repository] <https://github.com/bensondaled/eyeblink>`__.

-  **Variance Stabilization** `[9] <#vst>`__

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

[1] Giovannucci A., Friedrich J., Gunn P., Kalfon J., Koay S.A., Taxidis
J., Najafi F., Gauthier J.L., Zhou P., Tank D.W., Chklovskii D.B.,
Pnevmatikakis E.A. (2018). CaImAn: An open source tool for scalable
Calcium Imaging data Analysis. bioarXiv preprint.
`[paper] <https://doi.org/10.1101/339564>`__

Deconvolution and demixing of calcium imaging data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

[2] Pnevmatikakis, E.A., Soudry, D., Gao, Y., Machado, T., Merel, J., …
& Paninski, L. (2016). Simultaneous denoising, deconvolution, and
demixing of calcium imaging data. Neuron 89(2):285-299,
`[paper] <http://dx.doi.org/10.1016/j.neuron.2015.11.037>`__, `[Github
repository] <https://github.com/epnev/ca_source_extraction>`__.

[3] Pnevmatikakis, E.A., Gao, Y., Soudry, D., Pfau, D., Lacefield, C., …
& Paninski, L. (2014). A structured matrix factorization framework for
large scale calcium imaging data analysis. arXiv preprint
arXiv:1409.2903. `[paper] <http://arxiv.org/abs/1409.2903>`__.

[4] Zhou, P., Resendez, S. L., Stuber, G. D., Kass, R. E., & Paninski,
L. (2016). Efficient and accurate extraction of in vivo calcium signals
from microendoscopic video data. arXiv preprint arXiv:1605.07266.
`[paper] <https://arxiv.org/abs/1605.07266>`__, `[Github
repository] <https://github.com/zhoupc/CNMF_E>`__.

[5] Friedrich J. and Paninski L. Fast active set methods for online
spike inference from calcium imaging. NIPS, 29:1984-1992, 2016.
`[paper] <https://papers.nips.cc/paper/6505-fast-active-set-methods-for-online-spike-inference-from-calcium-imaging>`__,
`[Github repository] <https://github.com/j-friedrich/OASIS>`__.

Online Analysis
~~~~~~~~~~~~~~~

[6] Giovannucci, A., Friedrich J., Kaufman M., Churchland A., Chklovskii
D., Paninski L., & Pnevmatikakis E.A. (2017). OnACID: Online analysis of
calcium imaging data in real data. NIPS 2017, pp. 2378-2388.
`[paper] <http://papers.nips.cc/paper/6832-onacid-online-analysis-of-calcium-imaging-data-in-real-time>`__

Motion Correction
~~~~~~~~~~~~~~~~~

[7] Pnevmatikakis, E.A., and Giovannucci A. (2017). NoRMCorre: An online
algorithm for piecewise rigid motion correction of calcium imaging data.
Journal of Neuroscience Methods, 291:83-92
`[paper] <https://doi.org/10.1016/j.jneumeth.2017.07.031>`__, `[Github
repository] <https://github.com/simonsfoundation/normcorre>`__.

Behavioral Analysis
~~~~~~~~~~~~~~~~~~~

[8] Giovannucci, A., Pnevmatikakis, E. A., Deverett, B., Pereira, T.,
Fondriest, J., Brady, M. J., … & Masip, D. (2017). Automated gesture
tracking in head-fixed mice. Journal of Neuroscience Methods,
300:184-195.
`[paper] <https://doi.org/10.1016/j.jneumeth.2017.07.014>`__.

Variance Stabilization
~~~~~~~~~~~~~~~~~~~~~~

[9] Tepper, M., Giovannucci, A., and Pnevmatikakis, E (2018). Anscombe
meets Hough: Noise variance stabilization via parametric model
estimation. In ICASSP, 2018.
`[paper] <https://marianotepper.github.io/papers/anscombe-meets-hough.pdf>`__.
`[Github
repository] <https://github.com/marianotepper/hough-anscombe>`__
