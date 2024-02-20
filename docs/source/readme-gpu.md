CaImAn and GPUs
=====================
Some parts of CaImAn already use Tensorflow, and those parts can benefit from using a
hardware-accelerated version of Tensorflow. These versions can be installed using Conda
(presently only on the Linux and Windows platforms). Note that this is experimental
and our ability to support this environment is limited. To do this:

* First install the CaImAn environment as normal. Ideally test it.
* Next, install tensorflow-gpu as follows: `conda install tensorflow-gpu`
* Finally, reinstall opencv using the latest openblas build. `conda update opencv`

The last step should get you a current openblas-based build, although in some cases you will get an mkl
build of opencv (which may not have graphical bindings). If this happens, you may be able to
search for the most recent opencv with openblas and switch to it using existing conda tooling.

To do this, check after switching package versions that KERAS_BACKEND is set to tensorflow. If not, you can set it in your terminal by typing `KERAS_BACKEND=tensorflow`. On some platforms it defaults to theano, which will not be hardware-accelerated. This can affect the computational of the CaImAn online algorithm.

CaImAn and CUDA
---------------

CaImAn has experimental support for computing FFTs on your GPU,
using the pycuda libraries instead of OpenCV or numpy. This can be used during motion correction.

Installation
------------
We assume you have CaImAn generally working first. If you do not,
follow the general instructions first and verify you have a working
environment.

One (but less-tested with Caiman) route is to install pycuda from conda-forge; activate your
caiman environment and then do `conda install -c conda-forge pycuda`. This may be all you need.

It is possible to instead install CUDA, then use pip to install pycuda, but this is involved enough
that it's better to try the above first.

Use
---
The CUDA codepaths will only be active if the needed libraries are installed on your system. Otherwise non-CUDA codepaths will be active (even if CUDA is requested in our code).

The following functions have been extended with a
use_cuda flag (defaults to false) that if set true will use CUDA for FFT
computation (if the needed libraries are involved):

* The MotionCorrect class - this is most likely the interface you want
* MotionCorrect::motion_correct_rigid() - used internally by MotionCorrect class
* register_translation()
* tile_and_correct()
* motion_correct_batch_rigid()
* motion_correct_batch_pwrigid()
* motion_correction_piecewise()

If you have your environment set up correctly, you can add use_cuda=True to the arguments of MotionCorrect() in your code and it will use your GPU for motion correction.
