CaImAn and GPUs
===============
Some parts of Caiman use Tensorflow, and those parts can benefit from using a
hardware-accelerated version of Tensorflow. These versions can be installed using Conda
(presently only on the Linux and Windows platforms). Note that this is experimental
and our ability to support this environment is limited. To do this:

* First install the Caiman environment as normal. Ideally test it.
* Next, install tensorflow-gpu as follows: `conda install tensorflow-gpu`
* Finally, reinstall opencv using the latest openblas build. `conda update opencv`

The last step should get you a current openblas-based build, although in some cases you will get an mkl
build of opencv (which may not have graphical bindings). If this happens, you may be able to
search for the most recent opencv with openblas and switch to it using existing conda tooling.

To do this, check after switching package versions that KERAS_BACKEND is set to tensorflow. If not, you can set it in your terminal by typing `KERAS_BACKEND=tensorflow`. On some platforms it defaults to theano, which will not be hardware-accelerated. This can affect the computational of the Caiman online algorithm.
