On Clustering
=============

Some parts of the CaImAn library (and demos) are best done by farming work out over multiple CPUs (or in some cases, multiple separate systems in a cluster). Python provides (at least) two popular frameworks for this, multiprocessing and ipyparallel. Both of them have several modes and parameters, each with their own advantages and disadvantages. Their behaviour differs on different operating systems (e.g. Linux versus OSX) and different Python environments (script versus Spyder versus Jupyter). CaImAn provides some support for some modes of both frameworks, to make it easier for people to find a setup that works on their platform.

This document covers some of the reasons you might choose one framework over another, how you can pick a clustering option with the codebase, and some of the issues we have seen with each.

Docs on both
============
If you are already reading this, we encourage you to also do independent reading on both.
* [https://ipyparallel.readthedocs.io/en/latest/](Ipyparallel docs)
* [https://docs.python.org/3.4/library/multiprocessing.html](Multiprocessing docs - Python3)

Multiprocessing has three modes, each of which behaves differently:
* spawn - The normal on Windows, available on other platforms. Slightly slower, but safer.
* fork - The normal on all platforms but Windows. Efficient, but if any of the libraries Caiman uses are themselves multithreaded, can create problems
* forkserver - Fast, but your code needs to either be running under Spyder or Jupyter, or needs a certain organisation to make it work.

Comparison
==========
* Multiprocessing is designed to run multiple processes on the same system, to efficiently use multiple machines.
* Ipyparallel can do just that, but can also efficiently use multiple systems (and can integrate with cluster systems to do this automatically

How to select one with the codebase
===================================
Some of the demos use cm.cluster.setup_cluster() to setup a clustered environment. The "backend" parameter selects which configuration will be used
for the cluster. The parameter can take the following documented values:
* multiprocessing (or local) - Uses the multiprocessing module with its default backend for the platform
* ipyparallel - Use ipyparallel

Issues
======
* On OSX, some math libraries (blas, lapack) by default use a hardware-accelerated system framework that interacts badly with threads and might not work if run from a forked process. This causes some worker processes to hang forever if more than one of them tries to do some computations at the same time. You may be able to prevent this with "export VECLIB_MAXIMUM_THREADS=1" before running your code, or telling Conda to install a blas/lapack that is not built against Veclib.
* On Linux and OSX, some multithreaded builds of math libraries do their own multithreading and this can break if CaImAn forks to make new processes. This can result in hangs and/or core dumps. We have in our instructions other environment variables you should set to tell them not to multithread.

