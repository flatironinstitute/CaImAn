CaImAn and Older Linux Distros
=====================
On some older Linux distros, you may have trouble building a working environment because some Conda packages are introducing glibc version requirements into their binary builds. This has been seen on RHEL7/CentOS7 and may show up in other distributions as well. The errors turn up when tensorflow is imported from python code, with the loader failing because it cannot find symbols indicating a modern libc.

Fixing this is not trivial, but it is doable.

Fixing it
=========
To resolve this, you will first want to install a different build of Tensorflow, not from conda-forge. I used
```
conda search -c conda-forge tensorflow
```

to find a version from the main distro built against mkl (you will see mkl in the build string). Having done this, I installed that particular build, in my case by doing the following (your steps may vary):
```
conda install tensorflow=1.13.1=mkl_py37h54b294f_0
```

This works, but in doing so it changed your version of opencv and a number of other things to non-conda-forge versions. The non-conda-forge builds of opencv are built without graphical bindings, making them not useful for some things in CaImAn. We can switch them back by looking for available builds of opencv using a conda search command like the above, and then selecting a (similar version to what we got) suitable conda-forge build of opencv. In my case I finished fixing it with:

```
conda install -c conda-forge opencv=3.4.4
```

Support
=======
This is not a pretty procedure, but we will try to support it if it is necessary in your environment.
