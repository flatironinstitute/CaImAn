On file types and sizes
======================

CaImAn is designed to perform analysis on datasets saved over a single
or multiple files. However maximum efficiency is achieved when each
dataset is saved as a sequence of files of medium size (1-2GBs). Please
note the following:

-  If you’re using TIFF files make sure that the files are saved in
   multipage format. This is particularly important as multipage TIFF
   files can be indexed and individual frames can be read without
   loading the entire file in memory. On the contrary single page TIFFs
   would load the entire file before reading an individual frame. This
   can cause significant problems in CaImAn in terms of speed and memory
   consumption, as a lot of the parallelization (e.g. during motion
   correction) happens by passing the path to a file to multiple
   processes each of which will only read and process a small part of
   it. **Bear in mind that TIFF files of size 4GB or larger saved
   through ImageJ/FIJI are automatically save in single page format and
   should be avoided**. If you have such a file you can split into
   multiple shorter files through ImageJ/FIJI or through CaImAn using
   the following script

::

   import numpy as np
   import caiman as cm
   fname = ''  # path to file
   m = cm.load(fname)  # load the file
   T = m.shape(0)  # total number of frames for the file
   L = 1000  # length of each individual file
   fileparts = fname.split(".")
   for t in np.arange(0,T,L):
      m[t:t+L].save((".").join(fileparts[:-1]) + '_' + str(t//L) + '.' + fileparts[-1])

HDF5/H5 files in general do not suffer from this problem.

-  Single frame files should be avoided. The reason is that several
   functions, e.g. motion correction, memory mapping, are designed to
   work on small sets of frames and in general assume that each file has
   more than 1 frames. If your data is saved as a series of single frame
   files, you should convert them in a single (or multiple) files. You
   can do this by using the following script:

::

   import os
   import glob
   import caiman as cm
   fld = ''  # path to folder where the data is located
   fls = glob.glob(os.path.join(fld,'*.tif'))  #  change tif to the extension you need
   fls.sort()  # make sure your files are sorted alphanumerically
   m = cm.load_movie_chain(fls)
   m.save(os.path.join(fld,'data.tif'))

If the number of frames is too big, you can split into multiple files as
explained above. Make sure that your files are sorted alphanumerically
before combining them. This can be tricky if your files are initially
saved as ’file1.tif, file2.tif, …, file10.tif`. In this case you can
consult this `page <https://wiki.python.org/moin/HowTo/Sorting>`__.
