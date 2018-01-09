#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on Tue Feb 16 17:56:14 2016

@author: agiovann
"""

from __future__ import print_function

#%%
from builtins import str
from builtins import filter

try:
    get_ipython().magic('load_ext autoreload')
    get_ipython().magic('autoreload 2')
    print((1))
except:
    print('NOT IPYTHON')

import matplotlib as mpl
mpl.use('TKAgg')
from matplotlib import pyplot as plt
# plt.ion()

import sys
import numpy as np
import ca_source_extraction as cse

# sys.path.append('../SPGL1_python_port')
#%
from time import time
from scipy.sparse import coo_matrix
import tifffile
import subprocess
import time as tm
from time import time
import pylab as pl
import psutil
import calblitz as cb
import shutil
import glob
from ipyparallel import Client
import os
import glob
import h5py
import re


#%%
f_tot = []

f_all = []

rg = re.compile('^2016.*\.tif$', re.IGNORECASE)

for root, dirs, files in os.walk("."):

    path = root.split('/')

    f = list(filter(rg.findall, files))

    f.sort()

    if len(f) > 0:
        print(root)
        print((len(f)))

    f2 = set([f1[:-15] for f1 in f])

    f4 = [os.path.join(root, f3) for f3 in f2]

    f_tot.append(f4)
#%%

fnames = []
for f in f_tot:
    for f1 in f:
        if len(f1) > 0:
            f2 = glob.glob(f1 + '*.tif')
            f2 = list(f2)
            f2.sort()
            if len(f2) > 0:
                print('')
                print((f2[:20]))
                fnames = fnames + f2


#%% process files sequntially in case of failure
if 0:
    fnames1 = []
    for f in fnames:
        if os.path.isfile(f[:-3] + 'hdf5'):
            1
        else:
            print((1))
            fnames1.append(f)
    #%% motion correct
    t1 = time()
    file_res = cb.motion_correct_parallel(
        fnames1, fr=30, template=None, margins_out=0, max_shift_w=45, max_shift_h=45, dview=None, apply_smooth=True)
    t2 = time() - t1
    print(t2)
#%% LOGIN TO MASTER NODE
# TYPE salloc -n n_nodes --exclusive
# source activate environment_name

#%%#%%
slurm_script = '/mnt/xfs1/home/agiovann/SOFTWARE/Constrained_NMF/SLURM/slurmStart.sh'
cse.utilities.start_server(slurm_script=slurm_script)
# n_processes = 27#np.maximum(psutil.cpu_count() - 2,1) # roughly number of cores on your machine minus 1
pdir, profile = os.environ['IPPPDIR'], os.environ['IPPPROFILE']
client_ = Client(ipython_dir=pdir, profile=profile)
print(('Using ' + str(len(client_)) + ' processes'))

#%% motion correct
t1 = time()
file_res = cb.motion_correct_parallel(fnames, fr=30, template=None, margins_out=0,
                                      max_shift_w=45, max_shift_h=45, dview=client_[::2], apply_smooth=True)
t2 = time() - t1
print(t2)

#%%
all_movs = []
for f in fnames:
    print(f)
    with np.load(f[:-3] + 'npz') as fl:
        #        pl.subplot(1,2,1)
        #        pl.imshow(fl['template'],cmap=pl.cm.gray)
        #        pl.subplot(1,2,2)
        #        pl.plot(fl['shifts'])
        all_movs.append(fl['template'][np.newaxis, :, :])
#        pl.pause(.1)
#        pl.cla()
#%%
all_movs = cb.movie(np.concatenate(all_movs, axis=0), fr=10)
all_movs, shifts, corss, _ = all_movs.motion_correct(
    template=all_movs[1], max_shift_w=45, max_shift_h=45)
#%%
template = np.median(all_movs[:], axis=0)
np.save(base_folder + 'template_total', template)
pl.imshow(template, cmap=pl.cm.gray, vmax=120)
#%%
all_movs.play(backend='opencv', gain=5, fr=30)
#%%
t1 = time()
file_res = cb.motion_correct_parallel(fnames, 30, template=template, margins_out=0,
                                      max_shift_w=45, max_shift_h=45, dview=client_[::2], remove_blanks=False)
t2 = time() - t1
print(t2)
#%%
fnames = []
for file in glob.glob(base_folder + 'k31_20160107_MMP_150um_65mW_zoom2p2_000*[0-9].hdf5'):
    fnames.append(file)
fnames.sort()
print(fnames)
#%%
file_res = cb.utils.pre_preprocess_movie_labeling(client_[::2], fnames, median_filter_size=(2, 1, 1),
                                                  resize_factors=[.2, .1666666666], diameter_bilateral_blur=4)

#%%
client_.close()
cse.utilities.stop_server(is_slurm=True)

#%%

#%%
fold = os.path.split(os.path.split(fnames[0])[-2])[-1]
os.mkdir(fold)
#%%
files = glob.glob(fnames[0][:-20] + '*BL_compress_.tif')
files.sort()
print(files)
#%%
m = cb.load_movie_chain(files, fr=3)
m.play(backend='opencv', gain=10, fr=40)
#%%
m.save(files[0][:-20] + '_All_BL.tif')
#%%
files = glob.glob(fnames[0][:-20] + '*[0-9]._compress_.tif')
files.sort()
print(files)
#%%
m = cb.load_movie_chain(files, fr=3)
m.play(backend='opencv', gain=3, fr=40)
#%%
m.save(files[0][:-20] + '_All.tif')
