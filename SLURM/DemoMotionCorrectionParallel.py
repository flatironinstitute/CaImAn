# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 17:56:14 2016

@author: agiovann
"""
from __future__ import print_function

#%%
from builtins import range
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
#%% LOGIN TO MASTER NODE
# TYPE salloc -n n_nodes --exclusive
# source activate environment_name
#%%#%%
slurm_script = '/mnt/xfs1/home/agiovann/SOFTWARE/Constrained_NMF/SLURM/slurmStart.sh'
cse.utilities.start_server(ncpus=None, slurm_script=slurm_script)
#%%

#%%
import os
fnames = []
base_folder = '/mnt/ceph/users/agiovann/ImagingData/LABELLING/k31/20151223/'
for file in glob.glob(base_folder + 'k31_20151223_AM_150um_65mW_zoom2p2_00001_*.tif'):
    fnames.append(file)
fnames.sort()
print(fnames)

#%%
# np.maximum(psutil.cpu_count() - 2,1) # roughly number of cores on your machine minus 1
n_processes = 56

#%%
# low_SNR=False
# if low_SNR:
#    N=1000
#    mn1=m.copy().bilateral_blur_2D(diameter=5,sigmaColor=10000,sigmaSpace=0)
#
#    mn1,shifts,xcorrs, template=mn1.motion_correct()
#    mn2=mn1.apply_shifts(shifts)
#    #mn1=cb.movie(np.transpose(np.array(Y_n),[2,0,1]),fr=30)
#    mn=cb.concatenate([mn1,mn2],axis=1)
#    mn.play(gain=5.,magnification=4,backend='opencv',fr=30)
#%%
t1 = time()
file_res = cb.motion_correct_parallel(
    fnames, fr=30, template=None, margins_out=0, max_shift_w=45, max_shift_h=45, dview=dview, apply_smooth=True)
t2 = time() - t1
print(t2)
#%%
all_movs = []
for f in glob.glob(base_folder + '*.hdf5'):
    print(f)
    with np.load(f[:-4] + 'npz') as fl:
        #        pl.subplot(1,2,1)
        #        pl.imshow(fl['template'],cmap=pl.cm.gray)
        #        pl.subplot(1,2,2)
        #        pl.plot(fl['shifts'])
        all_movs.append(fl['template'][np.newaxis, :, :])
#        pl.pause(2)
#        pl.cla()
#%%
all_movs = cb.movie(np.concatenate(all_movs, axis=0), fr=10)
all_movs, shifts, corss, _ = all_movs.motion_correct(
    template=None, max_shift_w=45, max_shift_h=45)
#%%
template = np.median(all_movs[:], axis=0)
np.save(base_folder + 'template_total', template)
pl.imshow(template, cmap=pl.cm.gray, vmax=120)
#%%
all_movs.play(backend='opencv', gain=10, fr=10)
#%%
t1 = time()
file_res = cb.motion_correct_parallel(
    fnames, 30, template=template, margins_out=0, max_shift_w=45, max_shift_h=45, client=c, remove_blanks=False)
t2 = time() - t1
print(t2)
#%%
for f in file_res:
    with np.load(f + 'npz') as fl:
        pl.subplot(2, 2, 1)
        pl.imshow(fl['template'], cmap=pl.cm.gray, vmin=np.percentile(
            fl['template'], 1), vmax=np.percentile(fl['template'], 99))

        pl.subplot(2, 2, 3)
        pl.plot(fl['xcorrs'])
        pl.subplot(2, 2, 2)
        pl.plot(fl['shifts'])
        pl.pause(0.1)
        pl.cla()

print((time() - t1 - 200))
#%%
all_movs = []
for f in glob.glob(base_folder + '*.hdf5'):
    print(f)
    with np.load(f[:-4] + 'npz') as fl:
        #        pl.subplot(1,2,1)
        #        pl.imshow(fl['template'],cmap=pl.cm.gray)
        #        pl.subplot(1,2,2)
        #        pl.plot(fl['shifts'])
        all_movs.append(fl['template'][np.newaxis, :, :])
#        pl.pause(2)
#        pl.cla()

all_movs = cb.movie(np.concatenate(all_movs, axis=0), fr=10)
all_movs, shifts, corss, _ = all_movs.motion_correct(
    template=None, max_shift_w=45, max_shift_h=45)
all_movs.save(base_folder + 'avg_movies.tif')
template = np.median(all_movs[:], axis=0)
np.save(base_folder + 'template_total', template)
pl.imshow(template, cmap=pl.cm.gray, vmax=100)
#%%
all_movs.play(backend='opencv', gain=10, fr=5)
#%%
big_mov = []
big_shifts = []
fr_remove_init = 30
for f in fnames:
    with np.load(f[:-3] + 'npz') as fl:
        big_shifts.append(fl['shifts'])

    print(f)
    Yr = cb.load(f[:-3] + 'hdf5')[fr_remove_init:]
    Yr = Yr.resize(fx=1, fy=1, fz=.2)
    Yr = np.transpose(Yr, (1, 2, 0))
    d1, d2, T = Yr.shape
    Yr = np.reshape(Yr, (d1 * d2, T), order='F')
    print((Yr.shape))
#    np.save(fname[:-3]+'npy',np.asarray(Yr))
    big_mov.append(np.asarray(Yr))
#%%
big_mov = np.concatenate(big_mov, axis=-1)
big_shifts = np.concatenate(big_shifts, axis=0)
#%%
np.save('Yr_DS.npy', big_mov)
np.save('big_shifts.npy', big_shifts)

#%%
_, d1, d2 = np.shape(
    cb.load(fnames[0][:-3] + 'hdf5', subindices=list(range(3)), fr=10))
Yr = np.load('Yr_DS.npy', mmap_mode='r')
d, T = Yr.shape
Y = np.reshape(Yr, (d1, d2, T), order='F')
Y = cb.movie(np.array(np.transpose(Y, (2, 0, 1))), fr=30)
#%%
Y.play(backend='opencv', fr=30, gain=10, magnification=1)
#%%
cse.utilities.stop_server(is_slurm=True)
