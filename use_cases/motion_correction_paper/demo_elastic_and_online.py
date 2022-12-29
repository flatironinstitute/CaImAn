#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on Sat Nov 19 14:29:15 2016

@author: agiovann
"""

from __future__ import print_function

from builtins import zip
from builtins import str

try:
    if __IPYTHON__:
        # this is used for debugging purposes only. allows to reload classes when changed
        get_ipython().magic('load_ext autoreload')
        get_ipython().magic('autoreload 2')
except NameError:
    print('Not launched under iPython')

import sys
import numpy as np
import psutil
import glob
import os
import scipy
from ipyparallel import Client
# mpl.use('Qt5Agg')


import pylab as pl
pl.ion()
#%%
import caiman as cm
from caiman.components_evaluation import evaluate_components
from caiman.utils.visualization import plot_contours, view_patches_bar
from caiman.base.rois import extract_binary_masks_blob
from caiman.source_extraction import cnmf as cnmf
from caiman.motion_correction import motion_correct_online
from caiman.cluster import apply_to_patch
#%%
# backend='SLURM'
backend = 'local'
if backend == 'SLURM':
    n_processes = int(os.environ.get('SLURM_NPROCS'))
else:
    # roughly number of cores on your machine minus 1
    n_processes = np.maximum(int(psutil.cpu_count()), 1)
print(('using ' + str(n_processes) + ' processes'))
#%% start cluster for efficient computation
single_thread = False

if single_thread:
    dview = None
else:
    try:
        c.close()
    except:
        print('C was not existing, creating one')
    print("Stopping  cluster to avoid unnencessary use of memory....")
    sys.stdout.flush()
    if backend == 'SLURM':
        try:
            cm.stop_server(is_slurm=True)
        except:
            print('Nothing to stop')
        slurm_script = '/mnt/xfs1/home/agiovann/SOFTWARE/Constrained_NMF/SLURM/slurmStart.sh'
        cm.start_server(slurm_script=slurm_script)
        pdir, profile = os.environ['IPPPDIR'], os.environ['IPPPROFILE']
        c = Client(ipython_dir=pdir, profile=profile)
    else:
        cm.stop_server()
        cm.start_server()
        c = Client()

    print(('Using ' + str(len(c)) + ' processes'))
    dview = c[:len(c)]
#%%
# idx_x=slice(12,500,None)
# idx_y=slice(12,500,None)
# idx_xy=(idx_x,idx_y)
add_to_movie = 100  # the movie must be positive!!!
border_to_0 = 0
downsample_factor = 1  # use .2 or .1 if file is large and you want a quick answer
idx_xy = None
base_name = 'Yr'
#%%
#name_new=cm.save_memmap_each(['M_FLUO_t.tif'], dview=dview,base_name=base_name, resize_fact=(1, 1, downsample_factor), remove_init=0,idx_xy=idx_xy,add_to_movie=add_to_movie,border_to_0=border_to_0)
name_new = cm.save_memmap_each(['M_FLUO_t.tif'], dview=dview, base_name=base_name, resize_fact=(
    1, 1, downsample_factor), remove_init=0, idx_xy=idx_xy, add_to_movie=add_to_movie, border_to_0=border_to_0)
ame_new.sort()
print(name_new)
#%%
# Yr,dim,T=cm.load_memmap('Yr0_d1_64_d2_128_d3_1_order_C_frames_6764_.mmap')
Yr, dim, T = cm.load_memmap('Yr0_d1_512_d2_512_d3_1_order_C_frames_1076_.mmap')

res, idfl, shape_grid = apply_to_patch(Yr, (T,) + dim, None, dim[0], 8, motion_correct_online, 200, max_shift_w=15, max_shift_h=15,
                                       save_base_name='test_mmap',  init_frames_template=100, show_movie=False, remove_blanks=True, n_iter=2, show_template=False)
#%%
[pl.plot(np.array(r[0][-2])) for r in res]
[pl.plot(np.array(r[0][-1])) for r in res]

#%%
[pl.plot(np.array(r[1][-2])) for r in res]
[pl.plot(np.array(r[1][-1])) for r in res]
#%%
pl.imshow(res[0][2], cmap='gray', vmin=300, vmax=400)
#%%
# mr,dim_r,T=cm.load_memmap('test_mmap_d1_48_d2_114_d3_1_order_C_frames_6764_.mmap')
#m = cm.load('test_mmap_d1_48_d2_114_d3_1_order_C_frames_6764_.mmap')
mr, dim_r, T = cm.load_memmap(
    'test_mmap_d1_509_d2_504_d3_1_order_C_frames_1076_.mmap')
m = cm.load('test_mmap_d1_509_d2_504_d3_1_order_C_frames_1076_.mmap')
#%%
res_p, idfl, shape_grid = apply_to_patch(mr, (T,) + dim_r, dview, 24, 8, motion_correct_online, 0,
                                         max_shift_w=4, max_shift_h=4, save_base_name=None,  init_frames_template=100,
                                         show_movie=False, remove_blanks=False, n_iter=2, return_mov=True, use_median_as_template=True)
#%% video
res_p = apply_to_patch(mr, (T,) + dim_r, None, 48, 8, motion_correct_online, 0, show_template=True,
                       max_shift_w=3, max_shift_h=3, save_base_name=None,  init_frames_template=100,
                       show_movie=False, remove_blanks=False, n_iter=2, return_mov=True, use_median_as_template=True)

#%%

for idx, r in enumerate(res_p):
    pl.subplot(shape_grid[0], shape_grid[1], idx + 1)
#    pl.plot(np.reshape(np.ar4ray(r[0][0]).T,(4,-1)).T)
#    pl.plot(np.array(r[0][-2]))
#    idx_good = np.where(np.array(r[1][-1])>.15)[0]
#    pl.plot(np.array(r[0][-1])[idx_good])
    pl.plot(np.array(r[0][-1]))

#%%
for idx, r in enumerate(res_p):
    pl.subplot(shape_grid[0], shape_grid[1], idx + 1)
#    pl.plot(np.reshape(np.ar4ray(r[0][0]).T,(4,-1)).T)
#    pl.plot(np.array(r[0][-2]))
    pl.plot(np.array(r[1][-1]))
#%%
for idx, r in enumerate(res_p):
    pl.subplot(shape_grid[0], shape_grid[1], idx + 1)
    img = r[2]
    lq, hq = np.percentile(img, [10, 95])
    pl.imshow(img, cmap='gray', vmin=lq, vmax=hq)
# %%
#
# def cart2pol(x, y):
#    rho = np.sqrt(x**2 + y**2)
#    phi = np.arctan2(y, x)
#    return(rho, phi)
#
# def pol2cart(rho, phi):
#    x = rho * np.cos(phi)
#    y = rho * np.sin(phi)
#    return(x, y)
#%% find center of mass
movie_shifts_x = np.zeros((T,) + dim_r)
movie_shifts_y = np.zeros((T,) + dim_r)


for r, idx_mat in zip(res_p, idfl):
    img_temp = np.zeros(np.prod(dim_r))
    img_temp[idx_mat] = 1
    img_temp = np.reshape(img_temp, dim_r, order='F')
#    pl.imshow(img_temp)
    x1, x2 = np.round(scipy.ndimage.center_of_mass(img_temp)).astype(int)
    print((x1, x2))
    movie_shifts_x[:, x1, x2] = np.array(r[0][-1])[:, 0]
    movie_shifts_y[:, x1, x2] = np.array(r[0][-1])[:, 1]

#%%
pl.close()
mn = np.mean(m, 0)
pl.imshow(mn)

for imm_x, imm_y in zip(movie_shifts_x, movie_shifts_y):
    y, x = np.where((imm_x != 0) | (imm_y != 0))

    pl.cla()
    pl.imshow(mn)
    pl.quiver(x, y,  np.array(imm_y[y, x]), np.array(
        imm_x[y, x]), color='w', angles='xy', scale_units='xy', scale=.1)
#    pl.xlim([0, dim_r[1]])
#    pl.ylim([0, dim_r[0]])
    pl.pause(.01)


#%% final movie shifts size
np.multiply(shape_grid, 2)
movie_shifts_x = np.zeros(shape_grid + (T,))
movie_shifts_y = np.zeros(shape_grid + (T,))

for idx, r in enumerate(res_p):
    x1, x2 = np.unravel_index(idx, shape_grid)
    movie_shifts_x[x1, x2, :] = np.array(r[0][-1])[:, 0]
#%%
for idx, r in enumerate(res_p):
    mm = cm.movie(r[-1])
    mm.play(fr=100, magnification=7, gain=2.)
#%%

#%%
imgtot = np.zeros(np.prod(dim_r))
for idx, r in enumerate(res_p):
    #    pl.subplot(2,5,idx+1)
    img = r[0][2]
#    lq,hq = np.percentile(img,[10,90])
#    pl.imshow(img,cmap='gray',vmin=lq,vmax=hq)
    imgtot[r[1]] = np.maximum(img.T.flatten(), imgtot[r[1]])

m = cm.load('test_mmap_d1_49_d2_114_d3_1_order_C_frames_6764_.mmap')

lq, hq = np.percentile(imgtot, [1, 97])
pl.subplot(2, 1, 1)
pl.imshow(np.reshape(imgtot, dim_r, order='F'), cmap='gray',
          vmin=lq, vmax=hq, interpolation='none')
pl.subplot(2, 1, 2)
lq, hq = np.percentile(res[0][0][2], [10, 99])
pl.imshow(np.mean(m, 0), cmap='gray', vmin=lq, vmax=hq, interpolation='none')
