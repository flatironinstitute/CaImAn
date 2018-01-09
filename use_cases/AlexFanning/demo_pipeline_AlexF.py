#!/usr/bin/env python

# @package demos
#\brief      for the user/programmer to understand and try the code
#\details    all of other usefull functions (demos available on jupyter notebook) -*- coding: utf-8 -*-
#\version   1.0
#\pre       EXample.First initialize the system.
#\bug
#\warning
#\copyright GNU General Public License v2.0
#\date Created on Mon Nov 21 15:53:15 2016
#\author agiovann
# toclean

from __future__ import division
from __future__ import print_function
from builtins import zip
from builtins import str
from builtins import map
from builtins import range
from past.utils import old_div
import cv2
import glob

try:
    cv2.setNumThreads(1)
except:
    print('Open CV is naturally single threaded')

try:
    if __IPYTHON__:
        print(1)
        # this is used for debugging purposes only. allows to reload classes
        # when changed
        get_ipython().magic('load_ext autoreload')
        get_ipython().magic('autoreload 2')
except NameError:
    print('Not IPYTHON')

import caiman as cm
import numpy as np
import os
import time
import pylab as pl
import psutil
import sys
from ipyparallel import Client
from skimage.external.tifffile import TiffFile
import scipy
import copy

from caiman.utils.utils import download_demo
from caiman.base.rois import extract_binary_masks_blob
from caiman.utils.visualization import plot_contours, view_patches_bar
from caiman.source_extraction.cnmf import cnmf as cnmf
from caiman.motion_correction import MotionCorrect
from caiman.components_evaluation import estimate_components_quality

from caiman.components_evaluation import evaluate_components, evaluate_components_CNN

from caiman.tests.comparison import comparison
from caiman.motion_correction import tile_and_correct, motion_correction_piecewise
#%%
params_movie = {'fname': ['/mnt/ceph/neuro/ImagingData/AlexFanning/Concatenated0711d7b.tif'],
                'max_shifts': (3, 3),  # maximum allow rigid shift (2,2)
                'niter_rig': 1,
                'splits_rig': 14,  # for parallelization split the movies in  num_splits chuncks across time
                # if none all the splits are processed and the movie is saved
                'num_splits_to_process_rig': None,
                'p': 1,  # order of the autoregressive system
                'merge_thresh': 0.8,  # merging threshold, max correlation allow
                'rf': 14,  # half-size of the patches in pixels. rf=25, patches are 50x50    20
                'stride_cnmf': 10,  # amounpl.it of overlap between the patches in pixels
                'K': 6,  # number of components per patch
                # if dendritic. In this case you need to set init_method to sparse_nmf
                'is_dendrites': True,
                'init_method': 'sparse_nmf',
                'gSig': [5, 5],  # expected half size of neurons
                'alpha_snmf': 1e-6,  # this controls sparsity
                'final_frate': 10,
                'r_values_min_patch': .7,  # threshold on space consistency
                'fitness_min_patch': -20,  # threshold on time variability
                # threshold on time variability (if nonsparse activity)
                'fitness_delta_min_patch': -20,
                'Npeaks': 10,
                'r_values_min_full': .8,
                'fitness_min_full': - 40,
                'fitness_delta_min_full': - 40,
                'only_init_patch': True,
                'gnb': 1,
                'memory_fact': 1,
                'n_chunks': 10,
                # whether to update the background components in the spatial phase
                'update_background_components': True,
                'low_rank_background': True  # whether to update the using a low rank approximation. In the False case all the nonzero elements of the background components are updated using hals
                #(to be used with one background per patch)
                }

#%%
params_display = {
    'downsample_ratio': .2,
    'thr_plot': 0.8
}
# TODO: do find&replace on those parameters and delete this paragrph

# @params fname name of the movie
fname = params_movie['fname']
niter_rig = params_movie['niter_rig']
# @params max_shifts maximum allow rigid shift
max_shifts = params_movie['max_shifts']

# @params splits_rig for parallelization split the movies in  num_splits chuncks across time
splits_rig = params_movie['splits_rig']

# @params num_splits_to_process_ri if none all the splits are processed and the movie is saved
num_splits_to_process_rig = params_movie['num_splits_to_process_rig']


# %% download movie if not there
channel = 0
m_orig = cm.load_movie_chain(fname[:1], channel=(
    slice(None), channel, slice(None), slice(None)))
m_orig.save(fname[0][:-4] + '_channel_' + str(channel) + '.tif')
fname[0] = fname[0][:-4] + '_channel_' + str(channel) + '.tif'
# %% play movie
downsample_ratio = params_display['downsample_ratio']
offset_mov = -np.min(m_orig[:100])
m_orig.resize(1, 1, downsample_ratio).play(
    gain=10, offset=offset_mov, fr=30, magnification=5)

# %% RUN ANALYSIS
c, dview, n_processes = cm.cluster.setup_cluster(
    backend='local', n_processes=None, single_thread=False)

# %% INITIALIZING
t1 = time.time()
# movie must be mostly positive for this to work
# TODO : document
# setting timer to see how the changement in functions make the code react on a same computer.

min_mov = cm.load(fname[0], subindices=range(400)).min()
mc_list = []
new_templ = None
for each_file in fname:
    # TODO: needinfo how the classes works
    for i in range(2):
        mc = MotionCorrect(each_file, min_mov,
                           dview=dview, max_shifts=max_shifts, niter_rig=niter_rig, splits_rig=splits_rig,
                           num_splits_to_process_rig=num_splits_to_process_rig,
                           strides=None, overlaps=None, splits_els=None,
                           num_splits_to_process_els=None,
                           upsample_factor_grid=None, max_deviation_rigid=None,
                           shifts_opencv=True, nonneg_movie=True)
        mc.motion_correct_rigid(save_movie=True, template=new_templ)
        new_templ = mc.total_template_rig
        m_rig = cm.load(mc.fname_tot_rig)
        bord_px_rig = np.ceil(np.max(mc.shifts_rig)).astype(np.int)

    # TODO : needinfo
        pl.imshow(new_templ, cmap='gray')
        pl.pause(.1)

    mc_list.append(mc)
# we are going to keep this part because it helps the user understand what we need.
# needhelp why it is not the same as in the notebooks ?
# TODO: show screenshot 2,3

# %%
# load motion corrected movie
m_rig = cm.load(mc.fname_tot_rig)
pl.imshow(mc.total_template_rig, cmap='gray')
# %% visualize templates
cm.movie(np.array(mc.templates_rig)).play(
    fr=10, gain=5, magnification=5, offset=offset_mov)
# %% plot rigid shifts
pl.close()
pl.plot(mc.shifts_rig)
pl.legend(['x shifts', 'y shifts'])
pl.xlabel('frames')
pl.ylabel('pixels')
# %% inspect movie
downsample_ratio = params_display['downsample_ratio']
# TODO: todocument
offset_mov = -np.min(m_orig[:100])
m_rig.resize(1, 1, downsample_ratio).play(
    gain=10, offset=offset_mov * .25, fr=30, magnification=2, bord_px=bord_px_rig)


# %% visualize raw, rigid and pw-rigid motion correted moviews
downsample_factor = params_display['downsample_ratio']
# TODO : todocument
cm.concatenate(
    [m_orig.resize(1, 1, downsample_factor) + offset_mov, m_rig.resize(1, 1, downsample_factor)], axis=2).play(fr=60, gain=5, magnification=4, offset=0)
# TODO: show screenshot 8
# %% restart cluster to clean up memory
# TODO: todocument
c, dview, n_processes = cm.cluster.setup_cluster(
    backend='local', n_processes=None, single_thread=False)
# %% save each chunk in F format
t1 = time.time()
if not 'max_shifts' in params_movie:
    fnames = params_movie['fname']
    border_to_0 = 0
else:  # elif not params_movie.has_key('overlaps'):
    fnames = [mc.fname_tot_rig]
    border_to_0 = bord_px_rig
    m_els = m_rig
# else:
#   fnames = [mc.fname_tot_els]
#  border_to_0 = bord_px_els

# if you need to crop the borders use slicing
# idx_x=slice(border_nan,-border_nan,None)
# idx_y=slice(border_nan,-border_nan,None)
# idx_xy=(idx_x,idx_y)
idx_xy = None
# TODO: needinfo
add_to_movie = -np.nanmin(m_els) + 1  # movie must be positive
# if you need to remove frames from the beginning of each file
remove_init = 0
# downsample movie in time: use .2 or .1 if file is large and you want a quick answer
downsample_factor = 1
base_name = fname[0].split('/')[-1][:-4]
# TODO: todocument
name_new = cm.save_memmap_each(fnames, dview=dview, base_name=base_name, resize_fact=(
    1, 1, downsample_factor), remove_init=remove_init, idx_xy=idx_xy, add_to_movie=add_to_movie,
    border_to_0=border_to_0)
name_new.sort()
print(name_new)

# %% if multiple files were saved in C format, now put them together in a single large file.
if len(name_new) > 1:
    fname_new = cm.save_memmap_join(

        name_new, base_name='Yr', n_chunks=params_movie['n_chunks'], dview=dview)
else:
    print('One file only, not saving!')
    fname_new = name_new[0]

# %% LOAD MEMMAP FILE
# fname_new='Yr_d1_501_d2_398_d3_1_order_F_frames_369_.mmap'
Yr, dims, T = cm.load_memmap(fname_new)
d1, d2 = dims
images = np.reshape(Yr.T, [T] + list(dims), order='F')
# TODO: needinfo
Y = np.reshape(Yr, dims + (T,), order='F')
m_images = cm.movie(images)

# TODO: show screenshot 10
# %%  checks on movies
# computationnally intensive
if np.min(images) < 0:
    # TODO: should do this in an automatic fashion with a while loop at the 367 line
    raise Exception('Movie too negative, add_to_movie should be larger')
if np.sum(np.isnan(images)) > 0:
    # TODO: same here
    raise Exception('Movie contains nan! You did not remove enough borders')
# %% correlation image
# TODO: needinfo it is not the same and not used
# for fff in fname_new:
#    Cn = cm.movie(images[:1000]).local_correlations(eight_neighbours=True,swap_dim=True)
#    #Cn[np.isnan(Cn)] = 0
#    pl.imshow(Cn, cmap='gray', vmax=.35)
# %% correlation image
Cn = cm.local_correlations(Y)
Cn[np.isnan(Cn)] = 0
pl.imshow(Cn, cmap='gray', vmax=.95)
# TODO: show screenshot 11

# %% some parameter settings
# order of the autoregressive fit to calcium imaging in general one (slow gcamps) or two (fast gcamps fast scanning)
p = params_movie['p']
# merging threshold, max correlation allowed
merge_thresh = params_movie['merge_thresh']
# half-size of the patches in pixels. rf=25, patches are 50x50
rf = params_movie['rf']
# amounpl.it of overlap between the patches in pixels
stride_cnmf = params_movie['stride_cnmf']
# number of components per patch
K = params_movie['K']
# if dendritic. In this case you need to set init_method to sparse_nmf
is_dendrites = params_movie['is_dendrites']
# iinit method can be greedy_roi for round shapes or sparse_nmf for denritic data
init_method = params_movie['init_method']
# expected half size of neurons
gSig = params_movie['gSig']
# this controls sparsity
alpha_snmf = params_movie['alpha_snmf']
# frame rate of movie (even considering eventual downsampling)
final_frate = params_movie['final_frate']

if params_movie['is_dendrites'] == True:
    if params_movie['init_method'] is not 'sparse_nmf':
        raise Exception('dendritic requires sparse_nmf')
    if params_movie['alpha_snmf'] is None:
        raise Exception('need to set a value for alpha_snmf')
# %% Extract spatial and temporal components on patches
t1 = time.time()
border_pix = 4  # if motion correction introduces problems at the border remove pixels from border
# TODO: todocument
# TODO: warnings 3
cnm = cnmf.CNMF(n_processes=1, k=K, gSig=gSig, merge_thresh=params_movie['merge_thresh'], p=params_movie['p'],
                dview=dview, rf=rf, stride=stride_cnmf, memory_fact=1,
                method_init=init_method, alpha_snmf=alpha_snmf, only_init_patch=params_movie[
                    'only_init_patch'],
                gnb=params_movie['gnb'], method_deconvolution='oasis', border_pix=border_pix, low_rank_background=params_movie['low_rank_background'])
cnm = cnm.fit(images)

A_tot = cnm.A
C_tot = cnm.C
YrA_tot = cnm.YrA
b_tot = cnm.b
f_tot = cnm.f
sn_tot = cnm.sn
print(('Number of components:' + str(A_tot.shape[-1])))
# %%
pl.figure()

crd = plot_contours(A_tot, Cn, thr=params_display['thr_plot'])
# %% DISCARD LOW QUALITY COMPONENT
final_frate = params_movie['final_frate']
# threshold on space consistency
r_values_min = params_movie['r_values_min_patch']
# threshold on time variability
fitness_min = params_movie['fitness_delta_min_patch']
# threshold on time variability (if nonsparse activity)
fitness_delta_min = params_movie['fitness_delta_min_patch']
Npeaks = params_movie['Npeaks']
traces = C_tot + YrA_tot
# TODO: todocument
idx_components, idx_components_bad = estimate_components_quality(
    traces, Y, A_tot, C_tot, b_tot, f_tot, final_frate=final_frate, Npeaks=Npeaks, r_values_min=r_values_min,
    fitness_min=fitness_min, fitness_delta_min=fitness_delta_min)
print(('Keeping ' + str(len(idx_components)) +
       ' and discarding  ' + str(len(idx_components_bad))))
# %%
# TODO: show screenshot 13
pl.figure()
crd = plot_contours(
    A_tot.tocsc()[:, idx_components], Cn, thr=params_display['thr_plot'])
# %%
A_tot = A_tot.tocsc()[:, idx_components]
C_tot = C_tot[idx_components]
# %% rerun updating the components to refine
t1 = time.time()
cnm = cnmf.CNMF(n_processes=1, k=A_tot.shape, gSig=gSig, merge_thresh=merge_thresh, p=p, dview=dview, Ain=A_tot,
                Cin=C_tot, b_in=b_tot,
                f_in=f_tot, rf=None, stride=None, method_deconvolution='oasis', gnb=params_movie['gnb'],
                low_rank_background=params_movie['low_rank_background'], update_background_components=params_movie['update_background_components'], check_nan=True)

cnm = cnm.fit(images)

# %%
A, C, b, f, YrA, sn = cnm.A, cnm.C, cnm.b, cnm.f, cnm.YrA, cnm.sn
# %% again recheck quality of components, stricter criteria
final_frate = params_movie['final_frate']
# threshold on space consistency
r_values_min = params_movie['r_values_min_full']
fitness_min = params_movie['fitness_min_full']  # threshold on time variability
# threshold on time variability (if nonsparse activity)
fitness_delta_min = params_movie['fitness_delta_min_full']
Npeaks = params_movie['Npeaks']
traces = C + YrA
idx_components, idx_components_bad, fitness_raw, fitness_delta, r_values = estimate_components_quality(
    traces, Y, A, C, b, f, final_frate=final_frate, Npeaks=Npeaks, r_values_min=r_values_min, fitness_min=fitness_min,
    fitness_delta_min=fitness_delta_min, return_all=True)
print(' ***** ')
print((len(traces)))
print((len(idx_components)))
# %% save results
np.savez(os.path.join(os.path.split(fname_new)[0], os.path.split(fname_new)[1][:-4] + 'results_analysis.npz'), Cn=Cn,
         A=A,
         C=C, b=b, f=f, YrA=YrA, sn=sn, d1=d1, d2=d2, idx_components=idx_components,
         idx_components_bad=idx_components_bad,
         fitness_raw=fitness_raw, fitness_delta=fitness_delta, r_values=r_values)
# we save it
# %%
# TODO: show screenshot 14
pl.subplot(1, 2, 1)
crd = plot_contours(A.tocsc()[:, idx_components],
                    Cn, thr=params_display['thr_plot'])
pl.subplot(1, 2, 2)
crd = plot_contours(A.tocsc()[:, idx_components_bad],
                    Cn, thr=params_display['thr_plot'])
# %%
# TODO: needinfo
view_patches_bar(Yr, scipy.sparse.coo_matrix(A.tocsc()[:, idx_components]), C[idx_components, :], b, f, dims[0], dims[1],
                 YrA=YrA[idx_components, :], img=Cn)
# %%
view_patches_bar(Yr, scipy.sparse.coo_matrix(A.tocsc()[:, idx_components_bad]), C[idx_components_bad, :], b, f, dims[0],
                 dims[1], YrA=YrA[idx_components_bad, :], img=Cn)
# %% STOP CLUSTER and clean up log files
# TODO: todocument
cm.stop_server()

log_files = glob.glob('*_LOG_*')
for log_file in log_files:
    os.remove(log_file)
# %% reconstruct denoised movie
denoised = cm.movie(A.dot(C) + b.dot(f)).reshape(dims +
                                                 (-1,), order='F').transpose([2, 0, 1])
# %%
# TODO: show screenshot 15
denoised.play(gain=10, offset=0, fr=50, magnification=4)
#%% background only
denoised = cm.movie(b.dot(f)).reshape(
    dims + (-1,), order='F').transpose([2, 0, 1])
denoised.play(gain=5, offset=0, fr=20, magnification=4)


# %% reconstruct denoised movie without background
denoised = cm.movie(A.dot(C)).reshape(
    dims + (-1,), order='F').transpose([2, 0, 1])
# %%
# TODO: show screenshot 16
denoised.play(gain=10, offset=0, fr=100, magnification=5)
#%% show background(s)
BB = cm.movie(b.reshape(dims + (-1,), order='F').transpose(2, 0, 1))
BB.play(gain=2, offset=0, fr=2, magnification=4)
BB.zproject()

#%% *************************************************************************************************************************#%%

#%% *************************************************************************************************************************#%%

#%% *************************************************************************************************************************#%%

#%% *************************************************************************************************************************#%%

#%% *************************************************************************************************************************#%%

#%% RELOAD DATA
params_display = {
    'downsample_ratio': .2,
    'thr_plot': 0.8
}

analysis_file = os.path.join(os.path.split(fname_new)[0], os.path.split(
    fname_new)[1][:-4] + 'results_analysis.npz')
with np.load(analysis_file) as ld:
    print(ld.keys())
    locals().update(ld)
    dims_off = d1, d2
    A = scipy.sparse.coo_matrix(A[()])
    dims = (d1, d2)
