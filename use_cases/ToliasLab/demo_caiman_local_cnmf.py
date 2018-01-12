#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from builtins import str
from builtins import range
"""
Created on Wed Feb 24 18:39:45 2016

@author: Andrea Giovannucci

For explanation consult at https://github.com/agiovann/Constrained_NMF/releases/download/v0.4-alpha/Patch_demo.zip
and https://github.com/agiovann/Constrained_NMF

"""

try:
    if __IPYTHON__:
        print('Debugging!')
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
from caiman.source_extraction.cnmf import cnmf as cnmf
from caiman.components_evaluation import evaluate_components
from caiman.utils.visualization import plot_contours, view_patches_bar
from caiman.base.rois import extract_binary_masks_blob

#%%
c, dview, n_processes = cm.cluster.setup_cluster(
    backend='local', n_processes=None, single_thread=True)
#%%
is_patches = True
is_dendrites = True

if is_dendrites == True:
    # THIS METHOd CAN GIVE POSSIBLY INCONSISTENT RESULTS ON SOMAS WHEN NOT USED WITH PATCHES
    init_method = 'local_nmf'
    alpha_snmf = None  # this controls sparsity
else:
    init_method = 'greedy_roi'
    alpha_snmf = None  # 10e2  # this controls sparsity
#%%
m = cm.load('quietBlock.h5_at')
(m - np.min(m, 0)).save('example_movies/quiet_block_1000.tif')
#%% FOR LOADING ALL TIFF FILES IN A FILE AND SAVING THEM ON A SINGLE MEMORY MAPPABLE FILE
fnames = []
base_folder = './example_movies/'  # folder containing the demo files
for file in glob.glob(os.path.join(base_folder, '*.tif')):
    if file.endswith("1000.tif"):
        fnames.append(os.path.abspath(file))
fnames.sort()
if len(fnames) == 0:
    raise Exception("Could not find any file")

print(fnames)
fnames = fnames
#%%
# idx_x=slice(12,500,None)
# idx_y=slice(12,500,None)
# idx_xy=(idx_x,idx_y)

add_to_movie = 0  # the movie must be positive!!!
downsample_factor = 1  # use .2 or .1 if file is large and you want a quick answer
idx_xy = None
base_name = 'Yr'
name_new = cm.save_memmap_each(fnames, dview=dview, base_name=base_name, resize_fact=(
    1, 1, downsample_factor), remove_init=0, idx_xy=idx_xy, add_to_movie=add_to_movie)
name_new.sort()
print(name_new)
#%%
if len(name_new) > 1:
    fname_new = cm.save_memmap_join(
        name_new, base_name='Yr', n_chunks=12, dview=dview)
else:
    print('One file only, not saving!')
    fname_new = name_new[0]
#%%
# fname_new='Yr_d1_501_d2_398_d3_1_order_F_frames_369_.mmap'
Yr, dims, T = cm.load_memmap(fname_new)
d1, d2 = dims
images = np.reshape(Yr.T, [T] + list(dims), order='F')
Y = np.reshape(Yr, dims + (T,), order='F')
#%%
if np.min(images) < 0:
    raise Exception('Movie too negative, add_to_movie should be larger')
if np.sum(np.isnan(images)) > 0:
    raise Exception('Movie contains nan! You did not remove enough borders')
#%%
Cn = cm.local_correlations(Y[:, :, :1000])
pl.imshow(Cn, cmap='gray')

#%%
if not is_patches:
    #%%
    K = 35  # number of neurons expected per patch
    gSig = [7, 7]  # expected half size of neurons
    merge_thresh = 0.8  # merging threshold, max correlation allowed
    p = 2  # order of the autoregressive system
    cnm = cnmf.CNMF(n_processes, method_init=init_method, k=K, gSig=gSig, merge_thresh=merge_thresh,
                    p=p, dview=dview, Ain=None, method_deconvolution='oasis', skip_refinement=False)
    cnm = cnm.fit(images)
    crd = plot_contours(cnm.A, Cn, thr=0.9)
#%%
else:
    #%%
    rf = 100  # half-size of the patches in pixels. rf=25, patches are 50x50
    stride = 10  # amounpl.it of overlap between the patches in pixels
    K = None  # number of neurons expected per patch
    gSig = None  # expected half size of neurons
    merge_thresh = 0.8  # merging threshold, max correlation allowed
    p = 1  # order of the autoregressive system
    save_results = False
    #%% RUN ALGORITHM ON PATCHES
    options_local_NMF = {
        'NumCent': 400,
        # Define CNMF parameters
        'mbs': [10],  # temporal downsampling of data in intial phase of NMF
        'ds': 1,  # spatial downsampling of data in intial phase of NMF. Ccan be an integer or a list of the size of spatial dimensions
        # target sparsity range for spatial components
        'TargetAreaRatio': [0.01, 0.06],
        # repeats=1 # how many repeations to run NMF algorithm
        # 30 number of intial NMF iterations, in which we downsample data and add components
        'iters0': [5],
        'iters': 20,  # 100 number of main NMF iterations, in which we fine tune the components on the full data
        # l1 regularization parameter initialization (for increased sparsity). If zero, we have no l1 sparsity penalty
        'lam1_s': 10,
        # intialize of background shape at this percentile (over time) of video
        'bkg_per': 0.1,
        # merge components if activity is correlated above the this threshold (and sufficiently close)
        'MergeThreshold_activity': 0.85,
        # merge components if activity is correlated above the this threshold (and sufficiently close)
        'MergeThreshold_shapes': 0.99,
        'Connected': True,  # should we constrain all spatial component to be connected?
        'SigmaMask': 3  # if not [], then update masks so that they are non-zero a radius of SigmaMasks around previous non-zero support of shapes
    }

    cnm = cnmf.CNMF(n_processes, k=K, gSig=gSig, ssub=1, tsub=10, p_ssub=1, p_tsub=10, merge_thresh=0.8, p=0, dview=dview, Ain=None, rf=rf, stride=stride, memory_fact=1,
                    method_init=init_method, alpha_snmf=alpha_snmf, only_init_patch=True, gnb=1, method_deconvolution='oasis', normalize_init=False, options_local_NMF=options_local_NMF)
    cnm = cnm.fit(images)

    A_tot = cnm.A
    C_tot = cnm.C
    YrA_tot = cnm.YrA
    b_tot = cnm.b
    f_tot = cnm.f
    sn_tot = cnm.sn

    print(('Number of components:' + str(A_tot.shape[-1])))
    #%%
    pl.figure()
    crd = plot_contours(A_tot, Cn, thr=0.9)
    #%%
#    final_frate = 10# approx final rate  (after eventual downsampling )
#    Npeaks = 10
#    traces = C_tot + YrA_tot
#    #        traces_a=traces-scipy.ndimage.percentile_filter(traces,8,size=[1,np.shape(traces)[-1]/5])
#    #        traces_b=np.diff(traces,axis=1)
#    fitness_raw, fitness_delta, erfc_raw, erfc_delta, r_values, significant_samples = evaluate_components(
#        Y, traces, A_tot, C_tot, b_tot, f_tot, final_frate, remove_baseline=True, N=5, robust_std=False, Athresh=0.1, Npeaks=Npeaks,  thresh_C=0.3)
#
#    idx_components_r = np.where(r_values >= .5)[0]
#    idx_components_raw = np.where(fitness_raw < -40)[0]
#    idx_components_delta = np.where(fitness_delta < -20)[0]
#
#    idx_components = np.union1d(idx_components_r, idx_components_raw)
#    idx_components = np.union1d(idx_components, idx_components_delta)
#    idx_components_bad = np.setdiff1d(list(range(len(traces))), idx_components)
#
#    print(('Keeping ' + str(len(idx_components)) +
#           ' and discarding  ' + str(len(idx_components_bad))))
#    #%%
#    pl.figure()
#    crd = plot_contours(A_tot.tocsc()[:, idx_components], Cn, thr=0.9)
#    #%%
#    A_tot = A_tot.tocsc()[:, idx_components]
#    C_tot = C_tot[idx_components]
#    #%%
#    save_results = True
#    if save_results:
#        np.savez('results_analysis_patch.npz', A_tot=A_tot, C_tot=C_tot,
#                 YrA_tot=YrA_tot, sn_tot=sn_tot, d1=d1, d2=d2, b_tot=b_tot, f=f_tot)
#
#    #%%
#    cnm = cnmf.CNMF(n_processes, k=A_tot.shape, gSig=gSig, merge_thresh=merge_thresh, p=p, dview=dview, Ain=A_tot, Cin=C_tot,
#                    f_in=f_tot, rf=None, stride=None, method_deconvolution='oasis')
#    cnm = cnm.fit(images)
#
# %%
#A, C, b, f, YrA, sn = cnm.A, cnm.C, cnm.b, cnm.f, cnm.YrA, cnm.sn
# %%
#final_frate = 10
#
#Npeaks = 10
#traces = C + YrA
# traces_a=traces-scipy.ndimage.percentile_filter(traces,8,size=[1,np.shape(traces)[-1]/5])
# traces_b=np.diff(traces,axis=1)
# fitness_raw, fitness_delta, erfc_raw, erfc_delta, r_values, significant_samples = \
#    evaluate_components(Y, traces, A, C, b, f, final_frate, remove_baseline=True,
#                                      N=5, robust_std=False, Athresh=0.1, Npeaks=Npeaks,  thresh_C=0.3)
#
#idx_components_r = np.where(r_values >= .95)[0]
#idx_components_raw = np.where(fitness_raw < -100)[0]
#idx_components_delta = np.where(fitness_delta < -100)[0]
#
#
##min_radius = gSig[0] - 2
# masks_ws, idx_blobs, idx_non_blobs = extract_binary_masks_blob(
# A.tocsc(), min_radius, dims, num_std_threshold=1,
# minCircularity=0.7, minInertiaRatio=0.2, minConvexity=.5)
#
#idx_components = np.union1d(idx_components_r, idx_components_raw)
#idx_components = np.union1d(idx_components, idx_components_delta)
##idx_blobs = np.intersect1d(idx_components, idx_blobs)
#idx_components_bad = np.setdiff1d(list(range(len(traces))), idx_components)
#
#print(' ***** ')
# print((len(traces)))
# print((len(idx_components)))
# print((len(idx_blobs)))
# %%
#save_results = True
# if save_results:
#    np.savez(os.path.join(os.path.split(fname_new)[0], 'results_analysis.npz'), Cn=Cn, A=A.todense(), C=C, b=b, f=f, YrA=YrA, sn=sn, d1=d1, d2=d2, idx_components=idx_components, idx_components_bad=idx_components_bad)
#
# %% visualize components
# pl.figure();
#pl.subplot(1, 2, 1)
#crd = plot_contours(A.tocsc()[:, idx_components], Cn, thr=0.9)
##pl.subplot(1, 3, 2)
##crd = plot_contours(A.tocsc()[:, idx_blobs], Cn, thr=0.9)
#pl.subplot(1, 2, 2)
#crd = plot_contours(A.tocsc()[:, idx_components_bad], Cn, thr=0.9)
# %%
# view_patches_bar(Yr, scipy.sparse.coo_matrix(A.tocsc()[:, idx_components]), C[
#                               idx_components, :], b, f, dims[0], dims[1], YrA=YrA[idx_components, :], img=Cn)
# %%
# view_patches_bar(Yr, scipy.sparse.coo_matrix(A.tocsc()[:, idx_components_bad]), C[
#                               idx_components_bad, :], b, f, dims[0], dims[1], YrA=YrA[idx_components_bad, :], img=Cn)
# %% STOP CLUSTER and clean up log files


cm.stop_server()

log_files = glob.glob('Yr*_LOG_*')
for log_file in log_files:
    os.remove(log_file)
