#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Stripped demo for running the CNMF source extraction algorithm with CaImAn and
evaluation the components. The analysis can be run either in the whole FOV
or

For a complete pipeline (including motion correction) check demo_pipeline.py

Data courtesy of W. Yang, D. Peterka and R. Yuste (Columbia University)

This demo is designed to be run under spyder or jupyter; its plotting functions
are tailored for that environment.

@authors: @agiovann and @epnev

"""

from __future__ import print_function
from builtins import range
import cv2

try:
    cv2.setNumThreads(0)
except:
    pass

try:
    if __IPYTHON__:
        print("Detected iPython")
        get_ipython().magic('load_ext autoreload')
        get_ipython().magic('autoreload 2')
except NameError:
    pass

import numpy as np
import os
import glob
import matplotlib.pyplot as plt

import caiman as cm
from caiman.components_evaluation import estimate_components_quality_auto
from caiman.source_extraction.cnmf import cnmf as cnmf
from caiman.paths import caiman_datadir
#%% start a cluster

c, dview, n_processes =\
    cm.cluster.setup_cluster(backend='ipyparallel', n_processes=None,

                             single_thread=False)

#%% save files to be processed

# This datafile is distributed with Caiman
fnames = ['/Users/agiovann/Dropbox (Simons Foundation)/ImagingData/DataCalciumImaging/purkinje_out/corrected_movies/M_FLUO_1.tif']
fname_new  = '/Users/agiovann/example_movies_ALL/memmap__d1_131_d2_131_d3_1_order_C_frames_2991_.mmap' #fnames = ['/Users/agiovann/example_movies_ALL/2014-04-05-003.tif']

fnames = ['/Users/agiovann/example_movies_ALL/quietBlock_1_ds_5.hdf5']

# location of dataset  (can actually be a list of filed to be concatenated)
add_to_movie = -np.min(cm.load(fnames[0], subindices=range(200))).astype(float)
# determine minimum value on a small chunk of data
add_to_movie = np.maximum(add_to_movie, 0)
# if minimum is negative subtract to make the data non-negative
base_name = 'Yr'
name_new = cm.save_memmap_each(fnames, dview=dview, base_name=base_name,
                               add_to_movie=add_to_movie)
name_new.sort()
fname_new = cm.save_memmap_join(name_new, base_name='Yr', dview=dview)
#%% LOAD MEMORY MAPPABLE FILE
Yr, dims, T = cm.load_memmap(fname_new)
d1, d2 = dims
images = np.reshape(Yr.T, [T] + list(dims), order='F')

#%% play movie, press q to quit
play_movie = True
if play_movie:
    cm.movie(images).play(fr=50, magnification=1, gain=5.)

#%% correlation image. From here infer neuron size and density
Cn = cm.movie(images).local_correlations(swap_dim=False)
plt.cla()
plt.imshow(Cn, cmap='gray')
plt.title('Correlation Image')

#%% set up some parameters
is_patches = False      # flag for processing in patches or not

if is_patches:          # PROCESS IN PATCHES AND THEN COMBINE
    rf = [50,50]             # half size of each patch
    stride = [20,20]          # overlap between patches
    K = 100             # number of components in each patch
else:                   # PROCESS THE WHOLE FOV AT ONCE
    rf = None           # setting these parameters to None
    stride = None       # will run CNMF on the whole FOV
    K = 100              # number of neurons expected (in the whole FOV)

gSig = [6, 6]           # expected half size of neurons
merge_thresh = 0.80     # merging threshold, max correlation allowed
p = 1                   # order of the autoregressive system
gnb = 1                 # global background order
border_pix = 2

#%% Now RUN CNMF
cnm = cnmf.CNMF(n_processes, method_init='sparse_nmf', k=K, gSig=gSig,
                merge_thresh=merge_thresh, p=p, dview=dview, gnb=gnb,
                rf=rf, stride=stride, rolling_sum=False, alpha_snmf = 0, border_pix=border_pix)
cnm = cnm.fit(images)

#%% plot contour plots of components
plt.figure()
crd = cm.utils.visualization.plot_contours(cnm.A, Cn, thr=0.9)
plt.title('Contour plots of components')
#%%
cm.movie(np.reshape(Yr-cnm.A.dot(cnm.C) - cnm2.b.dot(cnm.f), dims + (-1,), order='F').transpose(2, 0, 1)).play(magnification=3, gain=10.)

#%%
A_in, C_in, b_in, f_in = cnm.A[:,:], cnm.C[:], cnm.b, cnm.f
cnm2 = cnmf.CNMF(n_processes=1, k=A_in.shape[-1], gSig=gSig, p=p, dview=dview,
                 merge_thresh=merge_thresh, Ain=A_in, Cin=C_in, b_in=b_in,
                 f_in=f_in, rf=None, stride=None, gnb=gnb,
                 method_deconvolution='oasis', check_nan=True,border_pix=border_pix)

cnm2 = cnm2.fit(images)
    #%% COMPONENT EVALUATION
# the components are evaluated in three ways:
#   a) the shape of each component must be correlated with the data
#   b) a minimum peak SNR is required over the length of a transient
#   c) each shape passes a CNN based classifier (this will pick up only neurons
#           and filter out active processes)
fr = 3             # approximate frame rate of data
decay_time = 1    # length of transient
min_SNR = 2.5       # peak SNR for accepted components (if above this, acept)
rval_thr = 0.90     # space correlation threshold (if above this, accept)
use_cnn = True     # use the CNN classifier
min_cnn_thr = 0.95  # if cnn classifier predicts below this value, reject

idx_components, idx_components_bad, SNR_comp, r_values, cnn_preds = \
    estimate_components_quality_auto(images, cnm2.A, cnm2.C, cnm2.b, cnm2.f,
                                     cnm2.YrA, fr, decay_time, gSig, dims,
                                     dview=dview, min_SNR=min_SNR,
                                     r_values_min=rval_thr, use_cnn=use_cnn,
                                     thresh_cnn_min=min_cnn_thr, thresh_cnn_lowest=0)
#%%
cm.utils.visualization.plot_contours(cnm2.A, Cn, thr=0.9)
#%% visualize selected and rejected components
plt.figure()
plt.subplot(1, 2, 1)
cm.utils.visualization.plot_contours(cnm2.A[:, idx_components], Cn, thr=0.9)
plt.title('Selected components')
plt.subplot(1, 2, 2)
plt.title('Discaded components')
cm.utils.visualization.plot_contours(cnm2.A[:, idx_components_bad], Cn, thr=0.9)

#%%
plt.figure()
crd = cm.utils.visualization.plot_contours(cnm2.A.tocsc()[:,idx_components], Cn, thr=0.9)
plt.title('Contour plots of components')
#%% visualize selected components
cm.utils.visualization.view_patches_bar(Yr, cnm2.A.tocsc()[:, idx_components],
                                        cnm2.C[idx_components, :], cnm2.b, cnm2.f,
                                        dims[0], dims[1],
                                        YrA=cnm2.YrA[idx_components, :], img=Cn)
#%% visualize selected components bad
cm.utils.visualization.view_patches_bar(Yr, cnm2.A.tocsc()[:, idx_components_bad],
                                        cnm2.C[idx_components_bad, :], cnm2.b, cnm2.f,
                                        dims[0], dims[1],
                                        YrA=cnm2.YrA[idx_components_bad, :], img=Cn)
#%%
cm.movie(np.reshape(cnm2.A.tocsc()[:, idx_components_bad].dot(
        cnm2.C[idx_components_bad]), dims + (-1,), order='F').transpose(2, 0, 1)).play(magnification=3, gain=10.)
#%%
cm.movie(np.reshape(cnm2.A.tocsc()[:, idx_components].dot(
        cnm2.C[idx_components]), dims + (-1,), order='F').transpose(2, 0, 1)).play(magnification=3, gain=10.)
#%%

#%%
cm.movie(np.reshape(Yr-cnm2.A.dot(cnm2.C) - cnm2.b.dot(cnm2.f), dims + (-1,), order='F').transpose(2, 0, 1)).play(magnification=3, gain=10.)
#%% STOP CLUSTER and clean up log files
cm.stop_server()

log_files = glob.glob('Yr*_LOG_*')
for log_file in log_files:
    os.remove(log_file)
