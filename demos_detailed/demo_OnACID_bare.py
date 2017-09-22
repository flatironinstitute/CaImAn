#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 17:52:23 2017

@author: jfriedrich
"""

import numpy as np
import pylab as pl
import caiman as cm
from caiman.source_extraction import cnmf as cnmf
from caiman.utils.visualization import view_patches_bar, plot_contours
from copy import deepcopy
from caiman.source_extraction.cnmf.online_cnmf import bare_initialization
import glob
from caiman.summary_images import max_correlation_image
from caiman.motion_correction import motion_correct_iteration_fast

#%%
fname = 'example_movies/demoMovie.tif'
#fname = 'example_movies/13592_3_0000_slice1.hdf5'
merge_thresh = 0.8  # merging threshold, max correlation allowed
p = 1  # order of the autoregressive system
initbatch = 100
T1 = 5000
expected_comps = 500
rf = 16
stride = 3
K = 1
gSig = [6, 6]  # expected half size of neurons
rval_thr = .9
thresh_fitness_delta = -30
thresh_fitness_raw = -30

Y = cm.load(fname, subindices = slice(0,initbatch,None)).astype(np.float32)
Yr = Y.transpose(1,2,0).reshape((np.prod(Y.shape[1:]),-1), order='F')

Cn_init = Y.local_correlations(swap_dim = False)
pl.imshow(Cn_init)
#%%
Cn = max_correlation_image(cm.load(fname, subindices = slice(0,T1,None)).astype(np.float32), swap_dim = False)
#%%
cnm_init = bare_initialization(Y[:initbatch].transpose(1, 2, 0),init_batch=initbatch,k=1,gnb=1,
                                 gSig=gSig, merge_thresh=0.8,
                                 p=1, minibatch_shape=100, minibatch_suff_stat=5,
                                 update_num_comps=True, rval_thr=rval_thr,
                                 thresh_fitness_delta=thresh_fitness_delta,
                                 thresh_fitness_raw=thresh_fitness_raw,
                                 batch_update_suff_stat=True, max_comp_update_shape=5)

crd = plot_contours(cnm_init.A.tocsc(), Cn_init, thr=0.9)
#%% RUN ALGORITHM ONLINE
cnm = deepcopy(cnm_init)
cnm._prepare_object(np.asarray(Yr), T1, expected_comps)
cnm.max_comp_update_shape = np.inf
cnm.update_num_comps = True
t = cnm.initbatch
max_shift = 5
shifts = []
Y_ = cm.load(fname, subindices = slice(t,T1,None)).astype(np.float32)
for frame_count, frame in enumerate(Y_):
#    templ = cnm.Ab.dot(cnm.C_on[:cnm.M, t - 1]).reshape(cnm.dims, order='F')
#    frame_cor, shift = motion_correct_iteration_fast(frame, templ, max_shift, max_shift)
#    shifts.append(shift)
    cnm.fit_next(t, frame.copy().reshape(-1, order='F'))
    t += 1

C = cnm.C_on[cnm.gnb:cnm.M]
A = cnm.Ab[:, cnm.gnb:cnm.M]
print(('Number of components:' + str(A.shape[-1])))
#%%
pl.figure() 
crd = cm.utils.visualization.plot_contours(A, Cn, thr=0.9)
#%%
view_patches_bar(Yr, A, C, cnm.b, cnm.C_on[:cnm.gnb],
                 dims[0], dims[1], YrA=cnm.noisyC[cnm.gnb:cnm.M] - C, img=Cn)
