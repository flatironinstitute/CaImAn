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

#%%
fname = './example_movies/demoMovie.tif'
Y = cm.load(fname).astype(np.float32)
Cn = cm.local_correlations(Y.transpose(1, 2, 0))

T1 = Y.shape[0]
merge_thresh = 0.8  # merging threshold, max correlation allowed
p = 1  # order of the autoregressive system
initbatch = 100
expected_comps = 50
rf = 16
stride = 3
K = 4
gSig = [8, 8]  # expected half size of neurons
rval_thr = .98
thresh_fitness_delta = -30
thresh_fitness_raw = -40

fname_new = Y[:initbatch].save('demo.mmap', order='C')

#%%
Yr, dims, T = cm.load_memmap(fname_new)
images = np.reshape(Yr.T, [T] + list(dims), order='F')
Cn_init = cm.local_correlations(np.reshape(Yr, dims + (T,), order='F'))


##%% RUN ALGORITHM ON SMALL INITIAL BATCH
#pl.close('all')
#
#cnm_init = cnmf.CNMF(2, k=1, gSig=gSig, merge_thresh=merge_thresh, gnb = 2,
#                     p=p, rf=None, stride=stride, skip_refinement=False,
#                     normalize_init=False, options_local_NMF=None,
#                     minibatch_shape=100, minibatch_suff_stat=5,
#                     update_num_comps=True, rval_thr=rval_thr,
#                     thresh_fitness_delta=thresh_fitness_delta,
#                     thresh_fitness_raw=thresh_fitness_raw,
#                     batch_update_suff_stat=True, max_comp_update_shape=5)
#
#cnm_init = cnm_init.fit(images)
#
#print(('Number of components:' + str(cnm_init.A.shape[-1])))
##%%
#pl.figure()

cnm_init = bare_initialization(Y[:initbatch].transpose(1, 2, 0),init_batch=initbatch,k=4,gnb=1,
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
t = 100
Y_ = cm.load(fname)[t:].astype(np.float32)
for frame_count, frame in enumerate(Y_):
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
