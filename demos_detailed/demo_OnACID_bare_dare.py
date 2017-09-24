#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 14:22:48 2017

@author: agiovann and me
"""
import numpy as np
try:
    if __IPYTHON__:
        print('Debugging!')
        # this is used for debugging purposes only. allows to reload classes when changed
        get_ipython().magic('load_ext autoreload')
        get_ipython().magic('autoreload 2')
except NameError:
    print('Not IPYTHON')
    pass

from time import time
import caiman as cm
from caiman.source_extraction import cnmf as cnmf
from caiman.utils.visualization import view_patches_bar
import pylab as pl
import scipy
from caiman.motion_correction import motion_correct_iteration_fast
from caiman.source_extraction.cnmf.online_cnmf import RingBuffer
from caiman.components_evaluation import evaluate_components
import cv2
from caiman.utils.visualization import plot_contours
import pickle as pickle
import glob
from caiman.source_extraction.cnmf.online_cnmf import bare_initialization


def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def load_object(filename):
    with open(filename, 'rb') as input_obj:
        obj = pickle.load(input_obj)
    return obj



#%%
fls_starts = ['/home/andrea/CaImAn/example_movies/demoMovie.tif', # quick demo
              'example_movies/13592_3_0000_slice1.hdf5', # Mesoscope
              'example_movies/14062_1_00004_00001_slice1.hdf5' #dense labeling
              ]

fls_start = fls_starts[2]
fls = glob.glob(fls_start)
fls.sort()
print(fls)

#%%
ds_factor = 1
try:
    del rf, stride
except:
    pass
if fls_start == fls_starts[0]:
    ds = 1 # downampling
    init_files = 1 # number of files used for initialization
    online_files = 0 # number of files used for online
    initbatch = 400 # number f frames for initialization
    T1 = 2000 #total number of frames
    expected_comps = 100 # exaggerate here
    K = 4 #initial number of components
    gSig = [6, 6]  # expected half size of neurons
    rval_thr = 0.9
    thresh_fitness_delta = -30
    thresh_fitness_raw = -50
    mot_corr = False

elif fls_start == fls_starts[1]:
    ds = 1
    init_files = 1
    online_files = 0
    initbatch = 300
    T1 = 18000
    expected_comps = 1000
    rval_thr = 0.85
    thresh_fitness_delta = -30
    thresh_fitness_raw = -30
    max_shift = 10 // ds
    mot_corr = True
    gSig = tuple(np.array([3, 3]) // ds + 1)  # expected half size of neurons
    K = 50
    merge_thresh = 0.8
    p = 1
elif fls_start == fls_starts[2]:
    ds = 1
    init_files = 1
    online_files = 0
    initbatch = 300
    T1 = 2000
    expected_comps = 1000
    rval_thr = 0.85
    thresh_fitness_delta = -30
    thresh_fitness_raw = -30
    max_shift = 10 // ds
    mot_corr = True
    gSig = tuple(np.array([10, 10]) // ds + 1)  # expected half size of neurons
    K = 50
    merge_thresh = 0.8
    p = 1
else:
    raise Exception('File not defined')
#%% initialize movie
if ds > 1:
    Y = cm.load(fls[0], subindices = slice(0,initbatch,None)).astype(np.float32).resize(1. / ds, 1. / ds)
else:
    Y =  cm.load(fls[0], subindices = slice(0,initbatch,None)).astype(np.float32)
    

if mot_corr:
    mc = Y.motion_correct(max_shift, max_shift)
    Y = mc[0].astype(np.float32)
    borders = np.max(mc[1])
else:
    Y = Y.astype(np.float32)
    
if ds_factor != 1:
    Y = Y.resize(ds_factor, ds_factor, 1)
    gSig = 1 + (np.multiply(gSig, ds_factor)).astype(np.int)

    
img_min = Y.min()
Y -= img_min
img_norm = np.std(Y, axis=0)
img_norm += np.median(img_norm)
Y = Y / img_norm[None, :, :]


_, d1, d2 = Y.shape
dims = (d1, d2)
Yr = Y.to_2D().T
merge_thresh = 0.8  # merging threshold, max correlation allowed
p = 1  # order of the autoregressive system
# T = Y.shape[0]
new_dims = (d1, d2)

Cn_init = Y.local_correlations(swap_dim = False)
pl.imshow(Cn_init)

#%% initialize OnAcid
cnm_init = bare_initialization(Y[:initbatch].transpose(1, 2, 0),init_batch=initbatch,k=K,gnb=1,
                                 gSig=gSig, merge_thresh=0.8,
                                 p=1, minibatch_shape=100, minibatch_suff_stat=5,
                                 update_num_comps=True, rval_thr=rval_thr,
                                 thresh_fitness_delta=thresh_fitness_delta,
                                 thresh_fitness_raw=thresh_fitness_raw,
                                 batch_update_suff_stat=True, max_comp_update_shape=5, 
                                 deconv_flag = False,
                                 simultaneously=False, n_refit=0)

crd = plot_contours(cnm_init.A.tocsc(), Cn_init, thr=0.9)

#%
A, C, b, f, YrA, sn = cnm_init.A, cnm_init.C, cnm_init.b, cnm_init.f, cnm_init.YrA, cnm_init.sn

#%%
view_patches_bar(Yr, scipy.sparse.coo_matrix(A.tocsc()[:, :]), C[:, :], b, f, dims[0], dims[1], YrA=YrA[:, :], img=Cn_init)
#%%
cnm_init.dview = None
save_object(cnm_init, fls[0][:-4] + '_DS_' + str(ds) + '.pkl')
#cnm_init.dview = None
#save_object(cnm_init, fls[0][:-4] + '_DS_' + str(ds) + '_init.pkl')
#%%
cnm_init = load_object(fls[0][:-4] + '_DS_' + str(ds) + '.pkl')
cnm_init._prepare_object(np.asarray(Yr), T1, expected_comps,
                           idx_components=None)
Cf = np.r_[cnm_init.C_on[:, :initbatch], cnm_init.f2]
#%%
# pl.figure(figsize=(20,13))
plot_contours = False
compute_cn = False
play_reconstr = True
resize_fact = .5

cnm2 = load_object(fls[0][:-4] + '_DS_' + str(ds) + '.pkl')
cnm2._prepare_object(np.asarray(Yr), T1, expected_comps, idx_components=None)
cnm2.max_comp_update_shape = np.inf
cnm2.update_num_comps = True
t = cnm2.initbatch
tottime = []
Cn = Cn_init.copy()
if online_files == 0:
    end_files = fls[:1]
    init_batc_iter = initbatch
    end_batch = T1
else:
    end_files = fls[init_files:init_files + online_files]
    init_batc_iter = 0
    T1 = None
#%
shifts = []
for ffll in end_files:  # np.array(fls)[np.array([1,2,3,4,5,-5,-4,-3,-2,-1])]:
    print(ffll)
    Y_ = cm.load(ffll, subindices=slice(init_batc_iter,T1,None))
    
    if plot_contours or compute_cn:
        if ds > 1:
            Y_1 = Y_.resize(1. / ds, 1. / ds, 1)
        else:
            Y_1 = Y_.copy()
                    
            if mot_corr:
                # templ = cnm2.Ab[:, -1].dot(cnm2.C_on[-1, t - 1]
                #                            ).toarray().reshape(cnm2.dims, order='F') * img_norm
                templ = (cnm2.Ab.data[:cnm2.Ab.indptr[1]] * cnm2.C_on[0, t - 1]).reshape(cnm2.dims, order='F') * img_norm
        
                newcn = (Y_1 - img_min).motion_correct(max_shift, max_shift, template=templ)[0].local_correlations(swap_dim=False)
                
                Cn = np.maximum(Cn, newcn)
            else:
                Cn = np.maximum(Cn, Y_1.local_correlations(swap_dim=False))

    # print())
    # print(np.sum(cnm2.Ab.power(2),0))
    old_comps = cnm2.N
    for frame_count, frame in enumerate(Y_):
        #        frame_ = frame.copy().astype(np.float32)/img_norm
        if np.isnan(np.sum(frame)):
            raise Exception('Frame ' + str(frame_count) + ' contains nan')
        if t % 100 == 0:
            print(t)
            print([np.max(cnm2.CC), np.max(cnm2.CY), np.max(cnm2.C_on[cnm2.gnb:cnm2.M, :t - 1]),
                   scipy.sparse.linalg.norm(cnm2.Ab, axis=0).min(), cnm2.N - old_comps, cnm2.Ab.shape])
            old_comps = cnm2.N

        t1 = time()
        frame_ = frame.copy().astype(np.float32)
        if ds > 1:
            frame_ = cv2.resize(frame_, img_norm.shape[::-1])

        frame_ -= img_min

        if mot_corr:
            templ = cnm2.Ab.dot(cnm2.C_on[:cnm2.M, t - 1]).reshape(cnm2.dims, order='F') * img_norm
            frame_cor, shift = motion_correct_iteration_fast(frame_, templ, max_shift, max_shift)
            shifts.append(shift)
        else:
            templ = None
            frame_cor = frame_

        frame_cor = frame_cor / img_norm
        cnm2.fit_next(t, frame_cor.reshape(-1, order='F'))
        tottime.append(time() - t1)

        t += 1

        A, b = cnm2.Ab[:, cnm2.gnb:], cnm2.Ab[:, :cnm2.gnb].toarray()
        C, f = cnm2.C_on[cnm2.gnb:cnm2.M, :], cnm2.C_on[:cnm2.gnb, :]
        noisyC = cnm2.C_on[:cnm2.M, :]
        if t % 1000 == 0 and plot_contours:
            pl.cla()
            crd = cm.utils.visualization.plot_contours(A, Cn, thr=0.9)
            pl.pause(1)
        if play_reconstr:
            comps_frame = A.dot(C[:,t-1]).reshape(cnm2.dims, order = 'F')*img_norm/np.max(img_norm)
            bgkrnd_frame = b.dot(f[:,t-1]).reshape(cnm2.dims, order = 'F')*img_norm/np.max(img_norm)
            all_comps = (np.array(A.sum(-1)).reshape(cnm2.dims, order = 'F'))*3#*img_norm/np.max(img_norm)
            frame_comp_1 = cv2.resize(np.concatenate([(frame)/np.max(img_norm),all_comps],axis = -1),(np.int(cnm2.dims[1]*resize_fact),np.int(cnm2.dims[0]*resize_fact) )).T
            frame_comp_2 = cv2.resize(np.concatenate([comps_frame,comps_frame+bgkrnd_frame],axis = -1),(np.int(cnm2.dims[1]*resize_fact),np.int(cnm2.dims[0]*resize_fact) )).T
            cv2.imshow('frame',np.concatenate([frame_comp_1,frame_comp_2],axis=-1))
#            cv2.imshow('frame',np.concatenate([frame_comp_1],axis=-1))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
                
#    if mot_corr:
#        np.savez('results_analysis_online_SUE_' + '_DS_' + str(ds) + '_' + str(t) + '.npz',
#                 Cn=Cn, Ab=cnm2.Ab, Cf=cnm2.C_on, b=cnm2.b, f=cnm2.f,
#                 dims=cnm2.dims, tottime=tottime, noisyC=cnm2.noisyC, shifts=shifts, templ=templ)
#    else:
#        np.savez('results_analysis_online_JEFF_LAST_' + '_DS_' + str(ds) + '_' + str(t) + '.npz',
#                 Cn=Cn, Ab=cnm2.Ab, Cf=cnm2.C_on, b=cnm2.b, f=cnm2.f,
#                 dims=cnm2.dims, tottime=tottime, noisyC=cnm2.noisyC)

    print((t - initbatch) / np.sum(tottime))

cv2.destroyAllWindows()
#%%
np.savez('results_analysis_online_MOT_CORR.npz',
         Cn=Cn, Ab=cnm2.Ab, Cf=cnm2.C_on, b=cnm2.b, f=cnm2.f,
         dims=cnm2.dims, tottime=tottime, noisyC=cnm2.noisyC, shifts=shifts)

#%%
A, b = cnm2.Ab[:, cnm2.gnb:], cnm2.Ab[:, :cnm2.gnb].toarray()
C, f = cnm2.C_on[cnm2.gnb:cnm2.M, :], cnm2.C_on[:cnm2.gnb, :]
b_trace = [osi.b for osi in cnm2.OASISinstances]
#%%
# Cn = Y.local_correlations(eight_neighbours = True,swap_dim=False)
# resCn = cv2.resize(Cn, (244, 244))
pl.figure()
crd = cm.utils.visualization.plot_contours(A, Cn, thr=0.9)
#%%m = 
view_patches_bar(Yr, scipy.sparse.coo_matrix(A.tocsc()[:, :]), C[:, :], b, f,
                 dims[0], dims[1], YrA=cnm2.noisyC[cnm2.gnb:cnm2.M] - C, img=Cn)
#%%



