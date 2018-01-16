#!/usr/bin/env python
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
    print('Not launched under iPython')

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


def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def load_object(filename):
    with open(filename, 'r') as input_obj:
        obj = pickle.load(input_obj)
    return obj


#%%
# Yr, trueC, trueS, trueA, trueb, centers, dims = gen_data(gamma = gamma, noise = noise, sig = gSig, seed = 3)
# T = Yr.shape[-1]
#    Y = cm.load('example_movies/Yr0000_d1_170_d2_170_d3_1_order_C_frames_3000_.mmap')[:2000,border_pix:-border_pix,border_pix:-border_pix]
# Y = cm.load('/mnt/ceph/users/agiovann/Jeff/J115_2015_12_09/Yr_d1_483_d2_492_d3_1_order_C_frames_96331_.mmap',subindices=np.arange(1000))[:,5:-5,5:-5]
# Y = cm.load('/tmp/Sue_2000_rig__d1_512_d2_512_d3_1_order_F_frames_2000_.mmap')[:,12:-12,12:-12]
# border_pix = 10
# Y = cm.load('/opt/local/Data/JGauthier-J115/Yr_d1_483_d2_492_d3_1_order_C_frames_96331_.mmap')[:,border_pix:-border_pix,border_pix:-border_pix]
# Y = cm.load('/Users/agiovann/J115_2015_12_09/Yr_d1_483_d2_492_d3_1_order_C_frames_96331_.mmap')[:,border_pix:-border_pix,border_pix:-border_pix]
# Y = cm.load('./test_jeff.tif')[:,10:-10,10:-10]
# T,d1,d2 = Y.shape
# Y = Y - np.min(Y[:5000])
#
#%%
# test
# Y = cm.load('example_movies/demoMovie.tif').astype(np.float32)
# T = 96331
# d1 = 483-10
# d2 = 492-10
#%%
c, dview, n_processes = cm.cluster.setup_cluster(
    backend='local', n_processes=None, single_thread=False)
#%%


# fls_starts = ['/opt/matlab/python/Data/Jeff/Yr_reduced*.mmap',
#              '/opt/local/privateCaImAn/example_movies/demoMovie.tif',\
#             '/opt/matlab/python/Data/Sue/k53_20160530_RSM_125um_41mW_zoom2p2_00001_000*.tif',
#             '/opt/local/Data/Sue/k53/k53_crop.tif',
#             '/opt/local/Data/Johannes/johannes_bad.tif',
#             '/opt/local/Data/Sue/k59_20161121_MMP_150um_47mW940nm_zoom4p2_*_g.tif']


# fls_starts = ['/opt/local/Data/JGauthier-J115/mmap_files/Yr_reduced*.mmap',
#              '/opt/local/privateCaImAn/example_movies/demoMovie.tif',
#              '/opt/local/Data/Sue/k53/k53_20160530_RSM_125um_41mW_zoom2p2_00001_000*.tif',
#              '/opt/local/Data/Sue/k53/k53_crop.tif',
#              '/opt/local/Data/Johannes/johannes_bad.tif',
#              '/opt/local/Data/Sue/k59_20161121_MMP_150um_47mW940nm_zoom4p2_*_g.tif']


fls_starts = ['/Users/agiovann/data/J115_2015_12_09/Yr_reduced*.mmap',
              '/opt/local/privateCaImAn/example_movies/demoMovie.tif',
              '/opt/local/Data/Sue/k53/k53_20160530_RSM_125um_41mW_zoom2p2_00001_000*.tif',
              '/opt/local/Data/Sue/k53/k53_crop.tif',
              '/opt/local/Data/Johannes/johannes_bad.tif',
              '/opt/local/Data/Sue/k59_20161121_MMP_150um_47mW940nm_zoom4p2_*_g.tif',
              '/home/andrea/CaImAn/example_movies/13800_1_0001_00002_0000*.tif'
              ]

fls_start = fls_starts[-1]
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
    ds = 2
    gSig = [7, 7]  # expected half size of neurons
    gSig = tuple(np.array(gSig) // ds + 1)
    init_files = 1
    online_files = 89
    initbatch = 1000
    T1 = 90000
    expected_comps = 1200
    rval_thr = 0.9
    thresh_fitness_delta = -30
    thresh_fitness_raw = -50
    mot_corr = False
    rf = 16 // ds
    stride = 6 // ds
    K = 5
elif fls_start == fls_starts[1]:
    ds = 1
    init_files = 1
    online_files = 0
    initbatch = 400
    T1 = 2000
    expected_comps = 50
    rf = 16
    stride = 3
    K = 4
    gSig = [6, 6]  # expected half size of neurons
    rval_thr = 0.9
    thresh_fitness_delta = -30
    thresh_fitness_raw = -50
    mot_corr = False
elif fls_start == fls_starts[2]:
    ds = 2
    init_files = 1
    online_files = 38
    initbatch = 3000
    T1 = 117000
    expected_comps = 1200
    rval_thr = 0.9
    thresh_fitness_delta = -30
    thresh_fitness_raw = -50
    max_shift = 20 // ds
    mot_corr = True
    gSig = tuple(np.array([7, 7]) // ds + 1)  # expected half size of neurons
    rf = 14 // ds
    stride = 4 // ds
    K = 4
    merge_thresh = 0.8
    p = 1
elif fls_start == fls_starts[3]:
    init_files = 1
    online_files = 0
    initbatch = 1000
    T1 = 3000
    expected_comps = 1200
    rval_thr = 0.9
    thresh_fitness_delta = -50
    thresh_fitness_raw = -50
    max_shift = 6
    mot_corr = True
    rf = 14
    stride = 6
    K = 6
    gSig = [4, 4]
    merge_thresh = 0.8
    p = 1

elif fls_start == fls_starts[4]:
    K = 200  # number of neurons expected per patch
    gSig = [7, 7]  # expected half size of neurons
    init_files = 1
    online_files = 0
    initbatch = 1000
    T1 = 3000
    expected_comps = 1400
    rval_thr = 0.9
    thresh_fitness_delta = -30
    thresh_fitness_raw = -50
    max_shift = 15
    mot_corr = True
elif fls_start == fls_starts[5]:
    K = 20  # number of neurons expected per patch
    gSig = [3, 3]  # expected half size of neurons
    init_files = 1
    online_files = 0
    initbatch = 200
    T1 = 2000
    expected_comps = 120
    rval_thr = 0.9
    thresh_fitness_delta = -50
    thresh_fitness_raw = -50
    max_shift = None
    mot_corr = False
elif fls_start == fls_starts[5]:
    ds = 2
    init_files = 1
    online_files = 38
    initbatch = 3000
    T1 = 117000
    expected_comps = 1200
    rval_thr = 0.9
    thresh_fitness_delta = -30
    thresh_fitness_raw = -50
    max_shift = 20 // ds
    mot_corr = True
    gSig = tuple(np.array([7, 7]) // ds + 1)  # expected half size of neurons
    rf = 14 // ds
    stride = 4 // ds
    K = 4
    merge_thresh = 0.8
    p = 1
else:
    raise Exception('File not defined')
#%%
# t1 = time()
# Cn = []
# for jj,mv in enumerate(fls[:1]):
#    print(jj)
#    m = cm.load(mv)[:initbatch]
#    if ds_factor != 1:
#        m=m.resize(ds_factor,ds_factor,1)
#    if len(Cn) == 0:
#        Cn = np.zeros(m.shape[1:])
#
#    Cn = np.maximum(Cn,m.local_correlations(swap_dim=False))
#    pl.imshow(Cn,cmap='gray')
#    pl.pause(.1)

#%%
# Cn = Y.local_correlations(eight_neighbours=True, swap_dim=False)
# np.save('Cn_90k.npy',Cn)

#%%
if ds > 1:
    Y = cm.load_movie_chain(fls[:init_files])[
        :initbatch].resize(1. / ds, 1. / ds)
else:
    Y = cm.load_movie_chain(fls[:init_files])[:initbatch]

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
base_name = '/'.join(fls[0].split('/')[:-1]) + '/' + \
    fls[0].split('/')[-1][:4] + '_' + str(init_files) + '.mmap'

fname_new = Y[:initbatch].save(base_name, order='C')
#%%
Yr, dims, T = cm.load_memmap(fname_new)
d1, d2 = dims
images = np.reshape(Yr.T, [T] + list(dims), order='F')
Y = np.reshape(Yr, dims + (T,), order='F')
Cn2 = cm.local_correlations(Y)
pl.imshow(Cn2)
#%% RUN ALGORITHM ON PATCHES
pl.close('all')
cnm_init = cnmf.CNMF(n_processes, method_init='greedy_roi', k=K, gSig=gSig, merge_thresh=merge_thresh,
                     p=0, dview=dview, Ain=None, rf=rf, stride=stride, method_deconvolution='oasis', skip_refinement=False,
                     normalize_init=False, options_local_NMF=None,
                     minibatch_shape=100, minibatch_suff_stat=5,
                     update_num_comps=True, rval_thr=rval_thr, thresh_fitness_delta=thresh_fitness_delta, thresh_fitness_raw=thresh_fitness_raw,
                     batch_update_suff_stat=True, max_comp_update_shape=5)


cnm_init = cnm_init.fit(images)
A_tot = cnm_init.A
C_tot = cnm_init.C
YrA_tot = cnm_init.YrA
b_tot = cnm_init.b
f_tot = cnm_init.f
sn_tot = cnm_init.sn

print(('Number of components:' + str(A_tot.shape[-1])))
#%%
pl.figure()
# crd = plot_contours(A_tot, Cn2, thr=0.9)
#%%
final_frate = 10  # approx final rate  (after eventual downsampling )
Npeaks = 10
traces = C_tot + YrA_tot
#        traces_a=traces-scipy.ndimage.percentile_filter(traces,8,size=[1,np.shape(traces)[-1]/5])
#        traces_b=np.diff(traces,axis=1)
fitness_raw, fitness_delta, erfc_raw, erfc_delta, r_values, significant_samples = \
    evaluate_components(Y, traces, A_tot, C_tot, b_tot, f_tot, final_frate, remove_baseline=True,
                        N=5, robust_std=False, Athresh=0.1, Npeaks=Npeaks, thresh_C=0.3)

idx_components_r = np.where(r_values >= .5)[0]
idx_components_raw = np.where(fitness_raw < -20)[0]
idx_components_delta = np.where(fitness_delta < -20)[0]

idx_components = np.union1d(idx_components_r, idx_components_raw)
idx_components = np.union1d(idx_components, idx_components_delta)
idx_components_bad = np.setdiff1d(list(range(len(traces))), idx_components)

print(('Keeping ' + str(len(idx_components)) +
       ' and discarding  ' + str(len(idx_components_bad))))
#%%
# pl.figure()
crd = plot_contours(A_tot.tocsc()[:, idx_components], Cn2, thr=0.9)
#%%
view_patches_bar(Yr, scipy.sparse.coo_matrix(A_tot.tocsc()[:, idx_components_bad]), C_tot[
    idx_components_bad, :], b_tot, f_tot, dims[0], dims[1], YrA=YrA_tot[idx_components_bad, :], img=Cn2)
#%%
A_tot = A_tot.tocsc()[:, idx_components]
C_tot = C_tot[idx_components]
#%%

#%%
# cnm2 = cnmf.CNMF(n_processes, k=A_tot.shape, gSig=gSig, merge_thresh=merge_thresh, p=p, dview=dview, Ain=A_tot, Cin=C_tot,
#                f_in=f_tot, rf=None, stride=None, method_deconvolution='oasis')
cnm_refine = cnmf.CNMF(n_processes, method_init='greedy_roi', k=A_tot.shape, gSig=gSig,
                       merge_thresh=merge_thresh, rf=None, stride=None, p=p, dview=dview,
                       Ain=A_tot, Cin=C_tot, f_in=f_tot, method_deconvolution='oasis',
                       skip_refinement=True, normalize_init=False, options_local_NMF=None,
                       minibatch_shape=100, minibatch_suff_stat=5, update_num_comps=True,
                       rval_thr=rval_thr, thresh_fitness_delta=thresh_fitness_delta,
                       thresh_fitness_raw=thresh_fitness_raw, batch_update_suff_stat=True,
                       max_comp_update_shape=5)


cnm_refine = cnm_refine.fit(images)
#%%
A, C, b, f, YrA, sn = cnm_refine.A, cnm_refine.C, cnm_refine.b, cnm_refine.f, cnm_refine.YrA, cnm_refine.sn
#%%
final_frate = 10
Npeaks = 10
traces = C + YrA
#        traces_a=traces-scipy.ndimage.percentile_filter(traces,8,size=[1,np.shape(traces)[-1]/5])
#        traces_b=np.diff(traces,axis=1)
fitness_raw, fitness_delta, erfc_raw, erfc_delta, r_values, significant_samples = \
    evaluate_components(Y, traces, A, C, b, f, final_frate, remove_baseline=True,
                        N=5, robust_std=False, Athresh=0.1, Npeaks=Npeaks,  thresh_C=0.3)

if fls_start == fls_starts[0]:
    #    if ds > 1:
    #        idx_components_r = np.where(r_values >= 0)[0]
    #        idx_components_raw = np.where(fitness_raw < -10)[0]
    #        idx_components_delta = np.where(fitness_delta < -10)[0]
    #
    #    else:
    idx_components_r = np.where(r_values >= 0.85)[0]
    idx_components_raw = np.where(fitness_raw < -50)[0]
    idx_components_delta = np.where(fitness_delta < -50)[0]

else:
    if ds > 1:
        idx_components_r = np.where(r_values >= .9)[0]
        idx_components_raw = np.where(fitness_raw < -100)[0]
        idx_components_delta = np.where(fitness_delta < -100)[0]
#
    else:
        idx_components_r = np.where(r_values >= .95)[0]
        idx_components_raw = np.where(fitness_raw < -100)[0]
        idx_components_delta = np.where(fitness_delta < -100)[0]


# min_radius = gSig[0] - 2
# masks_ws, idx_blobs, idx_non_blobs = extract_binary_masks_blob(
#    A.tocsc(), min_radius, dims, num_std_threshold=1,
#    minCircularity=0.7, minInertiaRatio=0.2, minConvexity=.5)

idx_components = np.union1d(idx_components_r, idx_components_raw)
idx_components = np.union1d(idx_components, idx_components_delta)
# idx_blobs = np.intersect1d(idx_components, idx_blobs)
idx_components_bad = np.setdiff1d(list(range(len(traces))), idx_components)

print(' ***** ')
print((len(traces)))
print((len(idx_components)))
#%%
cnm_refine.idx_components = idx_components
cnm_refine.idx_components_bad = idx_components_bad
cnm_refine.r_values = r_values
cnm_refine.fitness_raw = fitness_raw
cnm_refine.fitness_delta = fitness_delta
cnm_refine.Cn2 = Cn2

#%%
crd = plot_contours(A.tocsc()[:, idx_components], Cn2, thr=0.9, vmax=.55)
#%%
view_patches_bar(Yr, scipy.sparse.coo_matrix(A.tocsc()[:, idx_components]), C[
    idx_components, :], b, f, dims[0], dims[1], YrA=YrA[idx_components, :], img=Cn2)
#%%
view_patches_bar(Yr, scipy.sparse.coo_matrix(A.tocsc()[:, idx_components_bad]), C[
    idx_components_bad, :], b, f, dims[0], dims[1], YrA=YrA[idx_components_bad, :], img=Cn2)

#%%
cnm_refine.dview = None
save_object(cnm_refine, fls[0][:-4] + '_DS_' + str(ds) + '.pkl')
cnm_init.dview = None
save_object(cnm_init, fls[0][:-4] + '_DS_' + str(ds) + '_init.pkl')
#%%
cnm_refine = load_object(fls[0][:-4] + '_DS_' + str(ds) + '.pkl')
cnm_refine._prepare_object(np.asarray(Yr), T1, expected_comps,
                           idx_components=cnm_refine.idx_components)
Cf = np.r_[cnm_refine.C_on[:, :initbatch], cnm_refine.f2]
#%%
if mot_corr:
    np.savez('results_analysis_online_SUE_' + '_DS_' + str(ds) + '_init_.npz',
             Cn=Cn2, Ab=cnm_refine.Ab, Cf=Cf, b=cnm_refine.b2, f=cnm_refine.f2,
             dims=cnm_refine.dims, noisyC=cnm_refine.noisyC, shifts=mc[1], templ=mc[2])
else:
    np.savez('results_analysis_online_JEFF_' + '_DS_' + str(ds) + '_init_.npz',
             Cn=Cn2, Ab=cnm_refine.Ab, Cf=Cf, b=cnm_refine.b2, f=cnm_refine.f2,
             dims=cnm_refine.dims, noisyC=cnm_refine.noisyC)
#%%
cm.stop_server()
pl.close('all')
#%%
# pl.figure(figsize=(20,13))
cnm2 = load_object(fls[0][:-4] + '_DS_' + str(ds) + '.pkl')
cnm2._prepare_object(np.asarray(Yr), T1, expected_comps,
                     idx_components=cnm2.idx_components)
cnm2.max_comp_update_shape = np.inf
cnm2.update_num_comps = True
t = cnm2.initbatch
tottime = []
Cn = Cn2.copy()
if online_files == 0:
    end_files = fls[:1]
    init_batc_iter = initbatch
else:
    end_files = fls[init_files:init_files + online_files]
    init_batc_iter = 0
#%
shifts = []
for ffll in end_files:  # np.array(fls)[np.array([1,2,3,4,5,-5,-4,-3,-2,-1])]:
    print(ffll)
    Y_ = cm.load(ffll)[init_batc_iter:]
#
    if ds > 1:
        Y_1 = Y_.resize(1. / ds, 1. / ds, 1)
    else:
        Y_1 = Y_.copy()

    if mot_corr:
        # templ = cnm2.Ab[:, -1].dot(cnm2.C_on[-1, t - 1]
        #                            ).toarray().reshape(cnm2.dims, order='F') * img_norm
        templ = (cnm2.Ab.data[:cnm2.Ab.indptr[1]] * cnm2.C_on[0, t - 1]
                 ).reshape(cnm2.dims, order='F') * img_norm

        newcn = (Y_1 - img_min).motion_correct(
            max_shift, max_shift, template=templ)[0].local_correlations(swap_dim=False)
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
                   scipy.sparse.linalg.norm(cnm2.Ab, axis=0).min(), cnm2.N - old_comps])
            old_comps = cnm2.N

        t1 = time()
        frame_ = frame.copy().astype(np.float32)
        if ds > 1:
            frame_ = cv2.resize(frame_, img_norm.shape[::-1])

        frame_ -= img_min

        if mot_corr:
            templ = cnm2.Ab.dot(
                cnm2.C_on[:cnm2.M, t - 1]).reshape(cnm2.dims, order='F') * img_norm
            frame_cor, shift = motion_correct_iteration_fast(
                frame_, templ, max_shift, max_shift)
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
    if t % 1000 == 999:
        pl.cla()
        crd = cm.utils.visualization.plot_contours(A, Cn, thr=0.9)
        pl.pause(1)

    if mot_corr:
        np.savez('results_analysis_online_SUE_' + '_DS_' + str(ds) + '_' + str(t) + '.npz',
                 Cn=Cn, Ab=cnm2.Ab, Cf=cnm2.C_on, b=cnm2.b, f=cnm2.f,
                 dims=cnm2.dims, tottime=tottime, noisyC=cnm2.noisyC, shifts=shifts, templ=templ)
    else:
        np.savez('results_analysis_online_JEFF_LAST_' + '_DS_' + str(ds) + '_' + str(t) + '.npz',
                 Cn=Cn, Ab=cnm2.Ab, Cf=cnm2.C_on, b=cnm2.b, f=cnm2.f,
                 dims=cnm2.dims, tottime=tottime, noisyC=cnm2.noisyC)

    print((t - initbatch) / np.sum(tottime))

cv2.destroyAllWindows()
#%%
save = True
if save:
    if mot_corr:
        np.savez('results_analysis_online_SUE_DS_MOT_CORR.npz',
                 Cn=Cn, Ab=cnm2.Ab, Cf=cnm2.C_on, b=cnm2.b, f=cnm2.f,
                 dims=cnm2.dims, tottime=tottime, noisyC=cnm2.noisyC, shifts=shifts)
    else:
        np.savez('results_analysis_online_JEFF_DS_2.npz',
                 Cn=Cn, Ab=cnm2.Ab, Cf=cnm2.C_on, b=cnm2.b, f=cnm2.f,
                 dims=cnm2.dims, tottime=tottime, noisyC=cnm2.noisyC)

#%%
if False:
    import cv2
    tottime = []
    for t in range(initbatch, T1):
        if t % 10 == 0:
            print(t)
        t1 = time()
        frame = Yr[:, t].copy()
    #    frame = cv2.resize(frame.reshape(dims,order = 'F'),(244,244))
        cnm2.fit_next(t, frame.reshape(-1, order='F'))
        tottime.append(time() - t1)

    print((T1 - initbatch) / np.sum(tottime))
    # pl.figure()
    # pl.plot(tottime)
    #%%
    A, b = cnm2.Ab[:, cnm2.gnb:], cnm2.Ab[:, :cnm2.gnb].toarray()
    C, f = cnm2.C_on[cnm2.gnb:cnm2.M, :], cnm2.C_on[:cnm2.gnb, :]
    b_trace = [osi.b for osi in cnm2.OASISinstances]
    #%%
    # Cn = Y.local_correlations(eight_neighbours = True,swap_dim=False)
    # resCn = cv2.resize(Cn, (244, 244))
    pl.figure()
    crd = cm.utils.visualization.plot_contours(A, Cn, thr=0.9)
    #%%
    view_patches_bar(Yr, scipy.sparse.coo_matrix(A.tocsc()[:, :]), C[:, :], b, f,
                     dims[0], dims[1], YrA=cnm2.noisyC[cnm2.gnb:cnm2.M] - C, img=Cn)
    #%%

    #%%
    with np.load('results_full_movie_online_may5/results_analysis_online_JEFF_90k.take7_no_batch.npz') as ld:
        print(ld.keys())
        locals().update(ld)
        Ab = Ab[()]
    #    OASISinstances = OASISinstances[()]
        C_on = Cf
        A, b = cnm2.Ab[:, cnm2.gnb:], cnm2.Ab[:, :cnm2.gnb].toarray()
        C, f = cnm2.C_on[cnm2.gnb:cnm2.M, :], cnm2.C_on[:cnm2.gnb, :]
#%%
    pl.figure()
    crd = cm.utils.visualization.plot_contours(A, Cn, thr=0.9)

#%%
    view_patches_bar(None, scipy.sparse.coo_matrix(A.tocsc()[:, :]), C[:, :], b, f,
                     dims[0], dims[1], YrA=cnm2.noisyC[cnm2.gnb:cnm2.M] - C, img=Cn)
    #%%
    with np.load('results_analysis_offline_JEFF_90k.npz') as ld:
        print(ld.keys())
        locals().update(ld)
    view_patches_bar(None, scipy.sparse.coo_matrix(
        A), C[:, :], b, f, d1, d2, YrA=YrA, img=Cn)
    #%%
    fls_ld = glob.glob('results_analysis_online_JEFF_LAST__DS_2_*0.npz')
    fls_ld.sort(key=lambda x: np.int(x.split('_')[-1][:-5]))
    print(fls_ld)
    for fl in fls_ld:
        with np.load(fl) as ld:
            pl.plot(1 / scipy.sparse.linalg.norm(ld['Ab'][()], axis=0))
            pl.pause(1)
            break
