#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 11:55:48 2016

@author: agiovann
"""

#%%
from skimage.external import tifffile
import caiman as cm
from sklearn.decomposition import PCA, DictionaryLearning, MiniBatchDictionaryLearning, NMF
from sklearn.decomposition import FastICA
import pylab as pl
import glob
import numpy as np
from caiman.components_evaluation import compute_event_exceptionality  # import spams
#from PIL import Image
import scipy
from skimage.morphology import square
from skimage.filters import rank
#%%
def remove_baseline_fast(traces, bl_percentile=8):
        traces = traces.astype(np.float32)
        num_samps_bl = np.minimum(np.shape(traces)[-1]//5, 800)
        T = traces.shape[-1]
        downsampfact = num_samps_bl
        elm_missing = int(np.ceil(T * 1.0 / downsampfact)
                          * downsampfact - T)
        padbefore = int(np.floor(elm_missing//2.0))
        padafter = int(np.ceil(elm_missing// 2.0))
        tr_tmp = np.pad(
            traces.T, ((padbefore, padafter), (0, 0)), mode='reflect')
        numFramesNew, num_traces = np.shape(tr_tmp)
        #% compute baseline quickly
        print("binning data ...")
        tr_BL = np.reshape(tr_tmp, (downsampfact, int(
            numFramesNew//downsampfact), num_traces), order='F')
        tr_BL = np.percentile(tr_BL, bl_percentile, axis=0)
        print("interpolating data ...")
        print(tr_BL.shape)
        tr_BL = scipy.ndimage.zoom(np.array(tr_BL, dtype=np.float32), [
                                   downsampfact, 1], order=3, mode='constant', cval=0.0, prefilter=True)
        if padafter == 0:
                traces -= tr_BL.T
        else:
                traces -= tr_BL[padbefore:-padafter].T

        return traces.astype(np.float32)
#%%
from scipy.stats import norm
#a = cm.load('example_movies/demoMovie.tif')
a = cm.load('/mnt/ceph/neuro/labeling/yuste.Single_150u/images/tifs/Yr_d1_200_d2_256_d3_1_order_C_frames_3000_.mmap')
all_els = []
for it in range(1):
    print(it)
#    a = cm.movie(np.random.randn(*mns.shape).astype(np.float32))
    Yr = remove_baseline_fast(np.array(cm.movie.to_2D(a)).T).T

    #%
#    norm = lambda(x): np.exp(-x**2/2)/np.sqrt(2*np.pi)
    fitness, res, sd_r, md = compute_event_exceptionality(Yr.T)
    Yr_c = -np.log(norm.sf(np.array((Yr - md) /
                                            sd_r, dtype=np.float)))
    mns = cm.movie(scipy.ndimage.convolve(np.reshape(
        Yr_c, [-1, a.shape[1], a.shape[2]], order='F'), np.ones([5, 3, 3])))
    mns[mns < (38 * np.log10((mns.shape[0])))] = 0
    all_els.append(np.sum(mns > 0)/np.sum(Yr>0))
    print(all_els)


#%%
#m1 = cm.movie((np.array((Yr-md)/sd_r)).reshape([-1,60,80],order = 'F'))*(scipy.ndimage.convolve(mns>0,np.ones([5,3,3])))
m1 = cm.movie((np.array((Yr - md) / sd_r)
               ).reshape([-1, a.shape[1],a.shape[2]], order='F')) * (mns > 0)
#%%
if False:
    b1 = np.percentile(Yr,20,axis=0).reshape([a.shape[1],a.shape[2]], order='F')

    m1 = cm.movie((np.array((Yr - md))
               ).reshape([-1, a.shape[1],a.shape[2]], order='F'))

#%%
m1.save('/mnt/ceph/neuro/labeling/yuste.Single_150u/images/tifs/Single_150um_024_sparse.tif')
#%%
m1.save('example_movies/demoMovie_sparse.tif')
#%%
mov = Yr
#%%
mns = -cm.movie(np.reshape(res.clip(-1000, 0),
                           [m1.shape[0], m1.shape[1], -1]).transpose([2, 1, 0]))
#mns[mns > 1000] = 1000
mov = cm.movie.to_2D(mns)
#%%
noise = res[2]
mode = cm.components_evaluation.mode_robust(Yr, 0)
Yr_1 = (cm.movie.to_2D(a) - mode) / res[2]
mns_1 = (np.reshape(Yr_1, [-1, 80, 60], order='F'))
mov = np.maximum(0, cm.movie.to_2D(mns_1))
#%%


n_comps = 10
pca = PCA(n_comps)
pca = NMF(n_comps, alpha=10, l1_ratio=1, init='nndsvda')

pca.fit(mov)
#%%
import cv2
comps = np.reshape(pca.components_, [n_comps, 30, 30])
for count, comp in enumerate(comps):
    pl.subplot(4, 4, count + 1)
    blur = cv2.GaussianBlur(comp.astype(np.float32), (5, 5), 0)
    blur = np.array(blur / np.max(blur) * 255, dtype=np.uint8)
    ret3, th3 = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    pl.imshow((th3 * comp).T)

#%%
n_comps = 3
dl = DictionaryLearning(n_comps, alpha=1, verbose=True)
comps = dl.fit_transform(Yr.T)
comps = np.reshape(comps, [30, 30, n_comps]).transpose([2, 0, 1])
for count, comp in enumerate(comps):
    pl.subplot(4, 4, count + 1)
    pl.imshow(comp)
#%%
N_ICA_COMPS = 8
ica = FastICA(N_ICA_COMPS, max_iter=10000, tol=10e-8)
ica.fit(pca.components_)
#%
comps = np.reshape(ica.components_, [N_ICA_COMPS, 30, 30])
for count, comp in enumerate(comps):
    idx = np.argmax(np.abs(comp))
    comp = comp * np.sign(comp.flatten()[idx])
    pl.subplot(4, 4, count + 1)
    pl.imshow(comp.T)
#%%
n_comps = 5
a = cm.load('demoMovie.tif')[:100, 20:50, 20:50].IPCA_stICA(15, n_comps)
#%%
comps = np.reshape(a, [n_comps, 30, 30])
for count, comp in enumerate(comps):
    pl.subplot(3, 4, count + 1)
    pl.imshow(comp)
#%%
m = cm.load('demoMovie.tif')
m1 = m.IPCA_denoise(50)
#%%
m1.play(fr=30, gain=5., magnification=4)
#%%
m1.save('demoMovieDen.tif')
#%%
m1 = cm.load('demoMovieDen.tif')

m1.play(fr=30, gain=4., magnification=4)


#%%

fls = glob.glob('*Den.tif')
print fls
all_names, all_shifts, all_xcorrs, all_templates = cm.motion_correction.motion_correct_online_multifile(
    fls, 0)

#%%
Yr, dims, T = cm.load_memmap(all_names[0])
d1, d2 = dims
images = np.reshape(Yr.T, [T] + list(dims), order='F')
Y = np.reshape(Yr, dims + (T,), order='F')
#%%
K = 35  # number of neurons expected per patch
gSig = [7, 7]  # expected half size of neurons
merge_thresh = 0.8  # merging threshold, max correlation allowed
p = 2  # order of the autoregressive system
cnm = cnmf.CNMF(8,  k=K, gSig=gSig, merge_thresh=merge_thresh,
                p=p, dview=None, Ain=None, method_deconvolution='oasis')
cnm = cnm.fit(images)
#%%
A, C, b, f, YrA, sn = cnm.A, cnm.C, cnm.b, cnm.f, cnm.YrA, cnm.sn
#%%
final_frate = 10
tB = np.minimum(-2, np.floor(-5. / 30 * final_frate))
tA = np.maximum(5, np.ceil(25. / 30 * final_frate))
Npeaks = 10
traces = C + YrA
#        traces_a=traces-scipy.ndimage.percentile_filter(traces,8,size=[1,np.shape(traces)[-1]/5])
#        traces_b=np.diff(traces,axis=1)
fitness_raw, fitness_delta, erfc_raw, erfc_delta, r_values, significant_samples = \
    evaluate_components(Y, traces, A, C, b, f, remove_baseline=True,
                        N=5, robust_std=False, Athresh=0.1, Npeaks=Npeaks, tB=tB, tA=tA, thresh_C=0.3)

idx_components_r = np.where(r_values >= .5)[0]
idx_components_raw = np.where(fitness_raw < -40)[0]
idx_components_delta = np.where(fitness_delta < -20)[0]


idx_components = np.union1d(idx_components_r, idx_components_raw)
idx_components = np.union1d(idx_components, idx_components_delta)
idx_components_bad = np.setdiff1d(range(len(traces)), idx_components)

print(' ***** ')
print len(traces)
print(len(idx_components))


#%% visualize components
# pl.figure();
Cn = cm.local_correlations(Y[:, :, :3000])
crd = plot_contours(A.tocsc(), np.mean(images, 0), thr=0.9)

#%%
view_patches_bar(Yr, scipy.sparse.coo_matrix(A.tocsc()[:, idx_components]), C[
    idx_components, :], b, f, dims[0], dims[1], YrA=YrA[idx_components, :], img=Cn)
#%%
a.fr=10
m_bg, bg = a.computeDFF(quantilMin=10, secsWindow=10)
#%%
bg_ds = bg.resize(.1,.1,1)
dims_bg = bg_ds.shape[1:]
#%%
n_b = 1
model = NMF(n_components=n_b , init='nndsvdar')
b_in = model.fit_transform(np.maximum(bg_ds.reshape((-1,np.prod(dims_bg)), order='F').T, 0)).astype(np.float32)
f_in = model.components_.squeeze().astype(np.float32)
#%%
for i in range(n_b ):
    pl.subplot(1,n_b ,i+1)
    pl.imshow(b_in.reshape(dims_bg+(-1,),order='F')[:,:,i],vmax = np.mean(b_in[:,i])*3,cmap='gray')
#%%
def HALS4activity(Yr, A, C, iters=2):
        U = A.T.dot(Yr)
        V = A.T.dot(A)
        for _ in range(iters):
            for m in range(len(U)):  # neurons and background
                C[m] = np.clip(C[m] + (U[m] - V[m].dot(C)) /
                               V[m, m], 0, np.inf)
        return C

def HALS4shape(Yr, A, C, iters=2):
        U = C.dot(Yr.T)
        V = C.dot(C.T)
        for _ in range(iters):
            for m in range(A.shape[-1]):  # background
                A[:, m] = np.clip(A[:, m] + ((U[m] - V[m].dot(A.T)) /
                                                     V[m, m]), 0, np.inf)
        return A
#%%
b = np.array([cv2.resize(b_in[:,ii].reshape(dims_bg, order='F'),(a.shape[2],a.shape[1])).flatten(order='F') for ii in range(n_b)]) .T
f = HALS4activity(Yr.T, b, cv2.resize(f_in,(a.shape[0],n_b)), iters=5)
b = HALS4shape(Yr.T, b, f, iters=5)
f = HALS4activity(Yr.T, b, f, iters=5)
#%%
for i in range(n_b):
    pl.subplot(1,n_b ,i+1)
    pl.imshow(b.reshape((a.shape[1],a.shape[2])+(-1,),order='F')[:,:,i],vmax = np.mean(b[:,i])*3,cmap='gray')
