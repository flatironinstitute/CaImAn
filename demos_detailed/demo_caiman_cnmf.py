# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 18:39:45 2016

@author: Andrea Giovannucci

For explanation consult at https://github.com/agiovann/Constrained_NMF/releases/download/v0.4-alpha/Patch_demo.zip
and https://github.com/agiovann/Constrained_NMF
"""
from __future__ import print_function
from builtins import str
from builtins import range
try:
    if __IPYTHON__:
        print('Debugging!')
        # this is used for debugging purposes only. allows to reload classes when changed
        get_ipython().magic('load_ext autoreload')
        get_ipython().magic('autoreload 2')
except NameError:
    print('Not IPYTHON')
    pass

import numpy as np
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
from caiman.source_extraction.cnmf.utilities import extract_DF_F
from caiman.components_evaluation import evaluate_components
from caiman.utils.visualization import plot_contours,view_patches_bar
from caiman.utils.utils import download_demo
from caiman.cluster import setup_cluster
# %% RUN ANALYSIS
c, dview, n_processes = setup_cluster(
    backend='local', n_processes=None, single_thread=False)
#%%
is_patches=True
is_dendrites=False 

if is_dendrites == True:
# THIS METHOd CAN GIVE POSSIBLY INCONSISTENT RESULTS ON SOMAS WHEN NOT USED WITH PATCHES    
    init_method = 'sparse_nmf' 
    alpha_snmf=10e1  # this controls sparsity
else:
    init_method = 'greedy_roi'
    alpha_snmf=None #10e2  # this controls sparsity

#%% FOR LOADING ALL TIFF FILES IN A FILE AND SAVING THEM ON A SINGLE MEMORY MAPPABLE FILE
fnames = ['demoMovieJ.tif']
base_folder = './example_movies/'  # folder containing the demo files

download_demo(fnames[0])
fnames = [os.path.abspath(os.path.join(base_folder,fnames[0]))]
# TODO: todocument
m_orig = cm.load_movie_chain(fnames[:1])
print(fnames)  
fnames=fnames
#%%
#idx_x=slice(12,500,None)
#idx_y=slice(12,500,None)
#idx_xy=(idx_x,idx_y)

add_to_movie=300 # the movie must be positive!!!
downsample_factor=1 # use .2 or .1 if file is large and you want a quick answer
idx_xy=None
base_name='Yr'
name_new=cm.save_memmap_each(fnames, dview=dview,base_name=base_name, resize_fact=(1, 1, downsample_factor), remove_init=0,idx_xy=idx_xy,add_to_movie=add_to_movie)
name_new.sort()
print(name_new)
#%%
if len(name_new)>1:
    fname_new = cm.save_memmap_join(name_new, base_name='Yr', n_chunks=12, dview=dview)
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
if np.min(images)<0:
    raise Exception('Movie too negative, add_to_movie should be larger')
if np.sum(np.isnan(images))>0:
    raise Exception('Movie contains nan! You did not remove enough borders')   
#%%
Cn = cm.local_correlations(Y[:,:,:3000])
pl.imshow(Cn,cmap='gray')  

#%%
if not is_patches:
    #%%
    K = 35  # number of neurons expected per patch
    gSig = [7, 7]  # expected half size of neurons
    merge_thresh = 0.8  # merging threshold, max correlation allowed
    p = 2  # order of the autoregressive system
    cnm = cnmf.CNMF(n_processes, method_init=init_method, k=K, gSig=gSig, merge_thresh=merge_thresh,
                    p=p, dview=dview, Ain=None,method_deconvolution='oasis',skip_refinement = False)
    cnm = cnm.fit(images)
    crd = plot_contours(cnm.A, Cn, thr=0.9)
    C_dff = extract_DF_F(Yr, cnm.A, cnm.C, cnm.bl, quantileMin = 8, frames_window = 200, dview = dview)
    pl.figure()
    pl.plot(C_dff.T)
    #%%
else:
    #%%
    rf = 14  # half-size of the patches in pixels. rf=25, patches are 50x50
    stride = 6  # amounpl.it of overlap between the patches in pixels
    K = 6  # number of neurons expected per patch
    gSig = [6, 6]  # expected half size of neurons
    merge_thresh = 0.8  # merging threshold, max correlation allowed
    p = 1  # order of the autoregressive system
    save_results = False
    #%% RUN ALGORITHM ON PATCHES

    cnm = cnmf.CNMF(n_processes, k=K, gSig=gSig, merge_thresh=0.8, p=0, dview=dview, Ain=None, rf=rf, stride=stride, memory_fact=1,
                    method_init=init_method, alpha_snmf=alpha_snmf, only_init_patch=True, gnb=1,method_deconvolution='oasis', low_rank_background = False)
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
    final_frate = 10# approx final rate  (after eventual downsampling )
    Npeaks = 10
    traces = C_tot + YrA_tot
    #        traces_a=traces-scipy.ndimage.percentile_filter(traces,8,size=[1,np.shape(traces)[-1]/5])
    #        traces_b=np.diff(traces,axis=1)
    fitness_raw, fitness_delta, erfc_raw, erfc_delta, r_values, significant_samples = evaluate_components(
        Y, traces, A_tot, C_tot, b_tot, f_tot, final_frate, remove_baseline=True, N=5, robust_std=False, Athresh=0.1, Npeaks=Npeaks,  thresh_C=0.3)

    idx_components_r = np.where(r_values >= .5)[0]
    idx_components_raw = np.where(fitness_raw < -40)[0]
    idx_components_delta = np.where(fitness_delta < -20)[0]

    idx_components = np.union1d(idx_components_r, idx_components_raw)
    idx_components = np.union1d(idx_components, idx_components_delta)
    idx_components_bad = np.setdiff1d(list(range(len(traces))), idx_components)

    print(('Keeping ' + str(len(idx_components)) +
           ' and discarding  ' + str(len(idx_components_bad))))
    #%%
    pl.figure()
    crd = plot_contours(A_tot.tocsc()[:, idx_components], Cn, thr=0.9)
    #%%
    A_tot = A_tot.tocsc()[:, idx_components]
    C_tot = C_tot[idx_components]
    #%%
    save_results = True
    if save_results:
        np.savez('results_analysis_patch.npz', A_tot=A_tot, C_tot=C_tot,
                 YrA_tot=YrA_tot, sn_tot=sn_tot, d1=d1, d2=d2, b_tot=b_tot, f=f_tot)

    #%%
    cnm = cnmf.CNMF(n_processes, k=A_tot.shape, gSig=gSig, merge_thresh=merge_thresh, p=p, dview=dview, Ain=A_tot, Cin=C_tot, b_in = b_tot,
                    f_in=f_tot, rf=None, stride=None, method_deconvolution='oasis', gnb = 1,  low_rank_background = False)
    cnm = cnm.fit(images)

#%
A, C, b, f, YrA, sn = cnm.A, cnm.C, cnm.b, cnm.f, cnm.YrA, cnm.sn

#%%
final_frate = 10

Npeaks = 10
traces = C + YrA
#        traces_a=traces-scipy.ndimage.percentile_filter(traces,8,size=[1,np.shape(traces)[-1]/5])
#        traces_b=np.diff(traces,axis=1)
fitness_raw, fitness_delta, erfc_raw, erfc_delta, r_values, significant_samples = \
    evaluate_components(Y, traces, A, C, b, f, final_frate, remove_baseline=True,
                                      N=5, robust_std=False, Athresh=0.1, Npeaks=Npeaks,  thresh_C=0.3)
#%%
from caiman.components_evaluation import evaluate_components_CNN
predictions,final_crops = evaluate_components_CNN(A,dims,gSig,model_name = 'use_cases/CaImAnpaper/cnn_model')
#%%
threshold = .95
from caiman.utils.visualization import matrixMontage
pl.figure()
matrixMontage(np.squeeze(final_crops[np.where(predictions[:,1]>=threshold)[0]]))
pl.figure()
matrixMontage(np.squeeze(final_crops[np.where(predictions[:,0]>=threshold)[0]]))
#%%
thresh = .95
idx_components_cnn = np.where(predictions[:,1]>=thresh)[0]

print(' ***** ')
print((len(final_crops)))
print((len(idx_components_cnn)))
#print((len(idx_blobs)))    
#%
idx_components_r = np.where((r_values >= .99))[0]
idx_components_raw = np.where(fitness_raw < -60)[0]
idx_components_delta = np.where(fitness_delta < -60)[0]   

bad_comps = np.where((r_values <= .2) | (fitness_raw >= -4) | (predictions[:,1]<=.05))[0]
 
#idx_and_condition_1 = np.where((r_values >= .65) & ((fitness_raw < -20) | (fitness_delta < -20)) )[0]

idx_components = np.union1d(idx_components_r, idx_components_raw)
idx_components = np.union1d(idx_components, idx_components_delta)
idx_components = np.union1d(idx_components,idx_components_cnn)
idx_components = np.setdiff1d(idx_components,bad_comps)
#idx_components = np.intersect1d(idx_components,idx_size_neuro)
#idx_components = np.union1d(idx_components, idx_and_condition_1)
#idx_components = np.union1d(idx_components, idx_and_condition_2)

#idx_blobs = np.intersect1d(idx_components, idx_blobs)
#idx_components = idx_components_cnn
idx_components_bad = np.setdiff1d(list(range(len(r_values))), idx_components)


print(' ***** ')
print((len(r_values)))
print((len(idx_components)))
#%%
save_results = True
if save_results:
    np.savez(os.path.join(os.path.split(fname_new)[0], 'results_analysis.npz'), Cn=Cn, A=A.todense(), C=C, b=b, f=f, YrA=YrA, sn=sn, d1=d1, d2=d2, idx_components=idx_components, idx_components_bad=idx_components_bad)

#%% visualize components
# pl.figure();
pl.subplot(1, 2, 1)
crd = plot_contours(A.tocsc()[:, idx_components], Cn, thr=0.9)
#pl.subplot(1, 3, 2)
#crd = plot_contours(A.tocsc()[:, idx_blobs], Cn, thr=0.9)
pl.subplot(1, 2, 2)
crd = plot_contours(A.tocsc()[:, idx_components_bad], Cn, thr=0.9)
#%%
view_patches_bar(Yr, scipy.sparse.coo_matrix(A.tocsc()[:, idx_components]), C[
                               idx_components, :], b, f, dims[0], dims[1], YrA=YrA[idx_components, :], img=Cn)
#%%
view_patches_bar(Yr, scipy.sparse.coo_matrix(A.tocsc()[:, idx_components_bad]), C[
                               idx_components_bad, :], b, f, dims[0], dims[1], YrA=YrA[idx_components_bad, :], img=Cn)
#%%
C_dff = extract_DF_F(Yr, A.tocsc()[:, idx_components], C[idx_components, :], cnm.bl[idx_components], quantileMin = 8, frames_window = 200, dview = dview)
pl.plot(C_dff.T)

#%% STOP CLUSTER and clean up log files
cm.stop_server()

log_files = glob.glob('Yr*_LOG_*')
for log_file in log_files:
    os.remove(log_file)

