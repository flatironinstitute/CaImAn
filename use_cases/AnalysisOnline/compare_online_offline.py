#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 09:54:35 2017

@author: agiovann
"""

#%%
from __future__ import division
from __future__ import print_function
from builtins import zip
from builtins import str
from builtins import map
from builtins import range
from past.utils import old_div
import cv2

try:
    cv2.setNumThreads(1)
except:
    print('Open CV is naturally single threaded')

try:
    if __IPYTHON__:
        print((1))
        # this is used for debugging purposes only. allows to reload classes when changed
        get_ipython().magic('load_ext autoreload')
        get_ipython().magic('autoreload 2')
except NameError:
    print('Not IPYTHON')
    pass
from caiman.base.rois import nf_read_roi_zip
import os
import numpy as np
import pylab as pl
import caiman as cm
import scipy
from caiman.source_extraction.cnmf import cnmf as cnmf
from caiman.components_evaluation import evaluate_components
from caiman.utils.visualization import plot_contours,view_patches_bar
from caiman.base.rois import extract_binary_masks_blob
#%%
c,dview,n_processes = cm.cluster.setup_cluster(backend = 'local',n_processes = None,single_thread = True)
#%%
min_size_neuro = 50
n_frames_per_bin = 10
#offline_file = '/opt/local/Data/JGauthier-J115/offline_results/results_analysis_offline_JEFF_90k.npz'
offline_file = '/opt/local/Data/Sue/k53/Yr_d1_512_d2_512_d3_1_order_C_frames_116043_.results_analysis.npz'

#%%
with np.load(offline_file) as ld:
    print(ld.keys())
    locals().update(ld) 
    dims_off = d1,d2    
    A_off = A[()]
    C_off = C
    b_off = b
    f_off = f

#%%
#if offline_file == '/opt/local/Data/Sue/k53/Yr_d1_512_d2_512_d3_1_order_C_frames_116043_.results_analysis.npz':
#    idx_components_raw = np.where(r_values >= .95)[0]
#    idx_components_raw = np.where(fitness_raw < -75)[0]
#    idx_components_delta = np.where(fitness_delta < -75)[0]
#    
#else:
idx_components_r = np.where(r_values >= .95)[0]
idx_components_raw = np.where(fitness_raw < -55)[0]
idx_components_delta = np.where(fitness_delta < -55)[0]


#min_radius = gSig[0] - 2
#masks_ws, idx_blobs, idx_non_blobs = extract_binary_masks_blob(
#    A.tocsc(), min_radius, dims, num_std_threshold=1,
#    minCircularity=0.7, minInertiaRatio=0.2, minConvexity=.5)

idx_components = np.union1d(idx_components_r, idx_components_raw)
idx_components = np.union1d(idx_components, idx_components_delta)
#idx_blobs = np.intersect1d(idx_components, idx_blobs)
idx_components_bad = np.setdiff1d(list(range(len(r_values))), idx_components)

print(' ***** ')
print((len(r_values)))
print((len(idx_components)))  
#%%
A_off = A_off.toarray()[:,idx_components]
#A_off = A_off[:,idx_components]
C_off = C_off[idx_components]
#    OASISinstances = OASISinstances[()]
#%%
crd = plot_contours(scipy.sparse.coo_matrix(A_off), Cn, thr=0.9)

#%%
view_patches_bar(None, scipy.sparse.coo_matrix(A_off[:, :]), C_off[:, :], b_off, f_off, dims_off[0], dims_off[1], YrA=YrA[:, :], img=Cn)
    
#%%
A_off_thr = cm.source_extraction.cnmf.spatial.threshold_components(A_off[:, :], dims_off, medw=None, thr_method='max', maxthr=0.2, nrgthr=0.99, extract_cc=True,
                         se=None, ss=None, dview=dview) 

A_off_thr = A_off_thr > 0  
size_neurons = A_off_thr.sum(0)
A_off_thr = A_off_thr[:,size_neurons>min_size_neuro]
C_off_thr = C_off[size_neurons>min_size_neuro,:116000]
print(A_off_thr.shape)

C_off_thr = np.array([CC.reshape([-1,n_frames_per_bin]).max(1) for CC in C_off_thr])
#%%  
pl.figure()
pl.imshow(A_off_thr.sum(-1).reshape(dims_off,order = 'F'))

#%%
#with np.load('results_full_movie_online_may5/results_analysis_online_JEFF_90k.take7_no_batch.npz') as ld:
#online_file ='/opt/local/privateCaImAn/JEFF_MAY_14_AFT_BETTER_INIT_UPDATES_NO_STATS/results_analysis_online_JEFF_LAST_90000.npz'
#online_file ='/mnt/ceph/neuro/DataForPublications/OnlineCNMF/Jeff/EP_linux/results_analysis_online_JEFF_LAST__DS_2_90000.npz'
online_file ='/mnt/ceph/neuro/SUE_results_online_DS/results_analysis_online_SUE__DS_2_116043.npz'

with np.load(online_file) as ld:

    print(ld.keys())
    locals().update(ld)
    Ab = Ab[()]
    dims_on = dims
    if not np.all(dims_on == dims_off):            
        Ab = scipy.sparse.csc_matrix(np.concatenate([cv2.resize(Ab[:,comp_id].toarray().reshape(dims_on, order= 'F'),dims_off[::-1]).reshape((-1,1),order = 'F') for comp_id in range(Ab.shape[-1])],axis=1))
        Cn = cv2.resize(Cn,dims_off[::-1])
        dims_on = dims_off
#    OASISinstances = OASISinstances[()]
    
    C_on = Cf
    A_on,b_on = Ab[:,:-1],Ab[:,-1].toarray()
    C_on, f_on = C_on[:-1,:], C_on[-1:,:]
    print(A_on.shape)
#%%
pl.figure()
pl.imshow(Cn)
crd = plot_contours(scipy.sparse.coo_matrix(A_on), Cn, thr=0.9)    
#%%
view_patches_bar(None, scipy.sparse.coo_matrix(A_on.tocsc()[:, :]), C_on[
                               :, :], b_on, f_on, dims_on[0], dims_on[1], YrA=noisyC[:-1]-C_on, img=Cn)    
#%%
A_on_thr = cm.source_extraction.cnmf.spatial.threshold_components(A_on.toarray(), dims_on, medw=None, thr_method='max', maxthr=0.2, nrgthr=0.99, extract_cc=True,
                         se=None, ss=None, dview=dview) 

A_on_thr  = A_on_thr  > 0  
size_neurons = A_on_thr.sum(0)
A_on_thr = A_on_thr[:,size_neurons>min_size_neuro]
C_on_thr = C_on[size_neurons>min_size_neuro,:116000]
print(A_on_thr.shape)
C_on_thr = np.array([CC.reshape([-1,n_frames_per_bin]).max(1) for CC in C_on_thr])
#%%  
pl.figure()
pl.imshow(A_on_thr.sum(-1).reshape(dims_on,order = 'F'))
#%% load labelers
roi_rs = nf_read_roi_zip('/mnt/ceph/neuro/labeling/k53_20160530/regions/sonia_active_regions.zip',Cn.shape)
print(roi_rs.shape)
roi_bs = nf_read_roi_zip('/mnt/ceph/neuro/labeling/k53_20160530/regions/lindsey_active_regions.zip',Cn.shape)
print(roi_bs.shape)
#roi_ds = nf_read_roi_zip('/mnt/ceph/neuro/labeling/J115_2015-12-09_L01/regions/natalia_active_regions_els.zip',Cn.shape)
#print(roi_ds.shape)
#%%
roi_rs = nf_read_roi_zip('/mnt/ceph/neuro/labeling/J115_2015-12-09_L01/regions/sonia_active_regions_els.zip',Cn.shape)
print(roi_rs.shape)
roi_bs = nf_read_roi_zip('/mnt/ceph/neuro/labeling/J115_2015-12-09_L01/regions/lindsey_active_regions_els.zip',Cn.shape)
print(roi_bs.shape)
roi_ds = nf_read_roi_zip('/mnt/ceph/neuro/labeling/J115_2015-12-09_L01/regions/natalia_active_regions_els.zip',Cn.shape)
print(roi_ds.shape)
#%%
plot_results = True
#%%
pl.figure(figsize=(30,20))
tp_gt, tp_comp, fn_gt, fp_comp, performance_rs_on =  cm.base.rois.nf_match_neurons_in_binary_masks(roi_rs,A_on_thr[:,:].reshape([dims_on[0],dims_on[1],-1],order = 'F').transpose([2,0,1])*1.,thresh_cost=.7, min_dist = 10,
                                                                              print_assignment= False,plot_results=plot_results ,Cn=Cn, labels = ['RS','Online'])
pl.rcParams['pdf.fonttype'] = 42
font = {'family' : 'Myriad Pro',
        'weight' : 'regular',
        'size'   : 20}
pl.rc('font', **font)
#%%
pl.figure(figsize=(30,20))
_,_, _, _, performance_rs_off =  cm.base.rois.nf_match_neurons_in_binary_masks(roi_rs,A_off_thr[:,:].reshape([dims_off[0],dims_off[1],-1],order = 'F').transpose([2,0,1])*1.,
                                                                               thresh_cost=.7, min_dist = 10, print_assignment= False ,plot_results=plot_results ,Cn=Cn, labels = ['RS','offline'])
pl.rcParams['pdf.fonttype'] = 42
font = {'family' : 'Myriad Pro',
        'weight' : 'regular',
        'size'   : 20}
pl.rc('font', **font)

#%%
#pl.figure(figsize=(30,20))
#_,_, _, _, performance_ds_on =  cm.base.rois.nf_match_neurons_in_binary_masks(roi_ds,A_on_thr[:,:].reshape([dims_on[0],dims_on[1],-1],order = 'F').transpose([2,0,1])*1.,thresh_cost=.7, min_dist = 10,
#                                                                              print_assignment= False,plot_results=plot_results ,Cn=Cn, labels = ['DS','Online'])
#pl.rcParams['pdf.fonttype'] = 42
#font = {'family' : 'Myriad Pro',
#        'weight' : 'regular',
#        'size'   : 20}
#pl.rc('font', **font)
##%%
#pl.figure(figsize=(30,20))
#_,_, _, _, performance_ds_off =  cm.base.rois.nf_match_neurons_in_binary_masks(roi_ds,A_off_thr[:,:].reshape([dims_off[0],dims_off[1],-1],order = 'F').transpose([2,0,1])*1.,
#                                                                               thresh_cost=.7, min_dist = 10, print_assignment= False,plot_results=plot_results ,Cn=Cn, labels = ['DS','offline'])
#pl.rcParams['pdf.fonttype'] = 42
#font = {'family' : 'Myriad Pro',
#        'weight' : 'regular',
#        'size'   : 20}
#pl.rc('font', **font)




#%%
pl.figure(figsize=(30,20))
_,_, _, _, performance_rs_bs =  cm.base.rois.nf_match_neurons_in_binary_masks(roi_rs,roi_bs,
                                                                               thresh_cost=.7, min_dist = 10, print_assignment= False,plot_results=plot_results ,Cn=Cn, labels = ['RS','BS'])
pl.rcParams['pdf.fonttype'] = 42
font = {'family' : 'Myriad Pro',
        'weight' : 'regular',
        'size'   : 20}
pl.rc('font', **font)
#%%
#pl.figure(figsize=(30,20))
#_,_, _, _, performance_ds_bs =  cm.base.rois.nf_match_neurons_in_binary_masks(roi_ds,roi_bs,
#                                                                               thresh_cost=.7, min_dist = 10, print_assignment= False,plot_results=plot_results ,Cn=Cn, labels = ['DS','BS'])
#pl.rcParams['pdf.fonttype'] = 42
#font = {'family' : 'Myriad Pro',
#        'weight' : 'regular',
#        'size'   : 20}
#pl.rc('font', **font)
#%%
#pl.figure(figsize=(30,20))
#_,_, _, _, performance_ds_rs =  cm.base.rois.nf_match_neurons_in_binary_masks(roi_ds,roi_rs,
#                                                                               thresh_cost=.7, min_dist = 10, print_assignment= False,plot_results=plot_results ,Cn=Cn, labels = ['DS','RS'])
#pl.rcParams['pdf.fonttype'] = 42
#font = {'family' : 'Myriad Pro',
#        'weight' : 'regular',
#        'size'   : 20}
#pl.rc('font', **font)
#%%
pl.figure(figsize=(30,20))
_,_, _, _, performance_bs_on =  cm.base.rois.nf_match_neurons_in_binary_masks(roi_bs,A_on_thr[:,:].reshape([dims_on[0],dims_on[1],-1],order = 'F').transpose([2,0,1])*1.,thresh_cost=.7, min_dist = 10,
                                                                              print_assignment= False,plot_results=plot_results ,Cn=Cn, labels = ['BS','Online'])
pl.rcParams['pdf.fonttype'] = 42
font = {'family' : 'Myriad Pro',
        'weight' : 'regular',
        'size'   : 20}
pl.rc('font', **font)
#%%
pl.figure(figsize=(30,20))
_,_, _, _, performance_bs_off =  cm.base.rois.nf_match_neurons_in_binary_masks(roi_bs,A_off_thr[:,:].reshape([dims_off[0],dims_off[1],-1],order = 'F').transpose([2,0,1])*1.,
                                                                               thresh_cost=.7, min_dist = 10, print_assignment= False,plot_results=True,Cn=Cn, labels = ['BS','offline'])
pl.rcParams['pdf.fonttype'] = 42
font = {'family' : 'Myriad Pro',
        'weight' : 'regular',
        'size'   : 20}
pl.rc('font', **font)
#%%
pl.figure(figsize=(30,20))
idx_tp_gt,idx_tp_comp, idx_fn, idx_fp_comp, performance_off_on =  cm.base.rois.nf_match_neurons_in_binary_masks(A_off_thr[:,:].reshape([dims_off[0],dims_off[1],-1],order = 'F').transpose([2,0,1])*1.,A_on_thr[:,:].reshape([dims_on[0],dims_on[1],-1],order = 'F').transpose([2,0,1])*1.,thresh_cost=.7, min_dist = 10, print_assignment= False,plot_results=True,Cn=Cn, labels = ['online','offline'])
pl.rcParams['pdf.fonttype'] = 42
font = {'family' : 'Myriad Pro',
        'weight' : 'regular',
        'size'   : 20}
pl.rc('font', **font)

#%%
corrs = np.array([scipy.stats.pearsonr(C_off_thr[gt,:],C_on_thr[comp,:])[0] for gt,comp in zip(idx_tp_gt,idx_tp_comp)])
#%%

pl.figure()
pl.hist(corrs,100)
pl.xlabel('Pearson\'s r')
pl.ylabel('Cell count')
pl.rcParams['pdf.fonttype'] = 42
font = {'family' : 'Myriad Pro',
        'weight' : 'regular',
        'size'   : 20}
pl.rc('font', **font)
#%%
hist, bin_edges = np.histogram(corrs, bins =100,normed=True)
pl.plot(bin_edges[1:],np.cumsum(hist)/np.sum(hist))
pl.xlabel('Correlation')
pl.ylabel('Probabillity')
#%%
pl.figure()
poor_perf = np.where(np.logical_and(corrs>0.85,corrs<=.9))[0]
for poor in poor_perf[:]:
    
    pl.subplot(1,2,1)
    pl.cla()
    if idx_tp_comp[poor]<300:
        continue
#    if idx_tp_gt[poor] not in [250,134,156,174]:
#        continue
    on, off  = np.array(C_on_thr[idx_tp_comp[poor],:]).T,  np.array(C_off_thr[idx_tp_gt[poor],:]).T
    pl.plot((on-np.min(on))/(np.max(on)-np.min(on)))
    pl.plot((off-np.min(off))/(np.max(off)-np.min(off)))
    pl.title(str(corrs[poor]))
    pl.legend(['Online','Offline'])
    pl.xlabel('Frames (eventually downsampled)')
    pl.ylabel('A.U.')
    pl.subplot(1,2,2)
    pl.cla()
    on, off  = A_on_thr[:,idx_tp_comp[poor]].copy(),  A_off_thr[:,idx_tp_gt[poor]].copy()
    on = on/np.max(on)
    off = off/np.max(off)
    pl.imshow(Cn,cmap = 'gray', vmax = np.percentile(Cn,90))
    pl.imshow(on.reshape(dims_on,order = 'F'),cmap = 'Blues',alpha = .3, vmax = 1 )
    pl.imshow(off.reshape(dims_on,order = 'F'),cmap = 'hot',alpha = .3, vmax = 3)
    print(idx_tp_gt[poor])
    pl.rcParams['pdf.fonttype'] = 42
    font = {'family' : 'Myriad Pro',
            'weight' : 'regular',
            'size'   : 20}
    pl.rc('font', **font)
    pl.pause(.1)
    pl.waitforbuttonpress(-1)
#%%
neur_to_print = [250,310,174]
poor_perf = np.where(np.logical_and(corrs>0.75,corrs<=1))[0]
for poor in poor_perf[:]:
    
    if idx_tp_gt[poor] not in neur_to_print:
        continue

    pl.figure()
    pl.subplot(2,1,1)
    on, off  = A_on_thr[:,idx_tp_comp[poor]].copy(),  A_off_thr[:,idx_tp_gt[poor]].copy()
    pl.imshow(Cn,cmap = 'gray', vmax = np.percentile(Cn,90))
    pl.imshow(on.reshape(dims_on,order = 'F'),cmap = 'Blues',alpha = .3, vmax = 1 )
    pl.imshow(off.reshape(dims_on,order = 'F'),cmap = 'hot',alpha = .3, vmax = 3)
    pl.ylim([180,310])
    pl.xlim([0,130])
    pl.subplot(2,1,2)
    on, off  = np.array(C_on_thr[idx_tp_comp[poor],:]).T,  np.array(C_off_thr[idx_tp_gt[poor],:]).T

    on = on/np.max(on)
    off = off/np.max(off)
    on = np.mean(np.reshape(on,(10,-1),order = 'F'),0)
    off = np.mean(np.reshape(off,(10,-1),order = 'F'),0)
    pl.plot(on)
    pl.plot(off)
    pl.title(str(corrs[poor]))
    pl.rcParams['pdf.fonttype'] = 42
        
    
    

        
