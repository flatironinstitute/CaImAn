# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 10:50:33 2016

@author: agiovann
"""

#%%
try:
    if __IPYTHON__:
        # this is used for debugging purposes only. allows to reload classes when changed
        get_ipython().magic(u'load_ext autoreload')
        get_ipython().magic(u'autoreload 2')
except NameError:
    print('Not IPYTHON')
    pass

import sys
import numpy as np
import psutil
import glob
import os
import scipy
from ipyparallel import Client
import caiman as cm
from caiman.components_evaluation import evaluate_components
from caiman.utils.visualization import plot_contours,view_patches_bar
from caiman.base.rois import extract_binary_masks_blob
from caiman.source_extraction import cnmf as cnmf



# YI 1
folder = '/mnt/ceph/users/agiovann/ImagingData/LABELLING/Ben_labelled/Yi/Data 1'
file_mmap = 'Yr_d1_512_d2_512_d3_1_order_C_frames_410_.mmap'
ben_zip = '/mnt/ceph/users/agiovann/ImagingData/LABELLING/Ben_labelled/Yi/Data 1/Yi_data1_combined_2.zip'
princeton_zip = '/mnt/xfs1/home/agiovann/labeling/Yi.data.001/regions/princeton_regions.mat'
thresh_noise = -7

# YI 2
folder = '/mnt/ceph/users/agiovann/ImagingData/LABELLING/Ben_labelled/Yi/Data 2'
file_mmap = 'Yr_d1_512_d2_512_d3_1_order_C_frames_621_.mmap'
ben_zip = '/mnt/ceph/users/agiovann/ImagingData/LABELLING/Ben_labelled/Yi/Data 2/Yi_data2_combined.zip'
princeton_zip = '/mnt/xfs1/home/agiovann/labeling/Yi.data.002/regions/princeton_regions.mat'
thresh_noise = -7
# AMG
folder = '/mnt/ceph/users/agiovann/ImagingData/LABELLING/Ben_labelled/Jan_Convnet/AMG1_2016_02_08'
file_mmap = 'Yr_d1_512_d2_512_d3_1_order_C_frames_3201_.mmap'
ben_zip = '/mnt/ceph/users/agiovann/ImagingData/LABELLING/Ben_labelled/Jan_Convnet/AMG1_2016_02_08/All_13.zip'
princeton_zip = '/mnt/xfs1/home/agiovann/labeling/Jan-AMG1_exp2_new_001/regions/princeton_regions.mat'
thresh_noise = -20

# AMG 2
folder = '/mnt/ceph/users/agiovann/ImagingData/LABELLING/Ben_labelled/Jan_Convnet/AMG2_2016_02_09'
file_mmap = 'Yr_d1_512_d2_512_d3_1_order_C_frames_3817_.mmap'
ben_zip = '/mnt/ceph/users/agiovann/ImagingData/LABELLING/Ben_labelled/Jan_Convnet/AMG2_2016_02_09/All_14.zip'
princeton_zip = '/mnt/xfs1/home/agiovann/labeling/Jan-AMG_exp3_001/regions/princeton_regions.mat'
thresh_noise = -20

# Jan 25
folder = '/mnt/ceph/users/agiovann/ImagingData/LABELLING/Ben_labelled/Jan_Convnet/Jan25 2015_07_13'
file_mmap = 'Yr_d1_512_d2_512_d3_1_order_C_frames_3587_.mmap'
ben_zip = '/mnt/ceph/users/agiovann/ImagingData/LABELLING/Ben_labelled/Jan_Convnet/Jan25 2015_07_13/All_19.zip'
princeton_zip = '/mnt/xfs1/home/agiovann/labeling/Jan25_2015_07_13/regions/princeton_regions.mat'
thresh_noise = -20

# Jan 40
folder = '/mnt/ceph/users/agiovann/ImagingData/LABELLING/Ben_labelled/Jan_Convnet/Jan40_2016_03_30'
file_mmap = 'Yr_d1_512_d2_512_d3_1_order_C_frames_3576_.mmap'
ben_zip = '/mnt/ceph/users/agiovann/ImagingData/LABELLING/Ben_labelled/Jan_Convnet/Jan40_2016_03_30/All_6.zip'
princeton_zip = '/mnt/xfs1/home/agiovann/labeling/Jan40_exp2_001/regions/princeton_regions.mat'
thresh_noise = -20

# Jan 42
folder = '/mnt/ceph/users/agiovann/ImagingData/LABELLING/Ben_labelled/Jan_Convnet/Jan42_2016_04_06_Green'
file_mmap = 'Yr_d1_512_d2_512_d3_1_order_C_frames_3699_.mmap'
ben_zip = '/mnt/ceph/users/agiovann/ImagingData/LABELLING/Ben_labelled/Jan_Convnet/Jan42_2016_04_06_Green/All_9.zip'
princeton_zip = '/mnt/xfs1/home/agiovann/labeling/Jan42_exp4_001/regions/princeton_regions.mat'
thresh_noise = -20
#%%
import pylab as pl
pl.ion()    
    
with np.load(os.path.join(folder,'results_analysis.npz')) as ld:
    locals().update(ld)

A=scipy.sparse.coo_matrix(A)    

Yr, dims, T = cm.load_memmap(os.path.join(folder,file_mmap))
d1, d2 = dims
images = np.reshape(Yr.T, [T] + list(dims), order='F')
Y = np.reshape(Yr, dims + (T,), order='F')
gSig = [8, 8]  # expected half size of neurons
final_frate=1
with np.load(os.path.join(folder,'results_blobs.npz')) as ld:
    print ld.keys()
    locals().update(ld)
    
masks_cnmf=masks[idx_blob]  
  
#%%
crd = plot_contours(A.tocsc()[:, idx_components], Cn, thr=0.9)
#%%
masks_ben=cm.base.rois.nf_read_roi_zip(ben_zip,dims)
if '.mat' in princeton_zip:
    aa = scipy.io.loadmat(princeton_zip) 
    try:
        masks_princeton = aa['allROIs'].transpose([2,0,1])*1.
    except:
        masks_princeton = aa['M'].transpose([2,0,1])*1.    
else:
    masks_princeton=cm.base.rois.nf_read_roi_zip(princeton_zip,dims)*1.
#masks_ben_ci=cm.base.rois.nf_read_roi_zip(os.path.join(folder,ben_zip_ci),dims)

##%%
#pl.imshow(masks_ben.sum(0))
#pl.imshow(masks_cnmf.sum(0),cmap='hot',alpha=.3)
#pl.imshow(masks_princeton.sum(0),cmap='Greens',alpha=.3)

#%%
#images=np.array(images)
Yr=np.array(Yr)
#%%
traces_ben = cm.base.rois.mask_to_2d(masks_ben).T.dot(Yr)
traces_princeton = cm.base.rois.mask_to_2d(masks_princeton).T.dot(Yr)

traces_ben=traces_ben-scipy.ndimage.percentile_filter(traces_ben,8,size=[1,np.shape(traces_ben)[-1]/5])
traces_princeton=traces_princeton-scipy.ndimage.percentile_filter(traces_princeton,8,size=[1,np.shape(traces_princeton)[-1]/5])


fitness_ben,_,_=cm.components_evaluation.compute_event_exceptionality(traces_ben)
fitness_princeton,_,_=cm.components_evaluation.compute_event_exceptionality(traces_princeton)

masks_ben=masks_ben[fitness_ben<thresh_noise]
masks_princeton=masks_princeton[fitness_princeton<thresh_noise]
#%%
#pl.plot(fitness_ben,'.')
#pl.plot(fitness_princeton,'.')
#
##pl.plot(fitness_ben_ci,'r.')
##%%
#pl.imshow(masks_ben.sum(0))
#pl.imshow(masks_cnmf.sum(0),cmap='hot',alpha=.5)
#pl.imshow(masks_princeton.sum(0),cmap='greens',alpha=.5)

#%%
#for m1,m2,cost in zip(matches[0],matches[1],costs):
#    print m1,m2
#    if cost > .55 and cost < .75:
#        pl.imshow(masks_ben[m1])
#        pl.imshow(masks_cnmf[m2],alpha=.5)
#        pl.xlabel('COST:'+ str(cost))
#        pl.pause(1)
#        pl.cla()
        
#%%


#print np.shape(masks_ben)[0]        
#print np.shape(masks_cnmf)[0]    
    


#%%
pl.figure(figsize=(30,20))
idx_tp_gt,idx_tp_comp, idx_fn, idx_fp_comp, performance =  cm.base.rois.nf_match_neurons_in_binary_masks(masks_ben,masks_cnmf,thresh_cost=.7, min_dist = 10, print_assignment= False,plot_results=True,Cn=Cn)
masks_gt = masks_ben
masks_comp = masks_cnmf
comp_str = 'cnmf'
pl.savefig('ben_cnmf.pdf')
#%%
pl.figure(figsize=(30,20))
idx_tp_gt,idx_tp_comp, idx_fn, idx_fp_comp, performance =  cm.base.rois.nf_match_neurons_in_binary_masks(masks_ben,masks_princeton,thresh_cost=.7, min_dist = 10, print_assignment= False,plot_results=True,Cn=Cn)
masks_gt = masks_ben
masks_comp = masks_princeton
comp_str = 'manual'
pl.savefig('ben_princeton.pdf')
#%%
pl.figure(figsize=(30,20))
idx_tp_gt,idx_tp_comp, idx_fn, idx_fp_comp, performance =  cm.base.rois.nf_match_neurons_in_binary_masks(masks_princeton,masks_cnmf,thresh_cost=.7, min_dist = 10, print_assignment= False,plot_results=True,Cn=Cn)
masks_gt = masks_princeton
masks_comp = masks_cnmf
comp_str = 'cnmf'
pl.savefig('princeton_cnmf.pdf')
#%% 
pl.rcParams['pdf.fonttype'] = 42
font = {'family' : 'Myriad Pro',
        'weight' : 'regular',
        'size'   : 20}
pl.rc('font', **font)

pl.subplot(1,2,1)
pl.imshow(Cn)
[pl.contour(mm,levels=[0],colors='w',linewidths=2) for mm in masks_comp[idx_tp_comp]] 
[pl.contour(mm,levels=[0],colors='r',linewidths=2) for mm in masks_gt[idx_tp_gt]] 
#pl.legend([comp_str,'GT'])
pl.axis('off')
pl.subplot(1,2,2)
pl.imshow(Cn)
[pl.contour(mm,levels=[0],colors='w',linewidths=2) for mm in masks_comp[idx_fp_comp]] 
[pl.contour(mm,levels=[0],colors='r',linewidths=2) for mm in masks_gt[idx_fn]] 
#pl.legend([comp_str,'GT'])
pl.axis('off')
#pl.subplot(1,3,3)
#pl.imshow(np.reshape(A.tocsc()[:,idx_components[idx_fp_cnmf]].sum(-1),(d1,d2),order='F'))
#%%
#pl.subplot(1,3,1)
#pl.imshow(masks_cnmf[idx_tp_comp].sum(0))
#pl.imshow(masks_ben[idx_tp_gt].sum(0),alpha=.5,cmap='hot',vmax=3)
#pl.subplot(1,3,2)
#pl.imshow(masks_cnmf[idx_fp_cnmf].sum(0))
#pl.imshow(masks_ben[idx_fn].sum(0),cmap='hot',vmax=3,alpha=.5)
#
#pl.subplot(1,3,3)
#pl.imshow(np.reshape(A.tocsc()[:,idx_components[idx_fp_cnmf]].sum(-1),(d1,d2),order='F'))

#%%
#traces_fp = np.mean(masks_cnmf[idx_fp_cnmf,np.newaxis,:,:]*images,(2,3))
#%%
#pl.plot(traces_fp.T)
#%%
#idx_components_fp = idx_components[idx_fp_cnmf]
#view_patches_bar(Yr, scipy.sparse.coo_matrix(A.tocsc()[:, idx_components_fp]), C[
#                               idx_components_fp, :], b, f, dims[0], dims[1], YrA=YrA[idx_components_fp, :], img=Cn)