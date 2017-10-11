#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 11:35:22 2017

@author: agiovann
"""
import os
import numpy as np
import caiman as cm
import scipy
import pylab as pl
import cv2
import glob
#%%
fname_new = '/mnt/ceph/neuro/labeling/neurofinder.03.00.test/images/final_map/Yr_d1_498_d2_467_d3_1_order_C_frames_2250_.mmap'
fname_new = '/mnt/ceph/neuro/labeling/neurofinder.04.00.test/images/final_map/Yr_d1_512_d2_512_d3_1_order_C_frames_3000_.mmap'
fname_new = '/mnt/ceph/neuro/labeling/neurofinder.02.00/images/final_map/Yr_d1_512_d2_512_d3_1_order_C_frames_8000_.mmap'
fname_new = '/mnt/ceph/neuro/labeling/yuste.Single_150u/images/final_map/Yr_d1_200_d2_256_d3_1_order_C_frames_3000_.mmap'
fname_new = '/mnt/ceph/neuro/labeling/neurofinder.00.00/images/final_map/Yr_d1_512_d2_512_d3_1_order_C_frames_2936_.mmap'
fname_new = '/mnt/ceph/neuro/labeling/neurofinder.01.01/images/final_map/Yr_d1_512_d2_512_d3_1_order_C_frames_1825_.mmap'

gt_file = os.path.join(os.path.split(fname_new)[0], os.path.split(fname_new)[1][:-4] + 'match_masks.npz')
#%% you guys sort out how to deal with these big beasts
fname_new = glob.glob('/mnt/ceph/neuro/labeling/k53_20160530/images/mmap/*.mmap')
fname_new.sort()
gt_file = '/mnt/ceph/neuro/labeling/k53_20160530/Yr_d1_512_d2_512_d3_1_order_C_frames_116043_.match_masks.npz'

fname_new = glob.glob('/mnt/ceph/neuro/labeling/J115_2015-12-09_L01_ELS/images/mmap/*.mmap')
fname_new.sort()
gt_file = ' /mnt/ceph/neuro/labeling/J115_2015-12-09_L01_ELS/images/final_map/Yr_d1_463_d2_472_d3_1_order_C_frames_90000_.match_masks.npz'

fname_new = glob.glob('/mnt/ceph/neuro/labeling/J123_2015-11-20_L01_0/images/mmap/*.mmap')
fname_new.sort()
gt_file = '/mnt/ceph/neuro/labeling/J123_2015-11-20_L01_0/images/final_map/Yr_d1_458_d2_477_d3_1_order_C_frames_41000_.match_masks.npz'
#%%
#
with np.load(gt_file, encoding = 'latin1') as ld:
    print(ld.keys())
    locals().update(ld)
    A_gt = scipy.sparse.coo_matrix(A_gt[()])
    dims = (d1,d2)
    try:
        fname_new = fname_new[()].decode('unicode_escape')    
    except:
        fname_new = fname_new[()]        

    A_gt_thr = cm.source_extraction.cnmf.spatial.threshold_components(A_gt.tocsc()[:,:].toarray(), dims, medw=None, thr_method='max', maxthr=0.2, nrgthr=0.99, extract_cc=True,
                     se=None, ss=None, dview=None)                
#%%
crd = cm.utils.visualization.plot_contours(A_gt, Cn, thr=.99)
#%%
Yr, dims, T = cm.load_memmap(fname_new)
d1, d2 = dims
images = np.reshape(Yr.T, [T] + list(dims), order='F')
# TODO: needinfo
Y = np.reshape(Yr, dims + (T,), order='F')
m_orig  = cm.movie(images)
#%%
idx_exclude = np.arange(100)
idx_comps = np.setdiff1d(np.arange(A_gt.shape[-1]),idx_exclude)
final_frate = 10
r_values_min = .8  # threshold on space consistency
fitness_min = -20  # threshold on time variability    
fitness_delta_min = -20
Npeaks = 5
#%%
traces_gt = YrA_gt + C_gt
downsampfact = 500
elm_missing=int(np.ceil(T*1.0/downsampfact)*downsampfact-T)
padbefore=int(np.floor(elm_missing/2))
padafter=int(np.ceil(elm_missing/2))    
tr_tmp = np.pad(traces_gt.T,((padbefore,padafter),(0,0)),mode='reflect')
numFramesNew,num_traces = np.shape(tr_tmp)    
#% compute baseline quickly
print("binning data ..."); 
tr_BL=np.reshape(tr_tmp,(downsampfact,int(numFramesNew/downsampfact),num_traces),order='F');
tr_BL=np.percentile(tr_BL,8,axis=0)            
print("interpolating data ..."); 
print(tr_BL.shape)    
tr_BL=scipy.ndimage.zoom(np.array(tr_BL,dtype=np.float32),[downsampfact ,1],order=3, mode='constant', cval=0.0, prefilter=True)
if padafter==0:
    traces_gt -= tr_BL.T
else:
    traces_gt -= tr_BL[padbefore:-padafter].T


#traces_gt = scipy.signal.savgol_filter(traces_gt,5,2)          
#%%
fitness,exceptionality,sd_r,md = cm.components_evaluation.compute_event_exceptionality(traces_gt[idx_exclude],robust_std=False,N=5,use_mode_fast=False)
#%%
m_res = m_orig - cm.movie(np.reshape(A_gt.tocsc()[:,idx_comps].dot(C_gt[idx_comps]) + b_gt.dot(f_gt),dims+(-1,),order = 'F').transpose([2,0,1]))
#%%
max_mov = m_res.max() 
#%%
m_res.play()
 
#%%
count_start = 1
bin_ = 1
for count in range(count_start,T):
    img_temp = (m_res[count-count_start:count].copy().mean(0)/max_mov).astype(np.float32)
#    img_temp = m_res[count].copy().astype(np.float32)/max_mov
    active = np.where(exceptionality[:,count]<-20)[0]
    print(active)
    cms = [np.array(scipy.ndimage.center_of_mass(np.reshape(a.toarray(),dims,order = 'F'))).astype(np.int) for a in  A_gt.tocsc()[:,idx_exclude[active]].T]
    for cm__ in cms:
        cm_=cm__[::]
        img_temp = cv2.rectangle(img_temp,(cm_[1]-10, cm_[0]-10),(cm_[1]+10, cm_[0]+10),1)
    
    cv2.imshow('frame', cv2.resize(img_temp*3,(dims[1]*2,dims[0]*2)))
    cv2.waitKey(100)