#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 18:08:25 2017

@author: agiovann
"""

from __future__ import division
from __future__ import print_function
from builtins import zip
from builtins import str
from builtins import map
from builtins import range
from past.utils import old_div
import cv2
import glob

try:
    cv2.setNumThreads(1)
except:
    print('Open CV is naturally single threaded')

try:
    if __IPYTHON__:
        print(1)
        # this is used for debugging purposes only. allows to reload classes
        # when changed
        get_ipython().magic('load_ext autoreload')
        get_ipython().magic('autoreload 2')
except NameError:
    print('Not IPYTHON')
    pass
import caiman as cm
import numpy as np
import os
import time
import pylab as pl
import psutil
import sys
from ipyparallel import Client
from skimage.external.tifffile import TiffFile
import scipy
import copy

from caiman.utils.utils import download_demo
from caiman.base.rois import extract_binary_masks_blob
from caiman.utils.visualization import plot_contours, view_patches_bar
from caiman.source_extraction.cnmf import cnmf as cnmf
from caiman.motion_correction import MotionCorrect
from caiman.components_evaluation import estimate_components_quality

from caiman.components_evaluation import evaluate_components

from caiman.tests.comparison import comparison
from caiman.motion_correction import tile_and_correct, motion_correction_piecewise

import glob
from caiman.base.rois import com
#from keras.preprocessing.image import ImageDataGenerator
from caiman.utils.image_preprocessing_keras import ImageDataGenerator
#%%
training_files = [os.path.join(dp, f) for dp, dn, filenames in os.walk('/mnt/ceph/neuro/labeling/') for f in filenames if 'set.npz' in f]
print(training_files)
#%%
crop_size = 50
half_crop = crop_size//2
zoom = [.7,1.,1.]
from sklearn.preprocessing import normalize
id_file=0
folder = '/mnt/xfs1/home/agiovann/SOFTWARE/CaImAn/images_examples'
all_masks_gt = []
labels_gt = []
for fl in training_files:
    
    with np.load(fl) as ld:
        print(ld.keys())
        locals().update(ld)
        fname_new = fname_new[()]
        name_base = os.path.split(fname_new)[-1][:-5]
#        pl.figure()
#        pl.subplot(1, 3, 1)
#        pl.imshow(A_matched[()].sum(1).reshape(dims,order='F'), vmax = A_matched[()].max()*.2)
#        pl.subplot(1, 3, 2)
#        pl.imshow(A_unmatched[()].sum(1).reshape(dims,order='F'), vmax = A_unmatched[()].max()*.2)    
#        pl.subplot(1, 3, 3)
#        pl.imshow(A_negative[()].sum(1).reshape(dims,order='F'), vmax = A_negative[()].max()*.2)    
        
#        coms = com(scipy.sparse.coo_matrix(A_matched[()]), dims[0], dims[1])
        if 'sparse' in str(type(A_matched[()])):
            A_matched = A_matched[()].toarray()
            A_unmatched = A_unmatched[()].toarray()
            A_negative = A_negative[()].toarray()
        
#        A_matched = normalize(A_matched,axis=0)
#        A_unmatched = normalize(A_unmatched,axis=0)
#        A_negative = normalize(A_negative,axis=0)
            
        masks_gt = np.concatenate([A_matched.reshape(tuple(dims)+(-1,),order = 'F').transpose([2,0,1]),A_unmatched.reshape(tuple(dims)+(-1,),order = 'F').transpose([2,0,1]),A_negative.reshape(tuple(dims)+(-1,),order = 'F').transpose([2,0,1])],axis=0)
        labels_gt = np.concatenate([labels_gt,np.ones(A_matched.shape[-1]),np.ones(A_unmatched.shape[-1]),np.zeros(A_negative.shape[-1])])
        
        coms = [ scipy.ndimage.center_of_mass(mm) for mm in masks_gt]
        coms = np.maximum(coms,half_crop )
        coms = np.array([np.minimum(cm,dims-half_crop) for cm in coms])
        
        
            
        count_neuro=0
        for com,img in zip(coms,masks_gt):
#            if zoom and zoom[counter]==1:
#            if zoom>1:
#                
#            elif zoom<1:
            crop_img = img[com[0]-half_crop:com[0]+half_crop, com[1]-half_crop:com[1]+half_crop].copy() # Crop from x, y, w, h -> 100, 200, 300, 400
#            crop_img = cv2.resize(crop_img,dsize=None,fx=zoom[id_file],fy=zoom[id_file])
#            newshape = np.array(crop_img.shape)//2
#            crop_img = crop_img[newshape[0]-half_crop:newshape[0]+half_crop,newshape[0]-half_crop:newshape[0]+half_crop] 
            
            borders = np.array(crop_img.shape)
            img_tmp = np.zeros_like(crop_img)
            crop_img = cv2.resize(crop_img,dsize=None,fx=zoom[id_file],fy=zoom[id_file])

            deltaw=(half_crop*2-crop_img.shape[0])//2
            deltah=(half_crop*2-crop_img.shape[1])//2
            img_tmp[deltaw:deltaw+crop_img.shape[0],deltah:deltah+crop_img.shape[1]] = crop_img
            crop_img = img_tmp
            crop_img = crop_img/np.linalg.norm(crop_img)
            all_masks_gt.append(crop_img[np.newaxis,:,:,np.newaxis])
            augment_test = False
            cv2.imshow("cropped", cv2.resize(crop_img,(480,480))*10)
            cv2.waitKey(1)
            if augment_test:
                datagen = ImageDataGenerator(
    #            featurewise_center=True,
    #            featurewise_std_normalization=True,
                shear_range=0.3,
                rotation_range=360,
                width_shift_range=0.2,
                height_shift_range=0.2,
                zoom_range = [.5,2],
                horizontal_flip=True,
                vertical_flip=True,
                random_mult_range = [.25,2]
                )
                
                count_neuro+=1
                for x_batch, y_batch in datagen.flow(np.repeat(crop_img[np.newaxis,:,:],10,0)[:,:,:,None], [1,1,1,1,1,1,1,1,0,0], batch_size=10):
                    print(y_batch)
                    for b_img in x_batch:
                        cv2.imshow("cropped", cv2.resize(b_img.squeeze(),(480,480))*10)
                        cv2.waitKey(300)
                        count_neuro+=1
                        print(count_neuro)
                    break    
                            
            
            
#            crop_img = cv2.resize(crop_img,dsize=None,fx=2,fy=2)
#            newshape = np.array(crop_img.shape)//2
#            crop_img = crop_img[newshape[0]-half_crop:newshape[0]+half_crop,newshape[0]-half_crop:newshape[0]+half_crop] 
            # NOTE: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]
            

        id_file+=1
        
all_masks_gt = np.vstack(all_masks_gt)
#%%
cm.movie(np.squeeze(all_masks_gt[labels_gt==0])).play(gain=3., magnification = 5)
#%%
np.savez('ground_truth_comoponents.npz',all_masks_gt = all_masks_gt,labels_gt = labels_gt)
#%%
import itertools

def grouper(n, iterable, fillvalue=None):
    "grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return itertools.zip_longest(*args, fillvalue=fillvalue)
#%% curate once more. Remove wrong negatives
negatives = np.where(labels_gt==1)[0]
wrong = []
count = 0
for a in grouper(50,negatives):
    print(np.max(a))
    print(count)
    a = np.array(a)[np.array(a)>0].astype(np.int)
    count+=1
    img_mont_ = all_masks_gt[np.array(a)].squeeze()
    shps_img = img_mont_.shape
    img_mont = montage2d(img_mont_)
    shps_img_mont = np.array(img_mont.shape)//50
    pl.figure(figsize=(20,30))
    pl.imshow(img_mont)    
    inp = pl.ginput(n=0,timeout = -100000)    
    imgs_to_exclude = []
    inp = np.ceil(np.array(inp)/50).astype(np.int)-1
    if len(inp)>0:
        
        imgs_to_exclude = img_mont_[np.ravel_multi_index([inp[:,1],inp[:,0]],shps_img_mont)]
#        pl.imshow(montage2d(imgs_to_exclude))    
        wrong.append(np.array(a)[np.ravel_multi_index([inp[:,1],inp[:,0]],shps_img_mont)])
    np.save('temp_label_pos.npy',wrong)    
    pl.close()
#%%
pl.imshow(montage2d(all_masks_gt[np.concatenate(wrong)].squeeze()))
#%% 
lab_pos_wrong = np.load('temp_label_pos.npy')
lab_neg_wrong = np.load('temp_label_neg_plus.npy')

labels_gt_cur = labels_gt.copy()
labels_gt_cur[np.concatenate(lab_pos_wrong)] = 0
labels_gt_cur[np.concatenate(lab_neg_wrong)] = 1

np.savez('ground_truth_comoponents_curated.npz',all_masks_gt = all_masks_gt,labels_gt_cur = labels_gt_cur)
#%%
pl.imshow(montage2d(all_masks_gt[labels_gt_cur==0].squeeze()))
