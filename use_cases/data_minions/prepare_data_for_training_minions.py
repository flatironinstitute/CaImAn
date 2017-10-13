#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 10:47:55 2017

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
from skimage.util.montage import montage2d
from caiman.components_evaluation import evaluate_components

from caiman.tests.comparison import comparison
from caiman.motion_correction import tile_and_correct, motion_correction_piecewise

import glob
from caiman.base.rois import com
#from keras.preprocessing.image import ImageDataGenerator
from caiman.utils.image_preprocessing_keras import ImageDataGenerator
from sklearn.preprocessing import normalize
from keras.models import model_from_json

#%%
import itertools

def grouper(n, iterable, fillvalue=None):
    "grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return itertools.zip_longest(*args, fillvalue=fillvalue)
#%%
with np.load('/mnt/home/agiovann/Dropbox/Data_minions/ground_truth_components_minions.npz') as ld:
    print(ld.keys())
    locals().update(ld)
#%% Existing classifier
def run_classifier(msks, model_name = 'use_cases/CaImAnpaper/cnn_model'):
    json_file = open(model_name +'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(model_name +'.h5')
    print("Loaded model from disk")    
    return loaded_model.predict(msks, batch_size=32, verbose=1)

predictions = run_classifier(all_masks_gt)
#%% show results classifier
pl.figure(figsize=(20,30))
is_positive = 0
for a in grouper(100,np.where(predictions[:,is_positive]>.5)[0]):
     a_ = [aa for aa in a if aa is not None]
     img_mont_ = all_masks_gt[np.array(a_)].squeeze()
     shps_img = img_mont_.shape
     img_mont = montage2d(img_mont_)
     shps_img_mont = np.array(img_mont.shape)//50
     pl.imshow(img_mont)    
     inp = pl.pause(.1)    
     pl.cla()
#%% Curate data. Remove wrong negatives or wrong positives
is_positive = 0 # should be 0 when processing negatives
to_be_checked = np.where(labels_gt==is_positive)[0] 

wrong = []
count = 0
for a in grouper(50,to_be_checked):

    a_ = [aa for aa in a if aa is not None]
    a_ = np.array(a_)[np.array(a_)>0].astype(np.int)
    print(np.max(a_))
    print(count)
    count+=1
    img_mont_ = all_masks_gt[np.array(a_)].squeeze()
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
        wrong.append(np.array(a_)[np.ravel_multi_index([inp[:,1],inp[:,0]],shps_img_mont)])
    if is_positive:
        np.save('temp_label_pos_minions.npy',wrong)    
    else:
        np.save('temp_label_neg_minions.npy',wrong)    
    pl.close()







#%% 
pl.imshow(montage2d(all_masks_gt[np.concatenate(wrong)].squeeze()))
#%% 
lab_pos_wrong = np.load('temp_label_pos_minions.npy')
lab_neg_wrong = np.load('temp_label_neg_minions.npy')

labels_gt_cur = labels_gt.copy()
labels_gt_cur[np.concatenate(lab_pos_wrong)] = 0
labels_gt_cur[np.concatenate(lab_neg_wrong)] = 1

np.savez('ground_truth_components_curated_minions.npz',all_masks_gt = all_masks_gt,labels_gt_cur = labels_gt_cur)
#%%
pl.imshow(montage2d(all_masks_gt[labels_gt_cur==0].squeeze()))