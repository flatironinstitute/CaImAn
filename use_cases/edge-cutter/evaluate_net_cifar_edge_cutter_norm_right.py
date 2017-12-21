#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 11:42:13 2017

@author: agiovann
"""

"""
Created on Thu Aug 24 12:30:19 2017

@author: agiovann
"""

'''From keras example of convnet on the MNIST dataset.

Evaluate networks trained with train_net_cifar_edge_cutter_FOV.py and train_net_cifar_edge_cutter.py

'''
#%%
#import os
# os.chdir('/mnt/home/agiovann/SOFTWARE/CaImAn')
#from __future__ import division
#from __future__ import print_function
#from builtins import zip
#from builtins import str
#from builtins import map
#from builtins import range
#from past.utils import old_div
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
import keras
from keras.datasets import mnist
from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Dropout, Activation, Flatten

import json as simplejson
from keras.models import model_from_json
from sklearn.utils import class_weight as cw
from caiman.utils.image_preprocessing_keras import ImageDataGenerator
from keras.layers import merge
from keras.layers.core import Lambda
from keras.models import Model

import tensorflow as tf
#%% model FULL FOV
def get_conv(input_shape=(48,48,1), filename=None):
    model = Sequential()
#    model.add(Lambda(lambda x: (x-np.mean(x))/np.std(x),input_shape=input_shape, output_shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu', name='conv1', input_shape=input_shape, padding="same"))
    model.add(Conv2D(32, (3, 3), activation='relu', name='conv2', padding="same"))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(48, (3, 3), name = 'conv3', padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(48, (3, 3), name = 'conv4', padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.25))

    model.add(Conv2D(256,(8,8), activation="relu", name="dense1")) # This was Dense(128)
    model.add(Dropout(0.5))
    model.add(Conv2D(1, (1,1), name="dense2", activation="tanh")) # This was Dense(1)
    if filename:
        model.load_weights(filename)
    return model


#def get_conv(input_shape=(48, 48, 1), filename=None):
#    model = Sequential()
##    model.add(Lambda(lambda x: (x-np.mean(x))/np.std(x),input_shape=input_shape, output_shape=input_shape))
#    model.add(Conv2D(32, (3, 3), activation='relu', name='conv1',
#                     input_shape=input_shape, padding="same"))
#    model.add(Conv2D(32, (3, 3), activation='relu',
#                     name='conv2', padding="same"))
#    model.add(MaxPooling2D(pool_size=(4, 4)))
#    model.add(Dropout(0.25))
#
#    model.add(Conv2D(64, (3, 3), name='conv3', padding='same'))
#    model.add(Activation('relu'))
#    model.add(Conv2D(64, (3, 3), name='conv4', padding='same'))
#    model.add(Activation('relu'))
#    model.add(MaxPooling2D(pool_size=(4, 4)))
#    model.add(Dropout(0.25))
#
#    model.add(Conv2D(256, (3, 3), activation="relu",
#                     name="dense1"))  # This was Dense(128)
#    model.add(Dropout(0.5))
#    # This was Dense(1)
#    model.add(Conv2D(1, (1, 1), name="dense2", activation="tanh"))
#    if filename:
#        model.load_weights(filename)
#    return model


#heatmodel = get_conv(input_shape=(None,None,1), filename='use_cases/edge-cutter/residual_net_2classes_FOV.h5')

heatmodel = get_conv(input_shape=(
    None, None, 1), filename='use_cases/edge-cutter/residual_net_2classes_FOV.h5')

import matplotlib.pylab as plt


def locate(data, plot=False, rectangle=False, total_pooling=6, prob = 0.99):
    #    data = cv2.cvtColor(cv2.imread("test1.jpg"), cv2.COLOR_BGR2RGB)

    heatmap = 1 - \
        heatmodel.predict(data.reshape(
            1, data.shape[0], data.shape[1], data.shape[2]))

    if plot:
        plt.imshow(heatmap[0, :, :, 0])
        plt.title("Heatmap")
        plt.show()
        plt.imshow(heatmap[0, :, :, 0] > 0.99, cmap="gray")
        plt.title("Car Area")
        plt.show()

    if rectangle:
        xx, yy = np.meshgrid(
            np.arange(heatmap.shape[2]), np.arange(heatmap.shape[1]))
        x = (xx[heatmap[0, :, :, 0] > prob])
        y = (yy[heatmap[0, :, :, 0] > prob])

        for i, j in zip(x, y):
            cv2.rectangle(data, (i * total_pooling + 0, j * total_pooling + 0),
                          (i * total_pooling + 48, j * total_pooling + 48), 3)

    else:
        data = heatmap

    return heatmap, data


#%% MODEL CLASSIFIER ON PATCH
#json_path = 'use_cases/edge-cutter/residual_classifier_2classes.json'
#model_path = 'use_cases/edge-cutter/residual_classifier_2classes.h5'
json_path = 'use_cases/edge-cutter/residual_classifier_2classes.json'
model_path = 'use_cases/edge-cutter/residual_classifier_2classes.h5'
json_file = open(json_path, 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights(model_path)
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
loaded_model.compile(loss=keras.losses.categorical_crossentropy,
                     optimizer=opt,
                     metrics=['accuracy'])
#%%
#annotated,hmap = locate(data[:,:,None].copy())
#
# plt.title("Augmented")
# plt.imshow(annotated.squeeze())
# plt.show()
#%%
REGENERATE = False
if REGENERATE:
    total_crops = []
    total_labels = []
    patch_size = 50
    for fl in glob.glob('use_cases/edge-cutter/training_data/*.npz'):
        print(fl)
        try:
            with np.load(fl) as ld:
                all_neg_crops = ld['all_neg_crops']
                all_pos_crops = ld['all_pos_crops']
                all_dubious_crops = ld['all_dubious_crops']
                gSig = ld['gSig']
                for class_id, pos in enumerate([all_pos_crops, all_neg_crops, all_dubious_crops]):
                    pos = pos - np.median(pos, axis=(1, 2))[:, None, None]
                    pos = pos / np.std(pos, axis=(1, 2))[:, None, None]
                    total_crops += [cv2.resize(ain / np.linalg.norm(ain),
                                               (patch_size, patch_size)) for ain in pos]
                    total_labels += [class_id] * len(pos)

                print([len(all_neg_crops), len(
                    all_pos_crops), len(all_dubious_crops)])
        except:
            pass
    rand_perm = np.random.permutation(len(total_crops))
    total_crops = np.array(total_crops)[rand_perm]
    total_labels = np.array(total_labels)[rand_perm]
    np.savez('use_cases/edge-cutter/residual_crops_all_classes.npz',
             all_masks_gt=total_crops, labels_gt=total_labels)
#%%
# the data, shuffled and split between train and test sets
with np.load('use_cases/edge-cutter/residual_crops_all_classes.npz') as ld:
    all_masks_gt = ld['all_masks_gt']
    labels_gt = ld['labels_gt']
#%%
num_sampl = 30000
predictions = loaded_model.predict(
    all_masks_gt[:num_sampl, :, :, None], batch_size=32, verbose=1)
#cm.movie(np.squeeze(all_masks_gt[np.where(predictions[:num_sampl,1]<0.1)[0]])).play(gain=3., magnification = 5, fr = 10)
#%%
from skimage.util.montage import montage2d
pl.imshow(montage2d(all_masks_gt[np.where((labels_gt[:num_sampl] == 2) & (
    predictions[:num_sampl, 0] <= 0.5))[0]].squeeze()))
#%%
fname_new = '/mnt/ceph/neuro/labeling/neurofinder.03.00.test/images/final_map/Yr_d1_498_d2_467_d3_1_order_C_frames_2250_.mmap'
gSig = [8, 8]
gt_file = os.path.join(os.path.split(fname_new)[0], os.path.split(
    fname_new)[1][:-4] + 'match_masks.npz')
base_name = fname_new.split('/')[5]
maxT = 8100
with np.load(gt_file, encoding='latin1') as ld:
    print(ld.keys())
    locals().update(ld)
    A_gt = scipy.sparse.coo_matrix(A_gt[()])
    dims = (d1, d2)
    C_gt = C_gt[:, :maxT]
    YrA_gt = YrA_gt[:, :maxT]
    f_gt = f_gt[:, :maxT]
    try:
        fname_new = fname_new[()].decode('unicode_escape')
    except:
        fname_new = fname_new[()]

    A_gt_thr = cm.source_extraction.cnmf.spatial.threshold_components(A_gt.tocsc()[:, :].toarray(), dims, medw=None, thr_method='max', maxthr=0.2, nrgthr=0.99, extract_cc=True,
                                                                      se=None, ss=None, dview=None)

Yr, dims, T = cm.load_memmap(fname_new)
T = np.minimum(T, maxT)
Yr = Yr[:, :T]
d1, d2 = dims
images = np.reshape(Yr.T, [T] + list(dims), order='F')
# TODO: needinfo
Y = np.reshape(Yr, dims + (T,), order='F')
m_orig = np.array(images)
patch_size = 50
half_crop = np.minimum(gSig[0] * 4 + 1, patch_size) // 2
idx_included = np.arange(A_gt.shape[-1])
# idx_included = idx_components_cnn[np.arange(50)] # neurons that are displayed
all_pos_crops = []
all_neg_crops = []
all_dubious_crops = []
base_name = fname_new.split('/')[5]
dims = (d1, d2)
idx_included = np.array([x for x in idx_included if x is not None])
base_name = base_name + '_' + \
    str(idx_included[0]) + '_' + str(idx_included[-1])
print(base_name)
idx_excluded = np.setdiff1d(np.arange(A_gt.shape[-1]), idx_included)
m_res = m_orig - cm.movie(np.reshape(A_gt.tocsc()[:, idx_excluded].dot(
    C_gt[idx_excluded]) + b_gt.dot(f_gt), dims + (-1,), order='F').transpose([2, 0, 1]))
mean_proj = np.mean(m_res, 0)
std_mov = mean_proj.std()
min_mov = np.median(mean_proj)
#%%
# will use either the overfeat like (full FOV, train_net_cifar_edge_cutter_FOV.py) network or the single patch one (train_net_cifar_edge_cutter.py)
full_fov = True
normalize_median_std = True
count_start = 30
dims = np.array(dims)
border_size = 6
#bin_ = 10
cms_total = [np.array(scipy.ndimage.center_of_mass(np.reshape(a.toarray(
), dims, order='F'))).astype(np.int) for a in A_gt.tocsc()[:, idx_included].T]
cms_total = np.maximum(cms_total, half_crop)
cms_total = np.array([np.minimum(cms, dims - half_crop)
                      for cms in cms_total]).astype(np.int)

all_heatmaps = []
for count in range(count_start, T):
    if count % 1000 == 0:
        print(count)
    print(count)
    img_avg = m_res[count - count_start:count].mean(0)
    image_no_neurons = np.array((img_avg).astype(np.float32))

    image_orig = np.array(image_no_neurons.copy())
    if normalize_median_std:
        image_orig = (image_orig - np.median(image_orig)) / np.std(image_orig)
    else:
        image_orig = (image_orig - min_mov) / 8 / std_mov

    if full_fov:
        image_orig = locate(cv2.copyMakeBorder(image_orig,border_size,border_size,border_size,border_size, cv2.BORDER_REFLECT_101)[:, :, None], rectangle=False, prob = .5)[
            1].squeeze()
        all_heatmaps.append(image_orig)
    else:

        avg_crops = []
        for cm__ in cms_total:
            cm_ = cm__[::]
            ain = img_avg[cm_[0] - half_crop:cm_[0] + half_crop,
                          cm_[1] - half_crop:cm_[1] + half_crop]
            avg_crops.append(cv2.resize(
                ain / np.linalg.norm(ain), (patch_size, patch_size)))

        predictions = loaded_model.predict(
            np.array(avg_crops)[:, :, :, np.newaxis], batch_size=32, verbose=1)

        for pred, cm_ in zip(predictions, cms_total):
            if pred[0] > 0.95:
                image_orig = cv2.rectangle(image_orig, (cm_[
                                           1] - half_crop, cm_[0] - half_crop), (cm_[1] + half_crop, cm_[0] + half_crop), 10)

    #
    #    cms_1 = cms_2q
    #    cms_1 = np.maximum(cms_1,half_crop)
    #    cms_1 = np.array([np.minimum(cms,dims-half_crop) for cms in cms_1]).astype(np.int)
    #    crop_imgs = [img_avg[com[0]-half_crop:com[0]+half_crop, com[1]-half_crop:com[1]+half_crop] for com in cms_1]
    #    final_crops = np.array([cv2.resize(im/np.linalg.norm(im),(patch_size ,patch_size)) for im in crop_imgs])

#    cv2.imshow('frame', cv2.resize(image_orig*1., (dims[1] * 2, dims[0] * 2)))
#    cv2.waitKey(10)
