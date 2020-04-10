#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 16:45:00 2020

@author: caichangjia
"""
#%% 
import os
import matplotlib.pyplot as plt
import numpy as np
from caiman.external.cell_magic_wand import cell_magic_wand_single_point
import tensorflow as tf
from caiman.source_extraction.volpy.mrcnn import visualize, neurons
import caiman.source_extraction.volpy.mrcnn.model as modellib
from caiman.paths import caiman_datadir
import caiman as cm
from matplotlib.widgets import Slider


def quick_annotation(image, min_radius, max_radius, roughtness=2):
    
    def tellme(s):
        print(s)
        plt.title(s, fontsize=16)
        plt.draw()
        
    keep_select=True
    ROIs = []
    while keep_select:
        # Plot image
        plt.clf()
        plt.imshow(image, cmap='gray', vmax=np.percentile(image, 98))            
        if len(ROIs) == 0:
            pass
        elif len(ROIs) == 1:
            plt.imshow(ROIs[0], alpha=0.3, cmap='Oranges')
        else:
            plt.imshow(np.array(ROIs).sum(axis=0), alpha=0.3, cmap='Oranges')
        
        # Plot point and ROI
        tellme('Click center of neuron')
        center = plt.ginput(1)[0]
        plt.plot(center[0], center[1], 'r+')
        ROI = cell_magic_wand_single_point(image, (center[1], center[0]), 
                                           min_radius=min_radius, max_radius=max_radius, 
                                           roughness=2, zoom_factor=1)[0]
        plt.imshow(ROI, alpha=0.3, cmap='Reds')
    
        # Select or not
        tellme('Select? Key click for yes, mouse click for no')
        select = plt.waitforbuttonpress()
        if select:
            ROIs.append(ROI)
            tellme('You have selected a neuron. \n Keep selecting? Key click for yes, mouse click for no')
        else:
            tellme('You did not select a neuron \n Keep selecting? Key click for yes, mouse click for no')
        keep_select = plt.waitforbuttonpress()
        
    plt.close()        
    ROIs = np.array(ROIs)
    
    return ROIs

def mrcnn_inference(img, weights_path, display_result=True):
    config = neurons.NeuronsConfig()
    class InferenceConfig(config.__class__):
        # Run detection on one image at a time
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        DETECTION_MIN_CONFIDENCE = 0.7
        IMAGE_RESIZE_MODE = "pad64"
        IMAGE_MAX_DIM = 512
        RPN_NMS_THRESHOLD = 0.7
        POST_NMS_ROIS_INFERENCE = 1000
    config = InferenceConfig()
    config.display()
    model_dir = os.path.join(caiman_datadir(), 'model')
    DEVICE = "/cpu:0"  # /cpu:0 or /gpu:0
    with tf.device(DEVICE):
        model = modellib.MaskRCNN(mode="inference", model_dir=model_dir,
                                  config=config)
    model.load_weights(weights_path, by_name=True)
    results = model.detect([img], verbose=1)
    r = results[0]
    ROIs = r['masks'].transpose([2, 0, 1])

    display_result = True
    if display_result:
        _, ax = plt.subplots(1,1, figsize=(16,16))
        visualize.display_instances(img, r['rois'], r['masks'], r['class_ids'], 
                                ['BG', 'neurons'], r['scores'], ax=ax,
                                title="Predictions")
        
    return ROIs


def reconstructed_video(estimates, fnames, idx, scope):
    mv = cm.load(fnames, fr=400)[scope[0]:scope[1]]
    dims = (mv.shape[1], mv.shape[2])
    mv_bl = mv.computeDFF(secsWindow=0.1)[0]
    mv = (mv-mv.min())/(mv.max()-mv.min())
    mv_bl = -mv_bl
    mv_bl[mv_bl<np.percentile(mv_bl,3)] = np.percentile(mv_bl,3)
    mv_bl[mv_bl>np.percentile(mv_bl,98)] = np.percentile(mv_bl,98)
    mv_bl = (mv_bl - mv_bl.min())/(mv_bl.max()-mv_bl.min())

    estimates['weights'][estimates['weights']<0] = 0
    
    A = estimates['weights'][idx].transpose([1,2,0]).reshape((-1,len(idx)))
    C = estimates['t_rec'][idx,scope[0]:scope[1]]
    mv_rec = np.dot(A, C).reshape((dims[0],dims[1],scope[1]-scope[0])).transpose((2,0,1))    
    mv_rec = cm.movie(mv_rec,fr=400)
    mv_rec = (mv_rec - mv_rec.min())/(mv_rec.max()-mv_rec.min())
    mv_all = cm.concatenate((mv,mv_bl,mv_rec),axis=2)
    
    return mv_all

def view_components(estimates, img, idx):
    n = len(idx) 
    fig = plt.figure(figsize=(10, 10))

    axcomp = plt.axes([0.05, 0.05, 0.9, 0.03])
    ax1 = plt.axes([0.05, 0.55, 0.4, 0.4])
    ax3 = plt.axes([0.55, 0.55, 0.4, 0.4])
    ax2 = plt.axes([0.05, 0.1, 0.9, 0.4])    
    s_comp = Slider(axcomp, 'Component', 0, n, valinit=0)
    vmax = np.percentile(img, 98)
    
    def arrow_key_image_control(event):

        if event.key == 'left':
            new_val = np.round(s_comp.val - 1)
            if new_val < 0:
                new_val = 0
            s_comp.set_val(new_val)

        elif event.key == 'right':
            new_val = np.round(s_comp.val + 1)
            if new_val > n :
                new_val = n  
            s_comp.set_val(new_val)
        else:
            pass
        
    def update(val):
        i = np.int(np.round(s_comp.val))
        print(('Component:' + str(i)))

        if i < n:
            
            ax1.cla()
            imgtmp = estimates['weights'][idx][i]
            ax1.imshow(imgtmp, interpolation='None', cmap=plt.cm.gray, vmax=np.max(imgtmp)*0.5, vmin=0)
            ax1.set_title('Spatial component ' + str(i + 1))
            ax1.axis('off')
            
            ax2.cla()
            ax2.plot(estimates['t'][idx][i], alpha=0.8)
            ax2.plot(estimates['t_sub'][idx][i])            
            ax2.plot(estimates['t_rec'][idx][i], alpha = 0.4, color='red')
            ax2.plot(estimates['spikes'][idx][i],
                     1.05 * np.max(estimates['t'][idx][i]) * np.ones(estimates['spikes'][idx][i].shape),
                     color='r', marker='.', fillstyle='none', linestyle='none')
            ax2.set_title('Signal and spike times' + str(i + 1))
            ax2.legend(labels=['t', 't_sub', 't_rec', 'spikes'])
            ax2.text(0.1, 0.1,'snr:'+str(round(estimates['snr'][idx][i],2)), horizontalalignment='center', verticalalignment='center', transform = ax2.transAxes)
            ax2.text(0.1, 0.07,'num_spikes:'+str(len(estimates['spikes'][idx][i])), horizontalalignment='center', verticalalignment='center', transform = ax2.transAxes)            
            ax2.text(0.1, 0.04,'locality_test:'+str(estimates['locality'][idx][i]), horizontalalignment='center', verticalalignment='center', transform = ax2.transAxes)            
            
            ax3.cla()
            ax3.imshow(img, interpolation='None', cmap=plt.cm.gray, vmax=vmax)
            imgtmp2 = imgtmp.copy()
            imgtmp2[imgtmp2 == 0] = np.nan
            ax3.imshow(imgtmp2, interpolation='None',
                       alpha=0.5, cmap=plt.cm.hot)
            ax3.axis('off')
            print('snr:{0}'.format(estimates['snr'][idx][i]))
            
    s_comp.on_changed(update)
    s_comp.set_val(0)
    fig.canvas.mpl_connect('key_release_event', arrow_key_image_control)
    plt.show()








