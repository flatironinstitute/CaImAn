#!/usr/bin/env python
"""
Created on Mon Mar 23 16:45:00 2020
This file create functions used for demo_pipeline_voltage_imaging.py
@author: caichangjia
"""
#%% 
from IPython import get_ipython
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
import os
import tensorflow as tf

import caiman as cm
from caiman.external.cell_magic_wand import cell_magic_wand_single_point
from caiman.paths import caiman_datadir

def correlation_image(fnames, fr):
    """ Compute correlation image with Gaussian Blur. The method is relative slow
    in speed, but it is suitable for high noise movie.
    Args:
        fnames: 3-D array
            motion corrected movie in F-order memory mapping format
        
        fr: int
            frame rate
            
    Return: 
        Cn: 2-D array
            correlation image
    """
    m = cm.load(fnames, fr=fr)
    ma = m.computeDFF(secsWindow=1)[0]
    Cn = ma.copy().gaussian_blur_2D().local_correlations(swap_dim=False, 
                eight_neighbours=True, frames_per_chunk=1000000000)
    return Cn

def quick_annotation(img, min_radius, max_radius, roughness=2):
    """ Quick annotation method in VolPy using cell magic wand plugin
    Args:
        img: 2-D array
            img as the background for selection
            
        min_radius: float
            minimum radius of the selection
            
        max_radius: float
            maximum raidus of the selection
            
        roughness: int
            roughness of the selection surface
            
    Return:
        ROIs: 3-D array
            region of interests 
            (# of components * # of pixels in x dim * # of pixels in y dim)
    """
    get_ipython().run_line_magic('matplotlib', 'auto')
    def tellme(s):
        print(s)
        plt.title(s, fontsize=16)
        plt.draw()
        
    keep_select=True
    ROIs = []
    while keep_select:
        # Plot img
        plt.clf()
        plt.imshow(img, cmap='gray', vmax=np.percentile(img, 98))            
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
        ROI = cell_magic_wand_single_point(img, (center[1], center[0]), 
                                           min_radius=min_radius, max_radius=max_radius, 
                                           roughness=roughness, zoom_factor=1)[0]
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
    get_ipython().run_line_magic('matplotlib', 'inline')
    return ROIs

def mrcnn_inference(img, weights_path, display_result=True):
    """ Mask R-CNN inference in VolPy
    Args: 
        img: 2-D array
            summary images for detection
            
        weights_path: str
            path for Mask R-CNN weight
            
        display_result: boolean
            if True, the function will plot the result of inference
        
    Return:
        ROIs: 3-D array
            region of interests 
            (# of components * # of pixels in x dim * # of pixels in y dim)
    """
    from caiman.source_extraction.volpy.mrcnn import visualize, neurons
    import caiman.source_extraction.volpy.mrcnn.model as modellib
    config = neurons.NeuronsConfig()
    class InferenceConfig(config.__class__):
        # Run detection on one img at a time
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

    if display_result:
        _, ax = plt.subplots(1,1, figsize=(16,16))
        visualize.display_instances(img, r['rois'], r['masks'], r['class_ids'], 
                                ['BG', 'neurons'], r['scores'], ax=ax,
                                title="Predictions")        
    return ROIs

def reconstructed_movie(estimates, fnames, idx, scope, flip_signal):
    """ Create reconstructed movie in VolPy. The movie has three panels: 
    motion corrected movie on the left panel, movie removed from the baseline
    on the mid panel and reconstructed movie on the right panel.
    Args: 
        estimates: dict
            estimates dictionary contain results of VolPy
            
        fnames: list
            motion corrected movie in F-order memory mapping format
            
        idx: list
            index of selected neurons
            
        scope: list
            scope of number of frames in reconstructed movie
            
        flip_signal: boolean
            if True the signal will be flipped (for voltron) 
    
    Return:
        mv_all: 3-D array
            motion corrected movie, movie removed from baseline, reconstructed movie
            concatenated into one matrix
    """
    # motion corrected movie and movie removed from baseline
    mv = cm.load(fnames, fr=400)[scope[0]:scope[1]]
    dims = (mv.shape[1], mv.shape[2])
    mv_bl = mv.computeDFF(secsWindow=0.1)[0]
    mv = (mv-mv.min())/(mv.max()-mv.min())
    if flip_signal:
        mv_bl = -mv_bl
    mv_bl[mv_bl<np.percentile(mv_bl,3)] = np.percentile(mv_bl,3)
    mv_bl[mv_bl>np.percentile(mv_bl,98)] = np.percentile(mv_bl,98)
    mv_bl = (mv_bl - mv_bl.min())/(mv_bl.max()-mv_bl.min())

    # reconstructed movie
    estimates['weights'][estimates['weights']<0] = 0    
    A = estimates['weights'][idx].transpose([1,2,0]).reshape((-1,len(idx)))
    C = estimates['t_rec'][idx,scope[0]:scope[1]]
    mv_rec = np.dot(A, C).reshape((dims[0],dims[1],scope[1]-scope[0])).transpose((2,0,1))    
    mv_rec = cm.movie(mv_rec,fr=400)
    mv_rec = (mv_rec - mv_rec.min())/(mv_rec.max()-mv_rec.min())
    mv_all = cm.concatenate((mv,mv_bl,mv_rec),axis=2)    
    return mv_all

def view_components(estimates, img, idx):
    """ View spatial and temporal components interactively
    Args:
        estimates: dict
            estimates dictionary contain results of VolPy
            
        img: 2-D array
            summary images for detection
            
        idx: list
            index of selected neurons
    """
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
        
    def update(val):
        i = np.int(np.round(s_comp.val))
        print(f'Component:{i}')

        if i < n:
            
            ax1.cla()
            imgtmp = estimates['weights'][idx][i]
            ax1.imshow(imgtmp, interpolation='None', cmap=plt.cm.gray, vmax=np.max(imgtmp)*0.5, vmin=0)
            ax1.set_title(f'Spatial component {i+1}')
            ax1.axis('off')
            
            ax2.cla()
            ax2.plot(estimates['t'][idx][i], alpha=0.8)
            ax2.plot(estimates['t_sub'][idx][i])            
            ax2.plot(estimates['t_rec'][idx][i], alpha = 0.4, color='red')
            ax2.plot(estimates['spikes'][idx][i],
                     1.05 * np.max(estimates['t'][idx][i]) * np.ones(estimates['spikes'][idx][i].shape),
                     color='r', marker='.', fillstyle='none', linestyle='none')
            ax2.set_title(f'Signal and spike times {i+1}')
            ax2.legend(labels=['t', 't_sub', 't_rec', 'spikes'])
            ax2.text(0.1, 0.1, f'snr:{round(estimates["snr"][idx][i],2)}', horizontalalignment='center', verticalalignment='center', transform = ax2.transAxes)
            ax2.text(0.1, 0.07, f'num_spikes: {len(estimates["spikes"][idx][i])}', horizontalalignment='center', verticalalignment='center', transform = ax2.transAxes)            
            ax2.text(0.1, 0.04, f'locality_test: {estimates["locality"][idx][i]}', horizontalalignment='center', verticalalignment='center', transform = ax2.transAxes)            
            
            ax3.cla()
            ax3.imshow(img, interpolation='None', cmap=plt.cm.gray, vmax=vmax)
            imgtmp2 = imgtmp.copy()
            imgtmp2[imgtmp2 == 0] = np.nan
            ax3.imshow(imgtmp2, interpolation='None',
                       alpha=0.5, cmap=plt.cm.hot)
            ax3.axis('off')
            
    s_comp.on_changed(update)
    s_comp.set_val(0)
    fig.canvas.mpl_connect('key_release_event', arrow_key_image_control)
    plt.show()