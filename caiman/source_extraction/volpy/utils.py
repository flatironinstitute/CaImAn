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
from skimage.morphology import disk, erosion
import torch
import caiman as cm
from caiman.external.cell_magic_wand import cell_magic_wand_single_point
from caiman.source_extraction.volpy.mrcnn.config import Config
from caiman.source_extraction.volpy.mrcnn.model import (
    get_model_instance_segmentation,
    load_mrcnn_weights,
    mrcnn_inference,
)
from caiman.source_extraction.volpy.mrcnn.utils import data_transform, prepare_mrcnn_image

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
    try:
        if __IPYTHON__:
            get_ipython().run_line_magic('matplotlib', 'auto')
    except NameError:
        pass

    def tellme(s):
        print(s)
        plt.title(s, fontsize=16)
        plt.draw()
        
    keep_select=True
    ROIs = []
    while keep_select:
        # Plot img
        plt.clf()
        plt.imshow(img, cmap='gray', vmax=np.percentile(img, 99))            
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
    
    try:
        if __IPYTHON__:
            get_ipython().run_line_magic('matplotlib', 'inline')
    except NameError:
        pass

    return ROIs

def mrcnn_inference_pytorch(img, size_range, weights_path, display_result=True,
                            confidence_threshold=None, mask_threshold=None,
                            box_nms_threshold=None, erosion_radius=None):
    """ 
    Mask R-CNN inference in VolPy using PyTorch.
    Args:
        img (np.ndarray):
            2-D or 3-D summary image for detection. If 2D, it's converted to 3-channel.

        size_range (list):
            Range of neuron size [min, max] for selection.

        weights_path (str):
            Path for the PyTorch Mask R-CNN weights file (.pt).

        display_result (bool):
            If True, the function will plot the result of the inference.
            
        confidence_threshold (float, optional):
            Minimum detection score. Defaults to ``Config.INFERENCE_THRESHOLD``.

        mask_threshold (float, optional):
            Probability threshold used to binarize masks.

        box_nms_threshold (float, optional):
            IoU threshold for torchvision's final box non-maximum suppression.

        erosion_radius (int, optional):
            Radius of the disk used to erode accepted masks. Set to zero to disable.

    Returns:
        ROIs: 3-D np.ndarray:
            A 3-D array of ROIs (# of components, height, width).
    """
    config = Config()
    confidence_threshold = (config.INFERENCE_THRESHOLD if confidence_threshold is None
                            else confidence_threshold)
    mask_threshold = config.MASK_THRESHOLD if mask_threshold is None else mask_threshold
    box_nms_threshold = (config.BOX_NMS_THRESHOLD if box_nms_threshold is None
                         else box_nms_threshold)
    erosion_radius = (config.MASK_EROSION_RADIUS if erosion_radius is None
                      else erosion_radius)

    if len(size_range) != 2 or size_range[0] < 0 or size_range[0] >= size_range[1]:
        raise ValueError("size_range must contain increasing non-negative radii")
    if erosion_radius < 0:
        raise ValueError("erosion_radius must be non-negative")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    #Load Model and weights
    model = get_model_instance_segmentation(num_classes=config.NUM_CLASSES, pretrained=False)
    load_mrcnn_weights(model, weights_path, device)
    model.roi_heads.nms_thresh = box_nms_threshold
    model.to(device)
    model.eval()

    # Pre-process Image
    img_tv_tensor = prepare_mrcnn_image(img)

    # Perform Inference
    _, predicted_boxes, binarized_masks, predicted_scores = mrcnn_inference(
        model=model,
        img=img_tv_tensor, 
        thresh=confidence_threshold,
        mask_threshold=mask_threshold,
        eval_transform=data_transform(train=False),
        device=device,
        return_scores=True,
    )

    # Post-process and Filter by Size
    if binarized_masks.size == 0:
        ROIs = np.empty((0, *img.shape[:2]), dtype=bool)
        display_boxes = np.empty((0, 4), dtype=np.float32)
        display_scores = np.empty((0,), dtype=np.float32)
    else:
        mask_areas = binarized_masks.sum(axis=(1, 2))
        selection = np.logical_and(mask_areas > size_range[0] ** 2,
                                   mask_areas < size_range[1] ** 2)
        ROIs = binarized_masks[selection].astype(bool)
        if erosion_radius and len(ROIs):
            footprint = disk(erosion_radius)
            ROIs = np.stack([erosion(mask, footprint) for mask in ROIs])
        display_boxes = predicted_boxes.detach().cpu().numpy()[selection]
        display_scores = predicted_scores.detach().cpu().numpy()[selection]

    print(f"Inference complete. Found {ROIs.shape[0]} neurons.")

    if display_result:
        from caiman.source_extraction.volpy.mrcnn.visualize import display_instances
        _, ax = plt.subplots(1, 1, figsize=(16, 16))
        boxes_yx = display_boxes[:, [1, 0, 3, 2]]
        masks_hwn = ROIs.astype(np.uint8).transpose(1, 2, 0)
        class_ids = np.ones(len(ROIs), dtype=int)
        display_instances(
            np.asarray(img) if np.asarray(img).ndim == 3 else np.repeat(np.asarray(img)[..., None], 3, axis=-1),
            boxes_yx,
            masks_hwn,
            class_ids,
            ['BG', 'neurons'],
            display_scores,
            ax=ax,
            title=f"PyTorch Predictions ({len(ROIs)} ROIs found)",
        )
        plt.show()

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

def view_components(estimates, img, idx, frame_times=None, gt_times=None):
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
    if frame_times is not None:
        pass
    else:
        frame_times = np.array(range(len(estimates['t'][0])))
    
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
        i = int(np.round(s_comp.val))
        print(f'Component:{i}')

        if i < n:
            
            ax1.cla()
            imgtmp = estimates['weights'][idx][i]
            ax1.imshow(imgtmp, interpolation='None', cmap=plt.cm.gray, vmax=np.max(imgtmp)*0.5, vmin=0)
            ax1.set_title(f'Spatial component {i+1}')
            ax1.axis('off')
            
            ax2.cla()
            ax2.plot(frame_times, estimates['t'][idx][i], alpha=0.8)
            ax2.plot(frame_times, estimates['t_sub'][idx][i])            
            ax2.plot(frame_times, estimates['t_rec'][idx][i], alpha = 0.4, color='red')
            ax2.plot(frame_times[estimates['spikes'][idx[i]]],
                     1.05 * np.max(estimates['t'][idx][i]) * np.ones(estimates['spikes'][idx[i]].shape),
                     color='r', marker='.', fillstyle='none', linestyle='none')
            if gt_times is not None:
                ax2.plot(gt_times,
                     1.15 * np.max(estimates['t'][idx][i]) * np.ones(gt_times.shape),
                     color='blue', marker='.', fillstyle='none', linestyle='none')
                ax2.legend(labels=['t', 't_sub', 't_rec', 'spikes', 'gt_spikes'])
            else:
                ax2.legend(labels=['t', 't_sub', 't_rec', 'spikes'])
            ax2.set_title(f'Signal and spike times {i+1}')
            ax2.text(0.1, 0.1, f'snr:{round(estimates["snr"][idx][i],2)}', horizontalalignment='center', verticalalignment='center', transform = ax2.transAxes)
            ax2.text(0.1, 0.07, f'num_spikes: {len(estimates["spikes"][idx[i]])}', horizontalalignment='center', verticalalignment='center', transform = ax2.transAxes)            
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
    
