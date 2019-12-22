#!/usr/bin/env python
"""
Demo pipeline for processing voltage imaging data. The processing pipeline
includes motion correction and spike detection with given ROIs.
"""

import os
import cv2
import glob
import logging
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import sys

try:
    cv2.setNumThreads(0)
except:
    pass

try:
    if __IPYTHON__:
        # this is used for debugging purposes only. allows to reload classes
        # when changed
        get_ipython().magic('load_ext autoreload')
        get_ipython().magic('autoreload 2')
except NameError:
    pass

import caiman as cm
from caiman.motion_correction import MotionCorrect
from caiman.utils.utils import download_demo
from caiman.source_extraction.volpy.Volparams import volparams
from caiman.source_extraction.volpy.volpy import VOLPY

use_maskrcnn = True
if use_maskrcnn:
    from caiman.paths import caiman_datadir
    model_dir = caiman_datadir()+'/model'
    sys.path.append(model_dir) 
    import mrcnn.model as modellib
    from mrcnn import visualize
    from mrcnn import neurons
    import tensorflow as tf
    
# %%
# Set up the logger (optional); change this if you like.
# You can log to a file using the filename parameter, or make the output more
# or less verbose by setting level to logging.DEBUG, logging.INFO,
# logging.WARNING, or logging.ERROR
logging.basicConfig(format=
                    "%(relativeCreated)12d [%(filename)s:%(funcName)20s():%(lineno)s]" \
                    "[%(process)d] %(message)s",
                    level=logging.INFO)

# %%
def main():
    pass  # For compatibility between running under Spyder and the CLI
   
    # %% Please download the .npz file manually to your file path. Example datasets.
    url = 'https://www.dropbox.com/s/av3223dred7h5tb/demo_voltage_imaging.npz?dl=0'
    n_processes = 8
    
    # %% Load demo movie and ROIs
    file_path = '/home/nel/Code/Voltage_imaging/exampledata/403106_3min/demo_voltage_imaging.npz'
    m = np.load(file_path)
    mv = cm.movie(m.f.arr_0)
    ROIs = m.f.arr_1
    
    # %%
    fnames = '/home/nel/Code/Voltage_imaging/exampledata/403106_3min/demo_voltage_imaging.hdf5'
    mv.save(fnames)

    # %% Setup some parameters for data and motion correction    
    # dataset parameters
    fr = 400                                        # sample rate of the movie
    ROIs = ROIs
    index = list(range(ROIs.shape[0]))              # index of neurons
    weights = None                                  # reuse spatial weights by 
                                                    # opts.change_params(params_dict={'weights':vpy.estimates['weights']})
    # motion correction parameters
    motion_correct = True                           # flag for motion correction
    pw_rigid = False                                # flag for pw-rigid motion correction
    gSig_filt = (3, 3)                              # size of filter, in general gSig (see below),
                                                    # change this one if algorithm does not work
    max_shifts = (5, 5)                             # maximum allowed rigid shift
    strides = (48, 48)                              # start a new patch for pw-rigid motion correction every x pixels
    overlaps = (24, 24)                             # overlap between pathes (size of patch strides+overlaps)
    max_deviation_rigid = 3                         # maximum deviation allowed for patch with respect to rigid shifts
    border_nan = 'copy'

    mc_dict = {
        'fnames': fnames,
        'fr': fr,
        'index': index,
        'ROIs':ROIs,
        'weights':weights,              
        'pw_rigid': pw_rigid,
        'max_shifts': max_shifts,
        'gSig_filt': gSig_filt,
        'strides': strides,
        'overlaps': overlaps,
        'max_deviation_rigid': max_deviation_rigid,
        'border_nan': border_nan
    }

    opts = volparams(params_dict=mc_dict)

    # %% play the movie (optional)
    # playing the movie using opencv. It requires loading the movie in memory.
    # To close the video press q    
    display_images = False

    if display_images:
        m_orig = cm.load(fnames)
        ds_ratio = 0.2
        moviehandle = m_orig.resize(1, 1, ds_ratio)
        moviehandle.play(q_max=99.5, fr=60, magnification=2)
   
    # %% start a cluster for parallel processing
    c, dview, n_processes = cm.cluster.setup_cluster(
        backend='local', n_processes=n_processes, single_thread=False)

    # %%% MOTION CORRECTION
    # Create a motion correction object with the specified parameters
    mc = MotionCorrect(fnames, dview=dview, **opts.get_group('motion'))
    # Run piecewise rigid motion correction 
    mc.motion_correct(save_movie=True)    
    dview.terminate()

    # %% motion correction compared with original movie
    display_images = False

    if display_images:
        m_orig = cm.load(fnames)
        m_rig = cm.load(mc.mmap_file)
        ds_ratio = 0.2
        moviehandle = cm.concatenate([m_orig.resize(1, 1, ds_ratio) - mc.min_mov * mc.nonneg_movie,
                                      m_rig.resize(1, 1, ds_ratio)], axis=2)
        moviehandle.play(fr=60, q_max=99.5, magnification=2)  # press q to exit

    # % movie subtracted from the mean
        m_orig2 = (m_orig - np.mean(m_orig, axis=0))
        m_rig2 = (m_rig - np.mean(m_rig, axis=0))
        moviehandle1 = cm.concatenate([m_orig2.resize(1, 1, ds_ratio),
                                       m_rig2.resize(1, 1, ds_ratio)], axis=2)
        moviehandle1.play(fr=60, q_max=99.5, magnification=2) 

   # %% MEMORY mAPPING
    c, dview, n_processes = cm.cluster.setup_cluster(
        backend='local', n_processes=n_processes, single_thread=False)
    
    border_to_0 = 0 if mc.border_nan is 'copy' else mc.border_to_0
    fname_new = cm.save_memmap_join(mc.mmap_file, base_name='memmap_',
                               add_to_mov=border_to_0, dview=dview, n_chunks=10)
    
    dview.terminate() 
    
   # %% change fnames to the new motion corrected one
    opts.change_params(params_dict={'fnames':fname_new})
   
    #%% SEGMENTATION
    # Create mean and correlation image
    use_maskrcnn = True
    if use_maskrcnn:        
        m = cm.load(mc.mmap_file[0])
        if m.shape[0] > 20000:
            m = m[:20000,:,:]
        
        m = cm.movie(np.array(m), fr=400)
        img = m.mean(axis=0)
        img = (img-np.mean(img))/np.std(img)   
        m1 = m.compute_BL(secsWindow=1)    
        m = m - m1       
        Cn = m.local_correlations(swap_dim=False, eight_neighbours=True)
        img_corr = (Cn-np.mean(Cn))/np.std(Cn)
        summary_image = np.stack([img,img,img_corr],axis=2).astype(np.float32)
        del m
        del m1
        
        # Mask R-CNN       
        config = neurons.NeuronsConfig()
        class InferenceConfig(config.__class__):
            # Run detection on one image at a time
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MIN_CONFIDENCE = 0.7
            IMAGE_RESIZE_MODE = "pad64"
            IMAGE_MAX_DIM=512
            RPN_NMS_THRESHOLD = 0.7
            POST_NMS_ROIS_INFERENCE = 1000

        config = InferenceConfig()
        config.display()
        
        DEVICE = "/cpu:0"  # /cpu:0 or /gpu:0
        with tf.device(DEVICE):
            model = modellib.MaskRCNN(mode="inference", model_dir=model_dir,
                                      config=config)
        weights_path = model_dir + '/mrcnn/mask_rcnn_neurons_0040.h5'
        model.load_weights(weights_path, by_name=True)
        
        
        results = model.detect([summary_image], verbose=1)
        r = results[0]
        ROIs_mrcnn = r['masks'].transpose([2,0,1])
        
    # %% visualize the result
        display_result = True
        
        if display_result:  
            _, ax = plt.subplots(1,1, figsize=(16,16))
            visualize.display_instances(summary_image, r['rois'], r['masks'], r['class_ids'], 
                                    ['BG', 'neurons'], r['scores'], ax=ax,
                                    title="Predictions")
    
    # %% set to rois
        opts.change_params(params_dict={'ROIs':ROIs_mrcnn,
                                        'method':'SpikePursuit',
                                        'Ridge_bg_coef':0.5,  
                                        'index':list(range(ROIs_mrcnn.shape[0]))})

    # %% SpikePursuit
    c, dview, n_processes = cm.cluster.setup_cluster(
            backend='local', n_processes=n_processes, single_thread=False)
    vpy = VOLPY(n_processes=n_processes, dview=dview, params=opts)
    vpy.fit(n_processes=n_processes, dview=dview)

    # %% some visualization
    vpy.estimates['cellN']
    n = 0  
    plt.figure()
    plt.plot(vpy.estimates['trace'][n])
    plt.plot(vpy.estimates['spikeTimes'][n], np.max(vpy.estimates['trace'][n]) * 1 * np.ones(vpy.estimates['spikeTimes'][n].shape),
             color='g', marker='o', fillstyle='none', linestyle='none')
    plt.title('signal and spike times')
    plt.show()

    plt.figure()
    plt.imshow(-vpy.estimates['spatialFilter'][n])
    plt.colorbar()
    plt.title('spatial filter')
    plt.show()

    # %% STOP CLUSTER and clean up log files
    cm.stop_server(dview=dview)
    log_files = glob.glob('*_LOG_*')
    for log_file in log_files:
        os.remove(log_file)
    

# %%
# This is to mask the differences between running this demo in Spyder
# versus from the CLI
if __name__ == "__main__":
    main()
