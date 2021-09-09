#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 14:33:54 2021
This file is used for testing fiola
@author: @caichangjia
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from threading import Thread
import tensorflow as tf

import caiman as cm
from caiman.fiola.fiolaparams import fiolaparams
from caiman.fiola.fiola import FIOLA
from caiman.utils.utils import download_demo

import logging
logging.basicConfig(format=
                    "%(relativeCreated)12d [%(filename)s:%(funcName)20s():%(lineno)s]"\
                    "[%(process)d] %(message)s",
                    level=logging.WARNING)

def demo_test_fiola():
    if len(tf.config.list_physical_devices('GPU')) < 1:
        logging.warning('GPU is not detected')
    fnames = download_demo('demo_voltage_imaging.hdf5', 'volpy')  # file path to movie file (will download if not present)
    path_ROIs = download_demo('demo_voltage_imaging_ROIs.hdf5', 'volpy')  # file path to ROIs file (will download if not present)
    mov = cm.load(fnames)
    mask = cm.load(path_ROIs)
    mov = mov[:2000]
    
    # setting params
    # dataset dependent parameters
    fnames = fnames                 # name of the movie, we don't put a name here as movie is already loaded above
    fr = 400                        # sample rate of the movie
    ROIs = mask                     # a 3D matrix contains all region of interests
    
    mode = 'voltage'                # 'voltage' or 'calcium 'fluorescence indicator
    init_method = 'binary_masks'    # initialization method 'caiman', 'weighted_masks' or 'binary_masks'. Needs to provide masks or using gui to draw masks if choosing 'masks'
    num_frames_init =  1500        # number of frames used for initialization
    num_frames_total =  2000       # estimated total number of frames for processing, this is used for generating matrix to store data
    offline_batch_size = 100        # number of frames for one batch to perform offline motion correction
    batch_size = 20                 # number of frames processing at the same time using gpu 
    flip = True                     # whether to flip signal to find spikes   
    ms = [10, 10]                   # maximum shift in x and y axis respectively. Will not perform motion correction if None.
    update_bg = True                # update background components for spatial footprints
    filt_window = 15                # window size for removing the subthreshold activities 
    minimal_thresh = 3              # minimal of the threshold 
    template_window = 2             # half window size of the template; will not perform template matching if window size equals 0
    
    options = {
        'fnames': fnames,
        'fr': fr,
        'ROIs': ROIs,
        'mode': mode,
        'init_method':init_method,
        'num_frames_init': num_frames_init, 
        'num_frames_total':num_frames_total,
        'offline_batch_size': offline_batch_size,
        'batch_size':batch_size,
        'flip': flip,
        'ms': ms,
        'update_bg': update_bg,
        'filt_window': filt_window,
        'minimal_thresh': minimal_thresh,
        'template_window':template_window}
    
    params = fiolaparams(params_dict=options)
    fio = FIOLA(params=params)
    scope = [fio.params.data['num_frames_init'], fio.params.data['num_frames_total']]
    fio.fit(mov[:scope[0]])
    fio.pipeline.load_frame_thread = Thread(target=fio.pipeline.load_frame, 
                                            daemon=True, 
                                            args=(mov[scope[0]:scope[1], :, :],))
    fio.pipeline.load_frame_thread.start()
    fio.fit_online()
    fio.compute_estimates()
