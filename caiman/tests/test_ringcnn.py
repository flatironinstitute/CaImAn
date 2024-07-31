import os
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ["KERAS_BACKEND"] = "torch"

import bokeh.plotting as bpl
import cv2
import holoviews as hv
import logging
import matplotlib.pyplot as plt
from multiprocessing import Process, Queue, set_start_method
import numpy as np
from scipy.sparse import csr_matrix
from sys import platform
import tensorflow as tf
import torch 
from time import time, sleep

try:
    if __IPYTHON__:
        get_ipython().run_line_magic('load_ext', 'autoreload')
        get_ipython().run_line_magic('autoreload', '2')
except NameError:
    pass

logging.basicConfig(format=
                          "%(relativeCreated)12d [%(filename)s:%(funcName)10s():%(lineno)s] [%(process)d] %(message)s",
                    # filename="/tmp/caiman.log",
                    level=logging.WARNING)

import caiman as cm
from caiman.source_extraction import cnmf
from caiman.source_extraction.cnmf.online_cnmf import demix1p
from caiman.utils.nn_models import (fit_NL_model, create_LN_model, quantile_loss, rate_scheduler)
from caiman.utils.utils import download_demo


def get_iterator(device=0, fr=None):
    """
    device: device number (int) or filename (string) for reading from camera or file respectively
    fr: frame rate
    """
    if isinstance(device, int):  # capture from camera
        def capture_iter(device=device, fr=fr):
            cap = cv2.VideoCapture(device)
            if fr is not None:  # set frame rate
                cap.set(cv2.CAP_PROP_FPS, fr)
            while True:
                yield cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2GRAY)
        iterator = capture_iter(device, fr)
    else:  # read frame by frame from file
        iterator = cm.base.movies.load_iter(device, var_name_hdf5='Y')
    return iterator 

if __name__ == "__main__":

    init_batch = 100  # number of frames to use for initialization
    T = 6000          # total number of frames 
    fr = 10           # frame rate (Hz)

    iterator = get_iterator(download_demo('blood_vessel_10Hz.mat'))

    m = cm.movie(np.array([next(iterator) for t in range(init_batch)], dtype='float32'))

    fname_init = m.save('init.mmap', order='C')

    reuse_model = False                                       # set to True to reuse an existing ring model
    path_to_model = None                                      # specify a pre-trained model here if needed 
    gSig = (7, 7)                                             # expected half size of neurons
    gnb = 2    

    params_dict = {'fnames': fname_init,                      # filename(s) to be processed
               'var_name_hdf5': 'Y',                      # name of variable inside mat file where the data is stored
               'fr': fr,                                  # frame rate (Hz)
               'decay_time': 0.5,                         # approximate length of transient event in seconds
               'gSig': gSig,                              # gaussian width of a 2D gaussian kernel, which approximates a neuron
               'p': 1,                                    # order of AR indicator dynamics
               'ring_CNN': True,                          # SET TO TRUE TO USE RING CNN 
               'min_SNR': 2.65,                           # minimum SNR for accepting new components
               'SNR_lowest': 0.75,                        # reject components with SNR below this value
               'use_cnn': False,                          # do not use CNN based test for components
               'use_ecc': True,                           # test eccentricity
               'max_ecc': 2.625,                          # reject components with eccentricity above this value
               'rval_thr': 0.70,                          # correlation threshold for new component inclusion
               'rval_lowest': 0.25,                       # reject components with corr below that value
               'ds_factor': 1,                            # spatial downsampling factor (increases speed but may lose some fine structure)
               'nb': gnb,                                 # number of background components (rank)
               'motion_correct': True,                    # Flag for motion correction
               'init_batch': init_batch,                  # number of frames for initialization (presumably from the first file)
               'init_method': 'bare',                     # initialization method
               'normalize': False,                        # Whether to normalize each frame prior to online processing
               'expected_comps': 700,                     # maximum number of expected components used for memory pre-allocation (exaggerate here)
               'sniper_mode': False,                      # flag using a CNN to detect new neurons (o/w space correlation is used)
               'dist_shape_update' : True,                # flag for updating shapes in a distributed way
               'min_num_trial': 5,                        # number of candidate components per frame
               'epochs': 1,                               # number of total passes over the data
               'stop_detection': False,                   # Run a last epoch without detecting new neurons  
               'K': 50,                                   # initial number of components
               'lr': 6e-4,                                # (initial) learning rate
               'lr_scheduler': [0.9, 6000, 10000],        # learning rate scheduler
               'pct': 0.01,                               # quantile of the quantile loss function
               'path_to_model': path_to_model,            # where the ring CNN model is saved/loaded
               'reuse_model': reuse_model,                # flag for re-using a ring CNN model
              }

    opts = cnmf.params.CNMFParams(params_dict=params_dict)

    cnm3 = cnmf.online_cnmf.OnACID(params=opts)

    if cnm3.params.get('ring_CNN', 'loss_fn') == 'pct':
        loss_fn = quantile_loss(cnm3.params.get('ring_CNN', 'pct'))
    else:
        loss_fn = cnm3.params.get('ring_CNN', 'loss_fn')
    if cnm3.params.get('ring_CNN', 'lr_scheduler') is None:
        sch = None
    else:
        sch = rate_scheduler(*cnm3.params.get('ring_CNN', 'lr_scheduler'))
    model_LN = create_LN_model(m, shape=m.shape[1:] + (1,),
                           n_channels=cnm3.params.get('ring_CNN', 'n_channels'),
                           lr=cnm3.params.get('ring_CNN', 'lr'),
                           gSig=cnm3.params.get('init', 'gSig')[0],
                           loss=loss_fn, width=cnm3.params.get('ring_CNN', 'width'),
                           use_add=cnm3.params.get('ring_CNN', 'use_add'),
                           use_bias=cnm3.params.get('ring_CNN', 'use_bias'))
    if cnm3.params.get('ring_CNN', 'reuse_model'):
        model_LN.load_weights(cnm3.params.get('ring_CNN', 'path_to_model'))
    else:
        model_LN, history, path_to_model = fit_NL_model(
            model_LN, m, epochs=cnm3.params.get('ring_CNN', 'max_epochs'),
            patience=cnm3.params.get('ring_CNN', 'patience'), schedule=sch)
        cnm3.params.set('ring_CNN', {'path_to_model': path_to_model})



    
