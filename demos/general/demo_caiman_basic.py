#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Stripped demo for running the CNMF source extraction algorithm with CaImAn and
evaluation the components. The analysis can be run either in the whole FOV
or in patches. For a complete pipeline (including motion correction) 
check demo_pipeline.py
Data courtesy of W. Yang, D. Peterka and R. Yuste (Columbia University)

This demo is designed to be run under spyder or jupyter; its plotting functions
are tailored for that environment.

@authors: @agiovann and @epnev

"""

from builtins import range

import cv2
from copy import deepcopy
import glob
import logging
import numpy as np
import os

try:
    cv2.setNumThreads(0)
except:
    pass

try:
    if __IPYTHON__:
        get_ipython().magic('load_ext autoreload')
        get_ipython().magic('autoreload 2')
except NameError:
    pass


import caiman as cm
from caiman.paths import caiman_datadir
from caiman.source_extraction.cnmf import cnmf as cnmf
from caiman.source_extraction.cnmf import params as params

#%%
# Set up the logger; change this if you like.
# You can log to a file using the filename parameter, or make the output more or less
# verbose by setting level to logging.DEBUG, logging.INFO, logging.WARNING, or logging.ERROR

logging.basicConfig(format=
                          "%(relativeCreated)12d [%(filename)s:%(funcName)20s():%(lineno)s] [%(process)d] %(message)s",
                    # filename="/tmp/caiman.log",

#%%
def main():
    pass # For compatibility between running under Spyder and the CLI

#%% start a cluster

    c, dview, n_processes =\
        cm.cluster.setup_cluster(backend='local', n_processes=None,
                                 single_thread=False)

#%% set up some parameters
    fnames = [os.path.join(caiman_datadir(), 'example_movies', 'demoMovie.tif')]
                            # file to be analyzed
    is_patches = True       # flag for processing in patches or not
    fr = 10                 # approximate frame rate of data
    decay_time = 5.0        # length of transient

    if is_patches:          # PROCESS IN PATCHES AND THEN COMBINE
        rf = 10             # half size of each patch
        stride = 4          # overlap between patches
        K = 4               # number of components in each patch
    else:                   # PROCESS THE WHOLE FOV AT ONCE
        rf = None           # setting these parameters to None
        stride = None       # will run CNMF on the whole FOV
        K = 30              # number of neurons expected (in the whole FOV)

    gSig = [6, 6]           # expected half size of neurons
    merge_thresh = 0.80     # merging threshold, max correlation allowed
    p = 2                   # order of the autoregressive system
    gnb = 2                 # global background order

    params_dict = {'fnames': fnames,
                   'fr': fr,
                   'decay_time': decay_time,
                   'rf': rf,
                   'stride': stride,
                   'K': K,
                   'gSig': gSig,
                   'merge_thr': merge_thresh,
                   'p': p,
                   'nb': gnb}

    opts = params.CNMFParams(params_dict=params_dict)
#%% Now RUN CNMF
    cnm = cnmf.CNMF(n_processes, params=opts, dview=dview)
    cnm = cnm.fit_file()

#%% plot contour plots of components
    Cn = cm.load(fnames[0], subindices=range(1000)).local_correlations(swap_dim=False)
    cnm.estimates.plot_contours(img=Cn)

#%% load memory mapped file
    Yr, dims, T = cm.load_memmap(cnm.mmap_file)
    images = np.reshape(Yr.T, [T] + list(dims), order='F')

#%% refit
    cnm2 = cnm.refit(images, dview=dview)

#%% COMPONENT EVALUATION
    # the components are evaluated in three ways:
    #   a) the shape of each component must be correlated with the data
    #   b) a minimum peak SNR is required over the length of a transient
    #   c) each shape passes a CNN based classifier (this will pick up only neurons
    #           and filter out active processes)

    min_SNR = 2.5       # peak SNR for accepted components (if above this, acept)
    rval_thr = 0.90     # space correlation threshold (if above this, accept)
    use_cnn = True      # use the CNN classifier
    min_cnn_thr = 0.95  # if cnn classifier predicts below this value, reject
    
    cnm2.params.set('quality', {'min_SNR': min_SNR,
                                'rval_thr': rval_thr,
                                'use_cnn': use_cnn,
                                'min_cnn_thr': min_cnn_thr})
                                
    cnm2.estimates.evaluate_components(images, cnm2.params, dview=dview)
#%% visualize selected and rejected components
    cnm2.estimates.plot_contours(img=Cn, idx=cnm2.estimates.idx_components)

#%% visualize selected components
    cnm2.estimates.view_components(images, idx=cnm2.estimates.idx_components, img=Cn)

#%% play movie with results
    cnm2.estimates.play_movie(images, magnification=4)

#%% STOP CLUSTER and clean up log files
    cm.stop_server(dview=dview)

    log_files = glob.glob('Yr*_LOG_*')
    for log_file in log_files:
        os.remove(log_file)

#%%
# This is to mask the differences between running this demo in Spyder
# versus from the CLI
if __name__ == "__main__":
    main()
