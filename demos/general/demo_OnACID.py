#!/usr/bin/env python

"""
Basic demo for the CaImAn Online algorithm (OnACID) using CNMF initialization.
It demonstrates the construction of the params and online_cnmf objects and
the fit function that is used to run the algorithm.
For a more complete demo check the script demo_OnACID_mesoscope.py
"""

import argparse
#import code
import logging
import numpy as np
import os

try:
    cv2.setNumThreads(0)
except:
    pass

import caiman as cm
from caiman.source_extraction import cnmf as cnmf
from caiman.paths import caiman_datadir


def main():
    cfg = handle_args()

    if cfg.logfile:
        logging.basicConfig(format=
            "%(relativeCreated)12d [%(filename)s:%(funcName)20s():%(lineno)s][%(process)d] %(message)s",
            level=logging.WARNING,
            filename=cfg.logfile)
        # You can make the output more or less verbose by setting level to logging.DEBUG, logging.INFO, logging.WARNING, or logging.ERROR
    else:
        logging.basicConfig(format=
            "%(relativeCreated)12d [%(filename)s:%(funcName)20s():%(lineno)s][%(process)d] %(message)s",
            level=logging.WARNING)

    if cfg.input is None:
        fnames = [os.path.join(caiman_datadir(), 'example_movies', 'demoMovie.tif')]
    else:
        fnames = cfg.input

    if cfg.configfile:
        opts = cnmf.params.CNMFParams(params_from_file=cfg.configfile)
        if opts.data['fnames'] is None:
            opts.set("data", {"fnames": fnames})
    else:
        # set up some parameters
        fr = 10  # frame rate (Hz)
        decay_time = .75  # approximate length of transient event in seconds
        gSig = [6, 6]  # expected half size of neurons
        p = 1  # order of AR indicator dynamics
        min_SNR = 1  # minimum SNR for accepting candidate components
        thresh_CNN_noisy = 0.65  # CNN threshold for candidate components
        gnb = 2  # number of background components
        init_method = 'cnmf'  # initialization method
    
        # set up CNMF initialization parameters
        init_batch = 400  # number of frames for initialization
        patch_size = 32  # size of patch
        stride = 3  # amount of overlap between patches
        K = 4  # max number of components in each patch
    
        params_dict = {'fr': fr,
                       'fnames': fnames,
                       'decay_time': decay_time,
                       'gSig': gSig,
                       'p': p,
                       'min_SNR': min_SNR,
                       'nb': gnb,
                       'init_batch': init_batch,
                       'init_method': init_method,
                       'rf': patch_size//2,
                       'stride': stride,
                       'sniper_mode': True,
                       'thresh_CNN_noisy': thresh_CNN_noisy,
                       'K': K}
    
        opts = cnmf.params.CNMFParams(params_dict=params_dict)

    # If you want to break into an interactive console session, move and uncomment this wherever you want in the code
    # (and uncomment the code import at the top)
    #code.interact(local=dict(globals(), **locals()) )

    # fit with online object
    cnm = cnmf.online_cnmf.OnACID(params=opts)
    cnm.fit_online()

    # plot contours
    logging.info(f"Number of components: {cnm.estimates.A.shape[-1]}")
    Cn = cm.load(fnames[0], subindices=slice(0,500)).local_correlations(swap_dim=False)
    cnm.estimates.plot_contours(img=Cn)

    # pass through the CNN classifier with a low threshold (keeps clearer neuron shapes and excludes processes)
    use_CNN = True
    if use_CNN:
        # threshold for CNN classifier
        opts.set('quality', {'min_cnn_thr': 0.05})
        cnm.estimates.evaluate_components_CNN(opts)
        cnm.estimates.plot_contours(img=Cn, idx=cnm.estimates.idx_components)
    
    # plot results
    cnm.estimates.view_components(img=Cn, idx=cnm.estimates.idx_components)

def handle_args():
    parser = argparse.ArgumentParser(description="Demonstrate basic Caiman Online functionality with CNMF initialization")
    parser.add_argument("--configfile", help="JSON Configfile for Caiman parameters")
    parser.add_argument("--input", action="append", help="File(s) to work on, provide multiple times for more files")
    parser.add_argument("--logfile",    help="If specified, log to the named file")
    return parser.parse_args()

########
main()
