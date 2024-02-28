#!/usr/bin/env python

"""
Basic demo for running the CNMF source extraction algorithm with
Caiman and evaluation the components. The analysis can be run either in the
whole FOV or in patches. For a complete pipeline (including motion correction)
check demo_pipeline.py

Data courtesy of W. Yang, D. Peterka and R. Yuste (Columbia University)

This demo is designed to be run in an IDE or from a CLI; its plotting functions
are tailored for that environment.

"""

import argparse
import cv2
import glob
import logging
import numpy as np
import os

try:
    cv2.setNumThreads(0)
except:
    pass

import caiman as cm
from caiman.paths import caiman_datadir
from caiman.source_extraction.cnmf import cnmf as cnmf
from caiman.source_extraction.cnmf import params as params
from caiman.summary_images import local_correlations_movie_offline
from caiman.utils.utils import download_demo


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
        # If no input is specified, use sample data, downloading if necessary
        fnames = [os.path.join(caiman_datadir(), 'example_movies', 'demoMovie.tif')] # file(s) to be analyzed
    else:
        fnames = cfg.input
    # If you prefer to hardcode filenames, you could do something like this:
    # fnames = ["/path/to/myfile1.avi", "/path/to/myfile2.avi"]

    if cfg.no_patches:
        is_patches = False # flag for processing in patches or not
    else:
        is_patches = True

    # set up some parameters
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

    min_SNR = 2      # peak SNR for accepted components (if above this, accept)
    rval_thr = 0.85     # space correlation threshold (if above this, accept)
    use_cnn = True      # use the CNN classifier
    min_cnn_thr = 0.99  # if cnn classifier predicts below this value, reject
    cnn_lowest = 0.1 # neurons with cnn probability lower than this value are rejected

    params_dict = {
        'data': {
            'fnames': fnames,
            'fr': fr,
            'decay_time': decay_time,
            },
        'init': {
            'gSig': gSig,
            'K': K,
            'nb': gnb
            },
        'patch': {
            'rf': rf,
            'stride': stride,
            },
        'merging': {
            'merge_thr': merge_thresh,
            },
        'preprocess': {
            'p': p,
            },
        'temporal': {
            'p': p,
            },
        'quality': {
                'min_SNR': min_SNR,
                'rval_thr': rval_thr,
                'use_cnn': use_cnn,
                'min_cnn_thr': min_cnn_thr,
                'cnn_lowest': cnn_lowest
            }
        }

    opts = params.CNMFParams(params_dict=params_dict)

    # start a cluster for parallel processing
    c, dview, n_processes = cm.cluster.setup_cluster(backend=cfg.cluster_backend, n_processes=cfg.cluster_nproc)


    # Run CaImAn Batch (CNMF)
    cnm = cnmf.CNMF(n_processes, params=opts, dview=dview)
    cnm = cnm.fit_file()

    # plot contour plots of components
    Cns = local_correlations_movie_offline(fnames[0],
                                           remove_baseline=True,
                                           swap_dim=False, window=1000, stride=1000,
                                           winSize_baseline=100, quantil_min_baseline=10,
                                           dview=dview)
    Cn = Cns.max(axis=0)
    Cn[np.isnan(Cn)] = 0
    if not cfg.no_play:
        cnm.estimates.plot_contours(img=Cn)

    # load memory mapped file
    Yr, dims, T = cm.load_memmap(cnm.mmap_file)
    images = np.reshape(Yr.T, [T] + list(dims), order='F')

    # refit
    cnm2 = cnm.refit(images, dview=dview)

    # Component Evaluation
    #   Components are evaluated in three ways:
    #   a) the shape of each component must be correlated with the data
    #   b) a minimum peak SNR is required over the length of a transient
    #   c) each shape passes a CNN based classifier (this will pick up only neurons
    #           and filter out active processes)


    cnm2.estimates.evaluate_components(images, opts, dview=dview)

    if not cfg.no_play:
        # visualize selected and rejected components
        cnm2.estimates.plot_contours(img=Cn, idx=cnm2.estimates.idx_components)
        # visualize selected components
        cnm2.estimates.view_components(images, idx=cnm2.estimates.idx_components, img=Cn)
   
    # only select high quality components (destructive)
    # cnm2.estimates.select_components(use_object=True)
    # cnm2.estimates.plot_contours(img=Cn)

    # save results
    cnm2.estimates.Cn = Cn
    cnm2.save(cnm2.mmap_file[:-4]+'hdf5')

    # play movie with results (original, reconstructed, amplified residual)
    if not cfg.no_play:
        cnm2.estimates.play_movie(images, magnification=4);

    # Stop the cluster and clean up log files
    cm.stop_server(dview=dview)

    if not cfg.keep_logs:
        log_files = glob.glob('Yr*_LOG_*')
        for log_file in log_files:
            os.remove(log_file)

def handle_args():
    parser = argparse.ArgumentParser(description="Demonstrate basic Caiman functionality")
    parser.add_argument("--keep_logs",  action="store_true", help="Keep temporary logfiles")
    parser.add_argument("--no_patches", action="store_true", help="Do not use patches")
    parser.add_argument("--no_play",    action="store_true", help="Do not display results")
    parser.add_argument("--cluster_backend", default="multiprocessing", help="Specify multiprocessing, ipyparallel, or single to pick an engine")
    parser.add_argument("--cluster_nproc", type=int, default=None, help="Override automatic selection of number of workers to use")
    parser.add_argument("--input", action="append", help="File(s) to work on, provide multiple times for more files")
    parser.add_argument("--logfile",    help="If specified, log to the named file")
    return parser.parse_args()

########
main()
