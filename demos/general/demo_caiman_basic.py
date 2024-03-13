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

import caiman
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

    opts = params.CNMFParams(params_from_file=cfg.configfile)

    if cfg.input is not None: # CLI arg can override all other settings for fnames, although other data-centric commands still must match source data
        opts.change_params({'data': {'fnames': cfg.input}})

    if not opts.data['fnames']: # Set neither by CLI arg nor through JSON, so use default data
        # If no input is specified, use sample data, downloading if necessary
        fnames = [os.path.join(caiman_datadir(), 'example_movies', 'demoMovie.tif')] # file(s) to be analyzed
        opts.change_params({'data': {'fnames': fnames}})

    # start a cluster for parallel processing
    c, dview, n_processes = caiman.cluster.setup_cluster(backend=cfg.cluster_backend, n_processes=cfg.cluster_nproc)


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
    Cn[np.isnan(Cn)] = 0 # Convert NaNs to zero
    if not cfg.no_play:
        cnm.estimates.plot_contours(img=Cn)

    # load memory mapped file
    Yr, dims, T = caiman.load_memmap(cnm.mmap_file)
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
    cnm2.save(cnm2.mmap_file[:-4] + 'hdf5') # FIXME use the path library to do this the right way

    # play movie with results (original, reconstructed, amplified residual)
    if not cfg.no_play:
        cnm2.estimates.play_movie(images, magnification=4);

    # Stop the cluster and clean up log files
    caiman.stop_server(dview=dview)

    if not cfg.keep_logs:
        log_files = glob.glob('Yr*_LOG_*')
        for log_file in log_files:
            os.remove(log_file)

def handle_args():
    parser = argparse.ArgumentParser(description="Demonstrate basic Caiman functionality")
    parser.add_argument("--configfile", default=os.path.join(caiman_datadir(), 'demos', 'general', 'params_demo_caiman_basic_with_patches.json'), help="JSON Configfile for Caiman parameters")
    parser.add_argument("--keep_logs",  action="store_true", help="Keep temporary logfiles")
    parser.add_argument("--no_play",    action="store_true", help="Do not display results")
    parser.add_argument("--cluster_backend", default="multiprocessing", help="Specify multiprocessing, ipyparallel, or single to pick an engine")
    parser.add_argument("--cluster_nproc", type=int, default=None, help="Override automatic selection of number of workers to use")
    parser.add_argument("--input", action="append", help="File(s) to work on, provide multiple times for more files")
    parser.add_argument("--logfile",    help="If specified, log to the named file")
    return parser.parse_args()

########
main()
