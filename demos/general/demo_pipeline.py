#!/usr/bin/env python

"""
Complete demo pipeline for processing two photon calcium imaging data using the
Caiman batch algorithm. The processing pipeline included motion correction,
source extraction and deconvolution. The demo shows how to construct the
params, MotionCorrect and cnmf objects and call the relevant functions. You
can also run a large part of the pipeline with a single method (cnmf.fit_file)
See inside for details.

Demo is also available as a jupyter notebook (see demo_pipeline.ipynb)
Dataset couresy of Sue Ann Koay and David Tank (Princeton University)

This demo pertains to two photon data. For a complete analysis pipeline for
one photon microendoscopic data see demo_pipeline_cnmfE.py
"""

import argparse
import cv2
import glob
import logging
import matplotlib
import numpy as np
import os

try:
    cv2.setNumThreads(0)
except:
    pass

import caiman
from caiman.motion_correction import MotionCorrect
from caiman.source_extraction.cnmf import cnmf as cnmf
from caiman.source_extraction.cnmf import params as params
from caiman.summary_images import local_correlations_movie_offline
from caiman.utils.utils import download_demo


def main():
    cfg = handle_args()

    if cfg.logfile:
        logging.basicConfig(format=
            "[%(filename)s:%(funcName)20s():%(lineno)s] %(message)s",
            level=logging.INFO,
            filename=cfg.logfile)
        # You can make the output more or less verbose by setting level to logging.DEBUG, logging.INFO, logging.WARNING, or logging.ERROR
    else:
        logging.basicConfig(format=
            "[%(filename)s:%(funcName)20s():%(lineno)s] %(message)s",
            level=logging.INFO)

    # First set up some parameters for data and motion correction
    opts = params.CNMFParams(params_from_file=cfg.configfile)

    if cfg.input is not None:
        opts.change_params({"data": {"fnames": cfg.input}})
    if not opts.data['fnames']: # Set neither by CLI arg nor through JSON, so use default data
        fnames = [download_demo('Sue_2x_3000_40_-46.tif')]
        opts.change_params({"data": {"fnames": fnames}})

    m_orig = caiman.load_movie_chain(opts.data['fnames'])

    # play the movie (optional)
    # playing the movie using opencv. It requires loading the movie in memory.
    # To close the video press q

    if not cfg.no_play:
        ds_ratio = 0.2
        moviehandle = m_orig.resize(1, 1, ds_ratio)
        moviehandle.play(q_max=99.5, fr=60, magnification=2)

    # start a cluster for parallel processing
    c, dview, n_processes = caiman.cluster.setup_cluster(backend=cfg.cluster_backend, n_processes=cfg.cluster_nproc)

    # Motion Correction
    # first we create a motion correction object with the specified parameters
    mc = MotionCorrect(opts.data['fnames'], dview=dview, **opts.get_group('motion'))
    # note that the file is not loaded in memory

    # Run (piecewise-rigid motion) correction using NoRMCorre
    mc.motion_correct(save_movie=True)

    # compare with original movie
    if not cfg.no_play:
        m_orig = caiman.load_movie_chain(opts.data['fnames'])
        m_els = caiman.load(mc.mmap_file)
        ds_ratio = 0.2
        moviehandle = caiman.concatenate([m_orig.resize(1, 1, ds_ratio) - mc.min_mov*mc.nonneg_movie,
                                      m_els.resize(1, 1, ds_ratio)], axis=2)
        moviehandle.play(fr=60, q_max=99.5, magnification=2)  # press q to exit

    # MEMORY MAPPING
    border_to_0 = 0 if mc.border_nan == 'copy' else mc.border_to_0
    # you can include the boundaries of the FOV if you used the 'copy' option
    # during motion correction, although be careful about the components near
    # the boundaries

    # memory map the file in order 'C'
    fname_new = caiman.save_memmap(mc.mmap_file, base_name='memmap_', order='C',
                               border_to_0=border_to_0)  # exclude borders

    # now load the file
    Yr, dims, T = caiman.load_memmap(fname_new)
    images = np.reshape(Yr.T, [T] + list(dims), order='F')
    # load frames in python format (T x X x Y)

    # restart cluster to clean up memory
    caiman.stop_server(dview=dview)
    c, dview, n_processes = caiman.cluster.setup_cluster(backend=cfg.cluster_backend, n_processes=cfg.cluster_nproc)

    # RUN CNMF ON PATCHES
    # First extract spatial and temporal components on patches and combine them
    # for this step deconvolution is turned off (p=0). If you want to have
    # deconvolution within each patch change params.patch['p_patch'] to a
    # nonzero value

    #opts.change_params({'p': 0})
    cnm = cnmf.CNMF(n_processes, params=opts, dview=dview)
    cnm = cnm.fit(images)

    # plot contours of found components
    Cns = local_correlations_movie_offline(mc.mmap_file[0],
                                           remove_baseline=True,
                                           window=1000, stride=1000,
                                           winSize_baseline=100, quantil_min_baseline=10,
                                           dview=dview)
    Cn = Cns.max(axis=0)
    Cn[np.isnan(Cn)] = 0
    if not cfg.no_play:
        cnm.estimates.plot_contours(img=Cn)

    # save results
    cnm.estimates.Cn = Cn
    cnm.save(fname_new[:-5] + '_init.hdf5') # FIXME

    # RE-RUN seeded CNMF on accepted patches to refine and perform deconvolution
    cnm2 = cnm.refit(images, dview=dview)

    # Component Evaluation
    #   Components are evaluated in three ways:
    #   a) the shape of each component must be correlated with the data
    #   b) a minimum peak SNR is required over the length of a transient
    #   c) each shape passes a CNN based classifier

    cnm2.estimates.evaluate_components(images, cnm2.params, dview=dview)

    if not cfg.no_play:
        # Plot Components
        cnm2.estimates.plot_contours(img=Cn, idx=cnm2.estimates.idx_components)

        # View Traces (accepted and rejected)
        cnm2.estimates.view_components(images, img=Cn,
                                      idx=cnm2.estimates.idx_components)
        cnm2.estimates.view_components(images, img=Cn,
                                      idx=cnm2.estimates.idx_components_bad)

    # update object with selected components (optional)
    cnm2.estimates.select_components(use_object=True)

    # Extract DF/F values
    cnm2.estimates.detrend_df_f(quantileMin=8, frames_window=250)

    # Show final traces
    if not cfg.no_play:
        cnm2.estimates.view_components(img=Cn)

    cnm2.estimates.Cn = Cn
    cnm2.save(cnm2.mmap_file[:-4] + 'hdf5')

    # reconstruct denoised movie (press q to exit)
    if not cfg.no_play:
        cnm2.estimates.play_movie(images, q_max=99.9, gain_res=2,
                                  magnification=2,
                                  bpx=border_to_0,
                                  include_bck=False)  # background not shown

    # Stop the cluster and clean up log files
    caiman.stop_server(dview=dview)
    matplotlib.pyplot.show(block=True)

    if not cfg.keep_logs:
        log_files = glob.glob('*_LOG_*')
        for log_file in log_files:
            os.remove(log_file)

def handle_args():
    parser = argparse.ArgumentParser(description="Demonstrate 2P Pipeline using batch algorithm")
    parser.add_argument("--configfile", default=os.path.join(caiman.paths.caiman_datadir(), 'demos', 'general', 'params_demo_pipeline.json'), help="JSON Configfile for Caiman parameters")
    parser.add_argument("--keep_logs",  action="store_true", help="Keep temporary logfiles")
    parser.add_argument("--no_play",    action="store_true", help="Do not display results")
    parser.add_argument("--cluster_backend", default="multiprocessing", help="Specify multiprocessing, ipyparallel, or single to pick an engine")
    parser.add_argument("--cluster_nproc", type=int, default=None, help="Override automatic selection of number of workers to use")
    parser.add_argument("--input", action="append", help="File(s) to work on, provide multiple times for more files")
    parser.add_argument("--logfile",    help="If specified, log to the named file")
    return parser.parse_args()

########
main()

