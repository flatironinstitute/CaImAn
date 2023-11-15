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
import numpy as np
import os

try:
    cv2.setNumThreads(0)
except:
    pass

import caiman as cm
from caiman.motion_correction import MotionCorrect
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
        fnames = ['Sue_2x_3000_40_-46.tif']  # filename to be processed
        if fnames[0] in ['Sue_2x_3000_40_-46.tif', 'demoMovie.tif']:
            fnames = [download_demo(fnames[0])]
    else:
        fnames = cfg.input
    # If you prefer to hardcode filenames, you could do something like this:
    # fnames = ["/path/to/myfile1.avi", "/path/to/myfile2.avi"]


    # First setup some parameters for data and motion correction

    # dataset dependent parameters
    fr = 30             # imaging rate in frames per second
    decay_time = 0.4    # length of a typical transient in seconds
    dxy = (2., 2.)      # spatial resolution in x and y in (um per pixel)
    # note the lower than usual spatial resolution here
    max_shift_um = (12., 12.)       # maximum shift in um
    patch_motion_um = (100., 100.)  # patch size for non-rigid correction in um

    # motion correction parameters
    pw_rigid = True       # flag to select rigid vs pw_rigid motion correction
    # maximum allowed rigid shift in pixels
    max_shifts = [int(a/b) for a, b in zip(max_shift_um, dxy)]
    # start a new patch for pw-rigid motion correction every x pixels
    strides = tuple([int(a/b) for a, b in zip(patch_motion_um, dxy)])
    # overlap between patches (size of patch in pixels: strides+overlaps)
    overlaps = (24, 24)
    # maximum deviation allowed for patch with respect to rigid shifts
    max_deviation_rigid = 3

    params_dict = {'fnames': fnames,
                   'fr': fr,
                   'decay_time': decay_time,
                   'dxy': dxy,
                   'pw_rigid': pw_rigid,
                   'max_shifts': max_shifts,
                   'strides': strides,
                   'overlaps': overlaps,
                   'max_deviation_rigid': max_deviation_rigid,
                   'border_nan': 'copy'}

    opts = params.CNMFParams(params_dict=params_dict)

    m_orig = cm.load_movie_chain(fnames)

    # play the movie (optional)
    # playing the movie using opencv. It requires loading the movie in memory.
    # To close the video press q

    if not cfg.no_play:
        ds_ratio = 0.2
        moviehandle = m_orig.resize(1, 1, ds_ratio)
        moviehandle.play(q_max=99.5, fr=60, magnification=2)

    # start a cluster for parallel processing
    c, dview, n_processes = cm.cluster.setup_cluster(backend=cfg.cluster_backend)

    # Motion Correction
    # first we create a motion correction object with the specified parameters
    mc = MotionCorrect(fnames, dview=dview, **opts.get_group('motion'))
    # note that the file is not loaded in memory

    # Run (piecewise-rigid motion) correction using NoRMCorre
    mc.motion_correct(save_movie=True)

    # compare with original movie
    if not cfg.no_play:
        m_orig = cm.load_movie_chain(fnames)
        m_els = cm.load(mc.mmap_file)
        ds_ratio = 0.2
        moviehandle = cm.concatenate([m_orig.resize(1, 1, ds_ratio) - mc.min_mov*mc.nonneg_movie,
                                      m_els.resize(1, 1, ds_ratio)], axis=2)
        moviehandle.play(fr=60, q_max=99.5, magnification=2)  # press q to exit

    # MEMORY MAPPING
    border_to_0 = 0 if mc.border_nan == 'copy' else mc.border_to_0
    # you can include the boundaries of the FOV if you used the 'copy' option
    # during motion correction, although be careful about the components near
    # the boundaries

    # memory map the file in order 'C'
    fname_new = cm.save_memmap(mc.mmap_file, base_name='memmap_', order='C',
                               border_to_0=border_to_0)  # exclude borders

    # now load the file
    Yr, dims, T = cm.load_memmap(fname_new)
    images = np.reshape(Yr.T, [T] + list(dims), order='F')
    # load frames in python format (T x X x Y)

    # restart cluster to clean up memory
    cm.stop_server(dview=dview)
    c, dview, n_processes = cm.cluster.setup_cluster(
        backend=cfg.cluster_backend, n_processes=None, single_thread=False)

    #  parameters for source extraction and deconvolution
    p = 1                    # order of the autoregressive system
    gnb = 2                  # number of global background components
    merge_thr = 0.85         # merging threshold, max correlation allowed
    rf = 15
    # half-size of the patches in pixels. e.g., if rf=25, patches are 50x50
    stride_cnmf = 6          # amount of overlap between the patches in pixels
    K = 4                    # number of components per patch
    gSig = [4, 4]            # expected half size of neurons in pixels
    # initialization method (if analyzing dendritic data using 'sparse_nmf')
    method_init = 'greedy_roi'
    ssub = 2                     # spatial subsampling during initialization
    tsub = 2                     # temporal subsampling during initialization

    # parameters for component evaluation
    opts_dict = {'fnames': fnames,
                 'p': p,
                 'fr': fr,
                 'nb': gnb,
                 'rf': rf,
                 'K': K,
                 'gSig': gSig,
                 'stride': stride_cnmf,
                 'method_init': method_init,
                 'rolling_sum': True,
                 'merge_thr': merge_thr,
                 'n_processes': n_processes,
                 'only_init': True,
                 'ssub': ssub,
                 'tsub': tsub}

    opts.change_params(params_dict=opts_dict);

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
    cnm.save(fname_new[:-5]+'_init.hdf5')

    # RE-RUN seeded CNMF on accepted patches to refine and perform deconvolution
    cnm2 = cnm.refit(images, dview=dview)

    # Component Evaluation
    #   Components are evaluated in three ways:
    #   a) the shape of each component must be correlated with the data
    #   b) a minimum peak SNR is required over the length of a transient
    #   c) each shape passes a CNN based classifier

    min_SNR = 2  # signal to noise ratio for accepting a component
    rval_thr = 0.85  # space correlation threshold for accepting a component
    use_cnn = True
    cnn_thr = 0.99  # threshold for CNN based classifier
    cnn_lowest = 0.1 # neurons with cnn probability lower than this value are rejected

    cnm2.params.set('quality', {'decay_time': decay_time,
                               'min_SNR': min_SNR,
                               'rval_thr': rval_thr,
                               'use_cnn': use_cnn,
                               'min_cnn_thr': cnn_thr,
                               'cnn_lowest': cnn_lowest})
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
    cm.stop_server(dview=dview)

    if not cfg.keep_logs:
        log_files = glob.glob('*_LOG_*')
        for log_file in log_files:
            os.remove(log_file)

def handle_args():
    parser = argparse.ArgumentParser(description="Demonstrate 2P Pipeline using batch algorithm")
    parser.add_argument("--keep_logs",  action="store_true", help="Keep temporary logfiles")
    parser.add_argument("--no_play",    action="store_true", help="Do not display results")
    parser.add_argument("--cluster_backend", default="multiprocessing", help="Specify multiprocessing, ipyparallel, or single to pick an engine")
    parser.add_argument("--input", action="append", help="File(s) to work on, provide multiple times for more files")
    parser.add_argument("--logfile",    help="If specified, log to the named file")
    return parser.parse_args()

########
main()

