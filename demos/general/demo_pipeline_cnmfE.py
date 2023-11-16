#!/usr/bin/env python

"""
Complete pipeline for motion correction, source extraction, and deconvolution
of one photon microendoscopic calcium imaging data using the CaImAn package.
The demo demonstrates how to use the params, MotionCorrect and cnmf objects
for processing 1p microendoscopic data. The analysis pipeline is similar as in
the case of 2p data processing with core difference being the usage of the
CNMF-E algorithm for source extraction (as opposed to plain CNMF). Check
the companion paper for more details.

You can also run a large part of the pipeline with a single method
(cnmf.fit_file) See inside for details.

Demo is also available as a jupyter notebook (see demo_pipeline_cnmfE.ipynb)
"""

import argparse
import glob
import logging
import matplotlib.pyplot as plt
import numpy as np


import caiman as cm
from caiman.motion_correction import MotionCorrect
from caiman.source_extraction import cnmf
from caiman.source_extraction.cnmf import params as params
from caiman.utils.utils import download_demo
from caiman.utils.visualization import inspect_correlation_pnr



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
        fnames = [download_demo('data_endoscope.tif')]
    else:
        fnames = cfg.input
    filename_reorder = fnames # XXX What is this?

    # First set up some parameters for data and motion correction
    # dataset dependent parameters
    fr = 10                          # movie frame rate
    decay_time = 0.4                 # length of a typical transient in seconds

    # motion correction parameters
    motion_correct = True            # flag for motion correction
    pw_rigid = False                 # flag for pw-rigid motion correction

    gSig_filt = (3, 3)   # size of filter, in general gSig (see below),
    #                      change this one if algorithm does not work
    max_shifts = (5, 5)  # maximum allowed rigid shift
    strides = (48, 48)   # start a new patch for pw-rigid motion correction every x pixels
    overlaps = (24, 24)  # overlap between patches (size of patch strides+overlaps)
    # maximum deviation allowed for patch with respect to rigid shifts
    max_deviation_rigid = 3
    border_nan = 'copy'

    params_dict = {'fnames': fnames,
                   'fr': fr,
                   'decay_time': decay_time,
                   'pw_rigid': pw_rigid,
                   'max_shifts': max_shifts,
                   'gSig_filt': gSig_filt,
                   'strides': strides,
                   'overlaps': overlaps,
                   'max_deviation_rigid': max_deviation_rigid,
                   'border_nan': border_nan}

    opts = params.CNMFParams(params_dict=params_dict)

    c, dview, n_processes = cm.cluster.setup_cluster(backend=cfg.cluster_backend, n_processes=cfg.cluster_nproc)
    # Motion Correction
    #  The pw_rigid flag set above, determines where to use rigid or pw-rigid
    #  motion correction
    if motion_correct:
        # do motion correction rigid
        mc = MotionCorrect(fnames, dview=dview, **opts.get_group('motion'))
        mc.motion_correct(save_movie=True)
        fname_mc = mc.fname_tot_els if pw_rigid else mc.fname_tot_rig
        if pw_rigid:
            bord_px = np.ceil(np.maximum(np.max(np.abs(mc.x_shifts_els)),
                                         np.max(np.abs(mc.y_shifts_els)))).astype(int)
        else:
            bord_px = np.ceil(np.max(np.abs(mc.shifts_rig))).astype(int)
            plt.subplot(1, 2, 1); plt.imshow(mc.total_template_rig)  # plot template
            plt.subplot(1, 2, 2); plt.plot(mc.shifts_rig)  # plot rigid shifts
            plt.legend(['x shifts', 'y shifts'])
            plt.xlabel('frames')
            plt.ylabel('pixels')

        bord_px = 0 if border_nan == 'copy' else bord_px
        fname_new = cm.save_memmap(fname_mc, base_name='memmap_', order='C',
                                   border_to_0=bord_px)
    else:  # if no motion correction just memory map the file
        fname_new = cm.save_memmap(filename_reorder, base_name='memmap_',
                                   order='C', border_to_0=0, dview=dview)

    # load memory mappable file
    Yr, dims, T = cm.load_memmap(fname_new)
    images = Yr.T.reshape((T,) + dims, order='F')

    # Parameters for source extraction and deconvolution (CNMF-E algorithm)
    p = 1               # order of the autoregressive system
    K = None            # upper bound on number of components per patch, in general None for 1p data
    gSig = (3, 3)       # gaussian width of a 2D gaussian kernel, which approximates a neuron
    gSiz = (13, 13)     # average diameter of a neuron, in general 4*gSig+1
    Ain = None          # possibility to seed with predetermined binary masks
    merge_thr = .7      # merging threshold, max correlation allowed
    rf = 40             # half-size of the patches in pixels. e.g., if rf=40, patches are 80x80
    stride_cnmf = 20    # amount of overlap between the patches in pixels
    #                     (keep it at least large as gSiz, i.e 4 times the neuron size gSig)
    tsub = 2            # downsampling factor in time for initialization,
    #                     increase if you have memory problems
    ssub = 1            # downsampling factor in space for initialization,
    #                     increase if you have memory problems
    #                     you can pass them here as boolean vectors
    low_rank_background = None  # None leaves background of each patch intact,
    #                     True performs global low-rank approximation if gnb>0
    gnb = 0             # number of background components (rank) if positive,
    #                     else exact ring model with following settings
    #                         gnb= 0: Return background as b and W
    #                         gnb=-1: Return full rank background B
    #                         gnb<-1: Don't return background
    nb_patch = 0        # number of background components (rank) per patch if gnb>0,
    #                     else it is set automatically
    min_corr = .8       # min peak value from correlation image
    min_pnr = 10        # min peak to noise ration from PNR image
    ssub_B = 2          # additional downsampling factor in space for background
    ring_size_factor = 1.4  # radius of ring is gSiz*ring_size_factor

    opts.change_params(params_dict={'dims': dims,
                                    'method_init': 'corr_pnr',  # use this for 1 photon
                                    'K': K,
                                    'gSig': gSig,
                                    'gSiz': gSiz,
                                    'merge_thr': merge_thr,
                                    'p': p,
                                    'tsub': tsub,
                                    'ssub': ssub,
                                    'rf': rf,
                                    'stride': stride_cnmf,
                                    'only_init': True,    # set it to True to run CNMF-E
                                    'nb': gnb,
                                    'nb_patch': nb_patch,
                                    'method_deconvolution': 'oasis',       # could use 'cvxpy' alternatively
                                    'low_rank_background': low_rank_background,
                                    'update_background_components': True,  # sometimes setting to False improve the results
                                    'min_corr': min_corr,
                                    'min_pnr': min_pnr,
                                    'normalize_init': False,               # just leave as is
                                    'center_psf': True,                    # leave as is for 1 photon
                                    'ssub_B': ssub_B,
                                    'ring_size_factor': ring_size_factor,
                                    'del_duplicates': True,                # whether to remove duplicates from initialization
                                    'border_pix': bord_px})                # number of pixels to not consider in the borders)

    # compute some summary images (correlation and peak to noise)
    # change swap dim if output looks weird, it is a problem with tiffile
    cn_filter, pnr = cm.summary_images.correlation_pnr(images[::1], gSig=gSig[0], swap_dim=False)
    # if your images file is too long this computation will take unnecessarily
    # long time and consume a lot of memory. Consider changing images[::1] to
    # images[::5] or something similar to compute on a subset of the data

    # inspect the summary images and set the parameters
    inspect_correlation_pnr(cn_filter, pnr)
    # print parameters set above, modify them if necessary based on summary images
    print(f"Minimum correlation: {min_corr}") # min correlation of peak (from correlation image)
    print(f"Minimum peak to noise ratio: {min_pnr}")  # min peak to noise ratio

    # Run CMNF in patches
    cnm = cnmf.CNMF(n_processes=n_processes, dview=dview, Ain=Ain, params=opts)
    cnm.fit(images)

    # ALTERNATE WAY TO RUN THE PIPELINE AT ONCE (optional -- commented out)
    #   you can also perform the motion correction plus cnmf fitting steps
    #   simultaneously after defining your parameters object using
    #    cnm1 = cnmf.CNMF(n_processes, params=opts, dview=dview)
    #    cnm1.fit_file(motion_correct=True)

    # Quality Control: DISCARD LOW QUALITY COMPONENTS
    min_SNR = 2.5           # adaptive way to set threshold on the transient size
    r_values_min = 0.85    # threshold on space consistency (if you lower more components
    #                        will be accepted, potentially with worst quality)
    cnm.params.set('quality', {'min_SNR': min_SNR,
                               'rval_thr': r_values_min,
                               'use_cnn': False})
    cnm.estimates.evaluate_components(images, cnm.params, dview=dview)

    print(' ***** ')
    print('Number of total components: ', len(cnm.estimates.C))
    print('Number of accepted components: ', len(cnm.estimates.idx_components))

    # Play result movies
    if not cfg.no_play:
        cnm.dims = dims
        cnm.estimates.plot_contours(img=cn_filter, idx=cnm.estimates.idx_components)
        cnm.estimates.view_components(images, idx=cnm.estimates.idx_components)
        # fully reconstructed movie
        cnm.estimates.play_movie(images, q_max=99.5, magnification=2,
                                 include_bck=True, gain_res=10, bpx=bord_px)
        # movie without background
        cnm.estimates.play_movie(images, q_max=99.9, magnification=2,
                                 include_bck=False, gain_res=4, bpx=bord_px)

    # Stop the cluster and clean up log files
    cm.stop_server(dview=dview)

    if not cfg.keep_logs:
        log_files = glob.glob('*_LOG_*')
        for log_file in log_files:
            os.remove(log_file)

def handle_args():
    parser = argparse.ArgumentParser(description="Demonstrate CNMFE Pipeline")
    parser.add_argument("--keep_logs",  action="store_true", help="Keep temporary logfiles")
    parser.add_argument("--no_play",    action="store_true", help="Do not display results")
    parser.add_argument("--cluster_backend", default="multiprocessing", help="Specify multiprocessing, ipyparallel, or single to pick an engine")
    parser.add_argument("--cluster_nproc", type=int, default=None, help="Override automatic selection of number of workers to use")
    parser.add_argument("--input", action="append", help="File(s) to work on, provide multiple times for more files")
    parser.add_argument("--logfile",    help="If specified, log to the named file")
    return parser.parse_args()

########
main()

