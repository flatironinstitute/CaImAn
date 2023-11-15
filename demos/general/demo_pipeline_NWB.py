#!/usr/bin/env python

"""
This script follows closely the demo_pipeline.py script but uses the
Neurodata Without Borders (NWB) file format for loading the input and saving
the output. It is meant as an example on how to use NWB files with CaImAn.
authors: @agiovann and @epnev
"""

import argparse
import cv2
from datetime import datetime
from dateutil.tz import tzlocal
import glob
import logging
import matplotlib.pyplot as plt
import numpy as np
import os

try:
    cv2.setNumThreads(0)
except:
    pass

import caiman as cm
from caiman.motion_correction import MotionCorrect
from caiman.paths import caiman_datadir
from caiman.source_extraction.cnmf import cnmf as cnmf
from caiman.source_extraction.cnmf import params as params
from caiman.utils.utils import download_demo

# TODO: Various code here isn't careful enough with paths and may produce files in
#       places alongside their sources, rather than in temporary directories. Come back
#       later and fix this once we've worked out where everything should go.

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

    # This demo will either accept a nwb file or it'll convert an existing movie into one
    if cfg.input is None:
        # default path, no input was provide, so we'll take the "Sue" sample data, convert it to a nwb, and then ingest it.
        # The convert target is the original fn, with the extension swapped to nwb.
        fr = float(15)      # imaging rate in frames per second
        pre_convert    = download_demo('Sue_2x_3000_40_-46.tif')
        convert_target = os.path.join(caiman_datadir(), 'Sue_2x_3000_40_-46.nwb')

        orig_movie = cm.load(pre_convert, fr=fr)
        # save file in NWB format with various additional info
        orig_movie.save(convert_target,
                        sess_desc="test",
                        identifier="demo 1",
                        imaging_plane_description='single plane',
                        emission_lambda=520.0, indicator='GCAMP6f',
                        location='parietal cortex',
                        experimenter='Sue Ann Koay', lab_name='Tank Lab',
                        institution='Princeton U',
                        experiment_description='Experiment Description',
                        session_id='Session 1',
                        var_name_hdf5='TwoPhotonSeries')
        fnames = [convert_target]
    elif cfg.input[0].endswith('.nwb'):
        if os.path.isfile(cfg.input[0]):
            # We were handed at least one nwb file and it exists either right here or as an absolute path
            for fn in cfg.input:
                if not os.path.isfile(fn):
                    raise Exception(f"Could not find input file {fn}")
                if not fn.endswith('.nwb'):
                    raise Exception(f"Cannot mix nwb and other file formats in input with this demo")
            fnames = cfg.input
        elif os.path.isfile(os.path.join(caiman_datadir(), cfg.input[0])):
            # We were handed at least one nwb file and it exists, but in the caiman_datadir()
            # So repeat the logic above except look in caiman_datadir()
            for fn in cfg.input:
                if not os.path.isfile(os.path.join(caiman_datadir(), fn)):
                    raise Exception(f"Could not find input file {fn}")
                if not fn.endswith('.nwb'):
                    raise Exception(f"Cannot mix nwb and other file formats in input with this demo")
            fnames = list(map(lambda n: os.path.join(caiman_datadir, n), cfg.input))

        else: # Someone mentioned an nwb file but we can't find it!
            raise Exception(f"Could not find referenced input file {cfg.input[0]}")
    else:
        # We were handed a file in some other format, so let's make an nwb file out of it and
        # then ingest that nwb file.
        if len(cfg.input) != 1:
            raise Exception("We're only willing to convert one file to nwb")
        pre_convert  = cfg.input[0]
        post_convert = os.path.splitext(pre_convert)[0] + ".nwb"
        fr = float(15) # TODO: Make this a cli parameter
        orig_movie = cm.load(pre_convert, fr=fr)
        # save file in NWB format with various additional info
        orig_movie.save(convert_target,
                        sess_desc="test",
                        identifier="demo 1",
                        imaging_plane_description='single plane',
                        emission_lambda=520.0, indicator='GCAMP6f', # Entirely synthetic, but if we don't know, what can we do?
                        location='unknown',
                        experimenter='Unknown Name', lab_name='Unknown Lab'
                        institution='Unknown University',
                        experiment_description='Experiment Description',
                        session_id='Session 1') # XXX do we need to provide var_name_hdf5?
        fnames = [post_convert]

    # estimates save path can be same or different from raw data path
    save_path = os.path.splitext(fnames[0])[0] + '_CNMF_estimates.nwb' # TODO: Make this a parameter?

    # We're done with all the file input parts

    # dataset dependent parameters
    decay_time = 0.4    # length of a typical transient in seconds
    dxy = (2., 2.)      # spatial resolution in x and y in (um per pixel)
    # note the lower than usual spatial resolution here
    max_shift_um = (12., 12.)       # maximum shift in um
    patch_motion_um = (100., 100.)  # patch size for non-rigid correction in um

    # First setup some parameters for data and motion correction
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
                   'border_nan': 'copy',
                   'var_name_hdf5': 'TwoPhotonSeries'}
    opts = params.CNMFParams(params_dict=params_dict)

    m_orig = cm.load_movie_chain(fnames, var_name_hdf5=opts.data['var_name_hdf5'])

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
    mc = MotionCorrect(fnames, dview=dview, var_name_hdf5=opts.data['var_name_hdf5'], **opts.get_group('motion'))
    # note that the file is not loaded in memory

    # Run (piecewise-rigid motion) correction using NoRMCorre
    mc.motion_correct(save_movie=True)

    # compare with original movie
    if not cfg.no_play:
        m_orig = cm.load_movie_chain(fnames, var_name_hdf5=opts.data['var_name_hdf5'])
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
    # for this step deconvolution is turned off (p=0)

    opts.change_params({'p': 0})
    cnm = cnmf.CNMF(n_processes, params=opts, dview=dview)
    cnm = cnm.fit(images)

    # ALTERNATE WAY TO RUN THE PIPELINE AT ONCE (optional)
    
    #   you can also perform the motion correction plus cnmf fitting steps
    #   simultaneously after defining your parameters object using
    #  cnm1 = cnmf.CNMF(n_processes, params=opts, dview=dview)
    #  cnm1.fit_file(motion_correct=True)

    # plot contours of found components
    Cn = cm.local_correlations(images, swap_dim=False)
    Cn[np.isnan(Cn)] = 0
    if not cfg.no_play:
        cnm.estimates.plot_contours(img=Cn)
    plt.title('Contour plots of found components')

    # save results in a separate file (just for demonstration purposes)
    cnm.estimates.Cn = Cn
    cnm.save(fname_new[:-4]+'hdf5')
    #cm.movie(Cn).save(fname_new[:-5]+'_Cn.tif')

    # RE-RUN seeded CNMF on accepted patches to refine and perform deconvolution
    cnm.params.change_params({'p': p})
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

    #  Save new estimates object (after adding correlation image)
    cnm2.estimates.Cn = Cn
    cnm2.save(fname_new[:-4] + 'hdf5')

    if not cfg.no_play:
        # Plot Components
        cnm2.estimates.plot_contours(img=Cn, idx=cnm2.estimates.idx_components)

        # View Traces (accepted and rejected)
        cnm2.estimates.view_components(images, img=Cn,
                                      idx=cnm2.estimates.idx_components)
        cnm2.estimates.view_components(images, img=Cn,
                                      idx=cnm2.estimates.idx_components_bad)

    # update object with selected components (optional)
    # cnm2.estimates.select_components(use_object=True)

    # Extract DF/F values
    cnm2.estimates.detrend_df_f(quantileMin=8, frames_window=250)

    # Show final traces
    if not cfg.no_play:
        cnm2.estimates.view_components(img=Cn)

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

    # save the results in the original NWB file
    cnm2.estimates.save_NWB(save_path, imaging_rate=fr, session_start_time=datetime.now(tzlocal()),
                            raw_data_file=fnames[0])

def handle_args():
    parser = argparse.ArgumentParser(description="Demonstrate 2P Pipeline using batch algorithm, with nwb files")
    parser.add_argument("--keep_logs",  action="store_true", help="Keep temporary logfiles")
    parser.add_argument("--no_play",    action="store_true", help="Do not display results")
    parser.add_argument("--cluster_backend", default="multiprocessing", help="Specify multiprocessing, ipyparallel, or single to pick an engine")
    parser.add_argument("--input", action="append", help="File(s) to work on, provide multiple times for more files")
    parser.add_argument("--logfile",    help="If specified, log to the named file")
    return parser.parse_args()

########
main()
