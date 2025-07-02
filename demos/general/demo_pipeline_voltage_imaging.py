#!/usr/bin/env python

"""
Demo pipeline for processing voltage imaging data. The processing pipeline
includes motion correction, memory mapping, segmentation, denoising and source
extraction. The demo shows how to construct the params, MotionCorrect and VOLPY 
objects and call the relevant functions. See inside for detail.
Dataset courtesy of Karel Svoboda Lab (Janelia Research Campus).
author: @caichangjia
"""

import argparse
import cv2
import glob
import h5py
import logging
import matplotlib.pyplot as plt
import numpy as np
import os

try:
    cv2.setNumThreads(0)
except:
    pass

import caiman
from caiman.motion_correction import MotionCorrect
from caiman.paths import caiman_datadir
from caiman.source_extraction.volpy import utils
from caiman.source_extraction.volpy.volparams import volparams
from caiman.source_extraction.volpy.volpy import VOLPY
from caiman.summary_images import local_correlations_movie_offline, mean_image
from caiman.utils.utils import download_demo, download_model


def main():
    cfg = handle_args()

    # Set up logging
    logger = logging.getLogger("caiman")
    logger.setLevel(logging.INFO) # Or another loglevel
    logfmt = logging.Formatter("[%(filename)s:%(funcName)20s():%(lineno)s] %(message)s")

    if cfg.logfile:
        handler = logging.FileHandler(cfg.logfile)
    else:
        handler = logging.StreamHandler()

    handler.setFormatter(logfmt)
    logger.addHandler(handler)

    # Figure out what data we're working on
    if cfg.input is None:
        # If no input is specified, use sample data, downloading if necessary
        fnames    = [download_demo('demo_voltage_imaging.hdf5')] # XXX do we need to use a separate directory?
        path_ROIs = [download_demo('demo_voltage_imaging_ROIs.hdf5')]
    else:
        fnames = cfg.input
        path_ROIs = cfg.pathinput

    #  Load demo movie and ROIs
    file_dir = os.path.split(fnames)[0]
    
    # dataset dependent parameters
    fr = 400                                        # sample rate of the movie

    # motion correction parameters
    pw_rigid = False                                # flag for pw-rigid motion correction
    gSig_filt = (3, 3)                              # size of filter, in general gSig (see below),
                                                    # change this one if algorithm does not work
    max_shifts = (5, 5)                             # maximum allowed rigid shift
    strides = (48, 48)                              # start a new patch for pw-rigid motion correction every x pixels
    overlaps = (24, 24)                             # overlap between patches (size of patch strides+overlaps)
    max_deviation_rigid = 3                         # maximum deviation allowed for patch with respect to rigid shifts
    border_nan = 'copy'

    opts_dict = {'fnames': fnames,
                   'fr': fr,
                   'pw_rigid': pw_rigid,
                   'max_shifts': max_shifts,
                   'gSig_filt': gSig_filt,
                   'strides': strides,
                   'overlaps': overlaps,
                   'max_deviation_rigid': max_deviation_rigid,
                   'border_nan': border_nan}

    opts = volparams(params_dict=opts_dict)

    # play the movie (optional)
    # playing the movie using opencv. It requires loading the movie in memory.
    # To close the movie press q
    if not cfg.no_play:
        m_orig = caiman.load(fnames)
        ds_ratio = 0.2
        moviehandle = m_orig.resize(1, 1, ds_ratio)
        moviehandle.play(q_max=99.5, fr=40, magnification=4)

    # start a cluster for parallel processing
    c, dview, n_processes = caiman.cluster.setup_cluster(backend=cfg.cluster_backend, n_processes=cfg.cluster_nproc)

    # %%% MOTION CORRECTION
    # first we create a motion correction object with the specified parameters
    mc = MotionCorrect(fnames, dview=dview, **opts.get_group('motion'))
    # Run correction
    do_motion_correction = True
    if do_motion_correction:
        mc.motion_correct(save_movie=True)
    else: 
        mc_list = [file for file in os.listdir(file_dir) if 
                   (os.path.splitext(os.path.split(fnames)[-1])[0] in file and '.mmap' in file)]
        mc.mmap_file = [os.path.join(file_dir, mc_list[0])]
        print(f'reuse previously saved motion corrected file:{mc.mmap_file}')

    # compare with original movie
    if not cfg.no_play:
        m_orig = caiman.load(fnames)
        m_rig = caiman.load(mc.mmap_file)
        ds_ratio = 0.2
        moviehandle = caiman.concatenate([m_orig.resize(1, 1, ds_ratio),
                                      m_rig.resize(1, 1, ds_ratio)], axis=2)
        moviehandle.play(fr=40, q_max=99.5, magnification=4)  # press q to exit

    # MEMORY MAPPING
    if not cfg.no_memmap:
        border_to_0 = 0 if mc.border_nan == 'copy' else mc.border_to_0
        # you can include the boundaries of the FOV if you used the 'copy' option
        # during motion correction, although be careful about the components near
        # the boundaries
        
        # memory map the file in order 'C'
        fname_new = caiman.save_memmap_join(mc.mmap_file, base_name='memmap_' + os.path.splitext(os.path.split(fnames)[-1])[0],
                                        add_to_mov=border_to_0, dview=dview)  # exclude border
    else: 
        mmap_list = [file for file in os.listdir(file_dir) if 
                     ('memmap_' + os.path.splitext(os.path.split(fnames)[-1])[0]) in file]
        fname_new = os.path.join(file_dir, mmap_list[0])
        print(f'reuse previously saved memory mapping file: {fname_new}')
    
    # SEGMENTATION
    # create summary images
    img = mean_image(mc.mmap_file[0], window = 1000, dview=dview)
    img = (img-np.mean(img))/np.std(img)
    
    gaussian_blur = False        # Use gaussian blur when there is too much noise in the video
    Cn = local_correlations_movie_offline(mc.mmap_file[0], fr=fr, window=fr*4, 
                                          stride=fr*4, winSize_baseline=fr, 
                                          remove_baseline=True, gaussian_blur=gaussian_blur,
                                          dview=dview).max(axis=0)
    img_corr = (Cn-np.mean(Cn))/np.std(Cn)
    summary_images = np.stack([img, img, img_corr], axis=0).astype(np.float32)
    # save summary images which are used in the VolPy GUI
    caiman.movie(summary_images).save(fnames[:-5] + '_summary_images.tif')
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(summary_images[0]); axs[1].imshow(summary_images[2])
    axs[0].set_title('mean image'); axs[1].set_title('corr image');

    if cfg.method == 'manual_annotation':
        with h5py.File(path_ROIs, 'r') as fl:
            ROIs = fl['mov'][()]  

    elif cfg.method == 'maskrcnn':
        weights_path = download_model('mask_rcnn')    
        ROIs = utils.mrcnn_inference(img=summary_images.transpose([1, 2, 0]), size_range=[5, 22],
                                     weights_path=weights_path, display_result=True) # size parameter decides size range of masks to be selected
        caiman.movie(ROIs).save(fnames[:-5] + '_mrcnn_ROIs.hdf5')

    elif cfg.method == 'gui_annotation':
        # run volpy_gui.py file in the caiman/source_extraction/volpy folder
        gui_ROIs =  os.path.join(caiman_datadir(), 'example_movies', 'gui_roi.hdf5')
        with h5py.File(gui_ROIs, 'r') as fl:
            ROIs = fl['mov'][()]

    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(summary_images[0]); axs[1].imshow(ROIs.sum(0))
    axs[0].set_title('mean image'); axs[1].set_title('masks')
        
    # restart cluster to clean up memory
    caiman.stop_server(dview=dview)
    c, dview, n_processes = caiman.cluster.setup_cluster(backend=cfg.cluster_backend, n_processes=cfg.cluster_nproc)


    # parameters for trace denoising and spike extraction
    ROIs = ROIs                                   # region of interests
    index = list(range(len(ROIs)))                # index of neurons
    weights = None                                # if None, use ROIs for initialization; to reuse weights check reuse weights block 

    template_size = 0.02                          # half size of the window length for spike templates, default is 20 ms 
    context_size = 35                             # number of pixels surrounding the ROI to censor from the background PCA
    visualize_ROI = False                         # whether to visualize the region of interest inside the context region
    flip_signal = True                            # Important!! Flip signal or not, True for Voltron indicator, False for others
    hp_freq_pb = 1 / 3                            # parameter for high-pass filter to remove photobleaching
    clip = 100                                    # maximum number of spikes to form spike template
    threshold_method = 'adaptive_threshold'       # adaptive_threshold or simple 
    min_spikes= 10                                # minimal spikes to be found
    pnorm = 0.5                                   # a variable deciding the amount of spikes chosen for adaptive threshold method
    threshold = 3                                 # threshold for finding spikes only used in simple threshold method, Increase the threshold to find less spikes
    do_plot = False                               # plot detail of spikes, template for the last iteration
    ridge_bg= 0.01                                # ridge regression regularizer strength for background removement, larger value specifies stronger regularization 
    sub_freq = 20                                 # frequency for subthreshold extraction
    weight_update = 'ridge'                       # ridge or NMF for weight update
    n_iter = 2                                    # number of iterations alternating between estimating spike times and spatial filters
    
    opts_dict={'fnames': fname_new,
               'ROIs': ROIs,
               'index': index,
               'weights': weights,
               'template_size': template_size, 
               'context_size': context_size,
               'visualize_ROI': visualize_ROI, 
               'flip_signal': flip_signal,
               'hp_freq_pb': hp_freq_pb,
               'clip': clip,
               'threshold_method': threshold_method,
               'min_spikes':min_spikes,
               'pnorm': pnorm, 
               'threshold': threshold,
               'do_plot':do_plot,
               'ridge_bg':ridge_bg,
               'sub_freq': sub_freq,
               'weight_update': weight_update,
               'n_iter': n_iter}

    opts.change_params(params_dict=opts_dict);          

    # TRACE DENOISING AND SPIKE DETECTION
    vpy = VOLPY(n_processes=n_processes, dview=dview, params=opts)
    vpy.fit(n_processes=n_processes, dview=dview);
   
    # visualization
    if not cfg.no_play:
        print(np.where(vpy.estimates['locality'])[0])    # neurons that pass locality test
        idx = np.where(vpy.estimates['locality'] > 0)[0]
        utils.view_components(vpy.estimates, img_corr, idx)
    
    # reconstructed movie 
    # note the negative spatial weights are cutoff    
    if not cfg.no_play:
        mv_all = utils.reconstructed_movie(vpy.estimates.copy(), fnames=mc.mmap_file,
                                           idx=idx, scope=(0,1000), flip_signal=flip_signal)
        mv_all.play(fr=40, magnification=3)
    
    # save the result in .npy format 
    save_result = True
    if save_result:
        vpy.estimates['ROIs'] = ROIs
        vpy.estimates['params'] = opts
        save_name = f'volpy_{os.path.split(fnames)[1][:-5]}_{threshold_method}'
        np.save(os.path.join(file_dir, save_name), vpy.estimates)
        
    # reuse weights 
    # set weights = reuse_weights in opts_dict dictionary
    estimates = np.load(os.path.join(file_dir, save_name+'.npy'), allow_pickle=True).item()
    reuse_weights = []
    for idx in range(ROIs.shape[0]):
        coord = estimates['context_coord'][idx]
        w = estimates['weights'][idx][coord[0][0]:coord[1][0]+1, coord[0][1]:coord[1][1]+1] 
        plt.figure(); plt.imshow(w);plt.colorbar(); plt.show()
        reuse_weights.append(w)
    
    # Stop the cluster and clean up log files
    caiman.stop_server(dview=dview)

    if not cfg.keep_logs:
        log_files = glob.glob('*_LOG_*')
        for log_file in log_files:
            os.remove(log_file)

def handle_args():
    parser = argparse.ArgumentParser(description="Demonstrate voltage imaging pipeline")
    parser.add_argument("--no_memmap",  action="store_true", help="Avoid memory mapping")
    parser.add_argument("--keep_logs",  action="store_true", help="Keep temporary logfiles")
    parser.add_argument("--no_play",    action="store_true", help="Do not display results")
    parser.add_argument("--cluster_backend", default="multiprocessing", help="Specify multiprocessing, ipyparallel, or single to pick an engine")
    parser.add_argument("--cluster_nproc", type=int, default=None, help="Override automatic selection of number of workers to use")
    parser.add_argument("--input", action="append", help="File(s) to work on, provide multiple times for more files")
    parser.add_argument("--pathinput", action="append", help="Path input file(s) to work on, provide multiple times for more files")
    parser.add_argument("--logfile",    help="If specified, log to the named file")
    parser.add_argument("--method", default="manual_annotation", choices=["manual_annotation", "maskrcnn", "gui_annotation"], help="Segmentation method")
    return parser.parse_args()

########
main()

