#!/usr/bin/env python
"""
Demo pipeline for processing voltage imaging data. The processing pipeline
includes motion correction, memory mapping, segmentation, denoising and source
extraction. The demo shows how to construct the params, MotionCorrect and VOLPY 
objects and call the relevant functions. See inside for detail.
Dataset courtesy of Karel Svoboda Lab (Janelia Research Campus).
author: @caichangjia
"""
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

try:
    if __IPYTHON__:
        # this is used for debugging purposes only. allows to reload classes
        # when changed
        get_ipython().magic('load_ext autoreload')
        get_ipython().magic('autoreload 2')
except NameError:
    pass

import caiman as cm
from caiman.motion_correction import MotionCorrect
from caiman.paths import caiman_datadir
from caiman.source_extraction.volpy import utils
from caiman.source_extraction.volpy.volparams import volparams
from caiman.source_extraction.volpy.volpy import VOLPY
from caiman.summary_images import local_correlations_movie_offline
from caiman.summary_images import mean_image
from caiman.utils.utils import download_demo, download_model

# %%
# Set up the logger (optional); change this if you like.
# You can log to a file using the filename parameter, or make the output more
# or less verbose by setting level to logging.DEBUG, logging.INFO,
# logging.WARNING, or logging.ERROR
logging.basicConfig(format=
                    "%(relativeCreated)12d [%(filename)s:%(funcName)20s():%(lineno)s]" \
                    "[%(process)d] %(message)s",
                    level=logging.INFO)

# %%
def main():
    pass  # For compatibility between running under Spyder and the CLI

    # %%  Load demo movie and ROIs
    fnames = download_demo('demo_voltage_imaging.hdf5', 'volpy')  # file path to movie file (will download if not present)
    path_ROIs = download_demo('demo_voltage_imaging_ROIs.hdf5', 'volpy')  # file path to ROIs file (will download if not present)
    file_dir = os.path.split(fnames)[0]
    
#%% dataset dependent parameters
    # dataset dependent parameters
    fr = 400                                        # sample rate of the movie

    # motion correction parameters
    pw_rigid = False                                 # flag for pw-rigid motion correction
    gSig_filt = (3, 3)                              # size of filter, in general gSig (see below),
                                                    # change this one if algorithm does not work
    max_shifts = (5, 5)                             # maximum allowed rigid shift
    strides = (48, 48)                              # start a new patch for pw-rigid motion correction every x pixels
    overlaps = (24, 24)                             # overlap between pathes (size of patch strides+overlaps)
    max_deviation_rigid = 3                         # maximum deviation allowed for patch with respect to rigid shifts
    border_nan = 'copy'

    opts_dict = {
        'fnames': fnames,
        'fr': fr,
        'pw_rigid': pw_rigid,
        'max_shifts': max_shifts,
        'gSig_filt': gSig_filt,
        'strides': strides,
        'overlaps': overlaps,
        'max_deviation_rigid': max_deviation_rigid,
        'border_nan': border_nan
    }

    opts = volparams(params_dict=opts_dict)

# %% play the movie (optional)
    # playing the movie using opencv. It requires loading the movie in memory.
    # To close the movie press q
    display_images = False

    if display_images:
        m_orig = cm.load(fnames)
        ds_ratio = 0.2
        moviehandle = m_orig.resize(1, 1, ds_ratio)
        moviehandle.play(q_max=99.5, fr=40, magnification=4)

# %% start a cluster for parallel processing
    c, dview, n_processes = cm.cluster.setup_cluster(
        backend='local', n_processes=None, single_thread=False)

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

# %% compare with original movie
    if display_images:
        m_orig = cm.load(fnames)
        m_rig = cm.load(mc.mmap_file)
        ds_ratio = 0.2
        moviehandle = cm.concatenate([m_orig.resize(1, 1, ds_ratio),
                                      m_rig.resize(1, 1, ds_ratio)], axis=2)
        moviehandle.play(fr=40, q_max=99.5, magnification=4)  # press q to exit

# %% MEMORY MAPPING
    do_memory_mapping = True
    if do_memory_mapping:
        border_to_0 = 0 if mc.border_nan == 'copy' else mc.border_to_0
        # you can include the boundaries of the FOV if you used the 'copy' option
        # during motion correction, although be careful about the components near
        # the boundaries
        
        # memory map the file in order 'C'
        fname_new = cm.save_memmap_join(mc.mmap_file, base_name='memmap_' + os.path.splitext(os.path.split(fnames)[-1])[0],
                                        add_to_mov=border_to_0, dview=dview)  # exclude border
    else: 
        mmap_list = [file for file in os.listdir(file_dir) if 
                     ('memmap_' + os.path.splitext(os.path.split(fnames)[-1])[0]) in file]
        fname_new = os.path.join(file_dir, mmap_list[0])
        print(f'reuse previously saved memory mapping file:{fname_new}')
    
# %% SEGMENTATION
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
    cm.movie(summary_images).save(fnames[:-5] + '_summary_images.tif')
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(summary_images[0]); axs[1].imshow(summary_images[2])
    axs[0].set_title('mean image'); axs[1].set_title('corr image')

    #%% methods for segmentation
    methods_list = ['manual_annotation',       # manual annotations need users to prepare annotated datasets same format as demo_voltage_imaging_ROIs.hdf5 
                    'maskrcnn',                # Mask R-CNN is a convolutional neural networks trained for detecting neurons in summary images
                    'gui_annotation']           # use VolPy GUI to correct outputs of Mask R-CNN or annotate new datasets easily 
                    
    method = methods_list[2]
    if method == 'manual_annotation':                
        with h5py.File(path_ROIs, 'r') as fl:
            ROIs = fl['mov'][()]  

    elif method == 'maskrcnn':                 # Important!! make sure install keras before using mask rcnn
        weights_path = download_model('mask_rcnn')
        ROIs = utils.mrcnn_inference(img=summary_images.transpose([1, 2, 0]), size_range=[5, 22],
                                     weights_path=weights_path, display_result=True) # size parameter decides size range of masks to be selected
        cm.movie(ROIs).save(fnames[:-5] + 'mrcnn_ROIs.hdf5')

    elif method == 'gui_annotation':
        # run volpy_gui.py file in the caiman/source_extraction/volpy folder
        # load the summary images you have just saved and select neurons. Save the ROIs to the video folder
        # the gui can be used to correct outputs of Mask R-CNN results by loading ROIs from the mrcnn_ROIs.hdf5 file
        gui_ROIs =  caiman_datadir() + '/example_movies/volpy/gui_roi.hdf5'
        with h5py.File(gui_ROIs, 'r') as fl:
            ROIs = fl['mov'][()]

    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(summary_images[0]); axs[1].imshow(ROIs.sum(0))
    axs[0].set_title('mean image'); axs[1].set_title('masks')


        
# %% restart cluster to clean up memory
    cm.stop_server(dview=dview)
    c, dview, n_processes = cm.cluster.setup_cluster(
        backend='local', n_processes=None, single_thread=False, maxtasksperchild=1)

# %% parameters for trace denoising and spike extraction
    ROIs = ROIs                                   # region of interests
    index = list(range(len(ROIs)))                # index of neurons
    weights = None                                # reuse spatial weights 

    context_size = 35                             # number of pixels surrounding the ROI to censor from the background PCA
    visualize_ROI = False                         # whether to visualize the region of interest inside the context region
    flip_signal = True                            # Important!! Flip signal or not, True for Voltron indicator, False for others
    hp_freq_pb = 1 / 3                            # parameter for high-pass filter to remove photobleaching
    clip = 100                                    # maximum number of spikes to form spike template
    threshold_method = 'adaptive_threshold'       # 'adaptive_threshold' or 'simple' 
    min_spikes= 10                                # minimal spikes to be found
    pnorm = 0.5                                   # a variable deciding the amount of spikes chosen for adaptive threshold method
    threshold = 3                                 # threshold for finding spikes only used in simple threshold method, Increase the threshold to find less spikes
    do_plot = False                               # plot detail of spikes, template for the last iteration
    ridge_bg= 0.01                                # ridge regression regularizer strength for background removement, larger value specifies stronger regularization 
    sub_freq = 20                                 # frequency for subthreshold extraction
    weight_update = 'ridge'                       # 'ridge' or 'NMF' for weight update
    n_iter = 2                                    # number of iterations alternating between estimating spike times and spatial filters
    
    opts_dict={'fnames': fname_new,
               'ROIs': ROIs,
               'index': index,
               'weights': weights,
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

#%% TRACE DENOISING AND SPIKE DETECTION
    vpy = VOLPY(n_processes=n_processes, dview=dview, params=opts)
    vpy.fit(n_processes=n_processes, dview=dview)

#%% visualization
    display_images = True
    if display_images:
        print(np.where(vpy.estimates['locality'])[0])    # neurons that pass locality test
        idx = np.where(vpy.estimates['locality'] > 0)[0]
        utils.view_components(vpy.estimates, img_corr, idx)
    
#%% reconstructed movie
# note the negative spatial weights is cutoff    
    if display_images:
        mv_all = utils.reconstructed_movie(vpy.estimates.copy(), fnames=mc.mmap_file,
                                           idx=idx, scope=(0,1000), flip_signal=flip_signal)
        mv_all.play(fr=40)
    
#%% save the result in .npy format 
    save_result = True
    if save_result:
        vpy.estimates['ROIs'] = ROIs
        vpy.estimates['params'] = opts
        save_name = f'volpy_{os.path.split(fnames)[1][:-5]}_{threshold_method}'
        np.save(os.path.join(file_dir, save_name), vpy.estimates)
    
# %% STOP CLUSTER and clean up log files
    cm.stop_server(dview=dview)
    log_files = glob.glob('*_LOG_*')
    for log_file in log_files:
        os.remove(log_file)

# %%
# This is to mask the differences between running this demo in Spyder
# versus from the CLI
if __name__ == "__main__":
    main()