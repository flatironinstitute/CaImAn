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
def run_volpy(fnames, options=None, do_motion_correction=True, do_memory_mapping=True, fr=400):
    #pass  # For compatibility between running under Spyder and the CLI

    # %%  Load demo movie and ROIs
    file_dir = os.path.split(fnames)[0]
    path_ROIs = [file for file in os.listdir(file_dir) if 'ROIs_gt' in file]
    if len(path_ROIs)>0:
        path_ROIs = path_ROIs[0]
    #path_ROIs = '/home/nel/NEL-LAB Dropbox/NEL/Datasets/voltage_lin/peyman_golshani/ROIs.hdf5'
    
#%% dataset dependent parameters
    # dataset dependent parameters
    fr = fr                                        # sample rate of the movie

    # motion correction parameters
    pw_rigid = False                                # flag for pw-rigid motion correction
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

# %% start a cluster for parallel processing
    c, dview, n_processes = cm.cluster.setup_cluster(
        backend='local', n_processes=None, single_thread=False)

# %%% MOTION CORRECTION
    # first we create a motion correction object with the specified parameters
    mc = MotionCorrect(fnames, dview=dview, **opts.get_group('motion'))
    # Run correction
    do_motion_correction = do_motion_correction
    if do_motion_correction:
        mc.motion_correct(save_movie=True)
    else: 
        mc_list = [file for file in os.listdir(file_dir) if 
                   (os.path.splitext(os.path.split(fnames)[-1])[0] in file and '.mmap' in file)]
        mc.mmap_file = [os.path.join(file_dir, mc_list[0])]
        print(f'reuse previously saved motion corrected file:{mc.mmap_file}')

# %% MEMORY MAPPING
    do_memory_mapping = do_memory_mapping
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
    # ! save summary image, it is used in GUI
    cm.movie(summary_images).save(fnames[:-5] + '_summary_images.tif')
    #plt.imshow(summary_images[0])    
    #%% three methods for segmentation
    methods_list = ['manual_annotation',       # manual annotation needs user to prepare annotated datasets same format as demo ROIs 
                    'gui_annotation',          # use gui to manually annotate neurons, but this is still under developing
                    'maskrcnn']                # maskrcnn is a convolutional network trained for finding neurons using summary images
    method = methods_list[0]
    if method == 'manual_annotation':                
        #with h5py.File(path_ROIs, 'r') as fl:
        #    ROIs = fl['mov'][()]
        ROIs = np.load(os.path.join(file_dir, path_ROIs))

    elif method == 'gui_annotation':
        # run volpy_gui file in the caiman/source_extraction/volpy folder
        # load the summary images you have just saved 
        # save the ROIs to the video folder
        path_ROIs =  caiman_datadir() + '/example_movies/volpy/gui_roi.hdf5'
        with h5py.File(path_ROIs, 'r') as fl:
            ROIs = fl['mov'][()]  
        
    elif method == 'maskrcnn':                 # Important!! make sure install keras before using mask rcnn
        weights_path = download_model('mask_rcnn')
        weights_path = '/home/nel/Code/NEL_LAB/Mask_RCNN/logs/neurons20200824T1032/mask_rcnn_neurons_0040.h5'
        ROIs = utils.mrcnn_inference(img=summary_images.transpose([1, 2, 0]), size_range=[5, 100],
                                     weights_path=weights_path, display_result=True) # size parameter decides size range of masks to be selected
        #np.save(os.path.join(file_dir, 'ROIs'), ROIs)            
# %% restart cluster to clean up memory
    cm.stop_server(dview=dview)
    c, dview, n_processes = cm.cluster.setup_cluster(
        backend='local', n_processes=None, single_thread=False, maxtasksperchild=1)

# %% parameters for trace denoising and spike extraction
    ROIs = ROIs                                   # region of interests
    index = list(range(len(ROIs)))                # index of neurons
    weights = None                                # reuse spatial weights 

    context_size = 35                             # number of pixels surrounding the ROI to censor from the background PCA
    flip_signal = True                            # Important!! Flip signal or not, True for Voltron indicator, False for others
    hp_freq_pb = 1 / 3                            # parameter for high-pass filter to remove photobleaching
    threshold_method = 'adaptive_threshold'                   # 'simple' or 'adaptive_threshold'
    min_spikes= 30                                # minimal spikes to be found
    threshold = 4                               # threshold for finding spikes, increase threshold to find less spikes
    do_plot = False                               # plot detail of spikes, template for the last iteration
    ridge_bg= 0.01                                 # ridge regression regularizer strength for background removement, larger value specifies stronger regularization 
    sub_freq = 20                                 # frequency for subthreshold extraction
    weight_update = 'ridge'                       # 'ridge' or 'NMF' for weight update
    n_iter = 2
    
    opts_dict={'fnames': fname_new,
               'ROIs': ROIs,
               'index': index,
               'weights': weights,
               'context_size': context_size,
               'flip_signal': flip_signal,
               'hp_freq_pb': hp_freq_pb,
               'threshold_method': threshold_method,
               'min_spikes':min_spikes,
               'threshold': threshold,
               'do_plot':do_plot,
               'ridge_bg':ridge_bg,
               'sub_freq': sub_freq,
               'weight_update': weight_update,
               'n_iter': n_iter}

    opts.change_params(params_dict=opts_dict);          
    
    if options is not None:
        print('using external options')
        opts.change_params(params_dict=options)
    else:
        print('not using external options')

#%% TRACE DENOISING AND SPIKE DETECTION
    vpy = VOLPY(n_processes=n_processes, dview=dview, params=opts)
    vpy.fit(n_processes=n_processes, dview=dview)

#%% visualization
    display_images = False
    if display_images:
        print(np.where(vpy.estimates['locality'])[0])    # neurons that pass locality test
        idx = np.where(vpy.estimates['locality'] > 0)[0]
        utils.view_components(vpy.estimates, img_corr, idx)
    
#%% reconstructed movie
# note the negative spatial weights is cutoff    
    if display_images:
        mv_all = utils.reconstructed_movie(vpy.estimates, fnames=mc.mmap_file,
                                           idx=idx, scope=(0,1000), flip_signal=flip_signal)
        mv_all.play(fr=40)
    
#%% save the result in .npy format 
    save_result = True
    if save_result:
        vpy.estimates['ROIs'] = ROIs
        save_name = f'volpy_{os.path.split(fnames)[1][:-5]}_{opts.volspike["threshold_method"]}_{opts.volspike["threshold"]}_{opts.volspike["weight_update"]}_bg_{opts.volspike["ridge_bg"]}'
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
    run_volpy