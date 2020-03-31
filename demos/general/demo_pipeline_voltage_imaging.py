#!/usr/bin/env python
"""
Demo pipeline for processing voltage imaging data. The processing pipeline
includes motion correction, memory mapping, segmentation, denoising and source
extraction. The demo shows how to construct the params, MotionCorrect and VOLPY 
objects and call the relevant functions. See inside for detail.

Dataset courtesy of Karel Svoboda Lab (Janelia Research Campus).
author: @caichangjia
"""
import os
import cv2
import glob
import logging
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import h5py

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
from caiman.utils.utils import download_demo, download_model
from caiman.source_extraction.volpy.volparams import volparams
from caiman.source_extraction.volpy.volpy import VOLPY
from caiman.source_extraction.volpy.mrcnn import visualize, neurons
import caiman.source_extraction.volpy.mrcnn.model as modellib
from caiman.paths import caiman_datadir
from caiman.summary_images import local_correlations_movie_offline
from caiman.summary_images import mean_image
from caiman.source_extraction.volpy.utils import quick_annotation


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
    #fnames = '/home/nel/data/voltage_data/voltage/peyman_golshani/movie3.hdf5'
    #path_ROIs = '/home/nel/data/voltage_data/voltage/peyman_golshani/ROIs.hdf5'
    #fnames = '/home/nel/data/voltage_data/simul_electr/johannes/09212017Fish1-1/registered.hdf5'
    #fnames = '/home/nel/data/voltage_data/volpy_paper/figure1/FOV4_50um.hdf5'
    #path_ROIs = '/home/nel/data/voltage_data/volpy_paper/figure1/FOV4_50um_ROIs.hdf5'
    #fnames = '/home/nel/data/voltage_data/volpy_paper/supp/IVQ48_S7_FOV5_15000.hdf5'
    #path_ROIs = '/home/nel/data/voltage_data/volpy_paper/supp/IVQ48_S7_FOV5_15000_ROIs.hdf5'
    #fnames = '/home/nel/data/voltage_data/volpy_paper/supp/06152017Fish1-2_5000.hdf5'
    #path_ROIs = '/home/nel/data/voltage_data/volpy_paper/supp/06152017Fish1-2_ROIs.hdf5'
    #fnames = '/home/nel/data/voltage_data/simul_electr/johannes/09282017Fish1-1/registered.hdf5'
    #fnames = '/home/nel/data/voltage_data/simul_electr/johannes/10052017Fish2-2/registered.hdf5'
    #fnames = '/home/nel/data/voltage_data/simul_electr/kaspar/Ground truth data/Session1/Session1.hdf5'
    #fnames = '/home/nel/data/voltage_data/volpy_paper/memory/403106_3min_10000.hdf5'
    #path_ROIs = '/home/nel/data/voltage_data/volpy_paper/memory/ROI.npz'
    #fnames = '/home/nel/data/voltage_data/volpy_paper/figure1/FOV4_50um.hdf5'

#%% dataset dependent parameters
    # dataset dependent parameters
    fr = 400                                        # sample rate of the movie
                                                   
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

# %% play the movie (optional)
    # playing the movie using opencv. It requires loading the movie in memory.
    # To close the video press q
    display_images = False

    if display_images:
        m_orig = cm.load(fnames)
        ds_ratio = 0.2
        moviehandle = m_orig.resize(1, 1, ds_ratio)
        moviehandle.play(q_max=99.5, fr=60, magnification=6)

# %% start a cluster for parallel processing
    c, dview, n_processes = cm.cluster.setup_cluster(
        backend='local', n_processes=None, single_thread=False)

# %%% MOTION CORRECTION
    # first we create a motion correction object with the specified parameters
    mc = MotionCorrect(fnames, dview=dview, **opts.get_group('motion'))
    # Run correction
    mc.motion_correct(save_movie=True)

# %% compare with original movie
    if display_images:
        m_orig = cm.load(fnames)
        m_rig = cm.load(mc.mmap_file)
        ds_ratio = 0.2
        moviehandle = cm.concatenate([m_orig.resize(1, 1, ds_ratio) - mc.min_mov * mc.nonneg_movie,
                                      m_rig.resize(1, 1, ds_ratio)], axis=2)
        moviehandle.play(fr=60, q_max=99.5, magnification=4)  # press q to exit

# %% MEMORY MAPPING
    border_to_0 = 0 if mc.border_nan == 'copy' else mc.border_to_0
    # you can include the boundaries of the FOV if you used the 'copy' option
    # during motion correction, although be careful about the components near
    # the boundaries
    
    # memory map the file in order 'C'
    fname_new = cm.save_memmap_join(mc.mmap_file, base_name='memmap_',
                                    add_to_mov=border_to_0, dview=dview)  # exclude border

    
    # %% change fnames to the new motion corrected one
    #fname_new = '/home/nel/caiman_data/example_movies/volpy/memmap__d1_100_d2_100_d3_1_order_C_frames_20000_.mmap'
    #fname_new = '/home/nel/data/voltage_data/voltage/peyman_golshani/memmap__d1_128_d2_128_d3_1_order_C_frames_3000_.mmap'
    #fname_new = '/home/nel/data/voltage_data/simul_electr/johannes/09212017Fish1-1/memmap__d1_44_d2_96_d3_1_order_C_frames_37950_.mmap'
    #fname_new = '/home/nel/data/voltage_data/volpy_paper/figure1/memmap__d1_128_d2_512_d3_1_order_C_frames_20000_.mmap'
    #fname_new = '/home/nel/data/voltage_data/volpy_paper/supp/memmap__d1_212_d2_96_d3_1_order_C_frames_15000_.mmap'
    #fname_new = '/home/nel/data/voltage_data/volpy_paper/supp/memmap__d1_364_d2_320_d3_1_order_C_frames_5000_.mmap'
    opts.change_params(params_dict={'fnames': fname_new})

# %% SEGMENTATION
    # create summary images
    img = mean_image(mc.mmap_file[0], window = 1000, dview=dview)
    img = (img-np.mean(img))/np.std(img)
    Cn = local_correlations_movie_offline(mc.mmap_file[0], fr=fr, window=1500, 
                                          stride=1500, winSize_baseline=400, remove_baseline=True, dview=dview).max(axis=0)
    img_corr = (Cn-np.mean(Cn))/np.std(Cn)
    summary_image = np.stack([img, img, img_corr], axis=2).astype(np.float32) 
    
    #%% three methods for segmentation
    methods_list = ['manual_annotation',        # manual annotation needs user to prepare annotated datasets same format as demo ROIs 
                    'quick_annotation',         # quick annotation annotates data with simple interface in python
                    'maskrcnn' ]                # maskrcnn is a convolutional network trained for finding neurons using summary images
    method = methods_list[0]
    if method == 'manual_annotation':                
        with h5py.File(path_ROIs, 'r') as fl:
            ROIs = fl['mov'][()]  # load ROIs

    elif method == 'quick_annotation': 
        ROIs = quick_annotation(img_corr, min_radius=4, max_radius=10)

    elif method == 'maskrcnn':
        config = neurons.NeuronsConfig()
        class InferenceConfig(config.__class__):
            # Run detection on one image at a time
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MIN_CONFIDENCE = 0.7
            IMAGE_RESIZE_MODE = "pad64"
            IMAGE_MAX_DIM = 512
            RPN_NMS_THRESHOLD = 0.7
            POST_NMS_ROIS_INFERENCE = 1000
        config = InferenceConfig()
        config.display()
        model_dir = os.path.join(caiman_datadir(), 'model')
        DEVICE = "/cpu:0"  # /cpu:0 or /gpu:0
        with tf.device(DEVICE):
            model = modellib.MaskRCNN(mode="inference", model_dir=model_dir,
                                      config=config)
        weights_path = download_model('mask_rcnn')
        model.load_weights(weights_path, by_name=True)
        results = model.detect([summary_image], verbose=1)
        r = results[0]
        ROIs = r['masks'].transpose([2, 0, 1])

        display_result = True
        if display_result:
            _, ax = plt.subplots(1,1, figsize=(16,16))
            visualize.display_instances(summary_image, r['rois'], r['masks'], r['class_ids'], 
                                    ['BG', 'neurons'], r['scores'], ax=ax,
                                    title="Predictions")

# %% restart cluster to clean up memory
    cm.stop_server(dview=dview)
    c, dview, n_processes = cm.cluster.setup_cluster(
        backend='local', n_processes=None, single_thread=False, maxtasksperchild=1)

# %% parameters for trace denoising and spike extraction
    fnames = fname_new                            # change file
    ROIs = ROIs                                   # region of interests
    index = list(range(len(ROIs)))                 # index of neurons
    weights = None                                # reuse spatial weights 

    tau_lp = 3                                    # parameter for high-pass filter to remove photobleaching
    threshold = 4                                 # threshold for finding spikes, increase threshold to find less spikes
    contextSize = 35                              # number of pixels surrounding the ROI to censor from the background PCA
    flip_signal = True                            # Important! Flip signal or not, True for Voltron indicator, False for others
    sub_freq = 50                                 # frequency for subthreshold extraction

    opts_dict={'fnames': fnames,
               'ROIs': ROIs,
               'index': index,
               'weights': weights,
               'tau_lp': tau_lp,
               'threshold': threshold,
               'contextSize': contextSize,
               'flip_signal': flip_signal,
               'sub_freq': sub_freq}

    opts.change_params(params_dict=opts_dict);          

#%% Trace Denoising and Spike Extraction
    vpy = VOLPY(n_processes=n_processes, dview=dview, params=opts)
    vpy.fit(n_processes=n_processes, dview=dview)

#%% visualization
    print(np.where(vpy.estimates['locality'])[0])    # neurons that pass locality test
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider
    n = len(opts.data['index']) 
    fig = plt.figure(figsize=(10, 10))
    img = summary_image[:,:,2]

    axcomp = plt.axes([0.05, 0.05, 0.9, 0.03])
    ax1 = plt.axes([0.05, 0.55, 0.4, 0.4])
    ax3 = plt.axes([0.55, 0.55, 0.4, 0.4])
    ax2 = plt.axes([0.05, 0.1, 0.9, 0.4])    
    s_comp = Slider(axcomp, 'Component', 0, n, valinit=0)
    vmax = np.percentile(img, 98)
    
    def arrow_key_image_control(event):

        if event.key == 'left':
            new_val = np.round(s_comp.val - 1)
            if new_val < 0:
                new_val = 0
            s_comp.set_val(new_val)

        elif event.key == 'right':
            new_val = np.round(s_comp.val + 1)
            if new_val > n :
                new_val = n  
            s_comp.set_val(new_val)
        else:
            pass
        
    def update(val):
        i = np.int(np.round(s_comp.val))
        print(('Component:' + str(i)))

        if i < n:
            
            ax1.cla()
            imgtmp = vpy.estimates['spatialFilter'][i]
            ax1.imshow(imgtmp, interpolation='None', cmap=plt.cm.gray, vmax=np.max(imgtmp)*0.5, vmin=0)
            ax1.set_title('Spatial component ' + str(i + 1))
            ax1.axis('off')
            
            ax2.cla()
            ax2.plot(vpy.estimates['trace_raw'][i], alpha=0.8)
            ax2.plot(vpy.estimates['trace_sub'][i])            
            ax2.plot(vpy.estimates['trace_recons'][i], alpha = 0.4, color='red')
            ax2.plot(vpy.estimates['trace_lp'][i], alpha = 0.4, color='green')

            ax2.plot(vpy.estimates['spikes'][i],
                     1.05 * np.max(vpy.estimates['trace_raw'][i]) * np.ones(vpy.estimates['spikes'][i].shape),
                     color='r', marker='.', fillstyle='none', linestyle='none')
            ax2.set_title('Signal and spike times' + str(i + 1))
            ax2.legend(labels=['Raw signal', 'Subthreshold activity', 'Reconstructed signal', 'Low pass signal''Spike time'])
            ax2.text(0.1, 0.1,'snr:'+str(round(vpy.estimates['snr'][i],2)), horizontalalignment='center', verticalalignment='center', transform = ax2.transAxes)
            ax2.text(0.1, 0.07,'num_spikes:'+str(len(vpy.estimates['spikes'][i])), horizontalalignment='center', verticalalignment='center', transform = ax2.transAxes)            
            ax2.text(0.1, 0.04,'locality_test:'+str(vpy.estimates['locality'][i]), horizontalalignment='center', verticalalignment='center', transform = ax2.transAxes)            
            
            ax3.cla()
            ax3.imshow(img, interpolation='None', cmap=plt.cm.gray, vmax=vmax)
            imgtmp2 = imgtmp.copy()
            imgtmp2[imgtmp2 == 0] = np.nan
            ax3.imshow(imgtmp2, interpolation='None',
                       alpha=0.5, cmap=plt.cm.hot)
            ax3.axis('off')
            print('snr:{0}'.format(vpy.estimates['snr'][i]))
            
    s_comp.on_changed(update)
    s_comp.set_val(0)
    fig.canvas.mpl_connect('key_release_event', arrow_key_image_control)
    plt.show()
    
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
