#!/usr/bin/env python

"""
Demo pipeline for processing voltage imaging data using the
CaImAn batch algorithm. The processing pipeline included motion correction 
and spike finding with given ROIs. 

"""

import cv2
import glob
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.io

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
from caiman.source_extraction.volpy.volpy_function import *  
from caiman.utils.utils import download_demo
from caiman.source_extraction.cnmf import params as params
import os
# %%
# Set up the logger (optional); change this if you like.
# You can log to a file using the filename parameter, or make the output more
# or less verbose by setting level to logging.DEBUG, logging.INFO,
# logging.WARNING, or logging.ERROR

logging.basicConfig(format=
                    "%(relativeCreated)12d [%(filename)s:%(funcName)20s():%(lineno)s]"\
                    "[%(process)d] %(message)s",
                    level=logging.INFO)

# %%
def main():
    pass  # For compatibility between running under Spyder and the CLI
    
# %% Set up directory for tif files, dividing files into several file_list
    dr = '/home/nel/Code/Voltage_imaging/exampledata/403106_3min_raw/raw_data/'
    num_files_total = len([f for f in os.listdir(dr) if f.endswith('.tif')])

    n_blocks = 1
    n_files = np.int(np.floor(num_files_total / n_blocks))

    for n in range(n_blocks):
        file_list = [dr + 'cameraTube051_{:05n}.tif'.format(i + 1) for i in
                     np.arange(n_files * n, n_files * (n + 1), 1)]
# %% Select file(s) to be processed
    fnames = [tuple([ff for ff in file_list[:4000]])]

# %% First setup some parameters for data and motion correction

    # dataset dependent parameters
    fr = 30             # imaging rate in frames per second
    decay_time = 0.4    # length of a typical transient in seconds
    dxy = (1., 1.)      # spatial resolution in x and y in (um per pixel)
    # note the lower than usual spatial resolution here
    max_shift_um = (10., 10.)       # maximum shift in um
    patch_motion_um = (64., 64.)  # patch size for non-rigid correction in um

    # motion correction parameters
    pw_rigid = False       # flag to select rigid vs pw_rigid motion correction
    # maximum allowed rigid shift in pixels
    max_shifts = [int(a/b) for a, b in zip(max_shift_um, dxy)]
    # start a new patch for pw-rigid motion correction every x pixels
    strides = tuple([int(a/b) for a, b in zip(patch_motion_um, dxy)])
    # overlap between pathes (size of patch in pixels: strides+overlaps)
    overlaps = (24, 24)
    # maximum deviation allowed for patch with respect to rigid shifts
    max_deviation_rigid = 3

    mc_dict = {
        'fnames': fnames,
        'fr': fr,
        'decay_time': decay_time,
        'dxy': dxy,
        'pw_rigid': pw_rigid,
        'max_shifts': max_shifts,
        'strides': strides,
        'overlaps': overlaps,
        'max_deviation_rigid': max_deviation_rigid,
        'border_nan': 'copy'
    }

    opts = params.CNMFParams(params_dict=mc_dict)

# %% play the movie (optional)
    # playing the movie using opencv. It requires loading the movie in memory.
    # To close the video press q
    display_images = True

    if display_images:
        m_orig = cm.load(list(fnames[0]))
        ds_ratio = 0.2
        moviehandle = m_orig.resize(1, 1, ds_ratio)
        moviehandle.play(q_max=99.5, fr=60, magnification=2)

# %% start a cluster for parallel processing
    c, dview, n_processes = cm.cluster.setup_cluster(
        backend='local', n_processes=None, single_thread=False)

# %%% MOTION CORRECTION
    # first we create a motion correction object with the specified parameters
    mc = MotionCorrect(fnames, dview=dview, **opts.get_group('motion'))
    # note that the file is not loaded in memory

# %% Run (piecewise-rigid motion) correction using NoRMCorre
    mc.motion_correct(save_movie=True)

#%% compare with original movie
    if display_images:
        m_orig = cm.load(list(fnames[0]))
        m_els = cm.load(mc.mmap_file)
        ds_ratio = 0.2
        moviehandle = cm.concatenate([m_orig.resize(1, 1, ds_ratio) - mc.min_mov*mc.nonneg_movie,
                                      m_els.resize(1, 1, ds_ratio)], axis=2)
        moviehandle.play(fr=60, q_max=99.5, magnification=2)  # press q to exit
        
        m_orig2 = m_orig - np.mean(m_orig, axis=0)
        m_els2 = m_els - np.mean(m_els, axis=0)
        moviehandle1 = cm.concatenate([m_orig2.resize(1, 1, ds_ratio) - mc.min_mov*mc.nonneg_movie,
                                      m_els2.resize(1, 1, ds_ratio)- mc.min_mov*mc.nonneg_movie], axis=2)
        moviehandle1.play(fr=60, q_max=99.5, magnification=2)     # subtract the background

#%% file name, index of cell, ROI, sample rate into args
    fname_new = mc.mmap_file[0]         # memory map file name    
    rois_path = '/home/nel/Code/Voltage_imaging/exampledata/ROIs/403106_3min_rois.mat'
    f = scipy.io.loadmat(rois_path)
    ROIs = f['roi'].T                   # ROIs that are given
    sampleRate = 400                    # sample rate of the video
    n_cells = 3                         # number of cells to process
    
    args = []
    for i in range(n_cells):
        args.append([fname_new, i, ROIs[i, :, :], sampleRate])
        
#%% restart cluster to clean up memory
    cm.stop_server(dview=dview)
    c, dview, n_processes = cm.cluster.setup_cluster(
        backend='local', n_processes=3, single_thread=False)
    
#%% process cells using volspike function
    output = dview.map_async(volspike, args).get()   
    
    
#%% some visualization
    n = 1                              # cell you are interested in
    %matplotlib auto
    
    plt.figure()
    plt.plot(output[n]['yFilt'])
    plt.plot(output[n]['spikeTimes'], np.max(output[n]['yFilt'])*1*np.ones(output[n]['spikeTimes'].shape), color='g', marker='o', fillstyle='none', linestyle='none')
    plt.title('signal and spike times')
    plt.show()
    
    plt.figure()
    plt.imshow(output[n]['spatialFilter'])
    plt.colorbar()
    plt.title('spatial filter')
    plt.show()    

#%% STOP CLUSTER and clean up log files
    cm.stop_server(dview=dview)
    log_files = glob.glob('*_LOG_*')
    for log_file in log_files:
        os.remove(log_file)

# %%
# This is to mask the differences between running this demo in Spyder
# versus from the CLI
if __name__ == "__main__":
    main()
