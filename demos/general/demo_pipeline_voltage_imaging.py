#!/usr/bin/env python

"""
Demo pipeline for processing voltage imaging data. The processing pipeline
includes motion correction and spike detection with given ROIs.

"""
import os
import cv2
import glob
import logging
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import sys

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
from caiman.utils.utils import download_demo
from caiman.source_extraction.volpy.Volparams import volparams
from caiman.source_extraction.volpy.volpy import VOLPY
import matplotlib.pyplot as plt

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

    # %% Please download the .npz file manually to your file path. 
    url = 'https://www.dropbox.com/s/tuv1bx9w6wdywnm/demo_voltage_imaging.npz?dl=0'  
    
    # %% Load demo movie and ROIs
    file_path = '/home/nel/Code/Voltage_imaging/exampledata/403106_3min/demo_voltage_imaging.npz'
    m = np.load(file_path)
    mv = cm.movie(m.f.arr_0)
    ROIs = m.f.arr_1
    
    # %% Save the movie  
    fnames = '/home/nel/Code/Voltage_imaging/exampledata/403106_3min/demomovie_voltage_imaging.hdf5'
    mv.save(fnames)                                 # neglect the warning

    # %% Setup some parameters for data and motion correction
    # dataset parameters
    fr = 400                                        # sample rate of the movie
    index = list(range(ROIs.shape[0]))              # index of neurons
    weights = None                                  # reuse spatial weights by 
                                                    # opts.change_params(params_dict={'weights':vpy.estimates['weights']})
    # motion correction parameters
    motion_correct = True                           # flag for motion correction
    pw_rigid = True                                 # flag for pw-rigid motion correction
    gSig_filt = (3, 3)                              # size of filter, in general gSig (see below),
                                                    # change this one if algorithm does not work
    max_shifts = (5, 5)                             # maximum allowed rigid shift
    strides = (48, 48)                              # start a new patch for pw-rigid motion correction every x pixels
    overlaps = (24, 24)                             # overlap between pathes (size of patch strides+overlaps)
    max_deviation_rigid = 3                         # maximum deviation allowed for patch with respect to rigid shifts
    border_nan = 'copy'

    mc_dict = {
        'fnames': fnames,
        'fr': fr,
        'index': index,
        'ROIs':ROIs,
        'weights':weights,              
        'pw_rigid': pw_rigid,
        'max_shifts': max_shifts,
        'gSig_filt': gSig_filt,
        'strides': strides,
        'overlaps': overlaps,
        'max_deviation_rigid': max_deviation_rigid,
        'border_nan': border_nan
    }

    opts = volparams(params_dict=mc_dict)

    # %% play the movie (optional)
    # playing the movie using opencv. It requires loading the movie in memory.
    # To close the video press q
    display_images = True

    if display_images:
        m_orig = cm.load(fnames)
        ds_ratio = 0.2
        moviehandle = m_orig.resize(1, 1, ds_ratio)
        moviehandle.play(q_max=99.5, fr=60, magnification=2)

    # %% start a cluster for parallel processing
    c, dview, n_processes = cm.cluster.setup_cluster(
        backend='local', n_processes=12, single_thread=False)

    # %%% MOTION CORRECTION
    # Create a motion correction object with the specified parameters
    mc_pw = MotionCorrect(fnames, dview=dview, **opts.get_group('motion'))
    # Run piecewise rigid motion correction 
    mc_pw.motion_correct(save_movie=True)

    # %% motion correction compared with original movie
    display_images = True

    if display_images:
        m_orig = cm.load(fnames)
        m_rig = cm.load(mc_pw.mmap_file)
        ds_ratio = 0.2
        moviehandle = cm.concatenate([m_orig.resize(1, 1, ds_ratio) - mc_pw.min_mov * mc_pw.nonneg_movie,
                                      m_rig.resize(1, 1, ds_ratio)], axis=2)
        moviehandle.play(fr=60, q_max=99.5, magnification=2)  # press q to exit

    # %% movie subtracted from the mean
        m_orig2 = (m_orig - np.mean(m_orig, axis=0))
        m_rig2 = (m_rig - np.mean(m_rig, axis=0))
        moviehandle1 = cm.concatenate([m_orig2.resize(1, 1, ds_ratio),
                                       m_rig2.resize(1, 1, ds_ratio)], axis=2)
        moviehandle1.play(fr=60, q_max=99.5, magnification=2) 

   # %% change fnames to the new motion corrected one
    fname_new = mc_pw.mmap_file[0]  # memory map file name
    opts.change_params(params_dict={'fnames':fname_new})

    # %% restart cluster to clean up memory
    cm.stop_server(dview=dview)
    c, dview, n_processes = cm.cluster.setup_cluster(
        backend='local', n_processes=12, single_thread=False)

    # %% process cells using volspike function
    vpy = VOLPY(n_processes=n_processes, dview=dview, params=opts)
    vpy.fit()

    # %% some visualization
    vpy.estimates['cellN']
    n = 0  # cell you are interested in

    plt.figure()
    plt.plot(vpy.estimates['trace'][n])
    plt.plot(vpy.estimates['spikeTimes'][n], np.max(vpy.estimates['trace'][n]) * 1 * np.ones(vpy.estimates['spikeTimes'][n].shape),
             color='g', marker='o', fillstyle='none', linestyle='none')
    plt.title('signal and spike times')
    plt.show()

    plt.figure()
    plt.imshow(vpy.estimates['spatialFilter'][n])
    plt.colorbar()
    plt.title('spatial filter')
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
