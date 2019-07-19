#!/usr/bin/env python

"""
This script follows closely the demo_pipeline_nwb.py script but for a 3d 
dataset and only through the motion correction process.
"""

import cv2
import glob
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
from caiman.source_extraction.cnmf import cnmf as cnmf
from caiman.source_extraction.cnmf import params as params
from caiman.utils.utils import download_demo
from caiman.paths import caiman_datadir

# %%
# Set up the logger (optional); change this if you like.
# You can log to a file using the filename parameter, or make the output more
# or less verbose by setting level to logging.DEBUG, logging.INFO,
# logging.WARNING, or logging.ERROR

logging.basicConfig(format=
                    "%(relativeCreated)12d [%(filename)s:%(funcName)20s():%(lineno)s]"\
                    "[%(process)d] %(message)s",
                    level=logging.WARNING)

#%%
def main():
    pass  # For compatibility between running under Spyder and the CLI

#%% Select file(s) to be processed (download if not present)
    fnames = [os.path.join(caiman_datadir(), 'example_movies/sampled3dMovie.nwb')]  
        # filename to be created or processed
    # dataset dependent parameters
    fr = 5  # imaging rate in frames per second
    decay_time = 0.4  # length of a typical transient in seconds

    starting_time = 0.
#%% load the file and save it in the NWB format (if it doesn't exist already)
    if not os.path.exists(fnames[0]):
        fnames_orig = [os.path.join(caiman_datadir(), 'example_movies/sampled3dMovie.h5')]  # filename to be processed
        orig_movie = cm.load(fnames_orig, fr=fr, is3D=True)
 #       orig_movie = cm.load_movie_chain(fnames_orig,fr=fr,is3D=True)
        
        # save file in NWB format with various additional info
        orig_movie.save(fnames[0], sess_desc='test', identifier='demo 3d',
             exp_desc='demo movie', imaging_plane_description='multi plane',
             emission_lambda=520.0, indicator='none',
             location='visual cortex', starting_time=starting_time,
             experimenter='NAOMi', lab_name='Tank Lab',
             institution='Princeton U',
             experiment_description='Experiment Description',
             session_id='Session 1',
             var_name_hdf5='TwoPhotonSeries')
#%% First setup some parameters for data and motion correction


    # motion correction parameters
    dxy = (5., 1., 1.)  # spatial resolution in z, x, and y in (um per pixel)
    # note the lower than usual spatial resolution here
    max_shift_um = (10., 10., 10.)  # maximum shift in um
    patch_motion_um = (30., 100., 100.)  # patch size for non-rigid correction in um
    pw_rigid = True       # flag to select rigid vs pw_rigid motion correction
    # maximum allowed rigid shift in pixels
    max_shifts = [int(a/b) for a, b in zip(max_shift_um, dxy)]
    # start a new patch for pw-rigid motion correction every x pixels
    strides = tuple([int(a/b) for a, b in zip(patch_motion_um, dxy)])
    # overlap between pathes (size of patch in pixels: strides+overlaps)
    overlaps = (4, 24, 24)
    # maximum deviation allowed for patch with respect to rigid shifts
    max_deviation_rigid = 3
    is3D = True
    
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
        'border_nan': 'copy',
        'var_name_hdf5': 'acquisition/TwoPhotonSeries',
        'is3D': is3D,
        'splits_els': 12,
        'splits_rig': 12
    }

    opts = params.CNMFParams(params_dict=mc_dict) #NOTE: default adjustments of parameters are not set yet, manually setting them now

# %% play the movie (optional)
    # playing the movie using opencv. It requires loading the movie in memory.
    # To close the video press q
    display_images = False
    if display_images:
        m_orig = cm.load_movie_chain(fnames, var_name_hdf5=opts.data['var_name_hdf5'],is3D=True)
        z, T, h, w = m_orig.shape # Time, plane, height, weight
        m_orig = np.reshape(m_orig, (T*z, h, w))
        ds_ratio = 0.2
        moviehandle = m_orig.resize(1, 1, ds_ratio)
        moviehandle.play(q_max=99.5, fr=60, magnification=2)

# %% start a cluster for parallel processing
# NOTE: ignore dview right now for debugging purposes
#    c, dview, n_processes = cm.cluster.setup_cluster(
#        backend='local', n_processes=None, single_thread=False)

# %%% MOTION CORRECTION
    # first we create a motion correction object with the specified parameters
    mc = MotionCorrect(fnames, dview=None, var_name_hdf5=opts.data['var_name_hdf5'], **opts.get_group('motion'))
#    mc = MotionCorrect(fnames, dview=dview, var_name_hdf5=opts.data['var_name_hdf5'], **opts.get_group('motion'))
    # note that the file is not loaded in memory

# %% Run (piecewise-rigid motion) correction using NoRMCorre
    mc.motion_correct(save_movie=True)

# %% compare with original movie
    if display_images:
        m_orig = cm.load_movie_chain(fnames, var_name_hdf5=opts.data['var_name_hdf5'],is3D=True)
        T, P, h, w = m_orig.shape # Time, plane, height, weight
        m_orig = np.reshape(m_orig, (T*P, h, w))
        m_els = cm.load(mc.mmap_file)
        ds_ratio = 0.2
        moviehandle = cm.concatenate([m_orig.resize(1, 1, ds_ratio) - mc.min_mov*mc.nonneg_movie,
                                      m_els.resize(1, 1, ds_ratio)], axis=2)
        moviehandle.play(fr=60, q_max=99.5, magnification=2)  # press q to exit

# %%
# This is to mask the differences between running this demo in Spyder
# versus from the CLI
if __name__ == "__main__":
    main()