#!/usr/bin/env python

"""
Complete demo pipeline for motion correction, source extraction, and
deconvolution of two-photon calcium imaging data using the CaImAn package.

Demo is also available as a jupyter notebook (see demo_pipeline.ipynb)
Dataset couresy of Sue Ann Koay and David Tank (Princeton University)

This demo pertains to two photon data. For a complete analysis pipeline for
one photon microendoscopic data see demo_pipeline_cnmfE.py

copyright GNU General Public License v2.0
authors: @agiovann and @epnev
"""

from __future__ import division
from __future__ import print_function

import os
import cv2
import glob

try:
    cv2.setNumThreads(0)
except():
    pass

try:
    if __IPYTHON__:
        print("Running under iPython")
        # this is used for debugging purposes only. allows to reload classes
        # when changed
        get_ipython().magic('load_ext autoreload')
        get_ipython().magic('autoreload 2')
except NameError:
    pass

import matplotlib.pyplot as plt
import numpy as np

import caiman as cm
from caiman.utils.utils import download_demo
from caiman.source_extraction.cnmf import cnmf as cnmf
from caiman.motion_correction import MotionCorrect
from caiman.source_extraction.cnmf import params as params
from copy import deepcopy

# %%
def main():
    pass  # For compatibility between running under Spyder and the CLI

# %% Select file(s) to be processed (download if not present)
    fname = ['Sue_2x_3000_40_-46.tif']  # filename to be processed
    if fname[0] in ['Sue_2x_3000_40_-46.tif', 'demoMovie.tif']:
        fname = [download_demo(fname[0])]

# %% First setup some parameters for data and motion correction

    # dataset dependent parameters
    fr = 30             # imaging rate in frames per second
    decay_time = 0.4    # length of a typical transient in seconds
    dxy = (2., 2.)      # spatial resolution in x and y in (um per pixel)
    max_shift_um = (12., 12.)           # maximum shift in um
    patch_motion_um = (100., 100.)  # patch size for non-rigid correction in um

    # motion correction parameters
    pw_rigid = True       # flag to select rigid vs pw_rigid motion correction
    # maximum allow rigid shift in pixels
    max_shifts = tuple([int(a/b) for a, b in zip(max_shift_um, dxy)])
    # for parallelization split the movies in  num_splits chuncks across time
    splits_rig = 56
    # start a new patch for pw-rigid motion correction every x pixels
    strides = tuple([int(a/b) for a, b in zip(patch_motion_um, dxy)])
    # overlap between pathes (size of patch in pixels: strides+overlaps)
    overlaps = (24, 24)
    # for parallelization split the movies in  num_splits chuncks across time
    splits_els = 56
    # maximum deviation allowed for patch with respect to rigid shifts
    max_deviation_rigid = 3

# %% play the movie
    # playing the movie using opencv. It requires loading the movie in memory.
    # To close the video press q
    display_images = False

    if display_images:
        m_orig = cm.load_movie_chain(fname)
        downsample_ratio = 0.2
        moviehandle = m_orig.resize(1, 1, downsample_ratio)
        moviehandle.play(q_max=99.5, fr=60, magnification=2)

# %% start a cluster for parallel processing
    c, dview, n_processes = cm.cluster.setup_cluster(
        backend='local', n_processes=None, single_thread=False)

# %%% MOTION CORRECTION
    # first we create a motion correction object with the specified parameters

    mc = MotionCorrect(fname, dview=dview, max_shifts=max_shifts,
                       splits_rig=splits_rig, splits_els=splits_els,
                       strides=strides, overlaps=overlaps, border_nan='copy',
                       max_deviation_rigid=max_deviation_rigid,
                       shifts_opencv=True, nonneg_movie=True, pw_rigid=pw_rigid)
    # note that the file is not loaded in memory

# %% Run (piecewise-rigid motion) correction using NoRMCorre
    mc.motion_correct(save_movie=True)

# %% compare with original movie
    if display_images:
        m_orig = cm.load_movie_chain(fname)
        m_els = cm.load(mc.mmap_file)
        downsample_ratio = 0.2
        moviehandle = cm.concatenate([m_orig.resize(1, 1, downsample_ratio) - mc.min_mov,
                                      m_els.resize(1, 1, downsample_ratio)], axis=2)
        moviehandle.play(fr=60, q_max=99.5, magnification=2)  # press q to exit

# %% MEMORY MAPPING
    border_to_0 = 0 if mc.border_nan is 'copy' else mc.border_to_0
    # you can include boundaries if you used the 'copy' option in the motion
    # correction, although be careful abou the components near the boundaries
    # memory map the file in order 'C'
    fname_new = cm.save_memmap(mc.mmap_file, base_name='memmap_', order='C',
                               border_to_0=border_to_0)  # exclude borders

    # now load the file
    Yr, dims, T = cm.load_memmap(fname_new)
    images = np.reshape(Yr.T, [T] + list(dims), order='F')
    # load frames in python format (T x X x Y)

# %% restart cluster to clean up memory
    cm.stop_server(dview=dview)
    c, dview, n_processes = cm.cluster.setup_cluster(
        backend='local', n_processes=None, single_thread=False)

# %%  parameters for source extraction and deconvolution
    p = 1                    # order of the autoregressive system
    gnb = 2                  # number of global background components
    merge_thresh = 0.8       # merging threshold, max correlation allowed
    # half-size of the patches in pixels. e.g., if rf=25, patches are 50x50
    rf = 15
    stride_cnmf = 6          # amount of overlap between the patches in pixels
    K = 4                    # number of components per patch
    gSig = [4, 4]            # expected half size of neurons
    # initialization method (if analyzing dendritic data using 'sparse_nmf')
    method_init = 'greedy_roi'

    # parameters for component evaluation
    opts_dict = {'fnames': fname,
                 'fr': fr,
                 'nb': gnb,
                 'rf': rf,
                 'gSig': gSig,
                 'K': K,
                 'stride': stride_cnmf,
                 'method_init': method_init,
                 'rolling_sum': True,
                 'merge_thr': merge_thresh,
                 'n_processes': n_processes}

    opts = params.CNMFParams(params_dict=opts_dict)
# %% RUN CNMF ON PATCHES
    # First extract spatial and temporal components on patches and combine them
    # for this step deconvolution is turned off (p=0)

    opts.set('temporal', {'p': 0})
    cnm = cnmf.CNMF(n_processes, params=opts, dview=dview)
    cnm = cnm.fit(images)

# %% plot contours of found components
    Cn = cm.local_correlations(images.transpose(1, 2, 0))
    Cn[np.isnan(Cn)] = 0
    cnm.estimates.plot_contours(img=Cn)
    plt.title('Contour plots of found components')

# %% COMPONENT EVALUATION
    # the components are evaluated in three ways:
    #   a) the shape of each component must be correlated with the data
    #   b) a minimum peak SNR is required over the length of a transient
    #   c) each shape passes a CNN based classifier
    min_SNR = 2.5       # signal to noise ratio for accepting a component
    rval_thr = 0.8      # space correlation threshold for accepting a component
    cnn_thr = 0.8       # threshold for CNN based classifier
    cnm.params.set('quality', {'decay_time': decay_time,
                               'min_SNR': min_SNR,
                               'rval_thr': rval_thr,
                               'use_cnn': True,
                               'min_cnn_thr': cnn_thr})
    cnm.estimates.evaluate_components(images, cnm.params, dview=dview)

# %% PLOT COMPONENTS
    cnm.estimates.plot_contours(img=Cn, idx=cnm.estimates.idx_components)

# %% VIEW TRACES (accepted and rejected)

    if display_images:
        cnm.estimates.view_components(images, img=Cn,
                                      idx=cnm.estimates.idx_components)
        cnm.estimates.view_components(images, img=Cn,
                                      idx=cnm.estimates.idx_components_bad)

# %% RE-RUN seeded CNMF on accepted patches to refine and perform deconvolution

#    cnm.dview = None
#    cnm2 = deepcopy(cnm)
#    cnm2.estimates.select_components(use_object=True)
#    cnm2.dview = dview
#    cnm2.params.set('patch', {'rf': None})
#    cnm2.params.set('temporal', {'p': p})
#    cnm2 = cnm2.fit(images)
    cnm2 = cnm.refit(images)

# %% Extract DF/F values
    cnm2.estimates.detrend_df_f(quantileMin=8, frames_window=250)

# %% Show final traces
    cnm2.estimates.view_components(Yr, img=Cn)

# %% reconstruct denoised movie (press q to exit)
    if display_images:
        cnm2.estimates.play_movie(images, q_max=99.9, gain_res=2,
                                  magnification=2,
                                  bpx=border_to_0,
                                  include_bck=True)

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
