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
from builtins import range

import os
import cv2
import glob

try:
    cv2.setNumThreads(0)
except:
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
import time

import caiman as cm
from caiman.utils.utils import download_demo
from caiman.source_extraction.cnmf import cnmf as cnmf
from caiman.motion_correction import MotionCorrect

#%%
def main():
    pass # For compatibility between running under Spyder and the CLI

#%% First setup some parameters

    # dataset dependent parameters
    display_images = False              # Set this to true to show movies and plots
    fname = ['Sue_2x_3000_40_-46.tif']  # filename to be processed
    fr = 30                             # imaging rate in frames per second
    decay_time = 0.4                    # length of a typical transient in seconds

    # motion correction parameters
    niter_rig = 1               # number of iterations for rigid motion correction
    max_shifts = (6, 6)         # maximum allow rigid shift
    # for parallelization split the movies in  num_splits chuncks across time
    splits_rig = 56
    # start a new patch for pw-rigid motion correction every x pixels
    strides = (48, 48)
    # overlap between pathes (size of patch strides+overlaps)
    overlaps = (24, 24)
    # for parallelization split the movies in  num_splits chuncks across time
    splits_els = 56
    upsample_factor_grid = 4    # upsample factor to avoid smearing when merging patches
    # maximum deviation allowed for patch with respect to rigid shifts
    max_deviation_rigid = 3

    # parameters for source extraction and deconvolution
    p = 1                       # order of the autoregressive system
    gnb = 2                     # number of global background components
    merge_thresh = 0.8          # merging threshold, max correlation allowed
    # half-size of the patches in pixels. e.g., if rf=25, patches are 50x50
    rf = 15
    stride_cnmf = 6             # amount of overlap between the patches in pixels
    K = 4                       # number of components per patch
    gSig = [4, 4]               # expected half size of neurons
    # initialization method (if analyzing dendritic data using 'sparse_nmf')
    init_method = 'greedy_roi'
    # sparsity penalty for dendritic data analysis through sparse NMF
    alpha_snmf = None

    # parameters for component evaluation
    min_SNR = 2.5               # signal to noise ratio for accepting a component
    rval_thr = 0.8              # space correlation threshold for accepting a component
    cnn_thr = 0.8               # threshold for CNN based classifier

#%% download the dataset if it's not present in your folder
    if fname[0] in ['Sue_2x_3000_40_-46.tif', 'demoMovie.tif']:
        fname = [download_demo(fname[0])]

#%% play the movie
    # playing the movie using opencv. It requires loading the movie in memory. To
    # close the video press q

    m_orig = cm.load_movie_chain(fname[:1])
    downsample_ratio = 0.2
    offset_mov = -np.min(m_orig[:100])
    moviehandle = m_orig.resize(1, 1, downsample_ratio)
    if display_images:
        moviehandle.play(gain=10, offset=offset_mov, fr=30, magnification=2)

#%% start a cluster for parallel processing
    c, dview, n_processes = cm.cluster.setup_cluster(
        backend='local', n_processes=None, single_thread=False)


#%%% MOTION CORRECTION
    # first we create a motion correction object with the parameters specified
    min_mov = cm.load(fname[0], subindices=range(200)).min()
    # this will be subtracted from the movie to make it non-negative

    mc = MotionCorrect(fname[0], min_mov, dview=dview, max_shifts=max_shifts, 
                       niter_rig=niter_rig, splits_rig=splits_rig,
                       strides=strides, overlaps=overlaps, 
                       splits_els=splits_els, border_nan='copy',
                       upsample_factor_grid=upsample_factor_grid,
                       max_deviation_rigid=max_deviation_rigid,
                       shifts_opencv=True, nonneg_movie=True)
    # note that the file is not loaded in memory

#%% Run piecewise-rigid motion correction using NoRMCorre
    mc.motion_correct_pwrigid(save_movie=True)
    m_els = cm.load(mc.fname_tot_els)
    bord_px_els = np.ceil(np.maximum(np.max(np.abs(mc.x_shifts_els)),
                                     np.max(np.abs(mc.y_shifts_els)))).astype(np.int)
    # maximum shift to be used for trimming against NaNs
#%% compare with original movie
    moviehandle = cm.concatenate([m_orig.resize(1, 1, downsample_ratio) + offset_mov,
                    m_els.resize(1, 1, downsample_ratio)],
                   axis=2)
    display_images = False
    if display_images:
        moviehandle.play(fr=60, q_max=99.5, magnification=2, offset=0)  # press q to exit

#%% MEMORY MAPPING
    # memory map the file in order 'C'
    fnames = mc.fname_tot_els   # name of the pw-rigidly corrected file.
    fname_new = cm.save_memmap(fnames, base_name='memmap_', order='C',
                               border_to_0=bord_px_els)  # exclude borders

    # now load the file
    Yr, dims, T = cm.load_memmap(fname_new)
    d1, d2 = dims
    images = np.reshape(Yr.T, [T] + list(dims), order='F')
    # load frames in python format (T x X x Y)

#%% restart cluster to clean up memory
    cm.stop_server(dview=dview)
    c, dview, n_processes = cm.cluster.setup_cluster(
        backend='local', n_processes=None, single_thread=False)

#%% RUN CNMF ON PATCHES

    # First extract spatial and temporal components on patches and combine them
    # for this step deconvolution is turned off (p=0)

    cnm = cnmf.CNMF(n_processes=1, k=K, gSig=gSig, merge_thresh=merge_thresh,
                    p=0, dview=dview, rf=rf, stride=stride_cnmf, memory_fact=1,
                    method_init=init_method, alpha_snmf=alpha_snmf,
                    only_init_patch=False, gnb=gnb, border_pix=bord_px_els)
    cnm = cnm.fit(images)
    cnm.dims = dims
#%% plot contours of found components
    Cn = cm.local_correlations(images.transpose(1, 2, 0))
    Cn[np.isnan(Cn)] = 0
    cnm.plot_contours(img=Cn)
    plt.title('Contour plots of found components')

#%% COMPONENT EVALUATION
    # the components are evaluated in three ways:
    #   a) the shape of each component must be correlated with the data
    #   b) a minimum peak SNR is required over the length of a transient
    #   c) each shape passes a CNN based classifier
    cnm.evaluate_components(images, fr=fr, decay_time=decay_time,
                            min_SNR=min_SNR, rval_thr=rval_thr,
                            use_cnn=False, min_cnn_thr=cnn_thr)

#%% PLOT COMPONENTS
    cnm.plot_contours(img=Cn, idx=cnm.idx_components)

#%% VIEW TRACES (accepted and rejected)

    if display_images:
        cnm.view_components(images, dims, img=Cn, idx=cnm.idx_components)
        cnm.view_components(images, dims, img=Cn, idx=cnm.idx_components_bad)

#%% RE-RUN seeded CNMF on accepted patches to refine and perform deconvolution
    A_in, C_in, b_in, f_in = cnm.A[:, cnm.idx_components],\
        cnm.C[cnm.idx_components], cnm.b, cnm.f
    cnm2 = cnmf.CNMF(n_processes=1, k=A_in.shape[-1], gSig=gSig, p=p, dview=dview,
                     merge_thresh=merge_thresh, Ain=A_in, Cin=C_in, b_in=b_in,
                     f_in=f_in, rf=None, stride=None, gnb=gnb,
                     method_deconvolution='oasis', check_nan=True)

    cnm2 = cnm2.fit(images)

#%% Extract DF/F values
    cnm2.detrend_df_f(quantileMin=8, frames_window=250)

#%% Show final traces
    cnm2.view_components(Yr, dims=dims, img=Cn)

#%% reconstruct denoised movie (press q to exit)
    if display_images:
        cnm2.play_movie(images, gain_res=2, magnification=2)

#%% STOP CLUSTER and clean up log files
    cm.stop_server(dview=dview)
    log_files = glob.glob('*_LOG_*')
    for log_file in log_files:
        os.remove(log_file)

#%%
# This is to mask the differences between running this demo in Spyder
# versus from the CLI
if __name__ == "__main__":
    main()
