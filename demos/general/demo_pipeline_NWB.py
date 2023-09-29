#!/usr/bin/env python

"""
This script follows closely the demo_pipeline.py script but uses the
Neurodata Without Borders (NWB) file format for loading the input and saving
the output. It is meant as an example on how to use NWB files with CaImAn.
authors: @agiovann and @epnev
"""
#%%
import cv2
import glob
from IPython import get_ipython
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
        ipython = get_ipython()
        ipython.run_line_magic('load_ext', 'autoreload')
        ipython.run_line_magic('autoreload', '2')
        ipython.run_line_magic('matplotlib', 'qt')
except NameError:
    pass


from datetime import datetime
from dateutil.tz import tzlocal

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
    pass  # For compatibility between running under an IDE and the CLI

    #%% Select file(s) to be processed (download if not present)
    fnames = [os.path.join(caiman_datadir(), 'example_movies/Sue_2x_3000_40_-46.nwb')]
    # estimates save path can be same or different from raw data path
    save_path = os.path.join(caiman_datadir(), 'example_movies/Sue_2x_3000_40_-46_CNMF_estimates.nwb')
        # filename to be created or processed
    # dataset dependent parameters
    fr = 15.  # imaging rate in frames per second
    decay_time = 0.4  # length of a typical transient in seconds

    #%% load the file and save it in the NWB format (if it doesn't exist already)
    if not os.path.exists(fnames[0]):
        fnames_orig = 'Sue_2x_3000_40_-46.tif'  # filename to be processed
        if fnames_orig in ['Sue_2x_3000_40_-46.tif', 'demoMovie.tif']:
            fnames_orig = [download_demo(fnames_orig)]
        orig_movie = cm.load(fnames_orig, fr=fr)

        # save file in NWB format with various additional info
        orig_movie.save(fnames[0], sess_desc='test', identifier='demo 1',
             imaging_plane_description='single plane',
             emission_lambda=520.0, indicator='GCAMP6f',
             location='parietal cortex',
             experimenter='Sue Ann Koay', lab_name='Tank Lab',
             institution='Princeton U',
             experiment_description='Experiment Description',
             session_id='Session 1',
             var_name_hdf5='TwoPhotonSeries')
        
    #%% First setup some parameters for data and motion correction
    # motion correction parameters
    dxy = (2., 2.)  # spatial resolution in x and y in (um per pixel)
    # note the lower than usual spatial resolution here
    max_shift_um = (12., 12.)  # maximum shift in um
    patch_motion_um = (100., 100.)  # patch size for non-rigid correction in um
    pw_rigid = True       # flag to select rigid vs pw_rigid motion correction
    # maximum allowed rigid shift in pixels
    max_shifts = [int(a/b) for a, b in zip(max_shift_um, dxy)]
    # start a new patch for pw-rigid motion correction every x pixels
    strides = tuple([int(a/b) for a, b in zip(patch_motion_um, dxy)])
    # overlap between patches (size of patch in pixels: strides+overlaps)
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
        'border_nan': 'copy',
        #'var_name_hdf5': 'acquisition/TwoPhotonSeries'
        'var_name_hdf5': 'TwoPhotonSeries'
    }
    opts = params.CNMFParams(params_dict=mc_dict)

    # %% play the movie (optional)
    # playing the movie using opencv. It requires loading the movie in memory.
    # To close the video press q
    display_images = False
    if display_images:
        m_orig = cm.load_movie_chain(fnames, var_name_hdf5=opts.data['var_name_hdf5'])
        ds_ratio = 0.2
        moviehandle = m_orig.resize(1, 1, ds_ratio)
        moviehandle.play(q_max=99.5, fr=60, magnification=2)

    # %% start a cluster for parallel processing
    c, dview, n_processes = cm.cluster.setup_cluster(
        backend='multiprocessing', n_processes=None, single_thread=False)

    # %%% MOTION CORRECTION
    # first we create a motion correction object with the specified parameters
    mc = MotionCorrect(fnames, dview=dview, var_name_hdf5=opts.data['var_name_hdf5'], **opts.get_group('motion'))
    # note that the file is not loaded in memory

    # %% Run (piecewise-rigid motion) correction using NoRMCorre
    mc.motion_correct(save_movie=True)

    # %% compare with original movie
    if display_images:
        m_orig = cm.load_movie_chain(fnames, var_name_hdf5=opts.data['var_name_hdf5'])
        m_els = cm.load(mc.mmap_file)
        ds_ratio = 0.2
        moviehandle = cm.concatenate([m_orig.resize(1, 1, ds_ratio) - mc.min_mov*mc.nonneg_movie,
                                      m_els.resize(1, 1, ds_ratio)], axis=2)
        moviehandle.play(fr=60, q_max=99.5, magnification=2)  # press q to exit

    # %% MEMORY MAPPING
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

    # %% restart cluster to clean up memory
    cm.stop_server(dview=dview)
    c, dview, n_processes = cm.cluster.setup_cluster(
        backend='multiprocessing', n_processes=None, single_thread=False)

    # %%  parameters for source extraction and deconvolution
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
    tsub = 2                     # temporal subsampling during intialization

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

    # %% RUN CNMF ON PATCHES
    # First extract spatial and temporal components on patches and combine them
    # for this step deconvolution is turned off (p=0)

    opts.change_params({'p': 0})
    cnm = cnmf.CNMF(n_processes, params=opts, dview=dview)
    cnm = cnm.fit(images)

    # %% ALTERNATE WAY TO RUN THE PIPELINE AT ONCE (optional)
    
    #   you can also perform the motion correction plus cnmf fitting steps
    #   simultaneously after defining your parameters object using
    #  cnm1 = cnmf.CNMF(n_processes, params=opts, dview=dview)
    #  cnm1.fit_file(motion_correct=True)

    # %% plot contours of found components
    Cn = cm.local_correlations(images, swap_dim=False)
    Cn[np.isnan(Cn)] = 0
    cnm.estimates.plot_contours(img=Cn)
    plt.title('Contour plots of found components')

    #%% save results in a separate file (just for demonstration purposes)
    cnm.estimates.Cn = Cn
    cnm.save(fname_new[:-4]+'hdf5')
    #cm.movie(Cn).save(fname_new[:-5]+'_Cn.tif')

    # %% RE-RUN seeded CNMF on accepted patches to refine and perform deconvolution
    cnm.params.change_params({'p': p})
    cnm2 = cnm.refit(images, dview=dview)

    # %% COMPONENT EVALUATION
    # the components are evaluated in three ways:
    #   a) the shape of each component must be correlated with the data
    #   b) a minimum peak SNR is required over the length of a transient
    #   c) each shape passes a CNN based classifier
    min_SNR = 2  # signal to noise ratio for accepting a component
    rval_thr = 0.85  # space correlation threshold for accepting a component
    cnn_thr = 0.99  # threshold for CNN based classifier
    cnn_lowest = 0.1 # neurons with cnn probability lower than this value are rejected

    cnm2.params.set('quality', {'decay_time': decay_time,
                               'min_SNR': min_SNR,
                               'rval_thr': rval_thr,
                               'use_cnn': True,
                               'min_cnn_thr': cnn_thr,
                               'cnn_lowest': cnn_lowest});
    cnm2.estimates.evaluate_components(images, cnm2.params, dview=dview)

    #%%  Save new estimates object (after adding correlation image)
    cnm2.estimates.Cn = Cn
    cnm2.save(fname_new[:-4] + 'hdf5')

    # %% PLOT COMPONENTS
    cnm2.estimates.plot_contours(img=Cn, idx=cnm2.estimates.idx_components)

    # %% VIEW TRACES (accepted and rejected)
    if display_images:
        cnm2.estimates.view_components(images, img=Cn,
                                      idx=cnm2.estimates.idx_components)
        cnm2.estimates.view_components(images, img=Cn,
                                      idx=cnm2.estimates.idx_components_bad)
    #%% update object with selected components (optional)
    # cnm2.estimates.select_components(use_object=True)

    #%% Extract DF/F values
    cnm2.estimates.detrend_df_f(quantileMin=8, frames_window=250)

    #%% Show final traces
    cnm2.estimates.view_components(img=Cn)

    #%% reconstruct denoised movie (press q to exit)
    if display_images:
        cnm2.estimates.play_movie(images, q_max=99.9, gain_res=2,
                                  magnification=2,
                                  bpx=border_to_0,
                                  include_bck=False)  # background not shown

    #%% STOP CLUSTER and clean up log files
    cm.stop_server(dview=dview)
    log_files = glob.glob('*_LOG_*')
    for log_file in log_files:
        os.remove(log_file)
        
    #%% save the results in the original NWB file
    cnm2.estimates.save_NWB(save_path, imaging_rate=fr, session_start_time=datetime.now(tzlocal()),
                            raw_data_file=fnames[0])

# %%
# This is to mask the differences between running this demo in IDE
# versus from the CLI
if __name__ == "__main__":
    main()
