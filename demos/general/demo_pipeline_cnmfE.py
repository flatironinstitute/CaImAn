#!/usr/bin/env python

"""
Complete demo pipeline for motion correction, source extraction, and deconvolution
of one photon microendoscopic calcium imaging data using the CaImAn package.

Demo is also available as a jupyter notebook (see demo_pipeline_cnmfE.ipynb)
"""

from __future__ import division
from __future__ import print_function


import matplotlib.pyplot as plt
import numpy as np

try:
    if __IPYTHON__:
        print('Detected iPython')
        # this is used for debugging purposes only. allows to reload classes when changed
        get_ipython().magic('load_ext autoreload')
        get_ipython().magic('autoreload 2')
except NameError:
    pass

import caiman as cm
from caiman.source_extraction import cnmf
from caiman.utils.utils import download_demo
from caiman.utils.visualization import inspect_correlation_pnr
from caiman.motion_correction import motion_correct_oneP_rigid, motion_correct_oneP_nonrigid
from caiman.source_extraction.cnmf import params as params
from copy import deepcopy

#%%
def main():
    pass # For compatibility between running under Spyder and the CLI

#%% First setup some parameters

    # dataset dependent parameters
    display_images = False           # Set to true to show movies and images
    fnames = ['data_endoscope.tif']  # filename to be processed
    fr = 10                          # movie frame rate
    decay_time = 0.4                 # length of a typical transient in seconds

    # motion correction parameters
    do_motion_correction_nonrigid = False
    do_motion_correction_rigid = True  # choose motion correction type

    gSig_filt = (3, 3)   # size of filter, in general gSig (see below),
    #                      change this one if algorithm does not work
    max_shifts = (5, 5)  # maximum allowed rigid shift
    strides = (48, 48)   # start a new patch for pw-rigid motion correction every x pixels
    overlaps = (24, 24)  # overlap between pathes (size of patch strides+overlaps)
    # for parallelization split the movies in  num_splits chuncks across time
    # (make sure that length_movie/num_splits_to_process_rig>100)
    splits_rig = 10
    splits_els = 10
    upsample_factor_grid = 4    # upsample factor to avoid smearing when merging patches
    # maximum deviation allowed for patch with respect to rigid shifts
    max_deviation_rigid = 3

#%% start the cluster
    try:
        cm.stop_server(dview=dview)  # stop it if it was running
    except():
        pass

    c, dview, n_processes = cm.cluster.setup_cluster(backend='local',
                                                     n_processes=24,  # number of process to use, if you go out of memory try to reduce this one
                                                     single_thread=False)

#%% download demo file
    fnames = [download_demo(fnames[0])]
    filename_reorder = fnames

#%% MOTION CORRECTION
    if do_motion_correction_nonrigid or do_motion_correction_rigid:
        # do motion correction rigid
        mc = motion_correct_oneP_rigid(fnames,
                                       gSig_filt=gSig_filt,
                                       max_shifts=max_shifts,
                                       dview=dview,
                                       splits_rig=splits_rig,
                                       save_movie=not(do_motion_correction_nonrigid),
                                       border_nan='copy'
                                       )
    
        new_templ = mc.total_template_rig

        plt.subplot(1, 2, 1); plt.imshow(new_templ)  # % plot template
        plt.subplot(1, 2, 2); plt.plot(mc.shifts_rig)  # % plot rigid shifts
        plt.legend(['x shifts', 'y shifts'])
        plt.xlabel('frames'); plt.ylabel('pixels')

        # borders to eliminate from movie because of motion correction
        bord_px = np.ceil(np.max(np.abs(mc.shifts_rig))).astype(np.int)
        filename_reorder = mc.fname_tot_rig

        # do motion correction nonrigid
        if do_motion_correction_nonrigid:
            mc = motion_correct_oneP_nonrigid(
                fnames,
                gSig_filt=gSig_filt,
                max_shifts=max_shifts,
                strides=strides,
                overlaps=overlaps,
                splits_els=splits_els,
                upsample_factor_grid=upsample_factor_grid,
                max_deviation_rigid=max_deviation_rigid,
                dview=dview,
                splits_rig=None,
                save_movie=True,     # whether to save movie in memory mapped format
                new_templ=new_templ,  # template to initialize motion correction
                border_nan='copy'
            )

            filename_reorder = mc.fname_tot_els
            bord_px = np.ceil(
                np.maximum(np.max(np.abs(mc.x_shifts_els)),
                           np.max(np.abs(mc.y_shifts_els)))).astype(np.int)

    # create memory mappable file in the right order on the hard drive (C order)
    fname_new = cm.save_memmap(
        filename_reorder,
        base_name='memmap_',
        order='C',
        border_to_0=bord_px,
        dview=dview)

    # load memory mappable file
    Yr, dims, T = cm.load_memmap(fname_new)
    Y = Yr.T.reshape((T,) + dims, order='F')

#%% parameters for source extraction and deconvolution
    
    p = 1               # order of the autoregressive system
    K = None            # upper bound on number of components per patch, in general None
    gSig = (3, 3)       # gaussian width of a 2D gaussian kernel, which approximates a neuron
    gSiz = (13, 13)     # average diameter of a neuron, in general 4*gSig+1
    Ain = None          # possibility to seed with predetermined binary masks
    merge_thresh = .7   # merging threshold, max correlation allowed
    rf = 40             # half-size of the patches in pixels. e.g., if rf=40, patches are 80x80
    stride_cnmf = 20    # amount of overlap between the patches in pixels
    #                     (keep it at least large as gSiz, i.e 4 times the neuron size gSig)
    tsub = 2            # downsampling factor in time for initialization,
    #                     increase if you have memory problems
    ssub = 1            # downsampling factor in space for initialization,
    #                     increase if you have memory problems
    #                     you can pass them here as boolean vectors
    low_rank_background = None  # None leaves background of each patch intact,
    #                             True performs global low-rank approximation 
    gnb = -1            # number of background components (rank) if positive,
    #                     else exact ring model with following settings
    #                         gnb= 0: Return background as b and W
    #                         gnb=-1: Return full rank background B
    #                         gnb<-1: Don't return background
    nb_patch = -1       # number of background components (rank) per patch,
    #                     use 0 or -1 for exact background of ring model (cf. gnb)
    min_corr = .8       # min peak value from correlation image
    min_pnr = 10        # min peak to noise ration from PNR image
    ssub_B = 2          # additional downsampling factor in space for background
    ring_size_factor = 1.4  # radius of ring is gSiz*ring_size_factor
    
    # parameters for component evaluation
    min_SNR = 3            # adaptive way to set threshold on the transient size
    r_values_min = 0.85    # threshold on space consistency (if you lower more components
    #                        will be accepted, potentially with worst quality)
    
    opts = params.CNMFParams(dims=dims, fr=fr, decay_time=decay_time,
                             method_init='corr_pnr',             # use this for 1 photon
                             k=K,
                             gSig=gSig,
                             gSiz=gSiz,
                             merge_thresh=merge_thresh,
                             p=p,
                             tsub=tsub,
                             ssub=ssub,
                             rf=rf,
                             stride=stride_cnmf,
                             only_init_patch=True,    # set it to True to run CNMF-E
                             gnb=gnb,
                             nb_patch=nb_patch,
                             method_deconvolution='oasis',       # could use 'cvxpy' alternatively
                             low_rank_background=low_rank_background,
                             update_background_components=True,  # sometimes setting to False improve the results
                             min_corr=min_corr,
                             min_pnr=min_pnr,
                             normalize_init=False,               # just leave as is
                             center_psf=True,                    # leave as is for 1 photon
                             ssub_B=ssub_B,
                             ring_size_factor=ring_size_factor,
                             del_duplicates=True,                # whether to remove duplicates from initialization
                             border_pix=bord_px)                 # number of pixels to not consider in the borders)

#%% compute some summary images (correlation and peak to noise)
    # change swap dim if output looks weird, it is a problem with tiffile
    cn_filter, pnr = cm.summary_images.correlation_pnr(Y, gSig=gSig[0], swap_dim=False)
    # inspect the summary images and set the parameters
    inspect_correlation_pnr(cn_filter, pnr)
    # print parameters set above, modify them if necessary based on summary images
    print(min_corr) # min correlation of peak (from correlation image)
    print(min_pnr)  # min peak to noise ratio

#%% RUN CNMF ON PATCHES
    cnm = cnmf.CNMF(n_processes=n_processes, dview=dview, Ain=Ain, params=opts)
    cnm.fit(Y)

#%% DISCARD LOW QUALITY COMPONENTS
    cnm.params.set('quality', {'min_SNR': min_SNR,
                               'rval_thr': r_values_min,
                               'use_cnn': False})
    cnm.evaluate_components(Y)

    print(' ***** ')
    print('Number of total components: ', len(cnm.estimates.C))
    print('Number of accepted components: ', len(cnm.estimates.idx_components))

#%% PLOT COMPONENTS
    cnm.dims = dims
    if display_images:
        cnm.plot_contours(img=cn_filter, idx=cnm.estimates.idx_components)
        cnm.view_components(Y, idx=cnm.estimates.idx_components)

#%% MOVIES
    if display_images:
        # fully reconstructed movie
        cnm.play_movie(Y, magnification=3, include_bck=True, gain_res=10)
        # movie without background
        cnm.play_movie(Y, magnification=3, include_bck=False, gain_res=4)

#%% STOP SERVER
    cm.stop_server(dview=dview)
# This is to mask the differences between running this demo in Spyder
# versus from the CLI
if __name__ == "__main__":
    main()
