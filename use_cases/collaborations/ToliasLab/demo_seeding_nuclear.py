#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Demo for detecting ROIs in a structural channel and then seeding CNMF with them.
Detection happens through simple adaptive thresholding of the mean image and 
could potentially be improved. Then the structural channel is processed. Both
offline and online approaches are included. 

The offline approach will only use the seeded masks, whereas online will also 
search for new components during the analysis.

The demo assumes that both channels are motion corrected prior to the analysis
although this is not necessary for the case of caiman online. 

@author: epnevmatikakis
"""

try:
    if __IPYTHON__:
        print('Debugging!')
        # this is used for debugging purposes only. allows to reload classes when changed
        get_ipython().magic('load_ext autoreload')
        get_ipython().magic('autoreload 2')
except NameError:
    print('Not launched under iPython')

import numpy as np
import glob
import matplotlib.pyplot as plt
import caiman as cm
from caiman.motion_correction import MotionCorrect
from caiman.components_evaluation import evaluate_components
from caiman.source_extraction import cnmf as cnmf
import os
from caiman.summary_images import max_correlation_image


## %% extract individual channels and motion correct them
#
#fname = '/Users/epnevmatikakis/Documents/Ca_datasets/Tolias/nuclear/gmc_960_30mw_00001.tif'
#fname_green = fname.split('.')[0] + '_green_raw.tif'
#fname_red = fname.split('.')[0] + '_red_raw.tif'
#m = cm.load(fname)
#
#m[::2].save(fname_green)
#m[1::2].save(fname_red)
#
#
## %% motion correction
#
## dataset dependent parameters
#fr = 30             # imaging rate in frames per second
#dxy = (1., 1.)      # spatial resolution in x and y in (um per pixel)
## note the lower than usual spatial resolution here
#max_shift_um = (12., 12.)       # maximum shift in um
#patch_motion_um = (100., 100.)  # patch size for non-rigid correction in um
#
## motion correction parameters
#pw_rigid = True       # flag to select rigid vs pw_rigid motion correction
## maximum allowed rigid shift in pixels
#max_shifts = [int(a/b) for a, b in zip(max_shift_um, dxy)]
## start a new patch for pw-rigid motion correction every x pixels
#strides = tuple([int(a/b) for a, b in zip(patch_motion_um, dxy)])
## overlap between pathes (size of patch in pixels: strides+overlaps)
#overlaps = (24, 24)
## maximum deviation allowed for patch with respect to rigid shifts
#max_deviation_rigid = 3
#
#mc_dict = {
#    'fnames': fname_red,
#    'fr': fr,
#    'dxy': dxy,
#    'pw_rigid': pw_rigid,
#    'max_shifts': max_shifts,
#    'strides': strides,
#    'overlaps': overlaps,
#    'max_deviation_rigid': max_deviation_rigid,
#    'border_nan': 'copy'
#}
#
#opts = params.CNMFParams(params_dict=mc_dict)
#
##%%
#
# c, dview, n_processes = cm.cluster.setup_cluster(
#        backend='local', n_processes=None, single_thread=False)
#
## %%% MOTION CORRECTION
#    # first we create a motion correction object with the specified parameters
#    mc = MotionCorrect(fname_red, dview=dview, **opts.get_group('motion'))
#    # note that the file is not loaded in memory
#
## %% Run (piecewise-rigid motion) correction using NoRMCorre
#    mc.motion_correct(save_movie=True)
#    R = cm.load(mc.mmap_file, in_memory=True)
#    R.save(fname.split('.')[0] + '_red.tif')
#    G = mc.apply_shifts_movie(fname_green)

# %% construct the seeding matrix using the structural channel (note that some
#  components are missed - thresholding can be improved)

fname_red = ('/Users/epnevmatikakis/Documents/Ca_datasets' +
                '/Tolias/nuclear/gmc_960_30mw_00001_red.tif')

Ain, mR = cm.base.rois.extract_binary_masks_from_structural_channel(
    cm.load(fname_red), expand_method='dilation', selem=np.ones((1, 1)))
plt.figure()
crd = cm.utils.visualization.plot_contours(
    Ain.astype('float32'), mR, thr=0.99, display_numbers=False)
plt.title('Contour plots of detected ROIs in the structural channel')

# %% choose whether to use online algorithm (OnACID) or offline (CNMF)
use_online = True

# specify some common parameters

fname_green = ('/Users/epnevmatikakis/Documents/Ca_datasets/Tolias' +
                '/nuclear/gmc_960_30mw_00001_green.tif')

#  %% some common parameters
K = 5  # number of neurons expected per patch (nuisance parameter in this case)
gSig = [7, 7]  # expected half size of neurons
fr = 30
decay_time = 0.5
merge_thresh = 0.8  # merging threshold, max correlation allowed
p = 1  # order of the autoregressive system
gnb = 1  # order of background
min_SNR = 2  # trace SNR threshold
rval_thr = .95

# %%  create and fit online object

if use_online:

    show_movie = True
    init_batch = 200   # use the first initbatch frames to initialize OnACID

    params_dict = {'fnames': fname_green,
                   'fr': fr,
                   'decay_time': decay_time,
                   'gSig': gSig,
                   'p': p,
                   'min_SNR': min_SNR,
                   'rval_thr': rval_thr,
                   'nb': gnb,
                   'motion_correct': False,
                   'init_batch': init_batch,
                   'init_method': 'seeded',
                   'normalize': True,
                   'K': K,
                   'dist_shape_update': True,
                   'show_movie': show_movie}
    opts = cnmf.params.CNMFParams(params_dict=params_dict)

    cnm = cnmf.online_cnmf.OnACID(params=opts)
    cnm.estimates.A = Ain
    cnm.fit_online()

    # %% plot some results

    cnm.estimates.plot_contours()
    cnm.estimates.view_components()
    # view components. Last components are the components added by OnACID

else:  # run offline CNMF algorithm (WIP)
    # %% start cluster
    c, dview, n_processes = cm.cluster.setup_cluster(
        backend='local', n_processes=None, single_thread=False)
    
    # %% 
    
    fname_map = cm.save_memmap([fname_green], base_name='Yr', order='C')
    Yr, dims, T = cm.load_memmap(fname_map)
    images = np.reshape(Yr.T, [T] + list(dims), order='F')
    cnm_b = cnmf.CNMF(n_processes, Ain=Ain, dview=dview, params=opts)
    cnm_b.fit(images)
    #%% FOR LOADING ALL TIFF FILES IN A FILE AND SAVING THEM ON A SINGLE MEMORY MAPPABLE FILE
    
    # can actually be a lost of movie to concatenate
    #fnames = ['example_movies/gmc_980_30mw_00001_green.tif']
    add_to_movie = 0  # the movie must be positive!!!
    downsample_factor = .5  # use .2 or .1 if file is large and you want a quick answer
    base_name = 'Yr'
    name_new = cm.save_memmap_each(fname_green, dview=None, base_name=base_name, resize_fact=(
        1, 1, downsample_factor), add_to_movie=add_to_movie)
    name_new.sort()
    fname_new = cm.save_memmap_join(name_new, base_name='Yr', dview=dview)

    #%% LOAD MEMORY MAPPABLE FILE

    Yr, dims, T = cm.load_memmap(fname_new)
    d1, d2 = dims
    images = np.reshape(Yr.T, [T] + list(dims), order='F')
    Y = np.reshape(Yr, dims + (T,), order='F')
        
    #%% run  seeded CNMF

    cnm = cnmf.CNMF(n_processes, method_init='greedy_roi', k=Ain.shape[1], gSig=gSig, merge_thresh=merge_thresh,
                    p=p, dview=dview, Ain=Ain, method_deconvolution='oasis', rolling_sum=False, rf=None)
    cnm = cnm.fit(images)
    A, C, b, f, YrA, sn = cnm.A, cnm.C, cnm.b, cnm.f, cnm.YrA, cnm.sn

    #%% plot contours of components

    pl.figure()
    crd = cm.utils.visualization.plot_contours(cnm.A, Cn, thr=0.9)
    pl.title('Contour plots against correlation image')

    #%% evaluate the quality of the components
    # a lot of components will be removed because presumably they are not active
    # during these 2000 frames of the experiment

    final_frate = 15  # approximate frame rate of data
    Npeaks = 10
    traces = C + YrA
    fitness_raw, fitness_delta, erfc_raw, erfc_delta, r_values, significant_samples = \
        evaluate_components(Y, traces, A, C, b, f, final_frate, remove_baseline=True,
                            N=5, robust_std=False, Npeaks=Npeaks, thresh_C=0.3)

    # filter based on spatial consistency
    idx_components_r = np.where(r_values >= .85)[0]
    # filter based on transient size
    idx_components_raw = np.where(fitness_raw < -50)[0]
    # filter based on transient derivative size
    idx_components_delta = np.where(fitness_delta < -50)[0]
    idx_components = np.union1d(idx_components_r, idx_components_raw)
    idx_components = np.union1d(idx_components, idx_components_delta)
    idx_components_bad = np.setdiff1d(list(range(len(traces))), idx_components)
    print((len(traces)))
    print((len(idx_components)))

    #%% visualize components
    # pl.figure();
    pl.subplot(1, 2, 1)
    crd = cm.utils.visualization.plot_contours(
        A.tocsc()[:, idx_components], Cn, thr=0.9)
    pl.title('selected components')
    pl.subplot(1, 2, 2)
    pl.title('discaded components')
    crd = cm.utils.visualization.plot_contours(
        A.tocsc()[:, idx_components_bad], Cn, thr=0.9)
    #%% visualize selected components
    cm.utils.visualization.view_patches_bar(Yr, A.tocsc()[:, idx_components], C[
        idx_components, :], b, f, dims[0], dims[1], YrA=YrA[idx_components, :], img=Cn)

    #%% STOP CLUSTER and clean up log files
    cm.stop_server()

    log_files = glob.glob('Yr*_LOG_*')
    for log_file in log_files:
        os.remove(log_file)
