#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Demo for detecting ROIs in a structural channel and then seeding CNMF with
them. Detection happens through simple adaptive thresholding of the mean image
and could potentially be improved. Then the structural channel is processed.
Both offline and online approaches are included.

The offline approach will only use the seeded masks, whereas online will also
search for new components during the analysis.

The demo assumes that both channels are motion corrected prior to the analysis
although this is not necessary for the case of caiman online.

@author: epnevmatikakis
"""

try:
    if __IPYTHON__:
        print('Debugging!')
        get_ipython().magic('load_ext autoreload')
        get_ipython().magic('autoreload 2')
except NameError:
    print('Not launched under iPython')

import numpy as np
import glob
import matplotlib.pyplot as plt
import caiman as cm
# from caiman.motion_correction import MotionCorrect
from caiman.source_extraction import cnmf as cnmf
import os

# %% use this cell for splitting channels and performing motion correction
# # %% extract individual channels and motion correct them
#
# fname = '/nuclear/gmc_960_30mw_00001.tif'
# fname_green = fname.split('.')[0] + '_green_raw.tif'
# fname_red = fname.split('.')[0] + '_red_raw.tif'
# m = cm.load(fname)
#
# m[::2].save(fname_green)
# m[1::2].save(fname_red)
#
#
# # %% motion correction
#
# # dataset dependent parameters
# fr = 30             # imaging rate in frames per second
# dxy = (1., 1.)      # spatial resolution in x and y in (um per pixel)
# # note the lower than usual spatial resolution here
# max_shift_um = (12., 12.)       # maximum shift in um
# patch_motion_um = (100., 100.)  # patch size for non-rigid correction in um
#
# # motion correction parameters
# pw_rigid = True       # flag to select rigid vs pw_rigid motion correction
# # maximum allowed rigid shift in pixels
# max_shifts = [int(a/b) for a, b in zip(max_shift_um, dxy)]
# # start a new patch for pw-rigid motion correction every x pixels
# strides = tuple([int(a/b) for a, b in zip(patch_motion_um, dxy)])
# # overlap between pathes (size of patch in pixels: strides+overlaps)
# overlaps = (24, 24)
# # maximum deviation allowed for patch with respect to rigid shifts
# max_deviation_rigid = 3
#
# mc_dict = {
#    'fnames': fname_red,
#    'fr': fr,
#    'dxy': dxy,
#    'pw_rigid': pw_rigid,
#    'max_shifts': max_shifts,
#    'strides': strides,
#    'overlaps': overlaps,
#    'max_deviation_rigid': max_deviation_rigid,
#    'border_nan': 'copy'
# }
#
#  opts = params.CNMFParams(params_dict=mc_dict)
#
# #%%
#
# c, dview, n_processes = cm.cluster.setup_cluster(
#        backend='local', n_processes=None, single_thread=False)
#
# # %%% MOTION CORRECTION
#    # first we create a motion correction object with the specified parameters
#    mc = MotionCorrect(fname_red, dview=dview, **opts.get_group('motion'))
#    # note that the file is not loaded in memory
#
# # %% Run (piecewise-rigid motion) correction using NoRMCorre
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

if use_online:  # run OnAcid algorithm (CaImAn online)

    show_movie = not True
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

    cnm_on = cnmf.online_cnmf.OnACID(params=opts)
    cnm_on.estimates.A = Ain
    cnm_on.fit_online()

    # %% plot some results

    cnm_on.estimates.plot_contours()
    cnm_on.estimates.view_components()
    # view components. Last components are the components added by OnACID

else:  # run offline CNMF algorithm (CaImAn Batch)
    # %% start cluster
    c, dview, n_processes = cm.cluster.setup_cluster(
        backend='local', n_processes=None, single_thread=False)

    # %% set parameters, cnmf object, and fit

    params_dict = {'fnames': fname_green,
                   'fr': fr,
                   'decay_time': decay_time,
                   'gSig': gSig,
                   'p': p,
                   'min_SNR': min_SNR,
                   'rval_thr': rval_thr,
                   'nb': gnb,
                   'only_init': False,
                   'rf': None}

    opts = cnmf.params.CNMFParams(params_dict=params_dict)
    cnm_b = cnmf.CNMF(n_processes, params=opts, dview=dview, Ain=Ain)
    cnm_b.fit_file(motion_correct=False)

    # %% load memory mapped file and evaluate components

    Yr, dims, T = cm.load_memmap(cnm_b.mmap_file)
    images = np.reshape(Yr.T, [T] + list(dims), order='F')
    Cn = cm.local_correlations(images, swap_dim=False)
    cnm_b.estimates.plot_contours(img=Cn)

    # %% evalute components and do some plotting
    cnm_b.estimates.evaluate_components(images, cnm_b.params, dview=dview)
    cnm_b.estimates.plot_contours(img=Cn, idx=cnm_b.estimates.idx_components)
    cnm_b.estimates.view_components(images, img=Cn,
                                    idx=cnm_b.estimates.idx_components)

    # %% STOP CLUSTER and clean up log files
    cm.stop_server()

    log_files = glob.glob('Yr*_LOG_*')
    for log_file in log_files:
        os.remove(log_file)
