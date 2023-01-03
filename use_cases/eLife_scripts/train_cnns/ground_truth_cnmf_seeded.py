#!/usr/bin/env python

# @package demos
#\brief      for the user/programmer to understand and try the code
#\details    all of other usefull functions (demos available on jupyter notebook) -*- coding: utf-8 -*-
#\version   1.0
#\pre       EXample.First initialize the system.
#\bug
#\warning
#\copyright GNU General Public License v2.0
#\date Created on Mon Nov 21 15:53:15 2016
#\author agiovann
# toclean

"""
Prepare ground truth built by matching with the results of CNMF
"""

from __future__ import division
from __future__ import print_function
from builtins import zip
from builtins import str
from builtins import map
from builtins import range
from past.utils import old_div
import cv2
import glob

try:
    cv2.setNumThreads(1)
except:
    print('Open CV is naturally single threaded')

try:
    if __IPYTHON__:
        print(1)
        # this is used for debugging purposes only. allows to reload classes
        # when changed
        get_ipython().magic('load_ext autoreload')
        get_ipython().magic('autoreload 2')
except NameError:
    print('Not launched under iPython')

import caiman as cm
import numpy as np
import os
import time
import pylab as pl
import psutil
import sys
from ipyparallel import Client
from skimage.external.tifffile import TiffFile
import scipy
import copy

from caiman.utils.utils import download_demo
from caiman.base.rois import extract_binary_masks_blob
from caiman.utils.visualization import plot_contours, view_patches_bar
from caiman.source_extraction.cnmf import cnmf as cnmf
from caiman.motion_correction import MotionCorrect
from caiman.components_evaluation import estimate_components_quality

from caiman.components_evaluation import evaluate_components

from caiman.tests.comparison import comparison
from caiman.motion_correction import tile_and_correct, motion_correction_piecewise

#%% neurofinder.03.00.test
params_movie = {'fname': ['/mnt/ceph/neuro/labeling/neurofinder.03.00.test/images/final_map/Yr_d1_498_d2_467_d3_1_order_C_frames_2250_.mmap'],
                'gtname': ['/mnt/ceph/neuro/labeling/neurofinder.03.00.test/regions/joined_consensus_active_regions.npy'],
                'p': 1,  # order of the autoregressive system
                'merge_thresh': 1,  # merging threshold, max correlation allow
                'final_frate': 10,
                #                 'r_values_min_patch': .7,  # threshold on space consistency
                #                 'fitness_min_patch': -20,  # threshold on time variability
                #                 # threshold on time variability (if nonsparse activity)
                #                 'fitness_delta_min_patch': -20,
                #                 'Npeaks': 10,
                #                 'r_values_min_full': .8,
                #                 'fitness_min_full': - 40,
                #                 'fitness_delta_min_full': - 40,
                #                 'only_init_patch': True,
                'gnb': 1,
                #                 'memory_fact': 1,
                #                 'n_chunks': 10,
                # whether to update the background components in the spatial phase
                'update_background_components': True,
                'low_rank_background': True,  # whether to update the using a low rank approximation. In the False case all the nonzero elements of the background components are updated using hals
                #(to be used with one background per patch)
                'swap_dim': False  # for some movies needed
                }
#%% neurofinder.04.00.test
params_movie = {'fname': ['/mnt/ceph/neuro/labeling/neurofinder.04.00.test/images/final_map/Yr_d1_512_d2_512_d3_1_order_C_frames_3000_.mmap'],
                'gtname': ['/mnt/ceph/neuro/labeling/neurofinder.04.00.test/regions/joined_consensus_active_regions.npy'],
                'p': 1,  # order of the autoregressive system
                'merge_thresh': 1,  # merging threshold, max correlation allow
                'final_frate': 10,
                #                 'r_values_min_patch': .7,  # threshold on space consistency
                #                 'fitness_min_patch': -20,  # threshold on time variability
                #                 # threshold on time variability (if nonsparse activity)
                #                 'fitness_delta_min_patch': -20,
                #                 'Npeaks': 10,
                #                 'r_values_min_full': .8,
                #                 'fitness_min_full': - 40,
                #                 'fitness_delta_min_full': - 40,
                #                 'only_init_patch': True,
                'gnb': 1,
                #                 'memory_fact': 1,
                #                 'n_chunks': 10,
                # whether to update the background components in the spatial phase
                'update_background_components': True,
                'low_rank_background': True,  # whether to update the using a low rank approximation. In the False case all the nonzero elements of the background components are updated using hals
                #(to be used with one background per patch)
                'swap_dim': False  # for some movies needed

                }
#%% packer: too problematic
# params_movie = {'fname': ['/mnt/ceph/neuro/labeling/packer.001/images/final_map/Yr_d1_512_d2_512_d3_1_order_C_frames_9900_.mmap'],
#                'gtname':['/mnt/ceph/neuro/labeling/packer.001/regions/joined_consensus_active_regions.npy'],
#                 'p': 1,  # order of the autoregressive system
#                 'merge_thresh': 1,  # merging threshold, max correlation allow
#                 'final_frate': 30,
# 'r_values_min_patch': .7,  # threshold on space consistency
# 'fitness_min_patch': -20,  # threshold on time variability
# threshold on time variability (if nonsparse activity)
# 'fitness_delta_min_patch': -20,
# 'Npeaks': 10,
# 'r_values_min_full': .8,
# 'fitness_min_full': - 40,
# 'fitness_delta_min_full': - 40,
# 'only_init_patch': True,
#                 'gnb': 1,
# 'memory_fact': 1,
# 'n_chunks': 10,
#                 'update_background_components': True,# whether to update the background components in the spatial phase
#                 'low_rank_background': True #whether to update the using a low rank approximation. In the False case all the nonzero elements of the background components are updated using hals
#                                     #(to be used with one background per patch)
#                 }

#%% Yi not clear neurons
params_movie = {'fname': ['/mnt/ceph/neuro/labeling/Yi.data.001/images/final_map/Yr_d1_512_d2_512_d3_1_order_C_frames_7826_.mmap'],
                'gtname': ['/mnt/ceph/neuro/labeling/Yi.data.001/regions/joined_consensus_active_regions.npy'],
                'p': 1,  # order of the autoregressive system
                'merge_thresh': 1,  # merging threshold, max correlation allow
                'final_frate': 30,
                #                 'r_values_min_patch': .7,  # threshold on space consistency
                #                 'fitness_min_patch': -20,  # threshold on time variability
                #                 # threshold on time variability (if nonsparse activity)
                #                 'fitness_delta_min_patch': -20,
                #                 'Npeaks': 10,
                #                 'r_values_min_full': .8,
                #                 'fitness_min_full': - 40,
                #                 'fitness_delta_min_full': - 40,
                #                 'only_init_patch': True,
                'gnb': 1,
                #                 'memory_fact': 1,
                #                 'n_chunks': 10,
                # whether to update the background components in the spatial phase
                'update_background_components': True,
                'low_rank_background': True  # whether to update the using a low rank approximation. In the False case all the nonzero elements of the background components are updated using hals
                #(to be used with one background per patch)
                }

#%% neurofinder.02.00
params_movie = {'fname': ['/mnt/ceph/neuro/labeling/neurofinder.02.00/images/final_map/Yr_d1_512_d2_512_d3_1_order_C_frames_8000_.mmap'],
                'gtname': ['/mnt/ceph/neuro/labeling/neurofinder.02.00/regions/joined_consensus_active_regions.npy'],
                'merge_thresh': .8,  # merging threshold, max correlation allow
                'final_frate': 10,
                'gnb': 1,
                # whether to update the background components in the spatial phase
                'update_background_components': True,
                'low_rank_background': True,  # whether to update the using a low rank approximation. In the False case all the nonzero elements of the background components are updated using hals
                                     #(to be used with one background per patch)
                'swap_dim': False  # for some movies needed
                }
#%% yuste: used kernel = np.ones((radius//4,radius//4),np.uint8)
params_movie = {'fname': ['/mnt/ceph/neuro/labeling/yuste.Single_150u/images/final_map/Yr_d1_200_d2_256_d3_1_order_C_frames_3000_.mmap'],
                'gtname': ['/mnt/ceph/neuro/labeling/yuste.Single_150u/regions/joined_consensus_active_regions.npy'],
                'p': 1,  # order of the autoregressive system
                'merge_thresh': 1,  # merging threshold, max correlation allow
                'final_frate': 10,
                'gnb': 1,
                # whether to update the background components in the spatial phase
                'update_background_components': True,
                'low_rank_background': True,  # whether to update the using a low rank approximation. In the False case all the nonzero elements of the background components are updated using hals
                #(to be used with one background per patch)
                'swap_dim': False  # for some movies needed
                }
#%% neurofinder 00 00
params_movie = {'fname': ['/mnt/ceph/neuro/labeling/neurofinder.00.00/images/final_map/Yr_d1_512_d2_512_d3_1_order_C_frames_2936_.mmap'],
                'gtname': ['/mnt/ceph/neuro/labeling/neurofinder.00.00/regions/joined_consensus_active_regions.npy'],
                'p': 1,  # order of the autoregressive system
                'merge_thresh': 1,  # merging threshold, max correlation allow
                'final_frate': 10,
                #                 'r_values_min_patch': .7,  # threshold on space consistency
                #                 'fitness_min_patch': -20,  # threshold on time variability
                #                 # threshold on time variability (if nonsparse activity)
                #                 'fitness_delta_min_patch': -20,
                #                 'Npeaks': 10,
                #                 'r_values_min_full': .8,
                #                 'fitness_min_full': - 40,
                #                 'fitness_delta_min_full': - 40,
                #                 'only_init_patch': True,
                'gnb': 1,
                #                 'memory_fact': 1,
                #                 'n_chunks': 10,
                # whether to update the background components in the spatial phase
                'update_background_components': True,
                'low_rank_background': True,  # whether to update the using a low rank approximation. In the False case all the nonzero elements of the background components are updated using hals
                #(to be used with one background per patch)
                'swap_dim': False  # for some movies needed
                }
#%% k56
params_movie = {'fname': ['/opt/local/Data/labeling/k53_20160530/Yr_d1_512_d2_512_d3_1_order_C_frames_116043_.mmap'],
                'gtname': ['/mnt/ceph/neuro/labeling/k53_20160530/regions/joined_consensus_active_regions.npy'],
                'seed_name': ['/mnt/ceph/neuro/labeling/k53_20160530/regions/joined_consensus_active_regions.npy'],
                'p': 1,  # order of the autoregressive system
                'merge_thresh': 1,  # merging threshold, max correlation allow
                'final_frate': 30,
                'gnb': 1,
                # whether to update the background components in the spatial phase
                'update_background_components': True,
                'low_rank_background': True,  # whether to update the using a low rank approximation. In the False case all the nonzero elements of the background components are updated using hals
                #(to be used with one background per patch)
                'swap_dim': False,  # for some movies needed
                'kernel': None
                }

#%% yuste: sue ann = np.ones((radius//4,radius//4),np.uint8)
# params_movie = {'fname': ['/mnt/ceph/neuro/labeling/yuste.Single_150u/images/final_map/Yr_d1_200_d2_256_d3_1_order_C_frames_3000_.mmap'],
#                'gtname':['/mnt/ceph/neuro/labeling/yuste.Single_150u/regions/joined_consensus_active_regions.npy'],
#                'seed_name':['/mnt/xfs1/home/agiovann/Downloads/yuste_sue_masks.mat'],
#                 'p': 1,  # order of the autoregressive system
#                 'merge_thresh': 1,  # merging threshold, max correlation allow
#                 'final_frate': 10,
#                 'gnb': 1,
#                 'update_background_components': True,# whether to update the background components in the spatial phase
#                 'low_rank_background': True, #whether to update the using a low rank approximation. In the False case all the nonzero elements of the background components are updated using hals
#                                     #(to be used with one background per patch)
#                 'swap_dim':False, #for some movies needed
#                 'kernel':None
#                 }
#%% neurofinder: 01.01
params_movie = {'fname': ['/mnt/ceph/neuro/labeling/neurofinder.01.01/images/final_map/Yr_d1_512_d2_512_d3_1_order_C_frames_1825_.mmap'],
                'gtname': ['/mnt/ceph/neuro/labeling/neurofinder.01.01/regions/joined_consensus_active_regions.npy'],
                'seed_name': ['/mnt/ceph/neuro/labeling/neurofinder.01.01/regions/joined_consensus_active_regions.npy'],
                'p': 1,  # order of the autoregressive system
                'merge_thresh': 1,  # merging threshold, max correlation allow
                'final_frate': 10,
                'gnb': 1,
                # whether to update the background components in the spatial phase
                'update_background_components': True,
                'low_rank_background': True,  # whether to update the using a low rank approximation. In the False case all the nonzero elements of the background components are updated using hals
                #(to be used with one background per patch)
                'swap_dim': False,  # for some movies needed
                'kernel': None
                }

#%% J115: 01.01
params_movie = {'fname': ['/mnt/ceph/neuro/labeling/J115_2015-12-09_L01_ELS/images/final_map/Yr_d1_463_d2_472_d3_1_order_C_frames_90000_.mmap'],
                'gtname': ['/mnt/ceph/neuro/labeling/J115_2015-12-09_L01_ELS/regions/joined_consensus_active_regions.npy'],
                'seed_name': ['/mnt/ceph/neuro/labeling/J115_2015-12-09_L01_ELS/regions/joined_consensus_active_regions.npy'],
                'p': 1,  # order of the autoregressive system
                'merge_thresh': 1,  # merging threshold, max correlation allow
                'final_frate': 10,
                'gnb': 1,
                # whether to update the background components in the spatial phase
                'update_background_components': True,
                'low_rank_background': True,  # whether to update the using a low rank approximation. In the False case all the nonzero elements of the background components are updated using hals
                #(to be used with one background per patch)
                'swap_dim': False,  # for some movies needed
                'kernel': None
                }


#%% J123
params_movie = {'fname': ['/mnt/ceph/neuro/labeling/J123_2015-11-20_L01_0/images/final_map/Yr_d1_458_d2_477_d3_1_order_C_frames_41000_.mmap'],
                'gtname': ['/mnt/ceph/neuro/labeling/J123_2015-11-20_L01_0/regions/joined_consensus_active_regions.npy'],
                'seed_name': ['/mnt/ceph/neuro/labeling/J123_2015-11-20_L01_0/regions/joined_consensus_active_regions.npy'],
                'p': 1,  # order of the autoregressive system
                'merge_thresh': 1,  # merging threshold, max correlation allow
                'final_frate': 10,
                'gnb': 1,
                # whether to update the background components in the spatial phase
                'update_background_components': True,
                'low_rank_background': True,  # whether to update the using a low rank approximation. In the False case all the nonzero elements of the background components are updated using hals
                #(to be used with one background per patch)
                'swap_dim': False,  # for some movies needed
                'kernel': None
                }
#%% Jan-AMG
params_movie = {'fname': ['/mnt/ceph/neuro/labeling/Jan-AMG_exp3_001/images/final_map/Yr_d1_512_d2_512_d3_1_order_C_frames_115897_.mmap'],
                'gtname': ['/mnt/ceph/neuro/labeling/Jan-AMG_exp3_001/regions/joined_consensus_active_regions.npy'],
                'seed_name': ['/mnt/ceph/neuro/labeling/Jan-AMG_exp3_001/regions/joined_consensus_active_regions.npy'],
                'p': 1,  # order of the autoregressive system
                'merge_thresh': 1,  # merging threshold, max correlation allow
                'final_frate': 10,
                'gnb': 1,
                # whether to update the background components in the spatial phase
                'update_background_components': True,
                'low_rank_background': True,  # whether to update the using a low rank approximation. In the False case all the nonzero elements of the background components are updated using hals
                #(to be used with one background per patch)
                'swap_dim': False,  # for some movies needed
                'kernel': None,
                'crop_pix': 8,
                }
#%% sue k37, not nice because few events
params_movie = {'fname': ['/mnt/ceph/neuro/labeling/k37_20160109_AM_150um_65mW_zoom2p2_00001_1-16/images/final_map/Yr_d1_512_d2_512_d3_1_order_C_frames_48000_.mmap'],
                'gtname': ['/mnt/ceph/neuro/labeling/k37_20160109_AM_150um_65mW_zoom2p2_00001_1-16/regions/joined_consensus_active_regions.npy'],
                'seed_name': ['/mnt/ceph/neuro/labeling/k37_20160109_AM_150um_65mW_zoom2p2_00001_1-16/regions/joined_consensus_active_regions.npy'],
                'p': 1,  # order of the autoregressive system
                'merge_thresh': 1,  # merging threshold, max correlation allow
                'final_frate': 30,
                'gnb': 2,
                # whether to update the background components in the spatial phase
                'update_background_components': True,
                'low_rank_background': True,  # whether to update the using a low rank approximation. In the False case all the nonzero elements of the background components are updated using hals
                #(to be used with one background per patch)
                'swap_dim': False,  # for some movies needed
                'kernel': None,
                'crop_pix': 7,
                }

#%%
params_display = {
    'downsample_ratio': .2,
    'thr_plot': 0.8
}
# TODO: do find&replace on those parameters and delete this paragrph

# @params fname name of the movie
fname_new = params_movie['fname'][0]
# %% RUN ANALYSIS
c, dview, n_processes = cm.cluster.setup_cluster(
    backend='local', n_processes=None, single_thread=False)
# %% LOAD MEMMAP FILE
# fname_new='Yr_d1_501_d2_398_d3_1_order_F_frames_369_.mmap'
Yr, dims, T = cm.load_memmap(fname_new)
d1, d2 = dims
images = np.reshape(Yr.T, [T] + list(dims), order='F')
Y = np.reshape(Yr, dims + (T,), order='F')
m_images = cm.movie(images)
# %% correlation image
if m_images.shape[0] < 10000:
    Cn = m_images.local_correlations(
        swap_dim=params_movie['swap_dim'], frames_per_chunk=1500)
    Cn[np.isnan(Cn)] = 0
else:
    Cn = np.array(cm.load(('/'.join(params_movie['gtname'][0].split('/')[:-2] + [
                  'projections', 'correlation_image_better.tif'])))).squeeze()
pl.imshow(Cn, cmap='gray', vmax=.95)
# TODO: show screenshot 11
#%%
import cv2
if not '.mat' in params_movie['seed_name'][0]:
    roi_cons = np.load(params_movie['seed_name'][0])
else:
    roi_cons = scipy.io.loadmat(params_movie['seed_name'][0])['comps'].reshape(
        (dims[1], dims[0], -1), order='F').transpose([2, 1, 0]) * 1.

radius = int(np.median(np.sqrt(np.sum(roi_cons, (1, 2)) / np.pi)))

print(radius)
#roi_cons = caiman.base.rois.nf_read_roi_zip('/mnt/ceph/neuro/labeling/neurofinder.03.00.test/regions/ben_active_regions_nd_sonia_active_regions_nd__lindsey_active_regions_nd_matches.zip',dims)
#roi_cons = np.concatenate([roi_cons, caiman.base.rois.nf_read_roi_zip('/mnt/ceph/neuro/labeling/neurofinder.03.00.test/regions/intermediate_regions/ben_active_regions_nd_sonia_active_regions_nd__lindsey_active_regions_nd_1_mismatches.zip',dims)],0)

print(roi_cons.shape)
pl.imshow(roi_cons.sum(0))

if params_movie['kernel'] is not None:  # kernel usually two
    kernel = np.ones(
        (radius // params_movie['kernel'], radius // params_movie['kernel']), np.uint8)
    roi_cons = np.vstack([cv2.dilate(rr, kernel, iterations=1)[
                         np.newaxis, :, :] > 0 for rr in roi_cons]) * 1.
    pl.imshow(roi_cons.sum(0), alpha=0.5)

A_in = np.reshape(roi_cons.transpose(
    [2, 1, 0]), (-1, roi_cons.shape[0]), order='C')
pl.figure()
crd = plot_contours(A_in, Cn, thr=.99999)

# %% some parameter settings
# order of the autoregressive fit to calcium imaging in general one (slow gcamps) or two (fast gcamps fast scanning)
p = params_movie['p']
# merging threshold, max correlation allowed
merge_thresh = params_movie['merge_thresh']

# %% Extract spatial and temporal components on patches
# TODO: todocument
if images.shape[0] > 10000:
    check_nan = False
else:
    check_nan = True

cnm = cnmf.CNMF(check_nan=check_nan, n_processes=1, k=A_in.shape[-1], gSig=[radius, radius], merge_thresh=params_movie['merge_thresh'], p=params_movie['p'], Ain=A_in.astype(bool),
                dview=dview, rf=None, stride=None, gnb=params_movie['gnb'], method_deconvolution='oasis', border_pix=0, low_rank_background=params_movie['low_rank_background'], n_pixels_per_process=1000)
cnm = cnm.fit(images)

A = cnm.A
C = cnm.C
YrA = cnm.YrA
b = cnm.b
f = cnm.f
snt = cnm.sn
print(('Number of components:' + str(A.shape[-1])))
# %%
pl.figure()
# TODO: show screenshot 12`
# TODO : change the way it is used
crd = plot_contours(A, Cn, thr=params_display['thr_plot'])


# %%
# TODO: needinfo
view_patches_bar(Yr, scipy.sparse.coo_matrix(A.tocsc()[:, :]), C[:, :], b, f, dims[0], dims[1],
                 YrA=YrA[:, :], img=Cn)

#%%
c, dview, n_processes = cm.cluster.setup_cluster(
    backend='local', n_processes=None, single_thread=False)
#%% thredshold components
min_size_neuro = 3 * 2 * np.pi
max_size_neuro = (2 * radius)**2 * np.pi
A_thr = cm.source_extraction.cnmf.spatial.threshold_components(A.tocsc()[:, :].toarray(), dims, medw=None, thr_method='max', maxthr=0.2, nrgthr=0.99, extract_cc=True,
                                                               se=None, ss=None, dview=dview)

A_thr = A_thr > 0
size_neurons = A_thr.sum(0)
idx_size_neuro = np.where((size_neurons > min_size_neuro)
                          & (size_neurons < max_size_neuro))[0]
A_thr = A_thr[:, idx_size_neuro]
print(A_thr.shape)
#%%
crd = plot_contours(scipy.sparse.coo_matrix(
    A_thr * 1.), Cn, thr=.99, vmax=0.35)
#%%
roi_cons = np.load(params_movie['gtname'][0])
print(roi_cons.shape)
pl.imshow(roi_cons.sum(0))
#%% compare CNMF sedded with ground truth
pl.figure(figsize=(30, 20))
tp_gt, tp_comp, fn_gt, fp_comp, performance_cons_off = cm.base.rois.nf_match_neurons_in_binary_masks(roi_cons, A_thr[:, :].reshape([dims[0], dims[1], -1], order='F').transpose([2, 0, 1]) * 1., thresh_cost=.7, min_dist=10,
                                                                                                     print_assignment=False, plot_results=False, Cn=Cn, labels=['GT', 'Offline'])
pl.rcParams['pdf.fonttype'] = 42
font = {'family': 'Myriad Pro',
        'weight': 'regular',
        'size': 20}
pl.rc('font', **font)

#%%
np.savez(os.path.join(os.path.split(fname_new)[0], os.path.split(fname_new)[1][:-4] + 'match_masks.npz'), Cn=Cn,
         tp_gt=tp_gt, tp_comp=tp_comp, fn_gt=fn_gt, fp_comp=fp_comp, performance_cons_off=performance_cons_off, idx_size_neuro_gt=idx_size_neuro, A_thr=A_thr,
         A_gt=A, C_gt=C, b_gt=b, f_gt=f, YrA_gt=YrA, d1=d1, d2=d2, idx_components_gt=idx_size_neuro[
             tp_comp],
         idx_components_bad_gt=idx_size_neuro[fp_comp], fname_new=fname_new)

#% now use the script match_seeded_gt if you want to create a training set
