    ##@package demos  
#\brief      for the user/programmer to understand and try the code
#\details    all of other usefull functions (demos available on jupyter notebook) -*- coding: utf-8 -*- 
#\version   1.0
#\pre       EXample.First initialize the system.
#\bug       
#\warning   
#\copyright GNU General Public License v2.0 
#\date Created on Mon Nov 21 15:53:15 2016
#\author agiovann
#toclean

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
    print('Not IPYTHON')
    pass
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
#%%
# @params params_movie set parameters and create template by RIGID MOTION CORRECTION
#params_movie = {'fname': ['Sue_2x_3000_40_-46.tif'],
#               'niter_rig': 1,
#               'max_shifts': (3, 3),  # maximum allow rigid shift
#               'splits_rig': 20,  # for parallelization split the movies in  num_splits chuncks across time
#               # if none all the splits are processed and the movie is saved
#               'num_splits_to_process_rig': None,
#               # intervals at which patches are laid out for motion correction
#               'strides': (48, 48),
#               # overlap between pathes (size of patch strides+overlaps)
#               'overlaps': (24, 24),
#               'splits_els': 28,  # for parallelization split the movies in  num_splits chuncks across time
#               # if none all the splits are processed and the movie is saved
#               'num_splits_to_process_els': [14, None],
#               'upsample_factor_grid': 4,  # upsample factor to avoid smearing when merging patches
#               # maximum deviation allowed for patch with respect to rigid
#               # shift
#               'max_deviation_rigid': 2,
#               'p': 1,  # order of the autoregressive system
#               'merge_thresh': 0.8,  # merging threshold, max correlation allowed
#               'rf': 15,  # half-size of the patches in pixels. rf=25, patches are 50x50
#               'stride_cnmf': 6,  # amounpl.it of overlap between the patches in pixels
#               'K': 4,  # number of components per patch
#               # if dendritic. In this case you need to set init_method to
#               # sparse_nmf
#               'is_dendrites': False,
#               'init_method': 'greedy_roi',
#               'gSig': [4, 4],  # expected half size of neurons
#               'alpha_snmf': None,  # this controls sparsity
#               'final_frate': 30,
#               'r_values_min_patch': .7,  # threshold on space consistency
#               'fitness_min_patch': -20,  # threshold on time variability
#                # threshold on time variability (if nonsparse activity)
#               'fitness_delta_min_patch': -20,
#               'Npeaks': 10,
#               'r_values_min_full': .8,
#               'fitness_min_full': - 40,
#               'fitness_delta_min_full': - 40,
#               'only_init_patch': True,
#               'gnb': 2,
#               'memory_fact': 1,
#               'n_chunks': 10,
#               'update_background_components': True,# whether to update the background components in the spatial phase
#               'low_rank_background': True  #whether to update the using a low rank approximation. In the False case all the nonzero elements of the background components are updated using hals    
#                                     #(to be used with one background per patch)                              
#               }

#%%
params_movie = {'fname': ['/mnt/ceph/users/agiovann/ImagingData/GrossmanAndres/AG07-04_04-23-17-Burlap-PRE_F400_5400.tif'],
                 'max_shifts': (12, 12),  # maximum allow rigid shift (2,2)
                 'niter_rig': 1,
                 'splits_rig': 14,  # for parallelization split the movies in  num_splits chuncks across time
                 'num_splits_to_process_rig': None,  # if none all the splits are processed and the movie is saved
                 'strides': (210, 210),  # intervals at which patches are laid out for motion correction
                 'overlaps': (32, 32),  # overlap between pathes (size of patch strides+overlaps)
                 'splits_els': 14,  # for parallelization split the movies in  num_splits chuncks across time
                 'num_splits_to_process_els': [14, None],  # if none all the splits are processed and the movie is saved
                 'upsample_factor_grid': 4,  # upsample factor to avoid smearing when merging patches
                 'max_deviation_rigid': 2,  # maximum deviation allowed for patch with respect to rigid shift
                 'p': 1,  # order of the autoregressive system
                 'merge_thresh': 0.8,  # merging threshold, max correlation allow
                 'rf': 15,  # half-size of the patches in pixels. rf=25, patches are 50x50    20
                 'stride_cnmf': 10,  # amounpl.it of overlap between the patches in pixels
                 'K': 6,  # number of components per patch
                 'is_dendrites': False,  # if dendritic. In this case you need to set init_method to sparse_nmf
                 'init_method': 'greedy_roi',
                 'gSig': [5, 5],  # expected half size of neurons
                 'alpha_snmf': None,  # this controls sparsity
                 'final_frate': 10,
                 'r_values_min_patch': .7,  # threshold on space consistency
                 'fitness_min_patch': -20,  # threshold on time variability
                 # threshold on time variability (if nonsparse activity)
                 'fitness_delta_min_patch': -20,
                 'Npeaks': 10,
                 'r_values_min_full': .8,
                 'fitness_min_full': - 40,
                 'fitness_delta_min_full': - 40,
                 'only_init_patch': True,
                 'gnb': 2,
                 'memory_fact': 1,
                 'n_chunks': 10,
                 'update_background_components': True,# whether to update the background components in the spatial phase
                 'low_rank_background': True #whether to update the using a low rank approximation. In the False case all the nonzero elements of the background components are updated using hals    
                                     #(to be used with one background per patch)          
                 }

#%%
params_display = {
    'downsample_ratio': .2,
    'thr_plot': 0.8
}

# TODO: do find&replace on those parameters and delete this paragrph

# @params fname name of the movie
fname = params_movie['fname']
niter_rig = params_movie['niter_rig']
# @params max_shifts maximum allow rigid shift
max_shifts = params_movie['max_shifts']

# @params splits_rig for parallelization split the movies in  num_splits chuncks across time
splits_rig = params_movie['splits_rig']

# @params num_splits_to_process_ri if none all the splits are processed and the movie is saved
num_splits_to_process_rig = params_movie['num_splits_to_process_rig']

# @params strides intervals at which patches are laid out for motion correction
strides = params_movie['strides']

# @ prams overlaps overlap between pathes (size of patch strides+overlaps)
overlaps = params_movie['overlaps']

# @params splits_els for parallelization split the movies in  num_splits chuncks across time
splits_els = params_movie['splits_els']

# @params num_splits_to_process_els  if none all the splits are processed and the movie is saved
num_splits_to_process_els = params_movie['num_splits_to_process_els']

# @params upsample_factor_grid upsample factor to avoid smearing when merging patches
upsample_factor_grid = params_movie['upsample_factor_grid']

# @params max_deviation_rigid maximum deviation allowed for patch with respect to rigid shift
max_deviation_rigid = params_movie['max_deviation_rigid']

# %% download movie if not there
if fname[0] in ['Sue_2x_3000_40_-46.tif','demoMovieJ.tif']:
    # TODO: todocument
    download_demo(fname[0])
    fname = [os.path.join('example_movies',fname[0])]
# TODO: todocument
m_orig = cm.load_movie_chain(fname[:1])

# %% play movie
downsample_ratio = params_display['downsample_ratio']
offset_mov = -np.min(m_orig[:100])
m_orig.resize(1, 1, downsample_ratio).play(
    gain=10, offset=offset_mov, fr=30, magnification=2)

# %% RUN ANALYSIS
c, dview, n_processes = cm.cluster.setup_cluster(
    backend='local', n_processes=None, single_thread=False)

# %% INITIALIZING
t1 = time.time()
# movie must be mostly positive for this to work
# TODO : document
# setting timer to see how the changement in functions make the code react on a same computer.

min_mov = cm.load(fname[0], subindices=range(400)).min()
mc_list = []
new_templ = None
for each_file in fname:
# TODO: needinfo how the classes works
    mc = MotionCorrect(each_file, min_mov,
                       dview=dview, max_shifts=max_shifts, niter_rig=niter_rig, splits_rig=splits_rig,
                       num_splits_to_process_rig=num_splits_to_process_rig,
                       strides=strides, overlaps=overlaps, splits_els=splits_els,
                       num_splits_to_process_els=num_splits_to_process_els,
                       upsample_factor_grid=upsample_factor_grid, max_deviation_rigid=max_deviation_rigid,
                       shifts_opencv=True, nonneg_movie=True)
    mc.motion_correct_rigid(save_movie=True,template = new_templ)
    new_templ = mc.total_template_rig
    m_rig = cm.load(mc.fname_tot_rig)
    bord_px_rig = np.ceil(np.max(mc.shifts_rig)).astype(np.int)

    # TODO : needinfo
    pl.imshow(new_templ, cmap='gray')
    pl.pause(.1)
    mc_list.append(mc)
# we are going to keep this part because it helps the user understand what we need.
# needhelp why it is not the same as in the notebooks ?
# TODO: show screenshot 2,3

# %%
# load motion corrected movie
m_rig = cm.load(mc.fname_tot_rig)
pl.imshow(mc.total_template_rig, cmap='gray')
# %% visualize templates
cm.movie(np.array(mc.templates_rig)).play(
    fr=10, gain=5, magnification=2, offset=offset_mov)
# %% plot rigid shifts
pl.close()
pl.plot(mc.shifts_rig)
pl.legend(['x shifts', 'y shifts'])
pl.xlabel('frames')
pl.ylabel('pixels')
# %% inspect movie
downsample_ratio = params_display['downsample_ratio']
# TODO: todocument
offset_mov = -np.min(m_orig[:100])
m_rig.resize(1, 1, downsample_ratio).play(
    gain=10, offset=offset_mov * .25, fr=30, magnification=2, bord_px=bord_px_rig)
# %%
# a computing intensive but parralellized part
t1 = time.time()
mc.motion_correct_pwrigid(save_movie=True,
                          template=mc.total_template_rig, show_template=True)
# TODO: change var name els= pwr
m_els = cm.load(mc.fname_tot_els)
pl.imshow(mc.total_template_els, cmap='gray')
# TODO: show screenshot 5
# TODO: bug sometimes saying there is no y_shifts_els
bord_px_els = np.ceil(np.maximum(np.max(np.abs(mc.x_shifts_els)),
                                 np.max(np.abs(mc.y_shifts_els)))).astype(np.int)
# %% visualize elastic shifts
pl.close()
pl.subplot(2, 1, 1)
pl.plot(mc.x_shifts_els)
pl.ylabel('x shifts (pixels)')
pl.subplot(2, 1, 2)
pl.plot(mc.y_shifts_els)
pl.ylabel('y_shifts (pixels)')
pl.xlabel('frames')
# TODO: show screenshot 6
# %% play corrected and downsampled movie
downsample_ratio = 0.2
m_els.resize(1, 1, downsample_ratio).play(
    gain=3, offset=0, fr=100, magnification=1, bord_px=bord_px_els)
# %% local correlation
_Cn = m_els.local_correlations(eight_neighbours=True, swap_dim=False)
pl.imshow(_Cn)
# TODO: show screenshot 7
# %% visualize raw, rigid and pw-rigid motion correted moviews
downsample_factor = params_display['downsample_ratio']
# TODO : todocument
cm.concatenate(
    [m_orig.resize(1, 1, downsample_factor) + offset_mov, m_rig.resize(1, 1, downsample_factor), m_els.resize(
        1, 1, downsample_factor)], axis=2)[700:1000].play(fr=60, gain=3, magnification=1, offset=0)
# TODO: show screenshot 8
# %% compute metrics for the results, just to check that motion correction worked properly
final_size = np.subtract(mc.total_template_els.shape, 2 * bord_px_els)
winsize = 100
swap_dim = False
resize_fact_flow = params_display['downsample_ratio']
# computationnaly intensive
# TODO: todocument
tmpl, correlations, flows_orig, norms, smoothness = cm.motion_correction.compute_metrics_motion_correction(
    mc.fname_tot_els, final_size[0], final_size[1], swap_dim, winsize=winsize, play_flow=False,
    resize_fact_flow=resize_fact_flow)
tmpl, correlations, flows_orig, norms, smoothness = cm.motion_correction.compute_metrics_motion_correction(
    mc.fname_tot_rig, final_size[0], final_size[1], swap_dim, winsize=winsize, play_flow=False,
    resize_fact_flow=resize_fact_flow)
tmpl, correlations, flows_orig, norms, smoothness = cm.motion_correction.compute_metrics_motion_correction(
    fname[0], final_size[0], final_size[1], swap_dim, winsize=winsize, play_flow=False,
    resize_fact_flow=resize_fact_flow)
# %% plot the results of metrics
fls = [mc.fname_tot_els[:-4] + '_metrics.npz', mc.fname_tot_rig[:-4] +
       '_metrics.npz', mc.fname[:-4] + '_metrics.npz']
# %%
for cnt, fl, metr in zip(range(len(fls)), fls, ['pw_rigid', 'rigid', 'raw']):
    with np.load(fl) as ld:
        print(ld.keys())
        #        pl.figure()
        print(fl)
        print(str(np.mean(ld['norms'])) + '+/-' + str(np.std(ld['norms'])) +
              ' ; ' + str(ld['smoothness']) + ' ; ' + str(ld['smoothness_corr']))
        # here was standing an iftrue ..
        pl.subplot(len(fls), 4, 1 + 4 * cnt)
        pl.ylabel(metr)
        try:
            mean_img = np.mean(
                cm.load(fl[:-12] + 'mmap'), 0)[12:-12, 12:-12]
        except:
            try:
                mean_img = np.mean(
                    cm.load(fl[:-12] + '.tif'), 0)[12:-12, 12:-12]
            except:
                mean_img = np.mean(
                    cm.load(fl[:-12] + 'hdf5'), 0)[12:-12, 12:-12]

        lq, hq = np.nanpercentile(mean_img, [.5, 99.5])
        pl.imshow(mean_img, vmin=lq, vmax=hq)
        pl.title('Mean')
        #        pl.plot(ld['correlations'])

        pl.subplot(len(fls), 4, 4 * cnt + 2)
        pl.imshow(ld['img_corr'], vmin=0, vmax=.35)
        pl.title('Corr image')
        #        pl.colorbar()
        pl.subplot(len(fls), 4, 4 * cnt + 3)
        #
        pl.plot(ld['norms'])
        pl.xlabel('frame')
        pl.ylabel('norm opt flow')
        pl.subplot(len(fls), 4, 4 * cnt + 4)
        flows = ld['flows']
        pl.imshow(np.mean(
            np.sqrt(flows[:, :, :, 0] ** 2 + flows[:, :, :, 1] ** 2), 0), vmin=0, vmax=0.3)
        pl.colorbar()
        pl.title('Mean optical flow')
# TODO: show screenshot 9
