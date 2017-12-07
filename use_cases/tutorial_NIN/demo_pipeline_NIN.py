# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 15:53:15 2016

@author: agiovann
"""
from __future__ import division
from __future__ import print_function
from builtins import zip
from builtins import str
from builtins import map
from builtins import range
from past.utils import old_div
import cv2
try:
    cv2.setNumThreads(1)
except:
    print('Open CV is naturally single threaded')

try:
    if __IPYTHON__:
        print((1))
        # this is used for debugging purposes only. allows to reload classes when changed
        get_ipython().magic('load_ext autoreload')
        get_ipython().magic('autoreload 2')
except NameError:
    print('Not IPYTHON')
    pass
#%%
import caiman as cm
import numpy as np
import os
import glob
import time
import pylab as pl
import psutil
import sys
from ipyparallel import Client
from skimage.external.tifffile import TiffFile
import scipy
#%%
from caiman.motion_correction import tile_and_correct, motion_correction_piecewise
from caiman.source_extraction.cnmf import cnmf as cnmf
from caiman.components_evaluation import evaluate_components
from caiman.utils.visualization import plot_contours, view_patches_bar
from caiman.base.rois import extract_binary_masks_blob
#%% set parameters and create template by RIGID MOTION CORRECTION
params_movie = {'fname': 'example_movies/Sue_2x_3000_40_-46.tif',
                'max_shifts': (6, 6),  # maximum allow rigid shift
                'splits_rig': 56,  # for parallelization split the movies in  num_splits chuncks across time
                # if none all the splits are processed and the movie is saved
                'num_splits_to_process_rig': None,
                # intervals at which patches are laid out for motion correction
                'strides': (48, 48),
                # overlap between pathes (size of patch strides+overlaps)
                'overlaps': (24, 24),
                'splits_els': 56,  # for parallelization split the movies in  num_splits chuncks across time
                # if none all the splits are processed and the movie is saved
                'num_splits_to_process_els': [28, None],
                'upsample_factor_grid': 4,  # upsample factor to avoid smearing when merging patches
                # maximum deviation allowed for patch with respect to rigid shift
                'max_deviation_rigid': 3,
                'p': 1,  # order of the autoregressive system
                'merge_thresh': 0.8,  # merging threshold, max correlation allowed
                'rf': 15,  # half-size of the patches in pixels. rf=25, patches are 50x50
                'stride_cnmf': 6,  # amounpl.it of overlap between the patches in pixels
                'K': 4,  # number of components per patch
                # if dendritic. In this case you need to set init_method to sparse_nmf
                'is_dendrites': False,
                'init_method': 'greedy_roi',
                'gSig': [4, 4],  # expected half size of neurons
                'alpha_snmf': None,  # this controls sparsity
                'final_frate': 30
                }
#%%
# params_movie = {'fname':'example_movies/demoMovie.tif',
#                'max_shifts':(1,1), # maximum allow rigid shift
#                'splits_rig':28, # for parallelization split the movies in  num_splits chuncks across time
#                'num_splits_to_process_rig':None, # if none all the splits are processed and the movie is saved
#                'strides': (48,48), # intervals at which patches are laid out for motion correction
#                'overlaps': (24,24), # overlap between pathes (size of patch strides+overlaps)
#                'splits_els':28, # for parallelization split the movies in  num_splits chuncks across time
#                'num_splits_to_process_els':[28,None], # if none all the splits are processed and the movie is saved
#                'upsample_factor_grid':4, # upsample factor to avoid smearing when merging patches
#                'max_deviation_rigid':3, #maximum deviation allowed for patch with respect to rigid shift
#                'p': 1, # order of the autoregressive system
#                'merge_thresh' : 0.8,  # merging threshold, max correlation allowed
#                'rf' : 15,  # half-size of the patches in pixels. rf=25, patches are 50x50
#                'stride_cnmf' : 6,  # amounpl.it of overlap between the patches in pixels
#                'K' : 5,  #  number of components per patch
#                'is_dendrites': False,  # if dendritic. In this case you need to set init_method to sparse_nmf
#                'init_method' : 'greedy_roi',
#                'gSig' : [4, 4],  # expected half size of neurons
#                'alpha_snmf' : None,  # this controls sparsity
#                'final_frate' : 30
#                }
#%%
# params_movie = {'fname':'/Users/agiovann/Dropbox (Simons Foundation)/CaImWorkshop2017/ABadura/20_12__002_cropped.tif',
#                'max_shifts':(6,6), # maximum allow rigid shift
#                'splits_rig':28, # for parallelization split the movies in  num_splits chuncks across time
#                'num_splits_to_process_rig':None, # if none all the splits are processed and the movie is saved
#                'strides': (96,96), # intervals at which patches are laid out for motion correction
#                'overlaps': (32,32), # overlap between pathes (size of patch strides+overlaps)
#                'splits_els':28, # for parallelization split the movies in  num_splits chuncks across time
#                'num_splits_to_process_els':[14,None], # if none all the splits are processed and the movie is saved
#                'upsample_factor_grid':4, # upsample factor to avoid smearing when merging patches
#                'max_deviation_rigid':3, #maximum deviation allowed for patch with respect to rigid shift
#                'p': 1, # order of the autoregressive system
#                'merge_thresh' : 0.8,  # merging threshold, max correlation allowed
#                'rf' : (10,30),  # half-size of the patches in pixels. rf=25, patches are 50x50
#                'stride_cnmf' : (5,10),  # amount of overlap between the patches in pixels
#                'K' : 6,  #  number of components per patch
#                'is_dendrites': True,  # if dendritic. In this case you need to set init_method to sparse_nmf
#                'init_method' : 'sparse_nmf',
#                'gSig' : [0,0],  # expected half size of neurons
#                'alpha_snmf' : 10,  # this controls sparsity
#                'final_frate' : 30
#                }
#%%
# params_movie = {'fname':'/Users/agiovann/SOFTWARE/CaImAn/example_movies/CaImWorkshop2017/LBina/160121_b2_arthiha_brian.tif',
#                'max_shifts':(3,3), # maximum allow rigid shift
#                'splits_rig':14, # for parallelization split the movies in  num_splits chuncks across time
#                'num_splits_to_process_rig':None, # if none all the splits are processed and the movie is saved
# 'strides': (96,96), # intervals at which patches are laid out for motion correction
# 'overlaps': (32,32), # overlap between pathes (size of patch strides+overlaps)
# 'splits_els':28, # for parallelization split the movies in  num_splits chuncks across time
# 'num_splits_to_process_els':[14,None], # if none all the splits are processed and the movie is saved
# 'upsample_factor_grid':4, # upsample factor to avoid smearing when merging patches
# 'max_deviation_rigid':3, #maximum deviation allowed for patch with respect to rigid shift
#                'p': 1, # order of the autoregressive system
#                'merge_thresh' : 0.8,  # merging threshold, max correlation allowed
#                'rf' : (20,21),  # half-size of the patches in pixels. rf=25, patches are 50x50
#                'stride_cnmf' : (10,1),  # amount of overlap between the patches in pixels
#                'K' : 4,  #  number of components per patch
#                'is_dendrites': True,  # if dendritic. In this case you need to set init_method to sparse_nmf
#                'init_method' : 'sparse_nmf',
#                'gSig' : [0,0],  # expected half size of neurons
#                'alpha_snmf' : 1,  # this controls sparsity
#                'final_frate' : 30
#                }
#%%
# params_movie = {'fname':'/Users/agiovann/Dropbox (Simons Foundation)/CaImWorkshop2017/THoogland/tmh_endoscope_0.5x.tif',
#                'max_shifts':(10,10), # maximum allow rigid shift
#                'splits_rig':7, # for parallelization split the movies in  num_splits chuncks across time
#                'num_splits_to_process_rig':None, # if none all the splits are processed and the movie is saved
#                'strides': (96,96), # intervals at which patches are laid out for motion correction
#                'overlaps': (32,32), # overlap between pathes (size of patch strides+overlaps)
#                'splits_els':7, # for parallelization split the movies in  num_splits chuncks across time
#                'num_splits_to_process_els':[7,None], # if none all the splits are processed and the movie is saved
#                'upsample_factor_grid':4, # upsample factor to avoid smearing when merging patches
#                'max_deviation_rigid':3, #maximum deviation allowed for patch with respect to rigid shift
#                'p': 1, # order of the autoregressive system
#                'merge_thresh' : 0.8,  # merging threshold, max correlation allowed
#                'rf' : (40,20),  # half-size of the patches in pixels. rf=25, patches are 50x50
#                'stride_cnmf' : (15,10),  # amount of overlap between the patches in pixels
#                'K' : 6,  #  number of components per patch
#                'is_dendrites': True,  # if dendritic. In this case you need to set init_method to sparse_nmf
#                'init_method' : 'sparse_nmf',
#                'gSig' : [0,0],  # expected half size of neurons
#                'alpha_snmf' : 1,  # this controls sparsity
#                'final_frate' : 30
#                }
#%%
# params_movie = {'fname':'/Users/agiovann/Dropbox (Simons Foundation)/CaImWorkshop2017/Z.Gao/calcium imaging Gao.tif',
#                'max_shifts':(12,12), # maximum allow rigid shift
#                'splits_rig':34, # for parallelization split the movies in  num_splits chuncks across time
#                'num_splits_to_process_rig':None, # if none all the splits are processed and the movie is saved
#                'strides': (32,32), # intervals at which patches are laid out for motion correction
#                'overlaps': (32,32), # overlap between pathes (size of patch strides+overlaps)
#                'splits_els':34, # for parallelization split the movies in  num_splits chuncks across time
#                'num_splits_to_process_els':[17,None], # if none all the splits are processed and the movie is saved
#                'upsample_factor_grid':4, # upsample factor to avoid smearing when merging patches
#                'max_deviation_rigid':3, #maximum deviation allowed for patch with respect to rigid shift
#                'p': 1, # order of the autoregressive system
#                'merge_thresh' : 0.8,  # merging threshold, max correlation allowed
#                'rf' : 20,  # half-size of the patches in pixels. rf=25, patches are 50x50
#                'stride_cnmf' : 15,  # amount of overlap between the patches in pixels
#                'K' : 6,  #  number of components per patch
#                'is_dendrites': True,  # if dendritic. In this case you need to set init_method to sparse_nmf
#                'init_method' : 'sparse_nmf',
#                'gSig' : [0,0],  # expected half size of neurons
#                'alpha_snmf' : .1,  # this controls sparsity
#                'final_frate' : 30
#                }
#%%
# params_movie = {'fname':'/Users/agiovann/Dropbox (Simons Foundation)/CaImWorkshop2017/GMeijer/GM_V1_1.tif',
#                'max_shifts':(12,12), # maximum allow rigid shift
#                'splits_rig':34, # for parallelization split the movies in  num_splits chuncks across time
#                'num_splits_to_process_rig':None, # if none all the splits are processed and the movie is saved
# 'strides': (128,128), # intervals at which patches are laid out for motion correction
# 'overlaps': (32,32), # overlap between pathes (size of patch strides+overlaps)
# 'splits_els':34, # for parallelization split the movies in  num_splits chuncks across time
# 'num_splits_to_process_els':[17,None], # if none all the splits are processed and the movie is saved
# 'upsample_factor_grid':4, # upsample factor to avoid smearing when merging patches
# 'max_deviation_rigid':3, #maximum deviation allowed for patch with respect to rigid shift
#                'p': 1, # order of the autoregressive system
#                'merge_thresh' : 0.8,  # merging threshold, max correlation allowed
#                'rf' : 15,  # half-size of the patches in pixels. rf=25, patches are 50x50
#                'stride_cnmf' : 6,  # amount of overlap between the patches in pixels
#                'K' : 3,  #  number of components per patch
#                'is_dendrites': False,  # if dendritic. In this case you need to set init_method to sparse_nmf
#                'init_method' : 'greedy_roi',
#                'gSig' : [7,7],  # expected half size of neurons
#                'alpha_snmf' : None,  # this controls sparsity
#                'final_frate' : 30
#                }
#%%
# params_movie = {'fname':'/Users/agiovann/Dropbox (Simons Foundation)/CaImWorkshop2017/Johny H. Pires/JHP_mpfc_A.tif',
# 'max_shifts':(12,12), # maximum allow rigid shift
# 'splits_rig':34, # for parallelization split the movies in  num_splits chuncks across time
# 'num_splits_to_process_rig':None, # if none all the splits are processed and the movie is saved
# 'strides': (128,128), # intervals at which patches are laid out for motion correction
# 'overlaps': (32,32), # overlap between pathes (size of patch strides+overlaps)
# 'splits_els':34, # for parallelization split the movies in  num_splits chuncks across time
# 'num_splits_to_process_els':[17,None], # if none all the splits are processed and the movie is saved
# 'upsample_factor_grid':4, # upsample factor to avoid smearing when merging patches
# 'max_deviation_rigid':3, #maximum deviation allowed for patch with respect to rigid shift
#                'p': 1, # order of the autoregressive system
#                'merge_thresh' : 0.8,  # merging threshold, max correlation allowed
#                'rf' : 15,  # half-size of the patches in pixels. rf=25, patches are 50x50
#                'stride_cnmf' : 6,  # amount of overlap between the patches in pixels
#                'K' : 3,  #  number of components per patch
#                'is_dendrites': False,  # if dendritic. In this case you need to set init_method to sparse_nmf
#                'init_method' : 'greedy_roi',
#                'gSig' : [7,7],  # expected half size of neurons
#                'alpha_snmf' : None,  # this controls sparsity
#                'final_frate' : 30
#                }

#%%
# params_movie = {'fname':'/Users/agiovann/Dropbox (Simons Foundation)/CaImWorkshop2017/E.Schut/ES_CA2_crop_1p.tif',
#                'p': 1, # order of the autoregressive system
#                'merge_thresh' : 0.8,  # merging threshold, max correlation allowed
#                'rf' : 30,  # half-size of the patches in pixels. rf=25, patches are 50x50
#                'stride_cnmf' : 15,  # amounpl.it of overlap between the patches in pixels
#                'K' : 4,  #  number of components per patch
#                'is_dendrites': False,  # if dendritic. In this case you need to set init_method to sparse_nmf
#                'init_method' : 'greedy_roi',
#                'gSig' : [10, 10],  # expected half size of neurons
#                'alpha_snmf' : None,  # this controls sparsity
#                'final_frate' : 30
#                }
#%% start local cluster
c, dview, n_processes = cm.cluster.setup_cluster(
    backend='local', n_processes=None, single_thread=False)
#%% RIGID MOTION CORRECTION
t1 = time.time()
fname = params_movie['fname']
max_shifts = params_movie['max_shifts']  # maximum allowed shifts
num_iter = 1  # number of times the algorithm is run
# for parallelization split the movies in  num_splits chuncks across time
splits = params_movie['splits_rig']
# if none all the splits are processed and the movie is saved
num_splits_to_process = params_movie['num_splits_to_process_rig']
shifts_opencv = True  # apply shifts fast way (but smoothing results)
save_movie_rigid = True  # save the movies vs just get the template
t1 = time.time()
fname_tot_rig, total_template_rig, templates_rig, shifts_rig = cm.motion_correction.motion_correct_batch_rigid(
    fname, max_shifts, dview=dview, splits=splits, num_splits_to_process=num_splits_to_process, num_iter=num_iter,  template=None, shifts_opencv=shifts_opencv, save_movie_rigid=save_movie_rigid)
t2 = time.time() - t1
print(t2)
pl.imshow(total_template_rig, cmap='gray',
          vmax=np.percentile(total_template_rig, 95))
#%%
pl.close()
pl.plot(shifts_rig)
#%%
m_rig = cm.load(fname_tot_rig)
#%%
add_to_movie = - np.min(total_template_rig) + 1
print(add_to_movie)
#%% visualize movies

m_rig.resize(1, 1, .2).play(fr=30, gain=10,
                            magnification=2, offset=add_to_movie)
#%% visualize templates
cm.movie(np.array(templates_rig)).play(
    fr=10, gain=10, magnification=1, offset=add_to_movie)
#%% PIECEWISE RIGID MOTION CORRECTION

t1 = time.time()
new_templ = total_template_rig.copy()
strides = params_movie['strides']
overlaps = params_movie['overlaps']
shifts_opencv = True
save_movie = True
splits = params_movie['splits_els']
num_splits_to_process_list = params_movie['num_splits_to_process_els']
upsample_factor_grid = params_movie['upsample_factor_grid']
max_deviation_rigid = params_movie['upsample_factor_grid']
add_to_movie = - np.min(total_template_rig) + 1
num_iter = 1
for num_splits_to_process in num_splits_to_process_list:
    fname_tot_els, total_template_wls, templates_els, x_shifts_els, y_shifts_els, coord_shifts_els = cm.motion_correction.motion_correct_batch_pwrigid(fname, max_shifts, strides, overlaps, add_to_movie, newoverlaps=None,  newstrides=None,
                                                                                                                                                       dview=dview, upsample_factor_grid=upsample_factor_grid, max_deviation_rigid=max_deviation_rigid,
                                                                                                                                                       splits=splits, num_splits_to_process=num_splits_to_process, num_iter=num_iter,
                                                                                                                                                       template=new_templ, shifts_opencv=shifts_opencv, save_movie=save_movie)
    new_templ = total_template_wls
#%%
pl.subplot(2, 1, 1)
pl.plot(x_shifts_els)
pl.subplot(2, 1, 2)
pl.plot(y_shifts_els)
borders_pix = np.ceil(np.maximum(np.max(np.abs(x_shifts_els)),
                                 np.max(np.abs(y_shifts_els)))).astype(np.int)

#%%
m_els = cm.load(fname_tot_els)
downs = 1
cm.concatenate([m_rig.resize(1, 1, downs), m_els.resize(1, 1, downs)], axis=1).play(
    fr=30, gain=25, magnification=1, offset=add_to_movie)
#%% compute metrics for the results
final_size = np.subtract(new_templ.shape, 2 * borders_pix)
winsize = 100
swap_dim = False
resize_fact_flow = .2
tmpl, correlations, flows_orig, norms, smoothness = cm.motion_correction.compute_metrics_motion_correction(
    fname_tot_els, final_size[0], final_size[1], swap_dim, winsize=winsize, play_flow=False, resize_fact_flow=resize_fact_flow)
tmpl, correlations, flows_orig, norms, smoothness = cm.motion_correction.compute_metrics_motion_correction(
    fname_tot_rig, final_size[0], final_size[1], swap_dim, winsize=winsize, play_flow=False, resize_fact_flow=resize_fact_flow)
tmpl, correlations, flows_orig, norms, smoothness = cm.motion_correction.compute_metrics_motion_correction(
    fname, final_size[0], final_size[1], swap_dim, winsize=winsize, play_flow=False, resize_fact_flow=resize_fact_flow)

#%% plot the results of metrics
fls = [fname_tot_els[:-4] + '_metrics.npz', fname_tot_rig[:-4] +
       '_metrics.npz', fname[:-4] + '_metrics.npz']
for cnt, fl in enumerate(fls):
    with np.load(fl) as ld:
        #        print(ld.keys())
        #        pl.figure()
        print(fl)
        print(str(np.mean(ld['norms'])) + '+/-' +
              str(np.std(ld['norms'])) + ' ; ' + str(ld['smoothness']))
        pl.subplot(len(fls), 4, 1 + 4 * cnt)
        try:
            mean_img = np.mean(cm.load(fl[:-12] + 'mmap'), 0)[12:-12, 12:-12]
        except:
            try:
                mean_img = np.mean(
                    cm.load(fl[:-12] + '.tif'), 0)[12:-12, 12:-12]
            except:
                mean_img = np.mean(
                    cm.load(fl[:-12] + 'hdf5'), 0)[12:-12, 12:-12]
        lq, hq = np.nanpercentile(mean_img, [.5, 99.5])
        pl.imshow(mean_img, vmin=lq, vmax=hq)
    #        pl.plot(ld['correlations'])

        pl.subplot(len(fls), 4, 4 * cnt + 2)
        pl.imshow(ld['img_corr'], vmin=0, vmax=.35)
#        pl.colorbar()
        pl.subplot(len(fls), 4, 4 * cnt + 3)
#
        pl.plot(ld['norms'])
        pl.subplot(len(fls), 4, 4 * cnt + 4)
        flows = ld['flows']
        pl.imshow(np.mean(
            np.sqrt(flows[:, :, :, 0]**2 + flows[:, :, :, 1]**2), 0), vmin=0, vmax=0.3)
        pl.colorbar()
#%% save each chunk in F format
if not params_movie.has_key('max_shifts'):
    fnames = [params_movie['fname']]
    border_to_0 = 0
elif not params_movie.has_key('overlaps'):
    fnames = [fname_tot_rig]
    border_to_0 = np.ceil(np.max(np.abs(shifts_rig))).astype(np.int)
    m_els = m_rig
else:
    fnames = [fname_tot_els]
    border_to_0 = borders_pix
# idx_x=slice(border_nan,-border_nan,None)
# idx_y=slice(border_nan,-border_nan,None)
# idx_xy=(idx_x,idx_y)
add_to_movie = -np.nanmin(m_els) + 1  # movie must be positive
idx_xy = None
downsample_factor = 1  # use .2 or .1 if file is large and you want a quick answer
base_name = 'Yr'
name_new = cm.save_memmap_each(fnames, dview=dview, base_name=base_name, resize_fact=(
    1, 1, downsample_factor), remove_init=0, idx_xy=idx_xy, add_to_movie=add_to_movie, border_to_0=border_to_0)
name_new.sort()
print(name_new)
#%% concatenate chunks if needed
if len(name_new) > 1:
    fname_new = cm.save_memmap_join(
        name_new, base_name='Yr', n_chunks=12, dview=dview)
else:
    print('One file only, not saving!')
    fname_new = name_new[0]

#%%
# fname_new='Yr_d1_501_d2_398_d3_1_order_F_frames_369_.mmap'
Yr, dims, T = cm.load_memmap(fname_new)
d1, d2 = dims
images = np.reshape(Yr.T, [T] + list(dims), order='F')
Y = np.reshape(Yr, dims + (T,), order='F')
m_images = cm.movie(images)
#%%  checks on movies
if np.min(images) < 0:
    raise Exception('Movie too negative, add_to_movie should be larger')
if np.sum(np.isnan(images)) > 0:
    raise Exception('Movie contains nan! You did not remove enough borders')
#%%
Cn = cm.local_correlations(Y[:, :, :3000])
Cn[np.isnan(Cn)] = 0
pl.imshow(Cn, cmap='gray', vmax=.99)
#%% some parameter settings
p = params_movie['p']  # order of the autoregressive system
# merging threshold, max correlation allowed
merge_thresh = params_movie['merge_thresh']
# half-size of the patches in pixels. rf=25, patches are 50x50
rf = params_movie['rf']
# amounpl.it of overlap between the patches in pixels
stride = params_movie['stride_cnmf']
K = params_movie['K']  # number of neurons expected per patch
gSig = params_movie['gSig']
init_method = params_movie['init_method']
alpha_snmf = params_movie['alpha_snmf']

if params_movie['is_dendrites'] == True:
    if params_movie['init_method'] is not 'sparse_nmf':
        raise Exception('dendritic requires sparse_nmf')
    if params_movie['alpha_snmf'] is None:
        raise Exception('need to set a value for alpha_snmf')
#%%
cnm = cnmf.CNMF(n_processes, k=K, gSig=gSig, merge_thresh=0.8, p=0, dview=dview, Ain=None, rf=rf, stride=stride, memory_fact=1,
                method_init=init_method, alpha_snmf=alpha_snmf, only_init_patch=True, gnb=1, method_deconvolution='oasis')
cnm = cnm.fit(images)

A_tot = cnm.A
C_tot = cnm.C
YrA_tot = cnm.YrA
b_tot = cnm.b
f_tot = cnm.f
sn_tot = cnm.sn

print(('Number of components:' + str(A_tot.shape[-1])))
#%%
pl.figure()
crd = plot_contours(A_tot, Cn, thr=0.9)
#%% DISCARD LOW QUALITY COMPONENT
final_frate = params_movie['final_frate']
r_values_min = .5  # threshold on space consistency
fitness_min = -20  # threshold on time variability
# threshold on time variability (if nonsparse activity)
fitness_delta_min = -20
Npeaks = 10
traces = C_tot + YrA_tot
idx_components, idx_components_bad = cm.components_evaluation.estimate_components_quality(
    traces, Y, A_tot, C_tot, b_tot, f_tot, final_frate=final_frate, Npeaks=Npeaks, r_values_min=r_values_min, fitness_min=fitness_min, fitness_delta_min=fitness_delta_min)

print(('Keeping ' + str(len(idx_components)) +
       ' and discarding  ' + str(len(idx_components_bad))))
#%%
pl.figure()
crd = plot_contours(A_tot.tocsc()[:, idx_components], Cn, thr=0.9)
#%%
A_tot = A_tot.tocsc()[:, idx_components]
C_tot = C_tot[idx_components]
#%% rerun updating the components
cnm = cnmf.CNMF(n_processes, k=A_tot.shape, gSig=gSig, merge_thresh=merge_thresh, p=p, dview=dview, Ain=A_tot, Cin=C_tot,
                f_in=f_tot, rf=None, stride=None, method_deconvolution='oasis')
cnm = cnm.fit(images)
#%%
A, C, b, f, YrA, sn = cnm.A, cnm.C, cnm.b, cnm.f, cnm.YrA, cnm.sn
#%% again recheck quality of components, stricter criteria
final_frate = params_movie['final_frate']
r_values_min = .8
fitness_min = - 30
fitness_delta_min = - 30
Npeaks = 10
traces = C + YrA
idx_components, idx_components_bad = cm.components_evaluation.estimate_components_quality(
    traces, Y, A, C, b, f, final_frate=final_frate, Npeaks=Npeaks, r_values_min=r_values_min, fitness_min=fitness_min, fitness_delta_min=fitness_delta_min)
print(' ***** ')
print((len(traces)))
print((len(idx_components)))
#%% save results
np.savez(os.path.join(os.path.split(fname_new)[0], os.path.split(fname_new)[1][:-4] + 'results_analysis.npz'), Cn=Cn, A=A.todense(
), C=C, b=b, f=f, YrA=YrA, sn=sn, d1=d1, d2=d2, idx_components=idx_components, idx_components_bad=idx_components_bad)
#%%
pl.subplot(1, 2, 1)
crd = plot_contours(A.tocsc()[:, idx_components], Cn, thr=0.9)
pl.subplot(1, 2, 2)
crd = plot_contours(A.tocsc()[:, idx_components_bad], Cn, thr=0.9)
#%%
view_patches_bar(Yr, scipy.sparse.coo_matrix(A.tocsc()[:, idx_components]), C[
    idx_components, :], b, f, dims[0], dims[1], YrA=YrA[idx_components, :], img=Cn)
#%%
view_patches_bar(Yr, scipy.sparse.coo_matrix(A.tocsc()[:, idx_components_bad]), C[
    idx_components_bad, :], b, f, dims[0], dims[1], YrA=YrA[idx_components_bad, :], img=Cn)
#%% STOP CLUSTER and clean up log files
cm.stop_server()

log_files = glob.glob('Yr*_LOG_*')
for log_file in log_files:
    os.remove(log_file)
