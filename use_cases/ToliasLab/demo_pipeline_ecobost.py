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

import matplotlib
matplotlib.use('Agg')
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

from caiman.components_evaluation import evaluate_components,evaluate_components_CNN

from caiman.tests.comparison import comparison
from caiman.motion_correction import tile_and_correct, motion_correction_piecewise
#%%
# @params params_movie set parameters and create template by RIGID MOTION CORRECTION
isscreen = False
#fls = ['/home/andrea/CaImAn/example_movies/demoMovieJ.tif','/home/andrea/CaImAn/example_movies/demoMovieJ.tif']
fls = glob.glob('/home/andrea/CaImAn/example_movies/12741_1_00003_0000*.tif')
#fls = glob.glob('/tmp/13800_1_0001_00002_0000*.tif')
fls.sort()
fls = fls[:2]
print(fls)
params_movie = {'fname': fls,
               'niter_rig': 1,
               'max_shifts': (10, 10),  # maximum allow rigid shift
               'splits_rig': 28,  # for parallelization split the movies in  num_splits chuncks across time
               # if none all the splits are processed and the movie is saved
               'num_splits_to_process_rig': None,
               # intervals at which patches are laid out for motion correction
               'strides': (48, 48),
               # overlap between pathes (size of patch strides+overlaps)
               'overlaps': (24, 24),
               'splits_els': 28,  # for parallelization split the movies in  num_splits chuncks across time
               # if none all the splits are processed and the movie is saved
               'num_splits_to_process_els': [14, None],
               'upsample_factor_grid': 4,  # upsample factor to avoid smearing when merging patches
               # maximum deviation allowed for patch with respect to rigid
               # shift
               'max_deviation_rigid': 2,
               'p': 1,  # order of the autoregressive system
               'merge_thresh': 0.8,  # merging threshold, max correlation allowed
               'rf': 15,  # half-size of the patches in pixels. rf=25, patches are 50x50
               'stride_cnmf': 6,  # amounpl.it of overlap between the patches in pixels
               'K': 4,  # number of components per patch
               # if dendritic. In this case you need to set init_method to
               # sparse_nmf
               'is_dendrites': False,
               'init_method': 'greedy_roi',
               'gSig': [7, 7],  # expected half size of neurons
               'alpha_snmf': None,  # this controls sparsity
               'final_frate': 30,
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
               'n_chunks': 20,
               'update_background_components': True,# whether to update the background components in the spatial phase
               'low_rank_background': True  #whether to update the using a low rank approximation. In the False case all the nonzero elements of the background components are updated using hals    
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
#%%
if isscreen:
    m_orig.resize(1, 1, downsample_ratio).play(
            gain=3, offset=offset_mov, fr=30, magnification=1)

# %% RUN ANALYSIS
c, dview, n_processes = cm.cluster.setup_cluster(
    backend='local', n_processes=None, single_thread=True)

# %% INITIALIZING
t1 = time.time()
# movie must be mostly positive for this to work
# TODO : document
# setting timer to see how the changement in functions make the code react on a same computer.

min_mov = cm.load(fname[0], subindices=range(400)).min()
mc_list = []
new_templ = None
print('Start motion correction loop')
for each_file in fname:
    print(each_file)
    # TODO: needinfo how the classes works
    mc = MotionCorrect(each_file, min_mov,
                       dview=dview, max_shifts=max_shifts, niter_rig=niter_rig, splits_rig=splits_rig,
                       num_splits_to_process_rig=num_splits_to_process_rig,
                       strides=strides, overlaps=overlaps, splits_els=splits_els,
                       num_splits_to_process_els=num_splits_to_process_els,
                       upsample_factor_grid=upsample_factor_grid, max_deviation_rigid=max_deviation_rigid,
                       shifts_opencv=True, nonneg_movie=True)

    mc.motion_correct_rigid(save_movie=True,template = new_templ)
    print("DONE")
    new_templ = mc.total_template_rig
    m_rig = cm.load(mc.fname_tot_rig)
    bord_px_rig = np.ceil(np.max(mc.shifts_rig)).astype(np.int)

    # TODO : needinfo
    if isscreen:
        pl.imshow(new_templ, cmap='gray')
        pl.pause(.1)

    mc_list.append(mc)
    print(time.time() - t1)


    
        
# we are going to keep this part because it helps the user understand what we need.
# needhelp why it is not the same as in the notebooks ?
# TODO: show screenshot 2,3

# %%
# load motion corrected movie
m_rig = cm.load(mc.fname_tot_rig)
if isscreen:
    m_orig.resize(1,1,.2).play(gain=1, offset=offset_mov, fr=30)

    pl.imshow(mc.total_template_rig, cmap='gray')
    # %% visualize templates
    cm.movie(np.array(mc.templates_rig)).play(
        fr=10, gain=1, magnification=1, offset=offset_mov)
    # %% plot rigid shifts
    pl.close()
    pl.plot(np.vstack([mc.shifts_rig for mc in mc_list]))
    pl.legend(['x shifts', 'y shifts'])
    pl.xlabel('frames')
    pl.ylabel('pixels')
# TODO: show screenshot 9
# %% restart cluster to clean up memory
# TODO: todocument
c, dview, n_processes = cm.cluster.setup_cluster(
    backend='local', n_processes=None, single_thread=False)
# %% save each chunk in F format
t1 = time.time()
fnames = [glob.glob(nm[:-4]+'*.mmap')[0] for nm in params_movie['fname']]
border_to_0 = bord_px_rig
m_els = m_rig
# else:
#   fnames = [mc.fname_tot_els]
#  border_to_0 = bord_px_els

# if you need to crop the borders use slicing
# idx_x=slice(border_nan,-border_nan,None)
# idx_y=slice(border_nan,-border_nan,None)
# idx_xy=(idx_x,idx_y)
idx_xy = None
# TODO: needinfo
add_to_movie = -np.nanmin(m_els) + 1  # movie must be positive
# if you need to remove frames from the beginning of each file
remove_init = 0
# downsample movie in time: use .2 or .1 if file is large and you want a quick answer
downsample_factor = 1
base_name = fname[0].split('/')[-1][:-4]
# TODO: todocument
dview_limited = c[:5]
name_new = cm.save_memmap_each(fnames, dview=dview_limited, base_name=base_name, resize_fact=(
    1, 1, downsample_factor), remove_init=remove_init, idx_xy=idx_xy, add_to_movie=add_to_movie,
                               border_to_0=border_to_0)
name_new.sort()
print(name_new)
print(time.time()-t1)

# %% if multiple files were saved in C format, now put them together in a single large file.
t1 = time.time()
if len(name_new) > 1:
    fname_new = cm.save_memmap_join(
        name_new, base_name='Yr', n_chunks=params_movie['n_chunks'], dview=dview)
else:
    print('One file only, not saving!')
    fname_new = name_new[0]
print(time.time()-t1)

# %% LOAD MEMMAP FILE
# fname_new='Yr_d1_501_d2_398_d3_1_order_F_frames_369_.mmap'
Yr, dims, T = cm.load_memmap(fname_new)
d1, d2 = dims
images = np.reshape(Yr.T, [T] + list(dims), order='F')
# TODO: needinfo
Y = np.reshape(Yr, dims + (T,), order='F')
m_images = cm.movie(images)

# TODO: show screenshot 10
# %%  checks on movies
# computationnally intensive
if isscreen:
    if np.min(images) < 0:
        # TODO: should do this in an automatic fashion with a while loop at the 367 line
        raise Exception('Movie too negative, add_to_movie should be larger')
    if np.sum(np.isnan(images)) > 0:
        # TODO: same here
        raise Exception('Movie contains nan! You did not remove enough borders')
# %% correlation image
# TODO: needinfo it is not the same and not used
# for fff in fname_new:
#    Cn = cm.movie(images[:1000]).local_correlations(eight_neighbours=True,swap_dim=True)
#    #Cn[np.isnan(Cn)] = 0
#    pl.imshow(Cn, cmap='gray', vmax=.35)
# %% correlation image
use_single_files = True
if use_single_files: # faster and more efficient
    Cns = [] 
    for fnn in fnames:
        print(fnn)
#        Cns.append(cm.summary_images.max_correlation_image(np.array(cm.load(fnn)), bin_size=1500, swap_dim=False))        
        Cns.append(cm.load(fnn).local_correlations(swap_dim=False))
        Cn = np.max(Cns,0)    
        if isscreen:

            pl.imshow(Cn, cmap='gray', vmax=.25)
            pl.pause(1)    
        
else:    # slower if you do not have smaller files
    Cn = cm.summary_images.max_correlation_image(Y, bin_size=1500)

Cn[np.isnan(Cn)] = 0

if isscreen:
    
    pl.imshow(Cn, cmap='gray', vmax=.25)
# TODO: show screenshot 11

# %% some parameter settings
# order of the autoregressive fit to calcium imaging in general one (slow gcamps) or two (fast gcamps fast scanning)
p = params_movie['p']
# merging threshold, max correlation allowed
merge_thresh = params_movie['merge_thresh']
# half-size of the patches in pixels. rf=25, patches are 50x50
rf = params_movie['rf']
# amounpl.it of overlap between the patches in pixels
stride_cnmf = params_movie['stride_cnmf']
# number of components per patch
K = params_movie['K']
# if dendritic. In this case you need to set init_method to sparse_nmf
is_dendrites = params_movie['is_dendrites']
# iinit method can be greedy_roi for round shapes or sparse_nmf for denritic data
init_method = params_movie['init_method']
# expected half size of neurons
gSig = params_movie['gSig']
# this controls sparsity
alpha_snmf = params_movie['alpha_snmf']
# frame rate of movie (even considering eventual downsampling)
final_frate = params_movie['final_frate']

if params_movie['is_dendrites'] == True:
    if params_movie['init_method'] is not 'sparse_nmf':
        raise Exception('dendritic requires sparse_nmf')
    if params_movie['alpha_snmf'] is None:
        raise Exception('need to set a value for alpha_snmf')
# %% Extract spatial and temporal components on patches
t1 = time.time()
border_pix = 0 # if motion correction introduces problems at the border remove pixels from border
# TODO: todocument
# TODO: warnings 3
cnm = cnmf.CNMF(n_processes=1, k=K, gSig=gSig, merge_thresh=params_movie['merge_thresh'], p=params_movie['p'],
                dview=dview, rf=rf, stride=stride_cnmf, memory_fact=1,
                method_init=init_method, alpha_snmf=alpha_snmf, only_init_patch=params_movie['only_init_patch'],
                gnb=params_movie['gnb'], method_deconvolution='oasis',border_pix = border_pix, low_rank_background = params_movie['low_rank_background']) 
cnm = cnm.fit(images)

A_tot = cnm.A
C_tot = cnm.C
YrA_tot = cnm.YrA
b_tot = cnm.b
f_tot = cnm.f
sn_tot = cnm.sn
print(('Number of components:' + str(A_tot.shape[-1])))
print(time.time()-t1)

# %%
if isscreen:
    pl.figure()
    # TODO: show screenshot 12`
    # TODO : change the way it is used
    crd = plot_contours(A_tot, Cn, thr=params_display['thr_plot'])
# %% DISCARD LOW QUALITY COMPONENT
t1 = time.time()
final_frate = params_movie['final_frate']
r_values_min = params_movie['r_values_min_patch']  # threshold on space consistency
fitness_min = params_movie['fitness_delta_min_patch']  # threshold on time variability
# threshold on time variability (if nonsparse activity)
fitness_delta_min = params_movie['fitness_delta_min_patch']
Npeaks = params_movie['Npeaks']
traces = C_tot + YrA_tot
# TODO: todocument
idx_components, idx_components_bad = estimate_components_quality(
    traces, Y, A_tot, C_tot, b_tot, f_tot, final_frate=final_frate, Npeaks=Npeaks, r_values_min=r_values_min,
    fitness_min=fitness_min, fitness_delta_min=fitness_delta_min)
print(('Keeping ' + str(len(idx_components)) +
       ' and discarding  ' + str(len(idx_components_bad))))
print(time.time()-t1)

# %%
# TODO: show screenshot 13
if isscreen:
    pl.figure()
    crd = plot_contours(A_tot.tocsc()[:, idx_components], Cn, thr=params_display['thr_plot'])
# %%
A_tot = A_tot.tocsc()[:, idx_components]
C_tot = C_tot[idx_components]
# %% rerun updating the components to refine
t1 = time.time()
cnm = cnmf.CNMF(n_processes=1, k=A_tot.shape, gSig=gSig, merge_thresh=merge_thresh, p=p, dview=dview, Ain=A_tot,
                Cin=C_tot, b_in = b_tot,
                f_in=f_tot, rf=None, stride=None, method_deconvolution='oasis',gnb = params_movie['gnb'],
                low_rank_background = params_movie['low_rank_background'], update_background_components = params_movie['update_background_components'],check_nan = True)

cnm = cnm.fit(images)
print(time.time()-t1)

# %%
A, C, b, f, YrA, sn = cnm.A, cnm.C, cnm.b, cnm.f, cnm.YrA, cnm.sn
# %% again recheck quality of components, stricter criteria
final_frate = params_movie['final_frate']
r_values_min = params_movie['r_values_min_full']  # threshold on space consistency
fitness_min = params_movie['fitness_min_full']  # threshold on time variability
# threshold on time variability (if nonsparse activity)
fitness_delta_min = params_movie['fitness_delta_min_full']
Npeaks = params_movie['Npeaks']
traces = C + YrA
idx_components, idx_components_bad, fitness_raw, fitness_delta, r_values = estimate_components_quality(
    traces, Y, A, C, b, f, final_frate=final_frate, Npeaks=Npeaks, r_values_min=r_values_min, fitness_min=fitness_min,
    fitness_delta_min=fitness_delta_min, return_all=True)
print(' ***** ')
print((len(traces)))
print((len(idx_components)))
# %% save results
np.savez(os.path.join(os.path.split(fname_new)[0], os.path.split(fname_new)[1][:-4] + 'results_analysis.npz'), Cn=Cn,
         A=A,
         C=C, b=b, f=f, YrA=YrA, sn=sn, d1=d1, d2=d2, idx_components=idx_components,
         idx_components_bad=idx_components_bad,
         fitness_raw=fitness_raw, fitness_delta=fitness_delta, r_values=r_values)
# we save it
# %%
# TODO: show screenshot 14
if isscreen:
    pl.subplot(1, 2, 1)
    crd = plot_contours(A.tocsc()[:, idx_components], Cn, thr=params_display['thr_plot'])
    pl.subplot(1, 2, 2)
    crd = plot_contours(A.tocsc()[:, idx_components_bad], Cn, thr=params_display['thr_plot'])
    # %%
    # TODO: needinfo
    view_patches_bar(Yr, scipy.sparse.coo_matrix(A.tocsc()[:, idx_components]), C[idx_components, :], b, f, dims[0], dims[1],
                     YrA=YrA[idx_components, :], img=Cn)
    # %%
    view_patches_bar(Yr, scipy.sparse.coo_matrix(A.tocsc()[:, idx_components_bad]), C[idx_components_bad, :], b, f, dims[0],
                     dims[1], YrA=YrA[idx_components_bad, :], img=Cn)
# %% STOP CLUSTER and clean up log files
# TODO: todocument
cm.stop_server()

log_files = glob.glob('*_LOG_*')
for log_file in log_files:
    os.remove(log_file)

