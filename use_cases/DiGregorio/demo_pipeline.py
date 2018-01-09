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
import matplotlib.pyplot as plt

from caiman.utils.utils import download_demo
from caiman.utils.visualization import plot_contours, view_patches_bar
from caiman.source_extraction.cnmf import cnmf as cnmf
from caiman.motion_correction import MotionCorrect
from caiman.source_extraction.cnmf.utilities import detrend_df_f
from caiman.components_evaluation import estimate_components_quality_auto
#%% First setup some parameters
# dataset dependent parameters
fname = ['GcC_FOV2.tif']  # filename to be processed
fr = 30                             # imaging rate in frames per second
decay_time = 0.4                    # length of a typical transient in seconds

# motion correction parameters
niter_rig = 1               # number of iterations for rigid motion correction
max_shifts = (25, 25)         # maximum allow rigid shift
splits_rig = 8             # for parallelization split the movies in  num_splits chuncks across time
strides = None#(48, 48)          # start a new patch for pw-rigid motion correction every x pixels
overlaps = None#(24, 24)         # overlap between pathes (size of patch strides+overlaps)
splits_els = None             # for parallelization split the movies in  num_splits chuncks across time
upsample_factor_grid = None    # upsample factor to avoid smearing when merging patches
max_deviation_rigid = None     # maximum deviation allowed for patch with respect to rigid shifts

# parameters for source extraction and deconvolution
p = 2                       # order of the autoregressive system
gnb = 2                     # number of global background components
merge_thresh = 0.8          # merging threshold, max correlation allowed
rf = 50                     # half-size of the patches in pixels. e.g., if rf=25, patches are 50x50
stride_cnmf = 30             # amount of overlap between the patches in pixels
K = 3                       # number of components per patch
gSig = [15, 15]               # expected half size of neurons
init_method = 'greedy_roi'  # initialization method (if analyzing dendritic data using 'sparse_nmf')
is_dendrites = False        # flag for analyzing dendritic data
alpha_snmf = None           # sparsity penalty for dendritic data analysis through sparse NMF

# parameters for component evaluation
min_SNR = 3.5               # signal to noise ratio for accepting a component
rval_thr = 0.9              # space correlation threshold for accepting a component
cnn_thr = 0.8               # threshold for CNN based classifier

#%% download the dataset if it's not present in your folder
if fname[0] in ['Sue_2x_3000_40_-46.tif','demoMovieJ.tif']:
    download_demo(fname[0])
    fname = [os.path.join('example_movies',fname[0])]

#%% play the movie 
# playing the movie using opencv. It requires loading the movie in memory. To 
# close the video press q
m_orig = cm.load_movie_chain(fname[:1])
downsample_ratio = 0.2
offset_mov = -np.min(m_orig[:100])
m_orig.resize(1, 1, downsample_ratio).play(
gain=10, offset=offset_mov, fr=30, magnification=1)

#%% start a cluster for parallel processing
c, dview, n_processes = cm.cluster.setup_cluster(
    backend='local', n_processes=None, single_thread=False)
#%%% MOTION CORRECTION
# first we create a motion correction object with the parameters specified
min_mov = cm.load(fname[0], subindices=range(200)).min() 
        # this will be subtracted from the movie to make it non-negative 

mc = MotionCorrect(fname, min_mov,
                   dview=dview, max_shifts=max_shifts, niter_rig=niter_rig,
                   splits_rig=splits_rig, 
                   strides= strides, overlaps= overlaps, splits_els=splits_els,
                   upsample_factor_grid=upsample_factor_grid,
                   max_deviation_rigid=max_deviation_rigid, 
                   shifts_opencv = True, nonneg_movie = True)
# note that the file is not loaded in memory
#%%
mc.motion_correct_rigid(save_movie=True)
m_els = cm.load(mc.fname_tot_rig[0])
bord_px_els = np.ceil(np.max(np.abs(mc.shifts_rig))).astype(np.int)  
fname_cor = mc.fname_tot_rig
#%% Run piecewise-rigid motion correction using NoRMCorre
non_rigid = False
if non_rigid:
    mc.motion_correct_pwrigid(save_movie=True)
    m_els = cm.load(mc.fname_tot_els)
    bord_px_els = np.ceil(np.maximum(np.max(np.abs(mc.x_shifts_els)),
                                 np.max(np.abs(mc.y_shifts_els)))).astype(np.int)  
    fname_cor = mc.fname_tot_els
    # maximum shift to be used for trimming against NaNs
#%% compare with original movie
cm.concatenate([m_orig.resize(1, 1, downsample_ratio)+offset_mov,
                m_els.resize(1, 1, downsample_ratio)], 
               axis=2).play(fr=60, gain=5, magnification=1, offset=0)  # press q to exit

#%% MEMORY MAPPING
# memory map the file in order 'C'
fnames = fname_cor   # name of the pw-rigidly corrected file.
border_to_0 = bord_px_els     # number of pixels to exclude
fname_new = cm.save_memmap(fnames, base_name='memmap_', order = 'C',
                           border_to_0 = bord_px_els) # exclude borders

# now load the file
Yr, dims, T = cm.load_memmap(fname_new)
d1, d2 = dims
images = np.reshape(Yr.T, [T] + list(dims), order='F') 
    #load frames in python format (T x X x Y)

#%% restart cluster to clean up memory
dview.terminate()
c, dview, n_processes = cm.cluster.setup_cluster(
    backend='local', n_processes=None, single_thread=False)

#%% RUN CNMF ON PATCHES

# First extract spatial and temporal components on patches and combine them
# for this step deconvolution is turned off (p=0)
t1 = time.time()

cnm = cnmf.CNMF(n_processes=1, k=K, gSig=gSig, merge_thresh= merge_thresh, 
                p = 0,  dview=dview, rf=rf, stride=stride_cnmf, memory_fact=1,
                method_init=init_method, alpha_snmf=alpha_snmf, 
                only_init_patch = False, gnb = gnb, border_pix = bord_px_els) 
cnm = cnm.fit(images)

#%% plot contours of found components
Cn = cm.local_correlations(images.transpose(1,2,0))
Cn[np.isnan(Cn)] = 0
plt.figure(); crd = plot_contours(cnm.A, Cn, thr=0.9)
plt.title('Contour plots of found components')


#%% COMPONENT EVALUATION
# the components are evaluated in three ways:
#   a) the shape of each component must be correlated with the data
#   b) a minimum peak SNR is required over the length of a transient
#   c) each shape passes a CNN based classifier

idx_components, idx_components_bad, SNR_comp, SNR_comp_delta, r_values, cnn_preds = \
    estimate_components_quality_auto(images, cnm.A, cnm.C, cnm.b, cnm.f, 
                                     cnm.YrA, fr, decay_time, gSig, dims, 
                                     dview = dview, min_SNR=min_SNR, 
                                     r_values_min = rval_thr, use_cnn = False, 
                                     thresh_cnn_lowest = cnn_thr)

#%% PLOT COMPONENTS

plt.figure();
plt.subplot(121); crd_good = cm.utils.visualization.plot_contours(cnm.A[:,idx_components], Cn, thr=.8, vmax=0.75)
plt.title('Contour plots of accepted components')
plt.subplot(122); crd_bad = cm.utils.visualization.plot_contours(cnm.A[:,idx_components_bad], Cn, thr=.8, vmax=0.75)
plt.title('Contour plots of rejected components')

#%% VIEW TRACES (accepted and rejected)
view_patches_bar(Yr, cnm.A.tocsc()[:, idx_components], cnm.C[idx_components], 
                 cnm.b, cnm.f, dims[0], dims[1],YrA=cnm.YrA[idx_components], 
                 img=Cn)
#%%

view_patches_bar(Yr, cnm.A.tocsc()[:, idx_components_bad], cnm.C[idx_components_bad], 
                 cnm.b, cnm.f, dims[0], dims[1],YrA=cnm.YrA[idx_components_bad], 
                 img=Cn)

#%% RE-RUN seeded CNMF on accepted patches to refine and perform deconvolution 
A_in, C_in, b_in, f_in = cnm.A[:,idx_components], cnm.C[idx_components], cnm.b, cnm.f
cnm2 = cnmf.CNMF(n_processes=1, k=A_in.shape[-1], gSig=gSig, p=p, dview=dview,
                merge_thresh=merge_thresh,  Ain=A_in, Cin=C_in, b_in = b_in,
                f_in=f_in, rf = None, stride = None, gnb = gnb, 
                method_deconvolution='oasis', check_nan = True)

cnm2 = cnm2.fit(images)

#%% Extract DF/F values

F_dff = detrend_df_f(cnm2.A, cnm2.b, cnm2.C, cnm2.f, YrA = cnm2.YrA, 
                      quantileMin=8, frames_window=250)

#%% Show final traces
cnm2.view_patches(Yr, dims = dims, img = Cn)

#%% STOP CLUSTER and clean up log files
cm.stop_server(dview=dview)
log_files = glob.glob('*_LOG_*')
for log_file in log_files:
    os.remove(log_file)

#%% reconstruct denoised movie
denoised = cm.movie(cnm2.A.dot(cnm2.C) + \
                    cnm2.b.dot(cnm2.f)).reshape(dims + (-1,), order='F').transpose([2, 0, 1])
#%%
denoised = cm.movie(cnm2.A.dot(cnm2.C)).reshape(dims + (-1,), order='F').transpose([2, 0, 1])


#%% play along side original data
cm.concatenate([m_els.resize(1, 1, downsample_ratio),
                denoised.resize(1, 1, downsample_ratio)], 
                axis=2).play(fr=60, gain=30, magnification=1, offset=0)  # press q to exit