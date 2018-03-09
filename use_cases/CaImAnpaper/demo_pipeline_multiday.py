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
        print("Running under iPython")
        # this is used for debugging purposes only. allows to reload classes
        # when changed
        get_ipython().magic('load_ext autoreload')
        get_ipython().magic('autoreload 2')
except NameError:
    pass

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
from caiman.summary_images import correlation_image_ecobost
#%% First setup some parameters

# dataset dependent parameters
fname = glob.glob('/mnt/ceph/neuro/Sue/k53/20160530/*0[0-9].tif') +\
        glob.glob('/mnt/ceph/neuro/Sue/k53/20160531/*0[0-9].tif') +\
        glob.glob('/mnt/ceph/neuro/Sue/k53/20160603/*0[0-9].tif') +\
        glob.glob('/mnt/ceph/neuro/Sue/k53/20160606/*0[0-9].tif') # filename to be processed
fname = glob.glob('/mnt/ceph/neuro/Sue/k53/20160607/*0[0-9].tif') +\
        glob.glob('/mnt/ceph/neuro/Sue/k53/20160608/*0[0-9].tif') # filename to be processed
fname.sort()
fr = 30                             # imaging rate in frames per second
decay_time = 0.4                    # length of a typical transient in seconds

# motion correction parameters
niter_rig = 1               # number of iterations for rigid motion correction
max_shifts = (12, 12)         # maximum allow rigid shift
# for parallelization split the movies in  num_splits chuncks across time
splits_rig = 28
# start a new patch for pw-rigid motion correction every x pixels
strides = (96, 96)
# overlap between pathes (size of patch strides+overlaps)
overlaps = (32, 32)
# for parallelization split the movies in  num_splits chuncks across time
splits_els = 28
upsample_factor_grid = 4    # upsample factor to avoid smearing when merging patches
# maximum deviation allowed for patch with respect to rigid shifts
max_deviation_rigid = 2

# parameters for source extraction and deconvolution
p = 1                       # order of the autoregressive system
gnb = 2                     # number of global background components
merge_thresh = 0.8          # merging threshold, max correlation allowed
# half-size of the patches in pixels. e.g., if rf=25, patches are 50x50
rf = 20
stride_cnmf = 10             # amount of overlap between the patches in pixels
K = 7                       # number of components per patch
gSig = [7, 7]               # expected half size of neurons
# initialization method (if analyzing dendritic data using 'sparse_nmf')
init_method = 'greedy_roi'
is_dendrites = False        # flag for analyzing dendritic data
# sparsity penalty for dendritic data analysis through sparse NMF
alpha_snmf = None

# parameters for component evaluation
min_SNR = 2.5               # signal to noise ratio for accepting a component
rval_thr = 0.85              # space correlation threshold for accepting a component
cnn_thr = 0.95               # threshold for CNN based classifier

#%% download the dataset if it's not present in your folder
if fname[0] in ['Sue_2x_3000_40_-46.tif', 'demoMovieJ.tif']:
    download_demo(fname[0])
    fname = [os.path.join('example_movies', fname[0])]

#%% play the movie
# playing the movie using opencv. It requires loading the movie in memory. To
# close the video press q
m_orig = cm.load_movie_chain(fname[:1])
downsample_ratio = 0.2
offset_mov = -np.min(m_orig[:100])
m_orig.resize(1, 1, downsample_ratio).play(
    gain=10, offset=offset_mov, fr=30, magnification=2)
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
                   strides=strides, overlaps=overlaps, splits_els=splits_els,
                   upsample_factor_grid=upsample_factor_grid,
                   max_deviation_rigid=max_deviation_rigid,
                   shifts_opencv=True, nonneg_movie=True)
#%% Run piecewise-rigid motion correction using NoRMCorre
mc.motion_correct_rigid(save_movie=True, template = template)
#%% look at the movie of all the templates to assess if there is deformations
# or changes in the FOV
mov_templates = np.array(mc.templates_rig)
min_mov_templates = mov_templates.min()
cm.movie(mov_templates)[::10].play(gain = 10., offset = -min_mov_templates, magnification = 2)
##%% compute overall deformable template
## this will be subtracted from the movie to make it non-negative
#mc_templ = MotionCorrect(mov_templates, min_mov_templates,
#                   dview=dview, max_shifts=max_shifts, niter_rig=None,
#                   splits_rig=None,
#                   strides=strides, overlaps=overlaps, splits_els=10,
#                   upsample_factor_grid=upsample_factor_grid,
#                   max_deviation_rigid=max_deviation_rigid,
#                   shifts_opencv=True, nonneg_movie=True)
##%%
#mc_templ.motion_correct_pwrigid(save_movie=True, template = mov_templates[len(mov_templates)//2])
##%%
#cm.load(mc_templ.fname_tot_els)[::100].play(gain = 5., offset = -min_mov_templates, magnification = 2)
#%%
names_tots = glob.glob('/mnt/ceph/neuro/Sue/k53/20160608/*_F_*.mmap')
names_tots.sort()
print(names_tots)
#%%
dview.terminate()
c, dview, n_processes = cm.cluster.setup_cluster(
    backend='local', n_processes=10, single_thread=False)

Cn = correlation_image_ecobost(names_tots,dview = dview)


# maximum shift to be used for trimming against NaNs
#%% MEMORY MAPPING
# memory map the file in order 'C'
fnames = names_tots   # name of the pw-rigidly corrected file.
border_to_0 = 6     # number of pixels to exclude
fname_new = cm.save_memmap(fnames, base_name='memmap_', order='C',
                           border_to_0=border_to_0, dview = dview)  # exclude borders

#%% now load the file
Yr, dims, T = cm.load_memmap(fname_new)
d1, d2 = dims
images = np.reshape(Yr.T, [T] + list(dims), order='F')
# load frames in python format (T x X x Y)

#%% restart cluster to clean up memory
dview.terminate()
c, dview, n_processes = cm.cluster.setup_cluster(
    backend='local', n_processes=None, single_thread=False)

#%% RUN CNMF ON PATCHES
# First extract spatial and temporal components on patches and combine them
# for this step deconvolution is turned off (p=0)
t1 = time.time()

cnm = cnmf.CNMF(n_processes=1, k=K, gSig=gSig, merge_thresh=merge_thresh,
                p=0, dview=dview, rf=rf, stride=stride_cnmf, memory_fact=1,
                method_init=init_method, alpha_snmf=alpha_snmf,
                only_init_patch=False, gnb=gnb, border_pix=border_to_0)
cnm = cnm.fit(images)

#%% plot contours of found components
#Cn = cm.local_correlations(images.transpose(1, 2, 0))
#Cn[np.isnan(Cn)] = 0
plt.figure()
crd = plot_contours(cnm.A, Cn, thr=0.9)
plt.title('Contour plots of found components')


#%% COMPONENT EVALUATION
# the components are evaluated in three ways:
#   a) the shape of each component must be correlated with the data
#   b) a minimum peak SNR is required over the length of a transient
#   c) each shape passes a CNN based classifier

idx_components, idx_components_bad, SNR_comp, r_values, cnn_preds = \
    estimate_components_quality_auto(images, cnm.A, cnm.C, cnm.b, cnm.f,
                                     cnm.YrA, fr, decay_time, gSig, dims,
                                     dview=dview, min_SNR=min_SNR,
                                     r_values_min=rval_thr, use_cnn=False,
                                     thresh_cnn_lowest=cnn_thr)

#%% PLOT COMPONENTS
plt.figure()
plt.subplot(121)
crd_good = cm.utils.visualization.plot_contours(
    cnm.A[:, idx_components], Cn, thr=.8, vmax=0.75)
plt.title('Contour plots of accepted components')
plt.subplot(122)
crd_bad = cm.utils.visualization.plot_contours(
    cnm.A[:, idx_components_bad], Cn, thr=.8, vmax=0.75)
plt.title('Contour plots of rejected components')

#%% VIEW TRACES (accepted and rejected)

view_patches_bar(Yr, cnm.A.tocsc()[:, idx_components], cnm.C[idx_components],
                 cnm.b, cnm.f, dims[0], dims[1], YrA=cnm.YrA[idx_components],
                 img=Cn)

view_patches_bar(Yr, cnm.A.tocsc()[:, idx_components_bad], cnm.C[idx_components_bad],
                 cnm.b, cnm.f, dims[0], dims[1], YrA=cnm.YrA[idx_components_bad],
                 img=Cn)

#%% RE-RUN seeded CNMF on accepted patches to refine and perform deconvolution
A_in, C_in, b_in, f_in = cnm.A[:,
                               :], cnm.C[:], cnm.b, cnm.f
cnm2 = cnmf.CNMF(n_processes=1, k=A_in.shape[-1], gSig=gSig, p=p, dview=dview,
                 merge_thresh=merge_thresh, Ain=A_in, Cin=C_in, b_in=b_in,
                 f_in=f_in, rf=None, stride=None, gnb=gnb,
                 method_deconvolution='oasis', check_nan=True)

cnm2 = cnm2.fit(images)
#%%
min_SNR = 2.5               # signal to noise ratio for accepting a component
rval_thr = 0.85              # space correlation threshold for accepting a component
cnn_thr = 0.95               # threshold for CNN based classifier
thresh_cnn_lowest = 0.1
idx_components, idx_components_bad, SNR_comp, r_values, cnn_preds = \
    estimate_components_quality_auto(images, cnm2.A, cnm2.C, cnm2.b, cnm2.f,
                                     cnm2.YrA, fr, decay_time, gSig, dims,
                                     dview=dview, min_SNR=min_SNR,
                                     r_values_min=rval_thr, use_cnn=True,
                                     thresh_cnn_min=cnn_thr, thresh_cnn_lowest = thresh_cnn_lowest)
#%%
np.savez(os.path.join(os.path.split(fname_new)[0], os.path.split(fname_new)[1][:-4] + 'results_analysis.npz'), Cn=Cn, fname_new=fname_new,
                     A=cnm2.A,
                     C=cnm2.C, b=cnm2.b, f=cnm2.f, YrA=cnm2.YrA, sn=cnm2.sn, d1=d1, d2=d2, idx_components=idx_components,
                     idx_components_bad=idx_components_bad,
                     SNR_comp=SNR_comp, cnn_preds=cnn_preds, r_values=r_values)

#%%
plt.figure()
plt.subplot(121)
crd_good = cm.utils.visualization.plot_contours(
    cnm2.A[:, idx_components], Cn, thr=.96, vmax=0.5)
plt.title('Contour plots of accepted components')
plt.subplot(122)
crd_bad = cm.utils.visualization.plot_contours(
    cnm2.A[:, idx_components_bad], Cn, thr=.96, vmax=0.5)
plt.title('Contour plots of rejected components')
#%%

#%% Extract DF/F values

F_dff = detrend_df_f(cnm2.A[:, idx_components], cnm2.b, cnm2.C[idx_components], cnm2.f, YrA=cnm2.YrA[idx_components],
                     quantileMin=8, frames_window=250)

