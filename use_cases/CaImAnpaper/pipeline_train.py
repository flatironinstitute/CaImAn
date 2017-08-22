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
params_movie = {'fname': ['/mnt/ceph/neuro/labeling/neurofinder.03.00.test/images/final_map/Yr_d1_498_d2_467_d3_1_order_C_frames_2250_.mmap'],
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
fname_new = params_movie['fname'][0]
# %% RUN ANALYSIS
c, dview, n_processes = cm.cluster.setup_cluster(
    backend='local', n_processes=None, single_thread=False)


# %% LOAD MEMMAP FILE
# fname_new='Yr_d1_501_d2_398_d3_1_order_F_frames_369_.mmap'
Yr, dims, T = cm.load_memmap(fname_new)
d1, d2 = dims
images = np.reshape(Yr.T, [T] + list(dims), order='F')
# TODO: needinfo
Y = np.reshape(Yr, dims + (T,), order='F')
m_images = cm.movie(images)

# TODO: show screenshot 10
# %% correlation image
Cn = cm.local_correlations(Y)
Cn[np.isnan(Cn)] = 0
pl.imshow(Cn, cmap='gray', vmax=.95)
# TODO: show screenshot 11
#%%
roi_cons = np.load('/mnt/ceph/neuro/labeling/neurofinder.03.00.test/regions/joined_consensus_active_regions.npy')
print(roi_cons.shape)
pl.imshow(roi_cons.sum(0))
A_in = np.reshape(roi_cons.transpose([1,2,0]),(-1,roi_cons.shape[0]),order = 'F')
pl.figure()
crd = plot_contours(A_in, Cn, thr=.99999)

# %% some parameter settings
# order of the autoregressive fit to calcium imaging in general one (slow gcamps) or two (fast gcamps fast scanning)
p = params_movie['p']
# merging threshold, max correlation allowed
merge_thresh = params_movie['merge_thresh']

# %% Extract spatial and temporal components on patches
t1 = time.time()
# TODO: todocument
# TODO: warnings 3
radius  = np.median(np.sqrt(np.sum(A_in,0)/np.pi))

cnm = cnmf.CNMF(n_processes=1, k=A_in.shape[-1], gSig=[radius,radius], merge_thresh=params_movie['merge_thresh'], p=params_movie['p'], Ain = A_in.astype(np.bool),
                dview=dview, rf=None, stride=None, gnb=params_movie['gnb'], method_deconvolution='oasis',border_pix = 0, low_rank_background = params_movie['low_rank_background']) 
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

# %% save results
np.savez(os.path.join(os.path.split(fname_new)[0], os.path.split(fname_new)[1][:-4] + 'results_analysis.npz'), Cn=Cn,
         A=A,
         C=C, b=b, f=f, YrA=YrA, sn=sn, d1=d1, d2=d2, idx_components=idx_components,
         idx_components_bad=idx_components_bad,
         fitness_raw=fitness_raw, fitness_delta=fitness_delta, r_values=r_values)
# we save it


# %%
# TODO: needinfo
view_patches_bar(Yr, scipy.sparse.coo_matrix(A.tocsc()[:, :]), C[:, :], b, f, dims[0], dims[1],
                 YrA=YrA[:, :], img=Cn)
# %% STOP CLUSTER and clean up log files
# TODO: todocument
cm.stop_server()

log_files = glob.glob('*_LOG_*')
for log_file in log_files:
    os.remove(log_file)

#%% *************************************************************************************************************************#%%

#%% *************************************************************************************************************************#%%

#%% *************************************************************************************************************************#%%

#%% *************************************************************************************************************************#%%

#%% *************************************************************************************************************************#%%

#%% LOAD DATA
params_display = {
    'downsample_ratio': .2,
    'thr_plot': 0.8
}


analysis_file = '/mnt/ceph/neuro/jeremie_analysis/neurofinder.03.00.test/Yr_d1_498_d2_467_d3_1_order_C_frames_2250_._results_analysis.npz'
with np.load(analysis_file) as ld:
    print(ld.keys())
    locals().update(ld) 
    dims_off = d1,d2    
    A = scipy.sparse.coo_matrix(A[()])
    dims = (d1,d2)
    
#%%    
idx_components_r = np.where((r_values >= .85))[0]
idx_components_raw = np.where(fitness_raw < -40)[0]
idx_components_delta = np.where(fitness_delta < -40)[0]    
#idx_and_condition_1 = np.where((r_values >= .65) & ((fitness_raw < -20) | (fitness_delta < -20)) )[0]

idx_components = np.union1d(idx_components_r, idx_components_raw)
idx_components = np.union1d(idx_components, idx_components_delta)
#idx_components = np.union1d(idx_components, idx_and_condition_1)
#idx_components = np.union1d(idx_components, idx_and_condition_2)

#idx_blobs = np.intersect1d(idx_components, idx_blobs)
idx_components_bad = np.setdiff1d(list(range(len(r_values))), idx_components)

print(' ***** ')
print((len(r_values)))
print((len(idx_components)))  
#%%
pl.subplot(1, 2, 1)
crd = plot_contours(A.tocsc()[:, idx_components], Cn, thr=params_display['thr_plot'], vmax = 0.35)
pl.subplot(1, 2, 2)
crd = plot_contours(A.tocsc()[:, idx_components_bad], Cn, thr=params_display['thr_plot'], vmax = 0.35)
#%%
c, dview, n_processes = cm.cluster.setup_cluster(
    backend='local', n_processes=None, single_thread=False)
#%% thredshold components
min_size_neuro = 3*2*np.pi
max_size_neuro = 15**2*np.pi
A_thr = cm.source_extraction.cnmf.spatial.threshold_components(A.tocsc()[:,:].toarray(), dims, medw=None, thr_method='max', maxthr=0.2, nrgthr=0.99, extract_cc=True,
                         se=None, ss=None, dview=dview) 

A_thr = A_thr > 0  
size_neurons = A_thr.sum(0)
A_thr = A_thr[:,(size_neurons>min_size_neuro) & (size_neurons<max_size_neuro)]
print(A_thr.shape)
#%%
crd = plot_contours(scipy.sparse.coo_matrix(A_thr*1.), Cn, thr=.99, vmax = 0.35)
#%%
roi_cons = np.load('/mnt/ceph/neuro/labeling/neurofinder.03.00.test/regions/joined_consensus_active_regions.npy')
print(roi_cons.shape)
pl.imshow(roi_cons.sum(0))
#%%
#%%
pl.figure(figsize=(30,20))
tp_gt, tp_comp, fn_gt, fp_comp, performance_cons_off =  cm.base.rois.nf_match_neurons_in_binary_masks(roi_cons,A_thr[:,:].reshape([dims[0],dims[1],-1],order = 'F').transpose([2,0,1])*1.,thresh_cost=.7, min_dist = 10,
                                                                              print_assignment= False,plot_results=True,Cn=Cn, labels = ['GT','Offline'])
pl.rcParams['pdf.fonttype'] = 42
font = {'family' : 'Myriad Pro',
        'weight' : 'regular',
        'size'   : 20}
pl.rc('font', **font)