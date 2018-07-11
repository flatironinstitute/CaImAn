#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on Wed Feb 24 18:39:45 2016

@author: Andrea Giovannucci

For explanation consult at https://github.com/agiovann/Constrained_NMF/releases/download/v0.4-alpha/Patch_demo.zip
and https://github.com/agiovann/Constrained_NMF

"""

from __future__ import division
from __future__ import print_function
#%%
from builtins import str
from builtins import range
from past.utils import old_div
try:
    if __IPYTHON__:
        # this is used for debugging purposes only. allows to reload classes when changed
        get_ipython().magic('load_ext autoreload')
        get_ipython().magic('autoreload 2')
except NameError:
    print('Not launched under iPython')

import sys
import numpy as np
from time import time
from scipy.sparse import coo_matrix
import psutil
import glob
import os
import scipy
from ipyparallel import Client
import matplotlib as mpl
# mpl.use('Qt5Agg')


import pylab as pl
pl.ion()
#%%
import caiman as cm
from caiman.components_evaluation import evaluate_components
from caiman.utils.visualization import plot_contours, view_patches_bar
from caiman.base.rois import extract_binary_masks_blob
import caiman.source_extraction.cnmf as cnmf
#%%
# frame rate in Hz
final_frate = 10

#%%
# backend='SLURM'
backend = 'local'
if backend == 'SLURM':
    n_processes = np.int(os.environ.get('SLURM_NPROCS'))
else:
    # roughly number of cores on your machine minus 1
    n_processes = np.maximum(np.int(psutil.cpu_count()), 1)
print(('using ' + str(n_processes) + ' processes'))
#%% start cluster for efficient computation
single_thread = False

if single_thread:
    dview = None
else:
    try:
        c.close()
    except:
        print('C was not existing, creating one')
    print("Stopping  cluster to avoid unnencessary use of memory....")
    sys.stdout.flush()
    if backend == 'SLURM':
        try:
            cm.stop_server(is_slurm=True)
        except:
            print('Nothing to stop')
        slurm_script = '/mnt/xfs1/home/agiovann/SOFTWARE/Constrained_NMF/SLURM/slurmStart.sh'
        cm.start_server(slurm_script=slurm_script)
        pdir, profile = os.environ['IPPPDIR'], os.environ['IPPPROFILE']
        c = Client(ipython_dir=pdir, profile=profile)
    else:
        cm.stop_server()
        cm.start_server()
        c = Client()

    print(('Using ' + str(len(c)) + ' processes'))
    dview = c[:len(c)]
#%% FOR LOADING ALL TIFF FILES IN A FILE AND SAVING THEM ON A SINGLE MEMORY MAPPABLE FILE
fnames = []
base_folder = './example_movies/'  # folder containing the demo files
for file in glob.glob(os.path.join(base_folder, '*.tif')):
    if file.endswith("ie.tif"):
        fnames.append(os.path.abspath(file))
fnames.sort()
if len(fnames) == 0:
    raise Exception("Could not find any tiff file")
print(fnames)
fnames = fnames
#%%
# idx_x=slice(12,500,None)
# idx_y=slice(12,500,None)
# idx_xy=(idx_x,idx_y)
downsample_factor = 1  # use .2 or .1 if file is large and you want a quick answer
idx_xy = None
base_name = 'Yr'
name_new = cm.save_memmap_each(fnames, dview=dview, base_name=base_name, resize_fact=(
    1, 1, downsample_factor), remove_init=0, idx_xy=idx_xy)
name_new.sort()
print(name_new)
#%%
name_new = cm.save_memmap_each(fnames, dview=dview, base_name='Yr', resize_fact=(
    1, 1, 1), remove_init=0, idx_xy=None)
name_new.sort()
#%%
fname_new = cm.save_memmap_join(
    name_new, base_name='Yr', n_chunks=12, dview=dview)
#%%
Yr, dims, T = cm.load_memmap(fname_new)
d1, d2 = dims
Y = np.reshape(Yr, dims + (T,), order='F')
#%% visualize correlation image
Cn = cm.local_correlations(Y)
pl.imshow(Cn, cmap='gray')
#%%
rf = 10  # half-size of the patches in pixels. rf=25, patches are 50x50
stride = 2  # amounpl.it of overlap between the patches in pixels
K = 3  # number of neurons expected per patch
gSig = [7, 7]  # expected half size of neurons
merge_thresh = 0.8  # merging threshold, max correlation allowed
p = 2  # order of the autoregressive system
memory_fact = 1  # unitless number accounting how much memory should be used. You will need to try different values to see which one would work the default is OK for a 16 GB system
save_results = True
#%% RUN ALGORITHM ON PATCHES
options_patch = cnmf.utilities.CNMFParams(dims, K=K, gSig=gSig, ssub=1, tsub=4, p=0, thr=merge_thresh)
A_tot, C_tot, YrA_tot, b, f, sn_tot, optional_outputs = cnmf.map_reduce.run_CNMF_patches(fname_new, (d1, d2, T), options_patch, rf=rf, stride=stride,
                                                                                         dview=dview, memory_fact=memory_fact, gnb=1)
print(('Number of components:' + str(A_tot.shape[-1])))
#%%
if save_results:
    np.savez('results_analysis_patch.npz', A_tot=A_tot.todense(),
             C_tot=C_tot, sn_tot=sn_tot, d1=d1, d2=d2, b=b, f=f)
#%% if you have many components this might take long!
pl.figure()
crd = plot_contours(A_tot, Cn, thr=0.9)
# %% set parameters for full field of view analysis
options = cnmf.utilities.CNMFParams(dims, K=A_tot.shape[-1], gSig=gSig, p=0, thr=merge_thresh)
pix_proc = np.minimum(np.int((d1 * d2) / n_processes / (old_div(T, 2000.))),
                      np.int(old_div((d1 * d2), n_processes)))  # regulates the amount of memory used
options['spatial_params']['n_pixels_per_process'] = pix_proc
options['temporal_params']['n_pixels_per_process'] = pix_proc
#%% merge spatially overlaping and temporally correlated components
A_m, C_m, nr_m, merged_ROIs, S_m, bl_m, c1_m, sn_m, g_m = cnmf.merging.merge_components(Yr, A_tot, [], np.array(C_tot), [], np.array(
    C_tot), [], options['temporal_params'], options['spatial_params'], dview=dview, thr=options['merging']['thr'], mx=np.Inf)
#%% update temporal to get Y_r
options['temporal_params']['p'] = 0
# change ifdenoised traces time constant is wrong
options['temporal_params']['fudge_factor'] = 0.96
options['temporal_params']['backend'] = 'ipyparallel'
C_m, A_m, b, f_m, S_m, bl_m, c1_m, neurons_sn_m, g2_m, YrA_m = cnmf.temporal.update_temporal_components(
    Yr, A_m, b, C_m, f, dview=dview, bl=None, c1=None, sn=None, g=None, **options['temporal_params'])

#%% get rid of evenrually noisy components.
# But check by visual inspection to have a feeling fot the threshold. Try to be loose, you will be able to get rid of more of them later!
tB = np.minimum(-2, np.floor(-5. / 30 * final_frate))
tA = np.maximum(5, np.ceil(25. / 30 * final_frate))
Npeaks = 10
traces = C_m + YrA_m
#        traces_a=traces-scipy.ndimage.percentile_filter(traces,8,size=[1,np.shape(traces)[-1]/5])
#        traces_b=np.diff(traces,axis=1)
fitness_raw, fitness_delta, erfc_raw, erfc_delta, r_values, significant_samples =\
    evaluate_components(Y, traces, A_m, C_m, b, f_m,
                        remove_baseline=True, N=5, robust_std=False, Athresh=0.1, Npeaks=Npeaks, tB=tB, tA=tA, thresh_C=0.3)

idx_components_r = np.where(r_values >= .5)[0]
idx_components_raw = np.where(fitness_raw < -20)[0]
idx_components_delta = np.where(fitness_delta < -10)[0]


idx_components = np.union1d(idx_components_r, idx_components_raw)
idx_components = np.union1d(idx_components, idx_components_delta)
idx_components_bad = np.setdiff1d(list(range(len(traces))), idx_components)

print(' ***** ')
print((len(traces)))
print((len(idx_components)))

#%%
A_m = A_m[:, idx_components]
C_m = C_m[idx_components, :]

#%% display components  DO NOT RUN IF YOU HAVE TOO MANY COMPONENTS
pl.figure()
crd = plot_contours(A_m, Cn, thr=0.9)
#%%
print(('Number of components:' + str(A_m.shape[-1])))
#%% UPDATE SPATIAL OCMPONENTS
t1 = time()
A2, b2, C2, f = cnmf.spatial.update_spatial_components(
    Yr, C_m, f, A_m, sn=sn_tot, dview=dview, dims=dims, **options['spatial_params'])
print((time() - t1))
#%% UPDATE TEMPORAL COMPONENTS
options['temporal_params']['p'] = p
# change ifdenoised traces time constant is wrong
options['temporal_params']['fudge_factor'] = 0.96
C2, A2, b2, f2, S2, bl2, c12, neurons_sn2, g21, YrA = cnmf.temporal.update_temporal_components(
    Yr, A2, b2, C2, f, dview=dview, bl=None, c1=None, sn=None, g=None, **options['temporal_params'])
#%% stop server and remove log files
log_files = glob.glob('Yr*_LOG_*')
for log_file in log_files:
    os.remove(log_file)
#%% order components according to a quality threshold and only select the ones wiht qualitylarger than quality_threshold.
B = np.minimum(-2, np.floor(-5. / 30 * final_frate))
tA = np.maximum(5, np.ceil(25. / 30 * final_frate))
Npeaks = 10
traces = C2 + YrA
#        traces_a=traces-scipy.ndimage.percentile_filter(traces,8,size=[1,np.shape(traces)[-1]/5])
#        traces_b=np.diff(traces,axis=1)
fitness_raw, fitness_delta, erfc_raw, erfc_delta, r_values, significant_samples = evaluate_components(
    Y, traces, A2, C2, b2, f2, remove_baseline=True, N=5, robust_std=False, Athresh=0.1, Npeaks=Npeaks, tB=tB, tA=tA, thresh_C=0.3)

idx_components_r = np.where(r_values >= .6)[0]
idx_components_raw = np.where(fitness_raw < -60)[0]
idx_components_delta = np.where(fitness_delta < -20)[0]


min_radius = gSig[0] - 2
masks_ws, idx_blobs, idx_non_blobs = extract_binary_masks_blob(
    A2.tocsc(), min_radius, dims, num_std_threshold=1,
    minCircularity=0.6, minInertiaRatio=0.2, minConvexity=.8)


idx_components = np.union1d(idx_components_r, idx_components_raw)
idx_components = np.union1d(idx_components, idx_components_delta)
idx_blobs = np.intersect1d(idx_components, idx_blobs)
idx_components_bad = np.setdiff1d(list(range(len(traces))), idx_components)

print(' ***** ')
print((len(traces)))
print((len(idx_components)))
print((len(idx_blobs)))
#%% visualize components
# pl.figure();
pl.subplot(1, 3, 1)
crd = plot_contours(A2.tocsc()[:, idx_components], Cn, thr=0.9)
pl.subplot(1, 3, 2)
crd = plot_contours(A2.tocsc()[:, idx_blobs], Cn, thr=0.9)
pl.subplot(1, 3, 3)
crd = plot_contours(A2.tocsc()[:, idx_components_bad], Cn, thr=0.9)
#%%
view_patches_bar(Yr, scipy.sparse.coo_matrix(A2.tocsc()[
                 :, idx_components]), C2[idx_components, :], b2, f2, dims[0], dims[1], YrA=YrA[idx_components, :], img=Cn)
#%%
view_patches_bar(Yr, scipy.sparse.coo_matrix(A2.tocsc()[
                 :, idx_components_bad]), C2[idx_components_bad, :], b2, f2, dims[0], dims[1], YrA=YrA[idx_components_bad, :], img=Cn)
#%% STOP CLUSTER
pl.close()
if not single_thread:
    c.close()
    cm.stop_server()
