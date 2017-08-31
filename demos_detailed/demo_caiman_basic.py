# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 18:39:45 2016

@author: Andrea Giovannucci

"""
import numpy as np
import glob
import pylab as pl
import caiman as cm
from caiman.components_evaluation import evaluate_components
from caiman.source_extraction.cnmf import cnmf as cnmf
import os
#%% start cluster
c,dview,n_processes = cm.cluster.setup_cluster(backend = 'local',n_processes = None,single_thread = False)
#%% FOR LOADING ALL TIFF FILES IN A FILE AND SAVING THEM ON A SINGLE MEMORY MAPPABLE FILE
fnames = ['example_movies/demoMovie.tif'] # can actually be a lost of movie to concatenate
add_to_movie=540 # the movie must be positive!!!
downsample_factor=1 # use .2 or .1 if file is large and you want a quick answer
base_name='Yr'
name_new=cm.save_memmap_each(fnames, dview=dview,base_name=base_name, resize_fact=(1, 1, downsample_factor),add_to_movie=add_to_movie)
name_new.sort()
fname_new = cm.save_memmap_join(name_new, base_name='Yr', dview=dview)
#%% LOAD MEMORY MAPPABLE FILE
Yr, dims, T = cm.load_memmap(fname_new)
d1, d2 = dims
images = np.reshape(Yr.T, [T] + list(dims), order='F')
Y = np.reshape(Yr, dims + (T,), order='F')
#%% play movie, press q to quit
play_movie = False
if play_movie:     
    cm.movie(images).play(fr=50,magnification=3,gain=2.)
#%% movie cannot be negative!
if np.min(images)<0:
    raise Exception('Movie too negative, add_to_movie should be larger')
#%% correlation image. From here infer neuron size and density
Cn = cm.movie(images)[:3000].local_correlations(swap_dim=False)
pl.imshow(Cn,cmap='gray')  
#%%
K = 35  # number of neurons expected per patch
gSig = [7, 7]  # expected half size of neurons
merge_thresh = 0.8  # merging threshold, max correlation allowed
p = 2  # order of the autoregressive system
cnm = cnmf.CNMF(n_processes, method_init='greedy_roi', k=K, gSig=gSig, merge_thresh=merge_thresh,
                p=p, dview=dview, Ain=None,method_deconvolution='oasis',rolling_sum = False)
cnm = cnm.fit(images)
A, C, b, f, YrA, sn = cnm.A, cnm.C, cnm.b, cnm.f, cnm.YrA, cnm.sn
#%%
crd = cm.utils.visualization.plot_contours(cnm.A, Cn, thr=0.9)
#%% evaluate the quality of the components
final_frate = 10# approximate frame rate of data
Npeaks = 10
traces = C + YrA
fitness_raw, fitness_delta, erfc_raw, erfc_delta, r_values, significant_samples = \
    evaluate_components(Y, traces, A, C, b, f,final_frate, remove_baseline=True,
                                      N=5, robust_std=False, Npeaks=Npeaks, thresh_C=0.3)

idx_components_r = np.where(r_values >= .85)[0] # filter based on spatial consistency
idx_components_raw = np.where(fitness_raw < -50)[0] # filter based on transient size
idx_components_delta = np.where(fitness_delta < -50)[0] #  filter based on transient derivative size
idx_components = np.union1d(idx_components_r, idx_components_raw)
idx_components = np.union1d(idx_components, idx_components_delta)
idx_components_bad = np.setdiff1d(list(range(len(traces))), idx_components)
print((len(traces)))
print((len(idx_components)))

#%% visualize components
# pl.figure();
pl.subplot(1, 2, 1)
crd = cm.utils.visualization.plot_contours(A.tocsc()[:, idx_components], Cn, thr=0.9)
pl.title('selected components')
pl.subplot(1, 2, 2)
pl.title('discaded components')
crd = cm.utils.visualization.plot_contours(A.tocsc()[:, idx_components_bad], Cn, thr=0.9)
#%% visualize selected components
cm.utils.visualization.view_patches_bar(Yr, A.tocsc()[:, idx_components], C[
                               idx_components, :], b, f, dims[0], dims[1], YrA=YrA[idx_components, :], img=Cn)

#%% STOP CLUSTER and clean up log files   
cm.stop_server()

log_files = glob.glob('Yr*_LOG_*')
for log_file in log_files:
    os.remove(log_file)
