# -*- coding: utf-8 -*-
"""
Stripped demo for running the CNMF source extraction algorithm with CaImAn and
evaluation the components. The analysis can be run either in the whole FOV 
or 

For a complete pipeline (including motion correction) check demo_pipeline.py

Data courtesy of W. Yang, D. Peterka and R. Yuste (Columbia University)

@authors: @agiovann and @epnev

"""
from __future__ import print_function
from builtins import range
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')
import numpy as np
import glob
import matplotlib.pyplot as plt
import caiman as cm
from caiman.components_evaluation import estimate_components_quality_auto
from caiman.source_extraction.cnmf import cnmf as cnmf
import os

#%% start a cluster

c, dview, n_processes =\
     cm.cluster.setup_cluster(backend = 'local', n_processes = None, 
                              single_thread = False)

#%% save files to be processed 

fnames = ['example_movies/demoMovie.tif']                                       
    # location of dataset  (can actually be a list of filed to be concatenated)
add_to_movie = -np.min(cm.load(fnames[0],subindices=range(200))).astype(float)  
    # determine minimum value on a small chunk of data
add_to_movie = np.maximum(add_to_movie,0)                                       
    # if minimum is negative subtract to make the data non-negative
base_name='Yr'
name_new=cm.save_memmap_each(fnames, dview=dview, base_name=base_name,
                             add_to_movie=add_to_movie)
name_new.sort()
fname_new = cm.save_memmap_join(name_new, base_name='Yr', dview=dview)
#%% LOAD MEMORY MAPPABLE FILE
Yr, dims, T = cm.load_memmap(fname_new)
d1, d2 = dims
images = np.reshape(Yr.T, [T] + list(dims), order='F')

#%% play movie, press q to quit
play_movie = False
if play_movie:     
    cm.movie(images[1400:]).play(fr=50,magnification=4,gain=3.)
    
#%% correlation image. From here infer neuron size and density
Cn = cm.movie(images).local_correlations(swap_dim=False)
plt.imshow(Cn,cmap='gray')  
plt.title('Correlation Image')

#%% set up some parameters

is_patches = True      # flag for processing in patches or not

if is_patches:          # PROCESS IN PATCHES AND THEN COMBINE
    rf = 10             # half size of each patch
    stride = 4          # overlap between patches
    K = 3               # number of components in each patch
else:                   # PROCESS THE WHOLE FOV AT ONCE
    rf = None           # setting these parameters to None
    stride = None       # will run CNMF on the whole FOV
    K = 30              # number of neurons expected (in the whole FOV)
    
gSig = [5, 5]           # expected half size of neurons
merge_thresh = 0.80     # merging threshold, max correlation allowed
p = 2                   # order of the autoregressive system
gnb = 2                 # global background order


#%% Now RUN CNMF
cnm = cnmf.CNMF(n_processes, method_init='greedy_roi', k=K, gSig=gSig, 
                merge_thresh=merge_thresh, p=p, dview=dview, gnb = gnb,
                rf = rf, stride = stride, rolling_sum = False)
cnm = cnm.fit(images)

#%% plot contour plots of components

plt.figure();
crd = cm.utils.visualization.plot_contours(cnm.A, Cn, thr=0.9)
plt.title('Contour plots of components')

#%% COMPONENT EVALUATION
# the components are evaluated in three ways:
#   a) the shape of each component must be correlated with the data
#   b) a minimum peak SNR is required over the length of a transient
#   c) each shape passes a CNN based classifier (this will pick up only neurons
#           and filter out active processes)

fr = 10             # approximate frame rate of data
decay_time = 5.0    # length of transient
min_SNR = 2.5       # peak SNR for accepted components (if above this, acept)
rval_thr = 0.90     # space correlation threshold (if above this, accept)
use_cnn = True     # use the CNN classifier
min_cnn_thr = 0.10  # if cnn classifier predicts below this value, reject

idx_components, idx_components_bad, SNR_comp, r_values, cnn_preds = \
   estimate_components_quality_auto(images, cnm.A, cnm.C, cnm.b, cnm.f, 
                                    cnm.YrA, fr, decay_time, gSig, dims, 
                                    dview = dview, min_SNR = min_SNR, 
                                    r_values_min = rval_thr, use_cnn = use_cnn,
                                    thresh_cnn_lowest = min_cnn_thr)


#%% visualize selected and rejected components
plt.figure();
plt.subplot(1, 2, 1)
cm.utils.visualization.plot_contours(cnm.A[:, idx_components], Cn, thr=0.9);
plt.title('Selected components')
plt.subplot(1, 2, 2)
plt.title('Discaded components')
cm.utils.visualization.plot_contours(cnm.A[:, idx_components_bad], Cn, thr=0.9);

#%% visualize selected components
cm.utils.visualization.view_patches_bar(Yr, cnm.A.tocsc()[:, idx_components], 
                                        cnm.C[idx_components, :], cnm.b, cnm.f,
                                        dims[0], dims[1], 
                                        YrA=cnm.YrA[idx_components, :], img=Cn)

#%% STOP CLUSTER and clean up log files   
cm.stop_server()

log_files = glob.glob('Yr*_LOG_*')
for log_file in log_files:
    os.remove(log_file)