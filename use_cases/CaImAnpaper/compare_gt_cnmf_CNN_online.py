#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Complete pipeline for online processing using OnACID. 

@author: Andrea Giovannucci @agiovann and Eftychios Pnevmatikakis @epnev
Special thanks to Andreas Tolias and his lab at Baylor College of Medicine 
for sharing their data used in this demo.
"""
from __future__ import division
from __future__ import print_function
import numpy as np
try:
    if __IPYTHON__:
        print('Debugging!')
        # this is used for debugging purposes only. allows to reload classes when changed
        get_ipython().magic('load_ext autoreload')
        get_ipython().magic('autoreload 2')
except NameError:
    print('Not IPYTHON')
    pass

from time import time
import caiman as cm
from caiman.source_extraction import cnmf as cnmf
from caiman.utils.visualization import view_patches_bar
from caiman.utils.utils import download_demo
import pylab as pl
import scipy
from caiman.motion_correction import motion_correct_iteration_fast
import cv2
from caiman.utils.visualization import plot_contours
import glob
from caiman.source_extraction.cnmf.online_cnmf import bare_initialization, initialize_movie_online, RingBuffer
#from caiman.source_extraction.cnmf.online_cnmf import load_object, save_object
from copy import deepcopy
import os

from builtins import str
from builtins import range

try:
    cv2.setNumThreads(1)
except:
    print('Open CV is naturally single threaded')



#%% set some global parameters here

global_params = {'min_SNR': 1.8,        # minimum SNR when considering adding a new neuron
                 'gnb' : 2,             # number of background components   
                 'epochs' : 2,          # number of passes over the data
                 'rval_thr' : 0.80,     # spatial correlation threshold
                 'batch_length_dt': 10, # length of mini batch for OnACID in decay time units (length would be batch_length_dt*decay_time*fr)
                 'max_thr': 0.30        # parameter for thresholding components when cleaning up shapes
                 }

params_movie = [{}]*10        # set up list of dictionaries
#% neurofinder.03.00.test
params_movie[0] = {'fname': '/mnt/ceph/neuro/labeling/neurofinder.03.00.test/images/final_map/Yr_d1_498_d2_467_d3_1_order_C_frames_2250_.mmap',
                 'folder_name' : '/mnt/ceph/neuro/labeling/neurofinder.03.00.test/',
                 'p': 1,  # order of the autoregressive system
                 'fr': 7,
                 'decay_time': 0.4,
                 'gSig': [12,12],  # expected half size of neurons              
                 'final_frate': 10,                 
                 'Npeaks': 5,
                 'r_values_min_full': .8,
                 'fitness_min_full': - 40,
                 'fitness_delta_min_full': - 40,
                 'gnb': 2,
                 'n_chunks': 10,
                 'swap_dim':False,
                 'T1': 2250
                 }
#% neurofinder.04.00.test
params_movie[1] = {'fname': '/mnt/ceph/neuro/labeling/neurofinder.04.00.test/images/final_map/Yr_d1_512_d2_512_d3_1_order_C_frames_3000_.mmap',
                 'folder_name' : '/mnt/ceph/neuro/labeling/neurofinder.04.00.test/',
                 'p': 1,  # order of the autoregressive system
                 'fr': 8, 
                 'gSig': [7,7],  # expected half size of neurons
                 'decay_time' : 0.5, # rough length of a transient
                 'gnb' : 2,
                 'T1' : 3000,
                 
                 'alpha_snmf': None,  # this controls sparsity
                 'final_frate': 10,
                 'r_values_min_patch': .5,  # threshold on space consistency
                 'fitness_min_patch': -10,  # threshold on time variability
                 # threshold on time variability (if nonsparse activity)
                 'fitness_delta_min_patch': -10,
                 'Npeaks': 5,
                 'merge_thresh': 0.8,  # merging threshold, max correlation allow
                 'rf': 20,  # half-size of the patches in pixels. rf=25, patches are 50x50    20
                 'stride_cnmf': 10,  # amounpl.it of overlap between the patches in pixels
                 'K': 5,  # number of components per patch
                 'is_dendrites': False,  # if dendritic. In this case you need to set init_method to sparse_nmf
                 'init_method': 'greedy_roi',                
                 'r_values_min_full': .8,
                 'fitness_min_full': - 40,
                 'fitness_delta_min_full': - 40,
                 'only_init_patch': True,
                 'memory_fact': 1,
                 'n_chunks': 10,
                 'update_background_components': True,# whether to update the background components in the spatial phase
                 'low_rank_background': True #whether to update the using a low rank approximation. In the False case all the nonzero elements of the background components are updated using hals    
                 }

#% neurofinder 02.00
params_movie[2] = {'fname': '/mnt/ceph/neuro/labeling/neurofinder.02.00/images/final_map/Yr_d1_512_d2_512_d3_1_order_C_frames_8000_.mmap',
                 'folder_name' : '/mnt/ceph/neuro/labeling/neurofinder.02.00/',
                 'p': 1,  # order of the autoregressive system
                 'fr' : 30, # imaging rate in Hz                 
                 'gSig': [8,8],  # expected half size of neuron
                 'decay_time': 0.3,                 
                 'gnb': 2,          
                 'T1':8000,
                 'r_values_min_full': .8,
                 
                 'merge_thresh': 0.8,  # merging threshold, max correlation allow
                 'rf': 20,  # half-size of the patches in pixels. rf=25, patches are 50x50    20
                 'stride_cnmf': 10,  # amounpl.it of overlap between the patches in pixels
                 'K': 6,  # number of components per patch
                 'is_dendrites': False,  # if dendritic. In this case you need to set init_method to sparse_nmf
                 'init_method': 'greedy_roi',            
                 'alpha_snmf': None,  # this controls sparsity
                 'final_frate': 10,
                 'r_values_min_patch': .5,  # threshold on space consistency
                 'fitness_min_patch': -10,  # threshold on time variability
                 # threshold on time variability (if nonsparse activity)
                 'fitness_delta_min_patch': -10,
                 'Npeaks': 5,              
                 'only_init_patch': True,
                 'memory_fact': 1,
                 'n_chunks': 10,
                 'update_background_components': True,# whether to update the background components in the spatial phase
                 'low_rank_background': True, #whether to update the using a low rank approximation. In the False case all the nonzero elements of the background components are updated using hals    
                                     #(to be used with one background per patch)     
                 'swap_dim':False,
                 'fitness_min_full': - 40,
                 'fitness_delta_min_full': - 40,
                 'crop_pix':10                 
                 }

#% yuste
params_movie[3] = {'fname': '/mnt/ceph/neuro/labeling/yuste.Single_150u/images/final_map/Yr_d1_200_d2_256_d3_1_order_C_frames_3000_.mmap',
                 'folder_name': '/mnt/ceph/neuro/labeling/yuste.Single_150u/', 
                 'p': 1,  # order of the autoregressive system
                 'fr' : 10,
                 'decay_time' : 0.75,
                 'T1' : 3000,
                 'gnb': 2,
                 'gSig': [7,7],  # expected half size of neurons
                 
                 'merge_thresh': 0.8,  # merging threshold, max correlation allow
                 'rf': 15,  # half-size of the patches in pixels. rf=25, patches are 50x50    20
                 'stride_cnmf': 10,  # amounpl.it of overlap between the patches in pixels
                 'K': 8,  # number of components per patch
                 'is_dendrites': False,  # if dendritic. In this case you need to set init_method to sparse_nmf
                 'init_method': 'greedy_roi',
                 'alpha_snmf': None,  # this controls sparsity
                 'final_frate': 10,
                 'r_values_min_patch': .5,  # threshold on space consistency
                 'fitness_min_patch': -10,  # threshold on time variability
                 # threshold on time variability (if nonsparse activity)
                 'fitness_delta_min_patch': -10,
                 'Npeaks': 5,
                 'r_values_min_full': .8,
                 'fitness_min_full': - 40,
                 'fitness_delta_min_full': - 40,
                 'only_init_patch': True,
                 'memory_fact': 1,
                 'n_chunks': 10,
                 'update_background_components': True,# whether to update the background components in the spatial phase
                 'low_rank_background': True, #whether to update the using a low rank approximation. In the False case all the nonzero elements of the background components are updated using hals    
                                     #(to be used with one background per patch)     
                 'swap_dim':False,
                 'crop_pix':0
                 }


#% neurofinder.00.00
params_movie[4] = {'fname': '/mnt/ceph/neuro/labeling/neurofinder.00.00/images/final_map/Yr_d1_512_d2_512_d3_1_order_C_frames_2936_.mmap',
                 'folder_name':  '/mnt/ceph/neuro/labeling/neurofinder.00.00/',
                 'p': 1,  # order of the autoregressive system
                 'decay_time' : 0.4, 
                 'fr' : 8,
                 'gSig': [8,8],  # expected half size of neurons
                 'gnb': 2,
                 'T1' : 2936,
                 
                 'merge_thresh': 0.8,  # merging threshold, max correlation allow
                 'rf': 20,  # half-size of the patches in pixels. rf=25, patches are 50x50    20
                 'stride_cnmf': 10,  # amounpl.it of overlap between the patches in pixels
                 'K': 6,  # number of components per patch
                 'is_dendrites': False,  # if dendritic. In this case you need to set init_method to sparse_nmf
                 'init_method': 'greedy_roi',
                 'alpha_snmf': None,  # this controls sparsity
                 'final_frate': 10,
                 'r_values_min_patch': .5,  # threshold on space consistency
                 'fitness_min_patch': -10,  # threshold on time variability
                 # threshold on time variability (if nonsparse activity)
                 'fitness_delta_min_patch': -10,
                 'Npeaks': 5,
                 'r_values_min_full': .8,
                 'fitness_min_full': - 40,
                 'fitness_delta_min_full': - 40,
                 'only_init_patch': True,                 
                 'memory_fact': 1,
                 'n_chunks': 10,
                 'update_background_components': True,# whether to update the background components in the spatial phase
                 'low_rank_background': True, #whether to update the using a low rank approximation. In the False case all the nonzero elements of the background components are updated using hals    
                                     #(to be used with one background per patch)     
                 'swap_dim':False,
                 'crop_pix':10
                 }
#% neurofinder.01.01
params_movie[5] = {'fname': '/mnt/ceph/neuro/labeling/neurofinder.01.01/images/final_map/Yr_d1_512_d2_512_d3_1_order_C_frames_1825_.mmap',
                 'folder_name': '/mnt/ceph/neuro/labeling/neurofinder.01.01/',
                 'p': 1,  # order of the autoregressive system
                 'merge_thresh': 0.9,  # merging threshold, max correlation allow
                 'rf': 20,  # half-size of the patches in pixels. rf=25, patches are 50x50    20
                 'stride_cnmf': 10,  # amounpl.it of overlap between the patches in pixels
                 'K': 6,  # number of components per patch
                 'is_dendrites': False,  # if dendritic. In this case you need to set init_method to sparse_nmf
                 'init_method': 'greedy_roi',
                 'gSig': [6,6],  # expected half size of neurons
                 'alpha_snmf': None,  # this controls sparsity
                 'final_frate': 10,
                 'r_values_min_patch': .5,  # threshold on space consistency
                 'fitness_min_patch': -10,  # threshold on time variability
                 # threshold on time variability (if nonsparse activity)
                 'fitness_delta_min_patch': -10,
                 'Npeaks': 5,
                 'r_values_min_full': .8,
                 'fitness_min_full': - 40,
                 'fitness_delta_min_full': - 40,
                 'only_init_patch': True,
                 'gnb': 2,
                 'memory_fact': 1,
                 'n_chunks': 10,
                 'update_background_components': True,# whether to update the background components in the spatial phase
                 'low_rank_background': True, #whether to update the using a low rank approximation. In the False case all the nonzero elements of the background components are updated using hals    
                                     #(to be used with one background per patch)     
                 'swap_dim':False,
                 'crop_pix':2,
                 'filter_after_patch':True
                 }
#% Sue Ann k56
params_movie[6] = {'fname': '/opt/local/Data/labeling/k53_20160530/Yr_d1_512_d2_512_d3_1_order_C_frames_116043_.mmap',
                'folder_name':'/opt/local/Data/labeling/k53_20160530/',
                'gtname':'/mnt/ceph/neuro/labeling/k53_20160530/regions/joined_consensus_active_regions.npy',
                 'p': 1,  # order of the autoregressive system
                 'merge_thresh': 0.8,  # merging threshold, max correlation allow
                 'rf': 20,  # half-size of the patches in pixels. rf=25, patches are 50x50    20
                 'stride_cnmf': 10,  # amounpl.it of overlap between the patches in pixels
                 'K': 9,  # number of components per patch
                 'is_dendrites': False,  # if dendritic. In this case you need to set init_method to sparse_nmf
                 'init_method': 'greedy_roi',
                 'gSig': [6,6],  # expected half size of neurons
                 'alpha_snmf': None,  # this controls sparsity
                 'final_frate': 30,
                 'r_values_min_patch': .5,  # threshold on space consistency
                 'fitness_min_patch': -10,  # threshold on time variability
                 # threshold on time variability (if nonsparse activity)
                 'fitness_delta_min_patch': -10,
                 'Npeaks': 5,
                 'r_values_min_full': .8,
                 'fitness_min_full': - 40,
                 'fitness_delta_min_full': - 40,
                 'only_init_patch': True,
                 'gnb': 2,
                 'memory_fact': 1,
                 'n_chunks': 10,
                 'update_background_components': True,# whether to update the background components in the spatial phase
                 'low_rank_background': True, #whether to update the using a low rank approximation. In the False case all the nonzero elements of the background components are updated using hals    
                                     #(to be used with one background per patch)     
                 'swap_dim':False,
                 'crop_pix':2,
                 'filter_after_patch':True
                 }

#% J115
params_movie[7] = {'fname': '/mnt/ceph/neuro/labeling/J115_2015-12-09_L01_ELS/images/final_map/Yr_d1_463_d2_472_d3_1_order_C_frames_90000_.mmap',
                'folder_name':'/mnt/ceph/neuro/labeling/J115_2015-12-09_L01_ELS/',
                'gtname':'/mnt/ceph/neuro/labeling/J115_2015-12-09_L01_ELS/regions/joined_consensus_active_regions.npy',
                 'p': 1,  # order of the autoregressive system
                 'merge_thresh': 0.8,  # merging threshold, max correlation allow
                 'rf': 20,  # half-size of the patches in pixels. rf=25, patches are 50x50    20
                 'stride_cnmf': 10,  # amounpl.it of overlap between the patches in pixels
                 'K': 7,  # number of components per patch
                 'is_dendrites': False,  # if dendritic. In this case you need to set init_method to sparse_nmf
                 'init_method': 'greedy_roi',
                 'gSig': [7,7],  # expected half size of neurons
                 'alpha_snmf': None,  # this controls sparsity
                 'final_frate': 30,
                 'r_values_min_patch': .5,  # threshold on space consistency
                 'fitness_min_patch': -10,  # threshold on time variability
                 # threshold on time variability (if nonsparse activity)
                 'fitness_delta_min_patch': -10,
                 'Npeaks': 5,
                 'r_values_min_full': .8,
                 'fitness_min_full': - 40,
                 'fitness_delta_min_full': - 40,
                 'only_init_patch': True,
                 'gnb': 2,
                 'memory_fact': 1,
                 'n_chunks': 10,
                 'update_background_components': True,# whether to update the background components in the spatial phase
                 'low_rank_background': True, #whether to update the using a low rank approximation. In the False case all the nonzero elements of the background components are updated using hals    
                                     #(to be used with one background per patch)     
                 'swap_dim':False,
                 'crop_pix':2,
                 'filter_after_patch':True
                 }

#% J123
params_movie[8] = {'fname': '/mnt/ceph/neuro/labeling/J123_2015-11-20_L01_0/images/final_map/Yr_d1_458_d2_477_d3_1_order_C_frames_41000_.mmap',
                'folder_name':'/mnt/ceph/neuro/labeling/J123_2015-11-20_L01_0/',
                'gtname':'/mnt/ceph/neuro/labeling/J123_2015-11-20_L01_0/regions/joined_consensus_active_regions.npy',
                 'p': 1,  # order of the autoregressive system
                 'merge_thresh': 0.8,  # merging threshold, max correlation allow
                 'rf': 40,  # half-size of the patches in pixels. rf=25, patches are 50x50    20
                 'stride_cnmf': 20,  # amounpl.it of overlap between the patches in pixels
                 'K': 10,  # number of components per patch
                 'is_dendrites': False,  # if dendritic. In this case you need to set init_method to sparse_nmf
                 'init_method': 'greedy_roi',
                 'gSig': [12,12],  # expected half size of neurons
                 'alpha_snmf': None,  # this controls sparsity
                 'final_frate': 15,
                 'r_values_min_patch': .5,  # threshold on space consistency
                 'fitness_min_patch': -10,  # threshold on time variability
                 # threshold on time variability (if nonsparse activity)
                 'fitness_delta_min_patch': -10,
                 'Npeaks': 5,
                 'r_values_min_full': .8,
                 'fitness_min_full': - 40,
                 'fitness_delta_min_full': - 40,
                 'only_init_patch': True,
                 'gnb': 2,
                 'memory_fact': 1,
                 'n_chunks': 10,
                 'update_background_components': True,# whether to update the background components in the spatial phase
                 'low_rank_background': True, #whether to update the using a low rank approximation. In the False case all the nonzero elements of the background components are updated using hals    
                                     #(to be used with one background per patch)     
                 'swap_dim':False,
                 'crop_pix':2,
                 'filter_after_patch':True
                 }

#%% Jan AMG
#params_movie = {'fname': '/opt/local/Data/Jan/Jan-AMG_exp3_001/Yr_d1_512_d2_512_d3_1_order_C_frames_115897_.mmap',
#                'gtname':'/mnt/ceph/neuro/labeling/Jan-AMG_exp3_001/regions/joined_consensus_active_regions.npy',
#                 'p': 1,  # order of the autoregressive system
#                 'merge_thresh': 0.8,  # merging threshold, max correlation allow
#                 'rf': 25,  # half-size of the patches in pixels. rf=25, patches are 50x50    20
#                 'stride_cnmf': 10,  # amounpl.it of overlap between the patches in pixels
#                 'K': 6,  # number of components per patch
#                 'is_dendrites': False,  # if dendritic. In this case you need to set init_method to sparse_nmf
#                 'init_method': 'greedy_roi',
#                 'gSig': [7,7],  # expected half size of neurons
#                 'alpha_snmf': None,  # this controls sparsity
#                 'final_frate':30,
#                 'r_values_min_patch': .5,  # threshold on space consistency
#                 'fitness_min_patch': -10,  # threshold on time variability
#                 # threshold on time variability (if nonsparse activity)
#                 'fitness_delta_min_patch': -10,
#                 'Npeaks': 5,
#                 'r_values_min_full': .8,
#                 'fitness_min_full': - 40,
#                 'fitness_delta_min_full': - 40,
#                 'only_init_patch': True,
#                 'gnb': 2,
#                 'memory_fact': 1,
#                 'n_chunks': 30,
#                 'update_background_components': True,# whether to update the background components in the spatial phase
#                 'low_rank_background': True, #whether to update the using a low rank approximation. In the False case all the nonzero elements of the background components are updated using hals    
#                                     #(to be used with one background per patch)     
#                 'swap_dim':False,
#                 'crop_pix':8,
#                 'filter_after_patch':True
#                 }
##%%
#params_movie = {'fname': '/mnt/ceph/neuro/labeling/k37_20160109_AM_150um_65mW_zoom2p2_00001_1-16/images/final_map/Yr_d1_512_d2_512_d3_1_order_C_frames_48000_.mmap',
#                'gtname':'/mnt/ceph/neuro/labeling/k37_20160109_AM_150um_65mW_zoom2p2_00001_1-16/regions/joined_consensus_active_regions.npy',
#                 'p': 1,  # order of the autoregressive system
#                 'merge_thresh': 0.8,  # merging threshold, max correlation allow
#                 'rf': 20,  # half-size of the patches in pixels. rf=25, patches are 50x50    20
#                 'stride_cnmf': 10,  # amounpl.it of overlap between the patches in pixels
#                 'K': 5,  # number of components per patch
#                 'is_dendrites': False,  # if dendritic. In this case you need to set init_method to sparse_nmf
#                 'init_method': 'greedy_roi',
#                 'gSig': [6,6],  # expected half size of neurons
#                 'alpha_snmf': None,  # this controls sparsity
#                 'final_frate':30,
#                 'r_values_min_patch': .5,  # threshold on space consistency
#                 'fitness_min_patch': -10,  # threshold on time variability
#                 # threshold on time variability (if nonsparse activity)
#                 'fitness_delta_min_patch': -10,
#                 'Npeaks': 5,
#                 'r_values_min_full': .8,
#                 'fitness_min_full': - 40,
#                 'fitness_delta_min_full': - 40,
#                 'only_init_patch': True,
#                 'gnb': 2,
#                 'memory_fact': 1,
#                 'n_chunks': 30,
#                 'update_background_components': True,# whether to update the background components in the spatial phase
#                 'low_rank_background': True, #whether to update the using a low rank approximation. In the False case all the nonzero elements of the background components are updated using hals    
#                                     #(to be used with one background per patch)     
#                 'swap_dim':False,
#                 'crop_pix':8,
#                 'filter_after_patch':True
#                 }


#%% Select a dataset
# 0: neuforinder.03.00.test
# 1: neurofinder.04.00.test
# 2: neurofinder.02.00
# 3: yuste
# 4: neurofinder.00.00
# 5: neurofinder,01.01
# 6: sue_ann_k56
# 7: J115
# 8: J123

ind_dataset = 4

#%% convert mmaps into tifs

fls = glob.glob(params_movie[ind_dataset]['folder_name']+'images/mmap/*.mmap')
for file_count, ffll in enumerate(fls):
    fl_temp = cm.movie(np.array(cm.load(ffll)))
    fl_temp.save(fls[file_count][:-4]+'tif')
#%%  download and list all files to be processed
fls = glob.glob('/'.join( params_movie[ind_dataset]['fname'].split('/')[:-3]+['images','tifs','*.tif']))
fls.sort()
print(fls)  
template = cm.load( '/'.join( params_movie[ind_dataset]['fname'].split('/')[:-3]+['projections','median_projection.tif']))                                      

#%% Set up some parameters
ds_factor = 1                                                        # spatial downsampling factor (increases speed but may lose some fine structure)
gSig = tuple(np.ceil(np.array(params_movie[ind_dataset]['gSig'])/ds_factor).astype(np.int))  # expected half size of neurons
init_files = 1                                                       # number of files used for initialization
online_files = len(fls) - 1                                          # number of files used for online
initbatch = 200                                                      # number of frames for initialization (presumably from the first file)
expected_comps = 3000                                                # maximum number of expected components used for memory pre-allocation (exaggerate here)
K = 2                                                                # initial number of components
N_samples = np.ceil(params_movie[ind_dataset]['fr']*params_movie[ind_dataset]['decay_time'])   # number of timesteps to consider when testing new neuron candidates
pr_inc = 1 - scipy.stats.norm.cdf(global_params['min_SNR'])           # inclusion probability of noise transient
thresh_fitness_raw = np.log(pr_inc)*N_samples       # 
thresh_fitness_delta = -1.                          # event exceptionality thresholds 
p = params_movie[ind_dataset]['p']                  # order of AR indicator dynamics
rval_thr = global_params['rval_thr']                # correlation threshold for new component inclusion
mot_corr = True                                     # flag for online motion correction 
max_shift = np.ceil(5./ds_factor).astype('int')     # maximum allowed shift during motion correction
gnb = global_params['gnb']                                               # number of background components
epochs = global_params['epochs']                    # number of passes over the data
#len_file = m.shape[0]                              # upper bound for number of frames in each file (used right below)
T1 = params_movie[ind_dataset]['T1'] *epochs        # total length of all files (if not known use a large number, then truncate at the end)
minibatch_length = int(global_params['batch_length_dt']*params_movie[ind_dataset]['fr']*params_movie[ind_dataset]['decay_time'])

#%%    Initialize movie
mot_corr = True
if ds_factor > 1:                                   # load only the first initbatch frames and possibly downsample them
    Y = cm.load(fls[0], subindices = slice(0,initbatch,None)).astype(np.float32).resize(1. / ds_factor, 1. / ds_factor)
else:
    Y =  cm.load(fls[0], subindices = slice(0,initbatch,None)).astype(np.float32)
    
if mot_corr:                                        # perform motion correction on the first initbatch frames
    mc = Y.motion_correct(max_shift, max_shift, template = template)
    Y = mc[0].astype(np.float32)
    borders = np.max(mc[1])
else:
    Y = Y.astype(np.float32)
      
img_min = Y.min()                                   # minimum value of movie. Subtract it to make the data non-negative
Y -= img_min
img_norm = np.std(Y, axis=0)                        
img_norm += np.median(img_norm)                     # normalizing factor to equalize the FOV
Y = Y / img_norm[None, :, :]                        # normalize data

_, d1, d2 = Y.shape
dims = (d1, d2)                                     # dimensions of FOV
Yr = Y.to_2D().T                                    # convert data into 2D array                                    

Cn_init = Y.local_correlations(swap_dim = False)    # compute correlation image
#pl.imshow(Cn_init); pl.title('Correlation Image on initial batch'); pl.colorbar()

#%% initialize OnACID with bare initialization

cnm_init = bare_initialization(Y[:initbatch].transpose(1, 2, 0), init_batch=initbatch, k=K, gnb=gnb,
                                 gSig=gSig, p=p, minibatch_shape=100, minibatch_suff_stat=5,
                                 update_num_comps = True, rval_thr=rval_thr,
                                 thresh_fitness_delta = thresh_fitness_delta,
                                 thresh_fitness_raw = thresh_fitness_raw,
                                 batch_update_suff_stat=True, max_comp_update_shape = 10, 
                                 deconv_flag = True,
                                 simultaneously=True, n_refit=0)

#crd = plot_contours(cnm_init.A.tocsc(), Cn_init, thr=0.9)

#% Plot initialization results

#A, C, b, f, YrA, sn = cnm_init.A, cnm_init.C, cnm_init.b, cnm_init.f, cnm_init.YrA, cnm_init.sn
#view_patches_bar(Yr, scipy.sparse.coo_matrix(A.tocsc()[:, :]), C[:, :], b, f, dims[0], dims[1], YrA=YrA[:, :], img=Cn_init)

#% Prepare object for OnACID

cnm_init._prepare_object(np.asarray(Yr[:,:initbatch]), T1, expected_comps, idx_components=None, N_samples_exceptionality = int(N_samples))

#%% Run OnACID and optionally plot results in real time

cnm2 = deepcopy(cnm_init)
cnm2.max_comp_update_shape = np.inf
cnm2.update_num_comps = True
t = cnm2.initbatch
tottime = []
Cn = Cn_init.copy()

plot_contours_flag = False               # flag for plotting contours of detected components at the end of each file
play_reconstr = False                     # flag for showing video with results online (turn off flags for improving speed)
save_movie = False                       # flag for saving movie (file could be quite large..)
#movie_name = folder_name + '/output.avi' # name of movie to be saved
resize_fact = 1.2                        # image resizing factor

if online_files == 0:                    # check whether there are any additional files
    process_files = fls[:init_files]     # end processing at this file
    init_batc_iter = [initbatch]         # place where to start
    end_batch = T1              
else:
    process_files = fls[:init_files + online_files]     # additional files
    init_batc_iter = [initbatch] + [0]*online_files     # where to start reading at each file

shifts = []
if save_movie and play_reconstr:
    fourcc = cv2.VideoWriter_fourcc('8', 'B', 'P', 'S') 
    out = cv2.VideoWriter(movie_name,fourcc, 30.0, tuple([int(2*x*resize_fact) for x in cnm2.dims]))

for iter in range(epochs):
    if iter > 0:
        process_files = fls[:init_files + online_files]     # if not on first epoch process all files from scratch
        init_batc_iter = [0]*(online_files+init_files)      #
        
    for file_count, ffll in enumerate(process_files):  # np.array(fls)[np.array([1,2,3,4,5,-5,-4,-3,-2,-1])]:
        print('Now processing file ' + ffll)
        Y_ = cm.load(ffll, subindices=slice(init_batc_iter[file_count],T1,None))
        
        if plot_contours_flag:   # update max-correlation (and perform offline motion correction) just for illustration purposes
            if ds_factor > 1:
                Y_1 = Y_.resize(1. / ds_factor, 1. / ds_factor, 1)
            else:
                Y_1 = Y_.copy()                    
                if mot_corr:
                    templ = (cnm2.Ab.data[:cnm2.Ab.indptr[1]] * cnm2.C_on[0, t - 1]).reshape(cnm2.dims, order='F') * img_norm        
                    newcn = (Y_1 - img_min).motion_correct(max_shift, max_shift, template=templ)[0].local_correlations(swap_dim=False)                
                    Cn = np.maximum(Cn, newcn)
                else:
                    Cn = np.maximum(Cn, Y_1.local_correlations(swap_dim=False))
    
        old_comps = cnm2.N                              # number of existing components
        for frame_count, frame in enumerate(Y_):        # now process each file
            if np.isnan(np.sum(frame)):
                raise Exception('Frame ' + str(frame_count) + ' contains nan')
            if t % 100 == 0:
                print('Epoch: ' + str(iter+1) + '. ' + str(t)+' frames have beeen processed in total. '+str(cnm2.N - old_comps)+' new components were added. Total number of components is '+str(cnm2.Ab.shape[-1]-gnb))
                old_comps = cnm2.N
    
            t1 = time()                                 # count time only for the processing part
            frame_ = frame.copy().astype(np.float32)    # 
            if ds_factor > 1:
                frame_ = cv2.resize(frame_, img_norm.shape[::-1])   # downsample if necessary 
    
            frame_ -= img_min                                       # make data non-negative
    
            if mot_corr:                                            # motion correct
                templ = cnm2.Ab.dot(cnm2.C_on[:cnm2.M, t - 1]).reshape(cnm2.dims, order='F') * img_norm
                frame_cor, shift = motion_correct_iteration_fast(frame_, templ, max_shift, max_shift)
                shifts.append(shift)
            else:
                templ = None
                frame_cor = frame_
    
            frame_cor = frame_cor / img_norm                        # normalize data-frame
            cnm2.fit_next(t, frame_cor.reshape(-1, order='F'))      # run OnACID on this frame
            tottime.append(time() - t1)                             # store time
    
            t += 1
            
            if t % 1000 == 0 and plot_contours_flag:
                pl.cla()
                A = cnm2.Ab[:, cnm2.gnb:]
                crd = cm.utils.visualization.plot_contours(A, Cn, thr=0.9)  # update the contour plot every 1000 frames
                pl.pause(1)
                
            if play_reconstr:                                               # generate movie with the results
                A, b = cnm2.Ab[:, cnm2.gnb:], cnm2.Ab[:, :cnm2.gnb].toarray()
                C, f = cnm2.C_on[cnm2.gnb:cnm2.M, :], cnm2.C_on[:cnm2.gnb, :]
                comps_frame = A.dot(C[:,t-1]).reshape(cnm2.dims, order = 'F')*img_norm/np.max(img_norm)   # inferred activity due to components (no background)
                bgkrnd_frame = b.dot(f[:,t-1]).reshape(cnm2.dims, order = 'F')*img_norm/np.max(img_norm)  # denoised frame (components + background)
                all_comps = (np.array(A.sum(-1)).reshape(cnm2.dims, order = 'F'))                         # spatial shapes
                frame_comp_1 = cv2.resize(np.concatenate([frame_/np.max(img_norm),all_comps*3.],axis = -1),(2*np.int(cnm2.dims[1]*resize_fact),np.int(cnm2.dims[0]*resize_fact) ))
                frame_comp_2 = cv2.resize(np.concatenate([comps_frame*10.,comps_frame+bgkrnd_frame],axis = -1),(2*np.int(cnm2.dims[1]*resize_fact),np.int(cnm2.dims[0]*resize_fact) ))
                frame_pn = np.concatenate([frame_comp_1,frame_comp_2],axis=0).T
                vid_frame = np.repeat(frame_pn[:,:,None],3,axis=-1)
                vid_frame = np.minimum((vid_frame*255.),255).astype('u1')
                cv2.putText(vid_frame,'Raw Data',(5,20),fontFace = 5, fontScale = 1.2, color = (0,255,0), thickness = 1)
                cv2.putText(vid_frame,'Inferred Activity',(np.int(cnm2.dims[0]*resize_fact) + 5,20),fontFace = 5, fontScale = 1.2, color = (0,255,0), thickness = 1)
                cv2.putText(vid_frame,'Identified Components',(5,np.int(cnm2.dims[1]*resize_fact)  + 20),fontFace = 5, fontScale = 1.2, color = (0,255,0), thickness = 1)
                cv2.putText(vid_frame,'Denoised Data',(np.int(cnm2.dims[0]*resize_fact) + 5 ,np.int(cnm2.dims[1]*resize_fact)  + 20),fontFace = 5, fontScale = 1.2, color = (0,255,0), thickness = 1)
                cv2.putText(vid_frame,'Frame = '+str(t),(vid_frame.shape[1]//2-vid_frame.shape[1]//10,vid_frame.shape[0]-20),fontFace = 5, fontScale = 1.2, color = (0,255,255), thickness = 1)                
                if save_movie:
                    out.write(vid_frame)
                cv2.imshow('frame',vid_frame)                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break                                
    
        print('Cumulative processing speed is ' + str((t - initbatch) / np.sum(tottime))[:5] + ' frames per second.')
        
if save_movie:
    out.release()
cv2.destroyAllWindows()
#%%  save results (optional)

save_results = False

if save_results:
    np.savez(params_movie[ind_dataset]['folder_name']+'results_analysis_online_MOT_CORR.npz',
             Cn=Cn, Ab=cnm2.Ab, Cf=cnm2.C_on, b=cnm2.b, f=cnm2.f,
             dims=cnm2.dims, tottime=tottime, noisyC=cnm2.noisyC, shifts=shifts, img=Cn, 
             params_movie = params_movie[ind_dataset], global_params = global_params)

#%% extract results from the objects and do some plotting
A, b = cnm2.Ab[:, cnm2.gnb:], cnm2.Ab[:, :cnm2.gnb].toarray()
C, f = cnm2.C_on[cnm2.gnb:cnm2.M, t-t//epochs:t], cnm2.C_on[:cnm2.gnb, t-t//epochs:t]
noisyC = cnm2.noisyC[:,t-t//epochs:t]
b_trace = [osi.b for osi in cnm2.OASISinstances]

#pl.figure()
#crd = cm.utils.visualization.plot_contours(A, Cn, thr=0.9)
#view_patches_bar(Yr, scipy.sparse.coo_matrix(A.tocsc()[:, :]), C[:, :], b, f,
#                 dims[0], dims[1], YrA=noisyC[cnm2.gnb:cnm2.M] - C, img=Cn)

#%% load, threshold and filter for size ground truth

c, dview, n_processes = cm.cluster.setup_cluster(backend='local', n_processes=None, single_thread = True)

gt_file = os.path.join(os.path.split(params_movie[ind_dataset]['fname'])[0], os.path.split(params_movie[ind_dataset]['fname'])[1][:-4] + 'match_masks.npz')
min_radius = gSig[0]/2.          # minimum acceptable radius
max_radius = 2.*gSig[0]          # maximum acceptable radius
min_size_neuro = min_radius**2*np.pi
max_size_neuro = max_radius**2*np.pi
with np.load(gt_file, encoding = 'latin1') as ld:
    print(ld.keys())
    locals().update(ld)
    A_gt = scipy.sparse.coo_matrix(A_gt[()])
    dims = (d1,d2)
    
A_gt_thr = cm.source_extraction.cnmf.spatial.threshold_components(A_gt.tocsc()[:,:].toarray(), dims, medw=None, thr_method='max', maxthr=global_params['max_thr'], nrgthr=0.99, extract_cc=True,
                         se=None, ss=None, dview=None) 

A_gt_thr_bin = A_gt_thr > 0
size_neurons_gt = A_gt_thr_bin.sum(0)
idx_size_neurons_gt = np.where((size_neurons_gt>min_size_neuro) & (size_neurons_gt < max_size_neuro) )[0]
print(A_gt_thr.shape)     
#%% filter for size found neurons

A_thr = cm.source_extraction.cnmf.spatial.threshold_components(A.tocsc()[:,:].toarray(), dims, medw=None, thr_method='max', maxthr=global_params['max_thr'], nrgthr=0.99, extract_cc=True,
                         se=None, ss=None, dview=dview) 
A_thr_bin = A_thr > 0  
size_neurons = A_thr_bin.sum(0)
idx_size_neurons = np.where((size_neurons>min_size_neuro) & (size_neurons<max_size_neuro))[0]
#A_thr = A_thr[:,idx_size_neuro]
print(A_thr.shape)

#%% compute results 

use_cnn = False  # Use CNN classifier
if use_cnn:    
    from caiman.components_evaluation import evaluate_components_CNN
    predictions,final_crops = evaluate_components_CNN(A,dims,gSig,model_name = 'use_cases/CaImAnpaper/cnn_model')
    thresh_cnn = .05
    idx_components_cnn = np.where(predictions[:,1]>=thresh_cnn)[0]
    idx_neurons = np.intersect1d(idx_components_cnn,idx_size_neurons)
else:
    idx_neurons = idx_size_neurons.copy()
        

plot_results = True
if plot_results:
    pl.figure(figsize=(30,20))

tp_gt, tp_comp, fn_gt, fp_comp, performance_cons_off =  cm.base.rois.nf_match_neurons_in_binary_masks(A_gt_thr_bin[:,idx_size_neurons_gt].reshape([dims[0],dims[1],-1],order = 'F').transpose([2,0,1])*1.,
                                                                              A_thr_bin[:,idx_neurons].reshape([dims[0],dims[1],-1],order = 'F').transpose([2,0,1])*1.,thresh_cost=.7, min_dist = 10,
                                                                              print_assignment= False,plot_results=plot_results,Cn=Cn, labels = ['GT','Offline'])

pl.rcParams['pdf.fonttype'] = 42
font = {'family' : 'Arial',
        'weight' : 'regular',
        'size'   : 20}

pl.rc('font', **font)
print({a:b.astype(np.float16) for a,b in performance_cons_off.items()})