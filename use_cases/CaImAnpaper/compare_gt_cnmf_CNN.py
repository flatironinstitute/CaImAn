#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 14:49:36 2017

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
import glob
import pickle

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
import scipy
from caiman.utils.visualization import plot_contours, view_patches_bar
from caiman.source_extraction.cnmf import cnmf as cnmf
from caiman.components_evaluation import estimate_components_quality_auto
from caiman.cluster import setup_cluster
#%%
global_params = {'min_SNR': 2,        # minimum SNR when considering adding a new neuron
                 'gnb': 2,             # number of background components   
                 'rval_thr' : 0.80,     # spatial correlation threshold
                 'min_cnn_thresh' : 0.95,
                 'p' : 1,
                 'min_rval_thr_rejected': 0, # length of mini batch for OnACID in decay time units (length would be batch_length_dt*decay_time*fr)
#                 'min_fitness_raw_rejected': -4,       # parameter for thresholding components when cleaning up shapes
                 'max_classifier_probability_rejected' : 0.1,    # flag for motion correction (set to False to compare directly on the same FOV)                 
                 'max_fitness_delta_accepted' : -20,
#                 'max_fitness_accepted' : -20,
                 'Npeaks' : 5, 
                 'min_SNR_patch' : -10,
                 'min_r_val_thr_patch': 0.5,
                 'fitness_delta_min_patch': -5,
                 'update_background_components' : True,# whether to update the background components in the spatial phase
                 'low_rank_background'  : True, #whether to update the using a low rank approximation. In the False case all the nonzero elements of the background components are updated using hals    
                                             #(to be used with one background per patch) 
                 'only_init_patch'  : True,
                 'is_dendrites'  : False,  # if dendritic. In this case you need to set init_method to sparse_nmf
                 'alpha_snmf'  : None,
                 'init_method'  : 'greedy_roi',
                 'filter_after_patch'  :  False
                 }

params_movies = []  
#%%
params_movie = {'fname': '/mnt/ceph/neuro/labeling/neurofinder.03.00.test/images/final_map/Yr_d1_498_d2_467_d3_1_order_C_frames_2250_.mmap',
                     # order of the autoregressive system
                 'merge_thresh': 0.8,  # merging threshold, max correlation allow
                 'rf': 25,  # half-size of the patches in pixels. rf=25, patches are 50x50    20
                 'stride_cnmf': 10,  # amounpl.it of overlap between the patches in pixels
                 'K': 4,  # number of components per patch                 
                 'gSig': [8,8],  # expected half size of neurons
                                       
                 'n_chunks': 10,
                 'swap_dim':False,
                 'crop_pix' : 0, 
                 'fr': 7,
                 'decay_time': 0.4,                       
                 }
params_movies.append(params_movie.copy())
#%%
params_movie = {'fname': '/mnt/ceph/neuro/labeling/neurofinder.04.00.test/images/final_map/Yr_d1_512_d2_512_d3_1_order_C_frames_3000_.mmap',
                     # order of the autoregressive system
                 'merge_thresh': 0.8,  # merging threshold, max correlation allow
                 'rf': 20,  # half-size of the patches in pixels. rf=25, patches are 50x50    20
                 'stride_cnmf': 10,  # amounpl.it of overlap between the patches in pixels
                 'K': 5,  # number of components per patch
                 'gSig': [5,5],  # expected half size of neurons
                
                 'n_chunks': 10,
                 'swap_dim':False,
                 'crop_pix' : 0,
                 'fr' : 8,
                 'decay_time' : 0.5, # rough length of a transient
                 }
params_movies.append(params_movie.copy())
#%% neurofinder 02.00
params_movie = {'fname': '/mnt/ceph/neuro/labeling/neurofinder.02.00/images/final_map/Yr_d1_512_d2_512_d3_1_order_C_frames_8000_.mmap',
                     # order of the autoregressive system
                 'merge_thresh': 0.8,  # merging threshold, max correlation allow
                 'rf': 20,  # half-size of the patches in pixels. rf=25, patches are 50x50    20
                 'stride_cnmf': 10,  # amounpl.it of overlap between the patches in pixels
                 'K': 6,  # number of components per patch
                 'gSig': [5,5],  # expected half size of neurons
                 'fr' : 30, # imaging rate in Hz
                 'n_chunks': 10,
                 'swap_dim':False,
                 'crop_pix':10,
                 'decay_time': 0.3,  
                 }
params_movies.append(params_movie.copy())
#%% packer
#params_movie = {'fname': '/mnt/ceph/neuro/labeling/packer.001/images/final_map/Yr_d1_512_d2_512_d3_1_order_C_frames_9900_.mmap',
#                     # order of the autoregressive system
#                 'merge_thresh': 0.99,  # merging threshold, max correlation allow
#                 'rf': 25,  # half-size of the patches in pixels. rf=25, patches are 50x50    20
#                 'stride_cnmf': 10,  # amounpl.it of overlap between the patches in pixels
#                 'K': 6,  # number of components per patch
#                 'gSig': [6,6],  # expected half size of neurons
#                 'fr': 30,
#                 'n_chunks': 10,
#                 'update_background_components': False,# whether to update the background components in the spatial phase
#                'swap_dim':False,
#                'crop_pix':0
#                 }
#%% yuste
params_movie = {'fname': '/mnt/ceph/neuro/labeling/yuste.Single_150u/images/final_map/Yr_d1_200_d2_256_d3_1_order_C_frames_3000_.mmap',
                     # order of the autoregressive system
                 'merge_thresh': 0.8,  # merging threshold, max correlation allow
                 'rf': 15,  # half-size of the patches in pixels. rf=25, patches are 50x50    20
                 'stride_cnmf': 10,  # amounpl.it of overlap between the patches in pixels
                 'K': 8,  # number of components per patch
                 'gSig': [5,5],  # expected half size of neurons
                 'fr' : 10, 
                  'decay_time' : 0.75,
                 'n_chunks': 10,
                 'swap_dim':False,
                 'crop_pix':0
                 }
params_movies.append(params_movie.copy())

#%% neurofinder 00.00
params_movie = {'fname': '/mnt/ceph/neuro/labeling/neurofinder.00.00/images/final_map/Yr_d1_512_d2_512_d3_1_order_C_frames_2936_.mmap',
                     # order of the autoregressive system
                 'merge_thresh': 0.8,  # merging threshold, max correlation allow
                 'rf': 20,  # half-size of the patches in pixels. rf=25, patches are 50x50    20
                 'stride_cnmf': 10,  # amounpl.it of overlap between the patches in pixels
                 'K': 6,  # number of components per patch
                 'gSig': [6,6],  # expected half size of neurons
                'decay_time' : 0.4, 
                 'fr' : 8,
                 'n_chunks': 10,
                 'swap_dim':False,
                 'crop_pix':10
                 }
params_movies.append(params_movie.copy())
#%% neurofinder 01.01
params_movie = {'fname': '/mnt/ceph/neuro/labeling/neurofinder.01.01/images/final_map/Yr_d1_512_d2_512_d3_1_order_C_frames_1825_.mmap',
                     # order of the autoregressive system
                 'merge_thresh': 0.9,  # merging threshold, max correlation allow
                 'rf': 20,  # half-size of the patches in pixels. rf=25, patches are 50x50    20
                 'stride_cnmf': 10,  # amounpl.it of overlap between the patches in pixels
                 'K': 6,  # number of components per patch
                 'gSig': [6,6],  # expected half size of neurons
                 'decay_time' : 1.4, 
                 'fr' : 8,
                 'n_chunks': 10,
                 'swap_dim':False,
                 'crop_pix':2,
                 }
params_movies.append(params_movie.copy())
#%% Sue Ann k56
params_movie = {'fname': '/opt/local/Data/labeling/k53_20160530/Yr_d1_512_d2_512_d3_1_order_C_frames_116043_.mmap',
                'gtname':'/mnt/ceph/neuro/labeling/k53_20160530/regions/joined_consensus_active_regions.npy',
                     # order of the autoregressive system
                 'merge_thresh': 0.8,  # merging threshold, max correlation allow
                 'rf': 20,  # half-size of the patches in pixels. rf=25, patches are 50x50    20
                 'stride_cnmf': 10,  # amounpl.it of overlap between the patches in pixels
                 'K': 9,  # number of components per patch
                 'gSig': [6,6],  # expected half size of neurons
                 'fr': 30, 
                 'decay_time' : 0.3,
                 'n_chunks': 10,     
                 'swap_dim':False,
                 'crop_pix':2,
                 }
params_movies.append(params_movie.copy())
#%% J115
params_movie = {'fname': '/opt/local/Data/labeling/J115_2015-12-09_L01_ELS/Yr_d1_463_d2_472_d3_1_order_C_frames_90000_.mmap',
                'gtname':'/mnt/ceph/neuro/labeling/J115_2015-12-09_L01_ELS/regions/joined_consensus_active_regions.npy',
                     # order of the autoregressive system
                 'merge_thresh': 0.8,  # merging threshold, max correlation allow
                 'rf': 20,  # half-size of the patches in pixels. rf=25, patches are 50x50    20
                 'stride_cnmf': 10,  # amounpl.it of overlap between the patches in pixels
                 'K': 7,  # number of components per patch
                 'gSig': [7,7],  # expected half size of neurons
                 'fr' : 30,
                'decay_time' : 0.4,            
                 'n_chunks': 10,    
                 'swap_dim':False,
                 'crop_pix':2,
                   
                 }
params_movies.append(params_movie.copy())
#%% J123
params_movie = {'fname': '/mnt/ceph/neuro/labeling/J123_2015-11-20_L01_0/images/final_map/Yr_d1_458_d2_477_d3_1_order_C_frames_41000_.mmap',
                'gtname':'/mnt/ceph/neuro/labeling/J123_2015-11-20_L01_0/regions/joined_consensus_active_regions.npy',
                     # order of the autoregressive system
                 'merge_thresh': 0.8,  # merging threshold, max correlation allow
                 'rf': 40,  # half-size of the patches in pixels. rf=25, patches are 50x50    20
                 'stride_cnmf': 20,  # amounpl.it of overlap between the patches in pixels
                 'K': 10,  # number of components per patch
                 'gSig': [8,12],  # expected half size of neurons
                 'decay_time' : 0.5,
                 'fr' : 30,
                 'n_chunks': 10,
                 'swap_dim':False,
                 'crop_pix':2,  
                 }
params_movies.append(params_movie.copy())
#%% Jan AMG DIPS IN AVERAGE
#params_movie = {'fname': '/opt/local/Data/Jan/Jan-AMG_exp3_001/Yr_d1_512_d2_512_d3_1_order_C_frames_115897_.mmap',
#                'gtname':'/mnt/ceph/neuro/labeling/Jan-AMG_exp3_001/regions/joined_consensus_active_regions.npy',
#                     # order of the autoregressive system
#                 'merge_thresh': 0.8,  # merging threshold, max correlation allow
#                 'rf': 25,  # half-size of the patches in pixels. rf=25, patches are 50x50    20
#                 'stride_cnmf': 10,  # amounpl.it of overlap between the patches in pixels
#                 'K': 6,  # number of components per patch
#                 'gSig': [7,7],  # expected half size of neurons
#                 'fr':30,
#                 'n_chunks': 30,
#                 'swap_dim':False,
#                 'crop_pix':8,
#                 }
#params_movies.append(params_movie.copy())
#%%
params_movie = { 'fname': '/mnt/ceph/neuro/labeling/k37_20160109_AM_150um_65mW_zoom2p2_00001_1-16/images/final_map/Yr_d1_512_d2_512_d3_1_order_C_frames_48000_.mmap',
                 'gtname':'/mnt/ceph/neuro/labeling/k37_20160109_AM_150um_65mW_zoom2p2_00001_1-16/regions/joined_consensus_active_regions.npy',
                     # order of the autoregressive system
                 'merge_thresh': 0.8,  # merging threshold, max correlation allow
                 'rf': 20,  # half-size of the patches in pixels. rf=25, patches are 50x50    20
                 'stride_cnmf': 10,  # amounpl.it of overlap between the patches in pixels
                 'K': 5,  # number of components per patch
                 'gSig': [6,6],  # expected half size of neurons
                 'fr' : 30,
                 'decay_time' : 0.3,
                 'n_chunks': 30,
                 'swap_dim':False,
                 'crop_pix':8,
                 }
params_movies.append(params_movie.copy())
#%% yuste: sue ann
#params_movie = {'fname': '/mnt/ceph/neuro/labeling/yuste.Single_150u/images/final_map/Yr_d1_200_d2_256_d3_1_order_C_frames_3000_.mmap',
#                'seed_name':'/mnt/xfs1/home/agiovann/Downloads/yuste_sue_masks.mat',
#                     # order of the autoregressive system
#                 'merge_thresh': 1,  # merging threshold, max correlation allow
#                 'fr': 10,
#                   
#                 'swap_dim':False, #for some movies needed
#                 'kernel':None,
#                 }
##%% SUE neuronfinder00.00
#params_movie = {'fname': '/mnt/ceph/neuro/labeling/neurofinder.00.00/images/final_map/Yr_d1_512_d2_512_d3_1_order_C_frames_2936_.mmap',
#                'seed_name':'/mnt/xfs1/home/agiovann/Downloads/neurofinder_00.00_sue_masks.mat',
#                     # order of the autoregressive system
#                 'merge_thresh': 1,  # merging threshold, max correlation allow
#                 'fr': 10,
#                   
#                 'swap_dim':False, #for some movies needed
#                 'kernel':None,
#                 }
##%% SUE neuronfinder02.00
#params_movie = {'fname': '/mnt/ceph/neuro/labeling/neurofinder.02.00/images/final_map/Yr_d1_512_d2_512_d3_1_order_C_frames_8000_.mmap',
#                'seed_name':'/mnt/xfs1/home/agiovann/Downloads/neurofinder_02.00_sue_masks.mat',
#                     # order of the autoregressive system
#                 'merge_thresh': 1,  # merging threshold, max correlation allow
#                 'fr': 10,
#                   
#                 'swap_dim':False, #for some movies needed
#                 'kernel':None,
#                 }
#%%
#params_movie = params_movies[4]
all_perfs = []
all_rvalues = []
all_comp_SNR_raw =[] 
all_comp_SNR_delta = []
all_predictions = []
all_labels = []
reload = True
plot_on = False
save_on = False
for params_movie in np.array(params_movies)[:]:
#    params_movie['gnb'] = 3
    params_display = {
        'downsample_ratio': .2,
        'thr_plot': 0.8
    }
    
    # @params fname name of the movie
    fname_new = params_movie['fname']
    print(fname_new)
    # %% RUN ANALYSIS
    try:
        dview.terminate()    
    except:
        print('No clusters to stop')
        
    c, dview, n_processes = setup_cluster(
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
    #%%
    if not reload:
    # %% correlation image
        if m_images.shape[0]<10000:
            Cn = m_images.local_correlations(swap_dim = params_movie['swap_dim'], frames_per_chunk = 1500)
            Cn[np.isnan(Cn)] = 0
            check_nan = False
        else:
            Cn = np.array(cm.load(('/'.join(params_movie['gtname'].split('/')[:-2]+['projections','correlation_image.tif'])))).squeeze()
            check_nan = False
        pl.imshow(Cn, cmap='gray', vmax=.95)
        
        # %% some parameter settings
        # order of the autoregressive fit to calcium imaging in general one (slow gcamps) or two (fast gcamps fast scanning)
        p = global_params['p']
        # merging threshold, max correlation allowed
        merge_thresh = params_movie['merge_thresh']
        # half-size of the patches in pixels. rf=25, patches are 50x50
        rf = params_movie['rf']
        # amounpl.it of overlap between the patches in pixels
        stride_cnmf = params_movie['stride_cnmf']
        # number of components per patch
        K = params_movie['K']
        # if dendritic. In this case you need to set init_method to sparse_nmf
        is_dendrites = global_params['is_dendrites']
        # iinit method can be greedy_roi for round shapes or sparse_nmf for denritic data
        init_method = global_params['init_method']
        # expected half size of neurons
        gSig = params_movie['gSig']
        # this controls sparsity
        alpha_snmf = global_params['alpha_snmf']
        # frame rate of movie (even considering eventual downsampling)
        final_frate = params_movie['fr']
        
        if global_params['is_dendrites'] == True:
            if global_params['init_method'] is not 'sparse_nmf':
                raise Exception('dendritic requires sparse_nmf')
            if global_params['alpha_snmf'] is None:
                raise Exception('need to set a value for alpha_snmf')
        # %% Extract spatial and temporal components on patches
        t1 = time.time()
        # TODO: todocument
        # TODO: warnings 3
        cnm = cnmf.CNMF(n_processes=1, nb_patch = 1, k=K, gSig=gSig, merge_thresh=params_movie['merge_thresh'], p=global_params['p'],
                        dview=dview, rf=rf, stride=stride_cnmf, memory_fact=1,
                        method_init=init_method, alpha_snmf=alpha_snmf, only_init_patch=global_params['only_init_patch'],
                        gnb=global_params['gnb'], method_deconvolution='oasis',border_pix =  params_movie['crop_pix'], 
                        low_rank_background = global_params['low_rank_background'], rolling_sum = True, check_nan=check_nan) 
        cnm = cnm.fit(images)
        
        A_tot = cnm.A
        C_tot = cnm.C
        YrA_tot = cnm.YrA
        b_tot = cnm.b
        f_tot = cnm.f
        sn_tot = cnm.sn
        print(('Number of components:' + str(A_tot.shape[-1])))
        t_patch = time.time() - t1
        dview.terminate()
        c, dview, n_processes = cm.cluster.setup_cluster(
        backend='local', n_processes=None, single_thread=False)
        # %%
        if plot_on:
            pl.figure()
            # TODO: show screenshot 12`
            # TODO : change the way it is used
            crd = plot_contours(A_tot, Cn, thr=params_display['thr_plot'])

            # %% rerun updating the components to refine
        t1 = time.time()
        cnm = cnmf.CNMF(n_processes=1, k=A_tot.shape, gSig=gSig, merge_thresh=merge_thresh, p=p, dview=dview, Ain=A_tot,
                        Cin=C_tot, b_in = b_tot,
                        f_in=f_tot, rf=None, stride=None, method_deconvolution='oasis',gnb = global_params['gnb'],
                        low_rank_background = global_params['low_rank_background'], 
                        update_background_components = global_params['update_background_components'], check_nan=check_nan)
        
        cnm = cnm.fit(images)
        t_refine = time.time() - t1
    
        A, C, b, f, YrA, sn = cnm.A, cnm.C, cnm.b, cnm.f, cnm.YrA, cnm.sn
        # %% again recheck quality of components, stricter criteria
        idx_components, idx_components_bad, comp_SNR, comp_SNR_delta, r_values, predictionsCNN = estimate_components_quality_auto(
                            Y, A, C, b, f, YrA, params_movie['fr'], params_movie['decay_time'], gSig, dims, 
                            dview = dview, min_SNR=global_params['min_SNR'],
                            r_values_min = global_params['rval_thr'], r_values_lowest = global_params['min_rval_thr_rejected'],
                            Npeaks =  global_params['Npeaks'], use_cnn = True, thresh_cnn_min = global_params['min_cnn_thresh'],
                            thresh_cnn_lowest = global_params['max_classifier_probability_rejected'],
                            thresh_fitness_delta = global_params['max_fitness_delta_accepted'])
        
#        N_samples = np.ceil(params_movie['fr']*params_movie['decay_time']).astype(np.int)   # number of timesteps to consider when testing new neuron candidates
#        min_SNR = global_params['min_SNR']                                      # adaptive way to set threshold (will be equal to min_SNR) 
#        #pr_inc = 1 - scipy.stats.norm.cdf(global_params['min_SNR'])           # inclusion probability of noise transient
#        #thresh_fitness_raw = np.log(pr_inc)*N_samples       # event exceptionality threshold
#        thresh_fitness_raw = scipy.special.log_ndtr(-min_SNR)*N_samples
#        thresh_fitness_delta = -20.     
#        t1 = time.time()
#        final_frate = params_movie['fr']
#        r_values_min = global_params['rval_thr']  # threshold on space consistency
#        fitness_min = scipy.special.log_ndtr(-min_SNR)*N_samples  # threshold on time variability
#        # threshold on time variability (if nonsparse activity)
#        fitness_delta_min = global_params['max_fitness_delta_accepted']
#        Npeaks = global_params['Npeaks']
#        traces = C + YrA
#        idx_components, idx_components_bad, fitness_raw, fitness_delta, r_values = estimate_components_quality(
#            traces, Y, A, C, b, f, final_frate=final_frate, Npeaks=Npeaks, r_values_min=r_values_min, fitness_min=fitness_min,
#            fitness_delta_min=fitness_delta_min, return_all=True, dview = dview, num_traces_per_group = 50, N = N_samples)
        t_eva_comps = time.time() - t1
        print(' ***** ')
        print((len(traces)))
        print((len(idx_components)))
        # %% save results
        if save_on:
            np.savez(os.path.join(os.path.split(fname_new)[0], os.path.split(fname_new)[1][:-4] + 'results_analysis_after_merge_3.npz'), Cn=Cn, fname_new = fname_new,
                     A=A,
                     C=C, b=b, f=f, YrA=YrA, sn=sn, d1=d1, d2=d2, idx_components=idx_components,
                     idx_components_bad=idx_components_bad,
                     comp_SNR=comp_SNR, comp_SNR_delta=comp_SNR_delta, r_values=r_values, predictionsCNN = predictionsCNN, params_movie = params_movie)
            
        # %%
        if plot_on:        
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
    
    #%% LOAD DATA
    else:
        
        params_display = {
            'downsample_ratio': .2,
            'thr_plot': 0.8
        }
        fn_old = fname_new 
        #analysis_file = '/mnt/ceph/neuro/jeremie_analysis/neurofinder.03.00.test/Yr_d1_498_d2_467_d3_1_order_C_frames_2250_._results_analysis.npz'
        with np.load(os.path.join(os.path.split(fname_new)[0], os.path.split(fname_new)[1][:-4] + 'results_analysis_after_merge_3.npz'), encoding = 'latin1') as ld:
            print(ld.keys())
            locals().update(ld) 
            dims_off = d1,d2    
            A = scipy.sparse.coo_matrix(A[()])
            dims = (d1,d2)
            gSig = params_movie['gSig']
            fname_new = fn_old
            
        
        idx_components, idx_components_bad, comp_SNR, comp_SNR_delta, r_values, predictionsCNN = estimate_components_quality_auto(
                            Y, A, C, b, f, YrA, params_movie['fr'], params_movie['decay_time'], gSig, dims, 
                            dview = dview, min_SNR=global_params['min_SNR'],
                            r_values_min = global_params['rval_thr'], r_values_lowest = global_params['min_rval_thr_rejected'],
                            Npeaks =  global_params['Npeaks'], use_cnn = True, thresh_cnn_min = global_params['min_cnn_thresh'],
                            thresh_cnn_lowest = global_params['max_classifier_probability_rejected'],
                            thresh_fitness_delta = global_params['max_fitness_delta_accepted'])   
        
#        with open('component_classifier.pkl', 'rb') as fid:
#            total_classifier = pickle.load(fid)
#            idx_components = total_classifier
#            X_SNR_delta = np.clip(comp_SNR_delta,0,20)
#            X_SNR_raw = np.clip(comp_SNR,0,20)
#            X_r = r_values
#            X_pred = predictionsCNN
#            X_all = np.vstack([X_SNR_delta, X_SNR_raw, X_r, X_pred]).T
#            Y_class =  total_classifier.predict(X_all)
#            idx_components = np.where(Y_class == 1)[0]
#            idx_components_bad = np.where(Y_class == 0)[0]
            
            
        print(' ***** ')
        print((len(C)))
        print((len(idx_components)))        
#%%
    all_matches = False
    filter_SNR = False
    gt_file = os.path.join(os.path.split(fname_new)[0], os.path.split(fname_new)[1][:-4] + 'match_masks.npz')
    with np.load(gt_file, encoding = 'latin1') as ld:
        print(ld.keys())
        locals().update(ld)
        A_gt = scipy.sparse.coo_matrix(A_gt[()])
        dims = (d1,d2)
        try:
            fname_new = fname_new[()].decode('unicode_escape')    
        except:
            fname_new = fname_new[()]        
        if all_matches:
            roi_all_match = cm.base.rois.nf_read_roi_zip(glob.glob('/'.join(os.path.split(fname_new)[0].split('/')[:-2]+['regions/*nd_matches.zip']))[0],dims)
    #        roi_1_match = cm.base.rois.nf_read_roi_zip('/mnt/ceph/neuro/labeling/neurofinder.02.00/regions/intermediate_regions/ben_active_regions_nd_natalia_active_regions_nd__sonia_active_regions_nd__lindsey_active_regions_nd_1_mismatches.zip',dims)
    #        roi_all_match = roi_1_match#np.concatenate([roi_all_match, roi_1_match],axis = 0)
            A_gt_thr = roi_all_match.transpose([1,2,0]).reshape((np.prod(dims),-1),order = 'F')
            A_gt = A_gt_thr 
        else:   
            if filter_SNR: # here you need to adapt to new function estimate_components_quality_auto
                final_frate = params_movie['fr']        
                Npeaks = global_params['Npeaks']
                traces_gt = C_gt + YrA_gt
                idx_filter_snr, idx_filter_snr_bad, fitness_raw_gt, fitness_delta_gt, r_values_gt = estimate_components_quality_auto(
                    traces_gt, Y, A_gt, C_gt, b_gt, f_gt, final_frate=final_frate, Npeaks=Npeaks, r_values_min=1, fitness_min=-10,
                    fitness_delta_min=-10, return_all=True)
                
                print(len(idx_filter_snr))
                print(len(idx_filter_snr_bad))
                A_gt, C_gt, b_gt, f_gt, traces_gt, YrA_gt = A_gt.tocsc()[:,idx_filter_snr], C_gt[idx_filter_snr], b_gt, f_gt, traces_gt[idx_filter_snr], YrA_gt[idx_filter_snr]
            
            A_gt_thr = cm.source_extraction.cnmf.spatial.threshold_components(A_gt.tocsc()[:,:].toarray(), dims, medw=None, thr_method='max', maxthr=0.2, nrgthr=0.99, extract_cc=True,
                             se=None, ss=None, dview=None) 
    #%%
    if plot_on:
        pl.figure()
        crd = plot_contours(A_gt_thr, Cn, thr=.99)
    #%%
    if plot_on:
        threshold = .95
        from caiman.utils.visualization import matrixMontage
        pl.figure()
        matrixMontage(np.squeeze(final_crops[np.where(predictions[:,1]>=threshold)[0]]))
        pl.figure()
        matrixMontage(np.squeeze(final_crops[np.where(predictions[:,0]>=threshold)[0]]))
        #%%
        cm.movie(final_crops).play(gain=3,magnification = 6,fr=5)
        #%%
        cm.movie(np.squeeze(final_crops[np.where(predictions[:,1]>=0.95)[0]])).play(gain=2., magnification = 8,fr=5)
        #%%
        cm.movie(np.squeeze(final_crops[np.where(predictions[:,0]>=0.95)[0]])).play(gain=4., magnification = 8,fr=5)
    #%%
    min_size_neuro = 3*2*np.pi
    max_size_neuro = (2*gSig[0])**2*np.pi
    A_thr = cm.source_extraction.cnmf.spatial.threshold_components(A.tocsc()[:,:].toarray(), dims, medw=None, thr_method='max', maxthr=0.2, nrgthr=0.99, extract_cc=True,
                             se=None, ss=None, dview=dview) 
    
    A_thr = A_thr > 0  
    size_neurons = A_thr.sum(0)
    idx_size_neuro = np.where((size_neurons>min_size_neuro) & (size_neurons<max_size_neuro) )[0]
    #A_thr = A_thr[:,idx_size_neuro]
    print(A_thr.shape)
    #%%
    
#    min_SNR = global_params['min_SNR']                                      # adaptive way to set threshold (will be equal to min_SNR) 
#    thresh_fitness_raw = scipy.special.log_ndtr(-min_SNR)*N_samples
#    thresh_fitness_delta = global_params['max_fitness_delta_accepted']     
#    thresh_fitness_raw_reject = scipy.special.log_ndtr(-0.5)*N_samples
#
#    thresh = global_params['min_cnn_thresh']
#    idx_components_cnn = np.where(predictions[:,neuron_class]>=thresh)[0]
#    print(' ***** ')
#    print((len(final_crops)))
#    print((len(idx_components_cnn)))
#    #print((len(idx_blobs)))    
#    idx_components_r = np.where((r_values >= global_params['rval_thr']))[0]
#    idx_components_raw = np.where(fitness_raw < thresh_fitness_raw)[0]
#    idx_components_delta = np.where(fitness_delta < thresh_fitness_delta)[0]   
#    bad_comps = np.where((r_values <= global_params['min_rval_thr_rejected']) | (fitness_raw >= thresh_fitness_raw_reject) | (predictions[:,neuron_class]<=global_params['max_classifier_probability_rejected']))[0]
#    #idx_and_condition_1 = np.where((r_values >= .65) & ((fitness_raw < -20) | (fitness_delta < -20)) )[0]
#    idx_components = np.union1d(idx_components_r, idx_components_raw)
#    idx_components = np.union1d(idx_components, idx_components_delta)
#    idx_components = np.union1d(idx_components,idx_components_cnn)
#    idx_components = np.setdiff1d(idx_components,bad_comps)
#    #idx_components = np.intersect1d(idx_components,idx_size_neuro)
#    #idx_components = np.union1d(idx_components, idx_and_condition_1)
#    #idx_components = np.union1d(idx_components, idx_and_condition_2)
#    #idx_blobs = np.intersect1d(idx_components, idx_blobs)
#    idx_components_bad = np.setdiff1d(list(range(len(r_values))), idx_components)
    print(' ***** ')
    print((len(r_values)))
    print((len(idx_components)))  
    #%%
    if plot_on:
        pl.figure()
        pl.subplot(1, 2, 1)
        crd = plot_contours(A.tocsc()[:, idx_components], Cn, thr=params_display['thr_plot'], vmax = 0.85)
        pl.subplot(1, 2, 2)
        crd = plot_contours(A.tocsc()[:, idx_components_bad], Cn, thr=params_display['thr_plot'], vmax = 0.85)
    
    #%%
    plot_results = plot_on
    if plot_results:
        pl.figure(figsize=(30,20))
        
#    idx_components = range(len(r_values))
    tp_gt, tp_comp, fn_gt, fp_comp, performance_cons_off =  cm.base.rois.nf_match_neurons_in_binary_masks(A_gt_thr[:,:].reshape([dims[0],dims[1],-1],order = 'F').transpose([2,0,1])*1.,
                                                                                  A_thr[:,idx_components].reshape([dims[0],dims[1],-1],order = 'F').transpose([2,0,1])*1.,thresh_cost=.7, min_dist = 10,
                                                                                  print_assignment= False,plot_results=plot_results,Cn=Cn, labels = ['GT','Offline'])
    
    pl.rcParams['pdf.fonttype'] = 42
    font = {'family' : 'Arial',
            'weight' : 'regular',
            'size'   : 20}
    
    pl.rc('font', **font)
    print({a:b.astype(np.float16) for a,b in performance_cons_off.items()})
    performance_cons_off['fname_new'] = fname_new
    all_perfs.append(performance_cons_off)
    all_rvalues.append(r_values)
    all_comp_SNR_raw.append( comp_SNR)
    all_comp_SNR_delta.append(comp_SNR_delta )
    all_predictions.append(predictionsCNN )
    lbs = np.zeros(len(r_values))
    lbs[tp_comp] = 1
    all_labels.append(lbs)
    print(all_perfs)
    if False:
        #%% LOGISTIC REGRESSOR
        from sklearn import linear_model
        from sklearn.metrics import accuracy_score
        from sklearn import preprocessing
        from sklearn import svm
        from sklearn.neural_network import MLPClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        
        X_SNR_delta = (np.clip(np.concatenate(all_comp_SNR_delta),0,20))
        X_SNR_raw = (np.clip(np.concatenate(all_comp_SNR_raw),0,20))
        X_r = (np.concatenate(all_rvalues))
        X_pred = (np.concatenate(all_predictions))
    #    logreg = linear_model.LogisticRegression( max_iter=100)
    #    logreg = svm.SVC(kernel = 'rbf')
        logreg = MLPClassifier(solver='lbfgs', alpha=1e-5,
                         hidden_layer_sizes=(10,10,5), max_iter = 2000)
    #    logreg = RandomForestClassifier(max_depth=10)
        
        X = np.vstack([X_SNR_delta, X_SNR_raw, X_r, X_pred]).T
        Y = np.concatenate(all_labels)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33)
    #    X = np.vstack([X_SNR_delta - X_SNR_delta.mean(), X_SNR_raw - X_SNR_raw.mean(), X_r - X_r.mean(), X_pred - X_pred.mean()]).T
        
        
        logreg.fit(X_train, Y_train)
        Z = logreg.predict(X_test)
        acc = accuracy_score(Y_test, Z) 
        print(acc)
        #%%
        
        
        with open('component_classifier.pkl', 'wb') as fid:
            pickle.dump(logreg, fid)    
    # load it again
        with open('component_classifier.pkl', 'rb') as fid:
            gnb_loaded = pickle.load(fid)
        #%%
        X_SNR_delta = (np.clip(np.concatenate(all_comp_SNR_delta),0,20))
        X_SNR_raw = (np.clip(np.concatenate(all_comp_SNR_raw),0,20))
        X_r = (np.concatenate(all_rvalues))
        X_pred = (np.concatenate(all_predictions))
        idx_components_r = np.where((X_r >= global_params['rval_thr']))[0]    
        idx_components_raw = np.where(X_SNR_raw > 2)[0]
        idx_components_delta = np.where(X_SNR_delta > 1.5)[0]   
        idx_components_cnn = np.where(X_pred >  global_params['min_cnn_thresh'])[0]
        bad_comps = np.where((X_r <= global_params['min_rval_thr_rejected']) | (X_SNR_raw <= .5) | (X_pred<=global_params['max_classifier_probability_rejected']))[0]
        #idx_and_condition_1 = np.where((r_values >= .65) & ((fitness_raw < -20) | (fitness_delta < -20)) )[0]
        idx_components = np.union1d(idx_components_r, idx_components_raw)
        idx_components = np.union1d(idx_components, idx_components_delta)
        idx_components = np.union1d(idx_components,idx_components_cnn)
        idx_components = np.setdiff1d(idx_components,bad_comps)
        #idx_components = np.intersect1d(idx_components,idx_size_neuro)
        #idx_components = np.union1d(idx_components, idx_and_condition_1)
        #idx_components = np.union1d(idx_components, idx_and_condition_2)
        lab_manual = np.zeros(len(X_r))
        lab_manual[idx_components] = 1
        acc_manual = accuracy_score(Y, lab_manual) 
        print(acc_manual)
    #%%
    if save_on:
        np.savez(os.path.join(os.path.split(fname_new)[0], os.path.split(fname_new)[1][:-4] + 'results_comparison_cnn_after_merge_3.npz'),tp_gt = tp_gt, tp_comp = tp_comp, fn_gt = fn_gt, fp_comp = fp_comp,
                 performance_cons_off = performance_cons_off,idx_components = idx_components, A_gt = A_gt, C_gt = C_gt,
                 b_gt = b_gt , f_gt = f_gt, dims = dims, YrA_gt = YrA_gt, A = A, C = C, b = b, f = f, YrA = YrA, Cn = Cn)
    #%%
    if plot_on:
        view_patches_bar(Yr, scipy.sparse.coo_matrix(A_gt.tocsc()[:, fn_gt]), C_gt[fn_gt, :], b_gt, f_gt, dims[0], dims[1],
                     YrA=YrA_gt[fn_gt, :], img=Cn)
    #%%
        view_patches_bar(Yr, scipy.sparse.coo_matrix(A.tocsc()[:, idx_components[fp_comp]]), C[idx_components[fp_comp]], b, f, dims[0], dims[1],
                     YrA=YrA[idx_components[fp_comp]], img=Cn)
