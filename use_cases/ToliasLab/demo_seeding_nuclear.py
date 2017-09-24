#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""

Demo for detecting ROIs in a structural channel and then seeding CNMF with them.
Detection happens through simple adaptive thresholding of the mean image and 
could potentially be improved. Then the structural channel is processed. Both
offline and online approaches are included. 

The offline approach will only use the seeded masks, whereas online will also 
search for new components during the analysis.

The demo assumes that both channels are motion corrected prior to the analysis. 

@author: epnevmatikakis
"""

try:
    if __IPYTHON__:
        print('Debugging!')
        # this is used for debugging purposes only. allows to reload classes when changed
        get_ipython().magic('load_ext autoreload')
        get_ipython().magic('autoreload 2')
except NameError:
    print('Not IPYTHON')
    pass

import numpy as np
import glob
import pylab as pl
import caiman as cm
from caiman.components_evaluation import evaluate_components
from caiman.source_extraction.cnmf import cnmf as cnmf
from caiman.source_extraction.cnmf.online_cnmf import seeded_initialization
import os
from copy import deepcopy
from caiman.summary_images import max_correlation_image

#%% construct the seeding matrix using the structural channel (note that some components are missed - thresholding can be improved)

filename = '/Users/epnevmatikakis/Documents/Ca_datasets/Tolias/nuclear/gmc_980_30mw_00001_red.tif'
Ain, mR = cm.base.rois.extract_binary_masks_from_structural_channel(cm.load(filename), expand_method='dilation', selem=np.ones((3,3)))
pl.figure(); crd = cm.utils.visualization.plot_contours(Ain.astype('float32'), mR, thr=0.99)
pl.title('Contour plots of detected ROIs in the structural channel')

#%% choose whether to use online algorithm (OnACID) or offline (CNMF)
use_online = False

#%% some common parameters
K = 5  # number of neurons expected per patch (nuisance parameter in this case)
gSig = [7, 7]  # expected half size of neurons
merge_thresh = 0.8  # merging threshold, max correlation allowed
p = 1  # order of the autoregressive system
#%%
if use_online:
    #%% prepare parameters
    fnames = '/Users/epnevmatikakis/Documents/Ca_datasets/Tolias/nuclear/gmc_980_30mw_00001_green.tif'
    rval_thr = .95
    thresh_fitness_delta = -30
    thresh_fitness_raw = -30
    initbatch = 100         # use the first initbatch frames to initialize OnACID
    T1 = 2000               # length of dataset (currently needed to allocate matrices)
    expected_comps = 500    # maximum number of components
    Y = cm.load(fnames, subindices = slice(0,initbatch,None)).astype(np.float32)
    Yr = Y.transpose(1,2,0).reshape((np.prod(Y.shape[1:]),-1), order='F')
        
    #%% run seeded initialization
    cnm_init = seeded_initialization(Y[:initbatch].transpose(1, 2, 0), Ain = Ain, init_batch=initbatch, gnb=1,
                                      gSig=gSig, merge_thresh=0.8,
                                      p=1, minibatch_shape=100, minibatch_suff_stat=5,
                                      update_num_comps=True, rval_thr=rval_thr,
                                      thresh_fitness_delta=thresh_fitness_delta,
                                      thresh_fitness_raw=thresh_fitness_raw,
                                      batch_update_suff_stat=True, max_comp_update_shape=5)
    
    Cn_init = Y.local_correlations(swap_dim = False)
    pl.figure();
    crd = cm.utils.visualization.plot_contours(cnm_init.A.tocsc(), Cn_init, thr=0.9);
    pl.title('Contour plots of detected ROIs in the structural channel')

        # contour plot after seeded initialization. Note how the contours are not clean since there is no activity
        # for most of the ROIs during the first initbatch frames
    #%% run OnACID
    cnm = deepcopy(cnm_init)
    cnm._prepare_object(np.asarray(Yr), T1, expected_comps) # prepare the object to run OnACID
    cnm.max_comp_update_shape = np.inf
    cnm.update_num_comps = True
    t = cnm.initbatch

    Y_ = cm.load(fnames, subindices = slice(t,T1,None)).astype(np.float32)
    
    Cn = max_correlation_image(Y_, swap_dim = False)

    for frame_count, frame in enumerate(Y_):
        if frame_count%100 == 99:
            print([frame_count, cnm.Ab.shape])

#   no motion correction here        
#    templ = cnm.Ab.dot(cnm.C_on[:cnm.M, t - 1]).reshape(cnm.dims, order='F') 
#    frame_cor, shift = motion_correct_iteration_fast(frame, templ, max_shift, max_shift)
#    shifts.append(shift)
        cnm.fit_next(t, frame.copy().reshape(-1, order='F'), simultaneously=True)
        t += 1

    C = cnm.C_on[cnm.gnb:cnm.M]
    A = cnm.Ab[:, cnm.gnb:cnm.M]
    print(('Number of components:' + str(A.shape[-1])))

    #%% plot some results
    pl.figure()     
    crd = cm.utils.visualization.plot_contours(A, Cn, thr=0.9)
    pl.title('Contour plots of components in the functional channel')

    #%% view components. Last components are the components added by OnACID
    dims = Y.shape[1:]
    cm.utils.visualization.view_patches_bar(Yr, A, C, cnm.b, cnm.C_on[:cnm.gnb],
                                            dims[0], dims[1], YrA=cnm.noisyC[cnm.gnb:cnm.M] - C, img=Cn)
#%%    
else:  # run offline CNMF algorithm
    #%% start cluster    
    c,dview,n_processes = cm.cluster.setup_cluster(backend = 'local',n_processes = None,single_thread = False)
    
    #%% FOR LOADING ALL TIFF FILES IN A FILE AND SAVING THEM ON A SINGLE MEMORY MAPPABLE FILE
    
    fnames = ['/Users/epnevmatikakis/Documents/Ca_datasets/Tolias/nuclear/gmc_980_30mw_00001_green.tif'] # can actually be a lost of movie to concatenate
    add_to_movie=0 # the movie must be positive!!!
    downsample_factor= .5 # use .2 or .1 if file is large and you want a quick answer
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
    
    #%% run  seeded CNMF 

    cnm = cnmf.CNMF(n_processes, method_init='greedy_roi', k=Ain.shape[1], gSig=gSig, merge_thresh=merge_thresh,
                    p=p, dview=dview, Ain=Ain,method_deconvolution='oasis',rolling_sum = False, rf=None)
    cnm = cnm.fit(images)
    A, C, b, f, YrA, sn = cnm.A, cnm.C, cnm.b, cnm.f, cnm.YrA, cnm.sn
    
    #%% plot contours of components
    
    pl.figure();
    crd = cm.utils.visualization.plot_contours(cnm.A, Cn, thr=0.9)
    pl.title('Contour plots against correlation image')
    
    #%% evaluate the quality of the components 
    # a lot of components will be removed because presumably they are not active
    # during these 2000 frames of the experiment
    
    final_frate = 15 # approximate frame rate of data
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
        