#!/usr/bin/env python3
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
from caiman.components_evaluation import evaluate_components_CNN
#from caiman.source_extraction.cnmf.online_cnmf import load_object, save_object
from copy import deepcopy
import os

from builtins import str
from builtins import range

try:
    cv2.setNumThreads(1)
except:
    print('Open CV is naturally single threaded')


#%% Select a dataset
# 0: neuforinder.03.00.test
# 1: neurofinder.04.00.test
# 2: neurofinder.02.00
# 3: yuste
# 4: neurofinder.00.00
# 5: neurofinder,01.01
# 6: sue_ann_k53_20160530
# 7: J115
# 8: J123
# 9: sue_ann_k37
#10: Jan-AMG_exp3_001

ind_dataset = 2

#%% set some global parameters here
#'use_cases/edge-cutter/binary_cross_bootstrapped.json'
global_params = {'min_SNR': .75,        # minimum SNR when considering adding a new neuron
                 'gnb' : 2,             # number of background components
                 'epochs' : 2,          # number of passes over the data
                 'rval_thr' : 0.70,     # spatial correlation threshold
                 'batch_length_dt': 10, # length of mini batch for OnACID in decay time units (length would be batch_length_dt*decay_time*fr)
                 'max_thr': 0.25,       # parameter for thresholding components when cleaning up shapes
                 'mot_corr' : False,    # flag for motion correction (set to False to compare directly on the same FOV)
                 'min_num_trial' : 5,   # minimum number of times to attempt to add a component
                 'path_to_model' : 'model/cnn_model_online.json', #'use_cases/edge-cutter/binary_cross_bootstrapped.json', #'use_cases/edge-cutter/residual_classifier_2classes.json',
                 'use_peak_max' : True,
                 'thresh_CNN_noisy' : .5,
                 'sniper_mode' : True,
                 'rm_flag' : False,
                 'T_rm' : 1100
                 }

params_movie = [{}]*11        # set up list of dictionaries
#% neurofinder.03.00.test
params_movie[0] = {'fname': '/mnt/ceph/neuro/labeling/neurofinder.03.00.test/images/final_map/Yr_d1_498_d2_467_d3_1_order_C_frames_2250_.mmap',
                 'folder_name' : '/mnt/ceph/neuro/labeling/neurofinder.03.00.test/',
                 'ds_factor' : 2,
                 'p': 1,  # order of the autoregressive system
                 'fr': 7,
                 'decay_time': 0.4,
                 'gSig': [12,12],  # expected half size of neurons
                 'gnb': 3,
                 'T1': 2250
                 }
#% neurofinder.04.00.test
params_movie[1] = {'fname': '/mnt/ceph/neuro/labeling/neurofinder.04.00.test/images/final_map/Yr_d1_512_d2_512_d3_1_order_C_frames_3000_.mmap',
                 'folder_name' : '/mnt/ceph/neuro/labeling/neurofinder.04.00.test/',
                 'epochs' : 2,
                 'ds_factor' : 1,
                 'p': 1,  # order of the autoregressive system
                 'fr': 8,
                 'gSig': [7,7],  # expected half size of neurons
                 'decay_time' : 0.5, # rough length of a transient
                 'gnb' : 3,
                 'T1' : 3000,
                 }

#% neurofinder 02.00
params_movie[2] = {'fname': '/mnt/ceph/neuro/labeling/neurofinder.02.00/images/final_map/Yr_d1_512_d2_512_d3_1_order_C_frames_8000_.mmap',
                 'folder_name' : '/mnt/ceph/neuro/labeling/neurofinder.02.00/',
                 'ds_factor' : 1,
                 'p': 1,  # order of the autoregressive system
                 'fr' : 30, # imaging rate in Hz
                 'gSig': [8,8],  # expected half size of neuron
                 'decay_time': 0.3,
                 'gnb': 2,
                 'T1':8000,
                 }

#% yuste
params_movie[3] = {'fname': '/mnt/ceph/neuro/labeling/yuste.Single_150u/images/final_map/Yr_d1_200_d2_256_d3_1_order_C_frames_3000_.mmap',
                 'folder_name': '/mnt/ceph/neuro/labeling/yuste.Single_150u/',
                 'epochs' : 2,
                 'ds_factor' : 1,
                 'p': 1,  # order of the autoregressive system
                 'fr' : 10,
                 'decay_time' : .75,
                 'T1' : 3000,
                 'gnb': 3,
                 'gSig': [6,6],  # expected half size of neurons
                 }


#% neurofinder.00.00
params_movie[4] = {'fname': '/mnt/ceph/neuro/labeling/neurofinder.00.00/images/final_map/Yr_d1_512_d2_512_d3_1_order_C_frames_2936_.mmap',
                 'folder_name':  '/mnt/ceph/neuro/labeling/neurofinder.00.00/',
                 'ds_factor' : 1,
                 'p': 1,  # order of the autoregressive system
                 'decay_time' : 0.4,
                 'fr' : 16,
                 'gSig': [8,8],  # expected half size of neurons
                 'gnb': 2,
                 'T1' : 2936,
                  }
#% neurofinder.01.01
params_movie[5] = {'fname': '/mnt/ceph/neuro/labeling/neurofinder.01.01/images/final_map/Yr_d1_512_d2_512_d3_1_order_C_frames_1825_.mmap',
                 'folder_name': '/mnt/ceph/neuro/labeling/neurofinder.01.01/',
                 'ds_factor' : 1,
                 'p': 1,  # order of the autoregressive system
                 'fr' : 8,
                 'gnb':1,
                 'T1' : 1825,
                 'decay_time' : 1.4,
                 'gSig': [7,7]
                 }
#% Sue Ann k53
params_movie[6] = {'fname': '/mnt/ceph/neuro/labeling/k53_20160530/images/final_map/Yr_d1_512_d2_512_d3_1_order_C_frames_116043_.mmap',
                 'folder_name':'/mnt/ceph/neuro/labeling/k53_20160530/',
                 'gtname':'/mnt/ceph/neuro/labeling/k53_20160530/regions/joined_consensus_active_regions.npy',
                 'epochs' : 1,
                 'ds_factor' : 2,
                 'p': 1,  # order of the autoregressive system
                 'T1': 3000, # number of frames per file
                 'fr': 30,
                 'decay_time' : 0.4,
                 'gSig': [8,8],  # expected half size of neurons
                 'gnb' : 2,
                 }

#% J115
params_movie[7] = {'fname': '/mnt/ceph/neuro/labeling/J115_2015-12-09_L01_ELS/images/final_map/Yr_d1_463_d2_472_d3_1_order_C_frames_90000_.mmap',
                'folder_name':'/mnt/ceph/neuro/labeling/J115_2015-12-09_L01_ELS/',
                'gtname':'/mnt/ceph/neuro/labeling/J115_2015-12-09_L01_ELS/regions/joined_consensus_active_regions.npy',
                'epochs' : 1,
                'ds_factor' : 2,
                'p': 1,  # order of the autoregressive system
                'T1' : 1000,
                'gnb' : 2,
                'fr' : 30,
                'decay_time' : 0.4,
                'gSig' : [8,8]
                 }

#% J123
params_movie[8] = {'fname': '/mnt/ceph/neuro/labeling/J123_2015-11-20_L01_0/images/final_map/Yr_d1_458_d2_477_d3_1_order_C_frames_41000_.mmap',
                'folder_name':'/mnt/ceph/neuro/labeling/J123_2015-11-20_L01_0/',
                'gtname':'/mnt/ceph/neuro/labeling/J123_2015-11-20_L01_0/regions/joined_consensus_active_regions.npy',
                 'ds_factor' : 2,
                 'epochs' : 1,
                 'p': 1,  # order of the autoregressive system
                 'fr' : 30,
                 'T1' : 1000,
                 'gnb' : 2,
                 'decay_time' : 0.5,
                 'gSig': [10,10]
                 }

#% Sue Ann k37
params_movie[9] = {'fname' : '/mnt/ceph/neuro/labeling/k37_20160109_AM_150um_65mW_zoom2p2_00001_1-16/images/final_map/Yr_d1_512_d2_512_d3_1_order_C_frames_48000_.mmap',
                    'folder_name' : '/mnt/ceph/neuro/labeling/k37_20160109_AM_150um_65mW_zoom2p2_00001_1-16/',
                    'gtname' : '/mnt/ceph/neuro/labeling/k37_20160109_AM_150um_65mW_zoom2p2_00001_1-16/regions/joined_consensus_active_regions.npy',
                    'ds_factor' : 2,
                    'p' : 1,
                    'fr' : 30,
                    'T1' : 3000,
                    'gnb' : 2,
                    'decay_time' : 0.5,
                    'gSig' : [8,8]
                    }
#% Jan-AMG_exp3_001
params_movie[10] = {'fname' : '/mnt/ceph/neuro/labeling/Jan-AMG_exp3_001/images/final_map/Yr_d1_512_d2_512_d3_1_order_C_frames_115897_.mmap',
                    'folder_name' : '/mnt/ceph/neuro/labeling/Jan-AMG_exp3_001/',
                    'gt_name' : '/mnt/ceph/neuro/labeling/Jan-AMG_exp3_001/regions/joined_consensus_active_regions.npy',
                    'ds_factor' : 2,
                    'p' : 1,
                    'fr' : 30,
                    'T1' : 1002,
                    'gnb' : 2,
                    'decay_time' : 0.5,
                    'gSig' : [7,7]
                    }

#%% convert mmaps into tifs
#import os.path
#
#for ind_dataset in [10]:
#    fls = glob.glob(params_movie[ind_dataset]['folder_name']+'images/mmap/*.mmap')
#    for file_count, ffll in enumerate(fls):
#        file_name = '/'.join(ffll.split('/')[:-2]+['mmap_tifs']+[ffll.split('/')[-1][:-4]+'tif'])
#        if not os.path.isfile(file_name):
#            fl_temp = cm.movie(np.array(cm.load(ffll)))
#            fl_temp.save(file_name)
#        print(file_name)
#    print(ind_dataset)
#%%  download and list all files to be processed

mot_corr = global_params['mot_corr']
use_VST = False
use_mmap = True

if mot_corr:
    fls = glob.glob('/'.join( params_movie[ind_dataset]['fname'].split('/')[:-3]+['images','tifs','*.tif']))
    template = cm.load( '/'.join( params_movie[ind_dataset]['fname'].split('/')[:-3]+['projections','median_projection.tif']))
else:
    if use_mmap:
        fls = glob.glob('/'.join( params_movie[ind_dataset]['fname'].split('/')[:-3]+['images','mmap','*.mmap']))
    elif not use_VST:
        fls = glob.glob('/'.join( params_movie[ind_dataset]['fname'].split('/')[:-3]+['images','mmap_tifs','*.tif']))
    else:
        fls = glob.glob('/'.join( params_movie[ind_dataset]['fname'].split('/')[:-3]+['images','tiff_VST','*.tif']))

fls.sort()
print(fls)

#%% Set up some parameters
ds_factor = params_movie[ind_dataset]['ds_factor']                            # spatial downsampling factor (increases speed but may lose some fine structure)
gSig = tuple(np.ceil(np.array(params_movie[ind_dataset]['gSig'])/ds_factor).astype(int))  # expected half size of neurons
init_files = 1                                                       # number of files used for initialization
online_files = len(fls) - 1                                          # number of files used for online
initbatch = 200                                                      # number of frames for initialization (presumably from the first file)
expected_comps = 4000                                                # maximum number of expected components used for memory pre-allocation (exaggerate here)
K = 2                                                                # initial number of components
min_SNR = global_params['min_SNR']
min_SNR = 3.0956*np.log(len(fls)*params_movie[ind_dataset]['T1']/(307.85*params_movie[ind_dataset]['fr']) + 0.7421)
N_samples = np.ceil(params_movie[ind_dataset]['fr']*params_movie[ind_dataset]['decay_time'])   # number of timesteps to consider when testing new neuron candidates
pr_inc = 1 - scipy.stats.norm.cdf(global_params['min_SNR'])           # inclusion probability of noise transient
thresh_fitness_raw = np.log(pr_inc)*N_samples       # event exceptionality threshold
thresh_fitness_delta = -80.                         # make this very neutral
p = params_movie[ind_dataset]['p']                  # order of AR indicator dynamics
#rval_thr = global_params['rval_thr']                # correlation threshold for new component inclusion
rval_thr = 0.06*np.log(len(fls)*params_movie[ind_dataset]['T1']/(2177.*params_movie[ind_dataset]['fr'])-0.0462) +0.8862

try:
    gnb = params_movie[ind_dataset]['gnb']
except:
    gnb = global_params['gnb']

try:
    epochs = params_movie[ind_dataset]['epochs']                     # number of background components
except:
    epochs = global_params['epochs']                    # number of passes over the data

try:
    thresh_CNN_noisy = params_movie[ind_dataset]['thresh_CNN_noisy']
except:
    thresh_CNN_noisy = global_params['thresh_CNN_noisy']

try:
    min_num_trial = params_movie[ind_dataset]['min_num_trial']
except:
    min_num_trial = global_params['min_num_trial']

T1 = params_movie[ind_dataset]['T1']*len(fls)*epochs        # total length of all files (if not known use a large number, then truncate at the end)
#minibatch_length = int(global_params['batch_length_dt']*params_movie[ind_dataset]['fr']*params_movie[ind_dataset]['decay_time'])


#%% Initialize timing structure

timings = {'init' : 0,
           'corr' : 0,
           'load' : 0,
           'deep' : 0,
           'fitn' : 0,
           'remv' : 0,
           'epoc' : []
            }


#%%    Initialize movie
t_init = time()
if ds_factor > 1:                                   # load only the first initbatch frames and possibly downsample them
    Y = cm.load(fls[0], subindices = slice(0,initbatch,None)).astype(np.float32).resize(1. / ds_factor, 1. / ds_factor)
else:
    Y =  cm.load(fls[0], subindices = slice(0,initbatch,None)).astype(np.float32)

if mot_corr:                                        # perform motion correction on the first initbatch frames
    max_shift = np.ceil(5./ds_factor).astype('int')     # maximum allowed shift during motion correction
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

t_corr = time()
Cn_init = Y.local_correlations(swap_dim = False)    # compute correlation image
timings['corr'] = time() - t_corr

#%% initialize OnACID with bare initialization
rval_thr = 1
cnm_init = bare_initialization(Y[:initbatch].transpose(1, 2, 0), init_batch=initbatch, k=K, gnb=gnb,
                                 gSig=gSig, p=p, minibatch_shape=100, minibatch_suff_stat=5,
                                 update_num_comps = True, rval_thr=rval_thr,
                                 thresh_fitness_delta = thresh_fitness_delta,
                                 thresh_fitness_raw = thresh_fitness_raw, use_dense = False,
                                 batch_update_suff_stat=True, max_comp_update_shape = 200,
                                 deconv_flag = True,  thresh_CNN_noisy = thresh_CNN_noisy,
                                 simultaneously = False, n_refit = 0)

#crd = plot_contours(cnm_init.A.tocsc(), Cn_init, thr=0.9)

#% Plot initialization results

#A, C, b, f, YrA, sn = cnm_init.A, cnm_init.C, cnm_init.b, cnm_init.f, cnm_init.YrA, cnm_init.sn
#view_patches_bar(Yr, scipy.sparse.coo_matrix(A.tocsc()[:, :]), C[:, :], b, f, dims[0], dims[1], YrA=YrA[:, :], img=Cn_init)

#%% Prepare object for OnACID
t_deep = time()
cnm2 = deepcopy(cnm_init)
timings['deep'] = time() - t_deep

cnm2._prepare_object(np.asarray(Yr[:,:initbatch]), T1, expected_comps, idx_components=None, N_samples_exceptionality = int(N_samples),
                         max_num_added = min_num_trial,
                         min_num_trial = min_num_trial,
                         sniper_mode = global_params['sniper_mode'],
                         path_to_model = global_params['path_to_model'], use_peak_max = global_params['use_peak_max'])

timings['init'] = time() - t_init
#%% Run OnACID and optionally plot results in real time

cnm2.max_comp_update_shape = np.inf
cnm2.update_num_comps = True
cnm2.A_epoch = []
cnm2.added = []
t = cnm2.initbatch
tottime = []
Cn = Cn_init.copy()

remove_flag = global_params['rm_flag']
T_rm = global_params['T_rm']
rm_thr = 0.1
plot_contours_flag = False               # flag for plotting contours of detected components at the end of each file
play_reconstr = False                    # flag for showing video with results online (turn off flags for improving speed)
save_movie = False                       # flag for saving movie (file could be quite large..)
movie_name = params_movie[ind_dataset]['folder_name'] + 'output.avi' # name of movie to be saved
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
    t_epoc = time()
    if iter > 0:
        process_files = fls[:init_files + online_files]     # if not on first epoch process all files from scratch
        init_batc_iter = [0]*(online_files+init_files)      #

    for file_count, ffll in enumerate(process_files):  # np.array(fls)[np.array([1,2,3,4,5,-5,-4,-3,-2,-1])]:
        print('Now processing file ' + ffll)
        t_load = time()
        Y_ = cm.load(ffll, subindices=slice(init_batc_iter[file_count],T1,None), in_memory=True)
        
        timings['load'] += time() - t_load

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
            if t % 200 == 0:
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
            timings['fitn'] += time() - t1
            tottime.append(time() - t1)                             # store time

            t += 1
            #    break

            t_remv = time()
            if t % T_rm == 0 and remove_flag and (t + 1000 < T1):
                prd, _ = evaluate_components_CNN(cnm2.Ab[:, gnb:], dims, gSig)
                ind_rem = np.where(prd[:, 1] < rm_thr)[0].tolist()
                cnm2.remove_components(ind_rem)
                print('Removing '+str(len(ind_rem))+' components')
            timings['remv'] += time() - t_remv

            if t % 1000 == 0 and plot_contours_flag:
            #if t>=4500:tours_flag:
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
                frame_comp_1 = cv2.resize(np.concatenate([frame_/np.max(img_norm),all_comps*3.],axis = -1),(2*int(cnm2.dims[1]*resize_fact),int(cnm2.dims[0]*resize_fact) ))
                frame_comp_2 = cv2.resize(np.concatenate([comps_frame*10.,comps_frame+bgkrnd_frame],axis = -1),(2*int(cnm2.dims[1]*resize_fact),int(cnm2.dims[0]*resize_fact) ))
                frame_pn = np.concatenate([frame_comp_1,frame_comp_2],axis=0).T
                vid_frame = np.repeat(frame_pn[:,:,None],3,axis=-1)
                vid_frame = np.minimum((vid_frame*255.),255).astype('u1')
                cv2.putText(vid_frame,'Raw Data',(5,20),fontFace = 5, fontScale = 1.2, color = (0,255,0), thickness = 1)
                cv2.putText(vid_frame,'Inferred Activity',(int(cnm2.dims[0]*resize_fact) + 5,20),fontFace = 5, fontScale = 1.2, color = (0,255,0), thickness = 1)
                cv2.putText(vid_frame,'Identified Components',(5,int(cnm2.dims[1]*resize_fact)  + 20),fontFace = 5, fontScale = 1.2, color = (0,255,0), thickness = 1)
                cv2.putText(vid_frame,'Denoised Data',(int(cnm2.dims[0]*resize_fact) + 5 ,int(cnm2.dims[1]*resize_fact)  + 20),fontFace = 5, fontScale = 1.2, color = (0,255,0), thickness = 1)
                cv2.putText(vid_frame,'Frame = '+str(t),(vid_frame.shape[1]//2-vid_frame.shape[1]//10,vid_frame.shape[0]-20),fontFace = 5, fontScale = 1.2, color = (0,255,255), thickness = 1)
                if save_movie:
                    out.write(vid_frame)
                cv2.imshow('frame',vid_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        print('Cumulative processing speed is ' + str((t - initbatch) / np.sum(tottime))[:5] + ' frames per second.')
    cnm2.A_epoch.append(cnm2.Ab.copy())
    timings['epoc'].append(time()-t_epoc)

if save_movie:
    out.release()
cv2.destroyAllWindows()
#%%  save results (optional)

save_results = False

if save_results:
    np.savez(params_movie[ind_dataset]['folder_name']+'results_analysis_online_sensitive_EP_05.npz',
             Cn=Cn, Ab=cnm2.Ab, Cf=cnm2.C_on, b=cnm2.b, f=cnm2.f,
             dims=cnm2.dims, tottime=tottime, noisyC=cnm2.noisyC, shifts=shifts, img=Cn,
             params_movie = params_movie[ind_dataset], global_params = global_params, timings = timings)

#%% extract results from the objects and do some plotting
A, b = cnm2.Ab[:, cnm2.gnb:], cnm2.Ab[:, :cnm2.gnb].toarray()
C, f = cnm2.C_on[cnm2.gnb:cnm2.M, t-t//epochs:t], cnm2.C_on[:cnm2.gnb, t-t//epochs:t]
noisyC = cnm2.noisyC[:,t-t//epochs:t]
if params_movie[ind_dataset]['p'] > 0:
    b_trace = [osi.b for osi in cnm2.OASISinstances]

pl.figure()
crd = cm.utils.visualization.plot_contours(A, Cn, thr=0.9)

#%%

view_patches_bar(Yr, scipy.sparse.coo_matrix(A.tocsc()[:, :]), C[:, :], b, f,
                 dims[0], dims[1], YrA=noisyC[cnm2.gnb:cnm2.M] - C, img=Cn)

#%% load, threshold and filter for size ground truth
#global_params['max_thr'] = 0.25
c, dview, n_processes = cm.cluster.setup_cluster(backend='local', n_processes=None, single_thread = True)

gt_file = os.path.join(os.path.split(params_movie[ind_dataset]['fname'])[0], os.path.split(params_movie[ind_dataset]['fname'])[1][:-4] + 'match_masks.npz')
min_radius = max(gSig[0]/2.,2.)          # minimum acceptable radius
max_radius = 2.*gSig[0]                  # maximum acceptable radius
min_size_neuro = min_radius**2*np.pi
max_size_neuro = max_radius**2*np.pi

with np.load(gt_file, encoding = 'latin1') as ld:
    print(ld.keys())
    d1_or = int(ld['d1'])
    d2_or = int(ld['d2'])
    dims_or = (d1_or,d2_or)
    A_gt = ld['A_gt'][()].toarray()
    Cn_orig = ld['Cn']
    #locals().update(ld)
    #A_gt = scipy.sparse.coo_matrix(A_gt[()])
    #dims = (d1,d2)

if ds_factor > 1:
    A_gt = cm.movie(np.reshape(A_gt,dims_or+(-1,),order='F')).transpose(2,0,1).resize(1./ds_factor,1./ds_factor)
    pl.figure(); pl.imshow(A_gt.sum(0))
    A_gt2 = np.array(np.reshape(A_gt,(A_gt.shape[0],-1),order='F')).T
    Cn_orig = cv2.resize(Cn_orig,None,fx=1./ds_factor,fy=1./ds_factor)
else:
    A_gt2 = A_gt.copy()

A_gt_thr = cm.source_extraction.cnmf.spatial.threshold_components(A_gt2, dims, medw=None, thr_method='max', maxthr=global_params['max_thr'], extract_cc=True,
                         se=None, ss=None, dview=None)

A_gt_thr_bin = A_gt_thr > 0
size_neurons_gt = A_gt_thr_bin.sum(0)
idx_size_neurons_gt = np.where((size_neurons_gt>min_size_neuro) & (size_neurons_gt < max_size_neuro) )[0]
print(A_gt_thr.shape)
#%% filter for size found neurons

A_thr = cm.source_extraction.cnmf.spatial.threshold_components(A.tocsc()[:,:].toarray(), dims, medw=None, thr_method='max', maxthr=global_params['max_thr'], extract_cc=True,
                         se=None, ss=None, dview=dview)
A_thr_bin = A_thr > 0
size_neurons = A_thr_bin.sum(0)
idx_size_neurons = np.where((size_neurons>min_size_neuro) & (size_neurons<max_size_neuro))[0]
#A_thr = A_thr[:,idx_size_neuro]
print(A_thr.shape)

#%% compute results

use_cnn = True  # Use CNN classifier
if use_cnn:
    from caiman.components_evaluation import evaluate_components_CNN
    predictions,final_crops = evaluate_components_CNN(A,dims,gSig,model_name = 'model/cnn_model')
    thresh_cnn = .0
    idx_components_cnn = np.where(predictions[:,1]>=thresh_cnn)[0]
    idx_neurons = np.intersect1d(idx_components_cnn,idx_size_neurons)
else:
    idx_neurons = idx_size_neurons.copy()


#%% detect duplicates

#%%

from caiman.base.rois import detect_duplicates_and_subsets

duplicates, indices_keep, indices_remove, D, overlap = detect_duplicates_and_subsets(
            A_thr_bin[:,idx_neurons].reshape([dims[0],dims[1],-1],order = 'F').transpose([2,0,1])*1.,
            predictions[idx_neurons,1], r_values = None,
            dist_thr=0.1, min_dist = 10, thresh_subset = 0.6)

idx_components_cnmf = idx_neurons.copy()

plot_on = False
if len(duplicates) > 0:
    if plot_on:

        pl.figure()
        pl.subplot(1,3,1)
        pl.imshow(A_thr_bin[:,idx_neurons].reshape([dims[0],dims[1],-1],order = 'F').transpose([2,0,1])[np.unique(duplicates).flatten()].sum(0))
        pl.colorbar()
        pl.subplot(1,3,2)
        pl.imshow(A_thr_bin[:,idx_neurons].reshape([dims[0],dims[1],-1],order = 'F').transpose([2,0,1])[np.array(indices_keep)[:]].sum(0))
        pl.colorbar()
        pl.subplot(1,3,3)
        pl.imshow(A_thr_bin[:,idx_neurons].reshape([dims[0],dims[1],-1],order = 'F').transpose([2,0,1])[np.array(indices_remove)[:]].sum(0))
        pl.colorbar()
        pl.pause(1)
    idx_components_cnmf = np.delete(idx_components_cnmf,indices_remove)

print('Duplicates CNMF:'+str(len(duplicates)))

#%%

duplicates_gt, indices_keep_gt, indices_remove_gt, D_gt, overlap_gt = detect_duplicates_and_subsets(
        A_gt_thr_bin[:,idx_size_neurons_gt].reshape([dims[0],dims[1],-1],order = 'F').transpose([2,0,1])*1.,
        predictions = None, r_values = None,
        dist_thr=0.1, min_dist = 10,thresh_subset = 0.6)

idx_components_gt = idx_size_neurons_gt.copy()

if len(duplicates_gt) > 0:
    if plot_on:
        pl.figure()
        pl.subplot(1,3,1)
        pl.imshow(A_gt_thr_bin[:,idx_size_neurons_gt].reshape([dims[0],dims[1],-1],order = 'F').transpose([2,0,1])[np.array(duplicates_gt).flatten()].sum(0))
        pl.colorbar()
        pl.subplot(1,3,2)
        pl.imshow(A_gt_thr_bin[:,idx_size_neurons_gt].reshape([dims[0],dims[1],-1],order = 'F').transpose([2,0,1])[np.array(indices_keep_gt)[:]].sum(0))
        pl.colorbar()
        pl.subplot(1,3,3)
        pl.imshow(A_gt_thr_bin[:,idx_size_neurons_gt].reshape([dims[0],dims[1],-1],order = 'F').transpose([2,0,1])[np.array(indices_remove_gt)[:]].sum(0))
        pl.colorbar()
        pl.pause(1)
    idx_components_gt = np.delete(idx_components_gt,indices_remove_gt)
print('Duplicates gt:'+str(len(duplicates_gt)))

#%%
plot_results = False
if plot_results:
    pl.figure(figsize=(30,20))

tp_gt, tp_comp, fn_gt, fp_comp, performance_cons_off =  cm.base.rois.nf_match_neurons_in_binary_masks(A_gt_thr_bin[:,idx_components_gt].reshape([dims[0],dims[1],-1],order = 'F').transpose([2,0,1])*1.,
                                                                              A_thr_bin[:,idx_components_cnmf].reshape([dims[0],dims[1],-1],order = 'F').transpose([2,0,1])*1.,thresh_cost=.8, min_dist = 10,
                                                                              print_assignment= False,plot_results=plot_results,Cn=Cn_orig, labels = ['GT','Offline'])

pl.rcParams['pdf.fonttype'] = 42
font = {'family' : 'Arial',
        'weight' : 'regular',
        'size'   : 20}

pl.rc('font', **font)
print({a:b.astype(np.float16) for a,b in performance_cons_off.items()})
##%%
## =============================================================================
##
## =============================================================================
#with np.load(params_movie[ind_dataset]['folder_name']+'results_analysis_online_standard.npz') as  ld:
#    print(ld.keys())
#    locals().update(ld)
#Ab = Ab[()]
##%%
#
#A, b = Ab[:, b.shape[-1]:], Ab[:, :b.shape[-1]].toarray()
#C, f = Cf[b.shape[-1]:Ab.shape[-1], :], Cf[:b.shape[-1], :]
##noisyC = noisyC[:,:]
##if params_movie[ind_dataset]['p'] > 0:
##    b_trace = [osi.b for osi in cnm2.OASISinstances]
#A_us = np.array([cv2.resize(a.toarray().reshape(dims,order='F').T,Cn.shape).reshape(-1) for a in A.tocsc().T]).T
#A_gt_thr_us = np.array([cv2.resize(a.reshape(dims,order='F').T,Cn.shape).reshape(-1) for a in A_gt_thr.T]).T
#
##%%
#
#
#with np.load('/mnt/ceph/neuro/labeling/J115_2015-12-09_L01_ELS/images/final_map/Yr_d1_463_d2_472_d3_1_order_C_frames_90000_.results_analysis_after_merge_4.npz') as ld:
#    print(ld.keys())
#    Cn = ld['Cn']
#    C_gt = ld['C_gt']
#
#
#with np.load('/mnt/ceph/neuro/DataForPublications/caiman-paper/all_results_Jan_2018.npz') as ld:
#     all_results = ld['all_results']
#     all_results = all_results[()]
#
#ld = all_results['Datak53_20160530']
#ag_idx = ld['idx_components_cnmf'][ld['tp_comp']][np.argsort(ld['predictionsCNN'][ld['idx_components_cnmf'][ld['tp_comp']]])[[-6,-5,-4,-3,-2]]]
#[np.where(ld['tp_comp']==aaa)[0] for aaa in ag_idx]
##pl.imshow(Cn)
#
##%%
#pl.figure()
#
#pl.subplot(1,2,1)
#
#a1 = plot_contours(A_us[:, idx_components_cnmf[tp_comp]], Cn, thr=0.9, colors='yellow', vmax = 0.75, display_numbers=False,cmap = 'gray')
#
#a2 = plot_contours(A_gt_thr_us[:, idx_components_gt[tp_gt]], Cn, thr=0.9, vmax = 0.85, colors='r', display_numbers=False,cmap = 'gray')
#
#pl.subplot(1,2,2)
#
#a3 = plot_contours(A_us[:, idx_components_cnmf[fp_comp]], Cn, thr=0.9, colors='yellow', vmax = 0.75, display_numbers=False,cmap = 'gray')
#
#a4 = plot_contours(A_gt_thr_us[:, idx_components_gt[fn_gt]], Cn, thr=0.9, vmax = 0.85, colors='r', display_numbers=False,cmap = 'gray')
#
##%%
#
#pl.figure()
#
#pl.ylabel('spatial components')
#
#idx_comps_high_r = [np.argsort(predictions[:,1][idx_components_cnmf[tp_comp]])[[-6,-5,-4,-3,-2]]]
#
#idx_comps_high_r_cnmf = idx_components_cnmf[tp_comp][idx_comps_high_r]
#
#idx_comps_high_r_gt = idx_components_gt[tp_gt][idx_comps_high_r]
#
#images_nice = (A_us[:,idx_comps_high_r_cnmf].reshape(Cn.shape+(-1,),order = 'F')).transpose(2,0,1)
#images_nice_gt =  (A_gt_thr_us[:,idx_comps_high_r_gt].reshape(Cn.shape+(-1,),order = 'F')).transpose(2,0,1)
#
#cms = np.array([scipy.ndimage.center_of_mass(img) for img in images_nice]).astype(int)
#
#images_nice_crop = [img[cm_[0]-15:cm_[0]+15,cm_[1]-15:cm_[1]+15] for  cm_,img in zip(cms,images_nice)]
#
#images_nice_crop_gt = [img[cm_[0]-15:cm_[0]+15,cm_[1]-15:cm_[1]+15] for  cm_,img in zip(cms,images_nice_gt)]
#
#
#
#indexes = [1,3,5,7,9,2,4,6,8,10]
#
#count = 0
#
#for img in images_nice_crop:
#
#    pl.subplot(5,2,indexes[count])
#
#    pl.imshow(img)
#
#    pl.axis('off')
#
#    count += 1
#
#
#
#for img in images_nice_crop_gt:
#
#
#
#    pl.subplot(5,2,indexes[count])
#
#    pl.imshow(img)
#
#    pl.axis('off')
#
#    count += 1
#
##%%
#
#pl.figure()
#
#traces_gt = C_gt[idx_comps_high_r_gt]# + YrA_gt[idx_comps_high_r_gt]
#
#traces_cnmf = C[idx_comps_high_r_cnmf]# + YrA[idx_comps_high_r_cnmf]
#
#traces_gt/=np.max(traces_gt,1)[:,None]
#
#traces_cnmf /=np.max(traces_cnmf,1)[:,None]
#
#pl.plot(scipy.signal.decimate(traces_cnmf,10,1).T-np.arange(5)*1,'y')
#
#pl.plot(scipy.signal.decimate(traces_gt,10,1).T-np.arange(5)*1,'k', linewidth = .5 )