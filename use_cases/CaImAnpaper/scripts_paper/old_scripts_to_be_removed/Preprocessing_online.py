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
import sys
from caiman.base.rois import detect_duplicates_and_subsets


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
try:
    ID = sys.argv[1]
    ID = str(np.int(ID)-1)
    print('Processing ID:'+ str(ID))
    ID = [np.int(ID)]
    plot_on = False
    save_results = True

except:
    ID = range(6,9)
    print('ID NOT PASSED')
    plot_on = False
    save_results = False


preprocessing_from_scratch = True
if preprocessing_from_scratch:
    reload = False
    save_results = False # set to True if you want to regenerate

mot_corr = False
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
                 'path_to_model' : 'use_cases/CaImAnpaper/cnn_model_online.json', #'use_cases/edge-cutter/binary_cross_bootstrapped.json', #'use_cases/edge-cutter/residual_classifier_2classes.json',
                 'use_peak_max' : True,
                 'thresh_CNN_noisy' : .5,
                 'sniper_mode' : True,
                 'rm_flag' : False,
                 'T_rm' : 1100
                 }

params_movie = [{}]*9        # set up list of dictionaries
#% neurofinder.03.00.test
params_movie[0] = {'fname': 'N.03.00.t/Yr_d1_498_d2_467_d3_1_order_C_frames_2250_.mmap',
                 'folder_name' : 'N.03.00.t/',
                 'ds_factor' : 2,
                 'p': 1,  # order of the autoregressive system
                 'fr': 7,
                 'decay_time': 0.4,
                 'gSig': [12,12],  # expected half size of neurons
                 'gnb': 3,
                 'T1': 2250
                 }
#% neurofinder.04.00.test
params_movie[1] = {'fname': 'N.04.00.t/Yr_d1_512_d2_512_d3_1_order_C_frames_3000_.mmap',
                 'folder_name' : 'N.04.00.t/',
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
params_movie[2] = {'fname': 'N.02.00/Yr_d1_512_d2_512_d3_1_order_C_frames_8000_.mmap',
                 'folder_name' : 'N.02.00/',
                 'ds_factor' : 2,
                 'p': 1,  # order of the autoregressive system
                 'fr' : 30, # imaging rate in Hz
                 'gSig': [8,8],  # expected half size of neuron
                 'decay_time': 0.3,
                 'gnb': 2,
                 'T1':8000,
                 }

#% yuste
params_movie[3] = {'fname': 'YST/Yr_d1_200_d2_256_d3_1_order_C_frames_3000_.mmap',
                 'folder_name': 'YST/',
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
params_movie[4] = {'fname': 'N.00.00/Yr_d1_512_d2_512_d3_1_order_C_frames_2936_.mmap',
                 'folder_name':  'N.00.00/',
                 'ds_factor' : 1,
                 'p': 1,  # order of the autoregressive system
                 'decay_time' : 0.4,
                 'fr' : 16,
                 'gSig': [8,8],  # expected half size of neurons
                 'gnb': 2,
                 'T1' : 2936,
                  }
#% neurofinder.01.01
params_movie[5] = {'fname': 'N.01.01/Yr_d1_512_d2_512_d3_1_order_C_frames_1825_.mmap',
                 'folder_name': 'N.01.01/',
                 'ds_factor' : 1,
                 'p': 1,  # order of the autoregressive system
                 'fr' : 8,
                 'gnb':1,
                 'T1' : 1825,
                 'decay_time' : 1.4,
                 'gSig': [6,6]
                 }
#% Sue Ann k53
params_movie[6] = {'fname': 'K53/Yr_d1_512_d2_512_d3_1_order_C_frames_116043_.mmap',
                 'folder_name':'K53/',
                 'epochs' : 1,
                 'ds_factor' : 2,
                 'thresh_CNN_noisy' : .75,
                 'p': 1,  # order of the autoregressive system
                 'T1': 3000, # number of frames per file
                 'fr': 30,
                 'decay_time' : 0.4,
                 'gSig': [8,8],  # expected half size of neurons
                 'gnb' : 2,
                 }

#% J115
params_movie[7] = {'fname': 'J115/Yr_d1_463_d2_472_d3_1_order_C_frames_90000_.mmap',
                'folder_name':'J115/',
                'epochs' : 1,
                'ds_factor' : 2,
                'thresh_CNN_noisy' : .75,
                'p': 1,  # order of the autoregressive system
                'T1' : 1000,
                'gnb' : 2,
                'fr' : 30,
                'decay_time' : 0.4,
                'gSig' : [8,8]
                 }

#% J123
params_movie[8] = {'fname': 'J123/Yr_d1_458_d2_477_d3_1_order_C_frames_41000_.mmap',
                'folder_name':'J123/',
                 'ds_factor' : 2,
                 'epochs' : 1,
                 'p': 1,  # order of the autoregressive system
                 'fr' : 30,
                 'T1' : 1000,
                 'gnb' : 1,
                 'decay_time' : 0.5,
                 'gSig': [10,10]
                 }

ALL_CCs = []
all_results = dict()
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
if preprocessing_from_scratch:

    for ind_dataset in ID[:]:
        use_mmap = True
        ffls = glob.glob(os.path.abspath(params_movie[ind_dataset]['folder_name'])+'/*.mmap')
        ffls.sort()
        fls = []
        if len(ffls)>1: # avoid selecting the joined memmap file
            for ffll in ffls:
                if '_d1' not in ffll.split('/')[-1][2:5]:
                    print(ffll.split('/')[-1][2:4])
                    fls.append(ffll)
        else:
            fls = ffls


        print(fls)

        #%% Set up some parameters
        ds_factor = params_movie[ind_dataset]['ds_factor']                            # spatial downsampling factor (increases speed but may lose some fine structure)
        gSig = tuple(np.ceil(np.array(params_movie[ind_dataset]['gSig'])/ds_factor).astype(np.int))  # expected half size of neurons
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
        #%%
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
        Cn = Cn_init.copy()

    #%%
        if not reload:

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
            cnm2.thresh_CNN_noisy = thresh_CNN_noisy
            #%% Run OnACID and optionally plot results in real time
            cnm2.when = []
            cnm2.how_many = []
            cnm2.max_comp_update_shape = np.inf
            cnm2.update_num_comps = True
            cnm2.A_epoch = []
            cnm2.added = []
            t = cnm2.initbatch
            tottime = []


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
                        cnm2.how_many.append(len(cnm2.ind_new))

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
                            comps_frame = np.reshape(cnm2.sv,dims)/np.max(img_norm)/5
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
                cnm2.A_epoch.append(cnm2.Ab.copy())
                timings['epoc'].append(time()-t_epoc)

            if save_movie:
                out.release()
            cv2.destroyAllWindows()
            #%%  save results (optional)


            if save_results:
                np.savez(params_movie[ind_dataset]['folder_name']+'results_analysis_online_sensitive_SLURM_01.npz',
                         Cn=Cn, Ab=cnm2.Ab, Cf=cnm2.C_on, b=cnm2.b, f=cnm2.f,
                         dims=cnm2.dims, tottime=tottime, noisyC=cnm2.noisyC, shifts=shifts, img=Cn,
                         params_movie = params_movie[ind_dataset], global_params = global_params, timings = timings, howmany = np.array(cnm2.how_many))

            A, b = cnm2.Ab[:, cnm2.gnb:], cnm2.Ab[:, :cnm2.gnb].toarray()
            C, f = cnm2.C_on[cnm2.gnb:cnm2.M, t-t//epochs:t], cnm2.C_on[:cnm2.gnb, t-t//epochs:t]
            noisyC = cnm2.noisyC[:,t-t//epochs:t]
            if params_movie[ind_dataset]['p'] > 0:
                b_trace = [osi.b for osi in cnm2.OASISinstances]
            if plot_on:
                pl.figure()
                crd = cm.utils.visualization.plot_contours(A, Cn, thr=0.9)
                view_patches_bar(Yr, scipy.sparse.coo_matrix(A.tocsc()[:, :]), C[:, :], b, f,
                             dims[0], dims[1], YrA=noisyC[cnm2.gnb:cnm2.M] - C, img=Cn)


        else:
            print('*****  reloading   **********')
            with np.load(params_movie[ind_dataset]['folder_name']+'results_analysis_online_sensitive_SLURM_01.npz') as ld:
                t = (ld['tottime'].shape[0]+initbatch)
                ld1 = {k:ld[k] for k in ['Ab','Cf','noisyC']}
                locals().update(ld1)

            Ab = Ab[()]
            A, b = Ab[:, gnb:], Ab[:, :gnb].toarray()
            M = A.shape[-1]+gnb
            C, f = Cf[gnb:M, t-t//epochs:t], Cf[:gnb, t-t//epochs:t]
            noisyC = noisyC[:,t-t//epochs:t]

            if plot_on:
                pl.figure()
                crd = cm.utils.visualization.plot_contours(A, Cn, thr=0.9)
                view_patches_bar(Yr, scipy.sparse.coo_matrix(A.tocsc()[:, :]), C[:, :], b, f,
                             dims[0], dims[1], YrA=noisyC[gnb:M] - C, img=Cn)

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
            C_gt = ld['C_gt']
            Cn_orig = ld['Cn']
            #locals().update(ld)
            #A_gt = scipy.sparse.coo_matrix(A_gt[()])
            #dims = (d1,d2)

        if ds_factor > 1:
            A_gt = cm.movie(np.reshape(A_gt,dims_or+(-1,),order='F')).transpose(2,0,1).resize(1./ds_factor,1./ds_factor)
            if plot_on:
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
        #%%
        duplicates_gt, indeces_keep_gt, indeces_remove_gt, D_gt, overlap_gt = detect_duplicates_and_subsets(
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
                pl.imshow(A_gt_thr_bin[:,idx_size_neurons_gt].reshape([dims[0],dims[1],-1],order = 'F').transpose([2,0,1])[np.array(indeces_keep_gt)[:]].sum(0))
                pl.colorbar()
                pl.subplot(1,3,3)
                pl.imshow(A_gt_thr_bin[:,idx_size_neurons_gt].reshape([dims[0],dims[1],-1],order = 'F').transpose([2,0,1])[np.array(indeces_remove_gt)[:]].sum(0))
                pl.colorbar()
                pl.pause(1)
            idx_components_gt = np.delete(idx_components_gt,indeces_remove_gt)
        print('Duplicates gt:'+str(len(duplicates_gt)))

        #%% compute results

        #%% compute results
        use_cnn = False

        if use_cnn:
            thresh_cnn = .10
        else:
            thresh_cnn = 0


        from caiman.components_evaluation import evaluate_components_CNN
        predictions,final_crops = evaluate_components_CNN(A,dims,gSig,model_name = 'use_cases/CaImAnpaper/cnn_model')

        idx_components_cnn = np.where(predictions[:,1]>=thresh_cnn)[0]
        idx_neurons = np.intersect1d(idx_components_cnn,idx_size_neurons)
        #%% detect duplicates
        from caiman.base.rois import detect_duplicates_and_subsets

        duplicates, indeces_keep, indeces_remove, D, overlap = detect_duplicates_and_subsets(
                    A_thr_bin[:,idx_neurons].reshape([dims[0],dims[1],-1],order = 'F').transpose([2,0,1])*1.,
                    predictions[idx_neurons,1], r_values = None,
                    dist_thr=0.1, min_dist = 10, thresh_subset = 0.6)

        idx_components_cnmf = idx_neurons.copy()

        if len(duplicates) > 0:
            if plot_on:

                pl.figure()
                pl.subplot(1,3,1)
                pl.imshow(A_thr_bin[:,idx_neurons].reshape([dims[0],dims[1],-1],order = 'F').transpose([2,0,1])[np.unique(duplicates).flatten()].sum(0))
                pl.colorbar()
                pl.subplot(1,3,2)
                pl.imshow(A_thr_bin[:,idx_neurons].reshape([dims[0],dims[1],-1],order = 'F').transpose([2,0,1])[np.array(indeces_keep)[:]].sum(0))
                pl.colorbar()
                pl.subplot(1,3,3)
                pl.imshow(A_thr_bin[:,idx_neurons].reshape([dims[0],dims[1],-1],order = 'F').transpose([2,0,1])[np.array(indeces_remove)[:]].sum(0))
                pl.colorbar()
                pl.pause(1)
            idx_components_cnmf = np.delete(idx_components_cnmf,indeces_remove)

        print('Duplicates CNMF:'+str(len(duplicates)))


        #%% FIGURE 4 a bottom (need dataset K53!)
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

        #%% GENERATE CORRELATIONS FOR FIGS 5 b,c

        pl.rc('font', **font)
        print(gt_file)
        print('NO CNN')
        print({a:b.astype(np.float16) for a,b in performance_cons_off.items()})
        xcorrs = [scipy.stats.pearsonr(a,b)[0] for a, b in zip(C_gt[idx_components_gt[tp_gt]],C[idx_components_cnmf[tp_comp]])]
        xcorrs = [x for x in xcorrs if str(x) != 'nan']
        ALL_CCs.append(xcorrs)

        performance_tmp = performance_cons_off.copy()
        performance_tmp['A_gt_thr'] = A_gt_thr
        performance_tmp['A_thr'] = A_thr
        performance_tmp['A'] = A
        performance_tmp['A_gt'] = A_gt
        performance_tmp['C'] = C
        performance_tmp['C_gt'] = C_gt

        performance_tmp['idx_components_gt'] = idx_components_gt
        performance_tmp['idx_components_cnmf'] = idx_components_cnmf
        performance_tmp['tp_gt'] = tp_gt
        performance_tmp['tp_comp'] = tp_comp
        performance_tmp['fn_gt'] = fn_gt
        performance_tmp['fp_comp'] = fp_comp
        all_results[params_movie[ind_dataset]['folder_name']] = performance_tmp

    if save_results:
        np.savez('RESULTS_SUMMARY/all_results_Jan_2018_online.npz',all_results = all_results)

else:
    print('**** LOADING PREPROCESSED RESULTS ****')
    with np.load('RESULTS_SUMMARY/all_results_Jan_2018_online.npz') as ld:
        all_results = ld['all_results'][()]

#%% The variables ALL_CCs and all_results contain all the info necessary to create the figures
print([[k,r['f1_score']] for k,r in all_results.items()])
