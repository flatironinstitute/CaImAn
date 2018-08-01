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

import caiman as cm
from caiman.source_extraction import cnmf as cnmf
from caiman.source_extraction.cnmf.estimates import Estimates, compare_components
from caiman.source_extraction.cnmf.online_cnmf import load_OnlineCNMF
import pylab as pl
import scipy
import cv2
import glob
import os
import sys
from caiman.base.rois import detect_duplicates_and_subsets

from builtins import str

try:
    cv2.setNumThreads(1)
except:
    print('Open CV is naturally single threaded')

# %% Select a dataset
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
# 10: Jan-AMG_exp3_001
try:
    if 'pydevconsole' in sys.argv[0]:
        raise Exception()
    ID = sys.argv[1]
    ID = str(np.int(ID) - 1)
    print('Processing ID:' + str(ID))
    ID = [np.int(ID)]
    plot_on = False
    save_results = True

except:
    ID = [0, 3]  # range(6,9)
    print('ID NOT PASSED')
    plot_on = False
    save_results = False

preprocessing_from_scratch = True
if preprocessing_from_scratch:
    reload = False
    save_results = False  # set to True if you want to regenerate

mot_corr = False
base_folder = '/mnt/ceph/users/agiovann/LinuxDropbox/Dropbox/DATA_PAPER_ELIFE/'
# %% set some global parameters here
# 'use_cases/edge-cutter/binary_cross_bootstrapped.json'
global_params = {'min_SNR': .75,  # minimum SNR when considering adding a new neuron
                 'gnb': 2,  # number of background components
                 'epochs': 2,  # number of passes over the data
                 'rval_thr': 0.70,  # spatial correlation threshold
                 'batch_length_dt': 10,
                 # length of mini batch for OnACID in decay time units (length would be batch_length_dt*decay_time*fr)
                 'max_thr': 0.25,  # parameter for thresholding components when cleaning up shapes
                 'mot_corr': False,  # flag for motion correction (set to False to compare directly on the same FOV)
                 'min_num_trial': 5,  # minimum number of times to attempt to add a component
                 'use_peak_max': True,
                 'thresh_CNN_noisy': .5,
                 'sniper_mode': True,
                 'merge_thr': 0.8
                 }

params_movie = [{}] * 9  # set up list of dictionaries
# % neurofinder.03.00.test
params_movie[0] = {
    # 'fname': '/mnt/ceph/neuro/labeling/neurofinder.03.00.test/images/final_map/Yr_d1_498_d2_467_d3_1_order_C_frames_2250_.mmap',
    # 'folder_name' : '/mnt/ceph/neuro/labeling/neurofinder.03.00.test/',
    'fname': 'N.03.00.t/Yr_d1_498_d2_467_d3_1_order_C_frames_2250_.mmap',
    'folder_name': 'N.03.00.t/',
    'ds_factor': 2,
    'p': 1,  # order of the autoregressive system
    'fr': 7,
    'decay_time': 0.4,
    'gSig': [12, 12],  # expected half size of neurons
    'gnb': 3,
    'T1': 2250
}
# % neurofinder.04.00.test
params_movie[1] = {
    # 'fname': '/mnt/ceph/neuro/labeling/neurofinder.04.00.test/images/final_map/Yr_d1_512_d2_512_d3_1_order_C_frames_3000_.mmap',
    # 'folder_name' : '/mnt/ceph/neuro/labeling/neurofinder.04.00.test/',
    'fname': 'N.04.00.t/Yr_d1_512_d2_512_d3_1_order_C_frames_3000_.mmap',
    'folder_name': 'N.04.00.t/',
    'epochs': 2,
    'ds_factor': 1,
    'p': 1,  # order of the autoregressive system
    'fr': 8,
    'gSig': [7, 7],  # expected half size of neurons
    'decay_time': 0.5,  # rough length of a transient
    'gnb': 3,
    'T1': 3000,
}

# % neurofinder 02.00
params_movie[2] = {
    # 'fname': '/mnt/ceph/neuro/labeling/neurofinder.02.00/images/final_map/Yr_d1_512_d2_512_d3_1_order_C_frames_8000_.mmap',
    # 'folder_name' : '/mnt/ceph/neuro/labeling/neurofinder.02.00/',
    'fname': 'N.02.00/Yr_d1_512_d2_512_d3_1_order_C_frames_8000_.mmap',
    'folder_name': 'N.02.00/',
    'ds_factor': 2,
    'p': 1,  # order of the autoregressive system
    'fr': 30,  # imaging rate in Hz
    'gSig': [8, 8],  # expected half size of neuron
    'decay_time': 0.3,
    'gnb': 2,
    'T1': 8000,
}

# % yuste
params_movie[3] = {
    # 'fname': '/mnt/ceph/neuro/labeling/yuste.Single_150u/images/final_map/Yr_d1_200_d2_256_d3_1_order_C_frames_3000_.mmap',
    # 'folder_name': '/mnt/ceph/neuro/labeling/yuste.Single_150u/',
    'fname': 'YST/Yr_d1_200_d2_256_d3_1_order_C_frames_3000_.mmap',
    'folder_name': 'YST/',
    'epochs': 2,
    'ds_factor': 1,
    'p': 1,  # order of the autoregressive system
    'fr': 10,
    'decay_time': .75,
    'T1': 3000,
    'gnb': 3,
    'gSig': [6, 6],  # expected half size of neurons
}

# % neurofinder.00.00
params_movie[4] = {
    # 'fname': '/mnt/ceph/neuro/labeling/neurofinder.00.00/images/final_map/Yr_d1_512_d2_512_d3_1_order_C_frames_2936_.mmap',
    # 'folder_name':  '/mnt/ceph/neuro/labeling/neurofinder.00.00/',
    'fname': 'N.00.00/Yr_d1_512_d2_512_d3_1_order_C_frames_2936_.mmap',
    'folder_name': 'N.00.00/',
    'ds_factor': 1,
    'p': 1,  # order of the autoregressive system
    'decay_time': 0.4,
    'fr': 16,
    'gSig': [8, 8],  # expected half size of neurons
    'gnb': 2,
    'T1': 2936,
}
# % neurofinder.01.01
params_movie[5] = {
    # 'fname': '/mnt/ceph/neuro/labeling/neurofinder.01.01/images/final_map/Yr_d1_512_d2_512_d3_1_order_C_frames_1825_.mmap',
    # 'folder_name': '/mnt/ceph/neuro/labeling/neurofinder.01.01/',
    'fname': 'N.01.01/Yr_d1_512_d2_512_d3_1_order_C_frames_1825_.mmap',
    'folder_name': 'N.01.01/',
    'ds_factor': 1,
    'p': 1,  # order of the autoregressive system
    'fr': 8,
    'gnb': 1,
    'T1': 1825,
    'decay_time': 1.4,
    'gSig': [6, 6]
}
# % Sue Ann k53
params_movie[6] = {
    # 'fname': '/mnt/ceph/neuro/labeling/k53_20160530/images/final_map/Yr_d1_512_d2_512_d3_1_order_C_frames_116043_.mmap',
    # 'folder_name':'/mnt/ceph/neuro/labeling/k53_20160530/',
    'fname': 'K53/Yr_d1_512_d2_512_d3_1_order_C_frames_116043_.mmap',
    'folder_name': 'K53/',
    'epochs': 1,
    'ds_factor': 2,
    'thresh_CNN_noisy': .75,
    'p': 1,  # order of the autoregressive system
    'T1': 3000,  # number of frames per file
    'fr': 30,
    'decay_time': 0.4,
    'gSig': [8, 8],  # expected half size of neurons
    'gnb': 2,
}

# % J115
params_movie[7] = {
    # 'fname': '/mnt/ceph/neuro/labeling/J115_2015-12-09_L01_ELS/images/final_map/Yr_d1_463_d2_472_d3_1_order_C_frames_90000_.mmap',
    # 'folder_name':'/mnt/ceph/neuro/labeling/J115_2015-12-09_L01_ELS/',
    'fname': 'J115/Yr_d1_463_d2_472_d3_1_order_C_frames_90000_.mmap',
    'folder_name': 'J115/',
    'epochs': 1,
    'ds_factor': 2,
    'thresh_CNN_noisy': .75,
    'p': 1,  # order of the autoregressive system
    'T1': 1000,
    'gnb': 2,
    'fr': 30,
    'decay_time': 0.4,
    'gSig': [8, 8]
}

# % J123
params_movie[8] = {
    # 'fname': '/mnt/ceph/neuro/labeling/J123_2015-11-20_L01_0/images/final_map/Yr_d1_458_d2_477_d3_1_order_C_frames_41000_.mmap',
    # 'folder_name':'/mnt/ceph/neuro/labeling/J123_2015-11-20_L01_0/',
    'fname': 'J123/Yr_d1_458_d2_477_d3_1_order_C_frames_41000_.mmap',
    'folder_name': 'J123/',
    'ds_factor': 2,
    'epochs': 1,
    'p': 1,  # order of the autoregressive system
    'fr': 30,
    'T1': 1000,
    'gnb': 1,
    'decay_time': 0.5,
    'gSig': [10, 10]
}

ALL_CCs = []
all_results = dict()
# %% convert mmaps into tifs
# import os.path
#
# for ind_dataset in [10]:
#    fls = glob.glob(params_movie[ind_dataset]['folder_name']+'images/mmap/*.mmap')
#    for file_count, ffll in enumerate(fls):
#        file_name = '/'.join(ffll.split('/')[:-2]+['mmap_tifs']+[ffll.split('/')[-1][:-4]+'tif'])
#        if not os.path.isfile(file_name):
#            fl_temp = cm.movie(np.array(cm.load(ffll)))
#            fl_temp.save(file_name)
#        print(file_name)
#    print(ind_dataset)
# %%  download and list all files to be processed
if preprocessing_from_scratch:
    # %%
    for ind_dataset in ID[:]:
        use_mmap = True
        ffls = glob.glob(os.path.abspath(base_folder + params_movie[ind_dataset]['folder_name']) + '/*.mmap')
        ffls.sort()
        fls = []
        if len(ffls) > 1:  # avoid selecting the joined memmap file
            for ffll in ffls:
                if '_d1' not in ffll.split('/')[-1][2:5]:
                    print(ffll.split('/')[-1][2:4])
                    fls.append(ffll)
        else:
            fls = ffls

        print(fls)

        Cn = np.array(cm.load(os.path.abspath(
            base_folder + params_movie[ind_dataset]['folder_name']) + '/correlation_image.tif')).squeeze()

        # %% Set up some parameters
        ds_factor = params_movie[ind_dataset][
            'ds_factor']  # spatial downsampling factor (increases speed but may lose some fine structure)
        fr = params_movie[ind_dataset]['fr']
        decay_time = params_movie[ind_dataset]['decay_time']
        gSig = tuple(np.ceil(np.array(params_movie[ind_dataset]['gSig']) / ds_factor).astype(
            np.int))  # expected half size of neurons
        init_batch = 200  # number of frames for initialization (presumably from the first file)
        expected_comps = 4000  # maximum number of expected components used for memory pre-allocation (exaggerate here)
        K = 2  # initial number of components
        min_SNR = global_params['min_SNR']
        min_SNR = 3.0956 * np.log(
            len(fls) * params_movie[ind_dataset]['T1'] / (307.85 * params_movie[ind_dataset]['fr']) + 0.7421)
        N_samples = np.ceil(params_movie[ind_dataset]['fr'] * params_movie[ind_dataset][
            'decay_time'])  # number of timesteps to consider when testing new neuron candidates
        pr_inc = 1 - scipy.stats.norm.cdf(global_params['min_SNR'])  # inclusion probability of noise transient
        thresh_fitness_raw = np.log(pr_inc) * N_samples  # event exceptionality threshold
        thresh_fitness_delta = -80.  # make this very neutral
        p = params_movie[ind_dataset]['p']  # order of AR indicator dynamics
        # rval_thr = global_params['rval_thr']                # correlation threshold for new component inclusion
        rval_thr = 0.06 * np.log(
            len(fls) * params_movie[ind_dataset]['T1'] / (2177. * params_movie[ind_dataset]['fr']) - 0.0462) + 0.8862

        try:
            gnb = params_movie[ind_dataset]['gnb']
        except:
            gnb = global_params['gnb']

        try:
            epochs = params_movie[ind_dataset]['epochs']  # number of background components
        except:
            epochs = global_params['epochs']  # number of passes over the data

        try:
            thresh_CNN_noisy = params_movie[ind_dataset]['thresh_CNN_noisy']
        except:
            thresh_CNN_noisy = global_params['thresh_CNN_noisy']

        try:
            min_num_trial = params_movie[ind_dataset]['min_num_trial']
        except:
            min_num_trial = global_params['min_num_trial']

        T1 = params_movie[ind_dataset]['T1'] * len(
            fls) * epochs  # total length of all files (if not known use a large number, then truncate at the end)
        # minibatch_length = int(global_params['batch_length_dt']*params_movie[ind_dataset]['fr']*params_movie[ind_dataset]['decay_time'])
        # %%
        params_dict = {'fnames': fls,
                       'fr': fr,
                       'decay_time': decay_time,
                       'gSig': gSig,
                       'p': p,
                       'min_SNR': min_SNR,
                       'rval_thr': rval_thr,
                       'ds_factor': ds_factor,
                       'nb': gnb,
                       'motion_correct': mot_corr,
                       'init_batch': init_batch,
                       'init_method': 'bare',
                       'normalize': True,
                       'expected_comps': expected_comps,
                       'K': K,
                       'epochs': epochs,
                       'show_movie': False,
                       'min_num_trial': min_num_trial,  # minimum number of times to attempt to add a component
                       'use_peak_max': True,
                       'thresh_CNN_noisy': thresh_CNN_noisy,
                       'sniper_mode': global_params['sniper_mode'],
                       'merge_thr': global_params['merge_thr'],
                       'ds_factor': ds_factor
                       }
        opts = cnmf.params.CNMFParams(params_dict=params_dict)

        if not reload:
            cnm = cnmf.online_cnmf.OnACID(params=opts)
            cnm.fit_online()
            cnm.save(fls[0][:-4]+'hdf5')

        else:
            print('*****  reloading   **********')
            cnm = load_OnlineCNMF(fls[0][:-4]+'hdf5')

            if plot_on:
                pl.figure()
                crd = cnm.estimates.plot_contours(img=Cn)
                cnm.estimates.view_components(img=Cn)

        # %% load, threshold and filter for size ground truth
        min_radius = max(cnm.params.init['gSig'][0] / 2., 2.)  # minimum acceptable radius
        max_radius = 2. * cnm.params.init['gSig'][0]  # maximum acceptable radius
        min_size_neuro = min_radius ** 2 * np.pi
        max_size_neuro = max_radius ** 2 * np.pi

        # %%
        c, dview, n_processes = cm.cluster.setup_cluster(backend='local', n_processes=None, single_thread=True)
        gt_file = os.path.join(base_folder, os.path.split(params_movie[ind_dataset]['fname'])[0],
                               os.path.split(params_movie[ind_dataset]['fname'])[1][:-4] + 'match_masks.npz')
        with np.load(gt_file, encoding='latin1') as ld:
            d1_or = int(ld['d1'])
            d2_or = int(ld['d2'])
            dims_or = (d1_or, d2_or)
            A_gt = ld['A_gt'][()].toarray()
            C_gt = ld['C_gt']
            Cn_orig = ld['Cn']
            if ds_factor > 1:
                A_gt = cm.movie(np.reshape(A_gt, dims_or + (-1,), order='F')).transpose(2, 0, 1).resize(1. / ds_factor,
                                                                                                        1. / ds_factor)
                A_gt2 = np.array(np.reshape(A_gt, (A_gt.shape[0], -1), order='F')).T
                Cn_orig = cv2.resize(Cn_orig, None, fx=1. / ds_factor, fy=1. / ds_factor)
            else:
                A_gt2 = A_gt.copy()

        # A_gt_thr = cm.source_extraction.cnmf.spatial.threshold_components(A_gt2, dims, medw=None, thr_method='max', maxthr=global_params['max_thr'], extract_cc=True,
        #                          se=None, ss=None, dview=None)
        gt_estimate = Estimates(A=scipy.sparse.csc_matrix(A_gt2), b=None, C=C_gt, f=None, R=None, dims=dims_or)
        gt_estimate.threshold_spatial_components(maxthr=global_params['max_thr'], dview=dview)
        gt_estimate.remove_small_large_neurons(min_size_neuro, max_size_neuro)
        _ = gt_estimate.remove_duplicates(predictions=None, r_values=None, dist_thr=0.1, min_dist=10, thresh_subset=0.6)
        print(gt_estimate.A.shape)

        # %% compute results
        use_cnn = False
        if use_cnn:
            cnm.params.set('quality', {'min_cnn_thr': 0.1})
            cnm.estimates.evaluate_components_CNN(cnm.params)
            cnm.estimates.select_components(use_object=True)
        # %% detect duplicates
        cnm.estimates.threshold_spatial_components(maxthr=global_params['max_thr'], dview=dview)
        cnm.estimates.remove_small_large_neurons(min_size_neuro, max_size_neuro)
        _ = cnm.estimates.remove_duplicates(r_values=None, dist_thr=0.1, min_dist=10, thresh_subset=0.6)

        # %% FIGURE 4 a bottom (need dataset K53!)
        plot_results = False
        if plot_results:
            pl.figure(figsize=(30, 20))

        tp_gt, tp_comp, fn_gt, fp_comp, performance_cons_off = compare_components(gt_estimate, cnm.estimates,
                                                                                  Cn=Cn_orig, thresh_cost=.8,
                                                                                  min_dist=10,
                                                                                  print_assignment=False,
                                                                                  labels=['GT', 'CMP'],
                                                                                  plot_results=False)

        pl.rcParams['pdf.fonttype'] = 42
        font = {'family': 'Arial',
                'weight': 'regular',
                'size': 20}

        # %% GENERATE CORRELATIONS FOR FIGS 5 b,c
        pl.rc('font', **font)
        print(gt_file)
        print('NO CNN')
        print({a: b.astype(np.float16) for a, b in performance_cons_off.items()})
        xcorrs = [scipy.stats.pearsonr(a, b)[0] for a, b in zip(gt_estimate.C[tp_gt], cnm.estimates.C[tp_comp])]
        xcorrs = [x for x in xcorrs if str(x) != 'nan']
        ALL_CCs.append(xcorrs)

        performance_tmp = performance_cons_off.copy()
        performance_tmp['A_gt_thr'] = gt_estimate.A_thr
        performance_tmp['A_thr'] = cnm.estimates.A_thr
        performance_tmp['A'] = cnm.estimates.A
        performance_tmp['A_gt'] = gt_estimate.A
        performance_tmp['C'] = cnm.estimates.C
        performance_tmp['C_gt'] = gt_estimate.C

        performance_tmp['idx_components_gt'] = np.arange(gt_estimate.A.shape[-1])
        performance_tmp['idx_components_cnmf'] = np.arange(cnm.estimates.A.shape[-1])
        performance_tmp['tp_gt'] = tp_gt
        performance_tmp['tp_comp'] = tp_comp
        performance_tmp['fn_gt'] = fn_gt
        performance_tmp['fp_comp'] = fp_comp
        all_results[params_movie[ind_dataset]['folder_name']] = performance_tmp

    if save_results:
        np.savez('RESULTS_SUMMARY/all_results_Jan_2018_online_refactor.npz', all_results=all_results)

else:
    print('**** LOADING PREPROCESSED RESULTS ****')
    lsls
    with np.load('RESULTS_SUMMARY/all_results_Jan_2018_online.npz') as ld:
        all_results = ld['all_results'][()]

# %% The variables ALL_CCs and all_results contain all the info necessary to create the figures
print([[k, r['f1_score']] for k, r in all_results.items()])
