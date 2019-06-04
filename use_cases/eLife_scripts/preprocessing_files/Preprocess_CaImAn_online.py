#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Complete pipeline for CaImAn online processing and comparison with consensus
annotation. The script processes one (or more) of the provided datasets and 
compares against the consensus annotations. It generates three figures:
i) contour plots of the detected components (Fig. 4)
ii) the correlation coefficient of the traces against consensus (Fig. 5) 
iii) timing information about the different steps of caiman online. (Fig. 8)
    For more information check the companion paper.
@author: Andrea Giovannucci @agiovann and Eftychios Pnevmatikakis @epnev
"""
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
from caiman.base.movies import from_zipfiles_to_movie_lists
import matplotlib.pyplot as plt
import scipy
import cv2
import glob
import os
import sys
import time
import gc
import keras

from builtins import str

try:
    cv2.setNumThreads(1)
except:
    print('Open CV is naturally single threaded')

# %% Select a dataset (specify the ID variable as a list of datasets to be processed)
# 0: neuforinder.03.00.test (N.03.00.t)
# 1: neurofinder.04.00.test (N.04.00.t)
# 2: neurofinder.02.00 (N.02.00)
# 3: yuste (YST)
# 4: neurofinder.00.00 (N.00.00)
# 5: neurofinder,01.01 (N.01.01)
# 6: sue_ann_k53_20160530 (K53)
# 7: J115 (J115)
# 8: J123 (J123)

try:
    if 'pydevconsole' in sys.argv[0]:
        raise Exception()
    ID = sys.argv[1]
    ID = str(np.int(ID))
    print('Processing ID:' + str(ID))
    ID = [np.int(ID)]

except:
    ID = [0]
    print('ID NOT PASSED')

if len(ID) == 9:
    save_file = 'all_res_online_web_bk.npz'
else:
    save_file = 'results_CaImAn_Online_web_' + "_".join([str(ind) for ind in ID]) + '.npz'
reload = False
save_results = False
plot_results = False

base_folder = '/mnt/ceph/neuro/DataForPublications/DATA_PAPER_ELIFE/WEBSITE/'
# %% set some global parameters here
# 'use_cases/edge-cutter/binary_cross_bootstrapped.json'
global_params = {'min_SNR': 0.8,       # minimum SNR when considering adding a new neuron
                 'p': 1,               # order of indicator dynamics 
                 'gnb': 2,             # number of background components
                 'epochs': 2,          # number of passes over the data
                 'rval_thr': 0.8,      # spatial correlation threshold
                 'max_thr': 0.25,      # parameter for thresholding components when cleaning up shapes
                 'mot_corr': False,    # flag for motion correction (set to False to compare directly on the same FOV)
                 'min_num_trial': 10,  # maximum number of candidate components per frame
                 'use_peak_max': True,
                 'thresh_CNN_noisy': .65,  # CNN threshold for accepting a candidate component
                 'sniper_mode': True,      # use the CNN for testing candidate components
                 'update_freq': 250
                 }

params_movie = [{}] * 9  # set up list of dictionaries
# % neurofinder.03.00.test
params_movie[0] = {
    'folder_name': 'N.03.00.t',
    'ds_factor': 2,
    'fr': 7,
    'decay_time': 0.4,
    'gSig': [12, 12],  # expected half size of neurons
    'gnb': 3,
}
# % neurofinder.04.00.test
params_movie[1] = {
    'folder_name': 'N.04.00.t',
    'epochs': 3,
    'ds_factor': 1,
    'fr': 8,
    'gSig': [7, 7],  # expected half size of neurons
    'decay_time': 1.2,  # rough length of a transient
    'gnb': 3,
}

# % neurofinder 02.00
params_movie[2] = {
    'folder_name': 'N.02.00',
    'ds_factor': 2,
    'fr': 30,  # imaging rate in Hz
    'gSig': [8, 8],  # expected half size of neuron
    'decay_time': 0.3,
    'gnb': 2,
}

# % YST
params_movie[3] = {
    'folder_name': 'YST',
    'epochs': 2,
    'ds_factor': 1,
    'fr': 10,
    'decay_time': .75,
    'gnb': 3,
    'gSig': [6, 6],  # expected half size of neurons
}

# % neurofinder.00.00
params_movie[4] = {
    'folder_name': 'N.00.00',
    'ds_factor': 1,
    'decay_time': 0.4,
    'epochs': 3,
    'fr': 16,
    'gSig': [8, 8],  # expected half size of neurons
    'gnb': 2,
}
# % neurofinder.01.01
params_movie[5] = {
    'folder_name': 'N.01.01',
    'ds_factor': 1,
    'fr': 8,
    'gnb': 1,
    'decay_time': 1.4,
    'gSig': [6, 6]
}
# % Sue Ann k53
params_movie[6] = {
    'folder_name': 'K53',
    'epochs': 1,
    'ds_factor': 2,
    'fr': 30,
    'decay_time': 0.4,
    'gSig': [8, 8],  # expected half size of neurons
    'gnb': 2,
}

# % J115
params_movie[7] = {
    'folder_name': 'J115',
    'epochs': 1,
    'ds_factor': 2,
    'gnb': 2,
    'fr': 30,
    'decay_time': 0.4,
    'gSig': [8, 8]
}

# % J123
params_movie[8] = {
    'folder_name': 'J123',
    'ds_factor': 2,
    'epochs': 2,
    'fr': 30,
    'gnb': 1,
    'decay_time': 0.5,
    'gSig': [10, 10]
}

all_results = dict()


# %% iterate over all datasets to be processed

for ind_dataset in ID:
    keras.backend.clear_session()
    gc.collect()
    fname_zip = os.path.join(base_folder, params_movie[ind_dataset]['folder_name'], 'images', 'images_' + params_movie[ind_dataset]['folder_name'] +'.zip')
    fls = glob.glob(os.path.join(base_folder, params_movie[ind_dataset]['folder_name'], 'images', 'mov*.tif'))
    if len(fls) == 0:
        fls = from_zipfiles_to_movie_lists(fname_zip)

    fls = sorted(fls, key=lambda x: np.int(x.split('_')[-1][:-4]))
    print(fls)

    Cn = np.array(cm.load(os.path.join(base_folder, params_movie[ind_dataset]['folder_name'], 
                                       'projections', 'correlation_image.tif'))).squeeze()

    # %% Set up some parameters
    ds_factor = params_movie[ind_dataset][
        'ds_factor']  # spatial downsampling factor (increases speed but may lose some fine structure)
    fr = params_movie[ind_dataset]['fr']
    decay_time = params_movie[ind_dataset]['decay_time']
    gSig = tuple(np.ceil(np.array(params_movie[ind_dataset]['gSig']) / ds_factor).astype(
        np.int))  # expected half size of neurons
    init_batch = 200  # number of frames for initialization (presumably from the first file)
    expected_comps = 1000  # maximum number of expected components used for memory pre-allocation (exaggerate here)
    K = 2  # initial number of components    
    rval_thr = global_params['rval_thr']                # correlation threshold for new component inclusion

    try:
        gnb = params_movie[ind_dataset]['gnb']
    except:
        gnb = global_params['gnb']

    try:
        epochs = params_movie[ind_dataset]['epochs']  # number of background components
    except:
        epochs = global_params['epochs']  # number of passes over the data

    # %%
    params_dict = {'fnames': fls,
                   'fr': fr,
                   'decay_time': decay_time,
                   'gSig': gSig,
                   'p': global_params['p'],
                   'min_SNR': global_params['min_SNR'],
                   'rval_thr': global_params['rval_thr'],
                   'ds_factor': ds_factor,
                   'nb': gnb,
                   'motion_correct': global_params['mot_corr'],
                   'init_batch': init_batch,
                   'init_method': 'bare',
                   'normalize': True,
                   'expected_comps': expected_comps,
                   'dist_shape_update': True,
                   'K': K,
                   'epochs': epochs,
                   'show_movie': False,
                   'min_num_trial': global_params['min_num_trial'],
                   'use_peak_max': True,
                   'thresh_CNN_noisy': global_params['thresh_CNN_noisy'],
                   'sniper_mode': global_params['sniper_mode'],
                   'use_dense': False,
                   'update_freq' : global_params['update_freq']
                   }
    opts = cnmf.params.CNMFParams(params_dict=params_dict)

    if not reload:
        t1 = time.time()
        cnm = cnmf.online_cnmf.OnACID(params=opts)
        cnm.fit_online()
        t_el = time.time() - t1
        #cnm.save(fls[0][:-4]+'hdf5')  # uncomment if you want to save the entire object
    else:
        print('*****  reloading   **********')
        cnm = load_OnlineCNMF(fls[0][:-4]+'hdf5')

    # %% filter results by using the batch CNN
    use_cnn = False
    if use_cnn:
        cnm.params.set('quality', {'min_cnn_thr': 0.1})
        cnm.estimates.evaluate_components_CNN(cnm.params)
        cnm.estimates.select_components(use_object=True)

    # %% remove small and duplicate components

    min_radius = max(cnm.params.init['gSig'][0] / 2., 2.)  # minimum acceptable radius
    max_radius = 2. * cnm.params.init['gSig'][0]  # maximum acceptable radius
    min_size_neuro = min_radius ** 2 * np.pi
    max_size_neuro = max_radius ** 2 * np.pi

    cnm.estimates.threshold_spatial_components(maxthr=global_params['max_thr'], dview=None)
    cnm.estimates.remove_small_large_neurons(min_size_neuro, max_size_neuro)
    _ = cnm.estimates.remove_duplicates(r_values=None, dist_thr=0.1, min_dist=10, thresh_subset=0.6)

    # %% load consensus annotations and filter for size
    gt_file = glob.glob(os.path.join(base_folder, params_movie[ind_dataset]['folder_name'], '*masks.npz'))[0]
    with np.load(gt_file, encoding='latin1', allow_pickle=True) as ld:
        d1_or = int(ld['d1'])
        d2_or = int(ld['d2'])
        dims_or = (d1_or, d2_or)
        A_gt = ld['A_gt'][()].toarray()
        C_gt = ld['C_gt']
        Cn_orig = ld['Cn']
        if ds_factor > 1:
            A_gt2= np.concatenate([cv2.resize(A_gt[:, fr_].reshape(dims_or, order='F'), cnm.dims[::-1]).reshape(-1, order='F')[:,None] for fr_ in range(A_gt.shape[-1])], axis = 1)
            Cn_orig = cv2.resize(Cn_orig, cnm.dims[::-1])
        else:
            A_gt2 = A_gt.copy()

    gt_estimate = Estimates(A=scipy.sparse.csc_matrix(A_gt2), b=None, C=C_gt, f=None, R=None, dims=cnm.dims)
    gt_estimate.threshold_spatial_components(maxthr=global_params['max_thr'], dview=None)
    nrn_size = gt_estimate.remove_small_large_neurons(min_size_neuro, max_size_neuro)
    nrn_dup = gt_estimate.remove_duplicates(predictions=None, r_values=None, dist_thr=0.1, min_dist=10,
                                            thresh_subset=0.6)
    idx_components_gt = nrn_size[nrn_dup]
    print(gt_estimate.A.shape)

    # %% compute performance and plot against consensus annotations
    tp_gt, tp_comp, fn_gt, fp_comp, performance_cons_off = compare_components(gt_estimate, cnm.estimates,
                                                                              Cn=Cn_orig, thresh_cost=.8,
                                                                              min_dist=10,
                                                                              print_assignment=False,
                                                                              labels=['CA', 'CMO'],
                                                                              plot_results=plot_results)

    plt.rcParams['pdf.fonttype'] = 42
    font = {'family': 'Arial',
            'weight': 'regular',
            'size': 20}

    # %% compute correlations for traces of caiman online vs consensus
    plt.rc('font', **font)
    print(gt_file)
    print(params_movie[ind_dataset]['folder_name']+ str({a: b.astype(np.float16) for a, b in performance_cons_off.items()}))
    xcorrs = [scipy.stats.pearsonr(a, b)[0] for a, b in zip(gt_estimate.C[tp_gt], cnm.estimates.C[tp_comp])]
    xcorrs = [x for x in xcorrs if str(x) != 'nan']
    if plot_results:
        plt.figure();
        plt.subplot(1,2,1); plt.hist(xcorrs, 100); plt.title('Empirical PDF of trace correlation coefficients')
        plt.subplot(1,2,2); plt.hist(xcorrs, 100, cumulative=True); plt.title('Empirical CDF of trace correlation coefficients')

    # %% Save some results
    performance_tmp = performance_cons_off.copy()
    performance_tmp['A_gt_thr'] = gt_estimate.A_thr
    performance_tmp['A_thr'] = cnm.estimates.A_thr
    performance_tmp['A'] = cnm.estimates.A
    performance_tmp['A_gt'] = gt_estimate.A
    performance_tmp['C'] = cnm.estimates.C
    performance_tmp['C_gt'] = gt_estimate.C
    performance_tmp['YrA'] = cnm.estimates.YrA
    performance_tmp['YrA_gt'] = gt_estimate.YrA

    performance_tmp['idx_components_gt'] = np.arange(gt_estimate.A.shape[-1])
    performance_tmp['idx_components_cnmf'] = np.arange(cnm.estimates.A.shape[-1])
    performance_tmp['tp_gt'] = tp_gt
    performance_tmp['tp_comp'] = tp_comp
    performance_tmp['fn_gt'] = fn_gt
    performance_tmp['fp_comp'] = fp_comp
    performance_tmp['t_online'] = cnm.t_online
    performance_tmp['comp_upd'] = cnm.comp_upd
    performance_tmp['dims'] = cnm.dims
    performance_tmp['t_el'] = t_el
    performance_tmp['t_online'] = cnm.t_online
    performance_tmp['comp_upd'] = cnm.comp_upd
    performance_tmp['t_detect'] = cnm.t_detect
    performance_tmp['t_shapes'] = cnm.t_shapes
    performance_tmp['CCs'] = xcorrs
    performance_tmp['params'] = params_movie
    all_results[params_movie[ind_dataset]['folder_name']] = performance_tmp

    # %% Plot Timing performance
    if plot_results:
        plt.figure(); 
        plt.stackplot(np.arange(len(cnm.t_detect)),  1e3*(np.array(cnm.t_online) - np.array(cnm.t_detect) - np.array(cnm.t_shapes)),
                      1e3*np.array(cnm.t_detect), 1e3*np.array(cnm.t_shapes))
        plt.title('Processing time per frame')
        plt.xlabel('Frame #')
        plt.ylabel('Processing time [ms]')
        plt.ylim([0,100])
        plt.legend(labels=['process','detect','shapes'])

if save_results:
    path_save_file = os.path.join(base_folder, save_file)
    np.savez(path_save_file, all_results=all_results)

# %% The variables ALL_CCs and all_results contain all the info necessary to create the figures
print([[k, r['f1_score']] for k, r in all_results.items()])
