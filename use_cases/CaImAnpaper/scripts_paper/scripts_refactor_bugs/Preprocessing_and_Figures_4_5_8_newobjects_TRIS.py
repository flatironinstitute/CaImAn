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
import pickle
from caiman.components_evaluation import select_components_from_metrics
from caiman.base.rois import nf_match_neurons_in_binary_masks
from caiman.utils.utils import apply_magic_wand
from caiman.base.rois import detect_duplicates_and_subsets
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
    print('Not launched under iPython')

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
#%%  ANALYSIS MODE AND PARAMETERS
preprocessing_from_scratch = True # whether to run the full pipeline or just creating figures

if preprocessing_from_scratch:
    reload = False
    plot_on = False
    save_on = False # set to true to recreate
else:
    reload = True
    plot_on = False
    save_on = False

print_figs = True
skip_refinement = False
backend_patch = 'local'
backend_refine = 'local'
n_processes = 24
#%%
def precision_snr(snr_gt, snr_gt_fn, snr_cnmf, snr_cnmf_fp, snr_thrs):
        all_results_fake = []
        all_results_OR = []
        all_results_AND = []
        for snr_thr in snr_thrs:
            snr_all_gt = np.array(list(snr_gt) + list(snr_gt_fn) + [0]*len(snr_cnmf_fp))
            snr_all_cnmf = np.array(list(snr_cnmf) + [0]*len(snr_gt_fn) + list(snr_cnmf_fp))

            ind_gt = np.where(snr_all_gt > snr_thr)[0]  # comps in gt above threshold
            ind_cnmf = np.where(snr_all_cnmf > snr_thr)[0] # same for cnmf

            # precision: how many detected components above a given SNR are true
            prec = np.sum(snr_all_gt[ind_cnmf] > 0)/len(ind_cnmf)

            # recall: how many gt components with SNR above the threshold are detected
            rec = np.sum(snr_all_cnmf[ind_gt] > 0)/len(ind_gt)

            f1 = 2*prec*rec/(prec + rec)

            results_fake = [prec, rec, f1]
            # f1 score with OR condition

            ind_OR = np.union1d(ind_gt, ind_cnmf)
                    # indeces of components that are above threshold in either direction
            ind_gt_OR = np.where(snr_all_gt[ind_OR] > 0)[0]     # gt components
            ind_cnmf_OR = np.where(snr_all_cnmf[ind_OR] > 0)[0] # cnmf components
            prec_OR = np.sum(snr_all_gt[ind_OR][ind_cnmf_OR] > 0)/len(ind_cnmf_OR)
            rec_OR = np.sum(snr_all_cnmf[ind_OR][ind_gt_OR] > 0)/len(ind_gt_OR)
            f1_OR = 2*prec_OR*rec_OR/(prec_OR + rec_OR)

            results_OR = [prec_OR, rec_OR, f1_OR]

            # f1 score with AND condition

            ind_AND = np.intersect1d(ind_gt, ind_cnmf)
            ind_fp = np.intersect1d(ind_cnmf, np.where(snr_all_gt == 0)[0])
            ind_fn = np.intersect1d(ind_gt, np.where(snr_all_cnmf == 0)[0])

            prec_AND = len(ind_AND)/(len(ind_AND) + len(ind_fp))
            rec_AND = len(ind_AND)/(len(ind_AND) + len(ind_fn))
            f1_AND = 2*prec_AND*rec_AND/(prec_AND + rec_AND)

            results_AND = [prec_AND, rec_AND, f1_AND]
            all_results_fake.append(results_fake)
            all_results_OR.append(results_OR)
            all_results_AND.append(results_AND)


        return np.array(all_results_fake), np.array(all_results_OR), np.array(all_results_AND)
#%%
global_params = {'min_SNR': 2,        # minimum SNR when considering adding a new neuron
                 'gnb': 2,             # number of background components
                 'rval_thr' : 0.80,     # spatial correlation threshold
                 'min_cnn_thresh' : 0.95,
                 'p' : 1,
                 'min_rval_thr_rejected': 0, # length of mini batch for OnACID in decay time units (length would be batch_length_dt*decay_time*fr)
                 'max_classifier_probability_rejected' : 0.1,    # flag for motion correction (set to False to compare directly on the same FOV)
                 'max_fitness_delta_accepted' : -20,
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
#%%
params_movies = []
#%%
params_movie = {'fname': 'N.03.00.t/Yr_d1_498_d2_467_d3_1_order_C_frames_2250_.mmap',
                'gtname':'N.03.00.t/joined_consensus_active_regions.npy',
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
params_movie = {'fname': 'N.04.00.t/Yr_d1_512_d2_512_d3_1_order_C_frames_3000_.mmap',
                'gtname':'N.04.00.t/joined_consensus_active_regions.npy',
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


#%% yuste
params_movie = {'fname': 'YST/Yr_d1_200_d2_256_d3_1_order_C_frames_3000_.mmap',
                 'gtname':'YST/joined_consensus_active_regions.npy',
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
params_movie = {'fname': 'N.00.00/Yr_d1_512_d2_512_d3_1_order_C_frames_2936_.mmap',
                  'gtname':'N.00.00/joined_consensus_active_regions.npy',
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
params_movie = {'fname': 'N.01.01/Yr_d1_512_d2_512_d3_1_order_C_frames_1825_.mmap',
                  'gtname':'N.01.01/joined_consensus_active_regions.npy',
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
#%% neurofinder 02.00
params_movie = {#'fname': '/opt/local/Data/labeling/neurofinder.02.00/Yr_d1_512_d2_512_d3_1_order_C_frames_8000_.mmap',
                'fname': 'N.02.00/Yr_d1_512_d2_512_d3_1_order_C_frames_8000_.mmap',
                 'gtname':'N.02.00/joined_consensus_active_regions.npy',
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
#%% Sue Ann k53
params_movie = {#'fname': '/opt/local/Data/labeling/k53_20160530/Yr_d1_512_d2_512_d3_1_order_C_frames_116043_.mmap',
                'fname':'K53/Yr_d1_512_d2_512_d3_1_order_C_frames_116043_.mmap',
                'gtname':'K53/joined_consensus_active_regions.npy',
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
params_movie = {#'fname': '/opt/local/Data/labeling/J115_2015-12-09_L01_ELS/Yr_d1_463_d2_472_d3_1_order_C_frames_90000_.mmap',
                'fname': 'J115/Yr_d1_463_d2_472_d3_1_order_C_frames_90000_.mmap',
                'gtname':'J115/joined_consensus_active_regions.npy',
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
params_movie = {#'fname': '/opt/local/Data/labeling/J123_2015-11-20_L01_0/Yr_d1_458_d2_477_d3_1_order_C_frames_41000_.mmap',
                'fname': 'J123/Yr_d1_458_d2_477_d3_1_order_C_frames_41000_.mmap',
                'gtname':'J123/joined_consensus_active_regions.npy',
                     # order of the autoregressive system
                 'merge_thresh': 0.8,  # merging threshold, max correlation allow
                 'rf': 40,  # half-size of the patches in pixels. rf=25, patches are 50x50    20
                 'stride_cnmf': 20,  # amounpl.it of overlap between the patches in pixels
                 'K': 10,  # number of components per patch
                 'gSig': [8,8],  # expected half size of neurons
                 'decay_time' : 0.5,
                 'fr' : 30,
                 'n_chunks': 10,
                 'swap_dim':False,
                 'crop_pix':2,
                 }
params_movies.append(params_movie.copy())

#%%
def myfun(x):
    from caiman.source_extraction.cnmf.deconvolution import constrained_foopsi
    dc = constrained_foopsi(*x)
    return (dc[0],dc[5])
def fun_exc(x):
    from scipy.stats import norm
    from caiman.components_evaluation import compute_event_exceptionality

    fluo, param = x
    N_samples = np.ceil(param['fr'] * param['decay_time']).astype(np.int)
    ev = compute_event_exceptionality(np.atleast_2d(fluo), N=N_samples)
    return  -norm.ppf(np.exp(np.array(ev[1]) / N_samples))
#%%
if preprocessing_from_scratch:
    all_perfs = []
    all_rvalues = []
    all_comp_SNR_raw =[]
    all_comp_SNR_delta = []
    all_predictions = []
    all_labels = []
    all_results = dict()
    n_pixels_per_process=4000
    block_size=4000
    num_blocks_per_run=10
    ALL_CCs = []

    for params_movie in np.array(params_movies)[4:5]:
    #    params_movie['gnb'] = 3
        params_display = {
            'downsample_ratio': .2,
            'thr_plot': 0.8
        }

        # @params fname name of the movie
        fname_new = params_movie['fname']
        print(fname_new)

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
        try:
            cm.stop_server()
            dview.terminate()
        except:
            print('No clusters to stop')

        c, dview, n_processes = setup_cluster(
                backend=backend_patch, n_processes=n_processes, single_thread=False)
        print('Not RELOADING')
        if not reload:
            # %% RUN ANALYSIS

        # %% correlation image
            if (plot_on or save_on):
                if False and m_images.shape[0]<10000:
                    Cn = m_images.local_correlations(swap_dim = params_movie['swap_dim'], frames_per_chunk = 1500)
                    Cn[np.isnan(Cn)] = 0
                else:
                    Cn = np.array(cm.load(('/'.join(params_movie['gtname'].split('/')[:-2]+['projections','correlation_image.tif'])))).squeeze()
                    #pl.imshow(Cn, cmap='gray', vmax=.95)

            check_nan = False
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
            print('Strating CNMF')
            cnm = cnmf.CNMF(n_processes=n_processes, nb_patch = 1, k=K, gSig=gSig, merge_thresh=params_movie['merge_thresh'], p=global_params['p'],
                            dview=dview, rf=rf, stride=stride_cnmf, memory_fact=1,
                            method_init=init_method, alpha_snmf=alpha_snmf, only_init_patch=global_params['only_init_patch'],
                            gnb=global_params['gnb'], method_deconvolution='oasis',border_pix =  params_movie['crop_pix'],
                            low_rank_background = global_params['low_rank_background'], rolling_sum = True, check_nan=check_nan,
                            block_size=block_size, num_blocks_per_run=num_blocks_per_run)
            cnm = cnm.fit(images)

            A_tot = cnm.estimates.A
            C_tot = cnm.estimates.C
            YrA_tot = cnm.estimates.YrA
            b_tot = cnm.estimates.b
            f_tot = cnm.estimates.f
            sn_tot = cnm.estimates.sn
            print(('Number of components:' + str(A_tot.shape[-1])))
            t_patch = time.time() - t1
            try:
                dview.terminate()
            except:
                pass
            c, dview, n_processes = cm.cluster.setup_cluster(
            backend=backend_refine, n_processes=n_processes, single_thread=False)
            # %%
            if plot_on:
                pl.figure()
                crd = plot_contours(A_tot, Cn, thr=params_display['thr_plot'])

                # %% rerun updating the components to refine
            t1 = time.time()
            cnm = cnmf.CNMF(n_processes=n_processes, k=A_tot.shape, gSig=gSig, merge_thresh=merge_thresh, p=p, dview=dview, Ain=A_tot,
                            Cin=C_tot, b_in = b_tot,
                            f_in=f_tot, rf=None, stride=None, method_deconvolution='oasis',gnb = global_params['gnb'],
                            low_rank_background = global_params['low_rank_background'],
                            update_background_components = global_params['update_background_components'], check_nan=check_nan,
                                n_pixels_per_process=n_pixels_per_process, block_size= block_size, num_blocks_per_run=num_blocks_per_run, skip_refinement=skip_refinement)

            cnm = cnm.fit(images)
            t_refine = time.time() - t1

            A, C, b, f, YrA, sn = cnm.estimates.A, cnm.estimates.C, cnm.estimates.b, cnm.estimates.f, cnm.estimates.YrA, cnm.estimates.sn
            # %% again recheck quality of components, stricter criteria
            t1 = time.time()
            idx_components, idx_components_bad, comp_SNR, r_values, predictionsCNN = estimate_components_quality_auto(
                                Y, A, C, b, f, YrA, params_movie['fr'], params_movie['decay_time'], gSig, dims,
                                dview = dview, min_SNR=global_params['min_SNR'],
                                r_values_min = global_params['rval_thr'], r_values_lowest = global_params['min_rval_thr_rejected'],
                                Npeaks =  global_params['Npeaks'], use_cnn = True, thresh_cnn_min = global_params['min_cnn_thresh'],
                                thresh_cnn_lowest = global_params['max_classifier_probability_rejected'],
                                thresh_fitness_delta = global_params['max_fitness_delta_accepted'], gSig_range = None)
    #                            [list(np.add(i,a)) for i,a in zip(range(0,1),[gSig]*3)]


            t_eva_comps = time.time() - t1
            print(' ***** ')
            print((len(C)))
            print((len(idx_components)))
            #%%
        #    all_matches = False
        #    filter_SNR = False
            gt_file = os.path.join(os.path.split(fname_new)[0], os.path.split(fname_new)[1][:-4] + 'match_masks.npz')
            with np.load(gt_file, encoding = 'latin1') as ld:
                print(ld.keys())
        #        locals().update(ld)
                C_gt = ld['C_gt']
                YrA_gt = ld['YrA_gt']
                b_gt = ld['b_gt']
                f_gt = ld['f_gt']
                A_gt = scipy.sparse.coo_matrix(ld['A_gt'][()])
                dims_gt = (ld['d1'],ld['d2'])

    #            t1 = time.time()
                idx_components_gt, idx_components_bad_gt, comp_SNR_gt, r_values_gt, predictionsCNN_gt = estimate_components_quality_auto(
                                    Y, A_gt, C_gt, b_gt, f_gt, YrA_gt, params_movie['fr'], params_movie['decay_time'], gSig, dims_gt,
                                    dview = dview, min_SNR=global_params['min_SNR'],
                                    r_values_min = global_params['rval_thr'], r_values_lowest = global_params['min_rval_thr_rejected'],
                                    Npeaks =  global_params['Npeaks'], use_cnn = True, thresh_cnn_min = global_params['min_cnn_thresh'],
                                    thresh_cnn_lowest = global_params['max_classifier_probability_rejected'],
                                    thresh_fitness_delta = global_params['max_fitness_delta_accepted'], gSig_range = None)
        #                            [list(np.add(i,a)) for i,a in zip(range(0,1),[gSig]*3)]


                print(' ***** ')
                print((len(C)))
                print((len(idx_components_gt)))


            #%%
            min_size_neuro = 3*2*np.pi
            max_size_neuro = (2*gSig[0])**2*np.pi
            A_gt_thr = cm.source_extraction.cnmf.spatial.threshold_components(A_gt.tocsc()[:,:], dims_gt, medw=None, thr_method='max', maxthr=0.2, nrgthr=0.99, extract_cc=True,
                                     se=None, ss=None, dview=dview)

            A_gt_thr = A_gt_thr > 0
        #    size_neurons_gt = A_gt_thr.sum(0)
        #    idx_size_neuro_gt = np.where((size_neurons_gt>min_size_neuro) & (size_neurons_gt<max_size_neuro) )[0]
        #    #A_thr = A_thr[:,idx_size_neuro]
            print(A_gt_thr.shape)
            #%%
            A_thr = cm.source_extraction.cnmf.spatial.threshold_components(A.tocsc()[:,:].toarray(), dims, medw=None, thr_method='max', maxthr=0.2, nrgthr=0.99, extract_cc=True,
                                     se=None, ss=None, dview=dview)

            A_thr = A_thr > 0
            size_neurons = A_thr.sum(0)
    #        idx_size_neuro = np.where((size_neurons>min_size_neuro) & (size_neurons<max_size_neuro) )[0]
        #    A_thr = A_thr[:,idx_size_neuro]
            print(A_thr.shape)
            # %% save results
            if save_on:
                np.savez(os.path.join(os.path.split(fname_new)[0],
                                  os.path.split(fname_new)[1][:-4] +
                                  'results_analysis_new_dev.npz'),
                                  Cn=Cn, fname_new = fname_new,
                                  A=A, C=C, b=b, f=f, YrA=YrA, sn=sn, d1=d1,
                                  d2=d2, idx_components=idx_components,
                                  idx_components_bad=idx_components_bad,
                                  comp_SNR=comp_SNR,  r_values=r_values,
                                  predictionsCNN = predictionsCNN,
                                  params_movie = params_movie,
                                  A_gt=A_gt, A_gt_thr=A_gt_thr, A_thr=A_thr,
                                  C_gt=C_gt, f_gt=f_gt, b_gt=b_gt, YrA_gt=YrA_gt,
                                  idx_components_gt=idx_components_gt,
                                  idx_components_bad_gt=idx_components_bad_gt,
                                  comp_SNR_gt=comp_SNR_gt,r_values_gt=r_values_gt,
                                  predictionsCNN_gt=predictionsCNN_gt,
                                  t_patch=t_patch, t_eva_comps=t_eva_comps,
                                  t_refine=t_refine)

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
    #%%
            params_display = {
                'downsample_ratio': .2,
                'thr_plot': 0.8
            }

            fn_old = fname_new
    # =============================================================================
    #         print(os.path.join(os.path.split(fname_new)[0], os.path.split(fname_new)[1][:-4] + 'results_analysis_after_merge_5.npz'))
    # =============================================================================

            with np.load(os.path.join(os.path.split(fname_new)[0], os.path.split(fname_new)[1][:-4] + 'results_analysis_new_dev.npz'), encoding = 'latin1') as ld:
                ld1 = {k:ld[k] for k in ['d1','d2','A','params_movie','fname_new',
                       'C','idx_components','idx_components_bad','Cn','b','f','YrA',
                       'sn','comp_SNR','r_values','predictionsCNN','A_gt','A_gt_thr',
                       'A_thr','C_gt','b_gt','f_gt','YrA_gt','idx_components_gt',
                       'idx_components_bad_gt', 'comp_SNR_gt', 'r_values_gt', 'predictionsCNN_gt',
                       't_eva_comps', 't_patch', 't_refine']}

                locals().update(ld1)
                dims_off = d1,d2
                A = scipy.sparse.coo_matrix(A[()])
                A_gt = A_gt[()]
                dims = (d1,d2)
                try:
                    params_movie = params_movie[()]
                except:
                    pass
                gSig = params_movie['gSig']
                fname_new = fn_old
                print([A.shape])
                print([ t_patch, t_refine,t_eva_comps])
                print(t_eva_comps+t_patch + t_refine)



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
            #%
            cm.movie(final_crops).play(gain=3,magnification = 6,fr=5)
            #%
            cm.movie(np.squeeze(final_crops[np.where(predictions[:,1]>=0.95)[0]])).play(gain=2., magnification = 8,fr=5)
            #%
            cm.movie(np.squeeze(final_crops[np.where(predictions[:,0]>=0.95)[0]])
                            ).play(gain=4., magnification = 8,fr=5)

        #%%
        thresh_fitness_raw_reject = 0.5
    #    global_params['max_classifier_probability_rejected'] = .1
        gSig_range =  [list(np.add(i,a)) for i,a in zip(range(0,1),[gSig]*1)]
    #    global_params['max_classifier_probability_rejected'] = .2
        idx_components, idx_components_bad, cnn_values =\
                    select_components_from_metrics(
                    A, dims, gSig, r_values,  comp_SNR , global_params['rval_thr'],
                    global_params['min_rval_thr_rejected'], global_params['min_SNR'],
                    thresh_fitness_raw_reject, global_params['min_cnn_thresh'],
                    global_params['max_classifier_probability_rejected'], True, gSig_range)

        print((len(idx_components)))
        #%%

        if plot_on:
            pl.figure()
            pl.subplot(1, 2, 1)
            crd = plot_contours(A.tocsc()[:, idx_components], Cn, thr=params_display['thr_plot'], vmax = 0.85)
            pl.subplot(1, 2, 2)
            crd = plot_contours(A.tocsc()[:, idx_components_bad], Cn, thr=params_display['thr_plot'], vmax = 0.85)
        #%%  detect duplicates
        thresh_subset = 0.6
        duplicates, indeces_keep, indeces_remove, D, overlap = detect_duplicates_and_subsets(
                A_thr[:,idx_components].reshape([dims[0],dims[1],-1],order = 'F').transpose([2,0,1])*1., predictionsCNN[idx_components], r_values = None,
                dist_thr=0.1, min_dist = 10,thresh_subset = thresh_subset)

        idx_components_cnmf = idx_components.copy()
        if len(duplicates) > 0:
            if plot_on:
                pl.figure()
                pl.subplot(1,3,1)
                pl.imshow(A_thr[:,idx_components].reshape([dims[0],dims[1],-1],order = 'F').transpose([2,0,1])[np.unique(duplicates).flatten()].sum(0))
                pl.colorbar()
                pl.subplot(1,3,2)
                pl.imshow(A_thr[:,idx_components].reshape([dims[0],dims[1],-1],order = 'F').transpose([2,0,1])[np.array(indeces_keep)[:]].sum(0))
                pl.colorbar()
                pl.subplot(1,3,3)
                pl.imshow(A_thr[:,idx_components].reshape([dims[0],dims[1],-1],order = 'F').transpose([2,0,1])[np.array(indeces_remove)[:]].sum(0))
                pl.colorbar()
                pl.pause(1)
            idx_components_cnmf = np.delete(idx_components_cnmf,indeces_remove)

        print('Duplicates CNMF:'+str(len(duplicates)))

        duplicates_gt, indeces_keep_gt, indeces_remove_gt, D_gt, overlap_gt = detect_duplicates_and_subsets(
                A_gt_thr.reshape([dims[0],dims[1],-1],order = 'F').transpose([2,0,1])*1., predictions = None, r_values = None,
                dist_thr=0.1, min_dist = 10,thresh_subset = thresh_subset)

        idx_components_gt = np.arange(A_gt_thr.shape[-1])
        if len(duplicates_gt) > 0:
            if plot_on:
                pl.figure()
                pl.subplot(1,3,1)
                pl.imshow(A_gt_thr.reshape([dims[0],dims[1],-1],order = 'F').transpose([2,0,1])[np.array(duplicates_gt).flatten()].sum(0))
                pl.colorbar()
                pl.subplot(1,3,2)
                pl.imshow(A_gt_thr.reshape([dims[0],dims[1],-1],order = 'F').transpose([2,0,1])[np.array(indeces_keep_gt)[:]].sum(0))
                pl.colorbar()
                pl.subplot(1,3,3)
                pl.imshow(A_gt_thr.reshape([dims[0],dims[1],-1],order = 'F').transpose([2,0,1])[np.array(indeces_remove_gt)[:]].sum(0))
                pl.colorbar()

                pl.pause(1)
            idx_components_gt = np.delete(idx_components_gt,indeces_remove_gt)
        print('Duplicates gt:'+str(len(duplicates_gt)))


        #%%
        remove_small_neurons = False
        if remove_small_neurons:
            min_size_neuro = 3*2*np.pi
            max_size_neuro = (2*gSig[0])**2*np.pi
            size_neurons_gt = A_gt_thr.sum(0)
            idx_size_neuro_gt = np.where((size_neurons_gt>min_size_neuro) & (size_neurons_gt<max_size_neuro) )[0]
            idx_components_gt = np.intersect1d(idx_components_gt,idx_size_neuro_gt)

            size_neurons = A_thr.sum(0)
            idx_size_neuro = np.where((size_neurons>min_size_neuro) & (size_neurons<max_size_neuro) )[0]
            idx_components_cnmf = np.intersect1d(idx_components_cnmf,idx_size_neuro)


        plot_results = plot_on
        if plot_results:
            pl.figure(figsize=(30,20))
            Cn_ = Cn
        else:
            Cn_ = None

        tp_gt, tp_comp, fn_gt, fp_comp, performance_cons_off =\
                nf_match_neurons_in_binary_masks(
                        A_gt_thr[:,idx_components_gt].reshape([dims[0],dims[1],-1],order = 'F').transpose([2,0,1])*1.,
                        A_thr[:,idx_components_cnmf].reshape([dims[0],dims[1],-1],order = 'F').transpose([2,0,1])*1.,
                        thresh_cost=.8, min_dist = 10,
                        print_assignment= False, plot_results=plot_results, Cn=Cn_, labels=['GT','Offline'])
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
        all_predictions.append(predictionsCNN )
        lbs = np.zeros(len(r_values))
        lbs[tp_comp] = 1
        all_labels.append(lbs)
        print({pf['fname_new'].split('/')[0]:pf['f1_score'] for pf in all_perfs})


        performance_tmp = performance_cons_off.copy()
        performance_tmp['comp_SNR_gt'] = comp_SNR_gt
        performance_tmp['comp_SNR'] = comp_SNR
        performance_tmp['A_gt_thr'] = A_gt_thr
        performance_tmp['A_thr'] = A_thr
        performance_tmp['A'] = A
        performance_tmp['A_gt'] = A_gt
        performance_tmp['C'] = C
        performance_tmp['C_gt'] = C_gt
        performance_tmp['YrA'] = YrA
        performance_tmp['YrA_gt'] = YrA_gt
        performance_tmp['predictionsCNN'] = predictionsCNN
        performance_tmp['predictionsCNN_gt'] = predictionsCNN_gt
        performance_tmp['r_values'] = r_values
        performance_tmp['r_values_gt'] = r_values_gt
        performance_tmp['idx_components_gt'] = idx_components_gt
        performance_tmp['idx_components_cnmf'] = idx_components_cnmf
        performance_tmp['tp_gt'] = tp_gt
        performance_tmp['tp_comp'] = tp_comp
        performance_tmp['fn_gt'] = fn_gt
        performance_tmp['fp_comp'] = fp_comp

        ALL_CCs.append([scipy.stats.pearsonr(a,b)[0] for a, b in zip(C_gt[idx_components_gt[tp_gt]],C[idx_components_cnmf[tp_comp]])])


        all_results[fname_new.split('/')[0]] = performance_tmp
        print([ t_patch, t_refine,t_eva_comps])
        print(t_eva_comps+t_patch + t_refine)
            #%%
    if preprocessing_from_scratch:
        np.savez('RESULTS_SUMMARY/all_results_Jan_2018.npz',all_results = all_results)

else:

    #%% RELOAD ALL THE RESULTS INSTEAD OF REGENERATING THEM
    with np.load('RESULTS_SUMMARY/all_results_Jan_2018.npz') as ld:
        all_results = ld['all_results'][()]
    #%% CREATE FIGURES
    if print_figs:
        #%% FIGURE 5a MASKS  (need to use the dataset k53)
        pl.figure('Figure 5a masks')
        idxFile = 7
        predictionsCNN = all_results['J115']['predictionsCNN']
        idx_components_cnmf = all_results['J115']['idx_components_cnmf']
        idx_components_gt = all_results['J115']['idx_components_gt']
        tp_comp = all_results['J115']['tp_comp']
        fp_comp = all_results['J115']['fp_comp']
        tp_gt = all_results['J115']['tp_gt']
        fn_gt = all_results['J115']['fn_gt']
        A = all_results['J115']['A']
        A_gt = all_results['J115']['A_gt']
        C = all_results['J115']['C']
        C_gt = all_results['J115']['C_gt']
        fname_new = params_movies[idxFile ]['fname']
        with np.load(os.path.join(os.path.split(fname_new)[0], os.path.split(fname_new)[1][:-4] + 'results_analysis_new_dev.npz'), encoding = 'latin1') as ld:
            dims = (ld['d1'],ld['d2'])
        Cn = np.array(cm.load(('/'.join(params_movies[idxFile]['gtname'].split('/')[:-1]+['correlation_image.tif'])))).squeeze()





        pl.ylabel('spatial components')
        idx_comps_high_r = [np.argsort(predictionsCNN[idx_components_cnmf[tp_comp]])[[-6,-5,-4,-3,-2]]]
        idx_comps_high_r_cnmf = idx_components_cnmf[tp_comp][idx_comps_high_r]
        idx_comps_high_r_gt = idx_components_gt[tp_gt][idx_comps_high_r]
        images_nice = (A.tocsc()[:,idx_comps_high_r_cnmf].toarray().reshape(dims+(-1,),order = 'F')).transpose(2,0,1)
        images_nice_gt =  (A_gt.tocsc()[:,idx_comps_high_r_gt].toarray().reshape(dims+(-1,),order = 'F')).transpose(2,0,1)
        cms = np.array([scipy.ndimage.center_of_mass(img) for img in images_nice]).astype(np.int)
        images_nice_crop = [img[cm_[0]-15:cm_[0]+15,cm_[1]-15:cm_[1]+15] for  cm_,img in zip(cms,images_nice)]
        images_nice_crop_gt = [img[cm_[0]-15:cm_[0]+15,cm_[1]-15:cm_[1]+15] for  cm_,img in zip(cms,images_nice_gt)]

        indexes = [1,3,5,7,9,2,4,6,8,10]
        count = 0
        for img in images_nice_crop:
            pl.subplot(5,2,indexes[count])
            pl.imshow(img)
            pl.axis('off')
            count += 1

        for img in images_nice_crop_gt:

            pl.subplot(5,2,indexes[count])
            pl.imshow(img)
            pl.axis('off')
            count += 1

        #%% FIGURE 5a Traces  (need to use the dataset J115)
        pl.figure('Figure 5a traces')
        traces_gt = C_gt[idx_comps_high_r_gt]# + YrA_gt[idx_comps_high_r_gt]
        traces_cnmf = C[idx_comps_high_r_cnmf]# + YrA[idx_comps_high_r_cnmf]
        traces_gt/=np.max(traces_gt,1)[:,None]
        traces_cnmf /=np.max(traces_cnmf,1)[:,None]
        pl.plot(scipy.signal.decimate(traces_cnmf,10,1).T-np.arange(5)*1,'y')
        pl.plot(scipy.signal.decimate(traces_gt,10,1).T-np.arange(5)*1,'k', linewidth = .5 )

        #%% FIGURE 5c
        pl.figure('Figure 5c')
        pl.rcParams['pdf.fonttype'] = 42

        with np.load('RESULTS_SUMMARY/ALL_CORRELATIONS_ONLINE_CONSENSUS.npz') as ld:
            print(ld.keys())
            xcorr_online = ld['ALL_CCs']
#            xcorr_offline = list(np.array(xcorr_offline[[0,1,3,4,5,2,6,7,8,9]]))

        with np.load('RESULTS_SUMMARY/ALL_CORRELATIONS_OFFLINE_CONSENSUS.npz') as ld:
            xcorr_offline = ld['ALL_CCs']

        names = ['0300.T',
                 '0400.T',
                 'YT',
                 '0000',
                 '0200',
                 '0101',
                 'k53',
                 'J115',
                 'J123']

        pl.subplot(1,2,1)

        pl.hist(np.concatenate(xcorr_online),bins = np.arange(0,1,.01))
        pl.hist(np.concatenate(xcorr_offline),bins = np.arange(0,1,.01))
        a = pl.hist(np.concatenate(xcorr_online[:]),bins = np.arange(0,1,.01))
        a3 = pl.hist(np.concatenate(xcorr_offline)[:],bins = np.arange(0,1,.01))

        a_on = np.cumsum(a[0]/a[0].sum())
        a_off = np.cumsum(a3[0]/a3[0].sum())
        #%
        pl.close()
        pl.figure('Figure 5c')

        pl.plot(np.arange(0.01,1,.01),a_on);pl.plot(np.arange(0.01,1,.01),a_off)
        pl.legend(['online', 'offline'])
        pl.xlabel('correlation (r)')
        pl.ylabel('cumulative probability')
        medians_on = []
        iqrs_on = []
        medians_off = []
        iqrs_off = []
        for onn,off in zip(xcorr_online,xcorr_offline):
            medians_on.append(np.median(onn))
            iqrs_on.append(np.percentile(onn,[25,75]))
            medians_off.append(np.median(off))
            iqrs_off.append(np.percentile(off,[25,75]))
        #%%
        pl.figure('Figure 5b')
        mu_on = np.array([np.median(xc) for xc in xcorr_online[:-1]])
        mu_off = np.array([np.median(xc) for xc in xcorr_offline[:-1]])
        pl.plot(np.concatenate([mu_off[:,None],mu_on[:,None]],axis=1).T,'o-')
        pl.xlabel('Offline and Online')
        pl.ylabel('Correlation coefficient')
        #%% FIGURE 4a TOP (need to use the dataset J115)
        pl.figure('Figure 4a top')
        pl.subplot(1,2,1)
        a1 = plot_contours(A.tocsc()[:, idx_components_cnmf[tp_comp]], Cn, thr=0.9, colors='yellow', vmax = 0.75, display_numbers=False,cmap = 'gray')
        a2 = plot_contours(A_gt.tocsc()[:, idx_components_gt[tp_gt]], Cn, thr=0.9, vmax = 0.85, colors='r', display_numbers=False,cmap = 'gray')
        pl.subplot(1,2,2)
        a3 = plot_contours(A.tocsc()[:, idx_components_cnmf[fp_comp]], Cn, thr=0.9, colors='yellow', vmax = 0.75, display_numbers=False,cmap = 'gray')
        a4 = plot_contours(A_gt.tocsc()[:, idx_components_gt[fn_gt]], Cn, thr=0.9, vmax = 0.85, colors='r', display_numbers=False,cmap = 'gray')
        #%% FIGURE 4 c and d
        from matplotlib.pyplot import cm as cmap
        pl.figure("Figure 4c and 4d")

        color=cmap.jet(np.linspace(0,1,10))
        i = 0
        legs = []
        all_ys = []
        SNRs = np.arange(0,10)
        pl.subplot(1,3,1)

        for k,fl_results in all_results.items():
            print(k)
            nm = k[:]
            nm = nm.replace('neurofinder','NF')
            nm = nm.replace('final_map','')
            nm = nm.replace('.','')
            nm = nm.replace('Data','')
            idx_components_gt = fl_results['idx_components_gt']
            idx_components_cnmf = fl_results['idx_components_cnmf']
            tp_gt = fl_results['tp_gt']
            tp_comp = fl_results['tp_comp']
            fn_gt = fl_results['fn_gt']
            fp_comp = fl_results['fp_comp']
            comp_SNR = fl_results['comp_SNR']
            comp_SNR_gt = fl_results['comp_SNR_gt']
            snr_gt = comp_SNR_gt[idx_components_gt[tp_gt]]
            snr_gt_fn = comp_SNR_gt[idx_components_gt[fn_gt]]
            snr_cnmf = comp_SNR[idx_components_cnmf[tp_comp]]
            snr_cnmf_fp = comp_SNR[idx_components_cnmf[fp_comp]]
            all_results_fake, all_results_OR, all_results_AND = precision_snr(snr_gt, snr_gt_fn, snr_cnmf, snr_cnmf_fp, SNRs)
            all_ys.append(all_results_fake[:,-1])
            pl.fill_between(SNRs,all_results_OR[:,-1],all_results_AND[:,-1], color=color[i], alpha = .1)
            pl.plot(SNRs,all_results_fake[:,-1], '.-',color=color[i])
#            pl.plot(x[::1]+np.random.normal(scale=.07,size=10),y[::1], 'o',color=color[i])
            pl.ylim([0.5,1])
            legs.append(nm[:7])
            i += 1
#            break
        pl.plot(SNRs,np.mean(all_ys,0),'k--', alpha=1, linewidth = 2)
        pl.legend(legs+['average'], fontsize=10)
        pl.xlabel('SNR threshold')

        pl.ylabel('F1 SCORE')
        #%
        i = 0
        legs = []
        for k,fl_results in all_results.items():
            x = []
            y = []
            if 'K53' in k:
                idx_components_gt = fl_results['idx_components_gt']
                idx_components_cnmf = fl_results['idx_components_cnmf']
                tp_gt = fl_results['tp_gt']
                tp_comp = fl_results['tp_comp']
                fn_gt = fl_results['fn_gt']
                fp_comp = fl_results['fp_comp']
                comp_SNR = fl_results['comp_SNR']
                comp_SNR_gt = fl_results['comp_SNR_gt']
                snr_gt = comp_SNR_gt[idx_components_gt[tp_gt]]
                snr_gt_fn = comp_SNR_gt[idx_components_gt[fn_gt]]
                snr_cnmf = comp_SNR[idx_components_cnmf[tp_comp]]
                snr_cnmf_fp = comp_SNR[idx_components_cnmf[fp_comp]]
                all_results_fake, all_results_OR, all_results_AND = precision_snr(snr_gt, snr_gt_fn, snr_cnmf, snr_cnmf_fp, SNRs)
                prec_idx = 0
                recall_idx = 1
                f1_idx = 2

                pl.subplot(1,3,2)
                pl.scatter(snr_gt,snr_cnmf,color='k', alpha = .15)
                pl.scatter(snr_gt_fn,np.random.normal(scale = .25, size=len(snr_gt_fn)),color='g', alpha = .15)
                pl.scatter(np.random.normal(scale = .25, size=len(snr_cnmf_fp)),snr_cnmf_fp,color='g', alpha = .15)
                pl.fill_between([20,40],[-2,-2],[40,40], alpha = .05, color='r')
                pl.fill_between([-2,40],[20,20],[40,40], alpha = .05 ,color='b')
                pl.xlabel('SNR GT')
                pl.ylabel('SNR CaImAn')
                pl.subplot(1,3,3)
                pl.fill_between(SNRs,all_results_OR[:,prec_idx],all_results_AND[:,prec_idx], color='b', alpha = .1)
                pl.plot(SNRs,all_results_fake[:,prec_idx], '.-',color='b')
                pl.fill_between(SNRs,all_results_OR[:,recall_idx],all_results_AND[:,recall_idx], color='r', alpha = .1)
                pl.plot(SNRs,all_results_fake[:,recall_idx], '.-',color= 'r')
                pl.fill_between(SNRs,all_results_OR[:,f1_idx],all_results_AND[:,f1_idx], color='g', alpha = .1)
                pl.plot(SNRs,all_results_fake[:,f1_idx], '.-',color='g')
                pl.legend(['precision','recall','f-1 score'], fontsize = 10)
                pl.xlabel('SNR threshold')
        #%% FIGURE 4 b  performance in detecting neurons (results have been manually annotated on an excel spreadsheet and reported here below)
        import pylab as plt
        plt.figure('Figure 4b')

        names = ['N.03.00.t',
                 'N.04.00.t',
                 'YST',
                 'N.00.00',
                 'N.02.00',
                 'N.01.01',
                 'K53',
                 'J115',
                 'J123']



        f1s = dict()
        f1s['batch'] = [all_results[nmm]['f1_score'] for nmm in names]
        f1s['online'] = [0.76,0.678,0.783,0.721,0.769,0.725,0.818,0.805,0.803] # these can be read from last part of script
        f1s['L1'] = [np.nan,np.nan,0.78,np.nan,0.89,0.8,0.89,np.nan,0.85] # Human 1
        f1s['L2'] = [0.9,0.69,0.9,0.92,0.87,0.89,0.92,0.93,0.83] # Human 2
        f1s['L3'] = [0.85,0.75,0.82,0.83,0.84,0.78,0.93,0.94,0.9] # Human 3
        f1s['L4'] = [0.78,0.87,0.79,0.87,0.82,0.75,0.83,0.83,0.91] # Human 4

        all_of = ((np.vstack([f1s['L1'],f1s['L2'],f1s['L3'],f1s['L4'],f1s['batch'],f1s['online']])))



        for i in range(6):
            pl.plot(i+np.random.random(9)*.2, all_of[i,:],'.')
            pl.plot([i-.5, i+.5], [np.nanmean(all_of[i,:])]*2,'k')
        plt.xticks(range(6), ['L1','L2','L3','L4','batch','online'], rotation=45)
        pl.ylabel('F1-score')

        #%% FIGURE  timing to create a mmap file (Figures 8 a and b left, "mem mapping")
        if False:
            import glob
            try:
                dview.terminate()
            except:
                print('No clusters to stop')

            c, dview, n_processes = setup_cluster(
            backend=backend, n_processes=n_processes, single_thread=False)

            t1 = time.time()
            ffllss = list(glob.glob('/opt/local/Data/Sue/k53/orig_tifs/*.tif')[:])
            ffllss.sort()
            print(ffllss)
            fname_new = cm.save_memmap(ffllss, base_name='memmap_', order='C',
                               border_to_0=0, dview = dview, n_chunks = 80)  # exclude borders
            t2 = time.time() - t1
        #%%FIGURE 8 a and b time performances (results have been manually annotated on an excel spreadsheet and reported here below)
        import pylab as plt

        pl.figure("Figure 8a and 8b")
        import numpy as np
        plt.rcParams['pdf.fonttype'] = 42
        font = {'family' : 'Arial',
                'weight' : 'regular',
                'size'   : 20}

        t_mmap = dict()
        t_patch = dict()
        t_refine = dict()
        t_filter_comps = dict()

        size = np.log10(np.array([2.1, 3.1,0.6,3.1,8.4,1.9,121.7,78.7,35.8,50.3])*1000)
        components= np.array([368,935,476,1060,1099,1387,1541,1013,398,1064])
        components= np.array([368,935,476,1060,1099,1387,1541,1013,398,1064])

        t_mmap['cluster'] = np.array([np.nan,41,np.nan,np.nan,109,np.nan,561,378,135,212])
        t_patch['cluster'] = np.array([np.nan,46,np.nan,np.nan,92,np.nan,1063,469,142,372])
        t_refine['cluster'] = np.array([np.nan,225,np.nan,np.nan,256,np.nan,1065,675,265,422])
        t_filter_comps['cluster'] = np.array([np.nan,7,np.nan,np.nan,11,np.nan,143,77,30,57])

        t_mmap['desktop'] = np.array([25,41,11,41,135,23,690,510,176,163])
        t_patch['desktop'] = np.array([21,43,16,48,85,45,2150,949,316,475])
        t_refine['desktop'] = np.array([105,205,43,279,216,254,1749,837,237,493])
        t_filter_comps['desktop'] = np.array([3,5,2,5,9,7,246,81,36,38])


        t_mmap['laptop'] = np.array([4.7,27,3.6,18,144,11,731,287,125,248])
        t_patch['laptop'] = np.array([58,84,47,77,174,85,2398,1587,659,1203])
        t_refine['laptop'] = np.array([195,321,87,236,414,354,5129,3087,807,1550])
        t_filter_comps['laptop'] = np.array([5,10,5,7,15,11,719,263,74,100])


        # these can be read from the final portion of of the script output
        t_mmap['online'] = np.array([18.98, 85.21458578,  40.50961256,  17.71901989,  85.23642993, 30.11493444,  34.09690762,  18.95380235,  10.85061121,  31.97082043])
        t_patch['online'] = np.array([75.73,266.81324172,   332.06756997,   114.17053413,   267.06141853, 147.59935951,  3297.18628764,  2573.04009032,   578.88080835, 1725.74687123])
        t_refine['online'] = np.array([12.41, 91.77891779,    84.74378371,    31.84973955,    89.29527831, 25.1676743 ,  1689.06246471,  1282.98535109,    61.20671248, 322.67962313])
        t_filter_comps['online'] = np.array([0,0, 0, 0, 0, 0, 0, 0, 0, 0])



        pl.subplot(1,4,1)
        for key in ['cluster','desktop', 'laptop','online']:
            np.log10(t_mmap[key]+t_patch[key]+t_refine[key]+t_filter_comps[key])
            plt.scatter((size),np.log10(t_mmap[key]+t_patch[key]+t_refine[key]+t_filter_comps[key]),s=np.array(components)/10)
            plt.xlabel('size log_10(MB)')
            plt.ylabel('time log_10(s)')

        plt.plot((np.sort(size)),np.log10((np.sort(10**size))/31.45),'--.k')
        plt.legend(['acquisition-time','cluster (112 CPUs)','workstation (24 CPUs)', 'laptop (6 CPUs)','online (6 CPUs)'])
        pl.title('Total execution time')
        pl.xlim([3.8,5.2])
        pl.ylim([2.35,4.2])

        counter=2
        for key in ['cluster','laptop','online']:
            pl.subplot(1,4,counter)
            counter+=1
            if counter == 3:
                pl.title('Time per phase (cluster)')
                plt.ylabel('time (10^3 s)')
            elif counter == 2:
                pl.title('Time per phase (workstation)')
            else:
                pl.title('Time per phase (online)')

            plt.bar((size),(t_mmap[key]), width = 0.12, bottom = 0)
            plt.bar((size),(t_patch[key]), width = 0.12, bottom = (t_mmap[key]))
            plt.bar((size),(t_refine[key]), width = 0.12, bottom = (t_mmap[key]+t_patch[key]))
            plt.bar((size),(t_filter_comps[key]), width = 0.12, bottom = (t_mmap[key]+t_patch[key]+t_refine[key]))
            plt.xlabel('size log_10(MB)')
            if counter == 5:
                plt.legend(['Initialization','track activity','update shapes'])
            else:
                plt.legend(['mem mapping','patch init','refine sol','quality  filter','acquisition time'])

            plt.plot((np.sort(size)),(10**np.sort(size))/31.45,'--k')
            pl.xlim([3.6,5.2])

    #%% Figures, performance and timing online algorithm
    print('**** LOADING PREPROCESSED RESULTS ONLINE ALGORITHM ****')
    with np.load('RESULTS_SUMMARY/all_results_Jan_2018_online.npz') as ld:
        all_results = ld['all_results'][()]
    #%% Figure 4 b performance ONLINE ALGORITHM
    print('**** Performance online algorithm ****')
    print([(k[:-1],res['f1_score'])   for k,res in all_results.items()])

    #%%
    pl.figure('Figure 4a bottom')
    idxFile = 7
    key = params_movies[idxFile ]['gtname'].split('/')[0]+'/'
    idx_components_cnmf = all_results[key]['idx_components_cnmf']
    idx_components_gt = all_results[key]['idx_components_gt']
    tp_comp = all_results[key]['tp_comp']
    fp_comp = all_results[key]['fp_comp']
    tp_gt = all_results[key]['tp_gt']
    fn_gt = all_results[key]['fn_gt']
    A = all_results[key]['A']
    A_gt = all_results[key]['A_gt']
    A_gt = A_gt.to_2D().T
    C = all_results[key]['C']
    C_gt = all_results[key]['C_gt']
    fname_new = params_movies[idxFile]['fname']
    with np.load(os.path.join(os.path.split(fname_new)[0],'results_analysis_online_sensitive_SLURM_01.npz'), encoding = 'latin1') as ld:
        dims = ld['dims']

    Cn = cv2.resize(np.array(cm.load(key+'correlation_image.tif')).squeeze(),tuple(dims[::-1]))
    pl.subplot(1,2,1)
    a1 = plot_contours(A.tocsc()[:, idx_components_cnmf[tp_comp]], Cn, thr=0.9, colors='yellow', vmax = 0.75, display_numbers=False,cmap = 'gray')
    a2 = plot_contours(A_gt[:, idx_components_gt[tp_gt]], Cn, thr=0.9, vmax = 0.85, colors='r', display_numbers=False,cmap = 'gray')
    pl.subplot(1,2,2)
    a3 = plot_contours(A.tocsc()[:, idx_components_cnmf[fp_comp]], Cn, thr=0.9, colors='yellow', vmax = 0.75, display_numbers=False,cmap = 'gray')
    a4 = plot_contours(A_gt[:, idx_components_gt[fn_gt]], Cn, thr=0.9, vmax = 0.85, colors='r', display_numbers=False,cmap = 'gray')



    #%% Figure 8 timings (from here times for the online approach on mouse data are computed)
    timings = []
    names = []
    num_comps = []
    pctiles = []
    medians = []
    realtime = []
    time_init = []
    time_track_activity = []
    time_track_neurons = []
    for ind_dataset in [0,1,3,4,5,2,6,7,8]:
        with np.load(params_movies[ind_dataset ]['gtname'].split('/')[0]+'/'+'results_analysis_online_sensitive_SLURM_01.npz') as ld:
            print(params_movies[ind_dataset ]['gtname'].split('/')[0]+'/')
            print(ld.keys())
            print(ld['timings'])
            aaa = ld['noisyC']

            timings.append(ld['tottime'])
            names.append(params_movies[ind_dataset ]['gtname'].split('/')[0]+'/')
            num_comps.append(ld['Ab'][()].shape[-1])
            t = ld['tottime']
            pctiles.append(np.percentile(t[t<np.percentile(t,99)],[25,75])*params_movies[ind_dataset]['fr'])
            medians.append(np.percentile(t[t<np.percentile(t,99)],50)*params_movies[ind_dataset]['fr'])
            realtime.append(1/params_movies[ind_dataset]['fr'])
            time_init.append(ld['timings'][()]['init'])
            time_track_activity.append(np.sum(t[t<np.percentile(t,99)]))
            time_track_neurons.append(np.sum(t[t>=np.percentile(t,99)]))


    print('**** Timing online algorithm ****')
    print('Time initialization')
    print([(tm,nm) for tm,nm in zip(time_init,names)])
    print('Time track activity')
    print([(tm,nm) for tm,nm in zip(time_track_activity,names)])
    print('Time update shapes')
    print([(tm,nm) for tm,nm in zip(time_track_neurons,names)])



