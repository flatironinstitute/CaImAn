# -*- coding: utf-8 -*-

"""
Created on Fri Aug 25 14:49:36 2017

@author: agiovann
"""
import cv2

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
import sys
from caiman.utils.visualization import plot_contours
from caiman.source_extraction.cnmf import cnmf as cnmf
from caiman.source_extraction.cnmf.estimates import Estimates, compare_components
from caiman.cluster import setup_cluster
from caiman.source_extraction.cnmf import params as params
from caiman.source_extraction.cnmf.cnmf import load_CNMF

# %%  ANALYSIS MODE AND PARAMETERS
preprocessing_from_scratch = True  # whether to run the full pipeline or just creating figures

if preprocessing_from_scratch:
    reload = False
    plot_on = False
    save_on = False  # set to true to recreate
else:
    reload = True
    save_on = False

try:
    if 'pydevconsole' in sys.argv[0]:
        raise Exception()
    ID = sys.argv[1]
    ID = str(np.int(ID) - 1)
    print('Processing ID:' + str(ID))
    ID = [np.int(ID)]

except:
    ID = np.arange(2)
    print('ID NOT PASSED')



print_figs = True
skip_refinement = False
backend_patch = 'local'
backend_refine = 'local'
n_processes = 24
base_folder = '/mnt/ceph/neuro/DataForPublications/DATA_PAPER_ELIFE/'
n_pixels_per_process = 10000
block_size = 10000
num_blocks_per_run = 12




# %%
global_params = {'SNR_lowest': 0.5,
                 'min_SNR': 2,  # minimum SNR when considering adding a new neuron
                 'gnb': 2,  # number of background components
                 'rval_thr': 0.8,  # spatial correlation threshold
                 'min_rval_thr_rejected': -1.1,
                 'min_cnn_thresh': 0.99,
                 'max_classifier_probability_rejected': 0.1,
                 'p': 1,
                 'max_fitness_delta_accepted': -20,
                 'Npeaks': 5,
                 'min_SNR_patch': -10,
                 'min_r_val_thr_patch': 0.5,
                 'fitness_delta_min_patch': -5,
                 'update_background_components': True,
                 # whether to update the background components in the spatial phase
                 'low_rank_background': True,
                 # whether to update the using a low rank approximation. In the False case all the nonzero elements of the background components are updated using hals
                 # (to be used with one background per patch)
                 'only_init_patch': True,
                 'is_dendrites': False,  # if dendritic. In this case you need to set init_method to sparse_nmf
                 'alpha_snmf': None,
                 'init_method': 'greedy_roi',
                 'filter_after_patch': False
                 }
# %%
params_movies = []
# %%
params_movie = {'fname': 'N.03.00.t/Yr_d1_498_d2_467_d3_1_order_C_frames_2250_.mmap',
                'gtname': 'N.03.00.t/joined_consensus_active_regions.npy',
                # order of the autoregressive system
                'merge_thresh': 0.8,  # merging threshold, max correlation allow
                'rf': 25,  # half-size of the patches in pixels. rf=25, patches are 50x50    20
                'stride_cnmf': 10,  # amounpl.it of overlap between the patches in pixels
                'K': 4,  # number of components per patch
                'gSig': [8, 8],  # expected half size of neurons
                'n_chunks': 10,
                'swap_dim': False,
                'crop_pix': 0,
                'fr': 7,
                'decay_time': 0.4,
                }
params_movies.append(params_movie.copy())
# %%
params_movie = {'fname': 'N.04.00.t/Yr_d1_512_d2_512_d3_1_order_C_frames_3000_.mmap',
                'gtname': 'N.04.00.t/joined_consensus_active_regions.npy',
                # order of the autoregressive system
                'merge_thresh': 0.8,  # merging threshold, max correlation allow
                'rf': 20,  # half-size of the patches in pixels. rf=25, patches are 50x50    20
                'stride_cnmf': 10,  # amounpl.it of overlap between the patches in pixels
                'K': 5,  # number of components per patch
                'gSig': [5, 5],  # expected half size of neurons
                'n_chunks': 10,
                'swap_dim': False,
                'crop_pix': 0,
                'fr': 8,
                'decay_time': 0.5,  # rough length of a transient
                }
params_movies.append(params_movie.copy())

# %% yuste
params_movie = {'fname': 'YST/Yr_d1_200_d2_256_d3_1_order_C_frames_3000_.mmap',
                'gtname': 'YST/joined_consensus_active_regions.npy',
                # order of the autoregressive system
                'merge_thresh': 0.8,  # merging threshold, max correlation allow
                'rf': 15,  # half-size of the patches in pixels. rf=25, patches are 50x50    20
                'stride_cnmf': 10,  # amounpl.it of overlap between the patches in pixels
                'K': 10,  # number of components per patch
                'gSig': [5, 5],  # expected half size of neurons
                'fr': 10,
                'decay_time': 0.75,
                'n_chunks': 10,
                'swap_dim': False,
                'crop_pix': 0
                }
params_movies.append(params_movie.copy())
# %% neurofinder 00.00
params_movie = {'fname': 'N.00.00/Yr_d1_512_d2_512_d3_1_order_C_frames_2936_.mmap',
                'gtname': 'N.00.00/joined_consensus_active_regions.npy',
                'merge_thresh': 0.8,  # merging threshold, max correlation allow
                'rf': 20,  # half-size of the patches in pixels. rf=25, patches are 50x50    20
                'stride_cnmf': 10,  # amounpl.it of overlap between the patches in pixels
                'K': 7,  # number of components per patch
                'gSig': [6, 6],  # expected half size of neurons
                'decay_time': 0.4,
                'fr': 8,
                'n_chunks': 10,
                'swap_dim': False,
                'crop_pix': 10
                }
params_movies.append(params_movie.copy())
# %% neurofinder 01.01
params_movie = {'fname': 'N.01.01/Yr_d1_512_d2_512_d3_1_order_C_frames_1825_.mmap',
                'gtname': 'N.01.01/joined_consensus_active_regions.npy',
                'merge_thresh': 0.9,  # merging threshold, max correlation allow
                'rf': 20,  # half-size of the patches in pixels. rf=25, patches are 50x50    20
                'stride_cnmf': 10,  # amounpl.it of overlap between the patches in pixels
                'K': 7,  # number of components per patch
                'gSig': [6, 6],  # expected half size of neurons
                'decay_time': 1.4,
                'fr': 8,
                'n_chunks': 10,
                'swap_dim': False,
                'crop_pix': 2,
                }
params_movies.append(params_movie.copy())
# %% neurofinder 02.00
params_movie = {
    # 'fname': '/opt/local/Data/labeling/neurofinder.02.00/Yr_d1_512_d2_512_d3_1_order_C_frames_8000_.mmap',
    'fname': 'N.02.00/Yr_d1_512_d2_512_d3_1_order_C_frames_8000_.mmap',
    'gtname': 'N.02.00/joined_consensus_active_regions.npy',
    # order of the autoregressive system
    'merge_thresh': 0.8,  # merging threshold, max correlation allow
    'rf': 20,  # half-size of the patches in pixels. rf=25, patches are 50x50    20
    'stride_cnmf': 10,  # amounpl.it of overlap between the patches in pixels
    'K': 6,  # number of components per patch
    'gSig': [5, 5],  # expected half size of neurons
    'fr': 30,  # imaging rate in Hz
    'n_chunks': 10,
    'swap_dim': False,
    'crop_pix': 10,
    'decay_time': 0.3,
}
params_movies.append(params_movie.copy())
# %% Sue Ann k53
params_movie = {  # 'fname': '/opt/local/Data/labeling/k53_20160530/Yr_d1_512_d2_512_d3_1_order_C_frames_116043_.mmap',
    'fname': 'K53/Yr_d1_512_d2_512_d3_1_order_C_frames_116043_.mmap',
    'gtname': 'K53/joined_consensus_active_regions.npy',
    # order of the autoregressive system
    'merge_thresh': 0.8,  # merging threshold, max correlation allow
    'rf': 20,  # half-size of the patches in pixels. rf=25, patches are 50x50    20
    'stride_cnmf': 10,  # amounpl.it of overlap between the patches in pixels
    'K': 10,  # number of components per patch
    'gSig': [6, 6],  # expected half size of neurons
    'fr': 30,
    'decay_time': 0.3,
    'n_chunks': 10,
    'swap_dim': False,
    'crop_pix': 2,
}
params_movies.append(params_movie.copy())
# %% J115
params_movie = {
    # 'fname': '/opt/local/Data/labeling/J115_2015-12-09_L01_ELS/Yr_d1_463_d2_472_d3_1_order_C_frames_90000_.mmap',
    'fname': 'J115/Yr_d1_463_d2_472_d3_1_order_C_frames_90000_.mmap',
    'gtname': 'J115/joined_consensus_active_regions.npy',
    # order of the autoregressive system
    'merge_thresh': 0.8,  # merging threshold, max correlation allow
    'rf': 20,  # half-size of the patches in pixels. rf=25, patches are 50x50    20
    'stride_cnmf': 10,  # amounpl.it of overlap between the patches in pixels
    'K': 8,  # number of components per patch
    'gSig': [7, 7],  # expected half size of neurons
    'fr': 30,
    'decay_time': 0.4,
    'n_chunks': 10,
    'swap_dim': False,
    'crop_pix': 2,

}
params_movies.append(params_movie.copy())
# %% J123
params_movie = {
    # 'fname': '/opt/local/Data/labeling/J123_2015-11-20_L01_0/Yr_d1_458_d2_477_d3_1_order_C_frames_41000_.mmap',
    'fname': 'J123/Yr_d1_458_d2_477_d3_1_order_C_frames_41000_.mmap',
    'gtname': 'J123/joined_consensus_active_regions.npy',
    # order of the autoregressive system
    'merge_thresh': 0.8,  # merging threshold, max correlation allow
    'rf': 40,  # half-size of the patches in pixels. rf=25, patches are 50x50    20
    'stride_cnmf': 20,  # amounpl.it of overlap between the patches in pixels
    'K': 11,  # number of components per patch
    'gSig': [8, 8],  # expected half size of neurons
    'decay_time': 0.5,
    'fr': 30,
    'n_chunks': 10,
    'swap_dim': False,
    'crop_pix': 2,
}
params_movies.append(params_movie.copy())


# %%
def myfun(x):
    from caiman.source_extraction.cnmf.deconvolution import constrained_foopsi
    dc = constrained_foopsi(*x)
    return (dc[0], dc[5])


# %%
def fun_exc(x):
    from scipy.stats import norm
    from caiman.components_evaluation import compute_event_exceptionality

    fluo, param = x
    N_samples = np.ceil(param['fr'] * param['decay_time']).astype(np.int)
    ev = compute_event_exceptionality(np.atleast_2d(fluo), N=N_samples)
    return -norm.ppf(np.exp(np.array(ev[1]) / N_samples))


# %%
if preprocessing_from_scratch:
    all_perfs = []
    all_rvalues = []
    all_comp_SNR_raw = []
    all_comp_SNR_delta = []
    all_predictions = []
    all_labels = []
    all_results = dict()

    ALL_CCs = []

    for params_movie in np.array(params_movies)[ID]:
        #    params_movie['gnb'] = 3
        params_display = {
            'downsample_ratio': .2,
            'thr_plot': 0.8
        }

        fname_new = os.path.join(base_folder, params_movie['fname'])
        print(fname_new)
        # %% LOAD MEMMAP FILE
        Yr, dims, T = cm.load_memmap(fname_new)
        d1, d2 = dims
        images = np.reshape(Yr.T, [T] + list(dims), order='F')
        # TODO: needinfo
        Y = np.reshape(Yr, dims + (T,), order='F')
        m_images = cm.movie(images)
        if plot_on:
            if m_images.shape[0] < 5000:
                Cn = m_images.local_correlations(swap_dim=params_movie['swap_dim'], frames_per_chunk=1500)
                Cn[np.isnan(Cn)] = 0
            else:
                Cn = np.array(cm.load(
                    ('/'.join(
                        params_movie['gtname'].split('/')[:-2] + ['projections', 'correlation_image.tif'])))).squeeze()

        check_nan = False
        # %% start cluster
        # TODO: show screenshot 10
        try:
            cm.stop_server()
            dview.terminate()
        except:
            print('No clusters to stop')

        c, dview, n_processes = setup_cluster(
            backend=backend_patch, n_processes=n_processes, single_thread=False)
        # %%
        params_dict = {'fnames': [fname_new],
                       'fr': params_movie['fr'],
                       'decay_time': params_movie['decay_time'],
                       'rf': params_movie['rf'],
                       'stride': params_movie['stride_cnmf'],
                       'K': params_movie['K'],
                       'gSig': params_movie['gSig'],
                       'merge_thr': params_movie['merge_thresh'],
                       'p': global_params['p'],
                       'nb': global_params['gnb'],
                       'only_init': global_params['only_init_patch'],
                       'dview': dview,
                       'method_deconvolution': 'oasis',
                       'border_pix': params_movie['crop_pix'],
                       'low_rank_background': global_params['low_rank_background'],
                       'rolling_sum': True,
                       'nb_patch': 1,
                       'check_nan': check_nan,
                       'block_size_temp': block_size,
                       'block_size_spat': block_size,
                       'num_blocks_per_run': num_blocks_per_run,
                       'n_pixels_per_process': 4000,
                       'ssub': 2,
                       'tsub': 2,
                       'p_tsub': 1,
                       'p_ssub': 1,
                       'thr_method': 'nrg'
                       }

        init_method = global_params['init_method']

        opts = params.CNMFParams(params_dict=params_dict)
        if reload:
            cnm2 = load_CNMF(fname_new[:-5] + '_cnmf_gsig.hdf5')
        else:
            # %% Extract spatial and temporal components on patches
            t1 = time.time()
            print('Starting CNMF')
            cnm = cnmf.CNMF(n_processes, params=opts, dview=dview)
            cnm = cnm.fit(images)
            t_patch = time.time() - t1
            # %%
            try:
                dview.terminate()
            except:
                pass
            c, dview, n_processes = cm.cluster.setup_cluster(
                backend=backend_refine, n_processes=n_processes, single_thread=False)
            # %%
            if plot_on:
                cnm.estimates.plot_contours(img=Cn)

            # %% UPDATE SOME PARAMETERS
            cnm.params.change_params({'update_background_components': global_params['update_background_components'],
                                      'skip_refinement': skip_refinement,
                                      'n_pixels_per_process': n_pixels_per_process, 'dview': dview})
            # %%
            t1 = time.time()
            cnm2 = cnm.refit(images, dview=dview)
            t_refit = time.time() - t1
            if save_on:
                cnm2.save(fname_new[:-5] + '_cnmf_gsig.hdf5')

        # %%
        if plot_on:
            cnm2.estimates.plot_contours(img=Cn)
        # %% check quality of components and eliminate low quality
        cnm2.params.set('quality', {'SNR_lowest': global_params['SNR_lowest'],
                                    'min_SNR': global_params['min_SNR'],
                                    'rval_thr': global_params['rval_thr'],
                                    'rval_lowest': global_params['min_rval_thr_rejected'],
                                    #'Npeaks': global_params['Npeaks'],
                                    'use_cnn': True,
                                    'min_cnn_thr': global_params['min_cnn_thresh'],
                                    'cnn_lowest': global_params['max_classifier_probability_rejected'],
                                    #'thresh_fitness_delta': global_params['max_fitness_delta_accepted'],
                                    'gSig_range': None})

        t1 = time.time()
        cnm2.estimates.evaluate_components(images, cnm2.params, dview=dview)
        cnm2.estimates.select_components(use_object=True)
        t_eva_comps = time.time() - t1
        print(' ***** ')
        print((len(cnm2.estimates.C)))
        # %%
        if plot_on:
            cnm2.estimates.plot_contours(img=Cn)
        # %% prepare ground truth masks
        gt_file = os.path.join(os.path.split(fname_new)[0], os.path.split(fname_new)[1][:-4] + 'match_masks.npz')
        with np.load(gt_file, encoding='latin1') as ld:
            print(ld.keys())
            Cn_orig = ld['Cn']

            gt_estimate = Estimates(A=scipy.sparse.csc_matrix(ld['A_gt'][()]), b=ld['b_gt'], C=ld['C_gt'],
                                    f=ld['f_gt'], R=ld['YrA_gt'], dims=(ld['d1'], ld['d2']))

        min_size_neuro = 3 * 2 * np.pi
        max_size_neuro = (2 * params_dict['gSig'][0]) ** 2 * np.pi
        gt_estimate.threshold_spatial_components(maxthr=0.2, dview=dview)
        nrn_size = gt_estimate.remove_small_large_neurons(min_size_neuro, max_size_neuro)
        nrn_dup = gt_estimate.remove_duplicates(predictions=None, r_values=None, dist_thr=0.1, min_dist=10,
                                          thresh_subset=0.6)
        idx_components_gt = nrn_size[nrn_dup]
        print(gt_estimate.A_thr.shape)
        # %% prepare CNMF maks
        cnm2.estimates.threshold_spatial_components(maxthr=0.2, dview=dview)
        cnm2.estimates.remove_small_large_neurons(min_size_neuro, max_size_neuro)
        cnm2.estimates.remove_duplicates(r_values=None, dist_thr=0.1, min_dist=10, thresh_subset=0.6)

        # %%
        params_display = {
            'downsample_ratio': .2,
            'thr_plot': 0.8
        }

        pl.rcParams['pdf.fonttype'] = 42
        font = {'family': 'Arial',
                'weight': 'regular',
                'size': 20}
        pl.rc('font', **font)

        plot_results = False
        if plot_results:
            pl.figure(figsize=(30, 20))

        tp_gt, tp_comp, fn_gt, fp_comp, performance_cons_off = compare_components(gt_estimate, cnm2.estimates,
                                                                                  Cn=Cn_orig, thresh_cost=.8,
                                                                                  min_dist=10,
                                                                                  print_assignment=False,
                                                                                  labels=['GT', 'Offline'],
                                                                                  plot_results=False)

        print(fname_new+str({a: b.astype(np.float16) for a, b in performance_cons_off.items()}))
        cnm2.estimates.A_thr = scipy.sparse.csc_matrix(cnm2.estimates.A_thr)
        if save_on:
            cnm2.save(fname_new[:-5] + '_cnmf_gsig_after_analysis.hdf5')

        performance_cons_off['fname_new'] = fname_new
        performance_tmp = performance_cons_off.copy()
        performance_tmp['tp_gt'] = tp_gt
        performance_tmp['tp_comp'] = tp_comp
        performance_tmp['fn_gt'] = fn_gt
        performance_tmp['fp_comp'] = fp_comp
        performance_tmp['predictionsCNN'] = cnm2.estimates.cnn_preds
        performance_tmp['A'] = cnm2.estimates.A
        performance_tmp['C'] = cnm2.estimates.C
        performance_tmp['SNR_comp'] = cnm2.estimates.SNR_comp
        performance_tmp['Cn'] = Cn_orig
        performance_tmp['params'] = params_movie
        performance_tmp['dims'] = dims
        performance_tmp['CCs'] = [scipy.stats.pearsonr(a, b)[0] for a, b in
                        zip(gt_estimate.C[tp_gt], cnm2.estimates.C[tp_comp])]

        with np.load(os.path.join(base_folder,fname_new.split('/')[-2],'gt_eval.npz')) as ld:
            print(ld.keys())
            performance_tmp.update(ld)

        performance_tmp['idx_components_gt'] = idx_components_gt
        # ALL_CCs.append([scipy.stats.pearsonr(a, b)[0] for a, b in
        #                 zip(gt_estimate.C[tp_gt], cnm2.estimates.C[tp_comp])])
        #
        # performance_tmp['ALL_CCs'] = ALL_CCs

        all_results[fname_new.split('/')[-2]] = performance_tmp

        if save_on :
            print('SAVING...' + fname_new[:-5] + '_perf_Sep_2018_gsig.npz')
            np.savez(fname_new[:-5] + '_perf_Sep_2018_gsig.npz', all_results=performance_tmp)

    if save_on:
        # here eventually save when in a loop
        np.savez(os.path.join(base_folder,'all_res_sept_2018.npz'), all_results=all_results)
        print('Saving not implementd')


else:

    # %% RELOAD ALL THE RESULTS INSTEAD OF REGENERATING THEM
    base_folder = '/mnt/ceph/neuro/DataForPublications/DATA_PAPER_ELIFE/'
    with np.load(os.path.join(base_folder,'all_res_sept_2018.npz')) as ld:
        all_results_ = ld['all_results'][()]

    #%%
    # for k, fl_results in all_results_.items():
    #     print({kk_:vv_ for kk_,vv_ in fl_results.items() if 'gt' in kk_ and kk_ not in ['tp_gt','fn_gt']}.keys())
    #     print(os.path.join(base_folder,k,'gt_eval.npz'))
    #     np.savez(os.path.join(base_folder,k,'gt_eval.npz'),**{kk_:vv_ for kk_,vv_ in fl_results.items()
    #                                                          if ('gt' in kk_ )and (kk_ not in ['tp_gt','fn_gt'])})
    # %% CREATE FIGURES
    if print_figs:
        # %% FIGURE 5a MASKS  (need to use the dataset k53)
        pl.figure('Figure 5a masks')
        idxFile = 7
        name_file = 'J115'
        predictionsCNN = all_results[name_file ]['predictionsCNN']
        idx_components_gt = all_results[name_file]['idx_components_gt']
        tp_comp = all_results[name_file ]['tp_comp']
        fp_comp = all_results[name_file ]['fp_comp']
        tp_gt = all_results[name_file ]['tp_gt']
        fn_gt = all_results[name_file ]['fn_gt']
        A = all_results[name_file ]['A']
        A_gt = all_results[name_file ]['A_gt'][()]
        C = all_results[name_file ]['C']
        C_gt = all_results[name_file ]['C_gt']
        fname_new = all_results[name_file ]['params']['fname']
        dims = all_results[name_file]['dims']
        Cn = all_results[name_file ]['Cn']
        pl.ylabel('spatial components')
        idx_comps_high_r = [np.argsort(predictionsCNN[tp_comp])[[-6, -5, -4, -3, -2]]]
        idx_comps_high_r_cnmf = tp_comp[idx_comps_high_r]
        idx_comps_high_r_gt = idx_components_gt[tp_gt][idx_comps_high_r]
        images_nice = (A.tocsc()[:, idx_comps_high_r_cnmf].toarray().reshape(dims + (-1,), order='F')).transpose(2, 0,

                                                                                                                 1)
        images_nice_gt = (A_gt.tocsc()[:, idx_comps_high_r_gt].toarray().reshape(dims + (-1,), order='F')).transpose(2,
                                                                                                                     0,
                                                                                                                     1)
        cms = np.array([scipy.ndimage.center_of_mass(img) for img in images_nice]).astype(np.int)
        images_nice_crop = [img[cm_[0] - 15:cm_[0] + 15, cm_[1] - 15:cm_[1] + 15] for cm_, img in zip(cms, images_nice)]
        images_nice_crop_gt = [img[cm_[0] - 15:cm_[0] + 15, cm_[1] - 15:cm_[1] + 15] for cm_, img in
                               zip(cms, images_nice_gt)]

        indexes = [1, 3, 5, 7, 9, 2, 4, 6, 8, 10]
        count = 0
        for img in images_nice_crop:
            pl.subplot(5, 2, indexes[count])
            pl.imshow(img)
            pl.axis('off')
            count += 1

        for img in images_nice_crop_gt:
            pl.subplot(5, 2, indexes[count])
            pl.imshow(img)
            pl.axis('off')
            count += 1

        # %% FIGURE 5a Traces  (need to use the dataset J115)
        pl.figure('Figure 5a traces')
        traces_gt = C_gt[idx_comps_high_r_gt]  # + YrA_gt[idx_comps_high_r_gt]
        traces_cnmf = C[idx_comps_high_r_cnmf]  # + YrA[idx_comps_high_r_cnmf]
        traces_gt /= np.max(traces_gt, 1)[:, None]
        traces_cnmf /= np.max(traces_cnmf, 1)[:, None]
        pl.plot(scipy.signal.decimate(traces_cnmf, 10, 1).T - np.arange(5) * 1, 'y')
        pl.plot(scipy.signal.decimate(traces_gt, 10, 1).T - np.arange(5) * 1, 'k', linewidth=.5)

        # %% FIGURE 5c
        pl.figure('Figure 5c')
        pl.rcParams['pdf.fonttype'] = 42

        with np.load('RESULTS_SUMMARY/ALL_CORRELATIONS_ONLINE_CONSENSUS.npz') as ld:
            print(ld.keys())
            xcorr_online = ld['ALL_CCs']
        #            xcorr_offline = list(np.array(xcorr_offline[[0,1,3,4,5,2,6,7,8,9]]))

        xcorr_offline = []
        for k, fl_results in all_results.items():
            xcorr_offline.append(fl_results['CCs'])


        names = ['0300.T',
                 '0400.T',
                 'YT',
                 '0000',
                 '0200',
                 '0101',
                 'k53',
                 'J115',
                 'J123']

        pl.subplot(1, 2, 1)

        pl.hist(np.concatenate(xcorr_online), bins=np.arange(0, 1, .01))
        pl.hist(np.concatenate(xcorr_offline), bins=np.arange(0, 1, .01))
        a = pl.hist(np.concatenate(xcorr_online[:]), bins=np.arange(0, 1, .01))
        a3 = pl.hist(np.concatenate(xcorr_offline)[:], bins=np.arange(0, 1, .01))

        a_on = np.cumsum(a[0] / a[0].sum())
        a_off = np.cumsum(a3[0] / a3[0].sum())
        # %
        pl.close()
        pl.figure('Figure 5c')

        pl.plot(np.arange(0.01, 1, .01), a_on);
        pl.plot(np.arange(0.01, 1, .01), a_off)
        pl.legend(['online', 'offline'])
        pl.xlabel('correlation (r)')
        pl.ylabel('cumulative probability')
        medians_on = []
        iqrs_on = []
        medians_off = []
        iqrs_off = []
        for onn, off in zip(xcorr_online, xcorr_offline):
            medians_on.append(np.median(onn))
            iqrs_on.append(np.percentile(onn, [25, 75]))
            medians_off.append(np.median(off))
            iqrs_off.append(np.percentile(off, [25, 75]))
        # %%
        pl.figure('Figure 5b')
        mu_on = np.array([np.median(xc) for xc in xcorr_online[:-1]])
        mu_off = np.array([np.median(xc) for xc in xcorr_offline[:]])
        pl.plot(np.concatenate([mu_off[:, None], mu_on[:, None]], axis=1).T, 'o-')
        pl.xlabel('Offline and Online')
        pl.ylabel('Correlation coefficient')
        # %% FIGURE 4a TOP (need to use the dataset J115)
        pl.figure('Figure 4a top')
        pl.subplot(1, 2, 1)
        a1 = plot_contours(A.tocsc()[:, tp_comp], Cn, thr=0.9, colors='yellow', vmax=0.75,
                           display_numbers=False, cmap='gray')
        a2 = plot_contours(A_gt.tocsc()[:, tp_gt], Cn, thr=0.9, vmax=0.85, colors='r',
                           display_numbers=False, cmap='gray')
        pl.subplot(1, 2, 2)
        a3 = plot_contours(A.tocsc()[:, fp_comp], Cn, thr=0.9, colors='yellow', vmax=0.75,
                           display_numbers=False, cmap='gray')
        a4 = plot_contours(A_gt.tocsc()[:, idx_components_gt[fn_gt]], Cn, thr=0.9, vmax=0.85, colors='r',
                           display_numbers=False, cmap='gray')
        # %% FIGURE 4 c and d
        from matplotlib.pyplot import cm as cmap
        pl.figure("Figure 4c and 4d")
        color = cmap.jet(np.linspace(0, 1, 10))
        i = 0
        legs = []
        all_ys = []
        SNRs = np.arange(0, 10)
        pl.subplot(1, 3, 1)

        for k, fl_results in all_results.items():
            print(k)
            nm = k[:]
            nm = nm.replace('neurofinder', 'NF')
            nm = nm.replace('final_map', '')
            nm = nm.replace('.', '')
            nm = nm.replace('Data', '')
            idx_components_gt = fl_results['idx_components_gt']
            tp_gt = fl_results['tp_gt']
            tp_comp = fl_results['tp_comp']
            fn_gt = fl_results['fn_gt']
            fp_comp = fl_results['fp_comp']
            comp_SNR = fl_results['SNR_comp']
            comp_SNR_gt = fl_results['comp_SNR_gt'][idx_components_gt]
            snr_gt = comp_SNR_gt[tp_gt]
            snr_gt_fn = comp_SNR_gt[fn_gt]
            snr_cnmf = comp_SNR[tp_comp]
            snr_cnmf_fp = comp_SNR[fp_comp]
            all_results_fake, all_results_OR, all_results_AND = precision_snr(snr_gt, snr_gt_fn, snr_cnmf, snr_cnmf_fp,
                                                                              SNRs)
            all_ys.append(all_results_fake[:, -1])
            pl.fill_between(SNRs, all_results_OR[:, -1], all_results_AND[:, -1], color=color[i], alpha=.1)
            pl.plot(SNRs, all_results_fake[:, -1], '.-', color=color[i])
            #            pl.plot(x[::1]+np.random.normal(scale=.07,size=10),y[::1], 'o',color=color[i])
            pl.ylim([0.65, 1.05])
            legs.append(nm[:7])
            i += 1
        #            break
        pl.plot(SNRs, np.mean(all_ys, 0), 'k--', alpha=1, linewidth=2)
        pl.legend(legs + ['average'], fontsize=10)
        pl.xlabel('SNR threshold')

        pl.ylabel('F1 SCORE')
        # %
        i = 0
        legs = []
        for k, fl_results in all_results.items():
            x = []
            y = []

            idx_components_gt = fl_results['idx_components_gt']
            tp_gt = fl_results['tp_gt']
            tp_comp = fl_results['tp_comp']
            fn_gt = fl_results['fn_gt']
            fp_comp = fl_results['fp_comp']
            comp_SNR = fl_results['SNR_comp']
            comp_SNR_gt = fl_results['comp_SNR_gt'][idx_components_gt]
            snr_gt = comp_SNR_gt[tp_gt]
            snr_gt_fn = comp_SNR_gt[fn_gt]
            snr_cnmf = comp_SNR[tp_comp]
            snr_cnmf_fp = comp_SNR[fp_comp]
            all_results_fake, all_results_OR, all_results_AND = precision_snr(snr_gt, snr_gt_fn, snr_cnmf,
                                                                              snr_cnmf_fp, SNRs)
            prec_idx = 0
            recall_idx = 1
            f1_idx = 2
            if 'K53' in k:
                pl.subplot(1, 3, 2)
                pl.scatter(snr_gt, snr_cnmf, color='k', alpha=.15)
                pl.scatter(snr_gt_fn, np.random.normal(scale=.25, size=len(snr_gt_fn)), color='g', alpha=.15)
                pl.scatter(np.random.normal(scale=.25, size=len(snr_cnmf_fp)), snr_cnmf_fp, color='g', alpha=.15)
                pl.fill_between([20, 40], [-2, -2], [40, 40], alpha=.05, color='r')
                pl.fill_between([-2, 40], [20, 20], [40, 40], alpha=.05, color='b')
                pl.xlabel('SNR CA')
                pl.ylabel('SNR CaImAn')
            pl.subplot(1, 3, 3)
            pl.fill_between(SNRs, all_results_OR[:, prec_idx], all_results_AND[:, prec_idx], color='b', alpha=.1)
            pl.plot(SNRs, all_results_fake[:, prec_idx], '.-', color='b')
            pl.fill_between(SNRs, all_results_OR[:, recall_idx], all_results_AND[:, recall_idx], color='r',
                                alpha=.1)
            pl.plot(SNRs, all_results_fake[:, recall_idx], '.-', color='r')

            # pl.fill_between(SNRs, all_results_OR[:, f1_idx], all_results_AND[:, f1_idx], color='g', alpha=.1)
            # pl.plot(SNRs, all_results_fake[:, f1_idx], '.-', color='g')
            pl.legend(['precision', 'recall'], fontsize=10)
            pl.xlabel('SNR threshold')
        # %% FIGURE 4 b  performance in detecting neurons (results have been manually annotated on an excel spreadsheet and reported here below)
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
        f1s['online'] = [0.76, 0.678, 0.783, 0.721, 0.769, 0.725, 0.818, 0.805,
                         0.803]  # these can be read from last part of script
        f1s['L1'] = [np.nan, np.nan, 0.78, np.nan, 0.89, 0.8, 0.89, np.nan, 0.85]  # Human 1
        f1s['L2'] = [0.9, 0.69, 0.9, 0.92, 0.87, 0.89, 0.92, 0.93, 0.83]  # Human 2
        f1s['L3'] = [0.85, 0.75, 0.82, 0.83, 0.84, 0.78, 0.93, 0.94, 0.9]  # Human 3
        f1s['L4'] = [0.78, 0.87, 0.79, 0.87, 0.82, 0.75, 0.83, 0.83, 0.91]  # Human 4

        all_of = ((np.vstack([f1s['L1'], f1s['L2'], f1s['L3'], f1s['L4'], f1s['batch'], f1s['online']])))

        for i in range(6):
            pl.plot(i + np.random.random(9) * .2, all_of[i, :], '.')
            pl.plot([i - .5, i + .5], [np.nanmean(all_of[i, :])] * 2, 'k')
        plt.xticks(range(6), ['L1', 'L2', 'L3', 'L4', 'batch', 'online'], rotation=45)
        pl.ylabel('F1-score')

        # %% FIGURE  timing to create a mmap file (Figures 8 a and b left, "mem mapping")
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
                                       border_to_0=0, dview=dview, n_chunks=80)  # exclude borders
            t2 = time.time() - t1
        # %%FIGURE 8 a and b time performances (results have been manually annotated on an excel spreadsheet and reported here below)
        import pylab as plt

        pl.figure("Figure 8a and 8b")
        import numpy as np

        plt.rcParams['pdf.fonttype'] = 42
        font = {'family': 'Arial',
                'weight': 'regular',
                'size': 20}

        t_mmap = dict()
        t_patch = dict()
        t_refine = dict()
        t_filter_comps = dict()

        size = np.log10(np.array([2.1, 3.1, 0.6, 3.1, 8.4, 1.9, 121.7, 78.7, 35.8, 50.3]) * 1000)
        components = np.array([368, 935, 476, 1060, 1099, 1387, 1541, 1013, 398, 1064])
        components = np.array([368, 935, 476, 1060, 1099, 1387, 1541, 1013, 398, 1064])

        t_mmap['cluster'] = np.array([np.nan, 41, np.nan, np.nan, 109, np.nan, 561, 378, 135, 212])
        t_patch['cluster'] = np.array([np.nan, 46, np.nan, np.nan, 92, np.nan, 1063, 469, 142, 372])
        t_refine['cluster'] = np.array([np.nan, 225, np.nan, np.nan, 256, np.nan, 1065, 675, 265, 422])
        t_filter_comps['cluster'] = np.array([np.nan, 7, np.nan, np.nan, 11, np.nan, 143, 77, 30, 57])

        t_mmap['desktop'] = np.array([25, 41, 11, 41, 135, 23, 690, 510, 176, 163])
        t_patch['desktop'] = np.array([21, 43, 16, 48, 85, 45, 2150, 949, 316, 475])
        t_refine['desktop'] = np.array([105, 205, 43, 279, 216, 254, 1749, 837, 237, 493])
        t_filter_comps['desktop'] = np.array([3, 5, 2, 5, 9, 7, 246, 81, 36, 38])

        t_mmap['laptop'] = np.array([4.7, 27, 3.6, 18, 144, 11, 731, 287, 125, 248])
        t_patch['laptop'] = np.array([58, 84, 47, 77, 174, 85, 2398, 1587, 659, 1203])
        t_refine['laptop'] = np.array([195, 321, 87, 236, 414, 354, 5129, 3087, 807, 1550])
        t_filter_comps['laptop'] = np.array([5, 10, 5, 7, 15, 11, 719, 263, 74, 100])

        # these can be read from the final portion of of the script output
        t_mmap['online'] = np.array(
            [18.98, 85.21458578, 40.50961256, 17.71901989, 85.23642993, 30.11493444, 34.09690762, 18.95380235,
             10.85061121, 31.97082043])
        t_patch['online'] = np.array(
            [75.73, 266.81324172, 332.06756997, 114.17053413, 267.06141853, 147.59935951, 3297.18628764, 2573.04009032,
             578.88080835, 1725.74687123])
        t_refine['online'] = np.array(
            [12.41, 91.77891779, 84.74378371, 31.84973955, 89.29527831, 25.1676743, 1689.06246471, 1282.98535109,
             61.20671248, 322.67962313])
        t_filter_comps['online'] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        pl.subplot(1, 4, 1)
        for key in ['cluster', 'desktop', 'laptop', 'online']:
            np.log10(t_mmap[key] + t_patch[key] + t_refine[key] + t_filter_comps[key])
            plt.scatter((size), np.log10(t_mmap[key] + t_patch[key] + t_refine[key] + t_filter_comps[key]),
                        s=np.array(components) / 10)
            plt.xlabel('size log_10(MB)')
            plt.ylabel('time log_10(s)')

        plt.plot((np.sort(size)), np.log10((np.sort(10 ** size)) / 31.45), '--.k')
        plt.legend(
            ['acquisition-time', 'cluster (112 CPUs)', 'workstation (24 CPUs)', 'laptop (6 CPUs)', 'online (6 CPUs)'])
        pl.title('Total execution time')
        pl.xlim([3.8, 5.2])
        pl.ylim([2.35, 4.2])

        counter = 2
        for key in ['cluster', 'laptop', 'online']:
            pl.subplot(1, 4, counter)
            counter += 1
            if counter == 3:
                pl.title('Time per phase (cluster)')
                plt.ylabel('time (10^3 s)')
            elif counter == 2:
                pl.title('Time per phase (workstation)')
            else:
                pl.title('Time per phase (online)')

            plt.bar((size), (t_mmap[key]), width=0.12, bottom=0)
            plt.bar((size), (t_patch[key]), width=0.12, bottom=(t_mmap[key]))
            plt.bar((size), (t_refine[key]), width=0.12, bottom=(t_mmap[key] + t_patch[key]))
            plt.bar((size), (t_filter_comps[key]), width=0.12, bottom=(t_mmap[key] + t_patch[key] + t_refine[key]))
            plt.xlabel('size log_10(MB)')
            if counter == 5:
                plt.legend(['Initialization', 'track activity', 'update shapes'])
            else:
                plt.legend(['mem mapping', 'patch init', 'refine sol', 'quality  filter', 'acquisition time'])

            plt.plot((np.sort(size)), (10 ** np.sort(size)) / 31.45, '--k')
            pl.xlim([3.6, 5.2])

    # %% Figures, performance and timing online algorithm
    print('**** LOADING PREPROCESSED RESULTS ONLINE ALGORITHM ****')
    with np.load('RESULTS_SUMMARY/all_results_Jan_2018_online.npz') as ld:
        all_results = ld['all_results'][()]
    # %% Figure 4 b performance ONLINE ALGORITHM
    print('**** Performance online algorithm ****')
    print([(k[:-1], res['f1_score']) for k, res in all_results.items()])

    # %%
    pl.figure('Figure 4a bottom')
    idxFile = 7
    key = params_movies[idxFile]['gtname'].split('/')[0] + '/'
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
    with np.load(os.path.join(os.path.split(fname_new)[0], 'results_analysis_online_sensitive_SLURM_01.npz'),
                 encoding='latin1') as ld:
        dims = ld['dims']

    Cn = cv2.resize(np.array(cm.load(key + 'correlation_image.tif')).squeeze(), tuple(dims[::-1]))
    pl.subplot(1, 2, 1)
    a1 = plot_contours(A.tocsc()[:, idx_components_cnmf[tp_comp]], Cn, thr=0.9, colors='yellow', vmax=0.75,
                       display_numbers=False, cmap='gray')
    a2 = plot_contours(A_gt[:, idx_components_gt[tp_gt]], Cn, thr=0.9, vmax=0.85, colors='r', display_numbers=False,
                       cmap='gray')
    pl.subplot(1, 2, 2)
    a3 = plot_contours(A.tocsc()[:, idx_components_cnmf[fp_comp]], Cn, thr=0.9, colors='yellow', vmax=0.75,
                       display_numbers=False, cmap='gray')
    a4 = plot_contours(A_gt[:, idx_components_gt[fn_gt]], Cn, thr=0.9, vmax=0.85, colors='r', display_numbers=False,
                       cmap='gray')

    # %% Figure 8 timings (from here times for the online approach on mouse data are computed)
    timings = []
    names = []
    num_comps = []
    pctiles = []
    medians = []
    realtime = []
    time_init = []
    time_track_activity = []
    time_track_neurons = []
    for ind_dataset in [0, 1, 3, 4, 5, 2, 6, 7, 8]:
        with np.load(params_movies[ind_dataset]['gtname'].split('/')[
                         0] + '/' + 'results_analysis_online_sensitive_SLURM_01.npz') as ld:
            print(params_movies[ind_dataset]['gtname'].split('/')[0] + '/')
            print(ld.keys())
            print(ld['timings'])
            aaa = ld['noisyC']

            timings.append(ld['tottime'])
            names.append(params_movies[ind_dataset]['gtname'].split('/')[0] + '/')
            num_comps.append(ld['Ab'][()].shape[-1])
            t = ld['tottime']
            pctiles.append(np.percentile(t[t < np.percentile(t, 99)], [25, 75]) * params_movies[ind_dataset]['fr'])
            medians.append(np.percentile(t[t < np.percentile(t, 99)], 50) * params_movies[ind_dataset]['fr'])
            realtime.append(1 / params_movies[ind_dataset]['fr'])
            time_init.append(ld['timings'][()]['init'])
            time_track_activity.append(np.sum(t[t < np.percentile(t, 99)]))
            time_track_neurons.append(np.sum(t[t >= np.percentile(t, 99)]))

    print('**** Timing online algorithm ****')
    print('Time initialization')
    print([(tm, nm) for tm, nm in zip(time_init, names)])
    print('Time track activity')
    print([(tm, nm) for tm, nm in zip(time_track_activity, names)])
    print('Time update shapes')
    print([(tm, nm) for tm, nm in zip(time_track_neurons, names)])
