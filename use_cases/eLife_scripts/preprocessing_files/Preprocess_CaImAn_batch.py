# -*- coding: utf-8 -*-
"""
Complete pipeline for CaImAn batch processing and comparison with consensus
annotation. The script processes one (or more) of the provided datasets and 
compares against the consensus annotations. It generates the contour plots
of the detected components (Fig. 4a)
    For more information check the companion paper.
@author: Andrea Giovannucci @agiovann and Eftychios Pnevmatikakis @epnev
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
from caiman.source_extraction.cnmf import cnmf as cnmf
from caiman.source_extraction.cnmf.estimates import Estimates, compare_components
from caiman.cluster import setup_cluster
from caiman.source_extraction.cnmf import params as params
from caiman.source_extraction.cnmf.cnmf import load_CNMF
from caiman.base.movies import from_zipfiles_to_movie_lists
import shutil
import glob
import logging
import warnings

logging.basicConfig(format=
                    "%(relativeCreated)12d [%(filename)s:%(funcName)20s():%(lineno)s]"\
                    "[%(process)d] %(message)s",
                    level=logging.INFO)

warnings.filterwarnings("ignore", category=FutureWarning)

# %%  ANALYSIS MODE AND PARAMETERS
reload = False
plot_on = False

save_on = False  # set to true to recreate results for each file
save_all = False  # set to True to generate results for all files
check_result_consistency = False


try:
    if 'pydevconsole' in sys.argv[0]:
        raise Exception()
    ID = sys.argv[1]
    ID = str(int(ID) - 1)
    print('Processing ID:' + str(ID))
    ID = [int(ID)]

except:
    ID = np.arange(9)
    print('ID NOT PASSED')

print_figs = True
skip_refinement = False
backend_patch = 'local'
backend_refine = 'local'
n_processes = 24
base_folder = '/mnt/ceph/neuro/DataForPublications/DATA_PAPER_ELIFE/WEBSITE'
n_pixels_per_process = 4000
block_size = 5000
num_blocks_per_run = 20

# %%
global_params = {'SNR_lowest': 0.5,
                 'min_SNR': 2,  # minimum SNR when considering adding a new neuron
                 'gnb': 2,  # number of background components
                 'rval_thr': 0.85,  # spatial correlation threshold
                 'min_rval_thr_rejected': -1.1,
                 'min_cnn_thresh': 0.99,
                 'max_classifier_probability_rejected': 0.1,
                 'p': 1,
                 'max_fitness_delta_accepted': -20,
                 'update_background_components': True,
                 # whether to update the background components in the spatial phase
                 'low_rank_background': True,
                 # whether to update the using a low rank approximation. In the False case all the nonzero elements of the background components are updated using hals
                 # (to be used with one background per patch)
                 'only_init_patch': True,
                 'init_method': 'greedy_roi',
                 'filter_after_patch': False,
                 'tsub': 2,
                 'ssub': 2
                 }
# %%
params_movies = []
# %%
params_movie = {'fname': 'N.03.00.t/Yr_d1_498_d2_467_d3_1_order_C_frames_2250_.mmap',
                'gtname': 'N.03.00.t/joined_consensus_active_regions.npy',
                'merge_thresh': 0.8,  # merging threshold, max correlation allow
                'rf': 25,  # half-size of the patches in pixels. rf=25, patches are 50x50    20
                'stride_cnmf': 10,  # amounpl.it of overlap between the patches in pixels
                'K': 4,  # number of components per patch
                'gSig': [8, 8],  # expected half size of neurons
                'crop_pix': 0,
                'fr': 7,
                'decay_time': 0.4,
                }
params_movies.append(params_movie.copy())
# %%
params_movie = {'fname': 'N.04.00.t/Yr_d1_512_d2_512_d3_1_order_C_frames_3000_.mmap',
                'gtname': 'N.04.00.t/joined_consensus_active_regions.npy',
                'merge_thresh': 0.8,  # merging threshold, max correlation allow
                'rf': 20,  # half-size of the patches in pixels. rf=25, patches are 50x50    20
                'stride_cnmf': 10,  # amounpl.it of overlap between the patches in pixels
                'K': 5,  # number of components per patch
                'gSig': [5, 5],  # expected half size of neurons
                'crop_pix': 0,
                'fr': 8,
                'decay_time': 1.4,  # rough length of a transient
                }
params_movies.append(params_movie.copy())

# %% yuste
params_movie = {'fname': 'YST/Yr_d1_200_d2_256_d3_1_order_C_frames_3000_.mmap',
                'gtname': 'YST/joined_consensus_active_regions.npy',
                'merge_thresh': 0.8,  # merging threshold, max correlation allow
                'rf': 15,  # half-size of the patches in pixels. rf=25, patches are 50x50    20
                'stride_cnmf': 10,  # amounpl.it of overlap between the patches in pixels
                'K': 10,  # number of components per patch
                'gSig': [5, 5],  # expected half size of neurons
                'fr': 10,
                'decay_time': 0.75,
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
                'decay_time': 0.7,
                'fr': 8,
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
                'decay_time': 0.4,
                'fr': 8,
                'crop_pix': 2,
                }
params_movies.append(params_movie.copy())
# %% neurofinder 02.00
params_movie = {
    'fname': 'N.02.00/Yr_d1_512_d2_512_d3_1_order_C_frames_8000_.mmap',
    'gtname': 'N.02.00/joined_consensus_active_regions.npy',
    'merge_thresh': 0.8,  # merging threshold, max correlation allow
    'rf': 20,  # half-size of the patches in pixels. rf=25, patches are 50x50    20
    'stride_cnmf': 10,  # amounpl.it of overlap between the patches in pixels
    'K': 6,  # number of components per patch
    'gSig': [5, 5],  # expected half size of neurons
    'fr': 30,  # imaging rate in Hz
    'crop_pix': 10,
    'decay_time': 0.3,
}
params_movies.append(params_movie.copy())
# %% Sue Ann k53
params_movie = {
    'fname': 'K53/Yr_d1_512_d2_512_d3_1_order_C_frames_116043_.mmap',
    'gtname': 'K53/joined_consensus_active_regions.npy',
    'merge_thresh': 0.8,  # merging threshold, max correlation allow
    'rf': 20,  # half-size of the patches in pixels. rf=25, patches are 50x50    20
    'stride_cnmf': 10,  # amounpl.it of overlap between the patches in pixels
    'K': 10,  # number of components per patch
    'gSig': [6, 6],  # expected half size of neurons
    'fr': 30,
    'decay_time': 0.3,
    'crop_pix': 2,
}
params_movies.append(params_movie.copy())
# %% J115
params_movie = {
    'fname': 'J115/Yr_d1_463_d2_472_d3_1_order_C_frames_90000_.mmap',
    'gtname': 'J115/joined_consensus_active_regions.npy',
    'merge_thresh': 0.8,  # merging threshold, max correlation allow
    'rf': 20,  # half-size of the patches in pixels. rf=25, patches are 50x50    20
    'stride_cnmf': 10,  # amounpl.it of overlap between the patches in pixels
    'K': 8,  # number of components per patch
    'gSig': [7, 7],  # expected half size of neurons
    'fr': 30,
    'decay_time': 0.4,
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
    'crop_pix': 2,
}
params_movies.append(params_movie.copy())


#%%

# %%
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

    #%%
    fname_new = os.path.join(base_folder, params_movie['fname'])
    print(fname_new)
    if not os.path.exists(fname_new): # in case we need to reload from zip files
        try:
            cm.stop_server()
            dview.terminate()
        except:
            print('No clusters to stop')

        c, dview, _ = setup_cluster(
            backend=backend_patch, n_processes=8, single_thread=False)
        fname_zip = os.path.join(base_folder, params_movie['fname'].split('/')[0], 'images', 'images_' + params_movie['fname'].split('/')[0] + '.zip')
        mov_names = glob.glob(os.path.join(base_folder, params_movie['fname'].split('/')[0], 'images', '*.tif'))
        if len(mov_names) > 0:
            mov_names = sorted(mov_names, key=lambda x: int(x.split('_')[-1][:-4]))
        else:
            mov_names = from_zipfiles_to_movie_lists(fname_zip)

        # add_to_mov = cm.load(mov_names[0]).min()
        fname_zip = cm.save_memmap(mov_names, dview=dview, order='C', add_to_movie=0)
        shutil.move(fname_zip,fname_new)  # we get it from the images subfolder

    try:
        cm.stop_server()
        dview.terminate()
    except:
        print('No clusters to stop')

    c, dview, n_processes = setup_cluster(
        backend=backend_patch, n_processes=n_processes, single_thread=False)



            # %% LOAD MEMMAP FILE
    Yr, dims, T = cm.load_memmap(fname_new)
    d1, d2 = dims
    images = np.reshape(Yr.T, [T] + list(dims), order='F')
    # TODO: needinfo
    Y = np.reshape(Yr, dims + (T,), order='F')
    m_images = cm.movie(images)
    if plot_on:
        if m_images.shape[0] < 5000:
            Cn = m_images.local_correlations(swap_dim=False, frames_per_chunk=1500)
            Cn[np.isnan(Cn)] = 0
        else:
            Cn = np.array(cm.load(
                ('/'.join(
                    params_movie['gtname'].split('/')[:-2] + ['projections', 'correlation_image.tif'])))).squeeze()

    check_nan = False

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
                   'num_blocks_per_run_spat': num_blocks_per_run,
                   'num_blocks_per_run_temp': num_blocks_per_run,
                   'n_pixels_per_process': n_pixels_per_process,
                   'ssub': global_params['ssub'],
                   'tsub': global_params['tsub'],
                   'thr_method': 'nrg'
                   }

    init_method = global_params['init_method']

    opts = params.CNMFParams(params_dict=params_dict)
    if reload:
        cnm2 = load_CNMF(fname_new[:-5] + '_cnmf_perf_web.hdf5')
        reproduce_images_website = False
        if reproduce_images_website:
            cnm2.estimates.evaluate_components(images,params=cnm2.params, dview=dview)
            pl.figure(figsize=(10,7))
            good_snr = np.where(cnm2.estimates.SNR_comp>5)[0]
            examples = np.argsort(cnm2.estimates.cnn_preds)[-100:]
            examples = np.intersect1d(examples, good_snr)[-5:]
            for count, ex in enumerate(examples):
                pl.subplot(6,2,2*count+1)
                img = cnm2.estimates.A[:, ex].toarray().reshape(cnm2.estimates.dims, order='F')
                cm_ = np.array(scipy.ndimage.measurements.center_of_mass(img)).astype(int)
                pl.axis('off')
                pl.imshow(img[cm_[0]-15:cm_[0]+15,cm_[1]-15:cm_[1]+15])
                pl.subplot(6, 2, 2 * count + 2)
                pl.plot(cnm2.estimates.C[ex])
                pl.axis('off')
            pl.subplot(7, 1, 7)
            pl.plot(images.mean(axis=(1,2)))
            pl.axis('off')
            pl.title('Average frame wise movie')
            pl.savefig(os.path.join(os.path.split(fname_new)[0],'projections','traces_and_masks.jpg'))
            pl.savefig(os.path.join(os.path.split(fname_new)[0], 'projections', 'traces_and_masks.png'))
            pl.close()
            continue
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
                                  'n_pixels_per_process': n_pixels_per_process, 'dview': dview});
        # %%
        t1 = time.time()
        cnm2 = cnm.refit(images, dview=dview)
        t_refit = time.time() - t1
        if save_on:
            cnm2.save(fname_new[:-5] + '_cnmf_perf_web.hdf5')



    # %%
    if plot_on:
        cnm2.estimates.plot_contours(img=Cn)
    # %% check quality of components and eliminate low quality
    cnm2.params.set('quality', {'SNR_lowest': global_params['SNR_lowest'],
                                'min_SNR': global_params['min_SNR'],
                                'rval_thr': global_params['rval_thr'],
                                'rval_lowest': global_params['min_rval_thr_rejected'],
                                'use_cnn': True,
                                'min_cnn_thr': global_params['min_cnn_thresh'],
                                'cnn_lowest': global_params['max_classifier_probability_rejected'],
                                'gSig_range': None})

    cnm2.params.set('data',{'decay_time':params_movie['decay_time']})

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
    with np.load(gt_file, encoding='latin1', allow_pickle=True) as ld:
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
    #idx_components_gt = nrn_size[nrn_dup]
    gt_estimate.select_components(use_object=True)
    print(gt_estimate.A_thr.shape)
    # %% prepare CNMF maks
    cnm2.estimates.threshold_spatial_components(maxthr=0.2, dview=dview)
    cnm2.estimates.remove_small_large_neurons(min_size_neuro, max_size_neuro)
    cnm2.estimates.remove_duplicates(r_values=None, dist_thr=0.1, min_dist=10, thresh_subset=0.6)
    cnm2.estimates.select_components(use_object=True)
    print('Num neurons to match:' + str(cnm2.estimates.A.shape))
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
        cnm2.save(fname_new[:-5] + '_cnmf_perf_web_after_analysis.hdf5')

    performance_cons_off['fname_new'] = fname_new
    performance_tmp = performance_cons_off.copy()
    performance_tmp['tp_gt'] = tp_gt
    performance_tmp['tp_comp'] = tp_comp
    performance_tmp['fn_gt'] = fn_gt
    performance_tmp['fp_comp'] = fp_comp
    performance_tmp['predictionsCNN'] = cnm2.estimates.cnn_preds
    performance_tmp['A'] = cnm2.estimates.A
    performance_tmp['C'] = cnm2.estimates.C
    performance_tmp['YrA'] = cnm2.estimates.YrA
    performance_tmp['SNR_comp'] = cnm2.estimates.SNR_comp
    performance_tmp['Cn'] = Cn_orig
    performance_tmp['params'] = params_movie
    performance_tmp['dims'] = dims
    performance_tmp['CCs'] = [scipy.stats.pearsonr(a, b)[0] for a, b in
                    zip(gt_estimate.C[tp_gt], cnm2.estimates.C[tp_comp])]
    if not reload:
        performance_tmp['t_patch'] = t_patch
        performance_tmp['t_refit'] = t_refit
        performance_tmp['t_eva'] = t_eva_comps

    with np.load(os.path.join(base_folder,fname_new.split('/')[-2],'gt_eval.npz'), allow_pickle=True) as ld:
        print(ld.keys())
        performance_tmp.update(ld)

    performance_tmp['idx_components_gt'] = gt_estimate.idx_components

    ALL_CCs.append([scipy.stats.pearsonr(a, b)[0] for a, b in
                    zip(gt_estimate.C[tp_gt], cnm2.estimates.C[tp_comp])])

    performance_tmp['ALL_CCs'] = ALL_CCs

    all_results[fname_new.split('/')[-2]] = performance_tmp

    if save_on:
        print('SAVING...' + fname_new[:-5] + '_perf_web_gsig.npz')
        np.savez(fname_new[:-5] + '_perf_web_gsig.npz', all_results=performance_tmp)

    #
if save_all:
    # here eventually save when in a loop
    np.savez(os.path.join(base_folder,'all_res_web.npz'), all_results=all_results)

#%%
join_npz_files_parallel = False
if join_npz_files_parallel:
    #%%
    all_results = dict()

    for params_movie in params_movies:
        npzfile = os.path.join(base_folder,params_movie['fname'][:-5] + '_perf_web_gsig.npz')
        print(npzfile)
        if os.path.exists(npzfile):
            with np.load(npzfile, allow_pickle=True) as ld:
                all_results[params_movie['fname'].split('/')[-2]] = ld['all_results'][()]
        else:
            print("*** NOT EXIST ***" + npzfile)
    np.savez(os.path.join(base_folder,'all_res_web.npz'), all_results=all_results)
