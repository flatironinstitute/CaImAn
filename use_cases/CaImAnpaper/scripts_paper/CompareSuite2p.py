import caiman as cm
import numpy as np
import os
import time
import pylab as pl
import scipy
import h5py
import sys
from caiman.utils.visualization import plot_contours
from caiman.source_extraction.cnmf import cnmf as cnmf

from caiman.source_extraction.cnmf.estimates import Estimates, compare_components
from caiman.cluster import setup_cluster
from caiman.source_extraction.cnmf import params as params
n_processes = 24
base_folder = '/mnt/ceph/neuro/DataForPublications/DATA_PAPER_ELIFE/'

# %%
params_movies = []
# %%
params_movie = {'fname': 'N.03.00.t/Yr_d1_498_d2_467_d3_1_order_C_frames_2250_.mmap',
                's2p_file': '/opt/local/Data/Example/DATA/F/N.03.00.t/2018-09-01/1/python_out.mat',
                'gSig': [8, 8],  # expected half size of neurons
                }
params_movies.append(params_movie.copy())
# %%
params_movie = {'fname': 'N.04.00.t/Yr_d1_512_d2_512_d3_1_order_C_frames_3000_.mmap',
                's2p_file': '/opt/local/Data/Example/DATA/F/N.04.00.t/2018-09-01/1/python_out.mat',
                'gSig': [5, 5],  # expected half size of neurons
                }
params_movies.append(params_movie.copy())

# %% yuste
params_movie = {'fname': 'YST/Yr_d1_200_d2_256_d3_1_order_C_frames_3000_.mmap',
                's2p_file': '/opt/local/Data/Example/DATA/F/YST/2018-09-01/1/python_out.mat',
                'gSig': [5, 5],  # expected half size of neurons
                }
params_movies.append(params_movie.copy())
# %% neurofinder 00.00
params_movie = {'fname': 'N.00.00/Yr_d1_512_d2_512_d3_1_order_C_frames_2936_.mmap',
                's2p_file': '/opt/local/Data/Example/DATA/F/N.00.00/2018-09-01/1/python_out.mat',
                'gSig': [6, 6],  # expected half size of neurons
                }
params_movies.append(params_movie.copy())
# %% neurofinder 01.01
params_movie = {'fname': 'N.01.01/Yr_d1_512_d2_512_d3_1_order_C_frames_1825_.mmap',
                's2p_file': '/opt/local/Data/Example/DATA/F/N.01.01/2018-09-01/1/python_out.mat',
                'gSig': [6, 6],  # expected half size of neurons
                }
params_movies.append(params_movie.copy())
# %% neurofinder 02.00
params_movie = {
    # 'fname': '/opt/local/Data/labeling/neurofinder.02.00/Yr_d1_512_d2_512_d3_1_order_C_frames_8000_.mmap',
    'fname': 'N.02.00/Yr_d1_512_d2_512_d3_1_order_C_frames_8000_.mmap',
    's2p_file': '/opt/local/Data/Example/DATA/F/N.02.00/2018-09-01/1/python_out.mat',
    'gSig': [5, 5],  # expected half size of neurons

}
params_movies.append(params_movie.copy())
#%% K53
params_movie = {
    # 'fname': '/opt/local/Data/labeling/neurofinder.02.00/Yr_d1_512_d2_512_d3_1_order_C_frames_8000_.mmap',
    'fname': 'K53/Yr_d1_512_d2_512_d3_1_order_C_frames_116043_.mmap',
    's2p_file': '/opt/local/Data/Example/DATA/F/K53/2018-09-01/1/python_out.mat',
    'gSig': [6, 6],  # expected half size of neurons

}
params_movies.append(params_movie.copy())
#%% J115
params_movie = {
    # 'fname': '/opt/local/Data/labeling/neurofinder.02.00/Yr_d1_512_d2_512_d3_1_order_C_frames_8000_.mmap',
    'fname': 'J115/Yr_d1_463_d2_472_d3_1_order_C_frames_90000_.mmap',
    's2p_file': '/opt/local/Data/Example/DATA/F/J115/2018-09-01/1/python_out.mat',
    'gSig': [7, 7],  # expected half size of neurons

}
params_movies.append(params_movie.copy())
#%% J123
params_movie = {
    # 'fname': '/opt/local/Data/labeling/neurofinder.02.00/Yr_d1_512_d2_512_d3_1_order_C_frames_8000_.mmap',
    'fname': 'J123/Yr_d1_458_d2_477_d3_1_order_C_frames_41000_.mmap',
    's2p_file': '/opt/local/Data/Example/DATA/F/J123/2018-09-01/1/python_out.mat',
    'gSig': [8, 8],  # expected half size of neurons

}
params_movies.append(params_movie.copy())

#%%
onlycell = True
#%%
for params_movie in np.array(params_movies)[range(6, 9)]:
    # %% start cluster
    #
    # TODO: show screenshot 10
    try:
        cm.stop_server()
        dview.terminate()
    except:
        print('No clusters to stop')

    c, dview, n_processes = setup_cluster(
        backend='multiprocessing', n_processes=20, single_thread=False)



    # %% prepare ground truth masks
    fname_new = os.path.join(base_folder, params_movie['fname'])
    gt_file = os.path.join(os.path.split(fname_new)[0], os.path.split(fname_new)[1][:-4] + 'match_masks.npz')
    gSig = params_movie['gSig']
    with np.load(gt_file, encoding='latin1') as ld:
        Cn_orig = ld['Cn']
        dims = (ld['d1'][()], ld['d2'][()])
        gt_estimate = Estimates(A=scipy.sparse.csc_matrix(ld['A_gt'][()]), b=ld['b_gt'], C=ld['C_gt'],
                                f=ld['f_gt'], R=ld['YrA_gt'], dims=(ld['d1'], ld['d2']))


    min_size_neuro = 3 * 2 * np.pi
    max_size_neuro = (2 * gSig[0]) ** 2 * np.pi
    gt_estimate.threshold_spatial_components(maxthr=0.2, dview=dview)
    gt_estimate.remove_small_large_neurons(min_size_neuro, max_size_neuro)
    _ = gt_estimate.remove_duplicates(predictions=None, r_values=None, dist_thr=0.1, min_dist=10,
                                      thresh_subset=0.6)
    print(gt_estimate.A_thr.shape)
    #%% create estimate from suite2p output
    suite2p_file = params_movie['s2p_file']
    with h5py.File(suite2p_file, 'r') as ld:
        masks = np.array(ld['masks']).transpose([2,1,0])
        traces = np.array(ld['traces']).T
        iscell = np.array(ld['iscell'])
#    ld = scipy.io.loadmat(suite2p_file)
        if onlycell:
            A_2p = scipy.sparse.csc_matrix(np.reshape(masks[:, :, np.where(iscell == 1)[0]], (np.prod(dims), -1)))
        else:
            A_2p = scipy.sparse.csc_matrix(np.reshape(masks[:, :, :], (np.prod(dims), -1)))

        s2p_estimate = Estimates(A=A_2p, b=None, C=traces,f=None, R=None, dims=dims)

        min_size_neuro = 3 * 2 * np.pi
        max_size_neuro = (2 *gSig[0]) ** 2 * np.pi
        s2p_estimate .threshold_spatial_components(maxthr=0.2, dview=dview)
        # s2p_estimate .remove_small_large_neurons(min_size_neuro, max_size_neuro)
        _ = s2p_estimate .remove_duplicates(predictions=None, r_values=None, dist_thr=0.1, min_dist=10,
                                          thresh_subset=0.6)
        print(s2p_estimate.A_thr.shape)


    #%%
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



    tp_gt, tp_comp, fn_gt, fp_comp, performance_suite2p = compare_components(gt_estimate, s2p_estimate,
                                                                                      Cn=Cn_orig, thresh_cost=.8,
                                                                                      min_dist=10,
                                                                                      print_assignment=False,
                                                                                      labels=['GT', 'Suite_2p'],
                                                                                      plot_results=plot_results)
    print(params_movie['fname'])
    print({a: b.astype(np.float16) for a, b in performance_suite2p.items()})
