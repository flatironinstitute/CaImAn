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

reload = False
plot_on = False
save_on = True  # set to true to recreate

try:
    if 'pydevconsole' in sys.argv[0]:
        raise Exception()
    ID = sys.argv[1]
    ID = str(np.int(ID))
    print('Processing ID:' + str(ID))
    ID_nproc = np.int(ID)
    ID = range(0, 4)
except:
    # ID = range(0,1)
    ID = range(0, 1)
    ID_nproc = 0
    print('ID NOT PASSED')



print_figs = True
skip_refinement = False
backend_patch = 'multiprocessing'
backend_refine = 'multiprocessing'
n_proc_list = [24, 12, 6, 3, 2, 1]
n_processes = n_proc_list[ID_nproc]
print('n_processes:' + str(n_processes))
n_pixels_per_process = 10000
block_size = 10000
num_blocks_per_run = 12
n_pixels_per_process = 4000
base_folder = '/mnt/ceph/neuro/DataForPublications/DATA_PAPER_ELIFE/'

#%%
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
# %% neurofinder 02.00
params_movie = {
   'fname': '/opt/local/Data/labeling/neurofinder.02.00/Yr_d1_512_d2_512_d3_1_order_C_frames_8000_.mmap',
     # 'fname': 'N.02.00/Yr_d1_512_d2_512_d3_1_order_C_frames_8000_.mmap',
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
    'decay_time': 0.75,
}
params_movies.append(params_movie.copy())
# %% J123
params_movie = {
    'fname': '/opt/local/Data/labeling/J123_2015-11-20_L01_0/Yr_d1_458_d2_477_d3_1_order_C_frames_41000_.mmap',
    # 'fname': 'J123/Yr_d1_458_d2_477_d3_1_order_C_frames_41000_.mmap',
    'gtname': 'J123/joined_consensus_active_regions.npy',
    # order of the autoregressive system
    'merge_thresh': 0.8,  # merging threshold, max correlation allow
    'rf': 40,  # half-size of the patches in pixels. rf=25, patches are 50x50    20
    'stride_cnmf': 20,  # amounpl.it of overlap between the patches in pixels
    'K': 11,  # number of components per patch
    'gSig': [8, 8],  # expected half size of neurons
    'decay_time': 0.75,
    'fr': 30,
    'n_chunks': 10,
    'swap_dim': False,
    'crop_pix': 2,
}
params_movies.append(params_movie.copy())

#%% J115
params_movie = {
    'fname': '/opt/local/Data/labeling/J115_2015-12-09_L01_ELS/Yr_d1_463_d2_472_d3_1_order_C_frames_90000_.mmap',
    #'fname': 'J115/Yr_d1_463_d2_472_d3_1_order_C_frames_90000_.mmap',
    'gtname': 'J115/joined_consensus_active_regions.npy',
    # order of the autoregressive system
    'merge_thresh': 0.8,  # merging threshold, max correlation allow
    'rf': 20,  # half-size of the patches in pixels. rf=25, patches are 50x50    20
    'stride_cnmf': 10,  # amounpl.it of overlap between the patches in pixels
    'K': 8,  # number of components per patch
    'gSig': [7, 7],  # expected half size of neurons
    'fr': 30,
    'decay_time': 0.8,
    'n_chunks': 10,
    'swap_dim': False,
    'crop_pix': 2,

}
params_movies.append(params_movie.copy())

# %% Sue Ann k53
params_movie = {
    'fname': '/opt/local/Data/labeling/k53_20160530/Yr_d1_512_d2_512_d3_1_order_C_frames_116043_.mmap',
    # 'fname': 'K53/Yr_d1_512_d2_512_d3_1_order_C_frames_116043_.mmap',
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




# %%
all_perfs = []
all_rvalues = []
all_comp_SNR_raw = []
all_comp_SNR_delta = []
all_predictions = []
all_labels = []
all_results = dict()
ALL_CCs = []
all_timings = dict()

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
                   'num_blocks_per_run_temp': num_blocks_per_run,
                    'block_size_spat': block_size,
                    'num_blocks_per_run_spat': num_blocks_per_run,
                   'n_pixels_per_process': n_pixels_per_process,
                   'ssub': 2,
                   'tsub': 2,
                   'p_tsub': 1,
                   'p_ssub': 1,
                   'thr_method': 'nrg'
                   }

    init_method = global_params['init_method']

    opts = params.CNMFParams(params_dict=params_dict)
    if reload:
        all_timings[fname_new.split('/')[-2]] = []
        for ID_nproc in range(len(n_proc_list)):
            print(fname_new[:-5] + '_perf_timing_proc_' + str(ID_nproc+1) + '_.npz')
            with np.load(fname_new[:-5] + '_perf_timing_proc_' + str(ID_nproc+1) + '_.npz') as ld:
                tmp_timings = dict()
                tmp_timings['t_eva_comps'] = ld['t_eva_comps']
                tmp_timings['t_patch'] = ld['t_patch']
                tmp_timings['t_refit'] = ld['t_refit']
                tmp_timings['n_proc'] = n_proc_list[ID_nproc]
                all_timings[fname_new.split('/')[-2]].append(tmp_timings)



    else:
        # %% Extract spatial and temporal components on patches
        t1 = time.time()
        print('Starting CNMF')
        cnm = cnmf.CNMF(n_processes, params=opts, dview=dview)
        cnm = cnm.fit(images)
        t_patch = time.time() - t1
        print(str(ID_nproc)+' ***** time_patch:' + str(t_patch))
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
        print(str(ID_nproc)+' ***** time_refit:' + str(t_refit))


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
        print(str(ID_nproc)+ ' ***** time_eva:' + str(t_eva_comps))
        print((len(cnm2.estimates.C)))

        if save_on :
            np.savez(fname_new[:-5] + '_perf_timing_proc_' + str(n_processes) + '_.npz', t_eva_comps=t_eva_comps, t_patch=t_patch, t_refit=t_refit)


#%% CREATE OVERALL DICT
if False:
    results_timing = []
    for n_processes in n_proc_list:
        for params_movie in np.array(params_movies)[ID]:

            #    params_movie['gnb'] = 3
            params_display = {
                'downsample_ratio': .2,
                'thr_plot': 0.8
            }

            fname_new = os.path.join(base_folder, params_movie['fname'])

            print(fname_new)

            with np.load(fname_new[:-5] + '_perf_timing_proc_' + str(n_processes) + '_.npz') as ld:
                print(ld.keys())
                results_timing.append([fname_new.split('/')[-2], (n_processes),  float(ld['t_patch']), float(ld['t_refit']), float(ld['t_eva_comps'])])

    #%%

    df = DataFrame(results_timing)
    df.columns= ['name','processes','t_patch','t_refit','t_eva']
    #np.savez('/mnt/ceph/neuro/DataForPublications/DATA_PAPER_ELIFE/TIMING/all_timings_single_machine.npz', df.to_dict())
    #%%
    res = df.groupby(['name'])
    df1 = df.set_index('processes')

    df1.groupby('name').plot( subplots=False)
else: # unfortunately the timing data was not saved because of an issue with the script, os I report here the results. Should be run again to
    #%%
    plt.rcParams['pdf.fonttype'] = 42
    font = {'family': 'Arial',
            'weight': 'regular',
            'size': 20}
    from pandas import DataFrame
    import numpy as np
    all_timings = DataFrame([
        {'name': 'N.02.00', 'n_processes': 24, 'time_patch': 83.30931520462036, 'time_refit': 150.8432295322418, 'time_eval': 8.844662427902222},
        {'name': 'J123', 'n_processes': 24, 'time_patch': 274.6004159450531, 'time_refit': 180.42933320999146, 'time_eval': 41.29946994781494},
        {'name': 'J115', 'n_processes': 24, 'time_patch': 858.1903641223907, 'time_refit': 881.201664686203, 'time_eval': 104.10698699951172},
        {'name': 'K53', 'n_processes': 24, 'time_patch': 1797.6236493587494, 'time_refit': 2088.205850839615, 'time_eval': 214.41207218170166},

        {'name': 'N.02.00', 'n_processes': 12, 'time_patch': 93.71227407455444, 'time_refit': 171.99205470085144, 'time_eval': 9.873874425888062},
        {'name': 'J123', 'n_processes': 12, 'time_patch': 297.5868630409241, 'time_refit': 209.39823293685913, 'time_eval': 37.03397536277771},
        {'name': 'J115', 'n_processes': 12, 'time_patch': 910.1763546466827, 'time_refit': 867.1253700256348, 'time_eval': 91.3763816356659},
        {'name': 'K53', 'n_processes': 12, 'time_patch': 2131.146340608597, 'time_refit': 1840.5280983448029, 'time_eval': 226.46999287605286},

        {'name': 'N.02.00', 'n_processes': 6, 'time_patch': 119.35262084007263, 'time_refit': 237.7939488887787, 'time_eval': 13.037508964538574},
        {'name': 'J123', 'n_processes': 6, 'time_patch': 474.36822843551636, 'time_refit': 291.2786304950714, 'time_eval': 48.68373775482178},
        {'name': 'J115', 'n_processes': 6, 'time_patch': 1234.7518901824951, 'time_refit': 988.6932106018066, 'time_eval': 168.36487412452698},
        {'name': 'K53', 'n_processes': 6, 'time_patch': 2756.9168894290924, 'time_refit': 2216.247531414032, 'time_eval': 399.46128821372986},

        {'name': 'N.02.00', 'n_processes': 3, 'time_patch': 177.89352083206177, 'time_refit': 364.8443193435669, 'time_eval': 19.78028917312622},
        {'name': 'J123', 'n_processes': 3, 'time_patch': 855.328578710556, 'time_refit': 540.6532158851624, 'time_eval': 81.68031716346741},
        {'name': 'J115', 'n_processes': 3, 'time_patch': 2253.3668415546417, 'time_refit': 1791.0478348731995, 'time_eval': 333.98589515686035},
        {'name': 'K53', 'n_processes': 3, 'time_patch': 4108.806590795517, 'time_refit': 3596.7499675750732, 'time_eval': 733.7702996730804},

        {'name': 'N.02.00', 'n_processes': 2, 'time_patch': 228.71087837219238, 'time_refit': 495.9476628303528, 'time_eval': 27.659457445144653},
        {'name': 'J123', 'n_processes': 2, 'time_patch': 1191.312772512436, 'time_refit': 712.5731451511383, 'time_eval': 109.54649567604065},
        {'name': 'J115', 'n_processes': 2, 'time_patch': 2983.893436193466, 'time_refit': 2210.7389879226685, 'time_eval': 471.05265831947327},
        {'name': 'K53', 'n_processes': 2, 'time_patch': 5008.163422584534, 'time_refit': np.nan, 'time_eval': np.nan},

        {'name': 'N.02.00', 'n_processes': 1, 'time_patch': 389.0271465778351, 'time_refit': 863.1781265735626, 'time_eval': 40.629820346832275},
        {'name': 'J123', 'n_processes': 1, 'time_patch': 2390.3756499290466, 'time_refit': 1399.526368379593, 'time_eval': 221.93205952644348},
        {'name': 'J115', 'n_processes': 1, 'time_patch': 5922.906694412231, 'time_refit': 4353.50389456749, 'time_eval': 919.8520407676697},
        {'name': 'K53', 'n_processes': 1, 'time_patch': np.nan, 'time_refit': np.nan, 'time_eval': np.nan},
    ])
    pl.figure()
    pl.subplot(2,4,1)
    all_timings['total'] = all_timings['time_patch']+ all_timings['time_refit']+ all_timings['time_eval']

    all_timings = all_timings[all_timings['name']!='K53']
    all_timings.index = all_timings['n_processes']
    all_timings.groupby(by=['name'])['time_patch'].plot(logx=True, logy=True, label='name')
    # pl.gca().invert_xaxis()
    pl.title('initialization')
    pl.subplot(2, 4, 2)
    all_timings.groupby(by=['name'])['time_refit'].plot(logx=True, logy=True, label='name')
    # pl.gca().invert_xaxis()
    pl.ylabel('time(s)')
    pl.title('refinement')
    pl.subplot(2, 4, 3)
    all_timings.groupby(by=['name'])['time_eval'].plot(logx=True, logy=True, label='name')
    # pl.gca().invert_xaxis()
    pl.title('quality evaluation')
    pl.subplot(2, 4, 4)
    all_timings.groupby(by=['name'])['total'].plot(x='n_processes', logx=True, logy=True, label='name')
    # pl.gca().invert_xaxis()
    pl.title('total')
    pl.tight_layout()


    all_timings_norm = all_timings.copy()
    fact = all_timings_norm[all_timings_norm['n_processes'] == 1]

    for indx, row in fact.iterrows():
        print(row['name'])
        for nm in ['time_patch','time_refit','time_eval','total']:
            all_timings_norm[nm][all_timings_norm['name'] == row['name']] /= row[nm]
            all_timings_norm[nm][all_timings_norm['name'] == row['name']] = 1/all_timings_norm[nm][all_timings_norm['name'] == row['name']]
    all_timings = all_timings_norm
    logx = False
    logy = False
    pl.subplot(2, 4, 5)
    all_timings.index = all_timings['n_processes']
    all_timings.groupby(by=['name'])['time_patch'].plot(logx=logx, logy=logy)
    # pl.gca().invert_xaxis()
    pl.title('initialization')
    pl.subplot(2, 4, 6)
    all_timings.groupby(by=['name'])['time_refit'].plot(logx=logx, logy=logy,)
    # pl.gca().invert_xaxis()
    pl.ylabel('time(s)')
    pl.title('refinement')
    pl.subplot(2, 4, 7)
    all_timings.groupby(by=['name'])['time_eval'].plot(logx=logx, logy=logy)
    # pl.gca().invert_xaxis()
    pl.title('quality evaluation')
    pl.subplot(2, 4, 8)
    all_timings.groupby(by=['name'])['total'].plot(x='n_processes',logx=logx, logy=logy, legend='name')
    # pl.gca().invert_xaxis()
    pl.title('total')
    pl.tight_layout()
    #%%
    # %%FIGURE 8 a and b time performances (results have been manually annotated on an excel spreadsheet and reported here below)
    import pylab as plt

    pl.figure("Figure 8a and 8b",figsize=(20,4))
    import numpy as np

    plt.rcParams['pdf.fonttype'] = 42
    font = {'family': 'Arial',
            'weight': 'regular',
            'size': 20}

    t_mmap = dict()
    t_patch = dict()
    t_refine = dict()
    t_filter_comps = dict()

    size = np.log10(np.array([8.4, 121.7, 78.7, 35.8]) * 1000)
    components = np.array([1099, 1541, 1013, 398])
    components = np.array([1099, 1541, 1013, 398])

    t_mmap['cluster'] = np.array([109, 561, 378, 135])
    t_patch['cluster'] = np.array([92, 1063, 469, 142])
    t_refine['cluster'] = np.array([256, 1065, 675, 265])
    t_filter_comps['cluster'] = np.array([11, 143, 77, 30])

    # t_mmap['desktop'] = np.array([25, 41, 11, 41, 135, 23, 690, 510, 176, 163])
    # t_patch['desktop'] = np.array([21, 43, 16, 48, 85, 45, 2150, 949, 316, 475])
    # t_refine['desktop'] = np.array([105, 205, 43, 279, 216, 254, 1749, 837, 237, 493])
    # t_filter_comps['desktop'] = np.array([3, 5, 2, 5, 9, 7, 246, 81, 36, 38])
    t_mmap['desktop'] = np.array([135, 690, 510, 176])
    t_patch['desktop'] = np.array([83.309, 1797, 858, 274])
    t_refine['desktop'] = np.array([150.8, 2088.20, 881.20, 180.4])
    t_filter_comps['desktop'] = np.array([8.844, 214.41, 104.1, 41.299])

    t_mmap['laptop'] = np.array([144, 731, 287, 125])
    t_patch['laptop'] = np.array([177.893, 4108.8, 2253.366, 855.3])
    t_refine['laptop'] = np.array([364.8, 3596.7, 1791.04, 540.6])
    t_filter_comps['laptop'] = np.array([19.78, 733.77, 33.985, 81.68])



    # these can be read from the final portion of of the script output
    t_mmap['online'] = np.array([0, 0, 0, 0])
    t_patch['online'] = np.array([0, 0, 0, 0])
    t_refine['online'] = np.array([0, 0, 0, 0])
    t_filter_comps['online'] = np.array([909.07959843,  9996.35542846,  7175.75227594,  1371.42159343])

    pl.subplot(1, 4, 1)
    for key in ['cluster', 'desktop', 'laptop', 'online']:
        np.log10(t_mmap[key] + t_patch[key] + t_refine[key] + t_filter_comps[key])
        plt.scatter((size), np.log10((t_mmap[key] + t_patch[key] + t_refine[key] + t_filter_comps[key])),
                    s=np.array(components) / 10)
        plt.xlabel('size (GB)')
        plt.ylabel('time (minutes)')






    plt.plot((np.sort(size)), np.log10((np.sort(10 ** size)) / 31.45), '--.k')

    axx = pl.gca()
    axx.locator_params(nbins=7)
    axx.set_yticklabels([str(int((10**ss)/60))[:5] for ss in axx.get_yticks()])
    axx.set_xticklabels([str(int((10 ** ss) / 1000))[:5] for ss in axx.get_xticks()])
    plt.legend(
        ['acquisition-time', 'cluster (112 CPUs)', 'workstation (24 CPUs)', 'workstation (3 CPUs)', 'online (6 CPUs)'])
    pl.title('Total execution time')
    # pl.xlim([3.8, 5.2])
    # pl.ylim([2.35, 4.2])

    counter = 2
    for key in ['cluster', 'desktop', 'online']:
        pl.subplot(1, 4, counter)
        counter += 1
        if counter == 3:
            pl.title('Time per phase (cluster)')

        elif counter == 4:
            pl.title('Time per phase (workstation)')
        else:
            pl.title('Time per phase (online)')

        plt.bar((size), (t_mmap[key]), width=0.12, bottom=0)
        plt.bar((size), (t_patch[key]), width=0.12, bottom=(t_mmap[key]))
        plt.bar((size), (t_refine[key]), width=0.12, bottom=(t_mmap[key] + t_patch[key]))
        plt.bar((size), (t_filter_comps[key]), width=0.12, bottom=(t_mmap[key] + t_patch[key] + t_refine[key]))
        if counter == 5:
            plt.legend(['Initialization', 'track activity', 'update shapes'])
        else:
            plt.legend(['mem mapping', 'patch init', 'refine sol', 'quality  filter', 'acquisition time'])

        plt.plot((np.sort(size)), (10 ** np.sort(size)) / 31.45, '--k')
        pl.xlim([3.6, 5.2])
        axx = pl.gca()
        axx.locator_params(nbins=7)
        axx.set_yticklabels([str(int((ss) / 60))[:5] for ss in axx.get_yticks()])
        axx.set_xticklabels([str(int((10 ** ss) / 1000))[:5] for ss in axx.get_xticks()])
        plt.xlabel('size (GB)')
        plt.ylabel('time (minutes)')


