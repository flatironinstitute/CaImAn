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
import pylab as pl
import scipy
from caiman.utils.visualization import plot_contours

# %%
pl.rcParams['pdf.fonttype'] = 42
font = {'family': 'Arial',
        'weight': 'regular',
        'size': 20}
pl.rc('font', **font)


base_folder = '/mnt/ceph/neuro/DataForPublications/DATA_PAPER_ELIFE/WEBSITE/'

# %% RELOAD ALL THE RESULTS INSTEAD OF REGENERATING THEM
with np.load(os.path.join(base_folder, 'all_res_web.npz')) as ld:
    all_results = ld['all_results'][()]
with np.load(os.path.join(base_folder, 'all_res_online_web_bk.npz')) as ld:
    all_results_online = ld['all_results'][()]

pl.rcParams['pdf.fonttype'] = 42
# %% FIGURE  timing to create a mmap file (Figures 8 a and b "mem mapping")
generate_mmap = False
if generate_mmap:
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

# %% OLD ONE!!!

# %% RELOAD ALL THE RESULTS INSTEAD OF REGENERATING THEM
base_folder = '/mnt/ceph/neuro/DataForPublications/DATA_PAPER_ELIFE/'
with np.load(os.path.join(base_folder, 'all_res_sept_2018.npz')) as ld:
    all_results_ = ld['all_results'][()]

# %%
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
    predictionsCNN = all_results[name_file]['predictionsCNN']
    idx_components_gt = all_results[name_file]['idx_components_gt']
    tp_comp = all_results[name_file]['tp_comp']
    fp_comp = all_results[name_file]['fp_comp']
    tp_gt = all_results[name_file]['tp_gt']
    fn_gt = all_results[name_file]['fn_gt']
    A = all_results[name_file]['A']
    A_gt = all_results[name_file]['A_gt'][()]
    C = all_results[name_file]['C']
    C_gt = all_results[name_file]['C_gt']
    fname_new = all_results[name_file]['params']['fname']
    dims = all_results[name_file]['dims']
    Cn = all_results[name_file]['Cn']
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
    t_mmap['online'] = np.array(
        [85.23642993, 34.09690762, 18.95380235, 10.85061121])
    t_patch['online'] = np.array(
        [267.06141853, 3297.18628764, 2573.04009032, 578.88080835])
    t_refine['online'] = np.array(
        [89.29527831, 1689.06246471, 1282.98535109, 61.20671248])
    t_filter_comps['online'] = np.array([0, 0, 0, 0])

    pl.subplot(1, 4, 1)
    for key in ['cluster', 'desktop', 'laptop', 'online']:
        np.log10(t_mmap[key] + t_patch[key] + t_refine[key] + t_filter_comps[key])
        plt.scatter((size), np.log10(t_mmap[key] + t_patch[key] + t_refine[key] + t_filter_comps[key]),
                    s=np.array(components) / 10)
        plt.xlabel('size log_10(MB)')
        plt.ylabel('time log_10(s)')

    plt.plot((np.sort(size)), np.log10((np.sort(10 ** size)) / 31.45), '--.k')
    plt.legend(
        ['acquisition-time', 'cluster (112 CPUs)', 'workstation (24 CPUs)', 'workstation (3 CPUs)', 'online (6 CPUs)'])
    pl.title('Total execution time')
    pl.xlim([3.8, 5.2])
    pl.ylim([2.35, 4.2])

    counter = 2
    for key in ['cluster', 'desktop', 'online']:
        pl.subplot(1, 4, counter)
        counter += 1
        if counter == 3:
            pl.title('Time per phase (cluster)')
            plt.ylabel('time (10^3 s)')
        elif counter == 4:
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
