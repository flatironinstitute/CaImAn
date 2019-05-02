# -*- coding: utf-8 -*-
"""
This script reproduces the results for all panels of Figure 4, except panel b
The script loads the saved results and uses them to plot contours and do the
SNR analysis. For running the CaImAn batch and CaImAn online algorithm and
obtain the results check the scripts:
/preprocessing_files/Preprocess_batch.py
/preprocessing_files/Preprocess_CaImAn_online.py

More info can be found in the companion paper
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
from caiman.utils.visualization import plot_contours

# %%
pl.rcParams['pdf.fonttype'] = 42
font = {'family': 'Arial',
        'weight': 'regular',
        'size': 20}
pl.rc('font', **font)

base_folder = '/mnt/ceph/neuro/DataForPublications/DATA_PAPER_ELIFE/WEBSITE/'

# %% function for performing the SNR analysis


def precision_snr(snr_gt, snr_gt_fn, snr_cnmf, snr_cnmf_fp, snr_thrs):
    all_results_fake = []
    all_results_OR = []
    all_results_AND = []
    for snr_thr in snr_thrs:
        snr_all_gt = np.array(list(snr_gt) + list(snr_gt_fn) + [0] * len(snr_cnmf_fp))
        snr_all_cnmf = np.array(list(snr_cnmf) + [0] * len(snr_gt_fn) + list(snr_cnmf_fp))

        ind_gt = np.where(snr_all_gt > snr_thr)[0]  # comps in gt above threshold
        ind_cnmf = np.where(snr_all_cnmf > snr_thr)[0]  # same for cnmf

        # precision: how many detected components above a given SNR are true
        prec = np.sum(snr_all_gt[ind_cnmf] > 0) / len(ind_cnmf)

        # recall: how many gt components with SNR above the threshold are detected
        rec = np.sum(snr_all_cnmf[ind_gt] > 0) / len(ind_gt)

        f1 = 2 * prec * rec / (prec + rec)

        results_fake = [prec, rec, f1]
        # f1 score with OR condition

        ind_OR = np.union1d(ind_gt, ind_cnmf)
        # indeces of components that are above threshold in either direction
        ind_gt_OR = np.where(snr_all_gt[ind_OR] > 0)[0]  # gt components
        ind_cnmf_OR = np.where(snr_all_cnmf[ind_OR] > 0)[0]  # cnmf components
        prec_OR = np.sum(snr_all_gt[ind_OR][ind_cnmf_OR] > 0) / len(ind_cnmf_OR)
        rec_OR = np.sum(snr_all_cnmf[ind_OR][ind_gt_OR] > 0) / len(ind_gt_OR)
        f1_OR = 2 * prec_OR * rec_OR / (prec_OR + rec_OR)

        results_OR = [prec_OR, rec_OR, f1_OR]

        # f1 score with AND condition

        ind_AND = np.intersect1d(ind_gt, ind_cnmf)
        ind_fp = np.intersect1d(ind_cnmf, np.where(snr_all_gt == 0)[0])
        ind_fn = np.intersect1d(ind_gt, np.where(snr_all_cnmf == 0)[0])

        prec_AND = len(ind_AND) / (len(ind_AND) + len(ind_fp))
        rec_AND = len(ind_AND) / (len(ind_AND) + len(ind_fn))
        f1_AND = 2 * prec_AND * rec_AND / (prec_AND + rec_AND)

        results_AND = [prec_AND, rec_AND, f1_AND]
        all_results_fake.append(results_fake)
        all_results_OR.append(results_OR)
        all_results_AND.append(results_AND)

    return np.array(all_results_fake), np.array(all_results_OR), np.array(all_results_AND)

# %% RELOAD ALL THE RESULTS INSTEAD OF REGENERATING THEM


with np.load(os.path.join(base_folder, 'all_res_web.npz'), allow_pickle=True) as ld:
    all_results = ld['all_results'][()]
with np.load(os.path.join(base_folder, 'all_res_online_web_bk.npz'), allow_pickle=True) as ld:
    all_results_online = ld['all_results'][()]

pl.rcParams['pdf.fonttype'] = 42

# %% CREATE FIGURES
idxFile = 7
name_file = 'J115'
name_file_online = 'J115/'
fname_new = all_results[name_file]['params']['fname']
dims = all_results[name_file]['dims']
Cn = all_results[name_file]['Cn']

predictionsCNN = all_results[name_file]['predictionsCNN']
idx_components_gt = all_results[name_file]['idx_components_gt']
idx_components_gt_online = all_results_online[name_file_online]['idx_components_gt']

tp_comp = all_results[name_file]['tp_comp']
fp_comp = all_results[name_file]['fp_comp']
tp_gt = all_results[name_file]['tp_gt']
fn_gt = all_results[name_file]['fn_gt']
A = all_results[name_file]['A']
A_gt_thr = all_results[name_file]['A_gt_thr']
A_gt = all_results[name_file]['A_gt'][()]
C = all_results[name_file]['C']
C_gt = all_results[name_file]['C_gt']
CCs_batch = all_results[name_file]['CCs']

tp_gt_online = all_results_online[name_file_online]['tp_gt']
tp_comp_online = all_results_online[name_file_online]['tp_comp']
fn_gt_online = all_results_online[name_file_online]['fn_gt']
fp_comp_online = all_results_online[name_file_online]['fp_comp']
A_gt_online = all_results_online[name_file_online]['A_gt']

A_online = all_results_online[name_file_online]['A']
# estimate the downsampling ratio and the original size
try:
    dims_online = all_results_online[name_file_online]['dims']
except:
    downsamp_factor = np.sqrt(A.shape[0]/A_online.shape[0])
    dims_online = tuple(np.array(np.round(np.divide(dims, downsamp_factor))).astype(np.int))

C_online = all_results_online[name_file_online]['C']
A_gt_online = all_results_online[name_file_online]['A_gt']
C_gt_online = all_results_online[name_file_online]['C_gt']
A_thr_online = all_results_online[name_file_online]['A_thr']
A_thr_online = np.vstack([cv2.resize(a.reshape(dims_online, order='F'), dims[::-1]).reshape(-1, order='F') for a in A_thr_online.T]).T
A_online = np.vstack([cv2.resize(a.toarray().reshape(dims_online, order='F'), dims[::-1]).reshape(-1, order='F') for a in A_online.T]).T
A_gt_online = np.vstack([cv2.resize(a.toarray().reshape(dims_online, order='F'), dims[::-1]).reshape(-1, order='F') for a in A_gt_online.T]).T

pl.ylabel('spatial components')
idx_comps_high_r = [np.argsort(predictionsCNN[tp_comp])][::-1][:35]
idx_comps_high_r = np.intersect1d(idx_comps_high_r, np.where(np.array(CCs_batch)>0.97)[0])[[2,4,7,9,11]]
idx_comps_high_r_cnmf = tp_comp[idx_comps_high_r]
idx_comps_high_r_gt = idx_components_gt[tp_gt][idx_comps_high_r]

# %% FIGURE 4a, c, d, e (need to use the dataset J115)
tp_gt_online, tp_comp_online, fn_gt_online, fp_comp_online, performance_cons_off = cm.base.rois.nf_match_neurons_in_binary_masks(
    A_gt_thr[:, idx_components_gt].reshape([dims[0], dims[1], -1], order='F').transpose([2, 0, 1]) * 1.,
    A_thr_online[:, :].reshape([dims[0], dims[1], -1], order='F').transpose([2, 0, 1]) * 1.,
    thresh_cost=.8, min_dist=10,
    print_assignment=False, plot_results=False, Cn=None, labels=['GT', 'Offline'])

# %%
pl.figure('Figure 4a top')
pl.subplot(2, 2, 1)
a1 = plot_contours(A.tocsc()[:, tp_comp], Cn, thr=0.9, colors='yellow', vmax=0.75,
                   display_numbers=False, cmap='gray')
a2 = plot_contours(A_gt.tocsc()[:, tp_gt], Cn, thr=0.9, vmax=0.85, colors='r',
                   display_numbers=False, cmap='gray')
pl.subplot(2, 2, 2)
a3 = plot_contours(A.tocsc()[:, fp_comp], Cn, thr=0.9, colors='yellow', vmax=0.75,
                   display_numbers=False, cmap='gray')
a4 = plot_contours(A_gt.tocsc()[:, idx_components_gt[fn_gt]], Cn, thr=0.9, vmax=0.85, colors='r',
                   display_numbers=False, cmap='gray')

pl.subplot(2, 2, 3)
a1 = plot_contours(A_online[:, tp_comp_online], Cn, thr=0.9, colors='yellow', vmax=0.75,
                   display_numbers=False, cmap='gray')
a2 = plot_contours(A_gt.tocsc()[:, tp_gt_online], Cn, thr=0.9, vmax=0.85, colors='r',
                   display_numbers=False, cmap='gray')
pl.subplot(2, 2, 4)
a3 = plot_contours(A_online[:, fp_comp_online], Cn, thr=0.9, colors='yellow', vmax=0.75,
                   display_numbers=False, cmap='gray')
a4 = plot_contours(A_gt.tocsc()[:, idx_components_gt[fn_gt_online]], Cn, thr=0.9, vmax=0.85, colors='r',
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
    pl.ylim([0.65, 1.02])
    legs.append(nm[:7])
    i += 1
#            break
pl.plot(SNRs, np.mean(all_ys, 0), 'k--', alpha=1, linewidth=2)
pl.legend(legs + ['average'], fontsize=10)
pl.xlabel('SNR threshold')

pl.ylabel('F1_SCORE')
# %
i = 0
legs = []
avg_res_recall = []
avg_res_prec = []
avg_res_f1 = []
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
    avg_res_recall.append(all_results_fake[:, recall_idx])
    avg_res_prec.append(all_results_fake[:, prec_idx])
    avg_res_f1.append(all_results_fake[:, f1_idx])
    # pl.fill_between(SNRs, all_results_OR[:, prec_idx], all_results_AND[:, prec_idx], color='b', alpha=.1)
    pl.plot(SNRs, all_results_fake[:, prec_idx], color='b')
    # pl.fill_between(SNRs, all_results_OR[:, recall_idx], all_results_AND[:, recall_idx], color='r',
    #                     alpha=.1)
    pl.plot(SNRs, all_results_fake[:, recall_idx], color='r')

    # pl.fill_between(SNRs, all_results_OR[:, f1_idx], all_results_AND[:, f1_idx], color='g', alpha=.1)
    # pl.plot(SNRs, all_results_fake[:, f1_idx], '.-', color='g')
# pl.plot(SNRs, np.array(avg_res_prec).mean(0), '.-', color='b')
# pl.plot(SNRs, np.array(avg_res_recall).mean(0), '.-', color='r')
# pl.plot(SNRs, np.array(avg_res_f1).mean(0), '.-', color='g')
pl.legend(['precision', 'recall'], fontsize=10)
pl.xlabel('SNR threshold')

pl.ylim([0.6, 1.02])
# %% FIGURE 4 b  --> See file Figure_4_b.py