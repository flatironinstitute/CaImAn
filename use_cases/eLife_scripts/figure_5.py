# -*- coding: utf-8 -*-
"""
This script reproduces the results for all panels of Figure 5, (trace quality)
The script loads the saved results and uses them to plot the traces and
correlation coefficients with consensus traces.
For running the CaImAn batch and CaImAn online algorithm and
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
import glob
import pylab as pl
import scipy

# %%
pl.rcParams['pdf.fonttype'] = 42
font = {'family': 'Arial',
        'weight': 'regular',
        'size': 20}
pl.rc('font', **font)


base_folder = '/mnt/ceph/neuro/DataForPublications/DATA_PAPER_ELIFE/WEBSITE/'

#%% to merge results from all files
mergefiles = False
if mergefiles:
    fls = glob.glob('.')  # here should go the path to the generated files
    all_results_online = dict()
    for fl in fls:
        with np.load(fl) as ld:
            all_results_online.update((ld['all_results'][()]))

    np.savez(os.path.join(base_folder, 'all_res_online_web.npz'), all_results=all_results_online)

# %% RELOAD ALL THE RESULTS INSTEAD OF REGENERATING THEM
with np.load(os.path.join(base_folder, 'all_res_web.npz')) as ld:
    all_results = ld['all_results'][()]
with np.load(os.path.join(base_folder, 'all_res_online_web_bk.npz')) as ld:
    all_results_online = ld['all_results'][()]

pl.rcParams['pdf.fonttype'] = 42
#%% FIGURE 5 a,b,c
xcorr_online = []
for k, fl_results in all_results_online.items():
    xcorr_online.append(fl_results['CCs'])

xcorr_offline = []
for k, fl_results in all_results.items():
    xcorr_offline.append(fl_results['CCs'])

xcorrs = [[np.median(all_results[key]['CCs']), np.median(all_results_online[key+'/']['CCs'])] for key in all_results.keys()]

names = ['0300.T',
         '0400.T',
         'YST',
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
# a_on = a[0]
# a_off =a3[0]
# %
pl.close()
pl.figure('Figure 5c')

pl.plot(np.arange(0.01, 1, .01), a_on)
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
    mu_on = np.array(xcorrs)[:,1]#np.array([np.median(xc) for xc in xcorr_online[:]])
    mu_off = np.array(xcorrs)[:,0]#np.array([np.median(xc) for xc in xcorr_offline[:]])
    pl.plot(np.concatenate([mu_off[:, None], mu_on[:, None]], axis=1).T, 'o-')
    pl.xlabel('Offline and Online')
    pl.ylabel('Correlation coefficient')
    for i, x in enumerate(names):
        pl.text(1.1, mu_on[i], x, rotation=0, verticalalignment='baseline')
        pl.text(-.25, mu_off[i]-i/200, x, rotation=0, verticalalignment='baseline')

        pl.xlim([-.5,1.5])

# %% FIGURE 5a MASKS  (need to use the dataset k53)
pl.figure('Figure 5a masks')
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
# YrA_gt = all_results[name_file]['YrA_gt']
# YrA = all_results[name_file]['YrA']


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
# A_gt_thr_online = all_results_online[name_file]['A_gt_thr']
A_thr_online = all_results_online[name_file_online]['A_thr']
A_thr_online = np.vstack([cv2.resize(a.reshape(dims_online, order='F'), dims[::-1]).reshape(-1, order='F') for a in A_thr_online.T]).T
A_online = np.vstack([cv2.resize(a.toarray().reshape(dims_online, order='F'), dims[::-1]).reshape(-1, order='F') for a in A_online.T]).T
A_gt_online = np.vstack([cv2.resize(a.toarray().reshape(dims_online, order='F'), dims[::-1]).reshape(-1, order='F') for a in A_gt_online.T]).T


pl.ylabel('spatial components')
idx_comps_high_r = [np.argsort(predictionsCNN[tp_comp])][::-1][:35]
idx_comps_high_r = np.intersect1d(idx_comps_high_r, np.where(np.array(CCs_batch)>0.97)[0])[[2,4,7,9,11]]
idx_comps_high_r_cnmf = tp_comp[idx_comps_high_r]
idx_comps_high_r_gt = idx_components_gt[tp_gt][idx_comps_high_r]

# match to GT
_, idx_comps_high_r_online, _, _ , _= cm.base.rois.nf_match_neurons_in_binary_masks(
    A_gt_thr[:, idx_comps_high_r_gt].reshape([dims[0], dims[1], -1], order='F').transpose([2, 0, 1]) * 1.,
    A_thr_online.reshape([dims[0], dims[1], -1], order='F').transpose([2, 0, 1]) * 1.,
    thresh_cost=.8, min_dist=10,
    print_assignment=False, plot_results=False, Cn=None, labels=['GT', 'Offline'])



images_nice = (A.tocsc()[:, idx_comps_high_r_cnmf].toarray().reshape(dims + (-1,), order='F')).transpose(2, 0, 1)
images_nice_gt = (A_gt.tocsc()[:, idx_comps_high_r_gt].toarray().reshape(dims + (-1,), order='F')).transpose(2, 0,
                                                                                                             1)

images_nice_online = (A_online[:, idx_comps_high_r_online].reshape(dims + (-1,), order='F')).transpose(2, 0, 1)

cms = np.array([scipy.ndimage.center_of_mass(img) for img in images_nice]).astype(np.int)


images_nice_crop = [img[cm_[0] - 15:cm_[0] + 15, cm_[1] - 15:cm_[1] + 15] for cm_, img in zip(cms, images_nice)]
images_nice_crop_gt = [img[cm_[0] - 15:cm_[0] + 15, cm_[1] - 15:cm_[1] + 15] for cm_, img in
                       zip(cms, images_nice_gt)]

images_nice_online_crop = [img[cm_[0] - 15:cm_[0] + 15, cm_[1] - 15:cm_[1] + 15] for cm_, img in zip(cms, images_nice_online)]



indexes = np.concatenate([range(1,16,3), range(2,16,3) , range(3,16,3)])
count = 0
for img in images_nice_crop:
    pl.subplot(5, 3, indexes[count])
    pl.imshow(img)
    pl.axis('off')
    count += 1

for img in images_nice_crop_gt:
    pl.subplot(5, 3, indexes[count])
    pl.imshow(img)
    pl.axis('off')
    count += 1

for img in images_nice_online_crop:
    pl.subplot(5, 3, indexes[count])
    pl.imshow(img)
    pl.axis('off')
    count += 1


# %% FIGURE 5a Traces  (need to use the dataset J115)
pl.figure('Figure 5a tces')
traces_gt = C_gt[idx_comps_high_r_gt] # + YrA_gt[idx_comps_high_r_gt]
traces_cnmf = C[idx_comps_high_r_cnmf] # + YrA[idx_comps_high_r_cnmf]
traces_online = C_online[idx_comps_high_r_online]  # + YrA[idx_comps_high_r_cnmf]

traces_gt /= np.max(traces_gt, 1)[:, None]
traces_cnmf /= np.max(traces_cnmf, 1)[:, None]
traces_online /= np.max(traces_online, 1)[:, None]


pl.plot(0.05+scipy.signal.decimate(traces_cnmf, 10, 1).T - np.arange(5) * 1, 'y')
pl.plot(0.1+scipy.signal.decimate(traces_gt, 10, 1).T - np.arange(5) * 1, 'k', linewidth=.5)
pl.plot(0.15+scipy.signal.decimate(traces_online, 10, 1).T - np.arange(5) * 1, 'r', linewidth=.5)