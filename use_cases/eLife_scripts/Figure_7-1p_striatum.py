# -*- coding: utf-8 -*-
"""
This script reproduces the results for Figure 7, analyzing 1p microendoscopic
data using the CaImAn implementation of the CNMF-E algorithm. The algorithm
using both a patch-based and a non patch-based approach and compares them with
the results obtained from the MATLAB implementation.

More info can be found in the companion paper
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.optimize import minimize
from scipy.sparse import csc_matrix
from scipy.stats import pearsonr
from scipy.ndimage import center_of_mass
from operator import itemgetter
import h5py
import caiman as cm
from caiman.base.rois import register_ROIs
from caiman.source_extraction import cnmf
import cv2
import os


# data from https://www.dropbox.com/sh/6395g5wwlv63f0s/AACTNVivxYs7IIyeS67SdV2Qa?dl=0
base_folder = '/mnt/ceph/neuro/DataForPublications/DATA_PAPER_ELIFE/WEBSITE'
fname = os.path.join(base_folder, 'blood_vessel_10Hz.mat')
Y = h5py.File(fname)['Y'].value.astype(np.float32)

gSig = 3   # gaussian width of a 2D gaussian kernel, which approximates a neuron
gSiz = 13  # average diameter of a neuron


#%% perform memory mapping and loading
fname_new = cm.save_memmap([Y], base_name='Yr', order='C')
Yr, dims, T = cm.load_memmap(fname_new)
Y = Yr.T.reshape((T,) + dims, order='F')

#%% run w/o patches
dview, n_processes = None, 2

cnm = cnmf.CNMF(n_processes=n_processes, method_init='corr_pnr', k=None, dview=dview,
                gSig=(gSig, gSig), gSiz=(gSiz, gSiz), merge_thresh=.65, p=1, tsub=1, ssub=1,
                only_init_patch=True, gnb=0, min_corr=.7, min_pnr=7, normalize_init=False,
                ring_size_factor=1.4, center_psf=True, ssub_B=2, init_iter=1, s_min=-10)
cnm.fit(Y)

#%% run w/ patches
c, dview, n_processes = cm.cluster.setup_cluster(
    backend='local', n_processes=None, single_thread=False)

cnmP = cnmf.CNMF(n_processes=n_processes, method_init='corr_pnr', k=None, dview=dview,
                 gSig=(gSig, gSig), gSiz=(gSiz, gSiz), merge_thresh=.65, p=1, tsub=1, ssub=1,
                 only_init_patch=True, gnb=0, min_corr=.7, min_pnr=7, normalize_init=False,
                 ring_size_factor=1.4, center_psf=True, ssub_B=2, init_iter=1, s_min=-10,
                 nb_patch=0, del_duplicates=True, rf=(64, 64), stride=(32, 32))
cnmP.fit(Y)


#%% DISCARD LOW QUALITY COMPONENT
def discard(cnm, final_frate=10,
            r_values_min=0.1,  # threshold on space consistency
            fitness_min=-20,  # threshold on time variability
            # threshold on time variability (if nonsparse activity)
            fitness_delta_min=-30,
            Npeaks=10):
    traces = cnm.estimates.C + cnm.estimates.YrA

    idx_components, idx_components_bad = cm.components_evaluation.estimate_components_quality(
        traces, Yr, cnm.estimates.A, cnm.estimates.C, cnm.estimates.b, cnm.estimates.f, final_frate=final_frate, Npeaks=Npeaks,
        r_values_min=r_values_min, fitness_min=fitness_min, fitness_delta_min=fitness_delta_min)

    print(('Keeping ' + str(len(idx_components)) +
           ' and discarding  ' + str(len(idx_components_bad))))
    A_ = cnm.estimates.A[:, idx_components]
    C_ = cnm.estimates.C[idx_components]
    return A_, C_, traces[idx_components]


A_, C_, traces = discard(cnm)
A_P, C_P, tracesP = discard(cnmP)


# DISCARD TOO SMALL COMPONENT
notsmall = np.sum(A_.toarray() > 0, 0) >= 125
A_ = A_[:, notsmall]
C_ = C_[notsmall]
notsmall = np.sum(A_P.toarray() > 0, 0) >= 125
A_P = A_P[:, notsmall]
C_P = C_P[notsmall]


# DISCARD TOO ECCENTRIC COMPONENT
def aspect_ratio(img):
    M = cv2.moments(img)
    cov = np.array([[M['mu20'], M['mu11']], [M['mu11'], M['mu02']]]) / M['m00']
    EV = np.sort(np.linalg.eigh(cov)[0])
    return np.sqrt(EV[1] / EV[0])


def keep_ecc(A, thresh=3):
    ar = np.array([aspect_ratio(a.reshape(dims)) for a in A.T])
    notecc = ar < thresh
    centers = np.array([center_of_mass(a.reshape(dims)) for a in A.T])
    border = np.array([c.min() < 3 or (np.array(dims) - c).min() < 3 for c in centers])
    return notecc | border


keep = keep_ecc(A_.toarray())
A_ = A_[:, keep]
C_ = C_[keep]
keep = keep_ecc(A_P.toarray())
A_P = A_P[:, keep]
C_P = C_P[keep]


#%% load matlab results and match ROIs
A, C_raw, C = itemgetter('A', 'C_raw', 'C')(loadmat(
    os.path.join(base_folder, 'results_bk.mat')))

A_ = csc_matrix(A_.toarray().reshape(
    dims + (-1,), order='C').reshape((-1, A_.shape[-1]), order='F'))
A_P = csc_matrix(A_P.toarray().reshape(
    dims + (-1,), order='C').reshape((-1, A_P.shape[-1]), order='F'))

ids = [616, 524, 452, 305, 256, 573, 181, 574, 575, 619]

def match(a, c):
    matched_ROIs1, matched_ROIs2, non_matched1, non_matched2, performance, A2 = register_ROIs(
        A, a, dims, align_flag=False, thresh_cost=.7)
    cor = [pearsonr(c1, c2)[0] for (c1, c2) in
           np.transpose([C[matched_ROIs1], c[matched_ROIs2]], (1, 0, 2))]
    print(np.mean(cor), np.median(cor))
    return matched_ROIs2[[list(matched_ROIs1).index(i) for i in ids]]


ids_ = match(A_, C_)
# {'f1_score': 0.9033778476040848, 'recall': 0.8468335787923417, 'precision': 0.968013468013468, 'accuracy': 0.8237822349570201}
# (0.7831526916134324, 0.8651673117308342)
ids_P = match(A_P, C_P)
# {'f1_score': 0.8979591836734694, 'recall': 0.8424153166421208, 'precision': 0.9613445378151261, 'accuracy': 0.8148148148148148}
# (0.7854095007286398, 0.870879114022712)


#%% plot ROIs and traces
cn_filter, pnr = cm.summary_images.correlation_pnr(
    Y, gSig=gSig, center_psf=True, swap_dim=False)

fig = plt.figure(figsize=(30, 10))
fig.add_axes([0, 0, .33, 1])
plt.rc('lines', lw=1.2)
cm.utils.visualization.plot_contours(
    A, cn_filter.T, thr=.6, vmax=0.95, colors='w', display_numbers=False)
cm.utils.visualization.plot_contours(
    A_P[:, np.array([i not in ids_P for i in range(A_P.shape[1])])], cn_filter,
    thr=.6, vmax=0.95, colors='r', display_numbers=False)
plt.rc('lines', lw=3.5)
for k, i in enumerate(ids_P):
    cm.utils.visualization.plot_contours(
        A_P[:, i], cn_filter.T, thr=.7, vmax=0.95, colors='C%d' % k, display_numbers=False)
plt.rc('lines', lw=1.5)
plt.plot([1, 2], zorder=-100, lw=3, c='w', label='Zhou et al.')
plt.plot([1, 2], zorder=-100, lw=3, c='r', label='patches')
lg = plt.legend(loc=(.44, .9), frameon=False, fontsize=20)
for text in lg.get_texts():
    text.set_color('w')
plt.axis('off')
plt.xticks([])
plt.yticks([])
fig.add_axes([.33, 0, .67, 1])
for i, n in enumerate(ids):
    plt.plot(.9 * C[n] / C[n].max() + (9 - i), c='k', lw=5.5, label='Zhou et al.')
    a, b = minimize(lambda x:
                    np.sum((x[0] + x[1] * C_[ids_[i]] - C[n] / C[n].max())**2), (0, 1e-3)).x
    plt.plot(.9 * (a + b * C_[ids_[i]]) + (9 - i), lw=4, c='cyan', label='no patches')
    a, b = minimize(lambda x:
                    np.sum((x[0] + x[1] * C_P[ids_P[i]] - C[n] / C[n].max())**2), (0, 1e-3)).x
    plt.plot(.9 * (a + b * C_P[ids_P[i]]) + (9 - i), lw=2.5, c='r', label='patches')
    if i == 0:
        plt.legend(ncol=3, loc=(.3, .96), frameon=False, fontsize=20, columnspacing=4)
    plt.scatter([-100], [.45 + (9 - i)], c='C%d' % i, s=80)
plt.ylim(0, 10)
plt.axis('off')
plt.xticks([])
plt.yticks([])
plt.xlim(-350, 6050)
plt.ylim(-.1, 10.2)
plt.show()