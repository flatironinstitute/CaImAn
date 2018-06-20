try:
    get_ipython().magic(u'load_ext autoreload')
    get_ipython().magic(u'autoreload 2')
except:
    print('NOT IPYTHON')

import caiman as cm
from caiman.source_extraction import cnmf
from caiman.base.rois import register_ROIs
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
import h5py
from operator import itemgetter
from scipy.io import loadmat
from scipy.stats import pearsonr

gSig = 3   # gaussian width of a 2D gaussian kernel, which approximates a neuron
gSiz = 13  # average diameter of a neuron
min_corr = .7
min_pnr = 7
s_min = -10
initbatch = 500
expected_comps = 100

# data from https://www.dropbox.com/sh/6395g5wwlv63f0s/AACTNVivxYs7IIyeS67SdV2Qa?dl=0
fname = 'eLife_submission/data/blood_vessel_10Hz.mat'
Y = h5py.File(fname)['Y'].value.astype(np.float32)
Y = Y[:, 64:104, 64:128]
T = len(Y)
dims = Y.shape[1:]
Yr = Y.T.reshape((-1, T))

# load matlab results
A, C_raw, C = itemgetter('A', 'C_raw', 'C')(loadmat(
    '/mnt/home/jfriedrich/CNMF_E/eLife_submission/code/' +
    'scripts_figures/fig_striatum/results_bk.mat'))
nA = np.sqrt(A.power(2).sum(0))
A = A.toarray().reshape((256, 256, -1), order='C')[64:104, 64:128].reshape((-1, A.shape[-1]), order='F')
A /= nA
A = A[:, (A**2).sum(0) > .2]


#%% RUN (offline) CNMF-E algorithm on the entire batch for sake of comparison

cnm_batch = cnmf.CNMF(2, method_init='corr_pnr', k=None, gSig=(gSig, gSig), gSiz=(gSiz, gSiz),
                      merge_thresh=.65, p=1, tsub=1, ssub=1, only_init_patch=True, gnb=0,
                      min_corr=min_corr, min_pnr=min_pnr, normalize_init=False,
                      ring_size_factor=1.4, center_psf=True, ssub_B=2, init_iter=1, s_min=s_min)

cnm_batch.fit(Y)

print(('Number of components:' + str(cnm_batch.A.shape[-1])))

Cn, pnr = cm.summary_images.correlation_pnr(Y, gSig=gSig, center_psf=True, swap_dim=False)
plt.figure()
cm.utils.visualization.plot_contours(A, Cn, thr=.6, lw=3)
cm.utils.visualization.plot_contours(cnm_batch.A, Cn, thr=.6, color='r')
cm.base.rois.register_ROIs(A, cnm_batch.A, dims, align_flag=0)

# # discard low quality components
# idx_components, idx_components_bad = cm.components_evaluation.estimate_components_quality(
#     cnm_batch.C + cnm_batch.YrA, Yr, cnm_batch.A, cnm_batch.C, cnm_batch.b, cnm_batch.f,
#     final_frate=10, Npeaks=10, r_values_min=.1, fitness_min=-20, fitness_delta_min=-30)
# print(('Keeping ' + str(len(idx_components)) +
#        ' and discarding  ' + str(len(idx_components_bad))))
# A_b = cnm_batch.A[:, idx_components]
# C_b = cnm_batch.C[idx_components]
# cm.base.rois.register_ROIs(A, A_b, dims, align_flag=0)

# plt.figure()
# crd = cm.utils.visualization.plot_contours(A, Cn, thr=.6, lw=3)
# crd = cm.utils.visualization.plot_contours(A_b, Cn, thr=.6, color='r')


#%% RUN (offline) CNMF-E algorithm on the initial batch

seeded = False
if not seeded:
    cnm_init = cnmf.CNMF(2, method_init='corr_pnr', k=None, gSig=(gSig, gSig), gSiz=(gSiz, gSiz),
                         merge_thresh=.8, p=1, tsub=1, ssub=1, only_init_patch=True, gnb=0,
                         min_corr=min_corr, min_pnr=min_pnr, normalize_init=False,
                         ring_size_factor=1.4, center_psf=True, ssub_B=2, init_iter=1, s_min=s_min,
                         minibatch_shape=100, minibatch_suff_stat=5, update_num_comps=True,
                         rval_thr=.9, thresh_fitness_delta=-20, thresh_fitness_raw=-40,
                         batch_update_suff_stat=True)

    cnm_init.fit(Y[:initbatch])

else:  # seeded from batch
    cnm_init = deepcopy(cnm_batch)
    cnm_init.initbatch = initbatch
    cnm_init.update_num_comps = False
    cnm_init.C = cnm_init.C[:, :initbatch]
    cnm_init.f = cnm_init.f[:, :initbatch]
    cnm_init.YrA = cnm_init.YrA[:, :initbatch]
    cnm_init.W, cnm_init.b0 = cnmf.initialization.compute_W(
        Yr[:, :initbatch], cnm_init.A.toarray(), cnm_init.C, dims, 18, ssub=2)

print(('Number of components:' + str(cnm_init.A.shape[-1])))

Cn_init, pnr_init = cm.summary_images.correlation_pnr(
    Y[:initbatch], gSig=gSig, center_psf=True, swap_dim=False)
plt.figure()
cm.utils.visualization.plot_contours(A, Cn_init, thr=.6, lw=3)
cm.utils.visualization.plot_contours(cnm_init.A, Cn_init, thr=.6, color='r')
cm.base.rois.register_ROIs(A, cnm_init.A, dims, align_flag=0)


#%% run (online) CNMF-E algorithm

cnm = deepcopy(cnm_init)
cnm._prepare_object(np.asarray(Yr[:, :initbatch]), T, expected_comps)
t = cnm.initbatch

for frame in Y[initbatch:]:
    cnm.fit_next(t, frame.copy().reshape(-1, order='F'))
    t += 1

print(('Number of components:' + str(cnm.Ab.shape[-1])))

plt.figure()
crd = cm.utils.visualization.plot_contours(A, Cn, thr=.6, lw=3)
crd = cm.utils.visualization.plot_contours(cnm.Ab, Cn, thr=.6, color='r')
cm.base.rois.register_ROIs(A, cnm.Ab, dims, align_flag=0)

# # discard low quality components
# idx_components, idx_components_bad = cm.components_evaluation.estimate_components_quality(
#     cnm.noisyC[:cnm.M], Yr, cnm.Ab, cnm.C_on[:cnm.M], cnm.b, cnm.f,
#     final_frate=10, Npeaks=10, r_values_min=.1, fitness_min=-20, fitness_delta_min=-30)
# print(('Keeping ' + str(len(idx_components)) +
#        ' and discarding  ' + str(len(idx_components_bad))))
# A_ = cnm.Ab[:, idx_components]
# C_ = cnm.C_on[idx_components]

# plt.figure()
# cm.utils.visualization.plot_contours(A, Cn, thr=.6, lw=3)
# cm.utils.visualization.plot_contours(A_, Cn, thr=.6, color='r')
# cm.base.rois.register_ROIs(A, A_, dims, align_flag=0)


#%% compare online to batch

print('RSS of W:  online vs init:  {0}'.format((cnm.W - cnm_init.W).power(2).sum()))
print('            batch vs init:  {0}'.format((cnm_batch.W - cnm_init.W).power(2).sum()))
print('           online vs batch: {0}'.format((cnm.W - cnm_batch.W).power(2).sum()))

# sorted ring weights

d1, d2 = np.array(dims) / 2
ringidx = cnm._ringidx
circ = len(ringidx[0])


def get_angle(x, y):
    x = float(x)
    y = float(y)
    if x == 0:
        return np.pi / 2 if y > 0 else np.pi / 2 * 3
    else:
        if x < 0:
            return np.arctan(y / x) + np.pi
        else:
            if y >= 0:
                return np.arctan(y / x)
            else:
                return np.arctan(y / x) + 2 * np.pi


sort_by_angle = np.argsort([get_angle(ringidx[0][i], ringidx[1][i])
                            for i in range(circ)])


def sort_W(W):
    Q = np.zeros((d1 * d2, circ)) * np.nan
    for i in range(d1 * d2):
        pixel = np.unravel_index(i, (d1, d2), order='F')
        x = pixel[0] + ringidx[0]
        y = pixel[1] + ringidx[1]
        inside = (x >= 0) * (x < d1) * (y >= 0) * (y < d2)
        # faster than  Q[i, inside] = W[i, np.ravel_multi_index((x[inside], y[inside]), (d1, d2), order='F')].toarray()
        Q[i, inside] = W.data[W.indptr[i]:W.indptr[i + 1]]
    Q = Q[:, sort_by_angle]
    return Q


Q_batch = sort_W(cnm_batch.W)
Q_init = sort_W(cnm_init.W)
Q = sort_W(cnm.W)

plt.figure()
plt.plot(np.nanmean(Q, 0))
plt.plot(np.nanmean(Q_init, 0))
plt.plot(np.nanmean(Q_batch, 0))
plt.xlabel('Pixel on ring')
plt.ylabel('Average ring weight')

# traces

matched_ROIs1, matched_ROIs2, non_matched1, non_matched2, performance, A2 = register_ROIs(
    cnm.Ab, cnm_batch.A, dims, align_flag=False, thresh_cost=.7)

cor = [pearsonr(c1, c2)[0] for (c1, c2) in
       np.transpose([cnm.C_on[matched_ROIs1], cnm_batch.C[matched_ROIs2]], (1, 0, 2))]
print(np.mean(cor), np.median(cor))

plt.figure(figsize=(20, 20))
for i in range(cnm.N):
    plt.subplot(cnm.N, 1, 1 + i)
    plt.plot(cnm.C_on[matched_ROIs1[i]])
    plt.plot(cnm_batch.C[matched_ROIs2[i]])
