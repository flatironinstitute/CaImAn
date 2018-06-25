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
from operator import itemgetter
from scipy.io import loadmat
from scipy.stats import pearsonr


gSig = 3   # gaussian width of a 2D gaussian kernel, which approximates a neuron
gSiz0 = 7   # average diameter of a neuron
min_corr = .9
min_pnr = 15
s_min = -10
initbatch = 500
expected_comps = 200


small = True

fname = 'test_sim.mat'
test_sim = loadmat(fname)
(A, C, b, A_cnmfe, f, C_cnmfe, Craw_cnmfe, b0, sn, Yr, S_cnmfe,
    A_cnmfe_patch, C_cnmfe_patch, Craw_cnmfe_patch) = itemgetter(
    'A', 'C', 'b', 'A_cnmfe', 'f', 'C_cnmfe', 'Craw_cnmfe', 'b0', 'sn', 'Y', 'S_cnmfe',
    'A_cnmfe_patch', 'C_cnmfe_patch', 'Craw_cnmfe_patch')(test_sim)
N, T = C.shape
dims = (253, 316)
nA = np.sqrt(np.sum(A**2, 0))
A /= nA
C *= nA[:, None]

if small:
    A = A.reshape(dims + (-1,), order='F')[64:128, 64:160].reshape((-1, A.shape[-1]), order='F')
    keep = (A**2).sum(0) > .05
    A = A[:, keep]
    C = C[keep]

try:
    Yr, dims, T = cm.load_memmap('Yr_d1_64_d2_96_d3_1_order_C_frames_2000_.mmap' if small
                                 else 'Yr_d1_253_d2_316_d3_1_order_C_frames_2000_.mmap')
    Y = Yr.T.reshape((T,) + dims, order='F')
except:
    Y = Yr.T.reshape((-1,) + dims, order='F')
    if small:
        Y = Y[:, 64:128, 64:160]
    fname_new = cm.save_memmap([Y], base_name='Yr', order='C')
    Yr, dims, T = cm.load_memmap(fname_new)
    Y = Yr.T.reshape((T,) + dims, order='F')


#%% RUN (offline) CNMF-E algorithm on the entire batch for sake of comparison

cnm_batch = cnmf.CNMF(2, method_init='corr_pnr', k=None, gSig=(gSig, gSig), gSiz=(gSiz0, gSiz0),
                      merge_thresh=.9, p=1, tsub=1, ssub=1, only_init_patch=True, gnb=0,
                      min_corr=min_corr, min_pnr=min_pnr, normalize_init=False,
                      ring_size_factor=18./gSiz0, center_psf=True, ssub_B=2, init_iter=1, s_min=s_min)

cnm_batch.fit(Y)

print(('Number of components:' + str(cnm_batch.A.shape[-1])))


def tight():
    plt.xlim(0, dims[1])
    plt.ylim(0, dims[0])
    plt.axis('off')
    plt.xticks([])
    plt.yticks([])

Cn, pnr = cm.summary_images.correlation_pnr(Y, gSig=gSig, center_psf=True, swap_dim=False)
plt.figure()
crd = cm.utils.visualization.plot_contours(A, Cn, thr=.8, lw=3, display_numbers=False)
crd = cm.utils.visualization.plot_contours(cnm_batch.A, Cn, thr=.8, c='r')
tight()
plt.savefig('online1p_batch.pdf', pad_inches=0, bbox_inches='tight')
cm.base.rois.register_ROIs(A, cnm_batch.A, dims, align_flag=0)


#%% RUN (offline) CNMF-E algorithm on the initial batch

seeded = False
if not seeded:
    cnm_init = cnmf.CNMF(2, method_init='corr_pnr', k=None, gSig=(gSig, gSig), gSiz=(gSiz0, gSiz0),
                         merge_thresh=.98, p=1, tsub=1, ssub=1, only_init_patch=True, gnb=0,
                         min_corr=min_corr, min_pnr=min_pnr, normalize_init=False,
                         ring_size_factor=18./gSiz0, center_psf=True, ssub_B=2, init_iter=1, s_min=s_min,
                         minibatch_shape=100, minibatch_suff_stat=5, update_num_comps=True,
                         rval_thr=.9, thresh_fitness_delta=-30, thresh_fitness_raw=-50,
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
crd = cm.utils.visualization.plot_contours(A, Cn_init, thr=.8, lw=3, display_numbers=False)
crd = cm.utils.visualization.plot_contours(cnm_init.A, Cn_init, thr=.8, c='r')
tight()
plt.savefig('online1p_init.pdf', pad_inches=0, bbox_inches='tight')
cm.base.rois.register_ROIs(A, cnm_init.A, dims, align_flag=0)


#%% run (online) CNMF-E algorithm

cnm = deepcopy(cnm_init)
gSiz = 13
cnm.gSiz = (gSiz, gSiz)
cnm.ring_size_factor = 18./gSiz
cnm.options['init_params']['gSiz'] = (gSiz, gSiz)
cnm.options['init_params']['ring_size_factor'] = 18./gSiz

if seeded:  # remove some components and let them be found in online mode
    idx = filter(lambda a: a not in [3, 14, 24], range(len(cnm.C)))
    cnm.C = cnm.C[idx]
    cnm.YrA = cnm.YrA[idx]
    cnm.A = cnm.A[:, idx]
    cnm_init.W, cnm_init.b0 = cnmf.initialization.compute_W(
        Yr[:, :initbatch], cnm.A.toarray(), cnm.C, dims, 18, ssub=2)

cnm._prepare_object(np.asarray(Yr[:, :initbatch]), T, expected_comps)
t = cnm.initbatch

for frame in Y[initbatch:]:
    cnm.fit_next(t, frame.copy().reshape(-1, order='F'))
    t += 1

print(('Number of components:' + str(cnm.Ab.shape[-1])))

plt.figure()
crd = cm.utils.visualization.plot_contours(A, Cn, thr=.8, lw=3, display_numbers=False)
crd = cm.utils.visualization.plot_contours(cnm.Ab, Cn, thr=.8, c='r')
tight()
plt.savefig('online1p_online.pdf', pad_inches=0, bbox_inches='tight')
cm.base.rois.register_ROIs(A, cnm.Ab, dims, align_flag=0)


#%% compare online to batch

print('RSS of W:  online vs init:  {0:.4f}'.format((cnm.W - cnm_init.W).power(2).sum()))
print('            batch vs init:  {0:.4f}'.format((cnm_batch.W - cnm_init.W).power(2).sum()))
print('           online vs batch: {0:.4f}'.format((cnm.W - cnm_batch.W).power(2).sum()))


#%% compare to ground truth 

matched_ROIs1b, matched_ROIs2b, non_matched1b, non_matched2b, performanceb, A2b = register_ROIs(
    A, cnm_batch.A, dims, align_flag=False)
matched_ROIs1, matched_ROIs2, non_matched1, non_matched2, performance, A2 = register_ROIs(
    A, cnm.Ab, dims, align_flag=False)
matched_ROIs1i, matched_ROIs2i, non_matched1i, non_matched2i, performancei, A2i = register_ROIs(
    A, cnm_init.A, dims, align_flag=False)

#traces
cor_batch = [pearsonr(c1, c2)[0] for (c1, c2) in
             np.transpose([C[matched_ROIs1b], cnm_batch.C[matched_ROIs2b]], (1, 0, 2))]
cor = [pearsonr(c1, c2)[0] for (c1, c2) in
       np.transpose([C[matched_ROIs1], cnm.C_on[matched_ROIs2]], (1, 0, 2))]

print('Correlation  mean   median')
print('    batch:  {0:.4f}  {1:.4f}'.format(np.mean(cor_batch), np.median(cor_batch)))
print('   online:  {0:.4f}  {1:.4f}'.format(np.mean(cor), np.median(cor)))

plt.figure(figsize=(20, cnm.N))
for i in range(len(C)):
    plt.subplot(cnm.N, 1, 1 + i)
    plt.plot(C[i], c='k', lw=4, label='truth')
    try:
        plt.plot(cnm_batch.C[matched_ROIs2b[list(matched_ROIs1b).index(i)]],
                 c='r', lw=3, label='batch')
    except ValueError:
        pass
    try:
        plt.plot(cnm.C_on[matched_ROIs2[list(matched_ROIs1).index(i)]],
                 c='y', lw=2, label='online')
    except ValueError:
        pass
    if i == 0:
        plt.legend(ncol=3)
plt.tight_layout()
plt.savefig('online1p_traces.pdf')

# shapes
over_batch = [c1.dot(c2) / np.sqrt(c1.dot(c1) * c2.dot(c2)) for (c1, c2) in
             np.transpose([A.T[matched_ROIs1b],
                           cnm_batch.A.toarray().T[matched_ROIs2b]], (1, 0, 2))]
over = [c1.dot(c2) / np.sqrt(c1.dot(c1) * c2.dot(c2)) for (c1, c2) in
       np.transpose([A.T[matched_ROIs1], cnm.Ab.toarray().T[matched_ROIs2]], (1, 0, 2))]
over_init = [c1.dot(c2) / np.sqrt(c1.dot(c1) * c2.dot(c2)) for (c1, c2) in
       np.transpose([A.T[matched_ROIs1i], cnm_init.A.toarray().T[matched_ROIs2i]], (1, 0, 2))]

print('Overlap   mean   median')
print(' batch:  {0:.4f}  {1:.4f}'.format(np.mean(over_batch), np.median(over_batch)))
print('online:  {0:.4f}  {1:.4f}'.format(np.mean(over), np.median(over)))
print('  init:  {0:.4f}  {1:.4f}'.format(np.mean(over_init), np.median(over_init)))


#%% sorted ring weights

d1, d2 = (np.array(dims) + 1) // 2
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
