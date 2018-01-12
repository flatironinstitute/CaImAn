#!/usr/bin/env python

try:
    get_ipython().magic(u'load_ext autoreload')
    get_ipython().magic(u'autoreload 2')
except:
    print('Not launched under iPython')

import numpy as np
from scipy.io import loadmat
from operator import itemgetter
import matplotlib.pyplot as plt
from scipy.ndimage import center_of_mass
import itertools
import caiman as cm
from caiman.source_extraction import cnmf
from caiman.source_extraction.cnmf.utilities import compute_residuals


#%%
def get_mapping(inferredC, trueC, A):
    """
    finds the mapping that maps each true neuron to the best inferred one
    such that overall Ca correlation is maximized, trueC[n] ~ inferredC[mapIdx[n]].
    For neurons that have not been found, mapIdx will contain NaNs.
    """
    N, T = trueC.shape
    cc = np.corrcoef(A.T.reshape(N, -1)) > .2
    blocks = [set(np.where(c)[0]) for c in cc]
    for k in range(len(blocks)):
        for _ in range(10):
            for j in range(len(blocks) - 1, k, -1):
                if len(blocks[k].intersection(blocks[j])):
                    blocks[k] = blocks[k].union(blocks[j])
                    blocks.pop(j)
    mapIdx = np.nan * np.zeros(N)
    corT = np.asarray([[np.corrcoef(s, tC)[0, 1]
                        for s in inferredC] for tC in trueC])
    # first assign neurons that have mutually highest correlation
    # indices that haven't been a target of the mapping yet
    noTarget = list(range(len(inferredC)))
    for _ in range(10):
        if np.any(np.isnan(mapIdx)) and len(noTarget):
            nanIdx = np.where(np.isnan(mapIdx))[0]
            q = corT[np.isnan(mapIdx)][:, noTarget]
            to_del = []
            for k in range(len(q)):
                if np.argmax(q[:, np.argmax(q[k])]) == k:  # mutually highest correlation
                    mapIdx[nanIdx[k]] = noTarget[np.argmax(q[k])]
                    to_del.append(noTarget[np.argmax(q[k])])
            for d in to_del:
                noTarget.remove(d)
    # check permutations of nearby neurons
    while np.any(np.isnan(mapIdx)) and len(noTarget):
        nanIdx = np.where(np.isnan(mapIdx))[0]
        block = filter(lambda b: nanIdx[0] in b, blocks)[0]
        idx = list(block.intersection(nanIdx))  # ground truth indices
        candidates = list([np.argmax(corT[i, noTarget])
                           for i in idx])  # inferred indices
        if len(candidates) == len(set(candidates)):
            # the easier part: neurons within the group of nearby ones are
            # highly correlated with different inferred neurons
            for i in idx:
                k = np.argmax(corT[i, noTarget])
                mapIdx[i] = noTarget[k]
                del noTarget[k]
        else:
            # the tricky part: neurons within the group of nearby ones are
            # highly correlated with the same inferred neurons
            candidates = list(
                set(np.concatenate([np.argsort(corT[i, noTarget])[-2:] for i in idx])))
            bestcorr = -np.inf
            for perm in itertools.permutations(candidates):
                perm = list(perm)
                c = np.diag(corT[idx][:, perm[:len(idx)]]).sum()
                if c > bestcorr:
                    bestcorr = c
                    bestperm = perm
            mapIdx[list(idx)] = bestperm[:len(idx)]
            for d in bestperm[:len(idx)]:
                noTarget.remove(d)
    return mapIdx


def plot_centers(inferredA, trueA):
    tc = np.array([center_of_mass(a.reshape(dims_in, order='F'))
                   for a in trueA.T])
    if not whole_FOV:
        tc -= dims
    center = [center_of_mass(a.reshape(dims, order='F'))
              for a in inferredA.toarray().T]
    plt.figure(figsize=(15, 15))
    if whole_FOV:
        plt.imshow(A.sum(-1).reshape(dims_in, order='F'))
    else:
        plt.imshow(A.sum(-1).reshape(dims_in, order='F')
                   [dims[0]:2 * dims[0], dims[1]:2 * dims[1]])
        plt.xlim(0, dims[0])
        plt.ylim(0, dims[1])
    plt.scatter(*np.transpose(tc)[::-1], marker='x',
                lw=3, s=100, c='r', label='true centers')
    plt.scatter(*np.transpose(center)[::-1], c='w', label='inferred centers')
    plt.legend()


#%%
fname = 'test_sim.mat'
test_sim = loadmat(fname)
(A, C, b, A_cnmfe, f, C_cnmfe, Craw_cnmfe, b0, sn, Yr, S_cnmfe,
    A_cnmfe_patch, C_cnmfe_patch, Craw_cnmfe_patch) = itemgetter(
    'A', 'C', 'b', 'A_cnmfe', 'f', 'C_cnmfe', 'Craw_cnmfe', 'b0', 'sn', 'Y', 'S_cnmfe',
    'A_cnmfe_patch', 'C_cnmfe_patch', 'Craw_cnmfe_patch')(test_sim)
N, T = C.shape
dims_in = (253, 316)
Y = Yr.T.reshape((-1,) + dims_in, order='F')

# cm.movie(Y).play(fr=30, magnification=2)


gSig = 3   # gaussian width of a 2D gaussian kernel, which approximates a neuron
gSiz = 10  # average diameter of a neuron
min_corr = .9
min_pnr = 15
# If True, the background can be roughly removed. This is useful when the background is strong.
center_psf = True
K = 200


#%%
whole_FOV = True
if whole_FOV:
    fname_new = cm.save_memmap([Y], base_name='Yr')
    dims = dims_in
else:

    fname_new = cm.save_memmap([Y], base_name='Yr',
                               idx_xy=(slice(120, 2 * 120), slice(120, 2 * 120)))
    dims = (120, 120)

Yr, dims, T = cm.load_memmap(fname_new)
Y = Yr.T.reshape((T,) + dims, order='F')

cn_filter, pnr = cm.summary_images.correlation_pnr(
    Y, gSig=gSig, center_psf=center_psf, swap_dim=False)


#%%
try:
    dview.terminate()
except:
    pass
c, dview, n_processes = cm.cluster.setup_cluster(
    backend='local', n_processes=None, single_thread=False)


#%%
patches = True
if patches:
    cnm = cnmf.CNMF(n_processes=n_processes, method_init='corr_pnr', k=50,
                    gSig=(3, 3), gSiz=(10, 10), merge_thresh=.7, p=1, dview=dview,
                    tsub=1, ssub=1, Ain=None, rf=(50, 50), stride=(32, 32), only_init_patch=True,
                    gnb=16, nb_patch=16, method_deconvolution='oasis', low_rank_background=True,
                    update_background_components=False, min_corr=min_corr, min_pnr=min_pnr,
                    normalize_init=False, deconvolve_options_init=None,
                    ring_size_factor=1.5, center_psf=True, del_duplicates=True)
else:
    cnm = cnmf.CNMF(n_processes=n_processes, method_init='corr_pnr', k=None,
                    gSig=(gSig, gSig), gSiz=(gSiz, gSiz), merge_thresh=.7, p=1, dview=None,
                    tsub=1, ssub=1, Ain=None, only_init_patch=True,
                    gnb=16, nb_patch=16, method_deconvolution='oasis', low_rank_background=False,
                    update_background_components=False, min_corr=min_corr, min_pnr=min_pnr,
                    normalize_init=False, deconvolve_options_init=None,
                    ring_size_factor=1.5, center_psf=True)

cnm.fit(Y)

# %% DISCARD LOW QUALITY COMPONENT
final_frate = 10
r_values_min = 0.9  # threshold on space consistency
fitness_min = -250 if patches else -80  # threshold on time variability
# threshold on time variability (if nonsparse activity)
fitness_delta_min = -250 if patches else -80
Npeaks = 5
traces = cnm.C + cnm.YrA
# TODO: todocument
idx_components, idx_components_bad = cm.components_evaluation.estimate_components_quality(
    traces, Yr, cnm.A, cnm.C, cnm.b, cnm.f, final_frate=final_frate, Npeaks=Npeaks,
    r_values_min=r_values_min, fitness_min=fitness_min, fitness_delta_min=fitness_delta_min, dview=dview)

print(('Keeping ' + str(len(idx_components)) +
       ' and discarding  ' + str(len(idx_components_bad))))


#%%
A_, C_, YrA_, b_, f_ = (cnm.A[:, idx_components], cnm.C[idx_components],
                        cnm.YrA[idx_components], cnm.b, cnm.f)
#%%
YrA_GT = compute_residuals(np.array(Yr) - b0, A, b, C, f, dview=None)

cm.utils.visualization.view_patches_bar(Yr, A, C, b, f,
                                        dims[0], dims[1], YrA=YrA_GT, img=cn_filter)

#%%
cm.utils.visualization.view_patches_bar(Yr, A_, C_, b_, f_,
                                        dims[0], dims[1], YrA=YrA_, img=cn_filter)
#%%

mapIdx = get_mapping(C_, C, A).astype(int)
if True:
    corC = np.array([np.corrcoef(C_[mapIdx[n]], C[n])[0, 1] for n in range(N)])
    corA = np.array([np.corrcoef(A_[:, mapIdx[n]].toarray().squeeze(), A[:, n])[0, 1]
                     for n in range(N)])
    corC_cnmfe = np.array([np.corrcoef(C_cnmfe[n], C[n])[0, 1]
                           for n in range(N)])
    corA_cnmfe = np.array(
        [np.corrcoef(A_cnmfe.toarray()[:, n], A[:, n])[0, 1] for n in range(N)])
    corC_cnmfe_patch = np.array(
        [np.corrcoef(C_cnmfe_patch[n], C[n])[0, 1] for n in range(N)])
    corA_cnmfe_patch = np.array(
        [np.corrcoef(A_cnmfe_patch.toarray()[:, n], A[:, n])[0, 1] for n in range(N)])

else:
    corC = np.array([np.corrcoef(C_[mapIdx[n]] + YrA_[mapIdx[n]],
                                 C[n] + YrA_GT[n])[0, 1] for n in range(N)])
    corA = np.array([np.corrcoef(A_[:, mapIdx[n]].toarray().squeeze(), A[:, n])[0, 1]
                     for n in range(N)])
    corC_cnmfe = np.array(
        [np.corrcoef(Craw_cnmfe[n], C[n] + YrA_GT[n])[0, 1] for n in range(N)])
    corA_cnmfe = np.array(
        [np.corrcoef(A_cnmfe.toarray()[:, n], A[:, n])[0, 1] for n in range(N)])
    corC_cnmfe_patch = np.array(
        [np.corrcoef(Craw_cnmfe_patch[n], C[n] + YrA_GT[n])[0, 1] for n in range(N)])
    corA_cnmfe_patch = np.array(
        [np.corrcoef(A_cnmfe_patch.toarray()[:, n], A[:, n])[0, 1] for n in range(N)])


print(np.median(corC), np.median(corA))
print(np.median(corC_cnmfe), np.median(corA_cnmfe))
print(np.median(corC_cnmfe_patch), np.median(corA_cnmfe_patch))
#%%
crd = cm.utils.visualization.plot_contours(A_, cn_filter, thr=.95, vmax=0.95)
plot_centers(A_, A)
#%%
cm.stop_server(dview=dview)
