try:
    get_ipython().magic(u'load_ext autoreload')
    get_ipython().magic(u'autoreload 2')
except:
    print('NOT IPYTHON')

import numpy as np
from scipy.io import loadmat
from operator import itemgetter
import matplotlib.pyplot as plt
from scipy.ndimage import center_of_mass
import itertools
import caiman as cm
from caiman.source_extraction import cnmf


test_sim = loadmat('test_sim.mat')
A, C, b, A_cnmfe, f, C_cnmfe, Craw_cnmfe, b0, sn, Yr, S_cnmfe = itemgetter(
    'A', 'C', 'b', 'A_cnmfe', 'f', 'C_cnmfe',
    'Craw_cnmfe', 'b0', 'sn', 'Y', 'S_cnmfe')(test_sim)
dims = (253, 316)
N, T = C.shape
Y = Yr.T.reshape((-1,) + dims, order='F')

plt.imshow(Y[0])
cm.movie(Y).play(fr=30, magnification=2)
plt.figure(figsize=(20, 4))
plt.plot(C.T)


gSig = 3   # gaussian width of a 2D gaussian kernel, which approximates a neuron
gSiz = 10  # average diameter of a neuron
min_corr = .9
min_pnr = 15
# If True, the background can be roughly removed. This is useful when the background is strong.
center_psf = True
K = 200


#%%
fname_new = cm.save_memmap([Y], base_name='Yr')
Yr, dims, T = cm.load_memmap(fname_new)

cn_filter, pnr = cm.summary_images.correlation_pnr(
    Y, gSig=gSig, center_psf=center_psf, swap_dim=False)


#%%
c, dview, n_processes = cm.cluster.setup_cluster(
    backend='local', n_processes=None, single_thread=False)


#%%  no patches
cnm = cnmf.CNMF(n_processes=n_processes, method_init='corr_pnr', k=K,
                gSig=(gSig, gSig), gSiz=(gSiz, gSiz),
                merge_thresh=.8, p=1, dview=dview, tsub=1, ssub=1, Ain=None,
                only_init_patch=True, gnb=10, nb_patch=6, method_deconvolution='oasis',
                low_rank_background=False, update_background_components=False, min_corr=min_corr,
                min_pnr=min_pnr, normalize_init=False, deconvolve_options_init=None,
                ring_size_factor=1.5, center_psf=True)

cnm.fit(Yr.T.reshape((T,) + dims, order='F'))

crd = cm.utils.visualization.plot_contours(cnm.A, cn_filter, thr=.95, vmax=0.95)
plt.imshow(cnm.A.sum(-1).reshape(dims, order='F'))


# mapping of neuron indices to ground truth indices
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
    noTarget = list(range(len(inferredC)))  # indices that haven't been a target of the mapping yet
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
        candidates = list([np.argmax(corT[i, noTarget]) for i in idx])  # inferred indices
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


def plot_centers():
    tc = [center_of_mass(a.reshape(dims, order='F')) for a in A.T]
    center = [center_of_mass(a.reshape(dims, order='F')) for a in cnm.A.toarray().T]
    plt.figure(figsize=(15, 15))
    plt.imshow(A.sum(-1).reshape(dims, order='F'))
    plt.scatter(*np.transpose(tc)[::-1], marker='x', lw=3, s=100, c='r', label='true centers')
    plt.scatter(*np.transpose(center)[::-1], c='w', label='inferred centers')
    plt.legend()


mapIdx = get_mapping(cnm.C, C, A).astype(int)

corC = np.array([np.corrcoef(cnm.C[mapIdx[n]], C[n])[0, 1] for n in range(N)])
corA = np.array([np.corrcoef(cnm.A.toarray()[:, mapIdx[n]], A[:, n])[0, 1] for n in range(N)])

corC_cnmfe = np.array([np.corrcoef(C_cnmfe[n], C[n])[0, 1] for n in range(N)])
corA_cnmfe = np.array([np.corrcoef(A_cnmfe.toarray()[:, n], A[:, n])[0, 1] for n in range(N)])

print(corC.mean(), corA.mean())
print(corC_cnmfe.mean(), corA_cnmfe.mean())

plot_centers()


#%% patches
cnm = cnmf.CNMF(n_processes=n_processes, method_init='corr_pnr', k=120,
                gSig=(gSig, gSig), gSiz=(gSiz, gSiz), merge_thresh=.7,
                p=1, dview=dview, tsub=1, ssub=2, Ain=None, rf=(100, 100), stride=(25, 25),
                only_init_patch=True, gnb=10, nb_patch=6, method_deconvolution='oasis',
                low_rank_background=False, update_background_components=False, min_corr=min_corr,
                min_pnr=min_pnr, normalize_init=False, deconvolve_options_init=None,
                ring_size_factor=1.5, center_psf=True, del_duplicates=True)

cnm.fit(Yr.T.reshape((T,) + dims, order='F'))

mapIdx = get_mapping(cnm.C, C, A).astype(int)

corC = np.array([np.corrcoef(cnm.C[mapIdx[n]], C[n])[0, 1] for n in range(N)])
corA = np.array([np.corrcoef(cnm.A.toarray()[:, mapIdx[n]], A[:, n])[0, 1] for n in range(N)])

corC_cnmfe = np.array([np.corrcoef(C_cnmfe[n], C[n])[0, 1] for n in range(N)])
corA_cnmfe = np.array([np.corrcoef(A_cnmfe.toarray()[:, n], A[:, n])[0, 1] for n in range(N)])

print(corC.mean(), corA.mean())
print(corC_cnmfe.mean(), corA_cnmfe.mean())

plot_centers()
