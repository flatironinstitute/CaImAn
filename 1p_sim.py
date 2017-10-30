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
import caiman as cm
from caiman.source_extraction import cnmf


test_sim = loadmat('test_sim.mat')
A, C, b, A_cnmfe, f, C_cnmfe, Craw_cnmfe, b0, sn, Yr, S_cnmfe = itemgetter(
    'A', 'C', 'b', 'A_cnmfe', 'f', 'C_cnmfe',
    'Craw_cnmfe', 'b0', 'sn', 'Y', 'S_cnmfe')(test_sim)
dims = (253, 316)
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
c, dview, n_processes = cm.cluster.setup_cluster(
    backend='local', n_processes=None, single_thread=False)


#%%
# cnm = cnmf.CNMF(n_processes=n_processes, method_init='corr_pnr', k=25, gSig=(3, 3), gSiz=(10, 10), merge_thresh=.8,
#                 p=1, dview=dview, tsub=1, ssub=1, Ain=None, rf=(25, 25), stride=(25, 25),
#                 only_init_patch=True, gnb=10, nb_patch=6, method_deconvolution='oasis',
#                 low_rank_background=False, update_background_components=False, min_corr=min_corr,
#                 min_pnr=min_pnr, normalize_init=False, deconvolve_options_init=None,
#                 ring_size_factor=1.5, center_psf=True)

cnm = cnmf.CNMF(n_processes=n_processes, method_init='corr_pnr', k=K, gSig=(gSig, gSig), gSiz=(gSiz, gSiz),
                merge_thresh=.8, p=1, dview=dview, tsub=1, ssub=1, Ain=None,
                only_init_patch=True, gnb=10, nb_patch=6, method_deconvolution='oasis',
                low_rank_background=False, update_background_components=False, min_corr=min_corr,
                min_pnr=min_pnr, normalize_init=False, deconvolve_options_init=None,
                ring_size_factor=1.5, center_psf=True)


#%%
fname_new = cm.save_memmap([Y], base_name='Yr')
Yr, dims, T = cm.load_memmap(fname_new)
cnm.fit(Yr.T.reshape((T,) + dims, order='F'))


#%%
cn_filter, pnr = cm.summary_images.correlation_pnr(
    Y, gSig=gSig, center_psf=center_psf, swap_dim=False)
crd = cm.utils.visualization.plot_contours(cnm.A, cn_filter, thr=.95, vmax=0.95)
plt.imshow(cnm.A.sum(-1).reshape(dims, order='F'))


# mapping of neuron indices to ground truth indices
N, T = C.shape
mapIdx = np.nan * np.zeros(N, dtype='uint8')
corT = np.asarray([[np.corrcoef(s, tC)[0, 1]
                    for s in cnm.C] for tC in C])
noTarget = list(range(N))
for _ in range(10):
    if np.any(np.isnan(mapIdx)):
        nanIdx = np.where(np.isnan(mapIdx))[0]
        q = corT[np.isnan(mapIdx)][:, noTarget]
        to_del = []
        for k in range(len(q)):
            if np.argmax(q[:, np.argmax(q[k])]) == k:  # mutually highest correlation
                mapIdx[nanIdx[k]] = noTarget[np.argmax(q[k])]
                to_del.append(noTarget[np.argmax(q[k])])
        for d in to_del:
            noTarget.remove(d)
mapIdx = mapIdx.astype(int)

corC = np.array([np.corrcoef(cnm.C[mapIdx[n]], C[n])[0, 1] for n in range(N)])
corA = np.array([np.corrcoef(cnm.A[:, mapIdx[n]], A[:, n])[0, 1] for n in range(N)])

corC_cnmfe = np.array([np.corrcoef(C_cnmfe[n], C[n])[0, 1] for n in range(N)])
corA_cnmfe = np.array([np.corrcoef(A_cnmfe.toarray()[:, n], A[:, n])[0, 1] for n in range(N)])

print(corC.mean(), corA.mean())
print(corC_cnmfe.mean(), corA_cnmfe.mean())


tc = [center_of_mass(a.reshape(dims, order='F')) for a in A.T]
center = [center_of_mass(a.reshape(dims, order='F')) for a in cnm.A.T]
plt.figure(figsize=(15, 15))
plt.imshow(A.sum(-1).reshape(dims, order='F'))
plt.scatter(*np.transpose(tc)[::-1], marker='x', s=90, c='k', label='true centers')
plt.scatter(*np.transpose(center)[::-1], c='w', label='inferred centers')
plt.legend()
