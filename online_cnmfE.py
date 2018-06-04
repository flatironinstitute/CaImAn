try:
    get_ipython().magic(u'load_ext autoreload')
    get_ipython().magic(u'autoreload 2')
except:
    print('NOT IPYTHON')

import caiman as cm
from caiman.source_extraction import cnmf
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np

gSig = 3   # gaussian width of a 2D gaussian kernel, which approximates a neuron
gSiz = 13  # average diameter of a neuron
min_corr = .7
min_pnr = 7
s_min = -10
initbatch = 3000
expected_comps = 100

Yr, dims, T = cm.load_memmap('Yr_d1_64_d2_64_d3_1_order_C_frames_6000_.mmap')
Y = Yr.T.reshape((T,) + dims, order='F')


#%% RUN (offline) CNMF algorithm on the entire batch for sake of comparison

cnm_batch = cnmf.CNMF(2, method_init='corr_pnr', k=None, gSig=(gSig, gSig), gSiz=(gSiz, gSiz),
                merge_thresh=.65, p=1, tsub=1, ssub=1, only_init_patch=True, gnb=0,
                min_corr=min_corr, min_pnr=min_pnr, normalize_init=False, ring_size_factor=1.4,
                center_psf=True, ssub_B=1, init_iter=1, s_min=s_min)

cnm_batch.fit(Y)

print(('Number of components:' + str(cnm_batch.A.shape[-1])))

Cn = cm.local_correlations(Y.transpose(1, 2, 0))
plt.figure()
crd = cm.utils.visualization.plot_contours(cnm_batch.A, Cn, thr=.6)


#%% RUN (offline) CNMF algorithm on the initial batch

cnm_init = cnmf.CNMF(2, method_init='corr_pnr', k=None, gSig=(gSig, gSig), gSiz=(gSiz, gSiz),
                merge_thresh=.65, p=1, tsub=1, ssub=1, only_init_patch=True, gnb=0,
                min_corr=min_corr, min_pnr=min_pnr, normalize_init=False, ring_size_factor=1.4,
                center_psf=True, ssub_B=1, init_iter=1, s_min=s_min,
                minibatch_shape=100, minibatch_suff_stat=5, update_num_comps=False)

cnm_init.fit(Y[:initbatch])

print(('Number of components:' + str(cnm_init.A.shape[-1])))

Cn_init = cm.local_correlations(Y[:initbatch].transpose(1, 2, 0))
plt.figure()
crd = cm.utils.visualization.plot_contours(cnm_init.A, Cn_init, thr=.6)


#%% run (online) OnACID algorithm

cnm = deepcopy(cnm_init)
cnm._prepare_object(np.asarray(Yr[:, :initbatch]), T, expected_comps)
t = cnm.initbatch

for frame in Y[initbatch:]:
    cnm.fit_next(t, frame.copy().reshape(-1, order='F'))
    t += 1

print(('Number of components:' + str(cnm.Ab.shape[-1])))

Cn = cm.local_correlations(Y.transpose(1, 2, 0))
plt.figure()
crd = cm.utils.visualization.plot_contours(cnm.Ab, Cn, thr=.6)
