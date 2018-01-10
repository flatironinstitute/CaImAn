#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on Thu Sep  7 13:29:49 2017

@author: agiovann
"""

#%%
try:
    get_ipython().magic(u'load_ext autoreload')
    get_ipython().magic(u'autoreload 2')
except:
    print('Not launched under iPython')

import numpy as np
import matplotlib.pyplot as pl
from matplotlib.widgets import Slider
from scipy.sparse import coo_matrix
import caiman as cm
from caiman.source_extraction import cnmf
from caiman.utils.visualization import inspect_correlation_pnr
import scipy

#%%
#fname = './example_movies/data_endoscope.tif'
# gSig = [3,3]   # gaussian width of a 2D gaussian kernel, which approximates a neuron
# gSiz = [10,10]  # average diameter of a neuron
#min_corr = .8
#min_pnr = 10

#fname = '/opt/local/Data/1photon/3168_PAG_TIFF.tif'
fname = '/opt/local/Data/1photon/Yr_d1_190_d2_198_d3_1_order_F_frames_35992_.mmap'
gSig = [3, 3]   # gaussian width of a 2D gaussian kernel, which approximates a neuron
gSiz = [16, 16]  # average diameter of a neuron
min_corr = .6
min_pnr = 10

# If True, the background can be roughly removed. This is useful when the background is strong.
center_psf = True


if 'mmap' in fname:
    Yr, dims, T = cm.load_memmap(fname)
    Y = Yr.T.reshape((T,) + dims, order='F')
else:
    Y = cm.load(fname)
T, d1, d2 = Y.shape
print('The dimension of data is ', Y.shape)

ax = pl.axes()
ax.axis('off')
pl.imshow(Y[100])
#%%
# show correlation image of the raw data; show correlation image and PNR image of the filtered data
cn_raw = cm.summary_images.max_correlation_image(
    Y, swap_dim=False, bin_size=3000)
#%% TAKES MEMORY!!!
cn_filter, pnr = cm.summary_images.correlation_pnr(
    Y, gSig=gSig, center_psf=center_psf, swap_dim=False)

#%%
pl.figure(figsize=(10, 5))
for i, (data, title) in enumerate(((Y.mean(0), 'Mean image (raw)'),
                                   (Y.max(0), 'Max projection (raw)'),
                                   (cn_raw[1:-1, 1:-1], 'Correlation (raw)'),
                                   (cn_filter, 'Correlation (filtered)'),
                                   (pnr, 'PNR (filtered)'),
                                   (cn_filter * pnr, 'Correlation*PNR (filtered)'))):
    pl.subplot(2, 3, 1 + i)
    pl.imshow(data, cmap='jet', aspect='equal')
    pl.axis('off')
    pl.colorbar()
    pl.title(title)


#%%
# pick thresholds
inspect_correlation_pnr(cn_filter, pnr)

#%% start cluster
try:
    dview.terminate()
    dview = None
except:
    pass
c, dview, n_processes = cm.cluster.setup_cluster(
    backend='local', n_processes=None, single_thread=False)
#%%
cnm = cnmf.CNMF(n_processes=n_processes, method_init='corr_pnr', k=10,
                gSig=gSig, gSiz=gSiz, merge_thresh=.8, p=1, dview=dview,
                tsub=1, ssub=1, Ain=None, rf=(50, 50), stride=(32, 32), only_init_patch=True,
                gnb=16, nb_patch=16, method_deconvolution='oasis', low_rank_background=True,
                update_background_components=True, min_corr=min_corr, min_pnr=min_pnr,
                normalize_init=False, deconvolve_options_init=None,
                ring_size_factor=1.5, center_psf=center_psf, del_duplicates=True)

cnm.fit(Y)
# %% DISCARD LOW QUALITY COMPONENT
final_frate = 10
r_values_min = 0.9  # threshold on space consistency
fitness_min = -100  # threshold on time variability
# threshold on time variability (if nonsparse activity)
fitness_delta_min = - 100
Npeaks = 5
traces = cnm.C + cnm.YrA
# TODO: todocument
idx_components, idx_components_bad = cm.components_evaluation.estimate_components_quality(
    traces, Yr, cnm.A, cnm.C, cnm.b, cnm.f, final_frate=final_frate, Npeaks=Npeaks,
    r_values_min=r_values_min, fitness_min=fitness_min, fitness_delta_min=fitness_delta_min, dview=dview)

print(('Keeping ' + str(len(idx_components)) +
       ' and discarding  ' + str(len(idx_components_bad))))


#%%
A_, C_, YrA_, b_, f_ = cnm.A[:,
                             idx_components], cnm.C[idx_components], cnm.YrA[idx_components], cnm.b, cnm.f
#%%
pl.figure()
crd = cm.utils.visualization.plot_contours(
    A_.tocsc()[:, idx_components], cn_filter, thr=.9)

#%%
pl.imshow(A_.sum(-1).reshape(dims, order='F'), vmax=200)
#%%
idx_components = np.arange(A_.shape[-1])
cm.utils.visualization.view_patches_bar(
    Yr, coo_matrix(A_.tocsc()[:, idx_components]), C_[idx_components],
    b_, f_, dims[0], dims[1], YrA=YrA_[idx_components], img=cn_filter)
