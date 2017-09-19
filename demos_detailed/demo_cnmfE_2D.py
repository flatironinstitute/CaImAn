#!/usr/bin/env python3
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
    print('NOT IPYTHON')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.sparse import csc_matrix
import caiman as cm
from caiman.source_extraction import cnmf
from caiman.source_extraction.cnmf.initialization import greedyROI_corr
import matplotlib as mpl
import matplotlib.pyplot as plt
import bokeh
import bokeh.plotting as bpl
from bokeh.models import CustomJS, ColumnDataSource, Range1d
from bokeh.io import output_notebook, reset_output
import os
import scipy


#%%
def show_img(ax, img):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.pyplot as plt

    im = ax.imshow(img)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    plt.colorbar(im, cax=cax)
#%%
fname = './example_movies/data_endoscope.tif'
gSig = 3   # gaussian width of a 2D gaussian kernel, which approximates a neuron
gSiz = 10  # average diameter of a neuron
min_corr=.8
min_pnr=10
#fname = '/opt/local/Data/1photon/3168_PAG_TIFF.tif'
#fname = '/opt/local/Data/1photon/Yr_d1_190_d2_198_d3_1_order_F_frames_35992_.mmap'
#gSig = 3   # gaussian width of a 2D gaussian kernel, which approximates a neuron
#gSiz = 16  # average diameter of a neuron
#min_corr=.6
#min_pnr=10

# If True, the background can be roughly removed. This is useful when the background is strong.
center_psf = True


Y = cm.load(fname)
T, d1, d2 = Y.shape
print('The dimension of data is ', Y.shape)

ax = plt.axes()
ax.axis('off')
show_img(ax, Y[100, ])
#%%
# show correlation image of the raw data; show correlation image and PNR image of the filtered data
cn_raw = cm.summary_images.max_correlation_image(Y, swap_dim=False,bin_size = 3000)
#%% TAKES MEMORY!!!
cn_filter, pnr = cm.summary_images.correlation_pnr(
    Y, gSig=gSig, center_psf=center_psf, swap_dim=False)
plt.figure(figsize=(10, 5))
#%%
for i, (data, title) in enumerate(((Y.mean(0), 'Mean image (raw)'),
                                   (Y.max(0), 'Max projection (raw)'),
                                   (cn_raw[1:-1, 1:-1], 'Correlation (raw)'),
                                   (cn_filter, 'Correlation (filtered)'),
                                   (pnr, 'PNR (filtered)'),
                                   (cn_filter * pnr, 'Correlation*PNR (filtered)'))):
    plt.subplot(2, 3, 1 + i)
    plt.imshow(data, cmap='jet', aspect='equal')
    plt.axis('off')
    plt.colorbar()
    plt.title(title)
#%%
# pick thresholds
fig = plt.figure(figsize=(10, 4))
plt.axes([0.05, 0.2, 0.4, 0.7])
im_cn = plt.imshow(cn_filter, cmap='jet')
plt.title('correlation image')
plt.colorbar()
plt.axes([0.5, 0.2, 0.4, 0.7])
im_pnr = plt.imshow(pnr, cmap='jet')
plt.title('PNR')
plt.colorbar()

s_cn_max = Slider(plt.axes([0.05, 0.01, 0.35, 0.03]), 'vmax',
                  cn_filter.min(), cn_filter.max(), valinit=cn_filter.max())
s_cn_min = Slider(plt.axes([0.05, 0.07, 0.35, 0.03]), 'vmin',
                  cn_filter.min(), cn_filter.max(), valinit=cn_filter.min())
s_pnr_max = Slider(plt.axes([0.5, 0.01, 0.35, 0.03]), 'vmax',
                   pnr.min(), pnr.max(), valinit=pnr.max())
s_pnr_min = Slider(plt.axes([0.5, 0.07, 0.35, 0.03]), 'vmin',
                   pnr.min(), pnr.max(), valinit=pnr.min())


def update(val):
    im_cn.set_clim([s_cn_min.val, s_cn_max.val])
    im_pnr.set_clim([s_pnr_min.val, s_pnr_max.val])
    fig.canvas.draw_idle()
    
s_cn_max.on_changed(update)
s_cn_min.on_changed(update)
s_pnr_max.on_changed(update)
s_pnr_min.on_changed(update)
#%%
c, dview, n_processes = cm.cluster.setup_cluster(backend='local', n_processes=None, single_thread=False)

#%%
cnm = cnmf.CNMF(n_processes=2, method_init='corr_pnr', k=10, gSig=(3, 3), gSiz=(10, 10), merge_thresh=.8,
                p=1, dview=dview, tsub=1, ssub=1, Ain=None, rf=(15, 15), stride=(10, 10),
                only_init_patch=True, gnb=10, nb_patch=3, method_deconvolution='oasis',
                low_rank_background=False, update_background_components=False, min_corr=min_corr,
                min_pnr=min_pnr, normalize_init=False, deconvolve_options_init=None,
                ring_size_factor=1.5, center_psf=True)

#%%
#cnm = cnmf.CNMF(n_processes=2, method_init='corr_pnr', k=155, gSig=(3, 3), gSiz=(10, 10), merge_thresh=.8,
#                p=1, dview=None, tsub=1, ssub=1, Ain=None, rf=(64, 64), stride=(0, 0),
#                only_init_patch=True, gnb=10, nb_patch=3, method_deconvolution='oasis',
#                low_rank_background=False, update_background_components=False, min_corr=.8,
#                min_pnr=10, normalize_init=False, deconvolve_options_init=None,
#                ring_size_factor=1.5, center_psf=True)

#%%
#cnm.options['init_params']['gSiz'] = (10, 10)
#cnm.options['init_params']['gSig'] = (3, 3)
#cnm.options['init_params']['min_corr'] = .85
#cnm.options['init_params']['min_pnr'] = 20
# cnm.options['init_params']['normalize_init']=False
#%%
memmap = True  # must be True for patches
if memmap:
    if '.mmap' in fname:
        fname_new = fname
    else:
        fname_new = cm.save_memmap([fname], base_name='Yr')
    Yr, dims, T = cm.load_memmap(fname_new)
    cnm.fit(Yr.T.reshape((T,) + dims, order='F'))
else:
    cnm.fit(Y)
#%%
A_tot, C_tot, b_tot, f_tot, YrA_tot, sn = cnm.A, cnm.C, cnm.b, cnm.f, cnm.YrA, cnm.sn
#%%
crd = cm.utils.visualization.plot_contours(A_tot, cn_filter, thr=.95, vmax=0.95)
#%%
plt.imshow(A_tot.sum(-1).reshape(dims, order='F'))
# %% DISCARD LOW QUALITY COMPONENT
final_frate = 10
r_values_min = 0.1  # threshold on space consistency
fitness_min = - 10  # threshold on time variability
# threshold on time variability (if nonsparse activity)
fitness_delta_min = - 10
Npeaks = 10
traces = C_tot + YrA_tot
# TODO: todocument
idx_components, idx_components_bad = cm.components_evaluation.estimate_components_quality(
    traces, Yr, A_tot, C_tot, b_tot, f_tot, final_frate=final_frate, Npeaks=Npeaks, r_values_min=r_values_min,
    fitness_min=fitness_min, fitness_delta_min=fitness_delta_min)

print(('Keeping ' + str(len(idx_components)) + ' and discarding  ' + str(len(idx_components_bad))))
#%%
# TODO: show screenshot 14
plt.subplot(1, 2, 1)
crd = cm.utils.visualization.plot_contours(A_tot.tocsc()[:, idx_components], cn_filter, thr=.95)
plt.subplot(1, 2, 2)
crd = cm.utils.visualization.plot_contours(
    A_tot.tocsc()[:, idx_components_bad], cn_filter, thr=.95)
#%%
cm.utils.visualization.view_patches_bar(
    Yr, scipy.sparse.coo_matrix(A_tot.tocsc()[:, idx_components]), C_tot[
        idx_components, :], b_tot, f_tot, dims[0], dims[1], YrA=YrA_tot[idx_components, :], img=cn_filter)
#%%
cm.utils.visualization.view_patches_bar(
    Yr, scipy.sparse.coo_matrix(A_tot.tocsc()[:, idx_components_bad]), C_tot[
        idx_components_bad, :], b_tot, f_tot, dims[0], dims[1], YrA=YrA_tot[idx_components_bad, :], img=cn_filter)

# %% rerun updating the components to refine
cnm = cnmf.CNMF(n_processes=1, k=A_tot.shape, gSig=[gSig, gSig], merge_thresh=0.8, p=1, dview=dview, Ain=A_tot,
                Cin=C_tot, b_in=b_tot,
                f_in=f_tot, rf=None, stride=None, method_deconvolution='oasis', gnb=None,
                low_rank_background=False, update_background_components=False)

memmap = True  # must be True for patches
if memmap:
    fname_new = cm.save_memmap([fname], base_name='Yr')
    Yr, dims, T = cm.load_memmap(fname_new)
    cnm.fit(Yr.T.reshape((T,) + dims, order='F'))
else:
    cnm.fit(Y)


#%%
A, C, b, f, YrA, sn = cnm.A, cnm.C, cnm.b, cnm.f, cnm.YrA, cnm.sn
#%%
import pylab as pl
idx_components = range(np.shape(A)[-1])
# TODO: show screenshot 14
pl.subplot(1, 2, 1)
crd = cm.utils.visualization.plot_contours(A.tocsc()[:, idx_components], cn_filter, thr=.95)
pl.subplot(1, 2, 2)
crd = cm.utils.visualization.plot_contours(A.tocsc()[:, idx_components_bad], cn_filter, thr=.95)
#%%
plt.imshow(A.sum(-1).reshape(dims, order='F'), vmax = 200)
#%%
idx_components = np.where(np.array([scipy.stats.pearsonr(c,c_)[0] for c,c_ in zip(C,C_tot)])*np.array([scipy.stats.pearsonr(c,c_)[0] for c,c_ in zip(A.T.toarray(),A_tot.T.toarray())])<.8)[0]
cm.utils.visualization.view_patches_bar(
    Yr, scipy.sparse.coo_matrix(A.tocsc()[:,idx_components]), C[idx_components], b, f, dims[0], dims[1], YrA=YrA[idx_components], img=cn_filter)
pl.figure()
cm.utils.visualization.view_patches_bar(
    Yr, scipy.sparse.coo_matrix(A_tot.tocsc()[:,idx_components]), C_tot[idx_components], b_tot, f_tot, dims[0], dims[1], YrA=YrA_tot[idx_components], img=cn_filter)
