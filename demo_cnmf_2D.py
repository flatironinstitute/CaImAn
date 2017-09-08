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
Y = cm.load(fname)
T, d1, d2 = Y.shape
print('The dimension of data is ',Y.shape)

ax = plt.axes()
ax.axis('off')
show_img(ax, Y[100,])
#%%
# parameters 
gSig = 3   # gaussian width of a 2D gaussian kernel, which approximates a neuron 
gSiz = 10  # average diameter of a neuron 
center_psf = True     # If True, the background can be roughly removed. This is useful when the background is strong. 

# show correlation image of the raw data; show correlation image and PNR image of the filtered data
cn_raw = cm.summary_images.local_correlations_fft(Y, swap_dim=False)
cn_filter, pnr = cm.summary_images.correlation_pnr(Y, gSig=gSig, center_psf=center_psf, swap_dim=False)
plt.figure(figsize=(10, 5))

for i, (data, title) in enumerate(((Y.mean(0), 'Mean image (raw)'),
                                   (Y.max(0), 'Max projection (raw)'),
                                   (cn_raw[1:-1,1:-1], 'Correlation (raw)'),
                                   (cn_filter, 'Correlation (filtered)'),
                                   (pnr, 'PNR (filtered)'),
                                   (cn_filter*pnr, 'Correlation*PNR (filtered)'))):
    plt.subplot(2,3,1+i)
    plt.imshow(data, cmap='jet', aspect='equal')
    plt.axis('off')
    plt.colorbar() 
    plt.title(title);
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
plt.colorbar();

s_cn_max = Slider(plt.axes([0.05, 0.01, 0.35, 0.03]), 'vmax', cn_filter.min(), cn_filter.max(), valinit=cn_filter.max())
s_cn_min = Slider(plt.axes([0.05, 0.07, 0.35, 0.03]), 'vmin', cn_filter.min(), cn_filter.max(), valinit=cn_filter.min())
s_pnr_max = Slider(plt.axes([0.5, 0.01, 0.35, 0.03]), 'vmax', pnr.min(), pnr.max(), valinit=pnr.max())
s_pnr_min = Slider(plt.axes([0.5, 0.07, 0.35, 0.03]), 'vmin', pnr.min(), pnr.max(), valinit=pnr.min())

def update(val):
    im_cn.set_clim([s_cn_min.val, s_cn_max.val])
    im_pnr.set_clim([s_pnr_min.val, s_pnr_max.val])
    fig.canvas.draw_idle()
s_cn_max.on_changed(update)
s_cn_min.on_changed(update)
s_pnr_max.on_changed(update)
s_pnr_min.on_changed(update)

#%%
cnm = cnmf.CNMF(n_processes = 2, method_init='corr_pnr', k=10, gSig=(3,3), gSiz = (10,10), merge_thresh=.8,
                p=1, dview=None, tsub=1, ssub=1, Ain=None, rf=(15,15), stride=(10,10),
                only_init_patch=True, gnb=10, nb_patch=3, method_deconvolution='oasis', 
                low_rank_background=False, update_background_components=False, min_corr = .8, 
                min_pnr = 10, normalize_init = False, deconvolve_options_init = None, 
                ring_size_factor = 1.5, center_psf = True)


#%%
cnm = cnmf.CNMF(n_processes = 2, method_init='corr_pnr', k=155, gSig=(3,3), gSiz = (10,10), merge_thresh=.8,
                p=1, dview=None, tsub=1, ssub=1, Ain=None, rf=(64,64), stride=(0,0),
                only_init_patch=True, gnb=10, nb_patch=3, method_deconvolution='oasis', 
                low_rank_background=False, update_background_components=False, min_corr = .8, 
                min_pnr = 10, normalize_init = False, deconvolve_options_init = None, 
                ring_size_factor = 1.5, center_psf = True)

#%%
#cnm.options['init_params']['gSiz'] = (10, 10)
#cnm.options['init_params']['gSig'] = (3, 3)
#cnm.options['init_params']['min_corr'] = .85
#cnm.options['init_params']['min_pnr'] = 20
#cnm.options['init_params']['normalize_init']=False
#%%
memmap = True  # must be True for patches
if memmap:
    fname_new = cm.save_memmap([fname], base_name='Yr')
    Yr, dims, T = cm.load_memmap(fname_new)
    cnm.fit(Yr.T.reshape((T,) + dims, order='F'))
else:
    cnm.fit(Y)
#%%    
crd = cm.utils.visualization.plot_contours(cnm.A, cn_filter, thr=.99, vmax = 0.95)
#%%
plt.imshow(cnm.A.sum(-1).reshape(dims,order='F'))
