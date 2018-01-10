#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on Wed Aug 10 12:35:08 2016

@author: agiovann
"""

#%%
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')
from scipy.io import loadmat
import numpy as np
from PIL import Image
import calblitz as cb
import pylab as pl
import ca_source_extraction as cse
import os
base_folder = '/mnt/ceph/neuro/labeling/k31_20151223_AM_150um_65mW_zoom2p2/'
#%%


def extract_sue_ann_info(fname, base_folder):
    matvar = loadmat('./regions/sue_ann_regions.mat')
    idx = -1
    for counter, nm in enumerate(a['roiFile'][0]):
        if nm[0][0] in os.path.split(os.path.dirname(base_folder))[-1]:
            idx = counter
            A = matvar['A_s'][0][idx]
            A_init = matvar['A_ins'][0][idx]
            C = matvar['C_s'][0][idx]
            template = matvar['templates'][0][idx]
            idx_shapes = matvar['init'][0][idx]
            idx_global = matvar['globalId'][0][idx]

    if idx < 0:
        raise Exception('Matching name not found!')

    return A, C, template, idx_shapes, A_init


#%%
img_corr = cb.load(base_folder + 'projections/correlation_image.tif', fr=1)
img_median = cb.load(base_folder + 'projections/median_projection.tif', fr=1)
shape = np.shape(img_median)

pl.close()
if os.path.exist(base_folder + 'regions/princeton_regions.mat'):
    a = loadmat(base_folder + 'regions/princeton_regions.mat')
    try:
        rois_1 = a['allROIs']
    except:
        rois_1 = a['M']
elif os.path.exist(base_folder + 'regions/sue_ann_regions.mat'):
    A, C, template, idx_shapes, A_in = extract_sue_ann_info(
        base_folder + 'regions/sue_ann_regions.mat', base_folder)
    rois_1 = np.reshape(A.todense(), (shape[0], shape[1], -1), order='F')

else:
    rois_1 = np.transpose(cse.utilities.nf_read_roi_zip(
        base_folder + 'regions/princeton_regions.zip', shape), [1, 2, 0])


if os.path.exist(base_folder + 'regions/ben_regions.mat'):
    b = loadmat(base_folder + 'regions/ben_regions.mat')
    rois_2 = b['M']
else:
    rois_1 = np.transpose(cse.utilities.nf_read_roi_zip(
        base_folder + 'regions/ben_regions.zip', shape), [1, 2, 0])

vmax_corr_perc = 95
vmin_corr_perc = 5

vmax_median_perc = 97
vmin_median_perc = 5

rois_1 = rois_1 * 1.
rois_2 = rois_2 * 1.
rois_1[rois_1 == 0] = np.nan
rois_2[rois_2 == 0] = np.nan

pl.figure(facecolor="white")
pl.subplot(2, 4, 1)
pl.imshow(img_corr, cmap='gray', vmax=np.percentile(
    img_corr, vmax_corr_perc), vmin=np.percentile(img_corr, vmin_corr_perc))
pl.imshow(np.nanmax(rois_1, -1), cmap='ocean', vmax=2, alpha=.2, vmin=0)
pl.ylabel('CORR IMAGE')
pl.title('PRINCETON')
pl.axis('off')
pl.subplot(2, 4, 2)
pl.imshow(img_corr, cmap='gray', vmax=np.percentile(
    img_corr, vmax_corr_perc), vmin=np.percentile(img_corr, vmin_corr_perc))
pl.imshow(np.nanmax(rois_2, -1), cmap='hot', alpha=.2, vmin=0, vmax=3)
pl.title('BEN')
pl.axis('off')
pl.subplot(2, 4, 3)
pl.imshow(img_corr, cmap='gray', vmax=np.percentile(
    img_corr, vmax_corr_perc), vmin=np.percentile(img_corr, vmin_corr_perc))
pl.axis('off')
pl.subplot(2, 4, 5)
pl.imshow(img_median, cmap='gray', vmax=np.percentile(img_median,
                                                      vmax_median_perc), vmin=np.percentile(img_median, vmin_median_perc))
pl.imshow(np.nanmax(rois_1, -1), cmap='ocean', vmax=2, alpha=.2, vmin=0)
pl.ylabel('MEDIAN')
pl.axis('off')
pl.subplot(2, 4, 6)
pl.imshow(img_median, cmap='gray', vmax=np.percentile(img_median,
                                                      vmax_median_perc), vmin=np.percentile(img_median, vmin_median_perc))
pl.imshow(np.nanmax(rois_2, -1), cmap='hot', alpha=.2, vmin=0, vmax=3)
pl.axis('off')
pl.subplot(2, 4, 7)
pl.imshow(img_median, cmap='gray', vmax=np.percentile(img_median,
                                                      vmax_median_perc), vmin=np.percentile(img_median, vmin_median_perc))
pl.axis('off')
pl.subplot(2, 4, 4)

pl.imshow(np.nanmax(rois_1, -1), cmap='ocean', vmax=2, alpha=.5, vmin=0)
pl.imshow(np.nanmax(rois_2, -1), cmap='hot', alpha=.5, vmin=0, vmax=3)
pl.axis('off')

pl.subplot(2, 4, 8)
pl.axis('off')
pl.imshow(img_corr, cmap='gray', vmax=np.percentile(
    img_corr, vmax_corr_perc), vmin=np.percentile(img_corr, vmin_corr_perc))
pl.imshow(np.nanmax(rois_1, -1), cmap='ocean', vmax=2, alpha=.3, vmin=0)
pl.imshow(np.nanmax(rois_2, -1), cmap='hot', alpha=.3, vmin=0, vmax=3)
pl.axis('off')


font = {'family': 'Myriad Pro',
        'weight': 'regular',
        'size': 30}

pl.rc('font', **font)
#%%
pl.rcParams['pdf.fonttype'] = 42
pl.savefig(base_folder + 'comparison.pdf')
