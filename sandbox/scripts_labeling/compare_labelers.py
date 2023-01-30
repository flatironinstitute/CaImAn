#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 14:19:15 2017

Compare the performance of different labelers

@author: agiovann
"""
#%%
from __future__ import division
from __future__ import print_function
from builtins import zip
from builtins import str
from builtins import map
from builtins import range
from past.utils import old_div
import cv2

try:
    cv2.setNumThreads(1)
except:
    print('OpenCV is naturally single threaded')

try:
    if __IPYTHON__:
        print((1))
        # this is used for debugging purposes only. allows to reload classes when changed
        get_ipython().magic('load_ext autoreload')
        get_ipython().magic('autoreload 2')
except NameError:
    print('Not launched under iPython')

from caiman.base.rois import nf_read_roi_zip
import os
import numpy as np
import pylab as pl
import caiman as cm
from caiman.motion_correction import MotionCorrect
#%%
# files = {'/mnt/ceph/neuro/FromLabelers/To_be_labelled/Packer/',
#'/mnt/ceph/neuro/FromLabelers/To_be_labelled/Yuste/',
#'/mnt/ceph/neuro/FromLabelers/To_be_labelled/Neurofinder/01.01/',
#'/mnt/ceph/neuro/FromLabelers/To_be_labelled/Sue Ann/k37-20160109/',
#'/mnt/ceph/neuro/FromLabelers/To_be_labelled/Yi/Data 1/'}


params = [
    #['Jan25_2015_07_13',30,False,False,False], # fname, frate, do_rotate_template, do_self_motion_correct, do_motion_correct
    #['Jan40_exp2_001',30,False,False,False],
    #['Jan42_exp4_001',30,False,False,False],
    ['Jan-AMG1_exp2_new_001', 30, False, False, False],
    ['Jan-AMG_exp3_001', 30, False, False, False],
    ['Yi.data.001', 30, False, True, True],
    #['Yi.data.002',30,False,True,True],
    #['FN.151102_001',30,False,True,True],
    ['J115_2015-12-09_L01_ELS', 30, False, False, False],
    ['J123_2015-11-20_L01_0', 30, False, False, False],
    #['k26_v1_176um_target_pursuit_002_013',30,False,True,True],
    #['k31_20151223_AM_150um_65mW_zoom2p2',30,False,True,True],
    #['k31_20160104_MMA_150um_65mW_zoom2p2',30,False,True,True],
    #['k31_20160106_MMA_400um_118mW_zoom2p2_00001_1-19',30,True,True,True],
    #['k31_20160107_MMP_150um_65mW_zoom2p2_00001_1-15',30,True,True,True],
    #['k36_20151229_MMA_200um_65mW_zoom2p2_00001_1-17',30,True,True,True],
    #['k36_20160115_RSA_400um_118mW_zoom2p2_00001_20-38',30,True,True,True],
    #['k36_20160127_RL_150um_65mW_zoom2p2_00002_22-41',30,True,True,True],
    ['k37_20160109_AM_150um_65mW_zoom2p2_00001_1-16', 30, True, True, True],
    ['neurofinder.00.00', 7, False, False, False],
    #['neurofinder.00.01',7,False,False,False],
    #['neurofinder.00.02',7,False,False,False],
    #['neurofinder.00.03',7,False,False,False],
    #['neurofinder.00.04',7,False,False,False],
    #['neurofinder.01.00',7.5,False,False,False],
    ['neurofinder.01.01', 7.5, False, False, False],
    ['neurofinder.02.00', 8, False, False, False],
    #['neurofinder.03.00',6.75,False,False,False],
    ['neurofinder.03.00.test', 6.75, False, False, False],
    ['neurofinder.04.00', 6.75, False, False, False],
    ['neurofinder.04.00.test', 6.75, False, False, False],
    ['packer.001', 15, False, False, False],
    ['yuste.Single_150u', 10, False, False, False],
    ['k53_20160530', 10, False, False, False]
]


#%%
pl.close('all')
import itertools
# 0: with gt, 1: among themselves, 2: with all consensus, 3: with at least 2 consensus
compare_code = 0
label_name = ['regions/natalia_active_regions_nd.zip', 'regions/lindsey_active_regions_nd.zip',
              'regions/sonia_active_regions_nd.zip', 'regions/ben_active_regions_nd.zip']
#label_name = ['regions/intermediate_regions/natalia_all_regions.zip','regions/intermediate_regions/lindsey_all_regions.zip','regions/intermediate_regions/sonia_all_regions.zip','regions/intermediate_regions/ben_all_regions.zip']
consensus_name = 'regions/joined_consensus_active_regions.zip'
results = dict()
for count, par in enumerate(params):
    print(os.path.join('/mnt/ceph/neuro/labeling/',
                       par[0], 'projections/correlation_image.tif'))
    c_img = cm.load(os.path.join('/mnt/ceph/neuro/labeling/',
                                 par[0], 'projections/correlation_image.tif'))
    result = dict()
    if compare_code == 0:
        iterlabels = label_name
    elif compare_code == 1:
        iterlabels = itertools.combinations(label_name, 2)
    else:
        raise Exception('Not defined')

    for region_pairs in iterlabels:
        print(region_pairs)
        try:
            if compare_code == 0:
                roi_nat = nf_read_roi_zip(os.path.join(
                    '/mnt/ceph/neuro/labeling/', par[0], consensus_name), c_img.shape)
                roi_lin = nf_read_roi_zip(os.path.join(
                    '/mnt/ceph/neuro/labeling/', par[0], region_pairs), c_img.shape)

            else:
                roi_nat = nf_read_roi_zip(os.path.join(
                    '/mnt/ceph/neuro/labeling/', par[0], region_pairs[0]), c_img.shape)
                roi_lin = nf_read_roi_zip(os.path.join(
                    '/mnt/ceph/neuro/labeling/', par[0], region_pairs[1]), c_img.shape)

            print(roi_nat.shape)
            print(roi_lin.shape)
    #        roi_nat = nf_read_roi_zip(os.path.join('/mnt/ceph/neuro/labeling/',par[0],'regions/natalia_all_regions.zip'),c_img.shape)
    #        roi_lin = nf_read_roi_zip(os.path.join('/mnt/ceph/neuro/labeling/',par[0],'regions/lindsey_all_regions.zip'),c_img.shape)
        except:
            print('******************** PROBLEMS WITH ' + par[0])
            continue

    #
    #    pl.subplot(3,2,count+1)
    #    c_img[np.isnan(c_img )] = 0
    #    lq,hq=np.percentile(c_img,[5,95])
    #    c_img = (c_img-lq)/(hq-lq)
    # pl.imshow(c_img,cmap='gray',vmin=0.05,vmax=0.95)
    #    neur_nat = np.sum(roi_nat,0)
    #    neur_lin = np.sum(roi_lin,0)
    #    neur_nat[neur_nat == 0] = np.nan
    #    neur_lin[neur_lin == 0] = np.nan
    #    pl.imshow(neur_nat,cmap='hot',alpha=.3,vmin = 0, vmax = 3)
    #    pl.imshow(neur_lin,cmap='Greens',alpha=.3,vmin = 0,vmax = 3)
    #    pl.pause(0.01)
        plot_movie = False

        labels = [rrgg.replace('regions', '').replace(
            '/', '').replace('.zip', '') for rrgg in region_pairs]
#        pl.figure(figsize=(30,20))
        idx_tp_gt, idx_tp_comp, idx_fn, idx_fp_comp, performance = cm.base.rois.nf_match_neurons_in_binary_masks(
            roi_nat, roi_lin, thresh_cost=.7, min_dist=10, print_assignment=False, plot_results=plot_movie, Cn=c_img, labels=labels)
        result[region_pairs] = {'GTidx': idx_tp_gt, 'COMP_idx': idx_tp_comp,
                                'idx_fn': idx_fn, 'idx_fp_comp': idx_fp_comp, 'performance': performance}
        fig_name = (par[0] + '-' + region_pairs[0] + '_' + region_pairs[1]).replace(
            '/', '_').replace('regions.zip', '-').replace('.', '_') + '.pdf'
        if plot_movie:
            pl.rcParams['pdf.fonttype'] = 42
            font = {'family': 'Helvetica',
                    'weight': 'regular',
                    'size': 10}
            pl.rc('font', **font)
            pl.savefig(os.path.join(
                '/mnt/ceph/neuro/DataForPublications/Labeling/images/spatial_comparison', fig_name))
            pl.pause(1)
            pl.savefig(os.path.join(
                '/mnt/ceph/neuro/DataForPublications/Labeling/images/spatial_comparison', fig_name[:-4] + '.png'))
            pl.figure(figsize=(30, 20))

    results[par[0]] = result.copy()
    np.savez('result_comparison_gt.npz', results=results)
    #    pl.imshow(np.sum(masks_nf,0),cmap='Greens',alpha=.3)
#%%
import pandas as pd
ref_name = 'ben'
for count, ref_name in enumerate(['ben', 'lindsey', 'sonia', 'natalia']):
    pl.subplot(2, 2, count + 1)
    index = []
    cols = []
    vals = []
    all_values = []
    if compare_code == 0:
        iterations_on = [[(r, consensus_name[8:-12], a[8:-15], b['performance']['f1_score'])
                          for a, b in k.iteritems()] for r, k in results.iteritems()]
    elif compare_code == 1:
        iterations_on = [[(r, a[0][8:-12], a[1][8:-12], b['performance']['f1_score'])
                          for a, b in k.iteritems()] for r, k in results.iteritems()]

    for fl in iterations_on:
        for nm, lb1, lb2, perf in fl:
            if ref_name in lb1 + lb2:
                index.append(nm)
                vals.append(perf)
                if ref_name in lb1:
                    cols.append(lb2)
                else:
                    cols.append(lb1)
                all_values.append((index[-1], vals[-1], cols[-1]))

    df = pd.DataFrame(all_values, columns=('filename', 'perf', 'labeler'))
    #df = pd.DataFrame(vals, index=cols, columns=index)
    df_un = df.set_index(['filename', 'labeler']).unstack(0)
    pl.pcolor(df_un, norm=None, cmap='gray', vmax=1, vmin=0.5)
    pl.colorbar()
    r, c = df_un.shape
    pl.yticks(np.arange(0.5, r, 1), df_un.index)
    pl.xticks(np.arange(0.5, c, 1), [b[:7] for a, b in (df_un.columns.values)])
    pl.title(ref_name)
#%%
index = []
cols = []
vals = []
counter = 0
if compare_code == 0:
    iterations_on = [[(r, consensus_name[8:-12], a[8:-15], b['performance']['f1_score'])
                      for a, b in k.iteritems()] for r, k in results.iteritems()]
elif compare_code == 1:
    iterations_on = [[(r, a[0][8:-12], a[1][8:-12], b['performance']['f1_score'])
                      for a, b in k.iteritems()] for r, k in results.iteritems()]
for fl in iterations_on:
    all_values = []
    for nm, lb1, lb2, perf in fl:

        all_values.append((lb1, lb2, perf))
        all_values.append((lb2, lb1, perf))
    df = pd.DataFrame(all_values, columns=('labeler1', 'labeler2', 'perf'))
    df_un = df.set_index(['labeler1', 'labeler2']).unstack(0)
    counter += 1
    pl.subplot(3, 5, counter)
    pl.pcolor(df_un, norm=None, cmap='gray', vmax=0.9, vmin=0.3)
    pl.colorbar()
    r, c = df_un.shape
    pl.yticks(np.arange(0.5, r, 1), df_un.index)
    pl.xticks(np.arange(0.5, r, 1), df_un.index)
    pl.title(nm)
#%%

#df = pd.DataFrame(vals, index=cols, columns=index)
if compare_code == 0:
    iterations_on = [[(r, consensus_name[8:-12], a[8:-15], b['performance']['f1_score'], b['performance']
                       ['precision'], b['performance']['recall']) for a, b in k.iteritems()] for r, k in results.iteritems()]
elif compare_code == 1:
    iterations_on = [[(r, a[0][8:-12], a[1][8:-12], b['performance']['f1_score'], b['performance']['precision'],
                       b['performance']['recall']) for a, b in k.iteritems()] for r, k in results.iteritems()]
all_values = []
# [[(r,a[0][8:-12],a[1][8:-12],b['performance']['f1_score'],b['performance']['precision'],b['performance']['recall']) for a,b in k.iteritems()] for r,k in results.iteritems()]:
for fl in iterations_on:
    all_values += fl


df = pd.DataFrame(all_values, columns=('filename', 'labeler_1',
                                       'labeler_2', 'F1', 'precision', 'recall'))
df.to_excel('active_regions.xlsx')
#%%
g = df.groupby(df['labeler_2'])
g.describe()
#%% with dates
df = pd.read_excel('active_regions_with_date.xlsx')

df.describe()
#%%
df.groupby(['labeler_2'])['F1'].plot(x=np.arange(len(df)), legend=True)
#%%
df = df.sort_values('date', ascending=True)
df.index = np.arange(len(df))
pl.plot(df['date'], df['F1'], '*')
pl.xticks(rotation='vertical')
#%%
labels = []
for key, grp in df.groupby(['labeler_2']):
    #    pl.figure()
    labels.append(key)
    grp.sort_values('date', ascending=True)
    pl.plot(np.arange(len(grp)), grp['F1'], label=key)

#    pl.plot(grp['date'],grp['F1'], label=key)
#    pl.xticks(rotation='vertical')

pl.legend(labels)
pl.xlabel('Execution order')
pl.ylabel('F1 score')
#%%
import scipy
labels = []
for key, grp in df.groupby(['labeler_2']):
    #    pl.figure()
    labels.append(key)
    grp.sort_values('date', ascending=True)
    a = pl.hist(grp['F1'], label=key, cumulative=True,
                normed=1, bins=10, alpha=0)
    pl.plot(a[0], scipy.signal.savgol_filter(a[1][1:], 9, 8))
    pl.ylim([.6, 1])

pl.legend(labels)
pl.xlabel('F1 score')
pl.ylabel('probabiility (CDF)')
