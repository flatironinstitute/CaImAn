# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 14:49:36 2017

@author: agiovann
"""
import cv2

try:
    cv2.setNumThreads(1)
except:
    print('Open CV is naturally single threaded')

try:
    if __IPYTHON__:
        print(1)
        # this is used for debugging purposes only. allows to reload classes
        # when changed
        get_ipython().magic('load_ext autoreload')
        get_ipython().magic('autoreload 2')
except NameError:
    print('Not launched under iPython')

import caiman as cm
import numpy as np
import os
import pylab as pl
import scipy
from caiman.utils.visualization import plot_contours
import numpy as np
from pandas import DataFrame
import pylab as pl
pl.rcParams['pdf.fonttype'] = 42
font = {'family': 'Arial',
        'weight': 'regular',
        'size': 20}
pl.rc('font', **font)
base_folder = '/mnt/ceph/neuro/DataForPublications/DATA_PAPER_ELIFE/WEBSITE/'

#%% Figure 4b and GRID statistics
with np.load(os.path.join(base_folder,'ALL_RECORDS_GRID_FINAL.npz')) as ld:
    records = ld['records'][()]
    records = [list(rec) for rec in records]
    records = [rec[:5]+[float(rr) for rr in rec[5:]] for rec in records]
#%% Max of all datasets
df = DataFrame(records)
df.columns = ['name', 'gr_snr', 'grid_rval', 'grid_max_prob_rej', 'grid_thresh_CNN', 'recall',
              'precision', 'f1_score']
best_res = df.groupby(by=['name'])
best_res = best_res.describe()
max_caiman_batch = best_res['f1_score']['max']
print(max_caiman_batch)
print(max_caiman_batch.mean())
print(max_caiman_batch.std())
#%%
df = DataFrame(records)
df.columns = ['name', 'gr_snr', 'grid_rval', 'grid_max_prob_rej', 'grid_thresh_CNN','recall', 'precision', 'f1_score']
best_res = df.groupby(by=['gr_snr', 'grid_rval', 'grid_max_prob_rej', 'grid_thresh_CNN'])
best_res = best_res.describe()
print(best_res.loc[:, 'f1_score'].max())
#%%
df = DataFrame(records)
df.columns = ['name', 'gr_snr', 'grid_rval', 'grid_max_prob_rej', 'grid_thresh_CNN','recall', 'precision', 'f1_score']
best_res = df.groupby(by=['gr_snr', 'grid_rval', 'grid_max_prob_rej', 'grid_thresh_CNN'])
best_res = best_res.describe()
pars = best_res.loc[:, 'f1_score'].idxmax()['mean']
print(pars)
df_result = df[((df['gr_snr'] == pars[0]) & (df['grid_rval'] == pars[1]) & (df['grid_max_prob_rej'] == pars[2]) & (df['grid_thresh_CNN'] == pars[3]))]

print(df_result.sort_values(by='name')[['name','precision','recall','f1_score']])
print(df_result.mean())
print(df_result.std())



# %% Results for CaImAn Online (modify as needed)

with np.load(os.path.join(base_folder, 'all_records_grid_online.npz')) as ld:
    records_online = ld['records']
    records_online = [list(rec) for rec in records_online]

columns_online = ['name', 'trace_SNR', 'thresh_CNN', 'num_comp', 'recall',
                  'precision', 'F1_score']

#%%
df_result = df_result.sort_values(by='name')
max_res = df.groupby(by=['name'])
max_res = max_res.describe()
max_res = max_res.sort_values(by='name')
max_res = max_res['f1_score']['max']
df_result['f1_score_max'] = max_res.values
min_res = df.groupby(by=['name'])
min_res = min_res.describe()
min_res = min_res.sort_values(by='name')
min_res = min_res['f1_score']['min']
df_result['f1_score_min'] = min_res.values


names = ['N.03.00.t',
         'N.04.00.t',
         'YST',
         'N.00.00',
         'N.02.00',
         'N.01.01',
         'K53',
         'J115',
         'J123']

idx_sort = np.argsort(names)
df_result['L1_f1'] = np.array([np.nan, np.nan, 0.78, np.nan, 0.89, 0.8, 0.89,  0.85, np.nan])[idx_sort]  # Human 1
df_result['L2_f1'] = np.array([0.9, 0.69, 0.9, 0.92, 0.87, 0.89, 0.92, 0.93, 0.83])[idx_sort]  # Human 2
df_result['L3_f1'] = np.array([0.85, 0.75, 0.82, 0.83, 0.84, 0.78, 0.93, 0.94, 0.9])[idx_sort] # Human 3
df_result['L4_f1'] = np.array([0.78, 0.87, 0.79, 0.87, 0.82, 0.75, 0.83, 0.83, 0.91])[idx_sort]




df_result['f1_score_CaImAn_online'] = np.array([0.74213836,  0.71713147,  0.78541374,  0.77562327,  0.69266771,
    0.74285714,  0.80835509,  0.78950077,  0.83573487])[idx_sort]
#%%
ax = df_result.plot(x='name', y=['f1_score', 'f1_score_CaImAn_online','L4_f1','L3_f1','L2_f1','L1_f1'], xticks=range(len(df_result)),
                    kind='bar', color=[[1,0,0],[0,0,1],[.5,.5,.5],[.6,.6,.6],[.7,.7,.7],[.8,.8,.8]])

ax.set_xticklabels(df_result.name, rotation=45)
pl.legend(['CaImAn batch','CaImAn online','L4','L3','L2','L1'])
# ax.set_xticklabels(df_result.name)
# pl.xlabel('Dataset')
pl.ylabel('F1 score')
pl.ylim([0.55,0.95])
params_display = {
    'downsample_ratio': .2,
    'thr_plot': 0.8
}

pl.rcParams['pdf.fonttype'] = 42
font = {'family': 'Arial',
        'weight': 'regular',
        'size': 20}
pl.rc('font', **font)
pl.close()
#%%
pl.figure()
online_F1_max = np.array([0.75316456, 0.71713147, 0.79427083, 0.79733333, 0.718529,
                          0.76, 0.84371328, 0.81400438, 0.83965015])[idx_sort]
all_labels = np.vstack([df_result['L1_f1'],df_result['L2_f1'], df_result['L3_f1'] , df_result['L4_f1']])
mean_labels = np.nanmean(all_labels,0).T
df_cm = DataFrame({'Human average':mean_labels,'CaImAn online max':online_F1_max, 'CaImAn online avg': df_result['f1_score_CaImAn_online'].values
                    ,'CaImAn batch max':max_caiman_batch,'CaImAn batch avg': df_result['f1_score'].values})
df_cm.plot(kind='bar')
#%%
