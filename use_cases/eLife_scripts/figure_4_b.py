# -*- coding: utf-8 -*-
"""
This script reproduces the results for Figure 4b
The script loads the saved results for various parameters combinations and
plots the F1 score for the parameter combination that maximizes the performance
on average as well as the maximum F1 score attained for each dataset.
For running the CaImAn batch and CaImAn online algorithm and
obtain the results check the scripts:
/preprocessing_files/Preprocess_batch.py
/preprocessing_files/Preprocess_CaImAn_online.py

More info can be found in the companion paper
"""

try:
    if __IPYTHON__:
        print(1)
        # this is used for debugging purposes only. allows to reload classes
        # when changed
        get_ipython().magic('load_ext autoreload')
        get_ipython().magic('autoreload 2')
except NameError:
    print('Not launched under iPython')

import numpy as np
import os
import pylab as pl
from pandas import DataFrame
pl.rcParams['pdf.fonttype'] = 42
font = {'family': 'Arial',
        'weight': 'regular',
        'size': 20}
pl.rc('font', **font)
base_folder = '/mnt/ceph/neuro/DataForPublications/DATA_PAPER_ELIFE/WEBSITE/'

# %% Figure 4b and GRID statistics
with np.load(os.path.join(base_folder,'ALL_RECORDS_GRID_FINAL.npz')) as ld:
    records = ld['records'][()]
    records = [list(rec) for rec in records]
    records = [rec[:5]+[float(rr) for rr in rec[5:]] for rec in records]

columns = ['name', 'gr_snr', 'grid_rval', 'grid_max_prob_rej',
           'grid_thresh_CNN', 'recall', 'precision', 'f1_score']

with np.load(os.path.join(base_folder, 'all_records_grid_online.npz')) as ld:
    records_online = ld['records']
    records_online = [list(rec) for rec in records_online]
    records_online = [rec[:4] + [float(rr) for rr in rec[4:]] for rec in records_online]

columns_online = ['name', 'trace_SNR', 'thresh_CNN', 'num_comp', 'recall',
                  'precision', 'f1_score']
#%%
df_batch = DataFrame(records)
df_batch.columns = columns
df_online = DataFrame(records_online)
df_online.columns = columns_online
#%% Max of all datasets
for df in [df_batch, df_online]:
    best_res = df.groupby(by=['name'])
    best_res = best_res.describe()
    max_caiman_batch = best_res['f1_score']['max']
    print(max_caiman_batch)
    print(max_caiman_batch.mean())
    print(max_caiman_batch.std())
#%%
best_res = df_batch.groupby(by=['gr_snr', 'grid_rval', 'grid_max_prob_rej', 'grid_thresh_CNN'])
best_res_batch = best_res.describe()
print(best_res_batch.loc[:, 'f1_score'].max())
best_res = df_online.groupby(by=['trace_SNR', 'thresh_CNN', 'num_comp'])
best_res_online = best_res.describe()
print(best_res_online.loc[:, 'f1_score'].max())

#%%
pars_batch = best_res_batch.loc[:, 'f1_score'].idxmax()['mean']
print(pars_batch)
df_result_batch = df_batch[((df_batch['gr_snr'] == pars_batch[0]) & (df_batch['grid_rval'] == pars_batch[1]) & (df_batch['grid_max_prob_rej'] == pars_batch[2]) & (df_batch['grid_thresh_CNN'] == pars_batch[3]))]
print(df_result_batch.sort_values(by='name')[['name','precision','recall','f1_score']])
print(df_result_batch.mean())
print(df_result_batch.std())

pars_online = best_res_online.loc[:, 'f1_score'].idxmax()['mean']
print(pars_online)
df_result_online = df_online[((df_online['trace_SNR'] == pars_online[0]) & (df_online['thresh_CNN'] == pars_online[1]) & (df_online['num_comp'] == pars_online[2]))]
print(df_result_online.sort_values(by='name')[['name','precision','recall','f1_score']])
print(df_result_online.mean())
print(df_result_online.std())
#%%
df_result_batch = df_result_batch.sort_values(by='name')
max_res = df_batch.groupby(by=['name'])
max_res = max_res.describe()
max_res = max_res.sort_values(by='name')
max_res = max_res['f1_score']['max']
df_result_batch['f1_score_max'] = max_res.values
# min_res = df_result_batch.groupby(by=['name'])
# min_res = min_res.describe()
# min_res = min_res.sort_values(by='name')
# min_res = min_res['f1_score']['min']
# df_result_batch['f1_score_min'] = min_res.values


df_result_online = df_result_online.sort_values(by='name')
max_res = df_online.groupby(by=['name'])
max_res = max_res.describe()
max_res = max_res.sort_values(by='name')
max_res = max_res['f1_score']['max']
df_result_online['f1_score_max'] = max_res.values
# min_res = df_result_online.groupby(by=['name'])
# min_res = min_res.describe()
# min_res = min_res.sort_values(by='name')
# min_res = min_res['f1_score']['min']
# df_result_online['f1_score_min'] = min_res.values
df_result_all = df_result_batch.copy()
df_result_all['f1_score_online'] = df_result_online['f1_score'].values
df_result_all['f1_score_online_max'] = df_result_online['f1_score_max'].values



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
df_result_all['L1_f1'] = np.array([np.nan, np.nan, 0.78, np.nan, 0.89, 0.8, 0.89,  0.85, np.nan])[idx_sort]  # Human 1
df_result_all['L2_f1'] = np.array([0.9, 0.69, 0.9, 0.92, 0.87, 0.89, 0.92, 0.93, 0.83])[idx_sort]  # Human 2
df_result_all['L3_f1'] = np.array([0.85, 0.75, 0.82, 0.83, 0.84, 0.78, 0.93, 0.94, 0.9])[idx_sort] # Human 3
df_result_all['L4_f1'] = np.array([0.78, 0.87, 0.79, 0.87, 0.82, 0.75, 0.83, 0.83, 0.91])[idx_sort]


#
#
# df_result_batch['f1_score_CaImAn_online'] = np.array([0.74213836,  0.71713147,  0.78541374,  0.77562327,  0.69266771,
#     0.74285714,  0.80835509,  0.78950077,  0.83573487])[idx_sort]
#%%
ax = df_result_all.plot(x='name', y=['f1_score', 'f1_score_online','L4_f1','L3_f1','L2_f1','L1_f1'], xticks=range(len(df_result_all)),
                    kind='bar', color=[[1,0,0],[0,0,1],[.5,.5,.5],[.6,.6,.6],[.7,.7,.7],[.8,.8,.8]])

ax.set_xticklabels(df_result_all.name, rotation=45)
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

df_result_all['size_order'] = [7, 6, 8, 2, 0, 5, 1, 4, 3]
df_result_all['f1_score_mean_humans'] = np.nanmean(np.vstack([df_result_all['L1_f1'],df_result_all['L2_f1'], df_result_all['L3_f1'] , df_result_all['L4_f1']]),0).T
df_result_all.sort_values(by='size_order').plot(kind='bar', x='name', y=['f1_score_mean_humans','f1_score_online_max',
                                                                         'f1_score_online','f1_score_max','f1_score'],
                                                        label=['human', 'online max', 'online', 'batch max', 'batch'])

