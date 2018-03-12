#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 11:12:41 2018

@author: epnevmatikakis
"""


try:
    if __IPYTHON__:
        print('Debugging!')
        # this is used for debugging purposes only. allows to reload classes when changed
        get_ipython().magic('load_ext autoreload')
        get_ipython().magic('autoreload 2')
except NameError:
    print('Not launched under iPython')

import numpy as np
import matplotlib.pyplot as plt
import caiman as cm
from caiman.base.rois import register_multisession, extract_active_components, register_ROIs
import matplotlib.lines as mlines
from scipy.io import loadmat
from scipy.sparse import csc_matrix
from glob import glob
import itertools

base_folder = '/Users/epnevmatikakis/Desktop/untitled folder/'
#base_folder = '/mnt/ceph/neuro/Sue/k53/'

files = glob(base_folder+'*/*.npz')

#%% load data
data = []
Cns = []
for fl in files:
    print(fl)
    with np.load(fl) as ld:
        A_ = ld['A'][()].toarray()
        idx_ = np.where(ld['cnn_preds']>.75)[0]
#        idx_ = ld['idx_components']
        data.append(A_[:,idx_])
        Cns.append(ld['Cn'])

#%% normalizing function

def norm_nrg(a_):
    a = a_.copy()
    dims = a.shape
    a = a.reshape(-1, order='F')
    indx = np.argsort(a, axis=None)[::-1]
    cumEn = np.cumsum(a.flatten()[indx]**2)
    cumEn = cumEn/cumEn[-1]
    a = np.zeros(np.prod(dims))
    a[indx] = cumEn
    return a.reshape(dims, order='F')       

#%% normalize to sum 1 
dims = 512, 512

A = [csc_matrix(A1/A1.sum(0)) for A1 in data]
masks = [np.reshape(A_.toarray(), dims + (-1,),
         order='F').transpose(2, 0, 1) for A_ in A]

#%% register components across multiple days

max_thr = 0.1    
    
A_forw, assign_forw, match_forw = register_multisession(A, dims, Cns, max_thr=max_thr)
A_back, assign_back, match_back = register_multisession(A[::-1], dims, Cns[::-1], max_thr=max_thr)

#%% align all pairs separately
N = len(A)

pairs = list(itertools.combinations(range(N), 2))

pairs_matches = []

for pair in pairs:
    match_1, match_2, non_1, non_, perf_, _ = register_ROIs(A[pair[0]], A[pair[1]], dims, 
                                                            template1=Cns[pair[0]], 
                                                            template2=Cns[pair[1]],
                                                            plot_results=False,
                                                            max_thr=max_thr)
    pairs_matches.append(np.vstack((match_1, match_2)))

#%% read all pairing through the union
    
pairs_forw = []
pairs_back = []
assign_back_rev = assign_back[:,::-1]
for pair in pairs:
    match_f = extract_active_components(assign_forw, list(pair), only=False)
    match_f1 = assign_forw[list(match_f)][:,list(pair)]
    pairs_forw.append(match_f1.astype(int).T)
    match_b = extract_active_components(assign_back_rev, list(pair), only=False)
    match_b1 = assign_back_rev[list(match_b)][:,list(pair)]
    pairs_back.append(match_b1.astype(int).T)
#%% compare alignments

prec_forw = []
rec_forw = []
f1_forw = []
acc_forw = []

for pair_direct, pair_forw in zip(pairs_matches, pairs_forw):
    pair_direct = list(pair_direct.T.tolist())
    pair_forw = list(pair_forw.T.tolist())
    inc = [pair_f in pair_direct for pair_f in pair_forw]
    TP = np.array(inc).sum()
    FP = len(inc) - TP
    FN = len(pair_direct) - TP
    prec_forw.append(TP/(TP+FP))
    rec_forw.append(TP/(TP+FN))
    f1_forw.append(2*TP/(2*TP+FP+FN))
    acc_forw.append(TP/(TP+FP+FN))

prec_back = []
rec_back = []
f1_back = []
acc_back = []

for pair_direct, pair_back in zip(pairs_matches, pairs_back):
    pair_direct = list(pair_direct.T.tolist())
    pair_back = list(pair_back.T.tolist())
    inc = [pair_f in pair_direct for pair_f in pair_back]
    TP = np.array(inc).sum()
    FP = len(inc) - TP
    FN = len(pair_direct) - TP
    prec_back.append(TP/(TP+FP))
    rec_back.append(TP/(TP+FN))
    f1_back.append(2*TP/(2*TP+FP+FN))
    acc_back.append(TP/(TP+FP+FN))

#%%
    
combs = list(pairs)
for i in range(2, N):
    combs = combs + list(itertools.combinations(range(N), i+1))
    
prec_fb = []
rec_fb = []
f1_fb = []
acc_fb = []

for comb in combs:
    match_f = extract_active_components(assign_forw, list(comb), only=False)
    match_f1 = assign_forw[list(match_f)][:,list(comb)]
    comb_forw = list(match_f1.astype(int).tolist())
    match_b = extract_active_components(assign_back_rev, list(comb), only=False)
    match_b1 = assign_back_rev[list(match_b)][:,list(comb)]
    comb_back = list(match_b1.astype(int).tolist())
    inc = [comb_f in comb_back for comb_f in comb_forw]
    TP = np.array(inc).sum()
    FP = len(inc) - TP
    FN = len(comb_back) - TP
    prec_fb.append(TP/(TP+FP))
    rec_fb.append(TP/(TP+FN))
    f1_fb.append(2*TP/(2*TP+FP+FN))
    acc_fb.append(TP/(TP+FP+FN))
    
ln = [len(comb) for comb in combs]


#%%
dist_pair = [pair[1]-pair[0] for pair in pairs]

plt.figure()
plt.subplot(2,1,1); plt.scatter(dist_pair,f1_forw,c='r')
plt.scatter(dist_pair,f1_back,c='b')
plt.title('F_1 score, union vs direct matching')
day = [mlines.Line2D([], [], color='r', label='Forward')] + [mlines.Line2D([], [], color='b', label='Backward')]
plt.legend(handles=day, loc=3)
plt.xlabel('Time difference between session (days)')

plt.subplot(2,1,2); plt.scatter(ln,f1_fb)
plt.title('F_1 score, forward vs backward')
plt.xlabel('Number of sessions')

#%% make figure for the paper

match_f = extract_active_components(assign_forw, range(N), only=False)
match_f1 = assign_forw[list(match_f)][:,range(N)]
comb_forw = list(match_f1.astype(int).T.tolist())

plt.figure(figsize=(15,10))
plt.rcParams['pdf.fonttype'] = 42
font = {'family': 'Myriad Pro',
        'weight': 'regular',
        'size': 10}
plt.rc('font', **font)
cl = ['r', 'b', 'y', 'm', 'w', 'g']
ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=2)
lp, hp = np.nanpercentile(Cns[0], [5, 98])
plt.imshow(Cns[0], vmin=lp, vmax=hp, cmap='gray')
for i in range(N):
    [plt.contour(norm_nrg(mm), levels=[0.95], colors=cl[i], linewidths=2)
     for mm in masks[i][comb_forw[i]]]
plt.title('Components tracked across all days (forward)')
plt.xlim((3, dims[0]-7))
plt.axis('off')
day = [mlines.Line2D([], [], color=cl[i], label='Day '+str(i+1)) for i in range(N)]
plt.legend(handles=day, loc=3)

ax2 = plt.subplot2grid((3, 3), (0, 2))
plt.scatter(dist_pair,f1_forw,c='r')
plt.scatter(dist_pair,f1_back,c='b')
plt.title('F_1 score, union vs direct matching')
day = [mlines.Line2D([], [], color='r', label='Forward')] + [mlines.Line2D([], [], color='b', label='Backward')]
plt.legend(handles=day, loc=1)
plt.ylim((0.975,1.002))
plt.xlabel('Time difference between session (days)')

ax3 = plt.subplot2grid((3, 3), (1, 2))
plt.scatter(ln,f1_fb)
plt.title('F_1 score, forward vs backward')
plt.ylim((0.975, 1))
plt.xlabel('Number of sessions')
plt.tight_layout()