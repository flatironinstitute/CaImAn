#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script for generating the results presented in Figure9-supplement 1
(ROI registration across multiple days)
The script is intended to demonstrate the process that multiday registration
works, rather than performing registration across 3 different days which can
be done at once using the register_multisession function.

@author: epnevmatikakis
"""

import numpy as np
import matplotlib.pyplot as plt
import caiman as cm
from caiman.base.rois import register_multisession, extract_active_components, register_ROIs
import matplotlib.lines as mlines
from scipy.sparse import csc_matrix
import pickle
import os

# %% load spatial components and correlation images for each session

base_folder = '/mnt/ceph/neuro/DataForPublications/DATA_PAPER_ELIFE/WEBSITE/'
with open(os.path.join(base_folder, 'alignment.pickle'), 'rb') as f:
    A, CI = pickle.load(f)

# A is a list where each entry is the matrix of the spatial components for each session
# CI is a list where each entry is the correlation image for each session

# %% normalizing function


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


# %% normalize sum of each component to 1

dims = CI[0].shape
N = 3  # consider only the first three sessions

A = [csc_matrix(A1/A1.sum(0)) for A1 in A[:N]]
masks = [np.reshape(A_.toarray(), dims + (-1,),
         order='F').transpose(2, 0, 1) for A_ in A]

# %% contour plots for the components of each session (Fig. 14a)

plt.figure(figsize=(10, 15))
plt.rcParams['pdf.fonttype'] = 42
font = {'family': 'Myriad Pro',
        'weight': 'regular',
        'size': 10}
plt.rc('font', **font)
lp, hp = np.nanpercentile(CI[0], [5, 98])

for i in range(N):
    plt.subplot(3, 2, 1 + 2*i)
    plt.imshow(CI[i], vmin=lp, vmax=hp, cmap='gray')
    [plt.contour(norm_nrg(mm), levels=[0.95], colors='y', linewidths=1)
        for mm in masks[i]]
    plt.title('Session #'+str(i+1))

# %% register components across multiple days

max_thr = 0.1
A_reg, assign, match = register_multisession(A, dims, CI, max_thr=max_thr)
masks_reg = np.reshape(A_reg, dims + (-1,), order='F').transpose(2, 0, 1)
# %% first compare results from sessions 1 and 2 (Fig. 14b)
# If you just have two sessions you can use the register_ROIs function

match_1 = extract_active_components(assign, [0], only=False)
match_2 = extract_active_components(assign, [1], only=False)
match_12 = extract_active_components(assign, [0, 1], only=False)

cl = ['y', 'g', 'r']
labels = ['Both Sessions', 'Session 1 (only)', 'Session 2 (only)']
plt.subplot(3, 2, 2)
plt.imshow(CI[N], vmin=lp, vmax=hp, cmap='gray')
plt.title('Session #1 and #2 union')
[plt.contour(norm_nrg(mm), levels=[0.95], colors='y', linewidths=1)
    for mm in masks[0][match_12]]
[plt.contour(norm_nrg(mm), levels=[0.95], colors='g', linewidths=1)
    for mm in masks[0][np.setdiff1d(match_1, match_12)]]
[plt.contour(norm_nrg(mm), levels=[0.95], colors='r', linewidths=1)
    for mm in masks[1][assign[np.setdiff1d(match_2, match_12), 1].astype(int)]]
day = [mlines.Line2D([], [], color=cl[i], label=labels[i]) for i in range(N)]
plt.legend(handles=day, loc=3)

# %% Now align session 3 with union of sessions 1 and 2 (Fig. 14c)

match_13 = extract_active_components(assign, [0, 2], only=False)
match_23 = extract_active_components(assign, [1, 2], only=False)
match_3 = extract_active_components(assign, [2], only=False)

cl = ['y', 'g', 'r']
labels = ['Session 1 or 2 and session 3', 'Session 1 or 2 (only)', 'Session 3 (only)']
plt.subplot(3, 2, 4)
plt.imshow(CI[N], vmin=lp, vmax=hp, cmap='gray')
plt.title('Union of all sessions')
[plt.contour(norm_nrg(mm), levels=[0.95], colors='y', linewidths=1)
    for mm in masks_reg[np.union1d(match_13, match_23)]]
[plt.contour(norm_nrg(mm), levels=[0.95], colors='g', linewidths=1)
    for mm in masks_reg[np.setdiff1d(match_12, match_3)]]
[plt.contour(norm_nrg(mm), levels=[0.95], colors='r', linewidths=1)
    for mm in masks_reg[np.setdiff1d(match_3, match_12)]]
day = [mlines.Line2D([], [], color=cl[i], label=labels[i]) for i in range(N)]
plt.legend(handles=day, loc=3)

# %% Register sessions 1 and 3 directly from the union

cl = ['y', 'g', 'r']
labels = ['Session 1 and 3', 'Session 1 (only)', 'Session 3 (only)']
plt.subplot(3, 2, 6)
plt.imshow(CI[N], vmin=lp, vmax=hp, cmap='gray')
plt.title('Union of all sessions')
[plt.contour(norm_nrg(mm), levels=[0.95], colors='y', linewidths=1)
    for mm in masks_reg[match_13]]
[plt.contour(norm_nrg(mm), levels=[0.95], colors='g', linewidths=1)
    for mm in masks_reg[np.setdiff1d(match_1, match_3)]]
[plt.contour(norm_nrg(mm), levels=[0.95], colors='r', linewidths=1)
    for mm in masks_reg[np.setdiff1d(match_3, match_1)]]
day = [mlines.Line2D([], [], color=cl[i], label=labels[i]) for i in range(N)]
plt.legend(handles=day, loc=3)
