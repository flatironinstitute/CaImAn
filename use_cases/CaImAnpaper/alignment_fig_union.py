#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on Fri Jan  5 12:26:31 2018

@author: epnevmatikakis
Script for generating the alignment figure 4 and supplementary figure  using 
the union approach
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
import pylab as plt
import caiman as cm
#%%
from scipy.io import loadmat
from scipy.sparse import csc_matrix
from glob import glob

files = ['/Users/epnevmatikakis/Desktop/untitled folder/20160530/memmap__d1_512_d2_512_d3_1_order_C_frames_27000_.results_analysis.npz',
         '/Users/epnevmatikakis/Desktop/untitled folder/20160531/memmap__d1_512_d2_512_d3_1_order_C_frames_27000_.results_analysis.npz',
         '/Users/epnevmatikakis/Desktop/untitled folder/20160603/memmap__d1_512_d2_512_d3_1_order_C_frames_27000_.results_analysis.npz',
         '/Users/epnevmatikakis/Desktop/untitled folder/20160606/memmap__d1_512_d2_512_d3_1_order_C_frames_27000_.results_analysis.npz',
         '/Users/epnevmatikakis/Desktop/untitled folder/20160607/memmap__d1_512_d2_512_d3_1_order_C_frames_27000_.results_analysis.npz',
         '/Users/epnevmatikakis/Desktop/untitled folder/20160608/memmap__d1_512_d2_512_d3_1_order_C_frames_27000_.results_analysis.npz']

#files = ['/mnt/ceph/neuro/Sue/k53/20160530/memmap__d1_512_d2_512_d3_1_order_C_frames_27000_.results_analysis.npz',
#         '/mnt/ceph/neuro/Sue/k53/20160531/memmap__d1_512_d2_512_d3_1_order_C_frames_27000_.results_analysis.npz',
#         '/mnt/ceph/neuro/Sue/k53/20160603/memmap__d1_512_d2_512_d3_1_order_C_frames_27000_.results_analysis.npz']

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

#%%

A = [csc_matrix(A1 / A1.sum(0)) for A1 in data]

#A1, A2, A3 = data
#Cn1, Cn2, Cn3 = Cns
#
##%% normalize matrices
#A1 = csc_matrix(A1 / A1.sum(0))
#A2 = csc_matrix(A2 / A2.sum(0))
#A3 = csc_matrix(A3 / A3.sum(0))

#%% register components across multiple days

from caiman.base.rois import register_multisession, extract_active_components, register_ROIs
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

dims = 512, 512

A_union, assignments, matchings = register_multisession(A, dims, Cns)

#%% register backwards

A_back, assignments_back, matchings_back = register_multisession(A[::-1], dims, Cns[::-1])

#%%
N = len(A)
trip_forw = extract_active_components(assignments, list(range(N)), only = False)
trip_back = extract_active_components(assignments_back, list(range(N)), only = True)

#%%

matched_ROIs1, matched_ROIs2, non_matched1, non_matched2, performan,_ = register_ROIs(A_union, A_back, dims, 
                                                                                      template1=Cns[-1], 
                                                                                      template2=Cns[0],
                                                                                      plot_results=True,
                                                                                      thresh_cost=.8)

#%% plot neurons that appear in all days

Cn = np.reshape(np.array([A_.sum(axis=1) for A_ in A]).sum(axis=(0,2)), dims,
                         order='F')

masks = [np.reshape(A_.toarray(), dims + (-1,), 
         order='F').transpose(2, 0, 1) for A_ in A]

#%%
level = 0.95
cl=['b','r','y','m','w','g']
plt.figure(figsize=(20,10))
plt.rcParams['pdf.fonttype'] = 42
font = {'family': 'Myriad Pro',
        'weight': 'regular',
        'size': 10}
plt.rc('font', **font)
lp, hp = np.nanpercentile(Cns[0], [5, 98])
plt.subplot(1,2,1)
plt.imshow(Cns[0], vmin=lp, vmax=hp, cmap='gray')
for i in range(N):
    [plt.contour(norm_nrg(mm), levels=[level], colors=cl[i], linewidths=1)
     for mm in masks[i][list(assignments[trip_forw,i].astype(int))]]
plt.title('Components tracked across all days (forward)')
plt.axis('off')    
    
plt.subplot(1,2,2)
plt.imshow(Cns[-1], vmin=lp, vmax=hp, cmap='gray')
for i in range(N)[::-1]:
    [plt.contour(norm_nrg(mm), levels=[level], colors=cl[N-1-i], linewidths=1)
     for mm in masks[N-1-i][list(assignments_back[trip_back,i].astype(int))]]
plt.title('Components tracked across all days (backward)')
plt.axis('off')       

day = [mlines.Line2D([], [], color=cl[i], label='Day '+str(i)) for i in range(N)]
plt.legend(handles=day, loc=4)


#%%
comp_16 = extract_active_components(assignments, [0,N-1], only = False)

plt.figure()
matched_16_1, matched_16_6, non_matched1, non_matched2, performance,_ = register_ROIs(A[0], A[-1], dims, 
                                                                                      template1=Cns[0], 
                                                                                      template2=Cns[-1],
                                                                                      plot_results=False,
                                                                                      thresh_cost=.7)

#%%

plt.figure(figsize=(20,10))
plt.rcParams['pdf.fonttype'] = 42
font = {'family': 'Myriad Pro',
        'weight': 'regular',
        'size': 10}
plt.rc('font', **font)
lp, hp = np.nanpercentile(Cns[0], [5, 98])
plt.subplot(1,2,1)
plt.imshow(Cns[0], vmin=lp, vmax=hp, cmap='gray')
for i in [0,N-1]:
    [plt.contour(norm_nrg(mm), levels=[level], colors=cl[i], linewidths=1)
     for mm in masks[i][list(assignments[comp_16,i].astype(int))]]
plt.title('Components tracked across all days (forward)')
plt.axis('off') 

plt.subplot(1,2,2)
plt.imshow(Cns[0], vmin=lp, vmax=hp, cmap='gray')
[plt.contour(norm_nrg(mm), levels=[level], colors=cl[0], linewidths=1)
     for mm in masks[0][list(matched_16_1)]]
[plt.contour(norm_nrg(mm), levels=[level], colors=cl[5], linewidths=1)
     for mm in masks[5][list(matched_16_6)]]

#%%

plt.figure()
plt.imshow(Cns[0], vmin=lp, vmax=hp, cmap='gray')
[plt.contour(norm_nrg(mm), levels=[level], colors='r', linewidths=1)
     for mm in masks[0][df]]
[plt.contour(norm_nrg(mm), levels=[level], colors='y', linewidths=1)
     for mm in masks[0][list(assignments[df2,0].astype(int))]]

#%% OLD ANALYSIS

#%% find components that are active in certain sessions

triplets = assignments[np.where(np.isnan(assignments).sum(1) == 0)].astype(int)
    # extract triplets
    
matches_13 = assignments[np.intersect1d(
        np.where(np.isnan(assignments).sum(1) == 1), 
        np.where(np.isnan(assignments[:,1])))].astype(int)
    # example on how to extract components that are active on only days 1 and 3

matches_12 = assignments[np.intersect1d(
        np.where(np.isnan(assignments).sum(1) == 1), 
        np.where(np.isnan(assignments[:,2])))].astype(int)    

matches_23 = assignments[np.intersect1d(
        np.where(np.isnan(assignments).sum(1) == 1), 
        np.where(np.isnan(assignments[:,0])))].astype(int)    
   
matches_1 = assignments[np.intersect1d(
        np.where(np.isnan(assignments).sum(1) == 2), 
        np.where(assignments[:,0]>=0))].astype(int)    

matches_2 = assignments[np.intersect1d(
        np.where(np.isnan(assignments).sum(1) == 2), 
        np.where(assignments[:,1]>=0))].astype(int)    

matches_3 = assignments[np.intersect1d(
        np.where(np.isnan(assignments).sum(1) == 2), 
        np.where(assignments[:,2]>=0))].astype(int)   
    
    
#%% make figure
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


#%%


#%% figure for main paper. plots triplets and pairs
    
K_tot = assignments.shape[0]   # total number of components

Cn = np.reshape(A1.sum(axis=1) + A2.sum(axis=1) +
                A3.sum(axis=1), (512, 512), order='F')
plt.figure(figsize=(20,10))
masks_1 = np.reshape(A1.toarray(), dims + (-1,),
                     order='F').transpose(2, 0, 1)
masks_2 = np.reshape(A2.toarray(), dims + (-1,),
                     order='F').transpose(2, 0, 1)
masks_3 = np.reshape(A3.toarray(), dims + (-1,),
                     order='F').transpose(2, 0, 1)
#        try : #Plotting function
level = 0.95

plt.rcParams['pdf.fonttype'] = 42
font = {'family': 'Myriad Pro',
        'weight': 'regular',
        'size': 10}
plt.rc('font', **font)
lp, hp = np.nanpercentile(Cn, [5, 98])
plt.subplot(1,2,1)
plt.imshow(Cn, vmin=lp, vmax=hp, cmap='gray')
[plt.contour(norm_nrg(mm), levels=[level], colors='b', linewidths=1)
 for mm in masks_1[list(triplets[:,0]) + list(matches_12[:,0]) + list(matches_13[:,0])]]
[plt.contour(norm_nrg(mm), levels=[level], colors='r', linewidths=1)
 for mm in masks_2[list(triplets[:,1]) + list(matches_12[:,1]) + list(matches_23[:,1])]]
[plt.contour(norm_nrg(mm), levels=[level], colors='y', linewidths=1)
 for mm in masks_3[list(triplets[:,2]) + list(matches_13[:,2]) + list(matches_23[:,2])]]
# plt.legend(('Day1','Day2','Day3'))
plt.title('Components tracked across multiple days')
plt.axis('off')

day1 = mlines.Line2D([], [], color='b', label='Day 1')
day2 = mlines.Line2D([], [], color='r', label='Day 2')
day3 = mlines.Line2D([], [], color='y', label='Day 3')
plt.legend(handles=[day1, day2, day3], loc=4)

plt.show()

plt.subplot(1,2,2)
A_sum = np.reshape(A_union.sum(1),dims,order='F')
lv, hv = np.nanpercentile(A_sum, [5,98])
plt.imshow(A_sum, vmin = lv, vmax = hv, cmap = 'gray')
plt.title('Distinct components across the different sessions')
plt.axis('off')

plt.axis('off')

#%% compute union between sessions 1 and 2 (just for plotting purposes)

A_12, ass_12, mat_12 = register_multisession([A1, A2], dims, Cns[:-1])

masks_12 = np.reshape(A_12, dims + (-1,), order='F').transpose(2, 0, 1)
id_12 = np.where(np.isnan(ass_12).sum(-1) == 0)[0]
id_only_1 = np.where(np.isnan(ass_12[:,1]))[0]
id_only_2 = np.where(np.isnan(ass_12[:,0]))[0]

#%% figure for supplement. plots the various steps of the registration

lc = 0.05
hc = 0.5

masks_u = np.reshape(A_union, dims + (-1,),
                     order='F').transpose(2, 0, 1)

plt.figure(figsize=(21,14))

plt.subplot(2,3,1)
    # show all components from session 1
plt.imshow(Cn1, vmin = lc, vmax = hc, cmap='gray')
[plt.contour(norm_nrg(mm), levels=[level], colors='r', linewidths=1)
 for mm in masks_1]
plt.title('Session #1'); plt.axis('off')
#plt.colorbar()
                   
plt.subplot(2,3,2)
    # show all components from session 2
plt.imshow(Cn2, vmin = lc, vmax = hc, cmap='gray')
[plt.contour(norm_nrg(mm), levels=[level], colors='r', linewidths=1)
 for mm in masks_2]
plt.title('Session #2'); plt.axis('off')
          
plt.subplot(2,3,3)
    # show all components from session 3
plt.imshow(Cn3, vmin = lc, vmax = hc, cmap='gray')
[plt.contour(norm_nrg(mm), levels=[level], colors='r', linewidths=1)
 for mm in masks_3]
plt.title('Session #3'); plt.axis('off')          
         
plt.subplot(2,3,4)
    # show union between session 1 and session 2
        
plt.imshow(Cn2, vmin = lc, vmax = hc, cmap='gray')
[plt.contour(norm_nrg(mm), levels=[level], colors='r', linewidths=1)
 for mm in masks_12[id_12]]
[plt.contour(norm_nrg(mm), levels=[level], colors='b', linewidths=1)
 for mm in masks_12[id_only_1]]
[plt.contour(norm_nrg(mm), levels=[level], colors='w', linewidths=1)
 for mm in masks_12[id_only_2]]
day1 = mlines.Line2D([], [], color='r', label='Both sessions')
day2 = mlines.Line2D([], [], color='b', label='Session 1 (only)')
day3 = mlines.Line2D([], [], color='w', label='Session 2 (only)')
plt.legend(handles=[day1, day2, day3], loc=4)
plt.title('Session #1 and #2 union'); plt.axis('off')
          
plt.subplot(2,3,5)
    # show union between sessions 1 and 2 and session 3

id_1_or_2 = np.where(np.isnan(assignments[:,:-1]).sum(-1) < 2)[0]
id_3 = np.where(np.isnan(assignments[:,-1])==False)[0]
id_1_or_2_and_3 = np.intersect1d(id_1_or_2,id_3)
id_1_or_2_only = np.setdiff1d(range(A_union.shape[-1]),id_3)
id_3_only = np.setdiff1d(range(A_union.shape[-1]),id_1_or_2)

plt.imshow(Cn3, vmin = lc, vmax = hc, cmap = 'gray')
[plt.contour(norm_nrg(mm), levels=[level], colors='r', linewidths=1)
 for mm in masks_u[id_1_or_2_and_3]]
[plt.contour(norm_nrg(mm), levels=[level], colors='b', linewidths=1)
 for mm in masks_u[id_1_or_2_only]]
[plt.contour(norm_nrg(mm), levels=[level], colors='w', linewidths=1)
 for mm in masks_u[id_3_only]]
day1 = mlines.Line2D([], [], color='r', label='Session 1 or 2 and session 3')
day2 = mlines.Line2D([], [], color='b', label='Session 1 or 2 (only)')
day3 = mlines.Line2D([], [], color='w', label='Session 3 (only)')
plt.legend(handles=[day1, day2, day3], loc=4)
plt.title('Union of all sessions'); plt.axis('off')
          
plt.subplot(2,3,6)
    # compare sessions 1 and 3

id_1_and_3 = np.intersect1d(
        np.where(np.isnan(assignments[:,0]) == False)[0], 
        np.where(np.isnan(assignments[:,2]) == False)[0])

id_1_not_3 = np.setdiff1d(range(A1.shape[-1]),id_1_and_3)
id_3_not_1 = np.setdiff1d(id_3,id_1_and_3)

plt.imshow(Cn3, vmin = lc, vmax = hc, cmap = 'gray')
[plt.contour(norm_nrg(mm), levels=[level], colors='r', linewidths=1)
 for mm in masks_u[id_1_and_3]]
[plt.contour(norm_nrg(mm), levels=[level], colors='b', linewidths=1)
 for mm in masks_u[id_1_not_3]]
[plt.contour(norm_nrg(mm), levels=[level], colors='w', linewidths=1)
 for mm in masks_u[id_3_not_1]]
day1 = mlines.Line2D([], [], color='r', label='Session 1 and 3')
day2 = mlines.Line2D([], [], color='b', label='Session 1 (only)')
day3 = mlines.Line2D([], [], color='w', label='Session 3 (only)')
plt.legend(handles=[day1, day2, day3], loc=4)
plt.title('Session 1 vs session 3'); plt.axis('off')         

#%% compute union between sessions 1 and 2 (just for plotting purposes)

A_13, ass_13, mat_13 = register_multisession([A1, A3], dims, [Cn1, Cn3])
