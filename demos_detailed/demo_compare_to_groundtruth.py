#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 16:02:27 2017

@author: agiovann
"""

%% LOAD DATA
params_display = {
    'downsample_ratio': .2,
    'thr_plot': 0.8
}


analysis_file = '/mnt/ceph/neuro/jeremie_analysis/neurofinder.03.00.test/Yr_d1_498_d2_467_d3_1_order_C_frames_2250_._results_analysis.npz'
with np.load(analysis_file) as ld:
    print(ld.keys())
    locals().update(ld) 
    dims_off = d1,d2    
    A = scipy.sparse.coo_matrix(A[()])
    dims = (d1,d2)
    
#%%    
idx_components_r = np.where((r_values >= .85))[0]
idx_components_raw = np.where(fitness_raw < -40)[0]
idx_components_delta = np.where(fitness_delta < -40)[0]    
#idx_and_condition_1 = np.where((r_values >= .65) & ((fitness_raw < -20) | (fitness_delta < -20)) )[0]

idx_components = np.union1d(idx_components_r, idx_components_raw)
idx_components = np.union1d(idx_components, idx_components_delta)
#idx_components = np.union1d(idx_components, idx_and_condition_1)
#idx_components = np.union1d(idx_components, idx_and_condition_2)

#idx_blobs = np.intersect1d(idx_components, idx_blobs)
idx_components_bad = np.setdiff1d(list(range(len(r_values))), idx_components)

print(' ***** ')
print((len(r_values)))
print((len(idx_components)))  
#%%
pl.subplot(1, 2, 1)
crd = plot_contours(A.tocsc()[:, idx_components], Cn, thr=params_display['thr_plot'], vmax = 0.35)
pl.subplot(1, 2, 2)
crd = plot_contours(A.tocsc()[:, idx_components_bad], Cn, thr=params_display['thr_plot'], vmax = 0.35)
#%%
c, dview, n_processes = cm.cluster.setup_cluster(
    backend='local', n_processes=None, single_thread=False)
#%% thredshold components
min_size_neuro = 5**2*np.pi
max_size_neuro = 15**2*np.pi
A_thr = cm.source_extraction.cnmf.spatial.threshold_components(A.tocsc()[:,idx_components].toarray(), dims, medw=None, thr_method='max', maxthr=0.2, nrgthr=0.99, extract_cc=True,
                         se=None, ss=None, dview=dview) 

A_thr = A_thr > 0  
size_neurons = A_thr.sum(0)
A_thr = A_thr[:,(size_neurons>min_size_neuro) & (size_neurons<max_size_neuro)]
print(A_thr.shape)
#%%
crd = plot_contours(scipy.sparse.coo_matrix(A_thr*1.), Cn, thr=.99, vmax = 0.35)
#%%
roi_cons = np.load('/mnt/ceph/neuro/labeling/neurofinder.03.00.test/regions/joined_consensus_active_regions.npy')
print(roi_cons.shape)
pl.imshow(roi_cons.sum(0))
#%%
pl.figure(figsize=(30,20))
tp_gt, tp_comp, fn_gt, fp_comp, performance_cons_off =  cm.base.rois.nf_match_neurons_in_binary_masks(roi_cons,A_thr[:,:].reshape([dims[0],dims[1],-1],order = 'F').transpose([2,0,1])*1.,thresh_cost=.7, min_dist = 10,
                                                                              print_assignment= False,plot_results=True,Cn=Cn, labels = ['GT','Offline'])
pl.rcParams['pdf.fonttype'] = 42
font = {'family' : 'Myriad Pro',
        'weight' : 'regular',
        'size'   : 20}
pl.rc('font', **font)