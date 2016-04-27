# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 18:39:45 2016

@author: agiovann
"""
#%%
import os
import ca_source_extraction as cse 
import calblitz as cb
import time
import psutil
import sys
import scipy
import pylab as pl
import numpy as np

%load_ext autoreload
%autoreload 2

n_processes = np.maximum(psutil.cpu_count() - 1,1) # roughly number of cores on your machine minus 1
#print 'using ' + str(n_processes) + ' processes'
print "Stopping  cluster to avoid unnencessary use of memory...."
sys.stdout.flush()  
cse.utilities.stop_server() 
cse.utilities.start_server(n_processes)
#%%
#file_name='/home/agiovann/Dropbox (Simons Foundation)/Jeff/SmallExample/Yr_small.npy'
#file_name='/home/agiovann/Dropbox (Simons Foundation)/Jeff/SmallExample/Yr.npy'
#file_name='/home/agiovann/Dropbox (Simons Foundation)/Jeff/challenge/Yr.npy'

#file_name='/home/agiovann/GIT/Constrained_NMF/Yr.npy'

fnames=[]
for file in os.listdir("./"):
    if file.endswith(".hdf5"):
        fnames.append(file)
fnames.sort()
print fnames  
fnames=[fnames[0]]
#%%
fraction_downsample=.2;
idx_x=slice(150,350,None)
idx_y=slice(250,450,None)
fname_new=cse.utilities.save_memmap(fnames,base_name='Yr',resize_fact=(1,1,fraction_downsample),remove_init=30,idx_xy=(idx_x,idx_y))
#%%
fraction_downsample=.2;
fname_new=cse.utilities.save_memmap(fnames,base_name='Yr',remove_init=30*fraction_downsample)

#%%
Yr,d1,d2,T=cse.utilities.load_memmap(fname_new)
#%%
m=cb.movie(np.array(cb.to_3D(Yr.T,[T,d1,d2])),fr=30)
#%%
d,T=np.shape(Yr)
Y=np.reshape(Yr,(d1,d2,T),order='F')
merge_thresh=0.95
rf=15
stride = 2   
K=10
gSig=[5,5]

options_patch = cse.utilities.CNMFSetParms(Y,p=0,gSig=gSig,K=K)

options_patch['preprocess_params']['backend']='single_thread' 
options_patch['temporal_params']['fudge_factor'] = .96
options_patch['preprocess_params']['n_pixels_per_process']=np.int((rf*rf*4)/n_processes/(T/2000.))
options_patch['spatial_params']['n_pixels_per_process']=np.int((rf*rf*4)/n_processes/(T/2000.))
options_patch['temporal_params']['n_pixels_per_process']=np.int((rf*rf*4)/n_processes/(T/2000.))
options_patch['temporal_params']['p'] = 0 # set this to zero for fast updating without deconvolution

A_tot,C_tot,sn_tot, optional_outputs = cse.map_reduce.run_CNMF_patches(fname_new, (d1, d2, T), options_patch,rf=rf,stride = stride,
                                                                       n_processes=n_processes, backend='ipyparallel')

np.savez('results_analysis_patch.npz',A_tot=A_tot.todense(), C_tot=C_tot, sn_tot=sn_tot,d1=d1,d2=d2)    
#%%
if 1:
    Cn=cse.utilities.local_correlations(Y[:,:,:4000])
else:
    Cn=np.mean(Y[:,:,:4000],axis=-1)
#%%
crd = cse.utilities.plot_contours(A_tot,Cn,thr=0.9)
#%%

options = cse.utilities.CNMFSetParms(Y,p=2,gSig=gSig,K=A_tot.shape[-1])
pix_proc=np.minimum(np.int((d1*d2)/n_processes/(T/2000.)),np.int((d1*d2)/n_processes))
options['preprocess_params']['n_pixels_per_process']=pix_proc
options['spatial_params']['n_pixels_per_process']=pix_proc
options['temporal_params']['n_pixels_per_process']=pix_proc


options['temporal_params']['p'] = 0              
A_m,C_m,nr_m,merged_ROIs,S_m,bl_m,c1_m,sn_m,g_m=cse.merge_components(Yr,A_tot,[],np.array(C_tot),[],np.array(C_tot),[],options['temporal_params'],options['spatial_params'],thr=merge_thresh,mx=np.Inf)     

#%%
crd = cse.utilities.plot_contours(A_m,Cn,thr=0.9)
#%%
pixels_bckgrnd=np.nonzero(A_m.sum(axis=-1)==0)[0]
f=np.sum(Yr[pixels_bckgrnd,:],axis=0)
#%%
options['spatial_params']['backend']='ipyparallel'
t1 = time.time()
A2,b2,C2 = cse.spatial.update_spatial_components(Yr, C_m, f, A_m, sn=sn_tot, **options['spatial_params'])
print time.time() - t1
#%%
crd = cse.utilities.plot_contours(A2,Cn,thr=0.9)
#%%
#Cn = cse.utilities.local_correlations(Y)
#crd = cse.utilities.plot_contours(A2,Cn,thr=0.9)

options['temporal_params']['p'] = 0              
C2,f2,S2,bl2,c12,neurons_sn2,g21,YrA = cse.temporal.update_temporal_components(Yr,A2,b2,C2,f,bl=None,c1=None,sn=None,g=None,**options['temporal_params'])

A_or, C_or, srt = cse.utilities.order_components(A2,C2)
#cse.utilities.view_patches_bar(Yr,scipy.sparse.coo_matrix(A_or),C_or,b2,f2, d1,d2, YrA=YrA[srt,:])  

np.savez('results_analysis.npz',Cn=Cn,A_tot=A_tot.todense(), C_tot=C_tot, sn_tot=sn_tot, A2=A2.todense(),C2=C2,b2=b2,S2=S2,f2=f2,bl2=bl2,c12=c12, neurons_sn2=neurons_sn2, g21=g21,YrA=YrA,d1=d1,d2=d2)    
scipy.io.savemat('output_analysis_matlab.mat',{'A_or':A_or,'C_or':C_or , 'YrA_or':YrA[srt,:], 'S_or': S2[srt,:] })

#cse.utilities.stop_server() 
#%%
crd = cse.utilities.plot_contours(A_or,Cn,thr=0.9)
#%%
cse.utilities.view_patches_bar(Yr,scipy.sparse.coo_matrix(A_or),C_or,b2,f2, d1,d2, YrA=YrA[srt,:])  