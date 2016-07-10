# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 18:39:45 2016

@author: Andrea Giovannucci

For explanation consult at https://github.com/agiovann/Constrained_NMF/releases/download/v0.4-alpha/Patch_demo.zip
and https://github.com/agiovann/Constrained_NMF

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
import glob

%load_ext autoreload
%autoreload 2


#%% select all tiff files in current folder
fnames=[]
base_folder='./movies/' # folder containing the demo files
for file in glob.glob(os.path.join(base_folder,'*.tif')):
    if file.endswith(".tif"):
        fnames.append(file)
fnames.sort()
print fnames  
fnames=fnames
#%% Create a unique file fot the whole dataset
# THIS IS  ONLY IF YOU NEED TO SELECT A SUBSET OF THE FIELD OF VIEW 
#fraction_downsample=1;
#idx_x=slice(10,502,None)
#idx_y=slice(10,502,None)
#fname_new=cse.utilities.save_memmap(fnames,base_name='Yr',resize_fact=(1,1,fraction_downsample),remove_init=0,idx_xy=(idx_x,idx_y))
#%%  Memory Map. Create a unique file for the whole dataset
fraction_downsample=1; # useful to downsample the movie across time. fraction_downsample=.1 measn downsampling by a factor of 10
fname_new=cse.utilities.save_memmap(fnames,base_name='Yr',resize_fact=(1,1,fraction_downsample))
#%%
Yr,(d1,d2),T=cse.utilities.load_memmap(fname_new)
d,T=np.shape(Yr)
Y=np.reshape(Yr,(d1,d2,T),order='F') # 3D version of the movie
#%% build an image to check the presence of neurons
corr_image=1
if corr_image:
    # build correlation image
    Cn=cse.utilities.local_correlations(Y[:,:,:3000])
else:
    # build mean image
    Cn=np.mean(Y[:,:,:memory_fact*10000],axis=-1)
    
Cn[np.isnan(Cn)]=0
#%% USE this visualization to establish how large are neurons and how many neurons do you expect in a patch
pl.imshow(Cn,cmap='gray',vmin=np.percentile(Cn, 1), vmax=np.percentile(Cn, 99))    
#%%
rf=10 # half-size of the patches in pixels. rf=25, patches are 50x50
stride = 2 #amounpl.it of overlap between the patches in pixels    
K=3 # number of neurons expected per patch
gSig=[4,4] # expected half size of neurons
merge_thresh=0.8 # merging threshold, max correlation allowed
p=2 #order of the autoregressive system
memory_fact=1; #unitless number accounting how much memory should be used. You will need to try different values to see which one would work the default is OK for a 16 GB system
save_results=False
#%% Start the cluster
n_processes = np.maximum(np.int(psutil.cpu_count()*.75),1) # roughly number of cores on your machine minus 1
print 'using ' + str(n_processes) + ' processes'
print "Restarting cluster to avoid unnencessary use of memory...."
sys.stdout.flush()  
cse.utilities.stop_server() 
cse.utilities.start_server(n_processes)
#%% RUN ALGORITHM ON PATCHES
options_patch = cse.utilities.CNMFSetParms(Y,n_processes,p=0,gSig=gSig,K=K,ssub=1,tsub=4,thr=merge_thresh)
A_tot,C_tot,b,f,sn_tot, optional_outputs = cse.map_reduce.run_CNMF_patches(fname_new, (d1, d2, T), options_patch,rf=rf,stride = stride,
                                                                       n_processes=n_processes, backend='ipyparallel',memory_fact=memory_fact)
print 'Number of components:' + str(A_tot.shape[-1])      
#%%
if save_results:
    np.savez('results_analysis_patch.npz',A_tot=A_tot.todense(), C_tot=C_tot, sn_tot=sn_tot,d1=d1,d2=d2)    
#%% if you have many components this might take long!
pl.figure()
crd = cse.utilities.plot_contours(A_tot,Cn,thr=0.9)
#%% set parameters for full field of view analysis
options = cse.utilities.CNMFSetParms(Y,n_processes,p=0,gSig=gSig,K=A_tot.shape[-1],thr=merge_thresh)
pix_proc=np.minimum(np.int((d1*d2)/n_processes/(T/2000.)),np.int((d1*d2)/n_processes)) # regulates the amount of memory used
options['spatial_params']['n_pixels_per_process']=pix_proc
options['temporal_params']['n_pixels_per_process']=pix_proc
#%% merge spatially overlaping and temporally correlated components      
A_m,C_m,nr_m,merged_ROIs,S_m,bl_m,c1_m,sn_m,g_m=cse.merge_components(Yr,A_tot,[],np.array(C_tot),[],np.array(C_tot),[],options['temporal_params'],options['spatial_params'],thr=options['merging']['thr'],mx=np.Inf)     
#%% update temporal to get Y_r
options['temporal_params']['p']=0
options['temporal_params']['fudge_factor']=0.96 #change ifdenoised traces time constant is wrong
options['temporal_params']['backend']='ipyparallel'
C_m,f_m,S_m,bl_m,c1_m,neurons_sn_m,g2_m,YrA_m = cse.temporal.update_temporal_components(Yr,A_m,np.atleast_2d(b).T,C_m,f,bl=None,c1=None,sn=None,g=None,**options['temporal_params'])

#%%
traces=C_m+YrA_m
idx_components, fitness, erfc = cse.utilities.evaluate_components(traces,N=5,robust_std=False)
idx_components=idx_components[np.logical_and(True ,fitness < -10)]
print(len(idx_components))
cse.utilities.view_patches_bar(Yr,scipy.sparse.coo_matrix(A_m.tocsc()[:,idx_components]),C_m[idx_components,:],b,f_m, d1,d2, YrA=YrA_m[idx_components,:])  
#%%
A_m=A_m[:,idx_components]
C_m=C_m[idx_components,:]   

#%% display components  DO NOT RUN IF YOU HAVE TOO MANY COMPONENTS
pl.figure()
crd = cse.utilities.plot_contours(A_m,Cn,thr=0.9)
#%%
print 'Number of components:' + str(A_m.shape[-1])  
#%% UPDATE SPATIAL OCMPONENTS
options['spatial_params']['backend']='ipyparallel' #parallelize with ipyparallel
t1 = time.time()
A2,b2,C2 = cse.spatial.update_spatial_components(Yr, C_m, f, A_m, sn=sn_tot, **options['spatial_params'])
print time.time() - t1
#%% UPDATE TEMPORAL COMPONENTS
options['temporal_params']['p']=p
options['temporal_params']['fudge_factor']=0.96 #change ifdenoised traces time constant is wrong
options['temporal_params']['backend']='ipyparallel'
C2,f2,S2,bl2,c12,neurons_sn2,g21,YrA = cse.temporal.update_temporal_components(Yr,A2,b2,C2,f,bl=None,c1=None,sn=None,g=None,**options['temporal_params'])
#%% Order components
#A_or, C_or, srt = cse.utilities.order_components(A2,C2)
#%% stop server and remove log files
cse.utilities.stop_server() 
log_files=glob.glob('Yr*_LOG_*')
for log_file in log_files:
    os.remove(log_file)
#%% order components according to a quality threshold and only select the ones wiht qualitylarger than quality_threshold. 
#quality_threshold=0
traces=C2+YrA
idx_components, fitness, erfc = cse.utilities.evaluate_components(traces,N=5,robust_std=False)
#idx_components=idx_components[fitness<quality_threshold]
#print(idx_components.size*1./traces.shape[0])
#%% save analysis results in python and matlab format
if save_results:
    np.savez('results_analysis.npz',Cn=Cn,A_tot=A_tot.todense(), C_tot=C_tot, sn_tot=sn_tot, A2=A2.todense(),C2=C2,b2=b2,S2=S2,f2=f2,bl2=bl2,c12=c12, neurons_sn2=neurons_sn2, g21=g21,YrA=YrA,d1=d1,d2=d2,idx_components=idx_components, fitness=fitness, erfc=erfc)    
    scipy.io.savemat('output_analysis_matlab.mat',{'A2':A2,'C2':C2 , 'YrA':YrA, 'S2': S2 ,'YrA': YrA, 'd1':d1,'d2':d2,'idx_components':idx_components, 'fitness':fitness })
#%% 
cse.utilities.view_patches_bar(Yr,scipy.sparse.coo_matrix(A2.tocsc()[:,idx_components]),C2[idx_components,:],b2,f2, d1,d2, YrA=YrA[idx_components,:])  
#%%
# select only portion of components
pl.figure();
crd = cse.utilities.plot_contours(A2.tocsc()[:,idx_components],Cn,thr=0.9)

#%% RELOAD COMPONENTS!
if save_results:
    import sys
    import numpy as np
    import ca_source_extraction as cse
    from scipy.sparse import coo_matrix
    import scipy
    import pylab as pl
    import calblitz as cb
    
    
    
    with np.load('results_analysis.npz')  as ld:
          locals().update(ld)
    
    fname_new='Yr_d1_60_d2_80_d3_1_order_F_frames_2000_.mmap'
    
    Yr,(d1,d2),T=cse.utilities.load_memmap(fname_new)
    d,T=np.shape(Yr)
    Y=np.reshape(Yr,(d1,d2,T),order='F') # 3D version of the movie
    
    
    traces=C2+YrA
    idx_components, fitness, erfc = cse.utilities.evaluate_components(traces,N=5,robust_std=False)
    #cse.utilities.view_patches(Yr,coo_matrix(A_or),C_or,b2,f2,d1,d2,YrA = YrA[srt,:], secs=1)
    cse.utilities.view_patches_bar(Yr,scipy.sparse.coo_matrix(A2[:,idx_components]),C2[idx_components,:],b2,f2, d1,d2, YrA=YrA[idx_components,:])  
