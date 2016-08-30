# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 18:39:45 2016

@author: Andrea Giovannucci

For explanation consult at https://github.com/agiovann/Constrained_NMF/releases/download/v0.4-alpha/Patch_demo.zip
and https://github.com/agiovann/Constrained_NMF

"""
#%%
try:
    %load_ext autoreload
    %autoreload 2
    print 1
except:
    print 'NOT IPYTHON'

import matplotlib as mpl
mpl.use('TKAgg')
from matplotlib import pyplot as plt
#plt.ion()

import sys
import numpy as np
import ca_source_extraction as cse

#sys.path.append('../SPGL1_python_port')
#%
from time import time
from scipy.sparse import coo_matrix
import tifffile
import subprocess
import time as tm
from time import time
import pylab as pl
import psutil
import glob
import os
import scipy
from ipyparallel import Client
#%%
#backend='SLURM'
backend='local'
if backend == 'SLURM':
    n_processes = np.int(os.environ.get('SLURM_NPROCS'))
else:
    n_processes = np.maximum(np.int(psutil.cpu_count()),1) # roughly number of cores on your machine minus 1
print 'using ' + str(n_processes) + ' processes'
#%% start cluster for efficient computation
single_thread=False

if single_thread:
    dview=None
else:    
    try:
        c.close()
    except:
        print 'C was not existing, creating one'
    print "Stopping  cluster to avoid unnencessary use of memory...."
    sys.stdout.flush()  
    if backend == 'SLURM':
        try:
            cse.utilities.stop_server(is_slurm=True)
        except:
            print 'Nothing to stop'
        slurm_script='/mnt/xfs1/home/agiovann/SOFTWARE/Constrained_NMF/SLURM/slurmStart.sh'
        cse.utilities.start_server(slurm_script=slurm_script)
        pdir, profile = os.environ['IPPPDIR'], os.environ['IPPPROFILE']
        c = Client(ipython_dir=pdir, profile=profile)        
    else:
        cse.utilities.stop_server()
        cse.utilities.start_server()        
        c=Client()

    print 'Using '+ str(len(c)) + ' processes'
    dview=c[:len(c)]
#%% FOR LOADING ALL TIFF FILES IN A FILE AND SAVING THEM ON A SINGLE MEMORY MAPPABLE FILE
fnames=[]
base_folder='./movies' # folder containing the demo files
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

#%%
#idx_x=slice(12,500,None)
#idx_y=slice(12,500,None)
#idx_xy=(idx_x,idx_y)
downsample_factor=1 # use .2 or .1 if file is large and you want a quick answer
idx_xy=None
base_name='Yr'
name_new=cse.utilities.save_memmap_each(fnames, dview=dview,base_name=base_name, resize_fact=(1, 1, downsample_factor), remove_init=0,idx_xy=idx_xy )
name_new.sort(key=lambda fn: np.int(fn[fn.find(base_name)+len(base_name):fn.find('_')]))
print name_new
#%%
n_chunks=6 # increase this number if you have memory issues at this point
fname_new=cse.utilities.save_memmap_join(name_new,base_name='Yr', n_chunks=6, dview=dview)
#%%  Create a unique file fot the whole dataset
##
#fraction_downsample=1; # useful to downsample the movie across time. fraction_downsample=.1 measn downsampling by a factor of 10
#fname_new=cse.utilities.save_memmap(fnames,base_name='Yr',resize_fact=(1,1,fraction_downsample),order='F')
#%%

#%%
#fname_new='Yr_d1_501_d2_398_d3_1_order_F_frames_369_.mmap'
Yr,dims,T=cse.utilities.load_memmap(fname_new)
d1,d2=dims
Y=np.reshape(Yr,dims+(T,),order='F')
#%%
Cn = cse.utilities.local_correlations(Y[:,:,:3000])
pl.imshow(Cn,cmap='gray')  
#%%
rf=10 # half-size of the patches in pixels. rf=25, patches are 50x50
stride = 2 #amounpl.it of overlap between the patches in pixels    
K=3 # number of neurons expected per patch
gSig=[7,7] # expected half size of neurons
merge_thresh=0.8 # merging threshold, max correlation allowed
p=2 #order of the autoregressive system
memory_fact=1; #unitless number accounting how much memory should be used. You will need to try different values to see which one would work the default is OK for a 16 GB system
save_results=False
#%% RUN ALGORITHM ON PATCHES
options_patch = cse.utilities.CNMFSetParms(Y,n_processes,p=0,gSig=gSig,K=K,ssub=1,tsub=4,thr=merge_thresh)
A_tot,C_tot,YrA_tot,b,f,sn_tot, optional_outputs = cse.map_reduce.run_CNMF_patches(fname_new, (d1, d2, T), options_patch,rf=rf,stride = stride,
                                                                        dview=dview,memory_fact=memory_fact)
print 'Number of components:' + str(A_tot.shape[-1])      
#%%
if save_results:
    np.savez('results_analysis_patch.npz',A_tot=A_tot.todense(), C_tot=C_tot, sn_tot=sn_tot,d1=d1,d2=d2,b=b,f=f)    
#%% if you have many components this might take long!
pl.figure()
crd = cse.utilities.plot_contours(A_tot,Cn,thr=0.9)
#%% set parameters for full field of view analysis
options = cse.utilities.CNMFSetParms(Y,n_processes,p=0,gSig=gSig,K=A_tot.shape[-1],thr=merge_thresh)
pix_proc=np.minimum(np.int((d1*d2)/n_processes/(T/2000.)),np.int((d1*d2)/n_processes)) # regulates the amount of memory used
options['spatial_params']['n_pixels_per_process']=pix_proc
options['temporal_params']['n_pixels_per_process']=pix_proc
#%% merge spatially overlaping and temporally correlated components      
A_m,C_m,nr_m,merged_ROIs,S_m,bl_m,c1_m,sn_m,g_m=cse.merge_components(Yr,A_tot,[],np.array(C_tot),[],np.array(C_tot),[],options['temporal_params'],options['spatial_params'],dview=dview,thr=options['merging']['thr'],mx=np.Inf)     
#%% update temporal to get Y_r
options['temporal_params']['p']=0
options['temporal_params']['fudge_factor']=0.96 #change ifdenoised traces time constant is wrong
options['temporal_params']['backend']='ipyparallel'
C_m,f_m,S_m,bl_m,c1_m,neurons_sn_m,g2_m,YrA_m = cse.temporal.update_temporal_components(Yr,A_m,np.atleast_2d(b).T,C_m,f,dview=dview,bl=None,c1=None,sn=None,g=None,**options['temporal_params'])

#%% get rid of evenrually noisy components. 
# But check by visual inspection to have a feeling fot the threshold. Try to be loose, you will be able to get rid of more of them later!

traces=C_m+YrA_m
idx_components, fitness, erfc,r_values,num_significant_samples = cse.utilities.evaluate_components(Y,traces,A_m,N=5,robust_std=False)
idx_components = idx_components[np.logical_and(np.array(num_significant_samples)>0 ,np.array(r_values)>=.5)]

print(len(idx_components))
cse.utilities.view_patches_bar(Yr,scipy.sparse.coo_matrix(A_m.tocsc()[:,idx_components]),C_m[idx_components,:],b,f_m, d1,d2, YrA=YrA_m[idx_components,:]
                ,img=Cn)  
#%%
A_m=A_m[:,idx_components]
C_m=C_m[idx_components,:]   

#%% display components  DO NOT RUN IF YOU HAVE TOO MANY COMPONENTS
pl.figure()
crd = cse.utilities.plot_contours(A_m,Cn,thr=0.9)
#%%
print 'Number of components:' + str(A_m.shape[-1])  
#%% UPDATE SPATIAL OCMPONENTS
t1 = time()
A2,b2,C2 = cse.spatial.update_spatial_components(Yr, C_m, f, A_m, sn=sn_tot,dview=dview, **options['spatial_params'])
print time() - t1
#%% UPDATE TEMPORAL COMPONENTS
options['temporal_params']['p']=p
options['temporal_params']['fudge_factor']=0.96 #change ifdenoised traces time constant is wrong
C2,f2,S2,bl2,c12,neurons_sn2,g21,YrA = cse.temporal.update_temporal_components(Yr,A2,b2,C2,f,dview=dview, bl=None,c1=None,sn=None,g=None,**options['temporal_params'])
#%% Order components
#A_or, C_or, srt = cse.utilities.order_components(A2,C2)
#%% stop server and remove log files
cse.utilities.stop_server(is_slurm = (backend == 'SLURM')) 
log_files=glob.glob('Yr*_LOG_*')
for log_file in log_files:
    os.remove(log_file)
#%% order components according to a quality threshold and only select the ones wiht qualitylarger than quality_threshold. 
quality_threshold=-20
traces=C2+YrA
traces=C2+YrA
idx_components, fitness, erfc,r_values,num_significant_samples = cse.utilities.evaluate_components(Y,traces,A2,N=5,robust_std=False)

sure_in_idx= idx_components[np.logical_and(np.array(num_significant_samples)>1 ,np.array(r_values)>=.5)]
doubtful = idx_components[np.logical_and(np.array(num_significant_samples)==1 ,np.array(r_values)>=.5)]
they_suck = idx_components[np.logical_and(np.array(num_significant_samples)>=0 ,np.array(r_values)<.5)]
#%%
cse.utilities.view_patches_bar(Yr,scipy.sparse.coo_matrix(A2.tocsc()[:,sure_in_idx]),C2[sure_in_idx,:],b2,f2, dims[0],dims[1], YrA=YrA[sure_in_idx,:],img=Cn)  
#%% visualize components
#pl.figure();
pl.subplot(1,3,1)
crd = cse.utilities.plot_contours(A2.tocsc()[:,sure_in_idx],Cn,thr=0.9)
pl.subplot(1,3,2)
crd = cse.utilities.plot_contours(A2.tocsc()[:,doubtful],Cn,thr=0.9)
pl.subplot(1,3,3)
crd = cse.utilities.plot_contours(A2.tocsc()[:,they_suck],Cn,thr=0.9)
#%% save analysis results in python and matlab format
if save_results:
    np.savez('results_analysis.npz',Cn=Cn,A_tot=A_tot.todense(), C_tot=C_tot, sn_tot=sn_tot, A2=A2.todense(),C2=C2,b2=b2,S2=S2,f2=f2,bl2=bl2,c12=c12, neurons_sn2=neurons_sn2, g21=g21,YrA=YrA,d1=d1,d2=d2,idx_components=idx_components, fitness=fitness, erfc=erfc)    
    scipy.io.savemat('output_analysis_matlab.mat',{'A2':A2,'C2':C2 , 'YrA':YrA, 'S2': S2 ,'YrA': YrA, 'd1':d1,'d2':d2,'idx_components':idx_components, 'fitness':fitness })
#%% select  blobs
#min_radius=5 # min radius of expected blobs
#masks_ws,pos_examples,neg_examples=cse.utilities.extract_binary_masks_blob(A2.tocsc()[:,:], 
#     min_radius, dims, minCircularity= 0.5, minInertiaRatio = 0.2,minConvexity = .8)
#
#pl.subplot(1,2,1)
#
#final_masks=np.array(masks_ws)[pos_examples]
#pl.imshow(np.reshape(final_masks.max(0),dims,order='F'),vmax=1)
#pl.subplot(1,2,2)
#
#neg_examples_masks=np.array(masks_ws)[neg_examples]
#pl.imshow(np.reshape(neg_examples_masks.max(0),dims,order='F'),vmax=1)
##%%
#pl.imshow(np.reshape(A2.tocsc()[:,neg_examples].mean(1),dims, order='F'))
##%%
#pl.imshow(np.reshape(A2.tocsc()[:,pos_examples].mean(1),dims, order='F'))
#
##%%
#
#cse.utilities.view_patches_bar(Yr,scipy.sparse.coo_matrix(A2.tocsc()[:,pos_examples]),C2[pos_examples,:],b2,f2, dims[0],dims[1], YrA=YrA[pos_examples,:],img=Cn)  
#

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
    
    fname_new='Yr0_d1_60_d2_80_d3_1_order_C_frames_2000_.mmap'
    
    Yr,(d1,d2),T=cse.utilities.load_memmap(fname_new)
    d,T=np.shape(Yr)
    Y=np.reshape(Yr,(d1,d2,T),order='F') # 3D version of the movie
    
    
    traces=C2+YrA
    idx_components, fitness, erfc,r_values,num_significant_samples  = cse.utilities.evaluate_components(traces,N=5,robust_std=False)
    #cse.utilities.view_patches(Yr,coo_matrix(A_or),C_or,b2,f2,d1,d2,YrA = YrA[srt,:], secs=1)
    cse.utilities.view_patches_bar(Yr,scipy.sparse.coo_matrix(A2[:,idx_components]),C2[idx_components,:],b2,f2, d1,d2, YrA=YrA[idx_components,:])  
