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
    cse.utilities.stop_server()
    cse.utilities.start_server()
    c=Client()
    dview=c[:n_processes]

#%%
base_folder='./'
#%%
#fname_new='Yr_d1_501_d2_398_d3_1_order_F_frames_369_.mmap'
Yr,dims,T=cse.utilities.load_memmap(fname_new)
Y=np.reshape(Yr,dims+(T,),order='F')
#%%
Cn = cse.utilities.local_correlations(Y[:,:,:])
pl.imshow(Cn,cmap='gray')  
#%%  
rois_1=np.transpose(cse.utilities.nf_read_roi_zip(base_folder+'regions/ben_regions.zip',np.shape(Cn)),[1,2,0])
Ain=np.reshape(rois_1,[np.prod(np.shape(Cn)),-1],order='F')
#%%
K=Ain.shape[-1] # number of neurons expected per patch
gSig=[7,7] # expected half size of neurons
merge_thresh=1 # merging threshold, max correlation allowed
p=1 #order of the autoregressive system
options = cse.utilities.CNMFSetParms(Y,n_processes,p=p,gSig=gSig,K=K,ssub=2,tsub=2)
#%% PREPROCESS DATA AND INITIALIZE COMPONENTS
t1 = time()
Yr,sn,g,psx = cse.pre_processing.preprocess_data(Yr,dview=dview,**options['preprocess_params'])
print time() - t1

##%%
#t1 = time()
#Atmp, Ctmp, b_in, f_in, center=cse.initialization.initialize_components(Y, normalize=True, **options['init_params'])                                                    
#print time() - t1
#
##%% Refine manually component by clicking on neurons 
#refine_components=False
#if refine_components:
#    Ain,Cin = cse.utilities.manually_refine_components(Y,options['init_params']['gSig'],coo_matrix(Atmp),Ctmp,Cn,thr=0.9)
#else:
#    Ain,Cin = Atmp, Ctmp
#%% plot estimated component
pl.figure()
crd = cse.utilities.plot_contours(coo_matrix(Ain),Cn)  
pl.show()
#%% UPDATE SPATIAL COMPONENTS
pl.figure()
t1 = time()
A,b,Cin,f = cse.spatial.update_spatial_components(Yr, C=None, f=None, A_in=Ain.astype(np.bool), sn=sn, dview=None,**options['spatial_params'])
t_elSPATIAL = time() - t1
print t_elSPATIAL 
pl.figure()
crd = cse.utilities.plot_contours(A,Cn)


#%% update_temporal_components
#pl.close()
t1 = time()
options['temporal_params']['p'] = 0 # set this to zero for fast updating without deconvolution
C,f,S,bl,c1,neurons_sn,g,YrA = cse.temporal.update_temporal_components(Yr,A,b,Cin,f,dview=dview,bl=None,c1=None,sn=None,g=None,**options['temporal_params'])
t_elTEMPORAL = time() - t1
print t_elTEMPORAL 

#%%
traces=C+YrA
#traces=traces-scipy.ndimage.filters.percentile_filter(traces,20,(1,1000))
traces=traces-scipy.signal.savgol_filter(traces,np.shape(traces)[1]/2*2-1,1,axis=1)

idx_components, fitness, erfc = cse.utilities.evaluate_components(traces,N=5,robust_std=True)
idx_components=idx_components[fitness>-35]
print len(idx_components)
print np.shape(A)
#%%

cse.utilities.view_patches_bar(Yr,scipy.sparse.coo_matrix(A.tocsc()[:,idx_components]),C[idx_components,:],b,f, dims[0],dims[1], YrA=YrA[idx_components,:],img=Cn)  


#%% refine spatial and temporal 
pl.close()
t1 = time()
A2,b2,C2 = cse.spatial.update_spatial_components(Yr, C, f, A, sn=sn,dview=dview, **options['spatial_params'])
options['temporal_params']['p'] = p # set it back to original value to perform full deconvolution
C2,f2,S2,bl2,c12,neurons_sn2,g21,YrA = cse.temporal.update_temporal_components(Yr,A2,b2,C2,f,dview=dview, bl=None,c1=None,sn=None,g=None,**options['temporal_params'])
print time() - t1
#%%
plt.figure()
crd = cse.plot_contours(A2,Cn,thr=0.9)
#%%

traces=C2+YrA
traces=traces-scipy.signal.savgol_filter(traces,np.shape(traces)[1]/2*2-1,1,axis=1)
idx_components, fitness, erfc = cse.utilities.evaluate_components(traces,N=5,robust_std=True)
idx_components=idx_components[fitness<-15]
#%%
cse.utilities.view_patches_bar(Yr,scipy.sparse.coo_matrix(A2.tocsc()[:,idx_components]),C2[idx_components,:],b2,f2, dims[0],dims[1], YrA=YrA[idx_components,:],img=Cn)  
#%% visualize components
#pl.figure();
#crd = cse.utilities.plot_contours(A2.tocsc()[:,idx_components],Cn,thr=0.9)

#%% STOP CLUSTER
pl.close()
if not single_thread:    
    c.close()
    cse.utilities.stop_server()

#%% select  blobs
min_radius=5 # min radius of expected blobs
masks_ws,pos_examples,neg_examples=cse.utilities.extract_binary_masks_blob(A2.tocsc()[:,:], 
     min_radius, dims, minCircularity= 0.5, minInertiaRatio = 0.2,minConvexity = .8)

pl.subplot(1,2,1)

final_masks=np.array(masks_ws)[pos_examples]
pl.imshow(np.reshape(final_masks.max(0),dims,order='F'),vmax=1)
pl.subplot(1,2,2)

neg_examples_masks=np.array(masks_ws)[neg_examples]
pl.imshow(np.reshape(neg_examples_masks.max(0),dims,order='F'),vmax=1)
#%%
pl.imshow(np.reshape(A2.tocsc()[:,neg_examples].mean(1),dims, order='F'))
#%%
pl.imshow(np.reshape(A2.tocsc()[:,pos_examples].mean(1),dims, order='F'))

#%%
cse.utilities.view_patches_bar(Yr,scipy.sparse.coo_matrix(A2.tocsc()[:,pos_examples]),C2[pos_examples,:],b2,f2, dims[0],dims[1], YrA=YrA[pos_examples,:],img=Cn)  
#%%
cse.utilities.view_patches_bar(Yr,scipy.sparse.coo_matrix(A2.tocsc()[:,neg_examples]),C2[neg_examples,:],b2,f2, dims[0],dims[1], YrA=YrA[neg_examples,:],img=Cn)  
