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
n_processes = np.maximum(np.int(psutil.cpu_count()*.75),1) # roughly number of cores on your machine minus 1
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
#%% FOR LOADING ALL TIFF FILES IN A FILE AND SAVING THEM ON A SINGLE MEMORY MAPPABLE FILE
fnames=[]
base_folder='./movies/' # folder containing the demo files
for file in glob.glob(os.path.join(base_folder,'*.tif')):
    if file.endswith("ie.tif"):
        fnames.append(os.path.abspath(file))
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
name_new=cse.utilities.save_memmap_each(fnames, dview=dview,base_name='Yr', resize_fact=(1, 1, 1), remove_init=0, idx_xy=None)
name_new.sort()
#%%
fname_new=cse.utilities.save_memmap_join(name_new,base_name='Yr', n_chunks=6, dview=dview)
#%%
#fname_new='Yr_d1_501_d2_398_d3_1_order_F_frames_369_.mmap'
Yr,dims,T=cse.utilities.load_memmap(fname_new)
images=np.reshape(Yr.T,[T]+list(dims),order='F')
Y=np.transpose(images,[1,2,0])
#%%
cnmf=cse.CNMF(n_processes, k=30,gSig=[7,7],merge_thresh=0.8,p=2,dview=dview,Ain=None)
cnmf=cnmf.fit(images)
#%%
Cn = cse.utilities.local_correlations(Y[:,:,:])
pl.imshow(Cn,cmap='gray')    
#%%
crd = cse.utilities.plot_contours(cnmf.A,Cn)
#%%
traces=cnmf.C+cnmf.YrA
idx_components, fitness, erfc = cse.utilities.evaluate_components(traces,N=5,robust_std=True)
idx_components=idx_components[fitness<-10]
#%%
A,C,b,f,YrA=cnmf.A,cnmf.C,cnmf.b,cnmf.f,cnmf.YrA
cse.utilities.view_patches_bar(Yr,scipy.sparse.coo_matrix(A.tocsc()[:,idx_components]),C[idx_components,:],b,f, dims[0],dims[1], YrA=YrA[idx_components,:],img=Cn)  
#%% visualize components
#pl.figure();
#crd = cse.utilities.plot_contours(A2.tocsc()[:,idx_components],Cn,thr=0.9)
#%% STOP CLUSTER
pl.close()
if not single_thread:    
    c.close()
    cse.utilities.stop_server()

#%% select  blobs
#min_radius=5 # min radius of expected blobs
#masks_ws,pos_examples,neg_examples=cse.utilities.extract_binary_masks_blob(A.tocsc()[:,:], 
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
#pl.imshow(np.reshape(A.tocsc()[:,neg_examples].mean(1),dims, order='F'))
##%%
#pl.imshow(np.reshape(A.tocsc()[:,pos_examples].mean(1),dims, order='F'))