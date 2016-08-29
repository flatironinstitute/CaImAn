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
images=np.reshape(Yr.T,[T]+list(dims),order='F')
Y=np.reshape(Yr,dims+(T,),order='F')
#%%
Cn = cse.utilities.local_correlations(Y[:,:,:3000])
pl.imshow(Cn,cmap='gray')  
#%%
rf=10 # half-size of the patches in pixels. rf=25, patches are 50x50
stride = 4 #amounpl.it of overlap between the patches in pixels    
K=4 # number of neurons expected per patch
gSig=[7,7] # expected half size of neurons
merge_thresh=0.8 # merging threshold, max correlation allowed
p=2 #order of the autoregressive system
memory_fact=1; #unitless number accounting how much memory should be used. You will need to try different values to see which one would work the default is OK for a 16 GB system
save_results=False
#%% RUN ALGORITHM ON PATCHES
cnmf=cse.CNMF(n_processes, k=K,gSig=gSig,merge_thresh=0.8,p=0,dview=dview,Ain=None,rf=rf,stride=stride, memory_fact=memory_fact,\
        method_init='sparse_nmf',alpha_snmf=10e2)
cnmf=cnmf.fit(images)

A_tot=cnmf.A
C_tot=cnmf.C
YrA_tot=cnmf.YrA
b_tot=cnmf.b
f_tot=cnmf.f
sn_tot=cnmf.sn

print 'Number of components:' + str(A_tot.shape[-1])
#%% if you have many components this might take long!
pl.figure()
crd = cse.utilities.plot_contours(A_tot,Cn,thr=0.9)
#%%
cnmf=cse.CNMF(n_processes, k=A_tot.shape,gSig=gSig,merge_thresh=merge_thresh,p=p,dview=dview,Ain=A_tot,Cin=C_tot,\
                 f_in=f_tot, rf=None,stride=None)
cnmf=cnmf.fit(images)
#%%
A=cnmf.A
C=cnmf.C
YrA=cnmf.YrA
b=cnmf.b
f=cnmf.f
sn=cnmf.sn
#%% get rid of evenrually noisy components. 
# But check by visual inspection to have a feeling fot the threshold. Try to be loose, you will be able to get rid of more of them later!
traces=C+YrA
idx_components, fitness, erfc = cse.utilities.evaluate_components(traces,N=5,robust_std=False)
idx_components=idx_components[np.logical_and(True ,fitness < -30)]
print(len(idx_components))
cse.utilities.view_patches_bar(Yr,scipy.sparse.coo_matrix(A.tocsc()[:,idx_components]),C[idx_components,:],b,f, d1,d2, YrA=YrA[idx_components,:]
                ,img=Cn)  
#%%
A=A.tocsc()[:,idx_components]
C=C[idx_components,:]   

#%% display components  DO NOT RUN IF YOU HAVE TOO MANY COMPONENTS
pl.figure()
crd = cse.utilities.plot_contours(A,Cn,thr=0.9)

