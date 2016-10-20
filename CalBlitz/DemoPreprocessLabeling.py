# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 17:56:14 2016

@author: agiovann
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
import calblitz as cb
import shutil
import glob
from ipyparallel import Client
import os
import glob
import h5py
import re
#%% Sue Ann create data for labeling
diameter_bilateral_blur=4
median_filter_size=(2,1,1)
from scipy.ndimage import filters as ft
res=[]
with open('file_list.txt') as f:
    for ln in f:
        ln1=ln[:-1]
        print(ln1)
        with  h5py.File(ln1) as hh:
            print hh.keys()
            mov=np.array(hh['binnedF'],dtype=np.float32)
            mov=cb.movie(mov,fr=3)
            mov=mov.bilateral_blur_2D(diameter=diameter_bilateral_blur)
            mov1=cb.movie(ft.median_filter(mov,median_filter_size),fr=3)
            #mov1=mov1-np.median(mov1,0)
            mov1=mov1.resize(1,1,.3)
            mov1=mov1-cb.utils.mode_robust(mov1,0)
            mov=mov.resize(1,1,.3)
    #        mov=mov-np.percentile(mov,1)
            
            mov.save(ln1[:-3] + '_compress_.tif')
            mov1.save(ln1[:-3] + '_BL_compress_.tif')
            
            res.append(ln1[:-3] + '_compress_.tif')
            res.append(ln1[:-3] + '_BL_compress_.tif')
#%% extract correlation image

with open('file_list.txt') as f:
    for ln in f:
        
        ln1=ln[:ln.find('summary')]
        print(ln1)
        with  h5py.File(ln1 +'summary.mat') as hh:
            print hh.keys()
            mov=np.array(hh['binnedF'],dtype=np.float32)
            mov=cb.movie(mov,fr=3)
            img=cb.movie(mov.local_correlations(eight_neighbours=True),fr=1)
            
            img.save(ln1[:-3] + 'summary._correlation_image_full.tif')

#%%
with open('file_list.txt') as f:
    for ln in f:        
        ln1=ln[:ln.find('summary')]
        print(ln1)      
        ffinal=ln1[:-3] + 'summary._correlation_image_full.tif'
        shutil.copyfile(ffinal,os.path.join('./correlation_images/',os.path.split(ffinal)[-1]))                      
#%% Create images for Labeling
from glob import glob
import scipy.stats as st
import calblitz as cb
import numpy as np            
for fl  in res:            
    print(fl) 
    m=cb.load(fl,fr=3)
    
    img=m.local_correlations(eight_neighbours=True)
    im=cb.movie(img,fr=1)
    im.save(fl[:-4]+'correlation_image.tif')
    
    m=np.array(m)

    img=st.skew(m,0)
    im=cb.movie(img,fr=1)
    im.save(fl[:-4]+'skew.tif')
    
    img=st.kurtosis(m,0)
    im=cb.movie(img,fr=1)
    im.save(fl[:-4]+'kurtosis.tif')
    
    img=np.std(m,0)
    im=cb.movie(img,fr=1)
    im.save(fl[:-4]+'std.tif')
    
    img=np.median(m,0)
    im=cb.movie(img,fr=1)
    im.save(fl[:-4]+'median.tif')
    
    
    img=np.max(m,0)
    im=cb.movie(img,fr=1)
    im.save(fl[:-4]+'max.tif')
#%% create and save tail probability image
from scipy.io import loadmat
idxChunk=0

with open('file_list.txt') as f:
    for ln in f:
        nf=ln[:-19]+'.proto-roi.mat'
        print(nf)            
        c=loadmat(nf)        
        tailImg=c['prototypes']['metric'][0,idxChunk]['tailProb'][0][0]        
        cb.movie(tailImg,fr=1).save(ln[:-13]+'_compress_tail_img.tif')

    
#%% LOGIN TO MASTER NODE
# TYPE salloc -n n_nodes --exclusive
# source activate environment_name

#%%#%%
slurm_script='/mnt/xfs1/home/agiovann/SOFTWARE/Constrained_NMF/SLURM/slurmStart.sh'
cse.utilities.start_server(slurm_script=slurm_script)
#n_processes = 27#np.maximum(psutil.cpu_count() - 2,1) # roughly number of cores on your machine minus 1
pdir, profile = os.environ['IPPPDIR'], os.environ['IPPPROFILE']
client_ = Client(ipython_dir=pdir, profile=profile)
dview=client_[::2]
print 'Using '+ str(len(client_)) + ' processes'
#%% no server
if 0:
    dview=None
#%%
import os
fnames=[]
base_folder='/mnt/ceph/users/agiovann/ImagingData/LABELLING/Farzaneh/'
for file_ in glob.glob(base_folder+'151102_001_00*[0-9].tif'):
#        if not os.path.exists(file_[:-3]+'hdf5'):
            fnames.append(file_)
fnames.sort()
print fnames  
#%% motion correct
max_shift_w=45
max_shift_h=45
t1 = time()
file_res=cb.motion_correct_parallel(fnames,fr=30,template=None,margins_out=0,max_shift_w=max_shift_w, max_shift_h=max_shift_h,dview=dview,apply_smooth=True)
t2=time()-t1
print t2
#%%   
all_movs=[]
for f in  fnames:
    print f
    with np.load(f[:-3]+'npz') as fl:
#        pl.subplot(1,2,1)
#        pl.imshow(fl['template'],cmap=pl.cm.gray)
#        pl.subplot(1,2,2)
#        pl.plot(fl['shifts'])       
        all_movs.append(fl['template'][np.newaxis,:,:])
#        pl.pause(.1)
#        pl.cla()
#%%        
all_movs=cb.movie(np.concatenate(all_movs,axis=0),fr=10)
all_movs,shifts,corss,_=all_movs.motion_correct(template=all_movs[1],max_shift_w=max_shift_w, max_shift_h=max_shift_h)
#%%
template=np.median(all_movs[:],axis=0)
np.save(base_folder+'template_total',template)
#%%
pl.imshow(template,cmap=pl.cm.gray,vmax=120)
#%%
all_movs.play(backend='opencv',gain=1,fr=100)
#%%
t1 = time()
file_res=cb.motion_correct_parallel(fnames,30,template=template,margins_out=0,max_shift_w=max_shift_w, max_shift_h=max_shift_h,dview=dview,remove_blanks=False)
t2=time()-t1
print t2
#%%
fnames=[]
for file in glob.glob(base_folder+'15*[0-9].hdf5'):
        fnames.append(file)
fnames.sort()
print fnames  
#%%
file_res=cb.utils.pre_preprocess_movie_labeling(dview, fnames, median_filter_size=(2,1,1), 
                                  resize_factors=[.2,.1666666666*3],diameter_bilateral_blur=4)

#%%

client_.close()
cse.utilities.stop_server(is_slurm=True,pdir=pdir,profile=profile)

#%%

#%%
fold=os.path.split(os.path.split(fnames[0])[-2])[-1]
os.mkdir(fold)
#%%
files=glob.glob(fnames[0][:-19]+'*BL_compress_.tif')
files.sort()
print files
#%%
m=cb.load_movie_chain(files,fr=3)
m.play(backend='opencv',gain=10,fr=10)
#%%
m.save(files[0][:-20]+'_All_BL.tif')
#%%
files=glob.glob(fnames[0][:-19]+'*[0-9]._compress_.tif')
files.sort()
print files
#%%
m=cb.load_movie_chain(files,fr=3)
m.play(backend='opencv',gain=3,fr=40)
#%%
m.save(files[0][:-20]+'_All.tif')
