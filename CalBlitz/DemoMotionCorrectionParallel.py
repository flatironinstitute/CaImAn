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
from ipyparallel import Client
import os
import numpy as np
#%%
backend='SLURM'
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
#%%
import os
fnames=[]
for file in os.listdir("./"):
    if file.startswith("2016") and file.endswith(".tif"):
        fnames.append(os.path.abspath(file))
fnames.sort()
print fnames  
idx_start=[i for i in xrange(len(fnames[0])) if fnames[0][i] != fnames[- 1][i]][0]
base_name=fnames[0][:idx_start]
#%%
#low_SNR=False
#if low_SNR:
#    N=1000     
#    mn1=m.copy().bilateral_blur_2D(diameter=5,sigmaColor=10000,sigmaSpace=0)     
#    
#    mn1,shifts,xcorrs, template=mn1.motion_correct()
#    mn2=mn1.apply_shifts(shifts)     
#    #mn1=cb.movie(np.transpose(np.array(Y_n),[2,0,1]),fr=30)
#    mn=cb.concatenate([mn1,mn2],axis=1)
#    mn.play(gain=5.,magnification=4,backend='opencv',fr=30)
#%%
t1 = time()
file_res=cb.motion_correct_parallel(fnames,fr=30,template=None,margins_out=0,max_shift_w=45, max_shift_h=45,dview=dview,apply_smooth=True)
t2=time()-t1
print t2
#%%   
all_movs=[]
for f in  fls:
    idx=f.find('.')
    with np.load(f[:idx+1]+'npz') as fl:
        print f
#        pl.subplot(1,2,1)
#        pl.imshow(fl['template'],cmap=pl.cm.gray)
#        pl.subplot(1,2,2)
             
        all_movs.append(fl['template'][np.newaxis,:,:])
#        pl.plot(fl['shifts'])  
#        pl.pause(.001)
#        pl.cla()
#%%
all_movs=cb.movie(np.concatenate(all_movs,axis=0),fr=30)        
all_movs,shifts,_,_=all_movs.motion_correct(template=np.median(all_movs,axis=0))
all_movs.play(backend='opencv',gain=1.,fr=10)
#%%
all_movs=np.array(all_movs)
#%%
num_movies_per_chunk=50      
if num_movies_per_chunk < len(fnames):  
    chunks=range(0,len(fnames),num_movies_per_chunk)
    chunks[-1]=len(fnames)
else:
    chunks=[0, len(fnames)]
print chunks
movie_names=[]

for idx in range(len(chunks)-1):
        print chunks[idx], chunks[idx+1]       
        movie_names.append(fnames[chunks[idx]:chunks[idx+1]])

#%%
template_each=[];
all_movs_each=[];
movie_names=[]
for idx in range(len(chunks)-1):
        print chunks[idx], chunks[idx+1]
        all_mov=all_movs[chunks[idx]:chunks[idx+1]]
        all_mov=cb.movie(all_mov,fr=30)
        all_mov,shifts,_,_=all_mov.motion_correct(template=np.median(all_mov,axis=0))
        template=np.median(all_mov,axis=0)
        all_movs_each.append(all_mov)
        template_each.append(template)
        movie_names.append(fnames[chunks[idx]:chunks[idx+1]])
        pl.imshow(template,cmap=pl.cm.gray,vmax=100)
        
np.savez(base_name+'-template_total.npz',template_each=template_each, all_movs_each=np.array(all_movs_each),movie_names=movie_names)        
#%%
for idx,mov in enumerate(all_movs_each):
    mov.play(backend='opencv',gain=2.,fr=10)
#    mov.save(str(idx)+'sam_example.tif')
#%%

#%%
file_res=[]
for template,fn in zip(template_each,movie_names):
    print fn
    file_res.append(cb.motion_correct_parallel(fn,30,dview= dview,template=template,margins_out=0,max_shift_w=35, max_shift_h=35,remove_blanks=False))
#%%
for f1 in  file_res:
    for f in f1:
        with np.load(f+'npz') as fl:
            
            pl.subplot(1,2,1)
            pl.cla()
            pl.imshow(fl['template'],cmap=pl.cm.gray)
            pl.subplot(1,2,2)
            pl.plot(fl['shifts'])       
            pl.pause(0.001)
            pl.cla()
        
#%%
import shutil
names_new_each=[]
for mov_names_each in movie_names:   
    movie_names_hdf5=[]
    for mov_name in mov_names_each:
        movie_names_hdf5.append(mov_name[:-3]+'hdf5')
        #idx_x=slice(12,500,None)
        #idx_y=slice(12,500,None)
        #idx_xy=(idx_x,idx_y)
    idx_xy=None
8    name_new.sort()    
    names_new_each.append(name_new)
    print name_new
#%%
fnames_new_each=dview.map_sync(cse.utilities.save_memmap_join,names_new_each)
#%%
#for name_new in names_new_each:
#    fnames_new_each.append(cse.utilities.save_memmap_join(name_new, n_chunks=2, dview=None))
    

#%%
m=cb.load(fnames_new_each[-1],fr=5)
m.play(backend='opencv',gain=2.,fr=30)
#%%
