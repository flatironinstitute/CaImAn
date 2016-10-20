# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 19:47:04 2015

@author: agiovann
"""
#%%
path_to_CalBlitz_folder='/home/ubuntu/SOFTWARE/CalBlitz'
path_to_CalBlitz_folder='/Users/agiovann/Documents/SOFTWARE/CalBlitz/'

import sys
sys.path
sys.path.append(path_to_CalBlitz_folder)
#% add required packages
import calblitz as cb
import time
import pylab as pl
import numpy as np
import glob

#% set basic ipython functionalities
try: 
    pl.ion()
    %load_ext autoreload
    %autoreload 2
except:
    print "Probably not a Ipython interactive environment"

#%%
tif_files=glob.glob('*.tif')
num_frames_median=1000;
movs=[];
fr=30;
start_time=0;
templates=[];
for tif_file in tif_files:
    print(tif_file)
    m=cb.load(tif_file,fr=30,start_time=0,subindices=range(0,1500,20))
    min_val_add=np.percentile(m,.01)
    m=m-min_val_add
    movs.append(m)
    templ=np.nanmedian(m,axis=0);
    m,template,shifts,xcorrs=m.motion_correct(max_shift_w=5, max_shift_h=5, show_movie=False, template=templ, method='opencv')    
    templates.append(np.median(m,axis=0))

all_movs=cb.concatenate(movs)
m=cb.movie(np.array(templates),fr=1)
m=m.motion_correct(template=m[0])[0]
template=np.median(m,axis=0)
cb.matrixMontage(m,cmap=pl.cm.gray,vmin=0,vmax=1000)

#%%
all_shifts=[];
movs=[];
for tif_file in tif_files:
    print(tif_file)
    m=cb.load(tif_file,fr=30,start_time=0);   
    min_val_add=np.percentile(m,.01)
    m=m-min_val_add
    m,_,shifts,_=m.motion_correct(template=template, method='opencv')
    movs.append(m)    
    all_shifts.append(shifts)

all_movs=cb.concatenate(movs)
