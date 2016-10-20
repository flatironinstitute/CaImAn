# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 20:56:07 2015
@author: agiovann
Updated on Tue Apr 26 12:13:18 2016
@author: agiovann
Updated on Wed Aug 17 13:51:41 2016
@author: deep-introspection
"""

# init
import calblitz as cb
import pylab as pl
import numpy as np
import glob
from time import time
# set basic ipython functionalities
from IPython import get_ipython
ipython = get_ipython()

if '__IPYTHON__' in globals():
    ipython.magic('load_ext autoreload')
    ipython.magic('autoreload 2')
#%%
# define movie
def separate_channels(filename):    
    filename_red = filename[:-4]+'_red.hdf5'
    filename_green = filename[:-4]+'_green.hdf5'
    frameRate=30
    
    # load movie
    # for loading only a portion of the movie or only some channels
    # you can use the option: subindices=range(0,1500,10)
    m = cb.load(filename, fr=frameRate, subindices=slice(1,None,2))
    print m.shape
    m.save(filename_red)
    m = cb.load(filename, fr=frameRate, subindices=slice(0,None,2))
    m.save(filename_green)
    return((filename_green,filename_red))
#%%
file_names=glob.glob('*.tif')    
file_names.sort()
print file_names
#%%
res=map(separate_channels,file_names)
#%%
greens=[a for a,b in res]
#%%
file_res=cb.motion_correct_parallel(greens,fr=30,template=None,margins_out=0,max_shift_w=45, max_shift_h=45,dview=None,apply_smooth=True,save_hdf5=True)

#%% run a second time since it gives better results

file_res=cb.motion_correct_parallel(greens,fr=30,template=None,margins_out=0,max_shift_w=25, max_shift_h=25,dview=None,apply_smooth=True,save_hdf5=True)

#%%
m=cb.load(file_res[2]+'hdf5',fr=30)
m.resize(1,1,.2).play(backend='opencv',gain=3.,fr=100)
#%%
for f in file_res:
    m=cb.load(f+'hdf5',fr=30)
    m.save(f[:-1]+'_mc.tif')