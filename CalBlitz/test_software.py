# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 13:59:32 2015

@author: agiovann
"""

#%%
import h5py
import calblitz as cb
import time
import pylab as pl
import numpy as np


#%%
filename='movies/demoMovie_PC.tif'
frameRate=15.62;
start_time=0;

#%%
filename_py=filename[:-4]+'.npz'
filename_hdf5=filename[:-4]+'.hdf5'
filename_mc=filename[:-4]+'_mc.npz'

#%% load movie

print 'loading movie...'
m=cb.load(filename, fr=frameRate,start_time=start_time);

print 'motion correcting...'
max_shift_h=10;
max_shift_w=10;
#m=cb.load(filename_hdf5); 
m,shifts,xcorrs,template=m.motion_correct(max_shift_w=max_shift_w,max_shift_h=max_shift_h, num_frames_template=None, template = None,method='opencv')
max_h,max_w= np.max(shifts,axis=0)
min_h,min_w= np.min(shifts,axis=0)
m=m.crop(crop_top=max_h,crop_bottom=-min_h+1,crop_left=max_w,crop_right=-min_w,crop_begin=0,crop_end=0)

print 'saving motion corrected movie...'
m.save('demoTiff_mc.tif')

print 'computing delta f over sqrt(f)'
m=m-np.min(m)+1;
m,mbl=m.computeDFF(secsWindow=10,quantilMin=50)

print 'compute spatial components via ICA+PCA...'
initTime=time.time()
spcomps=m.IPCA_stICA(componentsPCA=70,componentsICA = 50, mu=1, batch=1000000, algorithm='parallel', whiten=True, ICAfun='logcosh', fun_args=None, max_iter=2000, tol=1e-8, w_init=None, random_state=None);
 
print 'Everything should be alright!'
