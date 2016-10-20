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

# set basic ipython functionalities
from IPython import get_ipython
ipython = get_ipython()

if '__IPYTHON__' in globals():
    ipython.magic('load_ext autoreload')
    ipython.magic('autoreload 2')
#%%
# define movie
filename = '09_09_2016_Running2_RGB.tif'
filename_hdf5 = filename[:-4]+'.hdf5'
filename_mc = filename[:-4]+'_mc.npz'
frameRate = 30
start_time = 0

# load movie
# for loading only a portion of the movie or only some channels
# you can use the option: subindices=range(0,1500,10)
m = cb.load(filename, fr=frameRate, start_time=start_time)

# red and green channels
m_r=m[:,:,:,0]
m=m[:,:,:,1]

# backend='opencv' is much faster
cb.concatenate([m,m_r],axis=1).resize(.5,.5,.1).play(fr=100, gain=1.0, magnification=1,backend='opencv')


# automatic parameters motion correction
max_shift_h = 20  # maximum allowed shifts in y
max_shift_w = 20  # maximum allowed shifts in x
m_r_mc, shifts, xcorrs, template = m_r.motion_correct(max_shift_w=max_shift_w,
                                               max_shift_h=max_shift_h,
                                               num_frames_template=None,
                                               template=None, remove_blanks=False,
                                               method='opencv')
#%%
pl.figure()
pl.imshow(template,cmap='gray')
pl.figure()
pl.plot(shifts)

#%% apply the shifts to the green channel
m_mc=m.apply_shifts(shifts, interpolation='linear', method='opencv', remove_blanks=True)
#%%
m_mc.resize(.5,.5,.1).play(fr=100, gain=1.0, magnification=1)
#%% SECOND APPROACH
filename = 'Context_CTRL_GC6s.tif'
filename_hdf5 = filename[:-4]+'.hdf5'
filename_mc = filename[:-4]+'_mc.npz'
frameRate = 30
start_time = 0

# load movie
# for loading only a portion of the movie or only some channels
# you can use the option: subindices=range(0,1500,10)
m = cb.load(filename, fr=frameRate, start_time=start_time)

# red and green channels

# backend='opencv' is much faster
m.resize(.5,.5,.1).play(fr=100, gain=1.0, magnification=1,backend='opencv')

#template=np.array(cb.load('Context_CTRL_Ai9_1000nm.tif',fr=1))
template=None # it works better correcting wiht the green channel!

# automatic parameters motion correction
max_shift_h = 40  # maximum allowed shifts in y
max_shift_w = 40  # maximum allowed shifts in x
m_mc, shifts, xcorrs, template = m.motion_correct(max_shift_w=max_shift_w,
                                               max_shift_h=max_shift_h,
                                               num_frames_template=None,
                                               template=template, remove_blanks=False,
                                               method='opencv')

#%%
m_mc.resize(.5,.5,.1).play(fr=100, gain=1.0, magnification=3,backend='opencv')
