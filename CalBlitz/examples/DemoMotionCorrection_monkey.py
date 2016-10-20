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
#%%
try:
    
    from IPython import get_ipython
    ipython = get_ipython()
    print(__IPYTHON__)
    ipython.magic('load_ext autoreload')
    ipython.magic('autoreload 2')
#    ipython.magic('matplotlib')
except:
    print ('Not Ipython') 

import pylab as pl
import numpy as np

import calblitz as cb


# set basic ipython functionalities



# define movie
filename = '/mnt/ceph/users/epnevmatikakis/Ca_datasets/Seidemann/ax3_002_000.sbx'
filename_hdf5 = '/mnt/ceph/users/agiovann/ax3_002_000_ag'+'.hdf5'
filename_mc =  '/mnt/ceph/users/agiovann/ax3_002_000_ag'+'_mc.tif'
frameRate = 50
start_time = 0
num_frames=1500

#%%

# load movie
# for loading only a portion of the movie or only some channels
# you can use the option: subindices=range(0,1500,10)
m = cb.load(filename, fr=frameRate, start_time=start_time,num_frames_sub_idx=num_frames)[0]

# save movie hdf5 format. fastest
m.save(filename_hdf5)
#%%
m=cb.load(filename_hdf5)
#%%
# backend='opencv' is much faster
m.play(fr=100, gain=1.0, magnification=1, backend='opencv')
#%%
# automatic parameters motion correction
max_shift_h = 30  # maximum allowed shifts in y
max_shift_w = 30  # maximum allowed shifts in x
m = cb.load(filename_hdf5)
m_mc, shifts, xcorrs, template = m.motion_correct(max_shift_w=max_shift_w,
                                               max_shift_h=max_shift_h,
                                               num_frames_template=None,
                                               template=None,
                                               method='opencv',remove_blanks=True)
#%%                                               

m_mc.save(filename_mc)
#np.savez(filename_mc,
#         template=template,
#         shifts=shifts,
#         xcorrs=xcorrs,
#         max_shift_h=max_shift_h,
#         max_shift_w=max_shift_w)
#
#%%
pl.subplot(2, 1, 1)
pl.plot(shifts)
pl.ylabel('pixels')
pl.subplot(2, 1, 2)
pl.plot(xcorrs)
pl.ylabel('cross correlation')
pl.xlabel('Frames')
#%%
m_mc2, shifts2, xcorrs2, template2 = m_mc.motion_correct(max_shift_w=max_shift_w,
                                               max_shift_h=max_shift_h,
                                               num_frames_template=None,
                                               template=template,
                                               method='opencv',remove_blanks=True)
#%%
pl.subplot(2, 1, 1)
pl.plot(shifts2)
pl.ylabel('pixels')
pl.subplot(2, 1, 2)
pl.plot(xcorrs2)
pl.ylabel('cross correlation')
pl.xlabel('Frames')                                             
#%%
m_mc3, shifts3, xcorrs3, template3 = m_mc.motion_correct(max_shift_w=max_shift_w,
                                               max_shift_h=max_shift_h,
                                               num_frames_template=None,
                                               template=m_mc2[10],
                                               method='opencv',remove_blanks=True)
#%%
pl.subplot(2, 1, 1)
pl.plot(shifts3)
pl.ylabel('pixels')
pl.subplot(2, 1, 2)
pl.plot(xcorrs3)
pl.ylabel('cross correlation')
pl.xlabel('Frames')                                             

#%%
m.play(fr=50, gain=5.0, magnification=1, backend='opencv')

# IF YOU WANT MORE CONTROL USE THE FOLLOWING
# motion correct for template purpose.
# Just use a subset to compute the template
m = cb.load(filename_hdf5)
min_val_add = np.min(np.mean(m, axis=0))  # movies needs to be not too negative
m = m-min_val_add

every_x_frames = 1  # for memory/efficiency reason can be increased
submov = m[::every_x_frames, :]
max_shift_h = 10
max_shift_w = 10

templ = np.nanmedian(submov, axis=0)  # create template with portion of movie
shifts, xcorrs = submov.extract_shifts(max_shift_w=max_shift_w,
                                       max_shift_h=max_shift_h,
                                       template=templ,
                                       method='opencv')
submov.apply_shifts(shifts, interpolation='cubic')
template = (np.nanmedian(submov, axis=0))
shifts, xcorrs = m.extract_shifts(max_shift_w=max_shift_w,
                                  max_shift_h=max_shift_h,
                                  template=template,
                                  method='opencv')
m = m.apply_shifts(shifts, interpolation='cubic')
template = (np.median(m, axis=0))

pl.imshow(template, cmap=pl.cm.gray)

# now use the good template to correct
max_shift_h = 10
max_shift_w = 10

m = cb.load(filename_hdf5)
min_val_add = np.float32(np.percentile(m, .0001))
m = m-min_val_add
shifts, xcorrs = m.extract_shifts(max_shift_w=max_shift_w,
                                  max_shift_h=max_shift_h,
                                  template=template,
                                  method='opencv')

if m.meta_data[0] is None:
    m.meta_data[0] = dict()

m.meta_data[0]['shifts'] = shifts
m.meta_data[0]['xcorrs'] = xcorrs
m.meta_data[0]['template'] = template
m.meta_data[0]['min_val_add'] = min_val_add

m.save(filename_hdf5)
# visualize shifts
pl.subplot(2, 1, 1)
pl.plot(shifts)
pl.subplot(2, 1, 2)
pl.plot(xcorrs)
# reload and apply shifts
m = cb.load(filename_hdf5)
meta_data = m.meta_data[0]
shifts = meta_data['shifts']
m = m.apply_shifts(shifts, interpolation='cubic')

# crop borders created by motion correction
max_h, max_w = np.max(shifts, axis=0)
min_h, min_w = np.min(shifts, axis=0)
m = m.crop(crop_top=max_h,
           crop_bottom=-min_h+1,
           crop_left=max_w,
           crop_right=-min_w,
           crop_begin=0,
           crop_end=0)

# play movie
m.play(fr=50, gain=3.0, magnification=1, backend='opencv')
