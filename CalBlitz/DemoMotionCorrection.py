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

# define movie
filename = 'movies/demoMovie_PC.tif'
filename_hdf5 = filename[:-4]+'.hdf5'
filename_mc = filename[:-4]+'_mc.npz'
frameRate = 15.62
start_time = 0

# load movie
# for loading only a portion of the movie or only some channels
# you can use the option: subindices=range(0,1500,10)
m = cb.load(filename, fr=frameRate, start_time=start_time)

# save movie hdf5 format. fastest
m.save(filename_hdf5)

# backend='opencv' is much faster
m.play(fr=100, gain=15.0, magnification=1, backend='opencv')

low_SNR = True
if low_SNR:
    N = 1000000
    mn1 = m.copy().bilateral_blur_2D(diameter=5,
                                     sigmaColor=10000,
                                     sigmaSpace=0)
    mn1, shifts, xcorrs, template = mn1.motion_correct()
    mn2 = mn1.apply_shifts(shifts)
    # mn1=cb.movie(np.transpose(np.array(Y_n),[2,0,1]),fr=30)
    mn = cb.concatenate([mn1, mn2], axis=1)

# automatic parameters motion correction
max_shift_h = 10  # maximum allowed shifts in y
max_shift_w = 10  # maximum allowed shifts in x
m = cb.load(filename_hdf5)
m, shifts, xcorrs, template = m.motion_correct(max_shift_w=max_shift_w,
                                               max_shift_h=max_shift_h,
                                               num_frames_template=None,
                                               template=None,
                                               method='opencv')

# removie the borders
max_h, max_w = np.max(shifts, axis=0)
min_h, min_w = np.min(shifts, axis=0)
m = m.crop(crop_top=max_h,
           crop_bottom=-min_h+1,
           crop_left=max_w,
           crop_right=-min_w,
           crop_begin=0,
           crop_end=0)
np.savez(filename_mc,
         template=template,
         shifts=shifts,
         xcorrs=xcorrs,
         max_shift_h=max_shift_h,
         max_shift_w=max_shift_w)

pl.subplot(2, 1, 1)
pl.plot(shifts)
pl.ylabel('pixels')
pl.subplot(2, 1, 2)
pl.plot(xcorrs)
pl.ylabel('cross correlation')
pl.xlabel('Frames')

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
