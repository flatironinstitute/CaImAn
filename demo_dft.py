# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 15:53:15 2016

@author: agiovann
"""
#%%
try:
    if __IPYTHON__:
        # this is used for debugging purposes only. allows to reload classes when changed
        get_ipython().magic(u'load_ext autoreload')
        get_ipython().magic(u'autoreload 2')
except NameError:
    print('Not IPYTHON')
    pass

import caiman as cm
import numpy as np
from caiman.dft_registration import register_translation
from skimage.util import view_as_windows
import time
import cv2
from caiman.motion_correction import apply_shift_iteration
import numbers
from numpy.lib.stride_tricks import as_strided
from warnings import warn
##%% SLOWER METHOD BUT COVER FULL FOVs
#def tile_and_correct(img,template, shapes,overlaps,upsample_factor_fft=10,upsample_factor_grid=1,max_shifts = (10,10),show_movie=False):
##    templates = view_as_windows(template.astype(np.float32),shapes,strides)
#    templates, _ = cm.cluster.get_patches_from_image(template.astype(np.float32),shapes=shapes,overlaps = overlaps)    
#    num_tiles = np.prod(templates.shape[:2])    
#    templates = list(np.reshape(templates,(num_tiles)))    
#    imgs, _ = cm.cluster.get_patches_from_image(img.astype(np.float32),shapes=shapes,overlaps = overlaps)    
#    dim_grid = np.shape(imgs)[:2]
#    imgs = list(np.reshape(imgs,(num_tiles)))
#    shfts = [register_translation(a,b,c,max_shifts = max_shifts) for a, b, c in zip (imgs,templates,[upsample_factor_fft]*num_tiles)]    
#    shift_img_x = np.reshape(np.array(shfts)[:,0],dim_grid)  
#    shift_img_y = np.reshape(np.array(shfts)[:,1],dim_grid)
#    
#    newshapes = tuple(np.ceil(np.multiply(shapes,1./upsample_factor_grid)).astype(np.int))    
#   
#    imgs, coords_2d = cm.cluster.get_patches_from_image(img.astype(np.float32),shapes=newshapes,overlaps = overlaps)           
#
#    dim_new_grid = tuple(imgs.shape[:2])
#    shift_img_x = cv2.resize(shift_img_x,dim_new_grid[::-1],interpolation = cv2.INTER_CUBIC)
#    shift_img_y = cv2.resize(shift_img_y,dim_new_grid[::-1],interpolation = cv2.INTER_CUBIC)
#    num_tiles = np.prod(imgs.shape[:2])    
#    total_shifts = [(-x,-y) for x,y in zip(shift_img_x.reshape(num_tiles),shift_img_y.reshape(num_tiles))]     
#    imgs = list(np.reshape(imgs,(num_tiles)))  
#
#    imgs = [apply_shift_iteration(im,sh,border_nan=True) for im,sh in zip(imgs, total_shifts)]
#
#
#    imgs = np.reshape(imgs,(dim_new_grid[0],dim_new_grid[1]))
#    
#    normalizer = np.zeros_like(img)
#    new_img = np.zeros_like(img)*np.nan
#    for idx_0,step_0 in enumerate(coords_2d):
#         for idx_1,step_1 in enumerate(step_0):          
#             prev_val = normalizer[step_1[0],step_1[1]]
#             normalizer[step_1[0],step_1[1]] = np.nansum(np.dstack([~np.isnan(imgs[idx_0,idx_1]),prev_val]),-1)
#             prev_val = new_img[step_1[0],step_1[1]]
#             new_img[step_1[0],step_1[1]] = np.nansum(np.dstack([imgs[idx_0,idx_1],prev_val]),-1)
#
#    new_img = new_img/normalizer
#    if show_movie:
#        img_show = np.hstack([new_img,img])
##        img_show = cv2.resize(img_show,None,fx=2,fy=2)
##        img_show = new_img
#        cv2.imshow('frame',img_show/5000)
#        cv2.waitKey(int(1./500*1000))      
#         
#    return shfts
#
#
#
#
##%%
#shapes = (32,32)
#overlaps = (0,0)
#strides = np.subtract(shapes,overlaps)
#
##%% slower covering full FOVs
#t1 = time.time()
#shfts_fft = [tile_and_correct(img,template, shapes, overlaps, max_shifts = (5,5), show_movie=False) for count,img in enumerate(np.array(m)[:300])]    
#print time.time()- t1
##%% THIS IS FASTER BUT NEED SOME TWEAKS TO COVER THE FULL FOV
#def tile_and_correct_fast(img,template, shapes,overlaps,upsample_factor_fft=10,upsample_factor_grid=1,max_shifts = (10,10),show_movie=True):
#    #%
#    strides = np.subtract(shapes,overlaps)
#    templates = view_as_windows(template.astype(np.float32),shapes,strides)        
#    num_tiles = np.prod(templates.shape[:2])    
#    templates = list(np.reshape(templates,(num_tiles ,shapes[0],shapes[1])))
#    
#    imgs= view_as_windows(img.astype(np.float32),shapes,strides) 
#
#    dim_grid = np.shape(imgs)[:2]
#    imgs = list(np.reshape(imgs,(num_tiles,shapes[0],shapes[1])))
#    shfts = [register_translation(a,b,c) for a, b, c in zip (imgs,templates,[upsample_factor_fft]*num_tiles)]    
#    shift_img_x = np.reshape(np.array(shfts)[:,0],dim_grid)  
#    shift_img_y = np.reshape(np.array(shfts)[:,1],dim_grid)
#    
#    newshapes = tuple(np.ceil(np.multiply(shapes,1./upsample_factor_grid)).astype(np.int))
#    newstrides = np.subtract(newshapes,overlaps)
#    imgs= view_as_windows(img.astype(np.float32),newshapes,newstrides)        
#
#    dim_new_grid = tuple(imgs.shape[:2])
#    shift_img_x = cv2.resize(shift_img_x,dim_new_grid[::-1],interpolation = cv2.INTER_CUBIC)
#    shift_img_y = cv2.resize(shift_img_y,dim_new_grid[::-1],interpolation = cv2.INTER_CUBIC)
#    num_tiles = np.prod(imgs.shape[:2])    
#    total_shifts = [(-x,-y) for x,y in zip(shift_img_x.reshape(num_tiles),shift_img_y.reshape(num_tiles))]     
#    imgs = list(np.reshape(imgs,(num_tiles,newshapes[0],newshapes[1])))    
#    imgs = [apply_shift_iteration(im,sh,border_nan=True) for im,sh in zip(imgs, total_shifts)]
#    imgs = np.reshape(np.array(imgs),(dim_new_grid[0],dim_new_grid[1],newshapes[0],newshapes[1]))
#    normalizer = np.zeros_like(img)
#    new_img = np.zeros_like(img)*np.nan
#
#    for idx_0,step_0 in enumerate(range(0,img.shape[0]-newshapes[0]+1,newstrides[0])):
#         for idx_1,step_1 in enumerate(range(0,img.shape[1]-newshapes[1]+1,newstrides[1])):
#
#             prev_val = normalizer[step_0:step_0+newshapes[0],step_1:step_1+newshapes[1]]
#             normalizer[step_0:step_0+newshapes[0],step_1:step_1+newshapes[1]] = np.nansum(np.dstack([~np.isnan(imgs[idx_0,idx_1]),prev_val]),-1)
#             prev_val = new_img[step_0:step_0+newshapes[0],step_1:step_1+newshapes[1]]
#             new_img[step_0:step_0+newshapes[0],step_1:step_1+newshapes[1]] = np.nansum(np.dstack([imgs[idx_0,idx_1],prev_val]),-1)
#
#    new_img = new_img/normalizer
#    if show_movie:
#        img_show = cv2.resize(np.hstack([new_img,img]),None,fx=2,fy=2)
#        cv2.imshow('frame',img_show/300)
#        cv2.waitKey(int(1./500*1000))
#               
#    return shfts
##%% faster needs some tweaking
#shapes = (32,32)
#overlaps = (0,0)
#strides = np.subtract(shapes,overlaps)
#
##%%
#t1 = time.time()
#shfts_fft = [tile_and_correct_fast(img, template, shapes,overlaps,max_shifts = (5,5),show_movie=False) for count,img in enumerate(np.array(m)[:])]    
#print time.time()- t1
#%% set parameters and create template
#m = cm.load('M_FLUO_t.tif')
#m = cm.load('M_FLUO_4.tif')

#m = cm.load('CA1_green.tif')
m = cm.load('/mnt/xfs1/home/agiovann/imaging/b23/4.24/20160424165602_00028_00001.tif')
t1  = time.time()
template = m.copy().motion_correct(18,18,template=None)[-1]
t2  = time.time() - t1
print t2

#%% RIGID MOTION CORRECTION TEST
t1  = time.time()
shifts_fft=[]
for count,img in enumerate(np.array(m)):
    if count%100 == 0:
            print count
            print time.time()- t1
    shts = register_translation(img.astype(np.float32),template.astype(np.float32),upsample_factor=10)
    shifts_fft.append(shts)
    
t2  = time.time() - t1
print t2


#%%
def sliding_window(image, windowSize, stepSize):
    # slide a window across the image
    range_1 = range(0, image.shape[0]-windowSize[0], stepSize[0]) + [image.shape[0]-windowSize[0]]
    range_2 = range(0, image.shape[1]-windowSize[1], stepSize[1]) + [image.shape[1]-windowSize[1]]
    for dim_1, x in enumerate(range_1):
		for dim_2,y in enumerate(range_2):
			# yield the current window
			yield (dim_1, dim_2 , x, y, image[ x:x + windowSize[0],y:y + windowSize[1]])
   
#%%
shapes = (32,32)
overlaps = (8,8)
strides = np.subtract(shapes,overlaps)
#%% 
def tile_and_correct_faster(img,template, shapes,overlaps,upsample_factor_fft=10,upsample_factor_grid=1,max_shifts = (10,10),show_movie=False,max_deviation_rigid=None):
#%    templates = view_as_windows(template.astype(np.float32),shapes,strides)
    rigid_shts = register_translation(img.astype(np.float32),template.astype(np.float32),upsample_factor=upsample_factor_fft,max_shifts=max_shifts)
    strides = np.subtract(shapes,overlaps)    
    iterator = sliding_window(template.astype(np.float32),windowSize=shapes,stepSize = strides)        
    templates = [it[-1] for it in iterator]
    iterator = sliding_window(template.astype(np.float32),windowSize=shapes,stepSize = strides)        
    xy_grid = [(it[0],it[1]) for it in iterator]    
    num_tiles = np.prod(np.add(xy_grid[-1],1))    
    iterator = sliding_window(img.astype(np.float32),windowSize=shapes,stepSize = strides)        
    imgs = [it[-1] for it in iterator]
    
    dim_grid = tuple(np.add(xy_grid[-1],1))
    
    if max_deviation_rigid is not None:
        lb_shifts = np.round(np.subtract(rigid_shts,max_deviation_rigid)).astype(int)
        ub_shifts = np.round(np.add(rigid_shts,max_deviation_rigid)).astype(int)
#        print rigid_shts, lb_shifts, ub_shifts 
    else:
        lb_shifts = None
        ub_shifts = None
        
    shfts = [register_translation(a,b,c, shifts_lb = lb_shifts, shifts_ub = ub_shifts, max_shifts = max_shifts) for a, b, c in zip (imgs,templates,[upsample_factor_fft]*num_tiles)]    
    
    shift_img_x = np.reshape(np.array(shfts)[:,0],dim_grid)  
    shift_img_y = np.reshape(np.array(shfts)[:,1],dim_grid)
    
    newshapes = tuple(np.ceil(np.multiply(shapes,1./upsample_factor_grid)).astype(np.int))    
    newstrides = np.subtract(newshapes,overlaps)
    
    iterator = sliding_window(img.astype(np.float32),windowSize=newshapes,stepSize = newstrides)        
    imgs = [it[-1] for it in iterator]
        
    iterator = sliding_window(img.astype(np.float32),windowSize=newshapes,stepSize = newstrides)        
    xy_grid = [(it[0],it[1]) for it in iterator]  
    iterator = sliding_window(img.astype(np.float32),windowSize=newshapes,stepSize = newstrides)        
    start_step = [(it[2],it[3]) for it in iterator]
    
    dim_new_grid = tuple(np.add(xy_grid[-1],1))
    shift_img_x = cv2.resize(shift_img_x,dim_new_grid[::-1],interpolation = cv2.INTER_CUBIC)
    shift_img_y = cv2.resize(shift_img_y,dim_new_grid[::-1],interpolation = cv2.INTER_CUBIC)
    num_tiles = np.prod(dim_new_grid)   
    
    total_shifts = [(-x,-y) for x,y in zip(shift_img_x.reshape(num_tiles),shift_img_y.reshape(num_tiles))]     

    imgs = [apply_shift_iteration(im,sh,border_nan=True) for im,sh in zip(imgs, total_shifts)]
    
    normalizer = np.zeros_like(img)
    new_img = np.zeros_like(img)*np.nan
    
    for (x,y),(idx_0,idx_1),im in zip(start_step,xy_grid,imgs):
#        print (x,y,idx_0,idx_1)
        prev_val = normalizer[x:x + newshapes[0],y:y + newshapes[1]]
        normalizer[x:x + newshapes[0],y:y + newshapes[1]] = np.nansum(np.dstack([~np.isnan(im),prev_val]),-1)
        prev_val = new_img[x:x + newshapes[0],y:y + newshapes[1]]
        new_img[x:x + newshapes[0],y:y + newshapes[1]] = np.nansum(np.dstack([im,prev_val]),-1)
    
    

    new_img = new_img/normalizer
    if show_movie:
        img = apply_shift_iteration(img,-rigid_shts,border_nan=True)
        img_show = np.vstack([new_img,img,new_img-img])
        img_show = cv2.resize(img_show,None,fx=5,fy=5)
#        img_show = new_img
        cv2.imshow('frame',img_show/200)
        cv2.waitKey(int(1./500*1000))      
              
    return shfts
#%%
t1 = time.time()
shfts_fft = [tile_and_correct_faster(img, template, shapes,overlaps,max_shifts = (20,20),show_movie=True,max_deviation_rigid=4) for count,img in enumerate(np.array(m)[:])]    
print time.time()- t1
#%%
#from cvxpy import *
#import numpy
#n = shift_img_x.size
## Problem data.
#A = np.eye(n)
#b = shift_img_x.flatten()
#t1 = time.time()
## Construct the problem.
#x = Variable(n)
#objective = Minimize(sum_entries(square(A*x - b)))
#Mx = 
#constraints = [0 <= x, x <= 1]
#prob = Problem(objective, constraints)
#print time.time()- t1
#print "Optimal value", prob.solve()
#print "Optimal var"
#print x.value # A numpy matrix.