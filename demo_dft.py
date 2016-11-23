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
from caiman.motion_correction import apply_shift_iteration
#%%

m = cm.load('M_FLUO_t.tif')
#%%
m = cm.load('J115_2015-12-09_L01_094.tif')
template=np.mean(m,0)

#%%
shifts, xcorrs, template = cm.motion_correction.motion_correct_online(m,0,18,18)
#%%
t1  = time.time()
shifts, xcorrs, template = m.motion_correct(18,18,template=template)
t2  = time.time() - t1
print t2

#%%

t1  = time.time()
shifts_fft=[]
for count,img in enumerate(np.array(m)):
    if count%100 == 0:
            print count
            print time.time()- t1
    shts = register_translation(img.astype(np.float32),template.astype(np.float32),upsample_factor=10)
    shifts_fft.append(shts[0])
    
t2  = time.time() - t1
print t2
#%%
shapes = (64,64)
stride = 8
#strides = np.subtract((32,32),stride)
strides = np.subtract(shapes,stride)
#imgs= view_as_windows(img.astype(np.float32),shapes,strides)    
#num_tiles = np.prod(imgs.shape[:2])
#imgs = list(np.reshape(imgs,(num_tiles,shapes[0],shapes[1])))

templates = view_as_windows(template.astype(np.float32),shapes,strides)    
num_tiles = np.prod(templates.shape[:2])
templates = list(np.reshape(templates,(num_tiles,shapes[0],shapes[1])))

#shfts = map(register_translation,imgs,templates,[10]*num_tiles)
#%%
def tile_and_correct(img,template, shapes,strides,upsample_factor_fft=10,upsample_factor_grid=2):
    #%%
    templates = view_as_windows(template.astype(np.float32),shapes,strides)        
    num_tiles = np.prod(templates.shape[:2])    
    templates = list(np.reshape(templates,(num_tiles ,shapes[0],shapes[1])))

    imgs= view_as_windows(img.astype(np.float32),shapes,strides)    
    dim_grid = np.shape(imgs)[:2]
    imgs = list(np.reshape(imgs,(num_tiles,shapes[0],shapes[1])))
    shfts = [register_translation(a,b,c) for a, b, c in zip (imgs,templates,[upsample_factor_fft]*num_tiles)]    
    shift_img_x = np.reshape(np.array(shfts)[:,0],dim_grid)  
    shift_img_y = np.reshape(np.array(shfts)[:,1],dim_grid)
    
    newshapes = tuple(np.ceil(np.multiply(shapes,1./upsample_factor_grid)).astype(np.int))
    newstrides = np.subtract(newshapes,stride)
    imgs= view_as_windows(img.astype(np.float32),newshapes,newstrides)        

    dim_new_grid = tuple(imgs.shape[:2])
    shift_img_x = cv2.resize(shift_img_x,dim_new_grid[::-1],interpolation = cv2.INTER_CUBIC)
    shift_img_y = cv2.resize(shift_img_y,dim_new_grid[::-1],interpolation = cv2.INTER_CUBIC)
    num_tiles = np.prod(imgs.shape[:2])    
    total_shifts = [(x,y) for x,y in zip(shift_img_x.reshape(num_tiles),shift_img_y.reshape(num_tiles))]     
    imgs = list(np.reshape(imgs,(num_tiles,newshapes[0],newshapes[1])))    
    imgs = [apply_shift_iteration(im,sh) for im,sh in zip(imgs, total_shifts)]
    
#%%    
    return shfts
#%%
t1 = time.time()
shfts_fft = [tile_and_correct(img,template, shapes,strides) for count,img in enumerate(np.array(m)[:50])]    
print time.time()- t1