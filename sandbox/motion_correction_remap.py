#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 18:30:19 2017

@author: agiovann
"""
#%%
import numpy as np
import caiman as cm
import pylab as pl
import cv2
from time import time
#%%

mc = cm.load('/mnt/ceph/users/agiovann/ImagingData/DanGoodwin/Somagcamp-Fish4-z13-100-400crop256.tif')[:2000].motion_correct(3,3)[0] 
#%%
import cv2
import numpy as np
from matplotlib import pyplot as plt
spctr= []
for fr,img in  enumerate(cm.load('/mnt/ceph/users/agiovann/ImagingData/DanGoodwin/Somagcamp-Fish4-z13-100-400crop256.tif')):
    print(fr)
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20*np.log(np.abs(fshift))
    spctr.append(magnitude_spectrum)
#    plt.subplot(121),plt.imshow(img, cmap = 'gray')
#    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
#    plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
#    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
#    plt.show()
#    pl.pause(.1)
#%%
dsfactor = 1
m = mc.resize(1,1,dsfactor)
num_frames = m.shape[0]
#%%

inputImage = m[10]
mapX = np.zeros((num_frames,inputImage.shape[0], inputImage.shape[1]), dtype=np.float32)
mapY = np.zeros((num_frames,inputImage.shape[0], inputImage.shape[1]), dtype=np.float32)
templ = np.median(m,0)
map_orig_x = np.zeros((inputImage.shape[0], inputImage.shape[1]), dtype=np.float32)
map_orig_y = np.zeros((inputImage.shape[0], inputImage.shape[1]), dtype=np.float32)

for j in range(inputImage.shape[0]):        
    print(j)
    for i in range(inputImage.shape[1]):                  
        map_orig_x[j,i] = i 
        map_orig_y[j,i] = j 
            
for k in range(num_frames):
    print(k)
    pyr_scale = .5; levels = 3; winsize = 20;  iterations = 15; poly_n = 7; poly_sigma = 1.2/5; flags = 0;
    flow = cv2.calcOpticalFlowFarneback(templ,m[k],None,pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags)       
    mapX[k,:] =  map_orig_x + flow[:,:,0]
    mapY[k,:] =  map_orig_y+ flow[:,:,1]
#    for j in range(inputImage.shape[0]):        
#            for i in range(inputImage.shape[1]):                  
#                mapX[k,j,i] = i + flow[j,i,0]
#                mapY[k,j,i] = j +  flow[j,i,1]
#%%
num_frames = mc.shape[0]
mapX_res = cm.movie(mapX).resize(1,1, 1/dsfactor)
mapY_res = cm.movie(mapY).resize(1,1, 1/dsfactor)
#%%    
fact = np.max(m)  
bl= np.min(m)     
times = []
new_ms = np.zeros(mc[:num_frames].shape)   
for counter,mm in enumerate(mc[:num_frames]):
    print(counter)
    t1 = time()
    new_img = cv2.remap(mm, mapX_res[counter], mapY_res[counter], cv2.INTER_CUBIC, None, cv2.BORDER_CONSTANT)
    new_ms[counter] = new_img
#    cv2.imshow('frame',(new_img-bl)*1./fact*5.)
#    if cv2.waitKey(1) & 0xFF == ord('q'):
#        break
    times.append(time()-t1)        
#cv2.destroyAllWindows()            
#%%
cm.movie(np.array(mc[:num_frames]-new_ms[:num_frames])).play(gain = 50.,magnification = 1)
#%%
cm.concatenate([cm.movie(np.array(new_ms[:num_frames]),fr = mc.fr),mc[:num_frames]],axis=2).resize(1,1,.5).play(gain = 2,magnification= 3 ,fr = 30, offset =-100)
#%%     
pl.subplot(1,3,1)
pl.imshow(np.mean(m[:2000],0),cmap = 'gray', vmax = 200)
pl.subplot(1,3,2)
pl.imshow(np.mean(new_ms[:2000],0),cmap = 'gray',vmax = 200)
pl.subplot(1,3,3)
pl.imshow(np.mean(new_ms[:2000],0)- np.mean(m[:2000],0),cmap = 'gray')
#%%
cm.movie(np.array(m[:2000]-new_ms[:2000])).play()
#%%
from multiprocessing import Pool
pl = Pool(processes=5)
def my_fun(X):
    import numpy as np
    return np.sum(range(X))

res = pl.map(my_fun,range(100000)*5)

