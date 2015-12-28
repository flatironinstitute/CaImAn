# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 17:44:17 2015

@author: agiovann
"""

from initialization import get_noise_fft,estimate_time_constant, get_noise_fft_parallel
import numpy as np
from scipy.interpolate import griddata    

#%%
def interpolate_missing_data(Y):
    """
    Interpolate any missing data using nearest neighbor interpolation
    """  
    coor=[];
    if np.any(np.isnan(Y)):
        raise Exception('The algorithm has not been tested with missing values (NaNs)')        
        for idx,row in enumerate(Y):
            nans=np.where(np.isnan(row))[0] 
            n_nans=np.where(~np.isnan(row))[0] 
            coor.append((idx,nans))
            Y[idx,nans]=np.interp(nans, n_nans, row[n_nans]) 
            
     
#    mis_data = np.isnan(Y)
#    coor = mis_data.nonzero()
#    ok_data = ~mis_data
#    coor_ok = ok_data.nonzero()
#    Yvals=[np.where(np.isnan(Y)) for row in Y]
#    
#    Yvals = griddata(coor_ok,Y[coor_ok],coor,method='nearest')
#    un_t = np.unique(coor[-1])
#    coorx = []
#    coory = []
#    Yvals = []
#    for i, unv in enumerate(un_t):
#        tm = np.where(coor[-1]==unv)
#        coorx.append(coor[0][tm].tolist())
#        coory.append(coor[1][tm].tolist())
#        Yt = Y[:,:,unv]
#        ok = ~np.isnan(Yt)
#        coor_ok = ok.nonzero()
#        ytemp = griddata(coor_ok,Yt[coor_ok],(coor[0][tm],coor[1][tm]),method='nearest')
#        Yvals.append(ytemp)
        
    return Y, coor

#%%
def find_unsaturated_pixels(Y, saturationValue = None, saturationThreshold = 0.9, saturationTime = 0.005):
    
    if saturationValue == None:
        saturationValue = np.power(2,np.ceil(np.log2(np.max(Y))))-1
        
    Ysat = (Y >= saturationThreshold*saturationValue)
    pix = np.mean(Ysat,Y.ndim-1).flatten('F') > saturationTime
    normalPixels = np.where(pix)
    
    return normalPixels

#%%
def preprocess_data(Y,  sn = None , n_processes=4, n_pixels_per_process=100,  noise_range = [0.25,0.5], noise_method = 'logmexp', compute_g=False,  p = 2, g = None,  lags = 5, include_noise = False, pixels = None):
    
    Y,coor=interpolate_missing_data(Y)            
    
    if sn is None:
        sn=get_noise_fft_parallel(Y, n_processes=n_processes,n_pixels_per_process=n_pixels_per_process, backend='threading', noise_range = noise_range, noise_method = noise_method)    
        #sn = get_noise_fft(Y, noise_range = noise_range, noise_method = noise_method)        
    
    if compute_g:
        g = estimate_time_constant(Y, sn, p = p, lags = lags, include_noise = include_noise, pixels = pixels)
    else:
        g=None
    
    return Y,sn,g    