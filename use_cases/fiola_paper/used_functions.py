#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 13:41:58 2021
This file includes other functions once used in FIOLA
@author: @caichangjia @agiovan
"""
import cv2
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats    
from scipy.signal import argrelextrema, butter, sosfilt, sosfilt_zi

import caiman as cm

class OnlineFilter(object):
    def __init__(self, freq, fr, order=3, mode='high'):
        '''
        Object encapsulating Online filtering for spike extraction traces
        Args:
            freq: float
            cutoff frequency
        
        order: int
            order of the filter
        
        mode: str
            'high' for high-pass filtering, 'low' for low-pass filtering
            
        '''
        self.freq = freq
        self.fr = fr
        self.mode = mode
        self.order=order
        self.normFreq = freq / (fr / 2)        
        self.filt = butter(self.order, self.normFreq, self.mode, output='sos')         
        self.z_init = sosfilt_zi(self.filt)
        
    
    def fit(self, sig):
        """
        Online filter initialization and running offline

        Parameters
        ----------
        sig : ndarray
            input signal to initialize
        num_frames_buf : int
            frames to use for buffering

        Returns
        -------
        sig_filt: ndarray
            filtered signal

        """
        sig_filt = signal_filter(sig, freq=self.freq, fr=self.fr, order=self.order, mode=self.mode)
        # result_init, z_init = signal.sosfilt(b, data[:,:20000], zi=z)
        self.z_init = np.repeat(self.z_init[:,None,:], sig.shape[0], axis=1)
    
        #sos_all = np.zeros(sig_filt.shape)

        for i in range(0,sig.shape[-1]-1):
            _ , self.z_init = sosfilt(self.filt, np.expand_dims(sig[:,i], axis=1), zi=self.z_init)
            
        return sig_filt 

    def fit_next(self, sig):
        
        sig_filt, self.z_init = sosfilt(self.filt, np.expand_dims(sig,axis=1), zi=self.z_init)
        return sig_filt.squeeze()

        
def rolling_window(ndarr, window_size, stride):   
        """
        generates efficient rolling window for running statistics
        Args:
            ndarr: ndarray
                input pixels in format pixels x time
            window_size: int
                size of the sliding window
            stride: int
                stride of the sliding window
        Returns:
                iterator with views of the input array
                
        """
        for i in range(0,ndarr.shape[-1]-window_size-stride+1,stride): 
            yield ndarr[:,i:np.minimum(i+window_size, ndarr.shape[-1])]
            
        if i+stride != ndarr.shape[-1]:
           yield ndarr[:,i+stride:]

def estimate_running_std(signal_in, win_size=20000, stride=5000, 
                         idx_exclude=None, q_min=25, q_max=75):
    """
    Function to estimate ROBUST runnning std
    
    Args:
        win_size: int
            window used to compute running std to normalize signals when 
            compensating for photobleaching
            
        stride: int
            corresponding stride to win_size
            
        idx_exclude: iterator
            indexes to exclude when computing std
        
        q_min: float
            lower percentile for estimation of signal variability (do not change)
        
        q_max: float
            higher percentile for estimation of signal variability (do not change)
        
        
    Returns:
        std_run: ndarray
            running standard deviation
    
    """
    if idx_exclude is not None:
        signal = signal_in[np.setdiff1d(range(len(signal_in)), idx_exclude)]        
    else:
        signal = signal_in
    iter_win = rolling_window(signal[None,:],win_size,stride)
    myperc = partial(np.percentile, q=[q_min,q_max], axis=-1)
    res = np.array(list(map(myperc,iter_win))).T.squeeze()
    iqr = (res[1]-res[0])/1.35
    std_run = cv2.resize(iqr,signal_in[None,:].shape).squeeze()
    return std_run

def compute_thresh(peak_height, prev_thresh=None, delta_max=0.03, number_maxima_before=1):
    kernel = stats.gaussian_kde(peak_height)
    x_val = np.linspace(0,np.max(peak_height),1000)
    pdf = kernel(x_val)
    second_der = np.diff(pdf,2)
    mean = np.mean(peak_height)
    min_idx = argrelextrema(kernel(x_val), np.less)

    minima = x_val[min_idx]
    minima = minima[minima>mean]
    minima_2nd = argrelextrema(second_der, np.greater)
    minima_2nd = x_val[minima_2nd]

    if prev_thresh is None:
        delta_max = np.inf
        prev_thresh = mean                   
    
    thresh = prev_thresh 

    if (len(minima)>0) and (np.abs(minima[0]-prev_thresh)< delta_max):
        thresh = minima[0]
        mnt = (minima_2nd-thresh)
        mnt = mnt[mnt<0]
        thresh += mnt[np.maximum(-len(mnt)+1,-number_maxima_before)]
    #else:
    #    thresh = 100
        
    thresh_7 = compute_std(peak_height) * 7.5
    
    """
    print(f'previous thresh: {prev_thresh}')
    print(f'current thresh: {thresh}')  
    """
    plt.figure()
    plt.plot(x_val, pdf,'c')    
    plt.plot(x_val[2:],second_der*500,'r')  
    plt.plot(thresh,0, '*')   
    plt.vlines(thresh_7, 0, 2, color='r')
    plt.pause(0.1)
    
    return thresh

def mode_robust_fast(inputData, axis=None):
    """
    Robust estimator of the mode of a data set using the half-sample mode.

    .. versionadded: 1.0.3
    """

    if axis is not None:

        def fnc(x):
            return mode_robust_fast(x)

        dataMode = np.apply_along_axis(fnc, axis, inputData)
    else:
        # Create the function that we can use for the half-sample mode
        data = inputData.ravel()
        # The data need to be sorted for this to work
        data = np.sort(data)
        # Find the mode
        dataMode = _hsm(data)

    return dataMode

def _hsm(data):
    if data.size == 1:
        return data[0]
    elif data.size == 2:
        return data.mean()
    elif data.size == 3:
        i1 = data[1] - data[0]
        i2 = data[2] - data[1]
        if i1 < i2:
            return data[:2].mean()
        elif i2 > i1:
            return data[1:].mean()
        else:
            return data[1]
    else:

        wMin = np.inf
        N = data.size//2 + data.size % 2

        for i in range(0, N):
            w = data[i + N - 1] - data[i]
            if w < wMin:
                wMin = w
                j = i

        return _hsm(data[j:j + N])
    
def combine_datasets(movies, masks, num_frames, x_shifts=[3,-3], y_shifts=[3,-3], weights=None, shape=(15,15)):
    """ Combine two datasets to create manually overlapping neurons
    Args: 
        movies: list
            list of movies
        masks: list
            list of masks
        num_frames: int
            number of frames selected
        x_shifts, y_shifts: list
            shifts in x and y direction relative to the original movie
        weights: list
            weights of each movie
        shape: tuple
            shape of the new combined movie
    
    Returns:
        new_mov: ndarray
            new combined movie
        new_masks: list
            masks for neurons
    """
    new_mov = 0
    new_masks = []
    if weights is None:
        weights = [1/len(movies)]*len(movies)
    for mov, mask, x_shift, y_shift, weight in zip(movies, masks, x_shifts,y_shifts, weights):
        new_mask = cm.movie(mask)
        if mov.shape[1] != shape[0]:
            mov = mov.resize(shape[0]/mov.shape[1],shape[1]/mov.shape[2],1)
            new_mask = new_mask.resize(shape[0]/mask.shape[1],shape[1]/mask.shape[2], 1)
            
        if num_frames > mov.shape[0]:
            num_diff = num_frames - mov.shape[0]
            mov = np.concatenate((mov, mov[(mov.shape[0] - num_diff) : mov.shape[0], :, :]), axis=0)
        new_mov += np.roll(mov[:num_frames]*weight, (x_shift, y_shift), axis=(1,2))
        new_mask = np.roll(new_mask, (x_shift, y_shift), axis=(1,2))
        new_masks.append(new_mask[0])   
        
    return new_mov, new_masks
    

