#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 12:12:44 2020
Fiola uses SignalAnalysisOnlineZ object for online spike extraction which
is based on template matching method 
@author: @caichangjia @andrea.giovannucci
"""
import logging
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import median_filter
from scipy import signal
from time import time

from caiman.source_extraction.volpy.spikepursuit import adaptive_thresh
from caiman.source_extraction.fiola.utilities import compute_std, non_symm_median_filter 

class SignalAnalysisOnlineZ(object):
    def __init__(self, mode='voltage', window = 10000, step = 5000, detrend=True, flip=True, 
                 do_scale=False, template_window=2, robust_std=False, adaptive_threshold=True, fr=400, freq=15, 
                 minimal_thresh=3.0, online_filter_method = 'median_filter', filt_window = 15, do_plot=False):
        '''
        Object encapsulating Online Spike extraction from input traces
        Args:
            mode: str
                voltage or calcium fluorescence indicator. Note that calcium deconvolution is not implemented here.
            window: int
                window over which to compute the running statistics
            step: int
                stride over which to compute the running statistics
            detrend: bool
                whether to remove the trend due to photobleaching 
            flip: bool
                whether to flip the signal after removing trend.
                True for voltron indicator. False for others
            do_scale: bool
                whether to scale the input trace or not
            template_window: int
                half window size for template matching. Will not perform template matching
                if 0.                
            robust_std: bool
                whether to use robust way to estimate noise
            adaptive_threshold: bool
                whether to use adaptive threshold method for deciding threshold level.
                Currently it should always be True.
            fr: float
                movie frame rate
            freq: float
                frequency for removing subthreshold activity
            minimal_thresh: float
                minimal threshold allowed for adaptive threshold. Threshold lower than minimal_thresh
                will be adjusted to minimal_thresh.
            online_filter_method: str
                Use 'median_filter' or 'median_filter_no_lag' for doing online filter. 'median filter'
                is more accurate but also introduce more lags
            filt_window: int or list
                window size for the median filter. The median filter is not symmetric when the two values 
                in the list is not the same(for example [8, 4] means using 8 past frames and 4 future frames for 
                                            removing the median of the current frame)                 
            do_plot: bool
                Whether to plot or not.
                
        '''
        for name, value in locals().items():
            if name != 'self':
                setattr(self, name, value)
                logging.debug(f'{name}, {value}')
        self.t_detect = []
        
    def fit(self, trace_in, num_frames):
        """
        Method for computing spikes in offline mode, and initializer for the online mode
        Args:
            trace_in: ndarray
                the vector (num neurons x time steps) used to initialize the online process
            num_frames: int
                total number of frames for processing including both initialization and online processing
        """
        nn, tm = trace_in.shape # number of neurons and time steps for initialization
        self.nn = nn
        self.frames_init = tm
        
        if self.mode == 'voltage':
            # contains all the extracted fluorescence traces
            self.trace, self.t_d, self.t0, self.t, self.t_s, self.t_sub = (np.zeros((nn, num_frames), dtype=np.float32) for _ in range(6))
            
            # contains running statistics
            self.median, self.scale, self.thresh, self.median2, self.std = (np.zeros((nn,1), dtype=np.float32) for _ in range(5))
    
            # contains spike time
            self.index = np.zeros((nn, num_frames), dtype=np.int32)
            self.index_track = np.zeros((nn), dtype=np.int32)        
            self.peak_to_std = np.zeros((nn, num_frames), dtype=np.float32)
            self.SNR, self.thresh_factor, self.peak_level = (np.zeros((nn, 1), dtype=np.float32) for _ in range(3))
            if self.template_window > 0:
                self.PTA = np.zeros((nn, 2*self.template_window+1), dtype=np.float32)
            else:
                self.PTA = np.zeros((nn, 5), dtype=np.float32)
            
            t_start = time()
            # initialize for each neuron: @todo parallelize
            if self.flip:
                self.trace[:, :tm] =  - trace_in.copy()
            else:
                self.trace[:, :tm] = trace_in.copy()
           
            if self.detrend:
                for tp in range(trace_in.shape[1]):
                    if tp > 0:
                        self.t_d[:, tp] = self.trace[:, tp] - self.trace[:, tp - 1] + 0.995 * self.t_d[:, tp - 1]
            else:
                self.t_d = self.trace.copy()
            
            for idx, tr in enumerate(self.t_d[:, :tm]):  
                output_list = find_spikes_tm(tr, self.freq, self.fr, self.do_scale, self.filt_window, self.template_window,
                                             self.robust_std, self.adaptive_threshold, self.minimal_thresh, self.do_plot)
                self.index_track[idx] = output_list[0].shape[0]
                self.index[idx, :self.index_track[idx]], self.thresh[idx], self.PTA[idx], self.t0[idx, :tm], \
                    self.t[idx, :tm], self.t_s[idx, :tm], self.t_sub[idx, :tm], self.median[idx], self.scale[idx], \
                        self.thresh_factor[idx], self.median2[idx], self.std[idx], \
                            self.peak_to_std[idx, :self.index_track[idx]], self.peak_level[idx] = output_list
                                    
            #self.filt.fit(self.t0[:, :tm])
            self.t_detect = self.t_detect + [(time() - t_start) / tm] * trace_in.shape[1]         
        elif self.mode == 'calcium':
            self.trace = np.zeros((nn, num_frames), dtype=np.float32)
            self.trace[:, :tm] = trace_in.copy()      
            # todo: add calcium deconvolution
            
        return self

    def fit_next(self, trace_in, n):
        '''
        Method to incrementally estimate the spikes at each time step
        Args:
            trace_in: ndarray
                vector containing the fluorescence value at a single time point
            n: time step    
        '''        
        self.n = n
        t_start = time()                                  
        
        # Update running statistics
        res = n % self.step        
        if ((n > 3.0 * self.window) and (res < 3 * self.nn)):
            idx = int(res / 3)
            temp = res - 3 * idx
            if temp == 0:
                self.update_median(idx)
            elif temp == 1:
                if self.do_scale:
                    self.update_scale(idx)
            elif temp == 2:
                self.update_thresh(idx)

        # Detrend, normalize 
        if self.flip:
            self.trace[:, n:(n + 1)] = - trace_in.copy()
        else:
            self.trace[:, n:(n + 1)] = trace_in.copy()
        if self.detrend:
            self.t_d[:, n:(n + 1)] = self.trace[:, n:(n + 1)] - self.trace[:, (n - 1):n] + 0.995 * self.t_d[:, (n - 1):n]
        else:
            self.t_d[:, n:(n + 1)] = self.trace[:, n:(n + 1)]
        temp = self.t_d[:, n:(n+1)].copy()        
        temp -= self.median[:, -1:]
        if self.do_scale == True:
            temp /= self.scale[:, -1:]
        self.t0[:, n:(n + 1)] = temp.copy()        

        # remove subthreshold
        if self.online_filter_method == 'median_filter':
            if type(self.filt_window) is int:
                sub_index = self.n - int((self.filt_window - 1) / 2)        
                lag = int((self.filt_window - 1) / 2)  
                if self.n >= self.frames_init + lag:
                    temp = self.t0[:, self.n - self.filt_window + 1: self.n + 1]
                    self.t_sub[:, sub_index] = np.median(temp, 1)
                    self.t[:, sub_index:sub_index + 1] = self.t0[:, sub_index:sub_index + 1] - self.t_sub[:, sub_index:sub_index + 1]
            elif type(self.filt_window) is list:
                sub_index = self.n - self.filt_window[1]
                lag = self.filt_window[1]  
                if self.n >= self.frames_init + lag:
                    temp = self.t0[:, self.n - sum(self.filt_window) : self.n + 1]
                    self.t_sub[:, sub_index] = np.median(temp, 1)
                    self.t[:, sub_index:sub_index + 1] = self.t0[:, sub_index:sub_index + 1] - self.t_sub[:, sub_index:sub_index + 1]
                
        elif self.online_filter_method == 'median_filter_no_lag':
            sub_index = self.n        
            lag = 0
            if self.n >= self.frames_init:
                temp = np.zeros((self.nn, self.filt_window))
                temp[:, :int((self.filt_window - 1) / 2) + 1] = self.t0[:, self.n - int((self.filt_window - 1) / 2): self.n + 1]
                temp[:, int((self.filt_window - 1) / 2) + 1:] = np.flip(self.t0[:, self.n - int((self.filt_window - 1) / 2): self.n])
                self.t_sub[:, sub_index] = np.median(temp, 1)
                self.t[:, sub_index:sub_index + 1] = self.t0[:, sub_index:sub_index + 1] - self.t_sub[:, sub_index:sub_index + 1]
           
        if self.template_window > 0:
            if self.n >= self.frames_init + lag + self.template_window:                                      
                temp = self.t[:, sub_index - 2*self.template_window : sub_index + 1]                         
                self.t_s[:, sub_index - self.template_window] = (temp * self.PTA).sum(axis=1)
                self.t_s[:, sub_index - self.template_window] = self.t_s[:, sub_index - self.template_window] - self.median2[:, -1]
        else:
            if self.n >= self.frames_init + lag + self.template_window:                                      
                self.t_s[:, sub_index] = self.t[:, sub_index]
                
        # Find spikes above threshold
        if self.n >= self.frames_init + lag + self.template_window + 1:
            temp = self.t_s[:, sub_index - self.template_window - 2: sub_index -self.template_window + 1].copy()
            idx_list = np.where((temp[:, 1] > temp[:, 0]) * (temp[:, 1] > temp[:, 2]) * (temp[:, 1] > self.thresh[:, -1]))[0]
            if idx_list.size > 0:
                self.index[idx_list, self.index_track[idx_list]] = sub_index - self.template_window - 1
                self.peak_to_std[idx_list, self.index_track[idx_list]] = self.t_s[idx_list, sub_index - self.template_window - 1] /self.std[idx_list, -1]
                self.index_track[idx_list] +=1
         
        self.t_detect.append(time() - t_start)
        return self
        
    def update_median(self, idx):
        tt = self.t_d[idx, self.n - np.int(self.window*2.5):self.n]
        if idx == 0:
            self.median = np.append(self.median, self.median[:, -1:], axis=1)
        self.median[idx, -1] = np.median(tt)
        return self
    
    def update_scale(self, idx):
        tt = self.t_d[idx, self.n - np.int(self.window*2.5):self.n]
        if idx == 0:
            self.scale = np.append(self.scale, self.scale[:, -1:], axis=1)
        self.scale[idx, -1] = - np.percentile(tt, 1)
        return self

    def update_thresh(self, idx):
        # add a new column of threshold
        if idx == 0:
            self.thresh = np.append(self.thresh, self.thresh[:, -1:], axis=1)   
            self.thresh_factor = np.append(self.thresh_factor, self.thresh_factor[:, -1:], axis=1)   
            self.peak_level = np.append(self.peak_level, self.peak_level[:, -1:], axis=1)   
            self.std = np.append(self.std, self.std[:, -1:], axis=1)   
        tt = self.t_s[idx, self.n - np.int(self.window * 2.5):self.n]
     
        if self.index[idx, self.index_track[idx]-100] > 300:       
            temp = self.peak_to_std[idx, self.index_track[idx]-100:self.index_track[idx]]
        else:
            temp = self.peak_to_std[idx, np.where(self.index[idx, :]>300)[0][0]: self.index_track[idx]]
        peak_level = np.percentile(temp, 95)
        thresh_factor = self.thresh_factor[idx, 0] * peak_level / self.peak_level[idx, -1]
        
        if thresh_factor >= self.minimal_thresh:
            if thresh_factor <= self.thresh_factor[idx, -1]:
                self.thresh_factor[idx, -1] = thresh_factor   # thresh_factor never increase
        else:
            self.thresh_factor[idx, -1] = self.minimal_thresh
        
        if self.robust_std:
            thresh_new = (-np.percentile(tt, 25) * 2 / 1.35) * self.thresh_factor[idx]
        else:    
            std = compute_std(tt)
            thresh_new =  std* self.thresh_factor[idx, -1]
        self.std[idx, -1] = std
        self.thresh[idx, -1] = thresh_new        
        return self
        
    def compute_SNR(self):
        for idx in range(self.trace.shape[0]):
            sg = np.percentile(self.t_s[idx], 99)
            std = compute_std(self.t_s[idx])
            snr = sg / std
            self.SNR[idx] = snr
        return self
            
    def reconstruct_signal(self):
        self.t_rec = np.zeros(self.trace.shape)
        for idx in range(self.trace.shape[0]):
            spikes = np.array(list(set(self.index[idx])-set([0])))
            if spikes.size > 0:
                self.t_rec[idx, spikes] = 1
                self.t_rec[idx] = np.convolve(self.t_rec[idx], np.flip(self.PTA[idx]), 'same')   #self.scale[idx,0]   
        return self
                
    def reconstruct_movie(self, A, shape, scope):
        self.A = A.reshape((shape[0], shape[1], -1), order='F')
        self.C = self.t_rec[:, scope[0]:scope[1]]
        self.mov_rec = np.dot(self.A, self.C).transpose((2,0,1))    
        return self

def find_spikes_tm(img, freq, fr, do_scale=False, filt_window=15, template_window=2, robust_std=False, 
                   adaptive_threshold=True, minimal_thresh=3.0, do_plot=False):
    """
    Offline template matching methods to find peaks, decide threshold
    Parameters
    ----------
    img : ndarray
        Input one dimensional detrended signal.
    freq : float
        Frequency for subthreshold extraction.
    fr : float
        Movie frame rate
    do_scale : bool, optional
        Whether scale the signal or not. The default is False.
    filt_window: int or list
        window size for the median filter. The median filter is not symmetric when the two values 
        in the list is not the same(for example [8, 4] means using 8 past frames and 4 future frames for 
                                    removing the median of the current frame)                 
    template_window: int
        half window size for template matching. Will not perform template matching
        if 0.                
    robust_std: bool
        whether to use robust way to estimate noise
    adaptive_threshold: bool
        whether to use adaptive threshold method for deciding threshold level.
        Currently it should always be True.
    minimal_thresh: float
        minimal threshold allowed for adaptive threshold. Threshold lower than minimal_thresh
        will be adjusted to minimal_thresh.
    do_plot: bool
        Whether to plot or not.
    
    Returns
    -------
    index : ndarray
        An array of spike times.
    thresh2 : float
        Threshold after template matching.
    PTA : ndarray
        Spike template. 
    t0 : ndarray
        Normalized trace.
    t : ndarray
        Trace removed from subthreshold.
    t_s : ndarray
        Trace after template matching.
    sub : ndarray
        Trace of subtrehsold.
    median : float
        Median of the original trace
    scale : float
        Scale of the original trace based on 1 percentile.
    thresh_factor : float
        A value determine the threshold level.
    median2 : float
        Median of the trace after template matching. 
    std: float
        Estimated standard deviation
    peak_to_std: float
        Spike height devided by std
    peak_level: float
        95 percentile of the peak_to_std. 
        It is updated to compensate for the photobleaching effect.
    """
    # Normalize and remove subthreshold
    median = np.median(img)
    scale = -np.percentile(img-median, 1)
    if do_scale == True:
        t0 = (img - median) / scale
    else:
        t0 = img.copy() - median
    
    #sub = signal_filter(t0, freq, fr, order=1, mode='low')
    #filt_window=15
    if type(filt_window) is list:
        sub = non_symm_median_filter(t0, filt_window)                
    else:
        sub = median_filter(t0, size=filt_window)
    t = t0 - sub

    # First time thresholding
    data = t.copy()
    if do_plot:
        plt.figure(); plt.plot(data);plt.show()
    pks = data[signal.find_peaks(data, height=None)[0]]
    thresh, _, _, _ = adaptive_thresh(pks, clip=100, pnorm=0.25, min_spikes=10)
    idx = signal.find_peaks(data, height=thresh)[0]
    
    # Form a template
    if template_window > 0:
        window_length = template_window
    else:
        window_length = 2 
    window = np.int64(np.arange(-window_length, window_length + 1, 1))
    idx = idx[np.logical_and(idx > (-window[0]), idx < (len(data) - window[-1]))]
    PTD = data[(idx[:, np.newaxis] + window)]
    PTA = np.median(PTD, 0)
    
    if template_window > 0:
        # Template matching, second time thresholding
        t_s = np.convolve(data, np.flipud(PTA), 'same')
        data = t_s.copy()
    else:
        logging.info('skip template matching')
        data = t.copy()
        t_s = t.copy()

    median2 = np.median(data)
    data = data - median2
    if robust_std:
        std = -np.percentile(data, 25) * 2 /1.35
    else:
        ff1 = -data * (data < 0)
        Ns = np.sum(ff1 > 0)
        std = np.sqrt(np.divide(np.sum(ff1**2), Ns)) 
        
    # Select best threshold based on estimated false positive rate
    if adaptive_threshold:
        pks2 = t_s[signal.find_peaks(t_s, height=None)[0]]
        try:
            thresh2, falsePosRate, detectionRate, low_spikes = adaptive_thresh(pks2, clip=0, pnorm=0.5, min_spikes=10)  # clip=0 means no clipping
            thresh_factor = thresh2 / std
            if thresh_factor < minimal_thresh:
                logging.info(f'Adaptive threshold factor is lower than minimal theshold: {minimal_thresh}, choose thresh factor to be {minimal_thresh}')
                thresh_factor = minimal_thresh
            thresh2 = thresh_factor * std
        except:
            logging.info('Adaptive threshold fails, automatically choose thresh factor to be 3')
            thresh_factor = 3
            thresh2 = thresh_factor * std
    else:
        raise ValueError('other methods are not implemented yet')

    index = signal.find_peaks(data, height=thresh2)[0]
    logging.info(f'final threshhold equals: {thresh2/std}')
                
    # plot signal, threshold, template and peak distribution
    if do_plot:
        plt.figure()
        plt.plot(t)
        plt.hlines(thresh, 0, len(t_s))
        plt.title('signal before template matching')
        plt.show()
        #plt.pause(3)

        plt.figure()
        plt.plot(t_s)
        plt.hlines(thresh2, 0, len(t_s))
        plt.title('signal after template matching')
        plt.show()
        #plt.pause(5)
        
        plt.figure();
        plt.plot(PTA);
        plt.title('template')
        plt.show()

        plt.figure()
        plt.hist(pks2, 500)
        plt.axvline(x=thresh2, c='r')
        plt.title('distribution of spikes')
        plt.tight_layout()
        plt.show()  
        plt.pause(5)
        
    peak_to_std = data[index] / std    
    try:
        peak_level = data[index[index>300]] / std # remove peaks in first three hundred frames to improve robustness
    except:
        peak_level = data[index] / std 
    peak_level = np.percentile(peak_level, 95)

    return index, thresh2, PTA, t0, t, t_s, sub, median, scale, thresh_factor, median2, std, peak_to_std, peak_level 
