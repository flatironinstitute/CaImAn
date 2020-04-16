#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import numpy as np
from . import atm
from . import spikepursuit

try:
    profile
except:
    def profile(a): return a


class VOLPY(object):
    """ Spike Detection in Voltage Imaging
        The general file class which is used to find spikes of voltage imaging.
        Its architecture is similar to the one of scikit-learn calling the function fit
        to run everything which is part of the structure of the class.
        The output will be recorded in self.estimates.
        In order to use VolPy within CaImAn, you must install Keras into your conda environment. 
        You can do this by activating your environment, and then issuing the command 
        "conda install -c conda-forge keras".
    """
    def __init__(self, n_processes, dview=None, context_size=35, censor_size=12, 
                 flip_signal=True, hp_freq_pb=1/3, nPC_bg=8, ridge_bg=0.01,  
                 hp_freq=1, threshold_method='simple', min_spikes=10, threshold=4, 
                 sigmas=np.array([1, 1.5, 2]), n_iter=2, weight_update='ridge', 
                 do_plot=True, do_cross_val=False, sub_freq=75, 
                 method='spikepursuit', superfactor=10, params=None):
        
        """
        Args:
            n_processes: int
                number of processes used 
        
            dview: Direct View object
                for parallelization pruposes when using ipyparallel
                
            context_size: int
                number of pixels surrounding the ROI to use as context

            censor_size: int
                number of pixels surrounding the ROI to censor from the background PCA; roughly
                the spatial scale of scattered/dendritic neural signals, in pixels
                
            flip_signal: boolean
                whether to flip signal upside down for spike detection 
                True for voltron, False for others

            hp_freq_pb: float
                high-pass frequency for removing photobleaching    
            
            nPC_bg: int
                number of principal components used for background subtraction
                
            ridge_bg: float
                regularization strength for ridge regression in background removal 

            hp_freq: float
                high-pass cutoff frequency to filter the signal after computing the trace

            threshold_method: str
                'simple' or 'adaptive_threshold' method for thresholding signals
                'simple' method threshold based on estimated noise level 
                'adaptive_threshold' method threshold based on estimated peak distribution
                
            min_spikes: int
                minimal number of spikes to be detected

            threshold: float
                threshold for spike detection in 'simple' threshold method 
                The real threshold is the value multiplied by the estimated noise level

            sigmas: 1-d array
                spatial smoothing radius imposed on high-pass filtered 
                movie only for finding weights

            n_iter: int
                number of iterations alternating between estimating spike times
                and spatial filters
                
            weight_update: str
                'ridge' or 'NMF' for weight update
                
            do_plot: boolean
                if Ture, plot trace of signals and spiketimes, 
                peak triggered average, histogram of heights in the last iteration

            do_cross_val: boolean
                whether to use cross validation to optimize regression regularization parameters
                
            sub_freq: float
                frequency for subthreshold extraction
                
            method: str
                'spikepursuit' or 'atm' method
                
            superfactor: int
                used in 'atm' method for regression
        """
        if params is None:
            logging.warning("Parameters are not set from volparams")
            raise Exception('Parameters are not set')
        else:
            self.params = params

        self.estimates = {}

    def fit(self, n_processes=None, dview=None):
        """Run the volspike function to detect spikes and save the result
        into self.estimates
        """
        results = []
        fnames = self.params.data['fnames']
        fr = self.params.data['fr']

        if self.params.volspike['method'] == 'spikepursuit':
            volspike = spikepursuit.volspike
        elif self.params.volspike['method'] == 'atm':
            volspike = atm.volspike    
   
        N = len(self.params.data['index'])
        times = int(np.ceil(N/n_processes))
        for j in range(times):
            if j < (times - 1):
                li = [k for k in range(j*n_processes, (j+1)*n_processes)]
            else:
                li = [k for k in range(j*n_processes, N )]
            args_in = []
            
            for i in li:
                idx = self.params.data['index'][i]
                ROIs = self.params.data['ROIs'][idx]
                if self.params.data['weights'] is None:
                    weights = None
                else:
                    weights = self.params.data['weights'][i]
                args_in.append([fnames, fr, idx, ROIs, weights, self.params.volspike])

            if 'multiprocessing' in str(type(dview)):
                results_part = dview.map_async(volspike, args_in).get(4294967)
            elif dview is not None:
                results_part = dview.map_sync(volspike, args_in)
            else:
                results_part = list(map(volspike, args_in))
            results = results + results_part
        
        for i in results[0].keys():
                self.estimates[i] = np.array([results[j][i] for j in range(N)])

        return self
