#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from . import atm
from . import spikepursuit
from .volparams import volparams

try:
    profile
except:
    def profile(a): return a


class VOLPY(object):
    """ Spike Detection in Voltage Imaging
        The general file class which is used to find spikes of voltage imaging.
        Its architecture is similar to the one of scikit-learn calling the function fit
        to run everything which is part of the structure of the class.
        The output will be recorded in self.estimate.
    """

    def __init__(self, n_processes, dview=None, doCrossVal=False, doGlobalSubtract=False,
                 contextSize=50, censorSize=12, nPC_bg=8, tau_lp=3, tau_pred=1,
                 sigmas=np.array([1, 1.5, 2]), nIter=5, localAlign=False, globalAlign=True,
                 highPassRegression=False, params=None):
        """
            n_processes: int
                number of processed used (if in parallel this controls memory usage)

            dview: Direct View object
                for parallelization pruposes when using ipyparallel

            doCrossVal: boolean
                whether to use cross validation to optimize regression regularization parameters

            doGlobalSubtract: boolean
                whether to subtract the signal which can be predicted by the entire video

            contextSize: int
                number of pixels surrounding the ROI to use as context

            censorSize: int
                number of pixels surrounding the ROI to censor from the background PCA; roughly
                the spatial scale of scattered/dendritic neural signals, in pixels

            nPC_bg: int
                number of principle components used for background subtraction

            tau_lp: int
                time window for lowpass filter (seconds); signals slower than this will be ignored

            tau_pred: int
                time window in seconds for high pass filtering to make predictor for regression

            sigmas: 1-d array
                spatial smoothing radius imposed on spatial filter

            nIter: int
                number of iterations alternating between estimating temporal and spatial filters

            localAlign: boolean

            globalAlign: boolean

            highPassRegression: boolean
                whether to regress on a high-passed version of the data.
                Slightly improves detection of spikes, but makes subthreshold unreliable
        """

        if params is None:
            self.params = volparams(doCrossVal=doCrossVal, doGlobalSubtract=doGlobalSubtract,
                                    contextSize=contextSize, censorSize=censorSize, nPC_bg=nPC_bg,
                                    tau_lp=tau_lp, tau_pred=tau_pred, sigmas=sigmas, nIter=nIter,
                                    localAlign=localAlign, globalAlign=globalAlign,
                                    highPassRegression=highPassRegression)
        else:
            self.params = params
            # params.set('patch', {'n_processes': n_processes})
        self.estimates = {}

    def fit(self, n_processes=None, dview=None):
        """Run the volspike function to detect spikes and save the result
        into self.estimate
        """
        results = []
        fnames = self.params.data['fnames']
        fr = self.params.data['fr']

        if self.params.volspike['method'] == 'SpikePursuit':
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
                ROIs = self.params.data['ROIs'][i]
                if self.params.data['weights'] is None:
                    weights = None
                else:
                    weights = self.params.data['weights'][i]
                args_in.append([fnames, fr, i, ROIs, weights, self.params.volspike])

            if 'multiprocessing' in str(type(dview)):
                results_part = dview.map_async(volspike, args_in).get(4294967)
            elif dview is not None:
                results_part = dview.map_sync(volspike, args_in)
            else:
                results_part = list(map(volspike, args_in))
                    
            results = results + results_part
        
        N = len(results)
        print(N)
        self.estimates['trace_raw'] = [results[i]['trace_raw'] for i in range(N)]
        self.estimates['trace_processed'] = [results[i]['trace_processed'] for i in range(N)]
        self.estimates['trace_recons'] = [results[i]['trace_recons'] for i in range(N)]
        self.estimates['trace_sub'] = [results[i]['trace_sub'] for i in range(N)]
        self.estimates['spikes'] = [results[i]['spikes'] for i in range(N)]
        self.estimates['num_spikes'] = [results[i]['num_spikes'] for i in range(N)]
        self.estimates['spatialFilter'] = [results[i]['spatialFilter'] for i in range(N)]
        self.estimates['locality'] = [results[i]['locality'] for i in range(N)]
        self.estimates['snr'] = [results[i]['snr'] for i in range(N)]
        self.estimates['cellN'] = [results[i]['cellN'] for i in range(N)]
        self.estimates['no_spike'] = [results[i]['no_spike'] for i in range(N)]

        

        return self

