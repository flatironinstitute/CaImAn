#!/usr/bin/env python
# -*- coding: utf-8 -*-

from builtins import object
from builtins import str

import cv2
import inspect
import logging
import numpy as np
import os
import psutil
import scipy
import sys
from .spikePursuit import volspike
from .Volparams import volparams

try:
    cv2.setNumThreads(0)
except:
    pass

try:
    profile
except:
    def profile(a): return a

class VOLPY(object):
    """ Spike Detection in Voltage Imaging
        The general file class which is used to find spikes of voltage imaging.
        Its architecture is similar to the one of scikit-learn calling the function fit to run everything which is part
        of the structure of the class. The output will be recorded in self.estimate.
    """
    def __init__(self, n_processes, dview=None, doCrossVal=False, doGlobalSubtract=False,
            contextSize=50, censorSize=12, nPC_bg=8, tau_lp=3, tau_pred=1, sigmas=np.array([1,1.5,2]),
            nIter=5, localAlign=False, globalAlign=True, highPassRegression=False, params=None):
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
                whether to regress on a high-passed version of the data. Slightly improves detection of spikes,
                but makes subthreshold unreliable"""

        self.dview = dview
        if params is None:
            self.params =volparams(doCrossVal=doCrossVal, doGlobalSubtract=doGlobalSubtract,
            contextSize=contextSize, censorSize=censorSize, nPC_bg=nPC_bg, tau_lp=tau_lp, tau_pred=tau_pred, sigmas=sigmas,
            nIter=nIter, localAlign=localAlign, globalAlign=globalAlign, highPassRegression=highPassRegression)
        else:
            self.params = params
            #params.set('patch', {'n_processes': n_processes})
        self.estimates = {}

    def fit(self):
        """Run the volspike function to detect spikes and save the result 
        into self.estimate        
        """
        args = dict()
        args['fnames'] = self.params.data['fnames']
        args['fr'] = self.params.data['fr']
        args['ROIs'] = self.params.data['ROIs']
        args['doCrossVal'] = self.params.volspike['doCrossVal']
        args['doGlobalSubtract'] = self.params.volspike['doGlobalSubtract']
        args['contextSize'] = self.params.volspike['contextSize']
        args['censorSize'] = self.params.volspike['censorSize']
        args['nPC_bg'] = self.params.volspike['nPC_bg']
        args['tau_lp'] = self.params.volspike['tau_lp']
        args['tau_pred'] = self.params.volspike['tau_pred']
        args['sigmas'] = self.params.volspike['sigmas']
        args['nIter'] = self.params.volspike['nIter']
        args['localAlign'] = self.params.volspike['localAlign']
        args['globalAlign'] = self.params.volspike['globalAlign']
        args['highPassRegression'] = self.params.volspike['highPassRegression']

        args_in = []
        for i in self.params.data['index']:
            args_in.append([args, i])

        if 'multiprocessing' in str(type(self.dview)):
            results = self.dview.map_async(volspike, args_in).get(4294967)
        elif self.dview is not None:
            results = self.dview.map_sync(volspike,args_in)
        else:
            results = list(map(volspike, args_in))

        N = len(results)
        self.estimates['spikeTimes'] = [results[i]['spikeTimes'] for i in range(N)]
        self.estimates['trace'] = [results[i]['yFilt'] for i in range(N)]
        self.estimates['spatialFilter'] = [results[i]['spatialFilter'] for i in range(N)]
        self.estimates['cellN'] = [results[i]['cellN'] for i in range(N)]
        self.estimates['templates'] = [results[i]['templates'] for i in range(N)]
        self.estimates['snr'] = [results[i]['snr'] for i in range(N)]
        self.estimates['num_spikes'] = [results[i]['num_spikes'] for i in range(N)]
        self.estimates['passedLocalityTest'] = [results[i]['passedLocalityTest'] for i in range(N)]
        self.estimates['low_spk'] = [results[i]['low_spk'] for i in range(N)]
        self.estimates['weights'] = [results[i]['weights'] for i in range(N)]

        return self


























