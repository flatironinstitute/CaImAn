#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
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

    def __init__(self, n_processes, dview=None, params=None):

        if params is None:
            logging.warning("Parameters are not set from volparams")
            raise Exception('Parameters are not set')
        else:
            self.params = params

        self.estimates = {}

    def fit(self, n_processes=None, dview=None):
        """Run the volspike function to detect spikes and save the result
        into self.estimate
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
        
        self.estimates['cell_n'] = np.array([results[i]['cell_n'] for i in range(N)])
        self.estimates['t'] = np.array([results[i]['t'] for i in range(N)])
        self.estimates['ts'] = np.array([results[i]['ts'] for i in range(N)])
        self.estimates['t_rec'] = np.array([results[i]['t_rec'] for i in range(N)])
        self.estimates['t_sub'] = np.array([results[i]['t_sub'] for i in range(N)])
        self.estimates['spikes'] = np.array([results[i]['spikes'] for i in range(N)])
        self.estimates['num_spikes'] = np.array([results[i]['num_spikes'] for i in range(N)])
        self.estimates['snr'] = np.array([results[i]['snr'] for i in range(N)])
        self.estimates['low_spikes'] = np.array([results[i]['low_spikes'] for i in range(N)])
        self.estimates['spatial_filter'] = np.array([results[i]['spatial_filter'] for i in range(N)])
        self.estimates['weights'] = np.array([results[i]['weights'] for i in range(N)])
        self.estimates['locality'] = np.array([results[i]['locality'] for i in range(N)])
        self.estimates['context_coord'] = np.array([results[i]['context_coord'] for i in range(N)])

        return self

