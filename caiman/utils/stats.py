#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on Thu Oct 20 13:49:57 2016

@author: agiovann
"""
import logging
from builtins import range
from past.utils import old_div
try:
    import numba
except:
    pass

import numpy as np

#%%


def mode_robust_fast(inputData, axis=None):
    """
    Robust estimator of the mode of a data set using the half-sample mode.

    .. versionadded: 1.0.3
    """

    if axis is not None:

        def fnc(x): return mode_robust_fast(x)
        dataMode = np.apply_along_axis(fnc, axis, inputData)
    else:
        # Create the function that we can use for the half-sample mode
        data = inputData.ravel()
        # The data need to be sorted for this to work
        data = np.sort(data)
        # Find the mode
        dataMode = _hsm(data)

    return dataMode
#%%


def mode_robust(inputData, axis=None, dtype=None):
    """
    Robust estimator of the mode of a data set using the half-sample mode.

    .. versionadded: 1.0.3
    """
    if axis is not None:
        def fnc(x): return mode_robust(x, dtype=dtype)
        dataMode = np.apply_along_axis(fnc, axis, inputData)
    else:
        # Create the function that we can use for the half-sample mode
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
                N = data.size // 2 + data.size % 2

                for i in range(0, N):
                    w = data[i + N - 1] - data[i]
                    if w < wMin:
                        wMin = w
                        j = i

                return _hsm(data[j:j + N])

        data = inputData.ravel()
        if type(data).__name__ == "MaskedArray":
            data = data.compressed()
        if dtype is not None:
            data = data.astype(dtype)

        # The data need to be sorted for this to work
        data = np.sort(data)

        # Find the mode
        dataMode = _hsm(data)

    return dataMode

#%%
#@numba.jit("void(f4[:])")


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
        N = old_div(data.size, 2) + data.size % 2

        for i in range(0, N):
            w = data[i + N - 1] - data[i]
            if w < wMin:
                wMin = w
                j = i

        return _hsm(data[j:j + N])


#%% kernel density estimation
   
def mode_robust_kde(inputData, axis = None):
    """
    Extracting the dataset of the mode using kernel density estimation
    """
    if axis is not None:

        def fnc(x): return mode_robust_kde(x)
        dataMode = np.apply_along_axis(fnc, axis, inputData)
    else:
        # Create the function that we can use for the half-sample mode
        bandwidth, mesh, density, cdf = kde(inputData)
        dataMode = mesh[np.argamax(density)]

    return dataMode

def df_percentile(inputData, axis = None):
    """
    Extracting the percentile of the data where the mode occurs and its value. 
    Used to determine the filtering level for DF/F extraction
    """
    if axis is not None:

        def fnc(x): return df_percentile(x)
        result = np.apply_along_axis(fnc, axis, inputData)
        data_prct = result[:, 0]
        val = result[:, 1]
    else:
        # Create the function that we can use for the half-sample mode
        err = True
        while err:
            try:
                bandwidth, mesh, density, cdf = kde(inputData)
                err = False
            except:
                logging.warning("There are no components for DF/F extraction!")
                if type(inputData) is not list:
                    inputData = inputData.tolist()
                inputData += inputData

        data_prct = cdf[np.argmax(density)]*100
        val = mesh[np.argmax(density)]

    return data_prct, val


"""
An implementation of the kde bandwidth selection method outlined in:
Z. I. Botev, J. F. Grotowski, and D. P. Kroese. Kernel density
estimation via diffusion. The Annals of Statistics, 38(5):2916-2957, 2010.
Based on the implementation in Matlab by Zdravko Botev.
Daniel B. Smith, PhD
Updated 1-23-2013
"""


import scipy as sci
import scipy.optimize
import scipy.fftpack

def kde(data, N=None, MIN=None, MAX=None):

    # Parameters to set up the mesh on which to calculate
    N = 2**12 if N is None else int(2**sci.ceil(sci.log2(N)))
    if MIN is None or MAX is None:
        minimum = min(data)
        maximum = max(data)
        Range = maximum - minimum
        MIN = minimum - Range/10 if MIN is None else MIN
        MAX = maximum + Range/10 if MAX is None else MAX

    # Range of the data
    R = MAX-MIN

    # Histogram the data to get a crude first approximation of the density
    M = len(data)
    DataHist, bins = sci.histogram(data, bins=N, range=(MIN, MAX))
    DataHist = DataHist/M
    DCTData = scipy.fftpack.dct(DataHist, norm=None)

    I = [iN*iN for iN in range(1, N)]
    SqDCTData = (DCTData[1:]/2)**2

    # The fixed point calculation finds the bandwidth = t_star
    guess = 0.1
    try:
        t_star = scipy.optimize.brentq(fixed_point, 0, guess, 
                                       args=(M, I, SqDCTData))
    except ValueError:
        print('Oops!')
        return None

    # Smooth the DCTransformed data using t_star
    SmDCTData = DCTData*sci.exp(-sci.arange(N)**2*sci.pi**2*t_star/2)
    # Inverse DCT to get density
    density = scipy.fftpack.idct(SmDCTData, norm=None)*N/R
    mesh = [(bins[i]+bins[i+1])/2 for i in range(N)]
    bandwidth = sci.sqrt(t_star)*R

    density = density/sci.trapz(density, mesh)
    cdf = np.cumsum(density)*(mesh[1]-mesh[0])

    return bandwidth, mesh, density, cdf

def fixed_point(t, M, I, a2):
    l=7
    I = sci.float64(I)
    M = sci.float64(M)
    a2 = sci.float64(a2)
    f = 2*sci.pi**(2*l)*sci.sum(I**l*a2*sci.exp(-I*sci.pi**2*t))
    for s in range(l, 1, -1):
        K0 = sci.prod(range(1, 2*s, 2))/sci.sqrt(2*sci.pi)
        const = (1 + (1/2)**(s + 1/2))/3
        time=(2*const*K0/M/f)**(2/(3+2*s))
        f=2*sci.pi**(2*s)*sci.sum(I**s*a2*sci.exp(-I*sci.pi**2*time))
    return t-(2*M*sci.sqrt(sci.pi)*f)**(-2/5)