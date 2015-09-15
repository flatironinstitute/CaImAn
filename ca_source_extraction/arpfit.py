# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 15:38:15 2015

@author: epnevmatikakis
"""

import numpy as np
from scipy.fftpack import fft, ifft

def arpfit(Y, p = 2, sn = None, g = None, noise_range = [0.25,0.5], noise_method = 'logmexp', lags = 5, include_noise = False, pixels = None):
        
    if pixels is None:
        pixels = np.arange(np.size(Y)/np.shape(Y)[-1])
        
    P = dict()
    if sn is None:
        sn = get_noise_fft(Y, noise_range = noise_range, noise_method = noise_method)
        
    P['sn'] = sn
    
    if g is None:
        g = estimate_time_constant(Y, sn, p = p, lags = lags, include_noise = include_noise, pixels = pixels)
        
    P['g'] = g
    
    return P

def estimate_time_constant(Y, sn, p = 2, lags = 5, include_noise = False, pixels = None):
        
    if pixels is None:
        pixels = np.arange(np.size(Y)/np.shape(Y)[-1])
    
    from scipy.linalg import toeplitz    
    
    npx = len(pixels)
    g = 0
    lags += p
    XC = np.zeros((npx,2*lags+1))
    for j in range(npx):
        XC[j,:] = np.squeeze(axcov(np.squeeze(Y[pixels[j],:]),lags))
        
    gv = np.zeros(npx*lags)
    if not include_noise:
        XC = XC[:,np.arange(lags-1,-1,-1)]
        lags -= p
        
    A = np.zeros((npx*lags,p))
    for i in range(npx):
        if not include_noise:
            A[i*lags+np.arange(lags),:] = toeplitz(np.squeeze(XC[i,np.arange(p-1,p+lags-1)]),np.squeeze(XC[i,np.arange(p-1,-1,-1)])) 
        else:
            A[i*lags+np.arange(lags),:] = toeplitz(np.squeeze(XC[i,lags+np.arange(lags)]),np.squeeze(XC[i,lags+np.arange(p)])) - (sn[i]**2)*np.eye(lags,p)
            gv[i*lags+np.arange(lags)] = np.squeeze(XC[i,lags+1:])
        
    if not include_noise:
        gv = XC[:,p:].T
        gv = np.squeeze(np.reshape(gv,(np.size(gv),1),order='F'))
        
    g = np.dot(np.linalg.pinv(A),gv)
    
    return g
    
def get_noise_fft(Y, noise_range = [0.25,0.5], noise_method = 'logmexp'):
    
    T = np.shape(Y)[-1]
    dims = len(np.shape(Y))
    ff = np.arange(0,0.5,1./T)
    ind1 = ff > noise_range[0]
    ind2 = ff <= noise_range[1]
    ind = np.logical_and(ind1,ind2)
    if dims > 1:
        sn = 0
        xdft = fft(Y,axis=-1)
        xdft = xdft[...,:T/2+2]
        psdx = (1./T)*np.abs(xdft)**2
        psdx[...,1:] *= 2
        sn = mean_psd(psdx[...,ind], method = noise_method)
        
    else:
        xdft = fft(Y)
        xdft = xdft[:T/2+2]
        psdx = (1./T)*np.abs(xdft)**2
        psdx[1:] *=2
        sn = mean_psd(psdx[ind], method = noise_method)
    
    return sn
    
def mean_psd(y, method = 'logmexp'):
    
    if method == 'mean':
        mp = np.sqrt(np.mean(y/2,axis=-1))
    elif method == 'median':
        mp = np.sqrt(np.median(y/2,axis=-1))
    else:
        mp = np.sqrt(np.exp(np.mean(np.log(y/2),axis=-1)))
        
    return mp

def axcov(data, maxlag=5):
    """
    Compute the autocovariance of data at lag = -maxlag:0:maxlag
    Parameters
    ----------
    data : array
        Array containing fluorescence data
    maxlag : int
        Number of lags to use in autocovariance calculation
    Returns
    -------
    axcov : array
        Autocovariances computed from -maxlag:0:maxlag
    """
    
    data = data - np.mean(data)
    T = len(data)
    bins = np.size(data)
    xcov = fft(data, np.power(2, nextpow2(2 * bins - 1)))
    xcov = ifft(np.square(np.abs(xcov)))    
    xcov = np.concatenate([xcov[np.arange(xcov.size - maxlag, xcov.size)],
                           xcov[np.arange(0, maxlag + 1)]])
    #xcov = xcov/np.concatenate([np.arange(T-maxlag,T+1),np.arange(T-1,T-maxlag-1,-1)])
    return np.real(xcov/T)
    
def nextpow2(value):
    """
    Find exponent such that 2^exponent is equal to or greater than abs(value).
    Parameters
    ----------
    value : int
    Returns
    -------
    exponent : int
    """
    
    exponent = 0
    avalue = np.abs(value)
    while avalue > np.power(2, exponent):
        exponent += 1
    return exponent                    