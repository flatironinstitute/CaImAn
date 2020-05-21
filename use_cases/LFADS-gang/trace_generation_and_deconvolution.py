#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed May 12 16:26:45 2020
This file gives examples of generation of calcium traces and deconvolution 
using OASIS
@author: @caichangjia and @andreagiovannucci adapted based on @j-friedrich code 
"""
#%%
import matplotlib.pyplot as plt
import numpy as np
from caiman.source_extraction.cnmf.temporal import constrained_foopsi
from scipy.interpolate import interp1d

#%%
def gen_data(trueSpikes=None, g=[.95], g_noise=0.01, sn=.3, poiss_noise_factor=1.5, T=3000, framerate=30, 
             firerate=.5, b=0, N=20, seed=13, nonlinearity=False, spike_noise = 0.1):
    """
    Generate data from homogenous Poisson Process

    Parameters
    ----------
    trueSpikes: array, optional , default None
        input spike train
    g : array, shape (p,), optional, default=[.95]
        Parameter(s) of the AR(p) process that models the fluorescence impulse response.
    sn : float, optional, default .3
        Noise standard deviation.
    poiss_noise_factor: float
        factor multiplying the poisson noise (computed over the range of the data), typical values, 2-4
    g_noise:  float 
        factor multiplying noise on g
    T : int, optional, default 3000
        Duration.
    framerate : int, optional, default 30
        Frame rate.
    firerate : int, optional, default .5
        Neural firing rate.
    b : int, optional, default 0
        Baseline.
    N : int, optional, default 20
        Number of generated traces.
    seed : int, optional, default 13
        Seed of random number generator.    
    nonlinearity: boolean, default False
        whether add nonlinearity or not
    spike_noise: float
        noise added to the spike    

    Returns
    -------
    y : array, shape (T,)
        Noisy fluorescence data.
    c : array, shape (T,)
        Calcium traces (without sn).
    s : array, shape (T,)
        Spike trains.
    g : ground truth autoregressive factors
    """
    
    if trueSpikes is None:
        np.random.seed(seed)
        Y = np.zeros((N, T))
        trueSpikes = np.random.rand(N, T) < firerate / float(framerate) 
    else:
        N, T = trueSpikes.shape
        Y = np.zeros((N, T))
    
    truth = trueSpikes.astype(float)
    truth += np.random.normal(0,spike_noise,truth.shape)
    g = np.vstack(g*N)
    g += g_noise*2*(np.random.random(g.shape)-0.5)
    
    for i in range(2, T):
        if np.shape(g)[-1] == 2:            
            raise Exception('Not compatible with other changes')
            truth[:, i] += g[:,0].T * truth[:, i - 1] + g[:,1].T * truth[:, i - 2]
        else:
            truth[:, i] += g[:,0] * truth[:, i - 1]
            
    if nonlinearity:
        f, inv_f = nonlinear_transformation()
        truth = f(truth)
    else:
        inv_f = None
#        truth = truth / truth.max(axis=1)[:, np.newaxis] * maximum[:, np.newaxis]
    fl_sig = b+truth
    fl_sig = (fl_sig-fl_sig.min())/(fl_sig.max()-fl_sig.min())
    poiss_noise = np.random.poisson(fl_sig*1000)/1000*poiss_noise_factor
    Y = b + truth + sn * np.random.randn(N, T) + poiss_noise           
    return Y, truth, trueSpikes, g
    
def nonlinear_transformation(model='dana_kim_gc6f'):
    """
    Output function for nonlinear transformation. The transformation is based on
    the property of GCAMP6f
    
    Args:
    ----     
    model: str
        which model to use: 'dana_kim_gc6f', 'druckman_gc6f-TG'
       
    Returns:
    -------
    f : function
        Nonlinear transformation
    """
    x = np.array(range(-2,160))
    if model == 'dana_kim_gc6f':
        x_orig = np.array([-2, 0.0, 1.0, 2.0,  3.0, 5.0,   10,  20, 40 , 160])
        y_orig = np.array([-0.01, 0.0, 0.1,0.25, 0.5,0.9, 1.85,3.6, 4.9, 6.1])    
    elif 'druckman_gc6f-TG': # for extrapolation used values from dana kim
        x_orig = np.array([-2,     0.0, 1.0, 2.0, 3.0,  4.0, 10,  20, 40 , 160])
        y_orig = np.array([-0.01,  0.0, 0.12, 0.3, 0.62, 1.1, 1.85, 3.6, 4.9, 6.1])    
         
    f = interp1d(x_orig, y_orig, kind='quadratic', fill_value='extrapolate')
    inv_f = interp1d(y_orig, x_orig, kind='quadratic', fill_value='extrapolate')
    # would be better to fit to a logistic
    xnew = np.arange(-2, 160, 0.01)
    ynew = f(xnew)
    plt.figure()
    plt.title('Nonlinear transformation')
    plt.plot(xnew, ynew)
    plt.plot(x_orig, y_orig,'o')
    plt.xlabel('# AP')
    plt.ylabel('deltaF/F')
    return f, inv_f

#%% Calcium trace generation
Y, truth, trueSpikes, gs = gen_data(firerate=2, N=1000, sn=0.3)
Y_nl, _, _, gs = gen_data(trueSpikes=trueSpikes, nonlinearity=True, sn=0.1)

#%% Deconvolution using OASIS
index = np.argmin(gs)
c, bl, c1, g, sn, s, lam = constrained_foopsi(Y[index], p=1)
c_nl, bl_nl, c1_nl, g_nl, sn_nl, s_nl, lam_nl = constrained_foopsi(Y_nl[index], p=1)

#%% Show without nonlinear transformation result
framerate=30
plt.figure()
tt = np.arange(0, len(c) * 1 / 30, 1 / 30)
plt.plot(tt, Y[index], label='trace')
plt.plot(tt, bl+c, label='deconvolve')
plt.plot(tt, truth[index], label='ground truth signal')
plt.plot(tt, trueSpikes[index],  label='ground truth spikes')
plt.plot(tt, s, label='deconvolved spikes')
plt.legend()
np.corrcoef(s, trueSpikes[index])


#%% Result with nonlinear transformation
plt.figure()
plt.plot(tt, Y_nl[index], label='trace')
plt.plot(tt, (bl_nl+(c_nl)), label='deconvolve')
plt.plot(tt, truth[index], label='ground truth signal')
plt.plot(tt, trueSpikes[index],  label='ground truth spikes')
plt.plot(tt, (s_nl), label='deconvolved spikes')    
plt.legend()
np.corrcoef(s_nl, trueSpikes[index])


    
    
    
    

    
