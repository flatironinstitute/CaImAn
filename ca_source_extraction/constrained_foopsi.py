# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 16:11:25 2015

@author: Eftychios A. Pnevmatikakis, based on an implementation by T. Machado,  Andrea Giovannucci & Ben Deverett
"""

# -*- coding: utf-8 -*-
# Written by 

import numpy as np
import scipy.signal 
import scipy.linalg

from warnings import warn    
#import time
#import sys    

#%%
def constrained_foopsi(fluor, 
                     b = None, 
                     c1 = None, 
                     g = None, 
                     sn = None, 
                     p= 2, 
                     method = 'cvx', 
                     bas_nonneg = True, 
                     noise_range = [.25,.5],
                     noise_method = 'logmexp',
                     lags = 5, 
                     resparse = 0,
                     fudge_factor = 1, 
                     verbosity = False):

    """
    Infer the most likely discretized spike train underlying a fluorescence
    trace, using a noise constrained deconvolution approach
    Inputs
    ----------
    fluor   : nparray
        One dimensional array containing the fluorescence intensities with
        one entry per time-bin.
    b       : float, optional
        Fluorescence baseline balue. If no value is given, then b is estimated 
        from the data
    c1      : 
    g       : float, optional
        Parameters of the AR process that models the fluorescence impulse response.
        Estimated from the data if no value is given
    sn      : float, optional
        Standard deviation of the noise distribution.  If no value is given, 
        then sn is estimated from the data.        
    options : dictionary
        list of user selected options (see more below)


    'p'             :         2, # AR order 
    'method'        :     'cvx', # solution method (no other currently supported)
    'bas_nonneg'    :      True, # bseline strictly non-negative
    'noise_range'   :  [.25,.5], # frequency range for averaging noise PSD
    'noise_method'  : 'logmexp', # method of averaging noise PSD
    'lags'          :         5, # number of lags for estimating time constants
    'resparse'      :         0, # times to resparse original solution (not supported)
    'fudge_factor'  :         1, # fudge factor for reducing time constant bias
    'verbosity'     :     False, # display optimization details
    
    Returns
    -------
    c            : ndarray of float
        The inferred denoised fluorescence signal at each time-bin.
    b, c1, g, sn : As explained above
    sp           : ndarray of float
        Discretized deconvolved neural activity (spikes)
    
    References
    ----------
    * Pnevmatikakis et al. 2015. Submitted (arXiv:1409.2903).
    * Machado et al. 2015. Cell 162(2):338-350
    """

    
    if g is None or sn is None:        
        g,sn = estimate_parameters(fluor, p=p, sn=sn, g = g, range_ff=noise_range, method=noise_method, lags=lags, fudge_factor=fudge_factor)
    
    if method == 'cvx':
        c,b,c1,g,sn,sp = cvxopt_foopsi(fluor, b =b, c1 = c1, g=g, sn=sn, p=p, bas_nonneg = bas_nonneg, verbosity = verbosity)
    elif method == 'spgl1':
        try:        
            c,b,c1,g,sn,sp = spgl1_foopsi(fluor, b =b, c1 = c1, g=g, sn=sn, p=p, bas_nonneg = bas_nonneg, verbosity = verbosity)
        except:
            print('SPGL1 produces an error. Using CVXOPT')
            c,b,c1,g,sn,sp = cvxopt_foopsi(fluor, b =b, c1 = c1, g=g, sn=sn, p=p, bas_nonneg = bas_nonneg, verbosity = verbosity)    
    
    return c,b,c1,g,sn,sp

def spgl1_foopsi(fluor, b, c1, g, sn, p, bas_nonneg, verbosity):
    
    import sys
    sys.path.append('../../SPGL1_python_port/')
    #sys.path.append('/Users/eftychios/Documents/Python/SPGL1_python_port/')
    from spgl1 import spg_bpdn
    import spgl_aux as spg
                     
    if b is None:
        bas_flag = True
        b = 0
    else:
        bas_flag = False
        
    if c1 is None:
        c1_flag = True
        c1 = 0
    else:
        c1_flag = False

    if bas_nonneg:
        b_lb = 0
    else:
        b_lb = np.min(fluor)
        
    T = len(fluor)
    w = np.ones(np.shape(fluor))
    if bas_flag:
        w = np.hstack((w,1e-10))
        
    if c1_flag:
        w = np.hstack((w,1e-10))
        
    gr = np.roots(np.concatenate([np.array([1]),-g.flatten()])) 
    gd_vec = np.max(gr)**np.arange(T)  # decay vector for initial fluorescence
    
    options = {'project' : spg.NormL1NN_project ,
               'primal_norm' : spg.NormL1NN_primal ,
               'dual_norm' : spg.NormL1NN_dual,
               'weights'   : w, 
               'verbosity' : verbosity}
    
    opA = lambda x,mode: G_inv_mat(x,mode,T,g,gd_vec,bas_flag,c1_flag)
    
    spikes = spg_bpdn(opA,np.squeeze(fluor)-bas_nonneg*b_lb - (1-bas_flag)*b -(1-c1_flag)*c1*gd_vec, sn*np.sqrt(T), options)[0]
    c = opA(np.hstack((spikes[:T],0)),1)
    if bas_flag:
        b = spikes[T] + b_lb
    
    if c1_flag:
        c1 = spikes[-1]
        
    return c,b,c1,g,sn,spikes
    

def G_inv_mat(x,mode,NT,gs,gd_vec,bas_flag = True, c1_flag = True):

    from scipy.signal import lfilter
    if mode == 1:
        b = lfilter(np.array([1]),np.concatenate([np.array([1.]),-gs]),x[:NT]) + bas_flag*x[NT-1+bas_flag] + c1_flag*gd_vec*x[-1]
        #b = np.dot(Emat,b)
    elif mode == 2:
        #x = np.dot(Emat.T,x)
        b = np.hstack((np.flipud(lfilter(np.array([1]),np.concatenate([np.array([1.]),-gs]),np.flipud(x))),np.ones(bas_flag)*np.sum(x),np.ones(c1_flag)*np.sum(gd_vec*x)))
            
    return b

  
def cvxopt_foopsi(fluor, b, c1, g, sn, p, bas_nonneg, verbosity):

    try:
        from cvxopt import matrix, spmatrix, spdiag, solvers
        import picos
    except ImportError:
        raise ImportError('Constrained Foopsi requires cvxopt and picos packages.')
    
    T = len(fluor)

    # construct deconvolution matrix  (sp = G*c) 
    G = spmatrix(1.,range(T),range(T),(T,T))

    for i in range(p):
        G = G + spmatrix(-g[i],np.arange(i+1,T),np.arange(T-i-1),(T,T))
        
    gr = np.roots(np.concatenate([np.array([1]),-g.flatten()])) 
    gd_vec = np.max(gr)**np.arange(T)  # decay vector for initial fluorescence
    gen_vec = G * matrix(np.ones(fluor.size))  
    
    # Initialize variables in our problem
    prob = picos.Problem()
    
    # Define variables
    calcium_fit = prob.add_variable('calcium_fit', fluor.size)    
    cnt = 0
    if b is None:
        flag_b = True
        cnt += 1
        b = prob.add_variable('b', 1)
        if bas_nonneg:
            b_lb = 0
        else:
            b_lb = np.min(fluor)
            
        prob.add_constraint(b >= b_lb)
    else:
        flag_b = False

    if c1 is None:
        flag_c1 = True
        cnt += 1
        c1 = prob.add_variable('c1', 1)
        prob.add_constraint(c1 >= 0)
    else:
        flag_c1 = False
    
    # Add constraints    
    prob.add_constraint(G * calcium_fit >= 0)
    res = abs(matrix(fluor.astype(float)) - calcium_fit - b*matrix(np.ones(fluor.size)) - matrix(gd_vec) * c1)
    prob.add_constraint(res < sn * np.sqrt(fluor.size))
    prob.set_objective('min', calcium_fit.T * gen_vec)
    
    # solve problem
    try:
        prob.solve(solver='mosek', verbose=verbosity)
        sel_solver = 'mosek'
#        prob.solve(solver='gurobi', verbose=verbosity)
#        sel_solver = 'gurobi'
    except ImportError:
        warn('MOSEK is not installed. Spike inference may be VERY slow!')
        sel_solver = []
        prob.solver_selection()
        prob.solve(verbose=verbosity)
        
    # if problem in infeasible due to low noise value then project onto the cone of linear constraints with cvxopt
    if prob.status == 'prim_infeas_cer' or prob.status == 'dual_infeas_cer' or prob.status == 'primal infeasible':
        warn('Original problem infeasible. Adjusting noise level and re-solving')   
        # setup quadratic problem with cvxopt        
        solvers.options['show_progress'] = verbosity
        ind_rows = range(T)
        ind_cols = range(T)
        vals = np.ones(T)
        if flag_b:
            ind_rows = ind_rows + range(T) 
            ind_cols = ind_cols + [T]*T
            vals = np.concatenate((vals,np.ones(T)))
        if flag_c1:
            ind_rows = ind_rows + range(T)
            ind_cols = ind_cols + [T+cnt-1]*T
            vals = np.concatenate((vals,np.squeeze(gd_vec)))            
        P = spmatrix(vals,ind_rows,ind_cols,(T,T+cnt))
        H = P.T*P
        Py = P.T*matrix(fluor.astype(float))
        sol = solvers.qp(H,-Py,spdiag([-G,-spmatrix(1.,range(cnt),range(cnt))]),matrix(0.,(T+cnt,1)))
        xx = sol['x']
        c = np.array(xx[:T])
        sp = np.array(G*matrix(c))
        c = np.squeeze(c)
        if flag_b:
            b = np.array(xx[T+1]) + b_lb
        if flag_c1:
            c1 = np.array(xx[-1])
        sn = np.linalg.norm(fluor-c-c1*gd_vec-b)/np.sqrt(T)   
    else: # readout picos solution
        c = np.squeeze(calcium_fit.value)
        sp = np.squeeze(np.asarray(G*calcium_fit.value))        
        if flag_b:    
            b = np.squeeze(b.value)        
        if flag_c1:    
            c1 = np.squeeze(c1.value)                    

    return c,b,c1,g,sn,sp


def estimate_parameters(fluor, p = 2, sn = None, g = None, range_ff = [0.25,0.5], method = 'logmexp', lags = 5, fudge_factor = 1):
    """
    Estimate noise standard deviation and AR coefficients if they are not present
    """
    
    if sn is None:
        sn = GetSn(fluor,range_ff,method)
        
    if g is None:
        g = estimate_time_constant(fluor,p,sn,lags,fudge_factor)

    return g,sn

def estimate_time_constant(fluor, p = 2, sn = None, lags = 5, fudge_factor = 1):
    """    
    Estimate AR model parameters through the autocovariance function    
    Inputs
    ----------
    fluor        : nparray
        One dimensional array containing the fluorescence intensities with
        one entry per time-bin.
    p            : positive integer
        order of AR system  
    sn           : float
        noise standard deviation, estimated if not provided.
    lags         : positive integer
        number of additional lags where he autocovariance is computed
    fudge_factor : float (0< fudge_factor <= 1)
        shrinkage factor to reduce bias
        
    Return
    -----------
    g       : estimated coefficients of the AR process
    """    
    

    if sn is None:
        sn = GetSn(fluor)
        
    lags += p
    xc = axcov(fluor,lags)        
    xc = xc[:,np.newaxis]
    
    A = scipy.linalg.toeplitz(xc[lags+np.arange(lags)],xc[lags+np.arange(p)]) - sn**2*np.eye(lags,p)
    g = np.linalg.lstsq(A,xc[lags+1:])[0]
    if fudge_factor < 1:
        gr = fudge_factor*np.roots(np.concatenate([np.array([1]),-g.flatten()]))
        gr = (gr+gr.conjugate())/2
        gr[gr>1] = 0.95
        gr[gr<0] = 0.15
        g = np.poly(gr)
        g = -g[1:]        
        
    return g.flatten()
    
def GetSn(fluor, range_ff = [0.25,0.5], method = 'logmexp'):
    """    
    Estimate noise power through the power spectral density over the range of large frequencies    
    Inputs
    ----------
    fluor    : nparray
        One dimensional array containing the fluorescence intensities with
        one entry per time-bin.
    range_ff : (1,2) array, nonnegative, max value <= 0.5
        range of frequency (x Nyquist rate) over which the spectrum is averaged  
    method   : string
        method of averaging: Mean, median, exponentiated mean of logvalues (default)
        
    Return
    -----------
    sn       : noise standard deviation
    """
    

    ff, Pxx = scipy.signal.welch(fluor)
    ind1 = ff > range_ff[0]
    ind2 = ff < range_ff[1]
    ind = np.logical_and(ind1,ind2)
    Pxx_ind = Pxx[ind]
    sn = {
        'mean': lambda Pxx_ind: np.sqrt(np.mean(Pxx_ind/2)),
        'median': lambda Pxx_ind: np.sqrt(np.median(Pxx_ind/2)),
        'logmexp': lambda Pxx_ind: np.sqrt(np.exp(np.mean(np.log(Pxx_ind/2))))
    }[method](Pxx_ind)

    return sn

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
    xcov = np.fft.fft(data, np.power(2, nextpow2(2 * bins - 1)))
    xcov = np.fft.ifft(np.square(np.abs(xcov)))
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
    
 


