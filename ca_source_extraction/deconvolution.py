# -*- coding: utf-8 -*-
"""Extract neural activity from a fluorescence trace using a constrained deconvolution approach
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

from SPGL1_python_port.spgl1 import spg_bpdn
from SPGL1_python_port import spgl_aux as spg
import sys
#%%
def constrained_foopsi(fluor, bl = None,  c1 = None, g = None,  sn = None, p = None, method = 'cvxpy', bas_nonneg = True,  
                     noise_range = [.25,.5], noise_method = 'logmexp', lags = 5, fudge_factor = 1., 
                    verbosity = False, solvers=None,**kwargs):
    
    """ Infer the most likely discretized spike train underlying a fluorescence trace 
    
    It relies on a noise constrained deconvolution approach
    
    
    Parameters
    ----------
    fluor: np.ndarray
        One dimensional array containing the fluorescence intensities with
        one entry per time-bin.
    bl: [optional] float
        Fluorescence baseline value. If no value is given, then bl is estimated 
        from the data.
    c1: [optional] float
        value of calcium at time 0
    g: [optional] list,float 
        Parameters of the AR process that models the fluorescence impulse response.
        Estimated from the data if no value is given
    sn: float, optional
        Standard deviation of the noise distribution.  If no value is given, 
        then sn is estimated from the data.        
    p: int
        order of the autoregression model
    method: [optional] string
        solution method for basis projection pursuit 'cvx' or 'spgl1' or 'debug' for fast but possibly imprecise temporal components    
    bas_nonneg: bool
        baseline strictly non-negative        
    noise_range:  list of two elms
        frequency range for averaging noise PSD
    noise_method: string
        method of averaging noise PSD
    lags: int 
        number of lags for estimating time constants    
    fudge_factor: float
        fudge factor for reducing time constant bias
    verbosity: bool     
         display optimization details
    solvers: list string
            primary and secondary (if problem unfeasible for approx solution) solvers to be used with cvxpy, default is ['ECOS','SCS']
    Returns
    -------
    c: np.ndarray float
        The inferred denoised fluorescence signal at each time-bin.
    bl, c1, g, sn : As explained above
    sp: ndarray of float
        Discretized deconvolved neural activity (spikes)
    
    References
    ----------
    * Pnevmatikakis et al. 2016. Neuron, in press, http://dx.doi.org/10.1016/j.neuron.2015.11.037
    * Machado et al. 2015. Cell 162(2):338-350
    """
    
    if p is None:
        raise Exception("You must specify the value of p")
        
    if g is None or sn is None:        
        g,sn = estimate_parameters(fluor, p=p, sn=sn, g = g, range_ff=noise_range, method=noise_method, lags=lags, fudge_factor=fudge_factor)
    if p == 0:       
        c1 = 0
        g = np.array(0)
        bl = 0
        c = np.maximum(fluor,0)
        sp = c.copy()
    else:
        if method == 'cvx':
            c,bl,c1,g,sn,sp = cvxopt_foopsi(fluor, b = bl, c1 = c1, g=g, sn=sn, p=p, bas_nonneg = bas_nonneg, verbosity = verbosity)
            
        elif method == 'spgl1':
    #        try:        
     
            c,bl,c1,g,sn,sp = spgl1_foopsi(fluor, bl = bl, c1 = c1, g=g, sn=sn, p=p, bas_nonneg = bas_nonneg, verbosity = verbosity)
    #        except:
    #            print('SPGL1 produces an error. Using CVXOPT')
    #            c,b,c1,g,sn,sp = cvxopt_foopsi(fluor, b =b, c1 = c1, g=g, sn=sn, p=p, bas_nonneg = bas_nonneg, verbosity = verbosity)    
        elif method == 'debug':
          
            c,bl,c1,g,sn,sp = spgl1_foopsi(fluor, bl =bl, c1 = c1, g=g, sn=sn, p=p, bas_nonneg = bas_nonneg, verbosity = verbosity,debug=True)
        
        elif method == 'cvxpy':
    
            c,bl,c1,g,sn,sp = cvxpy_foopsi(fluor,  g, sn, b=bl, c1=c1, bas_nonneg=bas_nonneg, solvers=solvers)
    
        else:
            raise Exception('Undefined Deconvolution Method')
    
    return c,bl,c1,g,sn,sp

def spgl1_foopsi(fluor, bl, c1, g, sn, p, bas_nonneg, verbosity, thr = 1e-2,debug=False):
    """Solve the deconvolution problem using the SPGL1 library
     available from https://github.com/epnev/SPGL1_python_port
    """
    fluor=fluor.copy()
    if 'spg' not in globals():
        raise Exception('The SPGL package could not be loaded, use a different method')
    
    if bl is None:
        bas_flag = True
#        kde = KernelDensity(kernel='gaussian', bandwidth=100).fit(fluor[:,np.newaxis])
#        X_plot=np.linspace(np.min(fluor),np.max(fluor),len(fluor))[:,np.newaxis]
#        p_dens = np.exp(kde.score_samples(X_plot))
#        print 'esimating baseline...'
#        n,bn=np.histogram(fluor,range=(np.percentile(fluor,1),np.percentile(fluor,99)),bins=100)
#        bl =  bn[np.argmax(n)]
        bl = 0
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
               'verbosity' : verbosity,
               'iterations': T}
    
    opA = lambda x,mode: G_inv_mat(x,mode,T,g,gd_vec,bas_flag,c1_flag)
    
    
    spikes,_,_,info = spg_bpdn(opA,np.squeeze(fluor)-bas_nonneg*b_lb - (1-bas_flag)*bl -(1-c1_flag)*c1*gd_vec, sn*np.sqrt(T))
    if np.min(spikes)<-thr*np.max(spikes) and not debug:
        spikes[:T][spikes[:T]<0]=0
        spikes,_,_,info = spg_bpdn(opA,np.squeeze(fluor)-bas_nonneg*b_lb - (1-bas_flag)*bl -(1-c1_flag)*c1*gd_vec, sn*np.sqrt(T), options)
        


    spikes[:T][spikes[:T]<0]=0
    
    c = opA(np.hstack((spikes[:T],0)),1)
    if bas_flag:
        bl = spikes[T] + b_lb
    
    if c1_flag:
        c1 = spikes[-1]
        
    return c,bl,c1,g,sn,spikes
    

def G_inv_mat(x,mode,NT,gs,gd_vec,bas_flag = True, c1_flag = True):
    """
    Fast computation of G^{-1}*x and G^{-T}*x required for using the SPGL1 method
    """
    from scipy.signal import lfilter
    if mode == 1:
        b = lfilter(np.array([1]),np.concatenate([np.array([1.]),-gs]),x[:NT]) + bas_flag*x[NT-1+bas_flag] + c1_flag*gd_vec*x[-1]
        #b = np.dot(Emat,b)
    elif mode == 2:
        #x = np.dot(Emat.T,x)
        b = np.hstack((np.flipud(lfilter(np.array([1]),np.concatenate([np.array([1.]),-gs]),np.flipud(x))),np.ones(bas_flag)*np.sum(x),np.ones(c1_flag)*np.sum(gd_vec*x)))
            
    return b

  
def cvxopt_foopsi(fluor, b, c1, g, sn, p, bas_nonneg, verbosity):
    """Solve the deconvolution problem using cvxopt and picos packages
    """
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

def cvxpy_foopsi(fluor,  g, sn, b=None, c1=None, bas_nonneg=True,solvers=None):
    '''Solves the deconvolution problem using the cvxpy package and the ECOS/SCS library. 
    Parameters:
    -----------
    fluor: ndarray
        fluorescence trace 
    g: list of doubles
        parameters of the autoregressive model, cardinality equivalent to p        
    sn: double
        estimated noise level
    b: double
        baseline level. If None it is estimated. 
    c1: double
        initial value of calcium. If None it is estimated.  
    bas_nonneg: boolean
        should the baseline be estimated        
    solvers: tuple of two strings
        primary and secondary solvers to be used. Can be choosen between ECOS, SCS, CVXOPT    

    Returns:
    --------
    c: estimated calcium trace
    b: estimated baseline
    c1: esimtated initial calcium value
    g: esitmated parameters of the autoregressive model
    sn: estimated noise level
    sp: estimated spikes 
        
    '''
    try:
        import cvxpy as cvx
    except ImportError:
        raise ImportError('cvxpy solver requires installation of cvxpy.')
    if solvers is None:
        solvers=['ECOS','SCS']
        
    T = fluor.size
    
    # construct deconvolution matrix  (sp = G*c)     
    G=scipy.sparse.dia_matrix((np.ones((1,T)),[0]),(T,T))
    
    for i,gi in enumerate(g):
        G = G + scipy.sparse.dia_matrix((-gi*np.ones((1,T)),[-1-i]),(T,T))
        
    gr = np.roots(np.concatenate([np.array([1]),-g.flatten()])) 
    gd_vec = np.max(gr)**np.arange(T)  # decay vector for initial fluorescence
    gen_vec = G.dot(scipy.sparse.coo_matrix(np.ones((T,1))))                          
 
    c = cvx.Variable(T) # calcium at each time step
    constraints=[]
    cnt = 0
    if b is None:
        flag_b = True
        cnt += 1
        b =  cvx.Variable(1) # baseline value
        if bas_nonneg:
            b_lb = 0
        else:
            b_lb = np.min(fluor)            
        constraints.append(b >= b_lb)
    else:
        flag_b = False

    if c1 is None:
        flag_c1 = True
        cnt += 1
        c1 =  cvx.Variable(1) # baseline value
        constraints.append(c1 >= 0)
    else:
        flag_c1 = False    
    
    thrNoise=sn * np.sqrt(fluor.size)
    
    try:
        objective=cvx.Minimize(cvx.norm(G*c,1)) # minimize number of spikes
        constraints.append(G*c >= 0)
        constraints.append(cvx.norm(-c + fluor - b - gd_vec*c1, 2) <= thrNoise) # constraints
        prob = cvx.Problem(objective, constraints) 
        result = prob.solve(solver=solvers[0])    
        
        if  not (prob.status ==  'optimal' or prob.status == 'optimal_inaccurate'):
            raise ValueError('Problem solved suboptimally or unfeasible')            
        
        print 'PROBLEM STATUS:' + prob.status 
        sys.stdout.flush()
    except (ValueError,cvx.SolverError) as err:     # if solvers fail to solve the problem           
         print(err) 
         sys.stdout.flush()
         lam=sn/500;
         constraints=constraints[:-1]
         objective = cvx.Minimize(cvx.norm(-c + fluor - b - gd_vec*c1, 2)+lam*cvx.norm(G*c,1))
         prob = cvx.Problem(objective, constraints)
         try: #in case scs was not installed properly
             try:
                 print('TRYING AGAIN ECOS') 
                 sys.stdout.flush()
                 result = prob.solve(solver=solvers[0]) 
             except:
                 print(solvers[0] + ' DID NOT WORK TRYING ' + solvers[1])
                 result = prob.solve(solver=solvers[1]) 
         except:             
             sys.stderr.write('***** SCS solver failed, try installing and compiling SCS for much faster performance. Otherwise set the solvers in tempora_params to ["ECOS","CVXOPT"]')
             sys.stderr.flush()
             #result = prob.solve(solver='CVXOPT')
             raise
             
         if not (prob.status ==  'optimal' or prob.status == 'optimal_inaccurate'):
            print 'PROBLEM STATUS:' + prob.status 
            raise Exception('Problem could not be solved')
            
    
    
    sp = np.squeeze(np.asarray(G*c.value))    
    c = np.squeeze(np.asarray(c.value))                
    if flag_b:    
        b = np.squeeze(b.value)        
    if flag_c1:    
        c1 = np.squeeze(c1.value)
        
    return c,b,c1,g,sn,sp
    
    
    
    
def estimate_parameters(fluor, p = 2, sn = None, g = None, range_ff = [0.25,0.5], method = 'logmexp', lags = 5, fudge_factor = 1):
    """
    Estimate noise standard deviation and AR coefficients if they are not present
    p: positive integer
        order of AR system  
    sn: float
        noise standard deviation, estimated if not provided.
    lags: positive integer
        number of additional lags where he autocovariance is computed
    range_ff : (1,2) array, nonnegative, max value <= 0.5
        range of frequency (x Nyquist rate) over which the spectrum is averaged  
    method: string
        method of averaging: Mean, median, exponentiated mean of logvalues (default)
    fudge_factor: float (0< fudge_factor <= 1)
        shrinkage factor to reduce bias
    """
    
    if sn is None:
        sn = GetSn(fluor,range_ff,method)
        
    if g is None:
        if p == 0:
            g = np.array(0)
        else:
            g = estimate_time_constant(fluor,p,sn,lags,fudge_factor)

    return g,sn

def estimate_time_constant(fluor, p = 2, sn = None, lags = 5, fudge_factor = 1.):
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
    gr = np.roots(np.concatenate([np.array([1]),-g.flatten()]))
    gr = (gr+gr.conjugate())/2.
    gr[gr>1] = 0.95 + np.random.normal(0,0.01,np.sum(gr>1))
    gr[gr<0] = 0.15 + np.random.normal(0,0.01,np.sum(gr<0))
    g = np.poly(fudge_factor*gr)
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
    
