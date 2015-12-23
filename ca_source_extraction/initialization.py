# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 11:56:06 2015

@author: epnevmatikakis
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
import scipy.ndimage as nd
import scipy.sparse as spr
import scipy
import numpy as np
from scipy.fftpack import fft, ifft
from skimage.transform import resize
from ca_source_extraction.utilities import com
#%%
def initialize_components(Y, K=30, gSig=[5,5], gSiz=None, ssub=1, tsub=1, nIter = 5, use_median = False, kernel = None): 
    """
    Initalize components using a greedy approach followed by hierarchical
    alternative least squares (HALS) NMF. Optional use of spatio-temporal
    downsampling to boost speed.
    
    Input:
    -------------------------------------------------------
    Y          d1 x d2 x T movie, raw data
    K          number of neurons to extract (default value: 30)
    tau        standard deviation of neuron size along x and y (default value: (5,5)
    
    fine-tuning parameters (optional)
    init_method: method of initialization ('greedy','sparse_NMF','both')
    nIter: number of iterations for shape tuning (default 5)
    gSiz: size of kernel (default 2*tau + 1)
    ssub: spatial downsampling factor (default 1)
    tsub: temporal downsampling factor (default 1)
            
  
    Output:
    -------------------------------------------------------
    Ain        (d1*d2) x K matrix, location of each neuron
    Cin        T x K matrix, calcium activity of each neuron
    center     K x 2 matrix, inferred center of each neuron
    bin        (d1*d2) X nb matrix, initialization of spatial background
    fin        nb X T matrix, initalization of temporal background

      
    Authors: Andrea Giovannucci, Eftychios A. Pnevmatikakis and Pengchen Zhou
    """
    if gSiz is None:
        gSiz=(2*gSig[0] + 1,2*gSig[1] + 1)
     
    d1,d2,T=np.shape(Y) 
    # rescale according to downsampling factor
    gSig = np.round([gSig[0]/ssub,gSig[1]/ssub]).astype(np.int)
    gSiz = np.round([gSiz[0]/ssub,gSiz[1]/ssub]).astype(np.int)    
    d1s = np.ceil(d1/ssub).astype(np.int)       #size of downsampled image
    d2s = np.ceil(d2/ssub).astype(np.int)     
    Ts = np.floor(T/tsub).astype(np.int)        #reduced number of frames
    
    # spatial downsampling
    if ssub!=1 or tsub!=1:
        print 'Spatial Downsampling...'
        Y_ds = resize(Y, [d1s, d2s, Ts], order=0,clip=False,mode='nearest')
    else:
        Y_ds = Y
        
    print 'Roi Extraction...'    
    Ain, Cin, _, b_in, f_in = greedyROI2d(Y_ds, nr = K, gSig = gSig, gSiz = gSiz, use_median = use_median, nIter=nIter, kernel = kernel)
    print 'Refining Components...'    
    Ain, Cin, b_in, f_in = hals_2D(Y_ds, Ain, Cin, b_in, f_in,maxIter=10);
    
    #center = ssub*com(Ain,d1s,d2s) 
    Ain = np.reshape(Ain, (d1s, d2s,K),order='F')
    Ain = resize(Ain, [d1, d2],order=0)
    
    Ain = np.reshape(Ain, (d1*d2, K),order='F') 
    
    b_in=np.reshape(b_in,(d1s, d2s),order='F')
    
    b_in = resize(b_in, [d1, d2]);
    
    b_in = np.reshape(b_in, (d1*d2, 1),order='F')
    
    Cin = resize(Cin, [K, T])
    f_in = resize(f_in, [1, T])    
    center = com(Ain,d1,d2)
    
    return Ain, Cin, b_in, f_in, center
    
    
#%%
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
    
    return 

#%%
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
    

#%%
def get_noise_fft(Y, noise_range = [0.25,0.5], noise_method = 'logmexp'):
    
    T = np.shape(Y)[-1]
    dims = len(np.shape(Y))
    ff = np.arange(0,0.5+1./T,1./T)
    ind1 = ff > noise_range[0]
    ind2 = ff <= noise_range[1]
    ind = np.logical_and(ind1,ind2)
    if dims > 1:
        sn = 0
        xdft = fft(Y,axis=-1)
        xdft = xdft[...,:T/2+1]
        psdx = (1./T)*np.abs(xdft)**2
        psdx[...,1:] *= 2
        sn = mean_psd(psdx[...,ind], method = noise_method)
        
    else:
        xdft = fft(Y)
        xdft = xdft[:T/2+1]
        psdx = (1./T)*np.abs(xdft)**2
        psdx[1:] *=2
        sn = mean_psd(psdx[ind], method = noise_method)
    
    return sn
    

#%%
def mean_psd(y, method = 'logmexp'):
    
    if method == 'mean':
        mp = np.sqrt(np.mean(y/2,axis=-1))
    elif method == 'median':
        mp = np.sqrt(np.median(y/2,axis=-1))
    else:
        mp = np.sqrt(np.exp(np.mean(np.log(y/2),axis=-1)))
        
    return mp


#%%
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


#%%
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


#%%
def greedyROI2d(Y, nr=30, gSig = [5,5], gSiz = [11,11], nIter = 5, use_median = False, kernel = None):
    """
    Greedy initialization of spatial and temporal components using spatial Gaussian filtering
    Inputs:
    Y: np.array
        3d array of fluorescence data with time appearing in the last axis.
        Support for 3d imaging will be added in the future
    nr: int
        number of components to be found
    gSig: scalar or list of integers
        standard deviation of Gaussian kernel along each axis
    gSiz: scalar or list of integers
        size of spatial component
    nIter: int
        number of iterations when refining estimates
    use_median: boolean
        add back fluorescence median values or not during refinement
        
    Outputs:
    A: np.array
        2d array of size (# of pixels) x nr with the spatial components. Each column is
        ordered columnwise (matlab format, order='F')
    C: np.array
        2d array of size nr X T with the temporal components
    center: np.array
        2d array of size nr x 2 with the components centroids
        
    Author: Eftychios A. Pnevmatikakis based on a matlab implementation by Yuanjun Gao
            Simons Foundation, 2015
    """
    d = np.shape(Y)
    med = np.median(Y,axis = -1)
    Y = Y - med[...,np.newaxis]
    gHalf = np.array(gSiz)/2
    gSiz = 2*gHalf + 1
    
    A = np.zeros((np.prod(d[0:-1]),nr))    
    C = np.zeros((nr,d[-1]))    
    center = np.zeros((nr,2))
    
    [M, N, T] = np.shape(Y)

    rho = imblur(Y, sig = gSig, siz = gSiz, nDimBlur = Y.ndim-1, kernel = kernel)
    v = np.sum(rho**2,axis=-1)
    
    if use_median:
        fact = 1
    else:
        fact = 0
    
    for k in range(nr):
        ind = np.argmax(v)
        ij = np.unravel_index(ind,d[0:-1])
        center[k,0] = ij[0]
        center[k,1] = ij[1]
        iSig = [np.maximum(ij[0]-gHalf[0],0),np.minimum(ij[0]+gHalf[0]+1,d[0])]
        jSig = [np.maximum(ij[1]-gHalf[1],0),np.minimum(ij[1]+gHalf[1]+1,d[1])]
        dataTemp = Y[iSig[0]:iSig[1],jSig[0]:jSig[1],:].copy() + fact*med[iSig[0]:iSig[1],jSig[0]:jSig[1],np.newaxis]
        traceTemp = np.squeeze(rho[ij[0],ij[1],:])
        coef, score = finetune2d(dataTemp, traceTemp, nIter = nIter)
        C[k,:] = np.squeeze(score) 
        score -= fact*np.median(score)
        dataSig = coef[...,np.newaxis]*score[np.newaxis,np.newaxis,...]
        [xSig,ySig] = np.meshgrid(np.arange(iSig[0],iSig[1]),np.arange(jSig[0],jSig[1]),indexing = 'xy')
        arr = np.array([np.reshape(xSig,(1,np.size(xSig)),order='F').squeeze(),np.reshape(ySig,(1,np.size(ySig)),order='F').squeeze()])
        indeces = np.ravel_multi_index(arr,d[0:-1],order='F')  
        A[indeces,k] = np.reshape(coef,(1,np.size(coef)),order='C').squeeze()
        #Atemp = np.zeros(d[0:-1])
        #Atemp[iSig[0]:iSig[1],jSig[0]:jSig[1]] = coef        
        #A[:,k] = np.squeeze(np.reshape(Atemp,(np.prod(d[0:-1]),1),order='F'))
        #Y[iSig[0]:iSig[1],jSig[0]:jSig[1],:] = Y[iSig[0]:iSig[1],jSig[0]:jSig[1],:] - dataSig
        Y[iSig[0]:iSig[1],jSig[0]:jSig[1],:] -= dataSig
        if k < nr-1:
            iMod = [np.maximum(ij[0]-2*gHalf[0],0),np.minimum(ij[0]+2*gHalf[0]+1,d[0])]
            iModLen = iMod[1]-iMod[0]
            jMod = [np.maximum(ij[1]-2*gHalf[1],0),np.minimum(ij[1]+2*gHalf[1]+1,d[1])]
            jModLen = jMod[1]-jMod[0]
            iLag = iSig - iMod[0] + 0
            jLag = jSig - jMod[0] + 0
            dataTemp = np.zeros((iModLen,jModLen))
            dataTemp[iLag[0]:iLag[1],jLag[0]:jLag[1]] = coef
            dataTemp = imblur(dataTemp[...,np.newaxis],sig=gSig,siz=gSiz,kernel = kernel)            
            rhoTEMP = dataTemp*score[np.newaxis,np.newaxis,...]
            #rho[iMod[0]:iMod[1],jMod[0]:jMod[1],:] = rho[iMod[0]:iMod[1],jMod[0]:jMod[1],:] - rhoTEMP
            rho[iMod[0]:iMod[1],jMod[0]:jMod[1],:] -= rhoTEMP
            v[iMod[0]:iMod[1],jMod[0]:jMod[1]] = np.sum(rho[iMod[0]:iMod[1],jMod[0]:jMod[1],:]**2,axis = -1)            

    res = np.reshape(Y,(M*N,T), order='F') + med.flatten()[:,None]
    model = NMF(n_components=1, init='random', random_state=0)
    
    b_in = model.fit_transform(np.maximum(res,0));
    f_in = model.components_.squeeze();    
   
#    f_in = model.components_.squeeze()
#    W = nmf_model.fit_transform(A);
#    H = nmf_model.components_;
    #[b_in,f_in] = NMF(n_components=1) 
    #np.maximum(res,0)      
    return A, C, center, b_in, f_in

#%%
def finetune2d(Y, cin, nIter = 5):

    for iter in range(nIter):
        a = np.maximum(np.dot(Y,cin),0)
        a = a/np.sqrt(np.sum(a**2))
        c = np.sum(Y*a[...,np.newaxis],tuple(np.arange(Y.ndim-1)))
    
    return a, c

#%%    
def imblur(Y, sig = 5, siz = 11, nDimBlur = None, kernel = None):
     
    from scipy.ndimage.filters import correlate        
    #from scipy.signal import correlate    
    
    X = np.zeros(np.shape(Y))
    
    if kernel is None:
        if nDimBlur is None:
            nDimBlur = Y.ndim - 1
        else:
            nDimBlur = np.min((Y.ndim,nDimBlur))
            
        if np.isscalar(sig):
            sig = sig*np.ones(nDimBlur)
            
        if np.isscalar(siz):
            siz = siz*np.ones(nDimBlur)
        
        xx = np.arange(-np.floor(siz[0]/2),np.floor(siz[0]/2)+1)
        yy = np.arange(-np.floor(siz[1]/2),np.floor(siz[1]/2)+1)
    
        hx = np.exp(-xx**2/(2*sig[0]**2))
        hx /= np.sqrt(np.sum(hx**2))
        
        hy = np.exp(-yy**2/(2*sig[1]**2))
        hy /= np.sqrt(np.sum(hy**2))    
    
        for t in range(np.shape(Y)[-1]):
            temp = correlate(Y[:,:,t],hx[:,np.newaxis],mode='wrap')
            X[:,:,t] = correlate(temp,hy[np.newaxis,:],mode='wrap')    
    else:
        for t in range(np.shape(Y)[-1]):
            X[:,:,t] = correlate(Y[:,:,t],kernel,mode='wrap')

## uncomment the following for general n-dim filtering
#    xx = []
#    hx = []
#    for i in range(nDimBlur):
#        vec = np.arange(-np.floor(siz[i]/2),np.floor(siz[i]/2)+1)
#        xx.append(vec)
#        fil = np.exp(-xx[i]**2/(2*sig[0]**2))
#        hx.append(fil/np.sqrt(np.sum(fil**2)))        
           
#    X = np.zeros(np.shape(Y))
#    siz = tuple([1]*nDimBlur)
#    sizY = np.shape(Y)
#    for t in range(sizY[-1]):
#        temp = Y[...,t]
#        for i in range(nDimBlur):
#            I = [0]*nDimBlur
#            I[i] = range(sizY[i])
#            siz[i] = sizY[i]
#            H = np.zeros(siz)
#            H[tuple(I)] = hx[i]
#            temp = correlate(temp,H,mode='wrap')
#        
#        X[...,t] = temp
            

    return X


#%%
def hals_2D(Y,A,C,b,f,bSiz=3,maxIter=5):
    """ Hierarchical alternating least square method for solving NMF problem  
    Y = A*C + b*f 
    
    input: 
       Y:      d1 X d2 X T, raw data. It will be reshaped to (d1*d2) X T in this
       function 
       A:      (d1*d2) X K, initial value of spatial components 
       C:      K X T, initial value of temporal components 
       b:      (d1*d2) X 1, initial value of background spatial component 
       f:      1 X T, initial value of background temporal component
    
       bSiz:   blur size. A box kernel (bSiz X bSiz) will be convolved
       with each neuron's initial spatial component, then all nonzero
       pixels will be picked as pixels to be updated, and the rest will be
       forced to be 0. 
       maxIter: maximum iteration of iterating HALS. 
    
    output:
    the updated A, C, b, f
    
    Author: Andrea Giovannucci, Columbia University, based on a matlab implementation from Pengcheng Zhou
       """
    
    #%% smooth the components
    d1, d2, T = np.shape(Y)
    ind_A = nd.filters.uniform_filter(np.reshape(A,(d1,d2, A.shape[1]),order='F'),size=(bSiz,bSiz,0))
    ind_A = np.reshape(ind_A>1e-10,(d1*d2, A.shape[1]),order='F')
    
    #%%
    ind_A = spr.csc_matrix(ind_A);  #indicator of nonnero pixels 
    K = np.shape(A)[1] #number of neurons 
    #%% update spatial and temporal components neuron by neurons
    Yres = np.reshape(Y, (d1*d2, T),order='F') - np.dot(A,C) - np.dot(b,f[None,:]);
    #print 'First residual is ' + str(scipy.linalg.norm(Yres, 'fro')) + '\n';
     
    for miter in range(maxIter):
        for mcell in range(K):
            ind_pixels = np.squeeze(np.asarray(ind_A[:, mcell].todense()))
            tmp_Yres = Yres[ind_pixels, :]
           # print 'First residual is ' + str(scipy.linalg.norm(tmp_Yres, 'fro')) + '\n';
            # update temporal component of the neuron
            c0 = C[mcell, :].copy();
            a0 = A[ind_pixels, mcell].copy()
            norm_a2 = scipy.linalg.norm(a0, ord=2)**2;
            C[mcell, :] = np.maximum(0, c0 + np.dot(a0.T,tmp_Yres/norm_a2))
            tmp_Yres = tmp_Yres + np.dot(a0[:,None],(c0-C[mcell,:])[None,:])
           # print 'First residual is ' + str(scipy.linalg.norm(tmp_Yres, 'fro')) + '\n';
            
            # update spatial component of the neuron
            norm_c2 = scipy.linalg.norm(C[mcell,:],ord=2)**2
            tmp_a = np.maximum(0, a0 + np.dot(tmp_Yres,C[mcell, :].T/norm_c2))
            A[ind_pixels, mcell] = tmp_a
            Yres[ind_pixels,:] = tmp_Yres + np.dot(a0[:,None]-tmp_a[:,None],C[None,mcell,:])
            #print 'First residual is ' + str(scipy.linalg.norm(tmp_Yres, 'fro')) + '\n';
    
    #        print 'First residual is ' + str(scipy.linalg.norm(Yres, 'fro')) + '\n';
    
        # update temporal component of the background
        f0 = f.copy();
        b0 = b.copy();
        norm_b2 = scipy.linalg.norm(b0,ord=2)**2
        f = np.maximum(0, f0 + np.dot(b0.T,Yres/norm_b2))
        Yres = Yres + np.dot(b0,f0-f)
        #print 'First residual is ' + str(scipy.linalg.norm(Yres, 'fro')) + '\n';
        
        
        # update spatial component of the background
        norm_f2 = scipy.linalg.norm(f, ord=2)**2
        b = np.maximum(0, b0 + np.dot(Yres,f.T/norm_f2))
        Yres = Yres + np.dot(b0-b,f)
        
        print 'Iteration:' + str(miter) + ', the norm of residual is ' + str(scipy.linalg.norm(Yres, 'fro'))
    
    return A, C, b, f
