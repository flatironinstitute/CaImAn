# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 11:56:06 2015

@author: epnevmatikakis
"""

import numpy as np
import matplotlib.pyplot as plt

def greedyROI2d(Y, nr=30, gSig = [5,5], gSiz = [11,11], nIter = 5, use_median = False):
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

    rho = imblur(Y, sig = gSig, siz = gSiz, nDimBlur = Y.ndim-1)
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
            dataTemp = imblur(dataTemp[...,np.newaxis],sig=gSig,siz=gSiz)            
            rhoTEMP = dataTemp*score[np.newaxis,np.newaxis,...]
            #rho[iMod[0]:iMod[1],jMod[0]:jMod[1],:] = rho[iMod[0]:iMod[1],jMod[0]:jMod[1],:] - rhoTEMP
            rho[iMod[0]:iMod[1],jMod[0]:jMod[1],:] -= rhoTEMP
            v[iMod[0]:iMod[1],jMod[0]:jMod[1]] = np.sum(rho[iMod[0]:iMod[1],jMod[0]:jMod[1],:]**2,axis = -1)            

    return A, C, center

#%%
def finetune2d(Y, cin, nIter = 5):

    for iter in range(nIter):
        a = np.maximum(np.dot(Y,cin),0)
        a = a/np.sqrt(np.sum(a**2))
        c = np.sum(Y*a[...,np.newaxis],tuple(np.arange(Y.ndim-1)))
    
    return a, c
    
def imblur(Y, sig = 5, siz = 11, nDimBlur = None):
     
    from scipy.ndimage.filters import correlate        
    #from scipy.signal import correlate    
    
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

    X = np.zeros(np.shape(Y))
    for t in range(np.shape(Y)[-1]):
        temp = correlate(Y[:,:,t],hx[:,np.newaxis],mode='wrap')
        X[:,:,t] = correlate(temp,hy[np.newaxis,:],mode='wrap')    

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