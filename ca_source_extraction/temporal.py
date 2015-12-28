# -*- coding: utf-8 -*-
"""A set of routines for estimating the temporal components, given the spatial components and temporal components

@author: agiovann
"""
from scipy.sparse import spdiags,coo_matrix,csgraph
import scipy
import numpy as np
import cPickle as pickle
from deconvolution import constrained_foopsi
import random
from scipy import linalg
from spatial import update_spatial_components
#%%
def make_G_matrix(T,g):
    ''' create matrix of autoregression to enforce indicator dynamics
    Inputs: 
    T: positive integer
        number of time-bins
    g: nd.array, vector p x 1
        Discrete time constants
        
    Output:
    G: sparse diagonal matrix
        Matrix of autoregression
    '''    
    if type(g) is np.ndarray:    
        if len(g) == 1 and g < 0:
            g=0
    
#        gs=np.matrix(np.hstack((-np.flipud(g[:]).T,1)))
        gs=np.matrix(np.hstack((1,-(g[:]).T)))
        ones_=np.matrix(np.ones((T,1)))
        G = spdiags((ones_*gs).T,range(0,-len(g)-1,-1),T,T)    
        
        return G
    else:
        raise Exception('g must be an array')
#%%
def update_temporal_components(Y,A,b,Cin,fin,ITER=2,method='constrained_foopsi',deconv_method = 'cvx', g='None',**kwargs):
    """update temporal components and background given spatial components using a block coordinate descent approach
    Inputs:
    Y: np.ndarray (2D)
        input data with time in the last axis (d x T)
    A: sparse matrix (crc format)
        matrix of temporal components (d x K)
    Cin: np.ndarray
        current estimate of temporal components (K x T)
    ITER: positive integer
        Maximum number of block coordinate descent loops. Default: 2
    fin: np.ndarray
        current estimate of temporal background (vector of length T)
    method: string
        Method of deconvolution of neural activity. 
        Default: constrained_foopsi (constrained deconvolution, the only method supported at the moment)
    deconv_method: string
        Solver for constrained foopsi ('cvx' or 'spgl1', default: 'cvx')
    g:  np.ndarray
        Global time constant (not used)
    **kwargs: all parameters passed to constrained_foopsi
                               b=None, 
                               c1=None,
                               g=None,
                               sn=None, 
                               p=2, 
                               method='cvx', 
                               bas_nonneg=True, 
                               noise_range=[0.25, 0.5], 
                               noise_method='logmexp', 
                               lags=5, 
                               resparse=0, 
                               fudge_factor=1, 
                               verbosity=False):
    """


    d,T = np.shape(Y);
    

    nr = np.shape(A)[-1]
    A = scipy.sparse.hstack((A,coo_matrix(b)))
    Cin =  np.vstack((Cin,fin));
    C = Cin;
    #%
    nA = np.squeeze(np.array(np.sum(np.square(A.todense()),axis=0)))
    
    Y=np.matrix(Y)
    C=np.matrix(C)
    Cin=np.matrix(Cin)
    Sp = np.zeros((nr,T))
    YrA = Y.T*A - Cin.T*(A.T*A);


    for iter in range(ITER):
        idxs=range(nr+1)
        random.shuffle(idxs)
        P_=[];
    #    perm = randperm(nr+1)
        for jj,ii in enumerate(idxs):            
            #ii=jj
            #print ii,jj
            pars=dict(kwargs)
    #        ii = perm(jj);
            if ii<nr:                
                if method == 'constrained_foopsi':
                        #print YrA.shape 
                        #print YrA.shape
                        YrA[:,ii] = YrA[:,ii] + nA[ii]*Cin[ii,:].T                  
                        cc,cb,c1,gn,sn,sp = constrained_foopsi(np.squeeze(np.asarray(YrA[:,ii]/nA[ii])), method = deconv_method, **pars)
                        #print pars
                        pars['gn'] = gn
                        
                        gd = np.max(np.roots(np.hstack((1,-gn.T))));  # decay time constant for initial concentration
                        gd_vec = gd**range(T)
                        
                        C[ii,:] = cc[:].T + cb + c1*gd_vec
                        Sp[ii,:] = sp[:T].T
                        YrA[:,ii] = YrA[:,ii] - np.matrix(nA[ii]*C[ii,:]).T
                        pars['b'] = cb
                        pars['c1'] = c1           
                        pars['neuron_sn'] = sn
                        pars['neuron_id'] = ii
                        P_.append(pars)
                else:
                        raise Exception('undefined  method')                        
                    
                
            else:
                YrA[:,ii] = YrA[:,ii] + nA[ii]*Cin[ii,:].T
                cc = np.maximum(YrA[:,ii]/nA[ii],0)
                #C[ii,:] = full(cc');
                YrA[:,ii] = YrA[:,ii] - nA[ii]*C[ii,:].T
            
            if (jj+1)%10 == 0:
                print str(jj+1) + ' out of total ' + str(nr+1) + ' temporal components updated \n'
    
    
        #%disp(norm(Fin(1:nr,:) - F,'fro')/norm(F,'fro'));
        if scipy.linalg.norm(Cin - C,'fro')/scipy.linalg.norm(C,'fro') <= 1e-3:
            # stop if the overall temporal component does not change by much
            break
        else:
            Cin = C
        
    
    
    Y_res = Y - A*C
    
    f = C[nr:,:]
    C = C[:nr,:]
        
    P_ = sorted(P_, key=lambda k: k['neuron_id']) 
    
    return C,f,Y_res,P_,Sp

#%%
