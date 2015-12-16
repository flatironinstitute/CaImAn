# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 10:11:26 2015

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
    ''' 
    create matrix of autoregression
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
def update_temporal_components(Y,A,b,Cin,fin,ITER=1,method='constrained_foopsi',deconv_method = 'cvx', g='None',**kwargs):
#                               b=None, 
#                               c1=None,
#                               g=None,
#                               sn=None, 
#                               p=2, 
#                               method='cvx', 
#                               bas_nonneg=True, 
#                               noise_range=[0.25, 0.5], 
#                               noise_method='logmexp', 
#                               lags=5, 
#                               resparse=0, 
#                               fudge_factor=1, 
#                               verbosity=False):
#    """
#    update temporal components and background given spatial components
#    **kwargs: all parameters passed to constrained_foopsi
#    """


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
                        
                        cc,cb,c1,gn,sn,_ = constrained_foopsi(np.squeeze(np.asarray(YrA[:,ii]/nA[ii])), method = deconv_method, **pars)
                        #print pars
                        pars['gn'] = gn
                        
                        gd = np.max(np.roots(np.hstack((1,-gn.T))));  # decay time constant for initial concentration
                        gd_vec = gd**range(T)
                        
                        C[ii,:] = cc[:].T + cb + c1*gd_vec
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
            
            if jj%10 == 0:
                print str(jj) + ' out of total ' + str(nr+1) + ' temporal components updated \n'
    
    
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
    
    return C,f,Y_res,P_

#%%
