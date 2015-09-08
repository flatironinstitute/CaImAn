# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 10:11:26 2015

@author: agiovann
"""
from scipy.sparse import spdiags,coo_matrix
import scipy
import numpy as np
import cPickle as pickle
from constrained_foopsi_AG import constrained_foopsi
import random
from scipy import linalg
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
def update_temporal_components(Y,A,b,Cin,fin,ITER=1,restimate_g=True,method='constrained_foopsi',g='None',**kwargs):
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
    #%
    if g is None:
        raise Exception('Add recompute g')

    d,T = np.shape(Y);
    
    flag_G = True
    if type(g) is not list:
        g=np.squeeze(g)
        flag_G = False
        G = make_G_matrix(T,np.array(g))
    
    #%
    nr = np.shape(A)[1]
    A = scipy.sparse.hstack((A,coo_matrix(b)))
    Cin =  np.vstack((Cin,fin));
    C = Cin;
    #%
    nA = np.squeeze(np.array(np.sum(np.square(A.todense()),axis=0)))
    
    Y=np.matrix(Y)
    Cin=np.matrix(Cin)
    
    YrA = Y.T*A - Cin.T*(A.T*A);

    #%
#    if method=='constrained_foopsi':
        #{'gn':None,'b':None,'cl':None,'neurons_sn':None,'fudge_factor','p':None}    
    #    P.gn = cell(nr,1);
    #    P.b = cell(nr,1);
    #    P.c1 = cell(nr,1);           
    #    P.neuron_sn = cell(nr,1);
       
            
    #    if isfield(P,'p'); options.p = P.p; else options.p = length(P.g); end
    #    if isfield(P,'fudge_factor'); options.fudge_factor = P.fudge_factor; end
    
    for iter in range(ITER):
        idxs=range(nr+1)
        random.shuffle(idxs)
        P_=[];
    #    perm = randperm(nr+1)
        for jj,ii in enumerate(idxs):
            
            ii=jj
            print ii,jj
            pars=dict(kwargs)
    #        ii = perm(jj);
            if ii<=nr:
                if flag_G:
                    if type(g) is list:
                        G = make_G_matrix(T,g[ii])
                    else:
                        raise Exception('Argument should be a list')
                
                if method == 'project':
                        YrA[:,ii] = YrA[:,ii] + nA[ii]*Cin[ii,:].T;
                        maxy = np.max(YrA[:,ii]/nA[ii])
                        raise Exception('Not Implemented')
    #                    cc = plain_foopsi(YrA(:,ii)/nA(ii)/maxy,G);
    #                    C(ii,:) = full(cc')*maxy;
    #                    YrA(:,ii) = YrA(:,ii) - nA(ii)*C(ii,:)';
                elif method == 'constrained_foopsi':
                        YrA[:,ii] = YrA[:,ii] + nA[ii]*Cin[ii,:].T                  
                        if restimate_g:
                            pars['g']=None
                        else:
                            pars['g']=g
                        
                        
                        cc,cb,c1,gn,sn,_ = constrained_foopsi(np.squeeze(np.asarray(YrA[:,ii]/nA[ii])),**pars)
                        print pars
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
                YrA[:,ii] = YrA[:,ii] + nA[ii]*Cin[ii,:].T
                cc = np.maximum(YrA[:,ii]/nA[ii],0)
                #C[ii,:] = full(cc');
                YrA[:,ii] = YrA[:,ii] - nA[ii]*C[ii,:].T
            
            if jj%10 == 0:
                print str(jj) + ' out of total ' + str(nr) + ' temporal components updated \n'
    
    
        #%disp(norm(Fin(1:nr,:) - F,'fro')/norm(F,'fro'));
        if scipy.linalg.norm(Cin - C,'fro')/scipy.linalg.norm(C,'fro') <= 1e-3:
            # stop if the overall temporal component does not change by much
            break
        else:
            Cin = C
        
    
    
    Y_res = Y - A*C
    
    f = C[nr:,:]
    C = C[:nr,:]
    
    return C,f,Y_res,P_
#%%

#%%
