# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 16:23:57 2015

@author: agiovann
"""
from scipy.sparse import spdiags,coo_matrix,csgraph,csr_matrix,csc_matrix
import scipy
import numpy as np
import cPickle as pickle
from constrained_foopsi import constrained_foopsi
import random
from scipy import linalg
from update_spatial_components import update_spatial_components
from update_temporal_components import update_temporal_components
import warnings
#%%
def mergeROIS(Y_res,A,b,C,f,d1,d2,P_,thr=0.8,mx=50,sn=None,deconv_method='spgl1',min_size=3,max_size=8,dist=3):
    """
    merging of spatially overlapping components that have highly correlated tmeporal activity
    % The correlation threshold for merging overlapping components is user specified in P.merge_thr (default value 0.85)
    % Inputs:
    % Y_res:        residual movie after subtracting all found components
    % A:            matrix of spatial components
    % b:            spatial background
    % C:            matrix of temporal components
    % f:            temporal background
    % P:            parameter struct
    
    % Outputs:
    % A:            matrix of new spatial components
    % C:            matrix of new temporal components
    % nr:           new number of components
    % merged_ROIs:  list of old components that were merged
    
    % Written by:
    % Andrea Giovannucci from implementation of Eftychios A. Pnevmatikakis, Simons Foundation, 2015
    """
    
#%
    
    nr = A.shape[1]
    [d,T] = np.shape(Y_res)
    C_corr = np.corrcoef(C[:nr,:],C[:nr,:])[:nr,:nr];
    FF1=C_corr>=thr; #find graph of strongly correlated temporal components 
    A_corr=A.T*A
    A_corr.setdiag(0)
    FF2=A_corr>0            # % find graph of overlapping spatial components
    FF3=np.logical_and(FF1,FF2.todense())
    FF3=coo_matrix(FF3)
    c,l=csgraph.connected_components(FF3) # % extract connected components
    
    p=len(P_[0]['gn'])
    MC=[];
    for i in range(c):     
        if np.sum(l==i)>1:
            MC.append((l==i).T)
    MC=np.asarray(MC).T
    
    if MC.ndim>1:
        cor = np.zeros((np.shape(MC)[1],1));
        
            
        for i in range(np.size(cor)):
            fm = np.where(MC[:,i])[0]
            for j1 in range(np.size(fm)):        
                for j2 in range(j1+1,np.size(fm)):
                    print j1,j2
                    cor[i] = cor[i] +C_corr[fm[j1],fm[j2]]
        
        
        Y_res = Y_res + np.dot(b,f);
        if np.size(cor) > 1:
            ind=np.argsort(np.squeeze(cor))[::-1]
        else:
            ind = [0]
    
        nm = min((np.size(ind),mx))   # number of merging operations
    
        A_merged = coo_matrix((d,nm)).tocsr();
        C_merged = np.zeros((nm,T));
        
        P_merged=[];
        merged_ROIs = []
    #%
        for i in range(nm):
            P_cycle=dict()
            merged_ROI=np.where(MC[:,ind[i]])[0]
            merged_ROIs.append(merged_ROI)
            nC = np.sqrt(np.sum(C[merged_ROI,:]**2,axis=1))
    #        A_merged[:,i] = np.squeeze((A[:,merged_ROI]*spdiags(nC,0,len(nC),len(nC))).sum(axis=1))    
            A_merged[:,i] = csr_matrix((A[:,merged_ROI]*spdiags(nC,0,len(nC),len(nC))).sum(axis=1))
    
            Y_res = Y_res + A[:,merged_ROI]*C[merged_ROI,:]
            
            aa_1=scipy.sparse.linalg.spsolve(spdiags(nC,0,len(nC),len(nC)),C[merged_ROI,:])
            aa_2=(aa_1).mean(axis=0)        
            
            ff = np.nonzero(A_merged[:,i])[0]     
            
            cc,_,_,Ptemp = update_temporal_components(np.asarray(Y_res[ff,:]),A_merged[ff,i],b[ff],aa_2,f,p=p,deconv_method=deconv_method)  
            
            aa,bb,cc = update_spatial_components(np.asarray(Y_res),cc,f,A_merged[:,i],d1=d1,d2=d2,sn=sn,min_size=min_size,max_size=max_size,dist=dist)
    
            A_merged[:,i] = aa.tocsr();        
    
            cc,_,_,Ptemp = update_temporal_components(Y_res[ff,:],A_merged[ff,i],bb[ff],cc,f,p=p,deconv_method=deconv_method)
            
            P_cycle=P_[merged_ROI[0]].copy()
            P_cycle['gn']=Ptemp[0]['gn']
            P_cycle['b']=Ptemp[0]['b']
            P_cycle['c1']=Ptemp[0]['c1']
            P_cycle['neuron_sn']=Ptemp[0]['neuron_sn']
            P_merged.append(P_cycle)
            C_merged[i,:] = cc
            if i+1 < nm:
                Y_res[ff,:] = Y_res[ff,:] - A_merged[ff,i]*cc
                
        #%
        neur_id = np.unique(np.hstack(merged_ROIs))
        
    
    
        good_neurons=np.setdiff1d(range(nr),neur_id)    
        
        A= scipy.sparse.hstack((A[:,good_neurons],A_merged.tocsc()))
        C = np.vstack((C[good_neurons,:],C_merged))
        
    #    P_new=list(P_[good_neurons].copy())
        P_new=[P_[pp] for pp in good_neurons]
        
        for p in P_merged:
            P_new.append(p)
    
        nr = nr - len(neur_id) + nm
    
    else:
        warnings.warn('No neurons merged!')
        merged_ROIs=[];
        P_new=P_
        
    return A,C,nr,merged_ROIs,P_new