# -*- coding: utf-8 -*-
"""Merging of spatially overlapping components that are temporally correlated
Created on Tue Sep  8 16:23:57 2015

@author: agiovann
"""
from scipy.sparse import coo_matrix,csgraph,csc_matrix, lil_matrix
import scipy
import numpy as np
from spatial import update_spatial_components
from temporal import update_temporal_components
from deconvolution import constrained_foopsi

#%%
def merge_components(Y,A,b,C,f,S,sn_pix,temporal_params,spatial_params,dview=None,thr=0.85,fast_merge=True,mx=1000,bl=None,c1=None,sn=None,g=None):
    """ Merging of spatially overlapping components that have highly correlated temporal activity
    The correlation threshold for merging overlapping components is user specified in thr
     
Parameters
-----------     

Y: np.ndarray
     residual movie after subtracting all found components (Y_res = Y - A*C - b*f) (d x T)
A: sparse matrix
     matrix of spatial components (d x K)
b: np.ndarray
     spatial background (vector of length d)
C: np.ndarray
     matrix of temporal components (K x T)
f:     np.ndarray
     temporal background (vector of length T)     
S:     np.ndarray            
     matrix of deconvolved activity (spikes) (K x T)
sn_pix: ndarray
     noise standard deviation for each pixel
temporal_params: dictionary 
     all the parameters that can be passed to the update_temporal_components function
spatial_params: dictionary 
     all the parameters that can be passed to the update_spatial_components function     
     
thr:   scalar between 0 and 1
     correlation threshold for merging (default 0.85)
mx:    int
     maximum number of merging operations (default 50)
sn_pix:    nd.array
     noise level for each pixel (vector of length d)
 
bl:        
     baseline for fluorescence trace for each row in C
c1:        
     initial concentration for each row in C
g:         
     discrete time constant for each row in C
sn:        
     noise level for each row in C

Returns
--------

A:     sparse matrix
        matrix of merged spatial components (d x K)
C:     np.ndarray
        matrix of merged temporal components (K x T)
nr:    int
    number of components after merging
merged_ROIs: list
    index of components that have been merged     
S:     np.ndarray            
        matrix of merged deconvolved activity (spikes) (K x T)
bl: float       
    baseline for fluorescence trace
c1: float       
    initial concentration
g:  float       
    discrete time constant
sn: float      
    noise level    
    """
    
#%
    
    nr = A.shape[1]
    if bl is not None and len(bl) != nr:
        raise Exception("The number of elements of bl must match the number of components")
    
    if c1 is not None and len(c1) != nr:
        raise Exception("The number of elements of c1 must match the number of components")
    
    if sn is not None and len(sn) != nr:
        raise Exception("The number of elements of bl must match the number of components")
    
    if g is not None and len(g) != nr:
        raise Exception("The number of elements of g must match the number of components")

    
    
    [d,T] = np.shape(Y)
#    C_corr = np.corrcoef(C[:nr,:],C[:nr,:])[:nr,:nr];
    C_corr = np.corrcoef(C)
    FF1=C_corr>=thr; #find graph of strongly correlated temporal components 
    A_corr=A.T*A
    A_corr.setdiag(0)
    FF2=A_corr>0            # % find graph of overlapping spatial components
    FF3=np.logical_and(FF1,FF2.todense())
    FF3=coo_matrix(FF3)
    c,l=csgraph.connected_components(FF3) # % extract connected components
    
    p=temporal_params['p']
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
                    cor[i] = cor[i] +C_corr[fm[j1],fm[j2]]
        
        if not fast_merge:
            Y_res = Y - A.dot(C)
            
        if np.size(cor) > 1:
            ind=np.argsort(np.squeeze(cor))[::-1]
        else:
            ind = [0]
    
        nm = min((np.size(ind),mx))   # number of merging operations
    
        A_merged = lil_matrix((d,nm));
        C_merged = np.zeros((nm,T));
        S_merged = np.zeros((nm,T));
        bl_merged=np.zeros((nm,1))
        c1_merged=np.zeros((nm,1))
        sn_merged=np.zeros((nm,1))
        g_merged=np.zeros((nm,p))
        
#        P_merged=[];
        merged_ROIs = []

        for i in range(nm):
#            P_cycle=dict()
            merged_ROI=np.where(MC[:,ind[i]])[0]
            merged_ROIs.append(merged_ROI)
            nC = np.sqrt(np.sum(np.array(C)[merged_ROI,:]**2,axis=1))
    #        A_merged[:,i] = np.squeeze((A[:,merged_ROI]*spdiags(nC,0,len(nC),len(nC))).sum(axis=1))    
            if fast_merge:
                Acsc = A.tocsc()[:,merged_ROI]
                Acsd = Acsc.toarray()
                Ctmp = np.array(C)[merged_ROI,:]
                print merged_ROI.T
                #aa  =  A.tocsc()[:,merged_ROI].dot(scipy.sparse.diags(nC,0,(len(nC),len(nC)))).sum(axis=1)
                aa  =  Acsc.dot(scipy.sparse.diags(nC,0,(len(nC),len(nC)))).sum(axis=1)
                for iter in range(10):
                    #cc = np.dot(aa.T.dot(A.toarray()[:,merged_ROI]),C[merged_ROI,:])/(aa.T*aa)
                    cc = np.dot(aa.T.dot(Acsd),Ctmp)/(aa.T*aa)
                    #aa = A.tocsc()[:,merged_ROI].dot(C[merged_ROI,:].dot(cc.T))/(cc*cc.T)
                    aa = Acsc.dot(Ctmp.dot(cc.T))/(cc*cc.T)
                
#                nC = np.sqrt(np.sum(A.toarray()[:,merged_ROI]**2,axis=0))*np.sqrt(np.sum(C[merged_ROI,:]**2,axis=1))
                nC = np.sqrt(np.sum(Acsd**2,axis=0))*np.sqrt(np.sum(Ctmp**2,axis=1))   
#                nA = sqrt( sum(aa.^2)/ max(sum(A(:,merged_ROIs{i}).^2)));

                nA = np.sqrt( np.sum(np.array(aa)**2) / A.tocsc()[:,merged_ROIs[i]].power(2).sum(0).max() )
#                nA = np.sqrt(np.sum(np.array(aa)**2)
                aa /= nA
                cc *= nA

                indx = np.argmax(nC)
                
                if g is not None:
                    cc,bm,cm,gm,sm,ss = constrained_foopsi(np.array(cc).squeeze(),g=g[merged_ROI[indx]],**temporal_params)
                else:
                    cc,bm,cm,gm,sm,ss = constrained_foopsi(np.array(cc).squeeze(),g=None,**temporal_params)
                    
                A_merged[:,i] = aa; 
                C_merged[i,:] = cc
                S_merged[i,:] = ss[:T]
                bl_merged[i] = bm
                c1_merged[i] = cm
                sn_merged[i] = sm
                g_merged[i,:] = gm 
            else:
                A_merged[:,i] = lil_matrix(( A.tocsc()[:,merged_ROI].dot(scipy.sparse.diags(nC,0,(len(nC),len(nC))))).sum(axis=1))        
                Y_res = Y_res + A.tocsc()[:,merged_ROI].dot(C[merged_ROI,:])                
                aa_1=scipy.sparse.linalg.spsolve(scipy.sparse.diags(nC,0,(len(nC),len(nC))),csc_matrix(C[merged_ROI,:]))
                aa_2=(aa_1).mean(axis=0)                        
                ff = np.nonzero(A_merged[:,i])[0]         
    #            cc,_,_,Ptemp,_ = update_temporal_components(np.asarray(Y_res[ff,:]),A_merged[ff,i],b[ff],aa_2,f,p=p,deconv_method=deconv_method)
                cc,_,_,_,bl__,c1__,sn__,g__,YrA = update_temporal_components(np.asarray(Y_res[ff,:]),A_merged[ff,i],b[ff],aa_2,f,dview=dview,bl=None,c1=None,sn=None,g=None,**temporal_params)                     
                aa,bb,cc = update_spatial_components(np.asarray(Y_res),cc,f,A_merged[:,i],sn=sn_pix,dview=dview,**spatial_params)
                A_merged[:,i] = aa.tocsr();                
                cc,_,_,ss,bl__,c1__,sn__,g__,YrA = update_temporal_components(Y_res[ff,:],A_merged[ff,i],bb[ff],cc,f,dview=dview,bl=bl__,c1=c1__,sn=sn__,g=g__,**temporal_params)                


                C_merged[i,:] = cc
                S_merged[i,:] = ss
                bl_merged[i] = bl__[0]
                c1_merged[i] = c1__[0]
                sn_merged[i] = sn__[0]
                g_merged[i,:] = g__[0]                
                if i+1 < nm:
                    Y_res[ff,:] = Y_res[ff,:] - A_merged[ff,i]*cc
                
        #%
        neur_id = np.unique(np.hstack(merged_ROIs))                
        good_neurons=np.setdiff1d(range(nr),neur_id)    

        
        A = scipy.sparse.hstack((A.tocsc()[:,good_neurons],A_merged.tocsc()))
        C = np.vstack((C[good_neurons,:],C_merged))
        if S is not None:
            S = np.vstack((S[good_neurons,:],S_merged))
        if bl is not None:    
            bl=np.hstack((bl[good_neurons],np.array(bl_merged).flatten()))
        if c1 is not None:            
            c1=np.hstack((c1[good_neurons],np.array(c1_merged).flatten()))
        if sn is not None:
            sn=np.hstack((sn[good_neurons],np.array(sn_merged).flatten()))        
        if g is not None:
            g=np.vstack((np.vstack(g)[good_neurons],g_merged))  
            

            
        
    #    P_new=list(P_[good_neurons].copy())
#        P_new=[P_[pp] for pp in good_neurons]
#        
#        for p in P_merged:
#            P_new.append(p)
#       
        nr = nr - len(neur_id) + nm
    
    else:
        print('********** No neurons merged! ***************')        
        merged_ROIs=[];        
        
    return A,C,nr,merged_ROIs,S,bl,c1,sn,g
    
