# -*- coding: utf-8 -*-
"""A set of routines for estimating the temporal components, given the spatial components and temporal components
@author: agiovann
"""
from __future__ import division
from __future__ import print_function
from builtins import str
from builtins import map
from builtins import range
from past.utils import old_div
from scipy.sparse import spdiags,coo_matrix#,csgraph
import scipy
import numpy as np
from .deconvolution import constrained_foopsi
from .utilities import update_order
import sys
import time
from caiman.mmapping import parallel_dot_product
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
        G = spdiags((ones_*gs).T,list(range(0,-len(g)-1,-1)),T,T)    

        return G
    else:
        raise Exception('g must be an array')
#%%
def constrained_foopsi_parallel(arg_in):
    """ necessary for parallel computation of the function  constrained_foopsi
    """  

    Ytemp, nT, jj_, bl, c1, g, sn, argss = arg_in
    T=np.shape(Ytemp)[0]
    #cc_,cb_,c1_,gn_,sn_,sp_ = constrained_foopsi(Ytemp/nT, bl = bl,  c1 = c1, g = g,  sn = sn, **argss)
    cc_,cb_,c1_,gn_,sn_,sp_ = constrained_foopsi(Ytemp, bl = bl,  c1 = c1, g = g,  sn = sn, **argss)
    gd_ = np.max(np.roots(np.hstack((1,-gn_.T))));  
    gd_vec = gd_**list(range(T))   


    C_ = cc_[:].T + cb_ + np.dot(c1_,gd_vec)
    Sp_ = sp_[:T].T
    #Ytemp_ = Ytemp - np.dot(nT,C_).T
    Ytemp_ = Ytemp - C_.T

    return C_,Sp_,Ytemp_,cb_,c1_,sn_,gn_,jj_


#%%
def update_temporal_components(Y, A, b, Cin, fin, bl = None,  c1 = None, g = None,  sn = None, nb = 1, ITER=2, method_foopsi='constrained_foopsi', block_size = 20000,  memory_efficient=False, debug=False, dview=None,**kwargs):
    """Update temporal components and background given spatial components using a block coordinate descent approach.

    Parameters
    -----------    

    Y: np.ndarray (2D)
        input data with time in the last axis (d x T)
    A: sparse matrix (crc format)
        matrix of temporal components (d x K)
    b: ndarray (dx1)
        current estimate of background component
    Cin: np.ndarray
        current estimate of temporal components (K x T)   
    fin: np.ndarray
        current estimate of temporal background (vector of length T)
    g:  np.ndarray
        Global time constant (not used)
    bl: np.ndarray
       baseline for fluorescence trace for each column in A
    c1: np.ndarray
       initial concentration for each column in A
    g:  np.ndarray       
       discrete time constant for each column in A
    sn: np.ndarray
       noise level for each column in A       
    nb: [optional] int
        Number of background components
    ITER: positive integer
        Maximum number of block coordinate descent loops. 
    method_foopsi: string
        Method of deconvolution of neural activity. constrained_foopsi is the only method supported at the moment.               
    n_processes: int
        number of processes to use for parallel computation. Should be less than the number of processes started with ipcluster.
    backend: 'str'
        single_thread no parallelization
        ipyparallel, parallelization using the ipyparallel cluster. You should start the cluster (install ipyparallel and then type 
        ipcluster -n 6, where 6 is the number of processes). 
        SLURM: using SLURM scheduler
    memory_efficient: Bool
        whether or not to optimize for memory usage (longer running times). nevessary with very large datasets  
    **kwargs: dict
        all parameters passed to constrained_foopsi except bl,c1,g,sn (see documentation). Some useful parameters are      
    p: int
        order of the autoregression model
    method: [optional] string
        solution method for constrained foopsi. Choices are
            'cvx':      using cvxopt and picos (slow especially without the MOSEK solver)
            'cvxpy':    using cvxopt and cvxpy with the ECOS solver (faster, default)

    solvers: list string
            primary and secondary (if problem unfeasible for approx solution) solvers to be used with cvxpy, default is ['ECOS','SCS']

    Note
    --------

    The temporal components are updated in parallel by default by forming of sequence of vertex covers.  

    Returns
    --------

    C:   np.ndarray
            matrix of temporal components (K x T)
    f:   np.array
            vector of temporal background (length T) 
    S:   np.ndarray            
            matrix of merged deconvolved activity (spikes) (K x T)
    bl:  float  
            same as input    
    c1:  float
            same as input    
    g:   float
            same as input    
    sn:  float
            same as input 
    YrA: np.ndarray
            matrix of spatial component filtered raw data, after all contributions have been removed.            
            YrA corresponds to the residual trace for each component and is used for faster plotting (K x T)

    """

    if 'p' not in kwargs or kwargs['p'] is None:
        raise Exception("You have to provide a value for p")

    d,T = np.shape(Y);    
    nr = np.shape(A)[-1]

    if b is not None:
        if b.shape[0]<b.shape[1]:
            b = b.T
        nb = b.shape[1]

    if  bl is None:
        bl=np.repeat(None,nr)

    if  c1 is None:
        c1=np.repeat(None,nr)

    if  g is None:
        g=np.repeat(None,nr)

    if  sn is None:
        sn=np.repeat(None,nr)                        

    A = scipy.sparse.hstack((A,coo_matrix(b)))
    S = np.zeros(np.shape(Cin));
    Cin =  np.vstack((Cin,fin));
    C = Cin;
    nA = np.squeeze(np.array(np.sum(np.square(A.todense()),axis=0)))


    Cin=coo_matrix(Cin)
    #YrA = ((A.T.dot(Y)).T-Cin.T.dot(A.T.dot(A)))
    print ('Generating residuals')
#    YA = (A.T.dot(Y).T)*spdiags(1./nA,0,nr+nb,nr+nb)

        
    if 'memmap' in str(type(Y)):
        if block_size <= 10000:
            print ('Forcing single thread for memory issues')
            dview_res = None
        else:
            
            dview_res = dview
            
        YA = parallel_dot_product(Y,A,dview=dview_res,block_size=block_size,transpose=True)*spdiags(old_div(1.,nA),0,nr+nb,nr+nb)
    else:
        YA = (A.T.dot(Y).T)*spdiags(old_div(1.,nA),0,nr+nb,nr+nb)
    print ('Done')
   # 
#    print np.allclose(YA,YA1)

    AA = ((A.T.dot(A))*spdiags(old_div(1.,nA),0,nr+nb,nr+nb)).tocsr()

    YrA = YA - Cin.T.dot(AA)
    #YrA = ((A.T.dot(Y)).T-Cin.T.dot(A.T.dot(A)))*spdiags(1./nA,0,nr+1,nr+1)


    Cin=np.array(Cin.todense())    
    for iter in range(ITER):
        O,lo = update_order(A.tocsc()[:,:nr])
        P_=[];
        for count,jo_ in enumerate(O):
            jo=np.array(list(jo_))           
            #Ytemp = YrA[:,jo.flatten()] + (np.dot(np.diag(nA[jo]),Cin[jo,:])).T
            Ytemp = YrA[:,jo.flatten()] + Cin[jo,:].T
            Ctemp = np.zeros((np.size(jo),T))
            Stemp = np.zeros((np.size(jo),T))
            btemp = np.zeros((np.size(jo),1))
            sntemp = btemp.copy()
            c1temp = btemp.copy()
            gtemp = np.zeros((np.size(jo),kwargs['p']));
            nT = nA[jo]            

#            args_in=[(np.squeeze(np.array(Ytemp[:,jj])), nT[jj], jj, bl[jo[jj]], c1[jo[jj]], g[jo[jj]], sn[jo[jj]], kwargs) for jj in range(len(jo))]
            args_in=[(np.squeeze(np.array(Ytemp[:,jj])), nT[jj], jj, None, None, None, None, kwargs) for jj in range(len(jo))]
#            import pdb
#            pdb.set_trace()
            if dview is not None:                    
                #
                if debug:                

                    results = dview.map_async(constrained_foopsi_parallel,args_in)  

                    results.get()

                    for outp in results.stdout:   

                        print((outp[:-1]))  

                        sys.stdout.flush()            

                    for outp in results.stderr:   

                        print((outp[:-1]))  

                        sys.stderr.flush()            

                else:

                    results = dview.map_sync(constrained_foopsi_parallel,args_in)

            else:

                results = list(map(constrained_foopsi_parallel,args_in))            


            for chunk in results:

                pars=dict()

                C_,Sp_,Ytemp_,cb_,c1_,sn_,gn_,jj_=chunk                    

                Ctemp[jj_,:] = C_[None,:]

                Stemp[jj_,:] = Sp_               

                Ytemp[:,jj_] = Ytemp_[:,None]            

                btemp[jj_] = cb_

                c1temp[jj_] = c1_

                sntemp[jj_] = sn_   

                gtemp[jj_,:] = gn_.T  

                bl[jo[jj_]] = cb_

                c1[jo[jj_]] = c1_

                sn[jo[jj_]] = sn_

                g[jo[jj_]]  = gn_.T if kwargs['p'] > 0 else [] #gtemp[jj,:]

                pars['b'] = cb_

                pars['c1'] = c1_                 

                pars['neuron_sn'] = sn_

                pars['gn'] = gtemp[jj_,np.abs(gtemp[jj_,:])>0] 

                pars['neuron_id'] = jo[jj_]

                P_.append(pars)

            YrA -= (Ctemp-C[jo,:]).T*AA[jo,:]
            #YrA[:,jo] = Ytemp
            C[jo,:] = Ctemp.copy()            

            S[jo,:] = Stemp

#            if (np.sum(lo[:jo])+1)%1 == 0:
            print((str(np.sum(lo[:count+1])) + ' out of total ' + str(nr) + ' temporal components updated'))

        ii=nr        


        #YrA[:,ii] = YrA[:,ii] + np.atleast_2d(Cin[ii,:]).T
        #cc = np.maximum(YrA[:,ii],0) 
        for ii in np.arange(nr,nr+nb):       
            cc = np.maximum(YrA[:,ii] + np.atleast_2d(Cin[ii,:]).T,0)
            YrA -= (cc-np.atleast_2d(Cin[ii,:]).T)*AA[ii,:]      
            C[ii,:] = cc.T
        #YrA = YA - C.T.dot(AA)
        #YrA[:,ii] = YrA[:,ii] - np.atleast_2d(C[ii,:]).T                

        if dview is not None:       
            dview.results.clear()   


        if old_div(scipy.linalg.norm(Cin - C,'fro'),scipy.linalg.norm(C,'fro')) <= 1e-3:
            # stop if the overall temporal component does not change by much
            print("stopping: overall temporal component not changing significantly")
            break
        else:
            Cin = C

    f = C[nr:,:]
    C = C[:nr,:]
    YrA = np.array(YrA[:,:nr]).T    
    P_ = sorted(P_, key=lambda k: k['neuron_id']) 


    return C,f,S,bl,c1,sn,g,YrA #,P_
