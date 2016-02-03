# -*- coding: utf-8 -*-
"""A set of routines for estimating the temporal components, given the spatial components and temporal components
@author: agiovann
"""
from scipy.sparse import spdiags,coo_matrix#,csgraph
import scipy
import numpy as np
#import cPickle as pickle
from deconvolution import constrained_foopsi
#import random
#from scipy import linalg
#from spatial import update_spatial_components
from utilities import update_order
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
def constrained_foopsi_parallel(arg_in):
    """ necessary for parallel computation of the function  constrained_foopsi
    """  
   
    Ytemp, nT, jj_, bl, c1, g, sn, argss = arg_in
    T=np.shape(Ytemp)[0]
    cc_,cb_,c1_,gn_,sn_,sp_ = constrained_foopsi(Ytemp/nT, bl = bl,  c1 = c1, g = g,  sn = sn, **argss)
    gd_ = np.max(np.roots(np.hstack((1,-gn_.T))));  
    gd_vec = gd_**range(T)   
    

    C_ = cc_[:].T + cb_ + np.dot(c1_,gd_vec)
    Sp_ = sp_[:T].T
    Ytemp_ = Ytemp - np.dot(nT,C_).T
    
    return C_,Sp_,Ytemp_,cb_,c1_,sn_,gn_,jj_
    
     
#%%
def update_temporal_components(Y, A, b, Cin, fin, bl = None,  c1 = None, g = None,  sn = None, ITER=2, method_foopsi='constrained_foopsi', n_processes=1, backend='single_thread',memory_efficient=False, **kwargs):
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
    memory_efficient: Bool
        whether or not to optimize for memory usage (longer running times). nevessary with very large datasets  
    **kwargs: dict
        all parameters passed to constrained_foopsi except bl,c1,g,sn (see documentation). Some useful parameters are      
    p: int
        order of the autoregression model
    method: [optional] string
        solution method for basis projection pursuit cvx or spgl1 or debug for fast but possibly imprecise temporal components    
  
    Returns
    --------
    
    C:     np.matrix
            matrix of temporal components (K x T)
    f:     np.array
            vector of temporal background (length T) 
    S:     np.ndarray            
            matrix of merged deconvolved activity (spikes) (K x T)
    bl:  float  
            same as input    
    c1:  float
            same as input    
    g:   float
            same as input    
    sn:  float
            same as input 
    
    """
    if not kwargs.has_key('p') or kwargs['p'] is None:
        raise Exception("You have to provide a value for p")

    d,T = np.shape(Y);    
    nr = np.shape(A)[-1]
    
    
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
    YrA = (A.T.dot(Y)).T-Cin.T.dot(A.T.dot(A))
    
    
    if backend == 'ipyparallel':
        try: # if server is not running and raise exception if not installed or not started        
            from ipyparallel import Client
            c = Client()
        except:
            print "this backend requires the installation of the ipyparallel (pip install ipyparallel) package and  starting a cluster (type ipcluster start -n 6) where 6 is the number of nodes"
            raise
    
        if len(c) <  n_processes:
            print len(c)
            raise Exception("the number of nodes in the cluster are less than the required processes: decrease the n_processes parameter to a suitable value")            
        
        dview=c[:n_processes] # use the number of processes
    
    Cin=np.array(Cin.todense())    
    for iter in range(ITER):
        O,lo = update_order(A.tocsc()[:,:nr])
        P_=[];
        for count,jo_ in enumerate(O):
            jo=np.array(list(jo_))           
            Ytemp = YrA[:,jo.flatten()] + (np.dot(np.diag(nA[jo]),Cin[jo,:])).T
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
            if backend == 'ipyparallel':                    
                
                results = dview.map_sync(constrained_foopsi_parallel,args_in)        

            elif backend == 'single_thread':
                
                results = map(constrained_foopsi_parallel,args_in)            
                
            else:
                
                raise Exception('Backend not defined. Use either single_thread or ipyparallel')
                
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
                g[jo[jj_]]  = gtemp[jj,:]#[jj_,np.abs(gtemp[jj,:])>0] 
                             
                
                pars['b'] = cb_
                pars['c1'] = c1_                 
                pars['neuron_sn'] = sn_
                pars['gn'] = gtemp[jj_,np.abs(gtemp[jj,:])>0] 
#                
##                for jj = 1:length(O{jo})
##                    P.gn(O{jo}(jj)) = {gtemp(jj,abs(gtemp(jj,:))>0)'};
##                end
                pars['neuron_id'] = jo[jj_]
                P_.append(pars)
            
            YrA[:,jo] = Ytemp
            C[jo,:] = Ctemp            
            S[jo,:] = Stemp
            
#            if (np.sum(lo[:jo])+1)%1 == 0:
            print str(np.sum(lo[:count+1])) + ' out of total ' + str(nr) + ' temporal components updated \n'
        
        ii=nr        
        YrA[:,ii] = YrA[:,ii] + nA[ii]*np.atleast_2d(Cin[ii,:]).T
        cc = np.maximum(YrA[:,ii]/nA[ii],0)
        C[ii,:] = cc[:].T
        YrA[:,ii] = YrA[:,ii] - nA[ii]*np.atleast_2d(C[ii,:]).T 
        
        if backend == 'ipyparallel':       
            dview.results.clear()   
            c.purge_results('all')
            c.purge_everything()

        if scipy.linalg.norm(Cin - C,'fro')/scipy.linalg.norm(C,'fro') <= 1e-3:
            # stop if the overall temporal component does not change by much
            print "stopping: overall temporal component not changing significantly"
            break
        else:
            Cin = C
    
    #Y_res = Y - A*C # this includes the baseline term
    
    f = C[nr:,:]
    C = C[:nr,:]
        
    P_ = sorted(P_, key=lambda k: k['neuron_id']) 
    if backend == 'ipyparallel':      
        c.close()
    
    return C,f,S,bl,c1,sn,g #,P_
    
#%%
#def update_temporal_components(Y,A,b,Cin,fin,ITER=2,method_foopsi='constrained_foopsi',n_processes=1, backend='single_thread', memory_efficient=False, **kwargs):
##def update_temporal_components(Y,A,b,Cin,fin,ITER=2,method_foopsi='constrained_foopsi',deconv_method = 'cvx', g='None',**kwargs):
#    """update temporal components and background given spatial components using a block coordinate descent approach
#    Inputs:
#    Y: np.ndarray (2D)
#        input data with time in the last axis (d x T)
#    A: sparse matrix (crc format)
#        matrix of temporal components (d x K)
#    Cin: np.ndarray
#        current estimate of temporal components (K x T)
#    ITER: positive integer
#        Maximum number of block coordinate descent loops. Default: 2
#    fin: np.ndarray
#        current estimate of temporal background (vector of length T)
#    method_foopsi: string
#        Method of deconvolution of neural activity. 
#        Default: constrained_foopsi (constrained deconvolution, the only method supported at the moment)
#    deconv_method: string
#        Solver for constrained foopsi ('cvx' or 'spgl1', default: 'cvx')
#    g:  np.ndarray
#        Global time constant (not used)
#    **kwargs: all parameters passed to constrained_foopsi
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
#                               
#    Outputs:
#    C:     np.matrix
#            matrix of temporal components (K x T)
#    f:     np.array
#            vector of temporal background (length T) 
#    P_:    dictionary
#            Dictionary with parameters for each temporal component:
#                P_.b:           baseline for fluorescence trace
#                P_.c1:          initial concentration
#                P_.gn:          discrete time constant
#                P_.neuron_sn:   noise level
#                P_.neuron_id:   index of component
#    Sp:    np.matrix
#            matrix of deconvolved neural activity (K x T)
#    """
#
#
#    d,T = np.shape(Y);
#    
#
#    nr = np.shape(A)[-1]
#    A = scipy.sparse.hstack((A,coo_matrix(b)))
#    Cin =  np.vstack((Cin,fin));
#    C = Cin;
#    #%
#    nA = np.squeeze(np.array(np.sum(np.square(A.todense()),axis=0)))
#    
#    Y=np.matrix(Y)
#    C=np.matrix(C)
#    Cin=np.matrix(Cin)
#    Sp = np.zeros((nr,T))
#    YrA = Y.T*A - Cin.T*(A.T*A);
#
#
#    for iter in range(ITER):
#        idxs=range(nr+1)
#        random.shuffle(idxs)
#        P_=[];
#    #    perm = randperm(nr+1)
#        for jj,ii in enumerate(idxs):            
#            #ii=jj
#            #print ii,jj
#            pars=dict()
#    #        ii = perm(jj);
#            if ii<nr:                
#                if method_foopsi == 'constrained_foopsi':
#                        #print YrA.shape 
#                        #print YrA.shape
#                        YrA[:,ii] = YrA[:,ii] + nA[ii]*Cin[ii,:].T                  
#                        cc,cb,c1,gn,sn,sp = constrained_foopsi(np.squeeze(np.asarray(YrA[:,ii]/nA[ii])), **pars)
#                        #print pars
#                        pars['gn'] = gn
#                        
#                        gd = np.max(np.roots(np.hstack((1,-gn.T))));  # decay time constant for initial concentration
#                        gd_vec = gd**range(T)
#                        
#                        C[ii,:] = cc[:].T + cb + c1*gd_vec
#                        Sp[ii,:] = sp[:T].T
#                        YrA[:,ii] = YrA[:,ii] - np.matrix(nA[ii]*C[ii,:]).T
#                        pars['b'] = cb
#                        pars['c1'] = c1           
#                        pars['neuron_sn'] = sn
#                        pars['neuron_id'] = ii
#                        P_.append(pars)
#                else:
#                        raise Exception('undefined  method')                        
#                    
#                
#            else:
#                YrA[:,ii] = YrA[:,ii] + nA[ii]*Cin[ii,:].T
#                cc = np.maximum(YrA[:,ii]/nA[ii],0)
#                C[ii,:] = cc[:].T # if you use this should give an error that we need to correct
#                YrA[:,ii] = YrA[:,ii] - nA[ii]*C[ii,:].T
#            
#            if (jj+1)%10 == 0:
#                print str(jj+1) + ' out of total ' + str(nr+1) + ' temporal components updated \n'
#    
#    
#        #%disp(norm(Fin(1:nr,:) - F,'fro')/norm(F,'fro'));
#        if scipy.linalg.norm(Cin - C,'fro')/scipy.linalg.norm(C,'fro') <= 1e-3:
#            # stop if the overall temporal component does not change by much
#            break
#        else:
#            Cin = C
#        
#    
#    
#    #Y_res = Y - A*C
#    
#    f = C[nr:,:]
#    C = C[:nr,:]
#        
#    P_ = sorted(P_, key=lambda k: k['neuron_id']) 
#    
#    return C,f,P_,Sp