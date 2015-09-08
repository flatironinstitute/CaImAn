# -*- coding: utf-8 -*-
"""
Created on Sat Aug 08 10:19:51 2015

@author: agiovann
"""
#%%
from cvxopt import matrix, spmatrix, spdiag, solvers
import picos
import numpy as np
import scipy

#%%


def lars_regression_noise(Yp, X, positive, noise,verbose=False):
    
    """
     Run LARS for regression problems with LASSO penalty, with optional positivity constraints
     Author: Andrea Giovannucci. Adapted code from Eftychios Pnevmatikakis
    
    
     Input Parameters:
       Yp:          Yp[:,t] is the observed data at time t
       X:           the regresion problem is Yp=X*W + noise
       maxcomps:    maximum number of active components to allow
       positive:    a flag to enforce positivity
       noise:       the noise of the observation equation. if it is not
                    provided as an argument, the noise is computed from the
                    variance at the end point of the algorithm. The noise is
                    used in the computation of the Cp criterion.
    
    
     Output Parameters:
       Ws: weights from each iteration
       lambdas: lambda_ values at each iteration
       TODO: W_lam, lam, flag
       Cps: C_p estimates
       last_break:     last_break(m) == n means that the last break with m non-zero weights is at Ws(:,:,n)
    """
    #%%
    
    #verbose=true;
    
    k=1;    
    
    Yp=np.expand_dims(Yp,axis=1) #necessary for matrix multiplications
       
    _,T = np.shape(Yp); # of time steps
    _,N = np.shape(X); # of compartments
    
    maxcomps = N;    
    W = np.zeros((N,k));
    active_set = np.zeros((N,k));
    visited_set = np.zeros((N,k));    
    lambdas = [];  
    Ws=[] #=np.zeros((W.shape[0],W.shape[1],maxcomps));  # Just preallocation. Ws may end with more or less than maxcomp columns       
    r = np.expand_dims(np.dot(X.T,Yp.flatten()),axis=1)       # N-dim vector
    M = np.dot(-X.T,X);            # N x N matrix 
    
    #%% begin main loop
    
    i = 0;
    flag = 0;

    while 1:
        
        if flag == 1:
            W_lam = 0
            break;
        
#        print i
        
    #% calculate new gradient component if necessary    
        if i>0 and new>=0 and visited_set[new] ==0: # AG NOT CLEAR HERE
    
            visited_set[new] =1;    #% remember this direction was computed    
        
    
    #% Compute full gradient of Q     
        dQ = r + np.dot(M,W);
            
    #% Compute new W
        if i == 0:
            
            if positive:
                dQa = dQ
            else:
                dQa = np.abs(dQ)
            
            lambda_, new = np.max(dQa),np.argmax(dQa)
            
            #[lambda_, new] = max(dQa(:));
        
            if lambda_ < 0:
                print 'All negative directions!'
                break
            
            
        else:
    
            #% calculate vector to travel along          
            
            avec, gamma_plus, gamma_minus = calcAvec(new, dQ, W, lambda_, active_set, M, positive)       
            
           # % calculate time of travel and next new direction 
                    
            if new==-1:                              # % if we just dropped a direction we don't allow it to emerge 
                if dropped_sign == 1:               # % with the same sign
                    gamma_plus[dropped] = np.inf;
                else:
                    gamma_minus[dropped] = np.inf;
                
            
                       
    
            gamma_plus[active_set == 1] = np.inf       #% don't consider active components 
            gamma_plus[gamma_plus <= 0] = np.inf       #% or components outside the range [0, lambda_]
            gamma_plus[gamma_plus> lambda_] = np.inf
            gp_min, gp_min_ind = np.min(gamma_plus),np.argmin(gamma_plus)
    
    
            if positive:
                gm_min = np.inf;                         #% don't consider new directions that would grow negative
            else:
                gamma_minus[active_set == 1] = np.inf            
                gamma_minus[gamma_minus> lambda_] =np.inf
                gamma_minus[gamma_minus <= 0] = np.inf
                gm_min, gm_min_ind = np.min(gamma_minus),np.argmin(gamma_minus)
    
            
    
            [g_min, which] = np.min(gp_min),np.argmin(gp_min)
            
        
    
            if g_min == np.inf:               #% if there are no possible new components, try move to the end
                g_min = lambda_;            #% This happens when all the components are already active or, if positive==1, when there are no new positive directions 
            
                     
            
    
            #% LARS check  (is g_min*avec too large?)
            gamma_zero = -W[active_set == 1]  / np.squeeze(avec);
            gamma_zero_full = np.zeros((N,k));
            gamma_zero_full[active_set == 1] = gamma_zero;
            gamma_zero_full[gamma_zero_full <= 0] = np.inf;
            gz_min, gz_min_ind = np.min(gamma_zero_full),np.argmin(gamma_zero_full)
            
            if gz_min < g_min:       
#                print 'check_here'
                if verbose:
                   print 'DROPPING active weight:' + str(gz_min_ind)
                
                active_set[gz_min_ind] = 0;
                dropped = gz_min_ind;
                dropped_sign = np.sign(W[dropped]);
                W[gz_min_ind] = 0;
                avec = avec[gamma_zero != gz_min];
                g_min = gz_min;
                new=-1 # new = 0;
                
                
            elif g_min < lambda_:            
                if  which == 0:
                    new = gp_min_ind;
                    if verbose:
                        print 'new positive component:' + str(new)
                    
                else:
                    new = gm_min_ind;
                    print 'new negative component:' +  str(new)
                                   
            W[active_set == 1] = W[active_set == 1] + np.dot(g_min,np.squeeze(avec));
            
            if positive:
                if any(W<0):
                    #min(W);
                    flag = 1;
                    #%error('negative W component');
    
            
            lambda_ = lambda_ - g_min;    
    
    #%  Update weights and lambdas 
            
        lambdas.append(lambda_);    
        Ws.append(W.copy())
        
    #    print Ws
        if len((Yp-np.dot(X,W)).shape)>2:
            res = scipy.linalg.norm(np.squeeze(Yp-np.dot(X,W)),'fro')**2;
        else:
            res = scipy.linalg.norm(Yp-np.dot(X,W),'fro')**2;
    
    
    #% Check finishing conditions                
        if lambda_ ==0 or (new>=0 and np.sum(active_set) == maxcomps) or (res < noise):
            if verbose:
                print 'end. \n'        
            break
    
    
    #%   
        if new>=0:        
            active_set[new] = 1
        
        
        i = i + 1
    
    Ws_old=Ws
    # end main loop 
    #%% final calculation of mus
    Ws=np.asarray(np.swapaxes(np.swapaxes(Ws_old,0,1),1,2))
    if flag == 0:
        if i > 0:
            Ws= np.squeeze(Ws[:,:,:len(lambdas)]);
            w_dir = -(Ws[:,i] - Ws[:,i-1])/(lambdas[i]-lambdas[i-1]);
            Aw = np.dot(X,w_dir);
            y_res = np.squeeze(Yp) - np.dot(X,Ws[:,i-1] + w_dir*lambdas[i-1]);
            ld = scipy.roots([scipy.linalg.norm(Aw)**2,-2*np.dot(Aw.T,y_res),np.dot(y_res.T,y_res)-noise]);
            lam = ld[np.intersect1d(np.where(ld>lambdas[i]),np.where(ld<lambdas[i-1]))];
            if len(lam) == 0  or np.any(lam)<0 or np.any(~np.isreal(lam)):
                lam = np.array([lambdas[i]]);
            
            W_lam = Ws[:,i-1] + np.dot(w_dir,lambdas[i-1]-lam[0]);
        else:
            problem = picos.Problem()
            W_lam = problem.add_variable('W_lam', X.shape[1])
            problem.set_objective('min', 1|W_lam)
            problem.add_constraint(W_lam >= 0)
            problem.add_constraint(picos.norm(matrix(Yp.astype(np.float))-matrix(X.astype(np.float))*W_lam,2)<=np.sqrt(noise))
            sel_solver = []
            problem.solver_selection()
            problem.solve(verbose=True)
           
            
    #        cvx_begin quiet
    #            variable W_lam(size(X,2));
    #            minimize(sum(W_lam));
    #            subject to
    #                W_lam >= 0;
    #                norm(Yp-X*W_lam)<= sqrt(noise);
    #        cvx_end
            lam = 10;
        
    else:
        W_lam = 0;
        Ws = 0;
        lambdas = 0; 
        lam = 0;
        
    return Ws, lambdas, W_lam, lam, flag

#%%  auxiliary functions
def calcAvec(new, dQ, W, lambda_, active_set, M, positive):
    # TODO: comment
    r,c=np.nonzero(active_set)    
#    [r,c] = find(active_set);
    Mm = -M.take(r,axis=0).take(r,axis=1)
    
    
    Mm=(Mm + Mm.T)/2;
    
    #% verify that there is no numerical instability 
    if len(Mm)>1:
#        print Mm.shape
        eigMm,_ = scipy.linalg.eig(Mm)
        eigMm=np.real(eigMm)
#        check_here
    else:
        eigMm=Mm
    
    if any(eigMm < 0):
        np.min(eigMm)
        #%error('The matrix Mm has negative eigenvalues')  
        flag = 1;
    
    
    
    b = np.sign(W);
    
    if new>=0:
        b[new] = np.sign(dQ[new]);
    
    b = b[active_set == 1];
    
    if len(Mm)>1:
        avec = np.linalg.solve(Mm,b)
    else:
        avec=b/Mm

    
    if positive: 
        if new>=0: 
            in_ = np.sum(active_set[:new]);
            if avec[in_] < 0:
                #new;
                #%error('new component of a is negative')
                flag = 1;                        
    
        
    
    one_vec = np.ones(W.shape);
    
    dQa = np.zeros(W.shape);
    for j in range(len(r)):
        dQa = dQa + np.expand_dims(avec[j]*M[:, r[j]],axis=1);
    
    
    gamma_plus = (lambda_ - dQ)/(one_vec + dQa);
    gamma_minus = (lambda_ + dQ)/(one_vec - dQa);
    
    return avec, gamma_plus, gamma_minus
