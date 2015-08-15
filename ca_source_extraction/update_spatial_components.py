# -*- coding: utf-8 -*-
"""
Created on Wed Aug 05 20:38:27 2015

@author: agiovann
"""
import scipy.io as sio
import numpy as np
from matplotlib import pylab as plt
from matplotlib.pylab import plot, imshow
from scipy.sparse import coo_matrix as coom
from scipy.sparse import spdiags
from scipy.linalg import eig
import scipy
#%% load example data
efty_params = sio.loadmat('efty_params.mat',struct_as_record=False) # load as structure matlab like
Y=efty_params['Yr']
C=efty_params['Cin']
f=efty_params['fin']
A_=efty_params['Ain']
P=efty_params['P'][0,0] # necessary because of the way it is stored
A=efty_params['A']
b=efty_params['b']
#%% set variables
[d,T] = np.shape(Y)
nr,_ = np.shape(C)       # number of neurons

d1=P.d1
d2=P.d2
dist=P.dist
g=P.g
sn=P.sn
min_size = 8
max_size = 3
show_sum = 0
Y_interp = P.interp
#%%
Coor=dict();
Coor['x'] = np.kron(np.ones((d2,1)),np.expand_dims(range(d1),axis=1)); 
Coor['y'] = np.kron(np.expand_dims(range(d2),axis=1),np.ones((d1,1)));

if not dist==np.inf:             # determine search area for each neuron
    cm = np.zeros((nr,2));        # vector for center of mass
    Vr = []    # cell(nr,1);
    IND = [];       # indicator for distance								   
    cm[:,0]=np.dot(Coor['x'].T,A_[:,:nr].todense())/A_[:,:nr].sum(axis=0)
    cm[:,1]=np.dot(Coor['y'].T,A_[:,:nr].todense())/A_[:,:nr].sum(axis=0) 
#    cm(:,1) = Coor.x'*A_(:,1:nr)./sum(A_(:,1:nr)); 
#    cm(:,2) = Coor.y'*A_(:,1:nr)./sum(A_(:,1:nr));          % center of mass for each components
    for i in range(nr):            # calculation of variance for each component and construction of ellipses
        dist_cm=scipy.sparse.coo_matrix(np.hstack((Coor['x'] - cm[i,0], Coor['y'] - cm[i,1])))            
        Vr.append(dist_cm.T*spdiags(A_[:,i].toarray().squeeze(),0,d,d)*dist_cm/A_[:,i].sum(axis=0))        
        D,V=eig(Vr[-1])        
        d11 = np.min((min_size**2,np.max((max_size**2,D[0].real))))
        d22 = np.min((min_size**2,np.max((max_size**2,D[1].real))))
        IND.append(np.sqrt((dist_cm*V[:,0])**2/d11 + (dist_cm*V[:,1])**2/d22)<=dist)       # search indexes for each component

    IND=(np.asarray(IND)).squeeze().T

Cf = np.vstack((C,f))
#%%
A = np.hstack((np.zeros((d,nr)),np.zeros((d,np.size(f,0)))))
sA = np.zeros((d1,d2))




for px in range(d):   # estimate spatial components
    fn = ~np.isnan(Y[px,:])       # identify missing data
    if dist == np.inf: # UP TO HERE ****************************************************************
#        [~, ~, a, ~] = lars_regression_noise(Y[px,fn].T, Cf[:,fn].T, 1, P.sn[px]**2*T);
#        A(px,:) = a';
#        sA(px) = sum(a);
        raise Exception('Not implemented')
    else:
        ind=np.where(IND[px,:])[0]
        if len(ind)>0:
            ind2 = [ind,nr+np.arange(f.shape[0])]
           # ind2 = [ind,nr+(1:size(f,1))];
            [~, ~, a, ~] = lars_regression_noise(Y[px,fn].T, Cf[ind2,fn].T, 1, P.sn[px]**2*T);
            A(px,ind2) = a';
            sA(px) = sum(a);
        end
    end
    if show_sum
        if mod(px,d1) == 0;
           figure(20); imagesc(sA); axis square;  
           title(sprintf('Sum of spatial components (%i out of %i columns done)',round(px/d1),d2)); drawnow;
        end
    end
end