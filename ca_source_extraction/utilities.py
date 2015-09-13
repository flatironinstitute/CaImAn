# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 15:52:53 2015

@author: eftychios
"""

import numpy as np
from scipy.sparse import spdiags
from matplotlib import pyplot as plt

def local_correlations(Y,eight_neighbours=False, swap_dim = True):
     # Output:
     #   rho M x N matrix, cross-correlation with adjacent pixel
     # if eight_neighbours=True it will take the diagonal neighbours too
     # swap_dim True indicates that time is listed in the last axis of Y (matlab format)
     # and moves it in the front     
     
     if swap_dim:
         Y = np.transpose(Y,tuple(np.hstack((Y.ndim-1,range(Y.ndim)[:-1]))))
    
     rho = np.zeros(np.shape(Y)[1:3])
     w_mov = (Y - np.mean(Y, axis = 0))/np.std(Y, axis = 0)
 
     rho_h = np.mean(np.multiply(w_mov[:,:-1,:], w_mov[:,1:,:]), axis = 0)
     rho_w = np.mean(np.multiply(w_mov[:,:,:-1], w_mov[:,:,1:,]), axis = 0)
     
     if True:
         rho_d1 = np.mean(np.multiply(w_mov[:,1:,:-1], w_mov[:,:-1,1:,]), axis = 0)
         rho_d2 = np.mean(np.multiply(w_mov[:,:-1,:-1], w_mov[:,1:,1:,]), axis = 0)


     rho[:-1,:] = rho[:-1,:] + rho_h
     rho[1:,:] = rho[1:,:] + rho_h
     rho[:,:-1] = rho[:,:-1] + rho_w
     rho[:,1:] = rho[:,1:] + rho_w
     
     if eight_neighbours:
         rho[:-1,:-1] = rho[:-1,:-1] + rho_d2
         rho[1:,1:] = rho[1:,1:] + rho_d1
         rho[1:,:-1] = rho[1:,:-1] + rho_d1
         rho[:-1,1:] = rho[:-1,1:] + rho_d2
     
     
     if eight_neighbours:
         neighbors = 8 * np.ones(np.shape(Y)[1:3])  
         neighbors[0,:] = neighbors[0,:] - 3;
         neighbors[-1,:] = neighbors[-1,:] - 3;
         neighbors[:,0] = neighbors[:,0] - 3;
         neighbors[:,-1] = neighbors[:,-1] - 3;
         neighbors[0,0] = neighbors[0,0] + 1;
         neighbors[-1,-1] = neighbors[-1,-1] + 1;
         neighbors[-1,0] = neighbors[-1,0] + 1;
         neighbors[0,-1] = neighbors[0,-1] + 1;
     else:
         neighbors = 4 * np.ones(np.shape(Y)[1:3]) 
         neighbors[0,:] = neighbors[0,:] - 1;
         neighbors[-1,:] = neighbors[-1,:] - 1;
         neighbors[:,0] = neighbors[:,0] - 1;
         neighbors[:,-1] = neighbors[:,-1] - 1;   

     rho = np.divide(rho, neighbors)

     return rho
     
def order_components(A,C):

     A = np.array(A.todense())
     nA4 = np.sum(A**4,axis=0)**0.25
     mC = np.ndarray.max(np.array(C),axis=1)
     srt = np.argsort(nA4**mC)[::-1]
     A_or = A[:,srt]
     C_or = C[srt,:]
          
     return A_or, C_or, srt
     
def view_patches(Yr,A,C,b,f,d1,d2):
    
    nr,T = C.shape
    nA2 = np.sqrt(np.sum(np.array(A.todense())**2,axis=0))
    Y_r = np.array(spdiags(1/nA2,0,nr,nr)*(A.T*np.matrix(Yr-b[:,np.newaxis]*f[np.newaxis])))
    fig = plt.figure()
    
    for i in range(nr+1):
        if i < nr:
            ax1 = fig.add_subplot(1,2,1)
            plt.imshow(np.reshape(np.array(A.todense())[:,i],(d1,d2),order='F'),interpolation='None')
            ax1.set_title('Spatial component ' + str(i+1))    
            ax2 = fig.add_subplot(1,2,2)
            plt.plot(np.arange(T),np.squeeze(np.array(Y_r[i,:]))) 
            plt.plot(np.arange(T),np.squeeze(np.array(C[i,:])))
            ax2.set_title('Temporal component ' + str(i+1)) 
            ax2.legend(labels = ['Filtered raw data','Inferred trace'])
            plt.waitforbuttonpress()
            fig.delaxes(ax2)
        else:
            ax1 = fig.add_subplot(1,2,1)
            plt.imshow(np.reshape(b,(d1,d2),order='F'),interpolation='None')
            ax1.set_title('Spatial background background')    
            ax2 = fig.add_subplot(1,2,2)
            plt.plot(np.arange(T),np.squeeze(np.array(f))) 
            ax2.set_title('Temporal background')      