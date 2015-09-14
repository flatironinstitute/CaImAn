# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 15:52:53 2015

@author: eftychios
"""

import numpy as np
from scipy.sparse import spdiags, diags
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
     nA2 = np.sqrt(np.sum(A**2,axis=0))
     A = np.array(np.matrix(A)*diags(1/nA2,0))
     nA4 = np.sum(A**4,axis=0)**0.25
     C = np.array(diags(nA2,0)*np.matrix(C))
     mC = np.ndarray.max(np.array(C),axis=1)
     srt = np.argsort(nA4**mC)[::-1]
     A_or = A[:,srt]
     C_or = C[srt,:]
          
     return A_or, C_or, srt
     

def extract_DF_F(Y,A,C,i=None):
    
    A2 = A.copy()
    A2.data **= 2
    nA2 = np.squeeze(np.array(A2.sum(axis=0)))
    A = A*diags(1/nA2,0)
    C = diags(nA2,0)*C         
     
    #if i is None:
    #    i = np.argmin(np.max(A,axis=0))

    Y = np.matrix(Y)         
    Yf =  A.transpose()*(Y - A*C) #+ A[:,i]*C[i,:])
    Df = np.median(np.array(Yf),axis=1)
    C_df = diags(1/Df,0)*C
             
    return C_df, Df

def find_unsaturated_pixels(Y, saturationValue = None, saturationThreshold = 0.9, saturationTime = 0.005):
    
    if saturationValue == None:
        saturationValue = np.power(2,np.ceil(np.log2(np.max(Y))))-1
        
    Ysat = (Y >= saturationThreshold*saturationValue)
    pix = np.mean(Ysat,Y.ndim-1).flatten('F') > saturationTime
    normalPixels = np.where(pix)
    
    return normalPixels
     
def com(A,d1,d2):

    nr = np.shape(A)[-1]
    Coor=dict();
    Coor['x'] = np.kron(np.ones((d2,1)),np.expand_dims(range(d1),axis=1)); 
    Coor['y'] = np.kron(np.expand_dims(range(d2),axis=1),np.ones((d1,1)));
    cm = np.zeros((nr,2));        # vector for center of mass							   
    cm[:,0]=np.dot(Coor['x'].T,A)/A.sum(axis=0)
    cm[:,1]=np.dot(Coor['y'].T,A)/A.sum(axis=0)       
    
    return cm
     
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
            
def plot_contours(A,Cn,thr = 0.995, display_numbers = True, max_number = None, **kwargs):

    A = np.array(A.todense())
    d1,d2 = np.shape(Cn)
    d,nr = np.shape(A)       
    if max_number is None:
        max_number = nr
        
    x,y = np.mgrid[0:d1:1,0:d2:1]    
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.imshow(Cn,interpolation=None)
    coordinates = []
    cm = com(A,d1,d2)
    for i in range(np.minimum(nr,max_number)):
        pars=dict(kwargs)
        indx = np.argsort(A[:,i],axis=None)[::-1]
        cumEn = np.cumsum(A[:,i].flatten()[indx]**2)
        cumEn /= cumEn[-1]
        Bvec = np.zeros(d)
        Bvec[indx] = cumEn
        Bmat = np.reshape(Bvec,np.shape(Cn),order='F')
        cs = plt.contour(y,x,Bmat,[thr])
        p = cs.collections[0].get_paths()[0]
        v = p.vertices
        pars['CoM'] = np.squeeze(cm[i,:])
        pars['coordinates'] = v           
        pars['bbox'] = [np.floor(np.min(v[:,1])),np.ceil(np.max(v[:,1])),np.floor(np.min(v[:,0])),np.ceil(np.max(v[:,0]))]
        pars['neuron_id'] = i+1
        coordinates.append(pars)        
        
    if display_numbers:
        for i in range(np.minimum(nr,max_number)):
            ax.text(cm[i,1],cm[i,0],str(i+1))
            
    return coordinates