# -*- coding: utf-8 -*-
"""
Created on Tue May 23 21:41:17 2017

@author: epnevmatikakis
"""

import numpy as np

def create_rotation_field(max_displacement=(10,10),center=(0,0),nx=512,ny=512):
    
    x = np.linspace(-max_displacement[0]-center[0],max_displacement[0]-center[0],nx)
    y = np.linspace(-max_displacement[1]-center[1],max_displacement[1]-center[1],ny)
    X,Y = np.meshgrid(x,y)
    rho = np.sqrt(X**2,Y**2)
    theta = np.arctan2(Y,X)
    
    vx = -np.sin(theta)*rho
    vy = np.cos(theta)*rho
    
    return vx,vy
    
def create_stretching_field(stretching_factor=(1.,1.),max_displacement=(5,5),center=(0,0),nx=512,ny=512):
    
    '''max shift would be equal to (max_displacement-center)*(stretching_factor-1)'''   
    
    x = np.linspace(-max_displacement[0]-center[0],max_displacement[0]-center[0],nx)
    y = np.linspace(-max_displacement[1]-center[1],max_displacement[1]-center[1],ny)
    X,Y = np.meshgrid(x,y)
    
    X_stretch = X*stretching_factor[0]
    Y_stretch = Y*stretching_factor[1]
    
    vx = X_stretch - X
    vy = Y_stretch - Y
    
    return vx,vy
    
def create_shearing_field(shearing_factor=0., max_displacement = (5,5), along_x = True, center = (0,0), nx = 512, ny = 512):

    '''set along_x to False for shearing along y axis'''

    x = np.linspace(-max_displacement[0]-center[0],max_displacement[0]-center[0],nx)
    y = np.linspace(-max_displacement[1]-center[1],max_displacement[1]-center[1],ny)
    X,Y = np.meshgrid(x,y)

    if along_x:
        vx = shearing_factor*Y
        vy = np.zeros((nx,ny))
    else:
        vx = np.zeros((nx,ny))
        vy = shearing_factor*X
        
    return vx,vy