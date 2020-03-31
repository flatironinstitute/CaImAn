#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 16:45:00 2020

@author: caichangjia
"""
#%% 
import matplotlib.pyplot as plt
import numpy as np
from caiman.external.cell_magic_wand import cell_magic_wand_single_point
def tellme(s):
    print(s)
    plt.title(s, fontsize=16)
    plt.draw()

def quick_annotation(image, min_radius, max_radius, roughtness=2):
    keep_select=True
    ROIs = []
    while keep_select:
        # Plot image
        plt.clf()
        plt.imshow(image, cmap='gray', vmax=np.percentile(image, 98))            
        if len(ROIs) == 0:
            pass
        elif len(ROIs) == 1:
            plt.imshow(ROIs[0], alpha=0.3, cmap='Oranges')
        else:
            plt.imshow(np.array(ROIs).sum(axis=0), alpha=0.3, cmap='Oranges')
        
        # Plot point and ROI
        tellme('Click center of neuron')
        center = plt.ginput(1)[0]
        plt.plot(center[0], center[1], 'r+')
        ROI = cell_magic_wand_single_point(image, (center[1], center[0]), 
                                           min_radius=min_radius, max_radius=max_radius, 
                                           roughness=2, zoom_factor=1)[0]
        plt.imshow(ROI, alpha=0.3, cmap='Reds')
    
        # Select or not
        tellme('Select? Key click for yes, mouse click for no')
        select = plt.waitforbuttonpress()
        if select:
            ROIs.append(ROI)
            tellme('You have selected a neuron. \n Keep selecting? Key click for yes, mouse click for no')
        else:
            tellme('You did not select a neuron \n Keep selecting? Key click for yes, mouse click for no')
        keep_select = plt.waitforbuttonpress()
        
    plt.close()        
    ROIs = np.array(ROIs)
    return ROIs