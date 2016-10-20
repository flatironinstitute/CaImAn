# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 13:22:26 2015

@author: agiovann
"""



#%%
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import label
import pylab as pl
import numpy as np


#%%
def extractROIsFromPCAICA(spcomps, numSTD=4, gaussiansigmax=2 , gaussiansigmay=2,thresh=None):
    """
    Given the spatial components output of the IPCA_stICA function extract possible regions of interest
    The algorithm estimates the significance of a components by thresholding the components after gaussian smoothing
    Parameters
    -----------
    spcompomps, 3d array containing the spatial components
    numSTD: number of standard deviation above the mean of the spatial component to be considered signiificant
    """        
    
    numcomps, width, height=spcomps.shape
    rowcols=int(np.ceil(np.sqrt(numcomps)));  
    
    #%
    allMasks=[];
    maskgrouped=[];
    for k in xrange(0,numcomps):
        comp=spcomps[k]
#            plt.subplot(rowcols,rowcols,k+1)
        comp=gaussian_filter(comp,[gaussiansigmay,gaussiansigmax])
        
        maxc=np.percentile(comp,99);
        minc=np.percentile(comp,1);
#            comp=np.sign(maxc-np.abs(minc))*comp;
        q75, q25 = np.percentile(comp, [75 ,25])
        iqr = q75 - q25
        minCompValuePos=np.median(comp)+numSTD*iqr/1.35;  
        minCompValueNeg=np.median(comp)-numSTD*iqr/1.35;            

        # got both positive and negative large magnitude pixels
        if thresh is None:
            compabspos=comp*(comp>minCompValuePos)-comp*(comp<minCompValueNeg);
        else:
            compabspos=comp*(comp>thresh)-comp*(comp<-thresh);

        #height, width = compabs.shape
        labeledpos, n = label(compabspos>0, np.ones((3,3)))
        maskgrouped.append(labeledpos)
        for jj in range(1,n+1):
            tmp_mask=np.asarray(labeledpos==jj)
            allMasks.append(tmp_mask)
#            labeledneg, n = label(compabsneg>0, np.ones((3,3)))
#            maskgrouped.append(labeledneg)
#            for jj in range(n):
#                tmp_mask=np.asarray(labeledneg==jj)
#                allMasks.append(tmp_mask)
#            plt.imshow(labeled)                             
#            plt.axis('off')         
    return allMasks,maskgrouped 