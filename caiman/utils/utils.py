# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 21:01:17 2015

@author: agiovann
"""
from __future__ import print_function

#%%
import numpy as np


#%% 
def val_parse(v):
    # parse values from si tags into python objects if possible
    try:
        return eval(v)
    except:
        if v == 'true':
            return True
        elif v == 'false':
            return False
        elif v == 'NaN':
            return np.nan
        elif v == 'inf' or v == 'Inf':
            return np.inf

        else:
            return v
#%%
def si_parse(imd):

    # parse image_description field embedded by scanimage
    imd = imd.split('\n')
    imd = [i for i in imd if '=' in i]
    imd = [i.split('=') for i in imd]
    imd = [[ii.strip(' \r') for ii in i] for i in imd]
    imd = {i[0]:val_parse(i[1]) for i in imd}
    return imd

#%%
def get_image_description_SI(fname):
    """Given a tif file acquired with Scanimage it returns a dictionary containing the information in the image description field
    """
    image_descriptions=[]
    
    try:
        
        from tifffile import TiffFile
    
    except:

        print('tifffile package not found, using skimage.external.tifffile')
        from skimage.external.tifffile import TiffFile 
        
    tf=TiffFile(fname)
    
    for idx,pag in enumerate(tf.pages):
        if idx%1000==0:
            print(idx)
    #        i2cd=si_parse(pag.tags['image_description'].value)['I2CData']
        field=pag.tags['image_description'].value

        image_descriptions.append(si_parse(field))

    return image_descriptions


