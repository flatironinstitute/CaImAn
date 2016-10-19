# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 14:50:22 2016

@author: agiovann
"""
import cv2
import numpy as np
import pylab as pl
from tempfile import NamedTemporaryFile
from IPython.display import HTML
#%%
def playMatrix(mov,gain=1.0,frate=.033):
    for frame in mov: 
        if gain!=1:
            cv2.imshow('frame',frame*gain)
        else:
            cv2.imshow('frame',frame)
            
        if cv2.waitKey(int(frate*1000)) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break  
    cv2.destroyAllWindows()        
#%% montage
def matrixMontage(spcomps,*args, **kwargs):
    numcomps, width, height=spcomps.shape
    rowcols=int(np.ceil(np.sqrt(numcomps)));           
    for k,comp in enumerate(spcomps):        
        pl.subplot(rowcols,rowcols,k+1)       
        pl.imshow(comp,*args, **kwargs)                             
        pl.axis('off')         
        
        
#%%
VIDEO_TAG = """<video controls>
 <source src="data:video/x-m4v;base64,{0}" type="video/mp4">
 Your browser does not support the video tag.
</video>"""

def anim_to_html(anim,fps=20):
    if not hasattr(anim, '_encoded_video'):
        with NamedTemporaryFile(suffix='.mp4') as f:
            anim.save(f.name, fps=fps, extra_args=['-vcodec', 'libx264'])
            video = open(f.name, "rb").read()
        anim._encoded_video = video.encode("base64")
    
    return VIDEO_TAG.format(anim._encoded_video)
    

def display_animation(anim,fps=20):
    pl.close(anim._fig)
    return HTML(anim_to_html(anim,fps=fps))          
        