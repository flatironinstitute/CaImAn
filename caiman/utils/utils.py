# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 21:01:17 2015

@author: agiovann
"""

#%%
import numpy as np
import os
import tifffile

#%%    
def process_movie_parallel(arg_in):

    import caiman as cm
    import numpy as np    
    

    fname,fr,margins_out,template,max_shift_w, max_shift_h,remove_blanks,apply_smooth,save_hdf5=arg_in
    
    if template is not None:
        if type(template) is str:
            if os.path.exists(template):
                template=cm.load(template,fr=1)
            else:
                raise Exception('Path to template does not exist:'+template)                
#    with open(fname[:-4]+'.stout', "a") as log:
#        print fname
#        sys.stdout = log
        
    #    import pdb
    #    pdb.set_trace()
    
    if type(fname) is cm.movie or type(fname) is cm.movies.movie:
        print type(fname)
        Yr=fname

    else:        
        
        Yr=cm.load(fname,fr=fr)
        
    if Yr.ndim>1:

        print 'loaded'    

        if apply_smooth:

            print 'applying smoothing'

            Yr=Yr.bilateral_blur_2D(diameter=10,sigmaColor=10000,sigmaSpace=0)

#        bl_yr=np.float32(np.percentile(Yr,8))    

 #       Yr=Yr-bl_yr     # needed to remove baseline

        print 'Remove BL'

        if margins_out!=0:

            Yr=Yr[:,margins_out:-margins_out,margins_out:-margins_out] # borders create troubles

        print 'motion correcting'

        Yr,shifts,xcorrs,template=Yr.motion_correct(max_shift_w=max_shift_w, max_shift_h=max_shift_h,  method='opencv',template=template,remove_blanks=remove_blanks) 

  #      Yr = Yr + bl_yr           
        
        if type(fname) is cm.movie:

            return Yr 

        else:     
            
            print 'median computing'        

            template=Yr.bin_median()

            print 'saving'  

            idx_dot=len(fname.split('.')[-1])

            if save_hdf5:

                Yr.save(fname[:-idx_dot]+'hdf5')        

            print 'saving 2'                 

            np.savez(fname[:-idx_dot]+'npz',shifts=shifts,xcorrs=xcorrs,template=template)

            print 'deleting'        

            del Yr

            print 'done!'
        
            return fname[:-idx_dot] 
        #sys.stdout = sys.__stdout__ 
    else:
        return None
           
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
    tf=tifffile.TiffFile(fname)
    for idx,pag in enumerate(tf.pages):
            if idx%1000==0:
                print(idx)
    #        i2cd=si_parse(pag.tags['image_description'].value)['I2CData']
            field=pag.tags['image_description'].value
            
            image_descriptions.append(si_parse(field))
    
    return image_descriptions
    
   