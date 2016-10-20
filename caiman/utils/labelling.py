# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 15:18:21 2016

@author: agiovann
"""
import numpy as np
import pylab as pl
import caiman as cm
#%%
def pre_preprocess_movie_labeling(dview, file_names, median_filter_size=(2,1,1), 
                                  resize_factors=[.2,.1666666666],diameter_bilateral_blur=4):
   def pre_process_handle(args):
        
        from scipy.ndimage import filters as ft
        import logging
        
        fil, resize_factors, diameter_bilateral_blur,median_filter_size=args
        
        name_log=fil[:-4]+ '_LOG'
        logger = logging.getLogger(name_log)
        hdlr = logging.FileHandler(name_log)
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        hdlr.setFormatter(formatter)
        logger.addHandler(hdlr) 
        logger.setLevel(logging.INFO)

        logger.info('START')
        logger.info(fil)
        mov=cm.load(fil,fr=30)
        logger.info('Read file')

        mov=mov.resize(1,1,resize_factors[0])
        logger.info('Resize')
        mov=mov.bilateral_blur_2D(diameter=diameter_bilateral_blur)
        logger.info('Bilateral')
        mov1=cm.movie(ft.median_filter(mov,median_filter_size),fr=30)
        logger.info('Median filter')
        #mov1=mov1-np.median(mov1,0)
        mov1=mov1.resize(1,1,resize_factors[1])
        logger.info('Resize 2')
        mov1=mov1-cm.utils.stats.mode_robust(mov1,0)
        logger.info('Mode')
        mov=mov.resize(1,1,resize_factors[1])
        logger.info('Resize')
#        mov=mov-np.percentile(mov,1)
        
        mov.save(fil[:-4] + '_compress_.tif')
        logger.info('Save 1')
        mov1.save(fil[:-4] + '_BL_compress_.tif')
        logger.info('Save 2')
        return 1
        
   args=[]
   for name in file_names:
            args.append([name,resize_factors,diameter_bilateral_blur,median_filter_size])
            
   if dview is not None: 
       file_res = dview.map_sync(pre_process_handle, args)                         
       dview.results.clear()       
   else:
       file_res = map(pre_process_handle, args)     
        
   return file_res
   
   