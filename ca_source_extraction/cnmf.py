# -*- coding: utf-8 -*-
"""
Constrained Nonnegative Matrix Factorization

Created on Fri Aug 26 15:44:32 2016

@author: agiovann
 

"""

import numpy as np
from utilities import local_correlations, CNMFSetParms, order_components
from pre_processing import preprocess_data
from initialization import initialize_components
from merging import merge_components
from spatial import update_spatial_components
from temporal import update_temporal_components

class CNMF(object):
   """
   Source extraction using constrained non-negative matrix factorization.
   """
   def __init__(self, n_processes, k=5, gSig=[4,4], merge_thresh=0.8 , p=2, dview=None, Ain=None, ssub=2, tsub=2,p_ssub=1, p_tsub=1, method_init= 'greedy_roi'):
      """ 
      Constructor of the CNMF method
      """
      self.k = k #number of neurons expected per patch
      self.gSig=gSig # expected half size of neurons
      self.merge_thresh=merge_thresh # merging threshold, max correlation allowed
      self.p=p #order of the autoregressive system
      self.dview=dview
      self.Ain=Ain
      self.ssub=ssub
      self.tsub=tsub
      self.p_ssub=p_ssub
      self.p_tsub=p_tsub
      self.method_init= method_init
      self.n_processes=n_processes
      self.A=None
      self.C=None
      self.S=None
      self.b=None
      self.f=None
      
   def fit(self, images):
       """
       This method uses the cnmf algorithm to find sources in data.
       
       Parameters
       ----------
       images : mapped np.ndarray of shape (t,x,y) containing the images that vary over time.
    	
       Returns
       --------
       self 
    
       """       
       T,d1,d2=images.shape
       dims=(d1,d2)
       Yr=images.reshape([T,np.prod(dims)],order='F').T
       Y=np.transpose(images,[1,2,0])
       print  T,d1,d2
       
       options = CNMFSetParms(Y,self.n_processes,p=self.p,gSig=self.gSig,K=self.k,ssub=self.ssub,tsub=self.tsub,\
                                       p_ssub=self.p_ssub, p_tsub=self.p_tsub, method_init= self.method_init)
   
        #	Cn = local_correlations(Y)
    
       Yr,sn,g,psx = preprocess_data(Yr,dview=self.dview,**options['preprocess_params'])
       
       do_merge=True
       if self.Ain is None:
           Ain,Cin , b_in, f_in, center=initialize_components(Y, normalize=True, **options['init_params'])                                                    
           A,b,Cin = update_spatial_components(Yr, Cin, f_in, Ain, sn=sn, dview=self.dview,**options['spatial_params'])

       else:
           A,b,Cin,f_in = update_spatial_components(Yr, C=None, f=None, A_in=self.Ain.astype(np.bool), sn=sn,dview=self.dview, **options['spatial_params'])
           do_merge=False
    	    	
       options['temporal_params']['p'] = 0 # set this to zero for fast updating without deconvolution
       
       C,f,S,bl,c1,neurons_sn,g,YrA = update_temporal_components(Yr,A,b,Cin,f_in,dview=self.dview,bl=None,c1=None,sn=None,g=None,**options['temporal_params'])             
    	
       if do_merge:
           A,C,nr,merged_ROIs,S,bl,c1,sn1,g1=merge_components(Yr,A,b,C,f,S,sn,options['temporal_params'], options['spatial_params'],dview=self.dview, bl=bl, c1=c1, sn=neurons_sn, g=g, thr=self.merge_thresh, mx=50, fast_merge = True)
       print A.shape    
    
       A,b,C = update_spatial_components(Yr, C, f, A, sn=sn,dview=self.dview, **options['spatial_params'])
       options['temporal_params']['p'] = self.p # set it back to original value to perform full deconvolution
       
       C,f,S,bl,c1,neurons_sn,g1,YrA = update_temporal_components(Yr,A,b,C,f,dview=self.dview,bl=None,c1=None,sn=None,g=None,**options['temporal_params'])
    
       self.A=A
       self.C=C
       self.b=b
       self.f=f
       self.YrA=YrA	
       self.options=options
       
       return self