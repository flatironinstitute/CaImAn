from time import time
import numpy as np
import pylab as pl
import scipy.io as sio
from scipy.sparse import coo_matrix
from greedyROI2d import greedyROI2d
from arpfit import arpfit
from sklearn.decomposition import ProjectedGradientNMF
from update_spatial_components import update_spatial_components
from update_temporal_components import update_temporal_components
from merge_rois import mergeROIS
from utilities import *
from sklearn.base import  BaseEstimator, TransformerMixin
 

class sNMF(BaseEstimator, TransformerMixin):
    """CONSTRAINED Non-Negative Matrix Factorization (CNMF)
    Find two non-negative matrices (A, C) whose product approximates the non-
    negative matrix F, under the constraints that C represents an autoregressive process of order p. F is a 3D matrix (x,y,t) representing fluorescence movie. 

        S = C*G^T
        F = A*C + b*f 
        Y = F + E,

    The optimization is carried out alternating solutions to the following two problems. 
    
    1) Find A,b by solving the optimization problem::

        minimize ||A||_1    

    Subject to::
    
        A,b>=0
        ||Y(i,:)−A(i,:)C−b(i)f^T|| <= \sigma_i * \sqrt(T), i=1,2,...,d
                
    Where

        ||A||_1 = \sum_{i,j} |a_{ij}| (L-1 norm)
        
      
    and 
    
    2) Find C,f by solving the optimization problem 
    
        minimize C*G^T (minimize number of spikes)
        
      
    """
    def __init__(self,  mov, n_neurons=100 , gSig=(5,5) , gSiz=(5,5), n_bakground_comps=1 , n_iter_NMF=1 , 
                 search_method='ellipse', min_size_ellipse=3 , max_size_ellipse=8 , expansion_ellipse=3 , dilation_element=None , #search location  
                 temporal_deconv_method = 'constrained_foopsi', re_estimate_g=True , temporal_iterations=2, #UPDATING TEMPORAL COMPONENTS
                 contr_deconv_method = 'spgl1', bas_nonneg = 1, fudge_factor=.9999, resparse=0,  #CONSTRAINED DECONVOLUTION 
                 merge_thr=0.85 
                 ):
                         
        self.mov = mov
        self.Y = np.transpose(np.asarray(self.mov),(1,2,0))
        
        if dilation_element is None and search_method == 'dilate':             
            dilation_element=strel('disk',4,0);
        
    def run(self):
        self.init_rois()
        self.update_spatial()
        self.update_temporal()
        #self.merge()
        self.order()

    def init_rois(self, n_components=100, show=False):
        Ain,Cin,center = greedyROI2d(self.Y, nr=n_components, gSig=[2,2], gSiz=[7,7], use_median=False)
        Cn = np.mean(self.Y, axis=-1)

        if show:
            pl1 = pl.imshow(Cn,interpolation='none')
            pl.colorbar()
            pl.scatter(x=center[:,1], y=center[:,0], c='m', s=40)
            pl.axis((-0.5,self.Y.shape[1]-0.5,-0.5,self.Y.shape[0]-0.5))
            pl.gca().invert_yaxis()

        active_pixels = np.squeeze(np.nonzero(np.sum(Ain,axis=1)))
        Yr = np.reshape(self.Y,(self.Y.shape[0]*self.Y.shape[1],self.Y.shape[2]),order='F')
        P = arpfit(Yr, p=2, pixels=active_pixels)
        Y_res = Yr - np.dot(Ain,Cin)
        model = ProjectedGradientNMF(n_components=1, init='random', random_state=0)
        model.fit(np.maximum(Y_res,0))
        fin = model.components_.squeeze()
        
        self.Yr,self.Cin,self.fin,self.Ain,self.P,self.Cn = Yr,Cin,fin,Ain,P,Cn

    def update_spatial(self):
        self.A,self.b = update_spatial_components(self.Yr, self.Cin, self.fin, self.Ain, d1=self.Y.shape[0], d2=self.Y.shape[1], sn=self.P['sn'])

    def update_temporal(self):
        self.C,self.f,self.Y_res,self.Pnew = update_temporal_components(self.Yr,self.A,self.b,self.Cin,self.fin,ITER=2,deconv_method = 'spgl1')

    def merge(self):
        A_m,C_m,nr_m,merged_ROIs,P_m=mergeROIS(self.Y_res,self.A.tocsc(),self.b,np.array(self.C),self.f,self.Y.shape[0],self.Y.shape[1],self.Pnew,sn=self.P['sn'])

    def order(self, show=True):
        self.A_or, self.C_or, self.srt = order_components(self.A,self.C)
        if show:
            crd = plot_contours(coo_matrix(self.A_or[:,::-1]),self.Cn,thr=0.9)

if __name__ == '__main__':
    from pyfluo import Movie
    mov = Movie('high_npil_22.tif', Ts=0.064)
    snmf = sNMF(mov)
    snmf.run()
