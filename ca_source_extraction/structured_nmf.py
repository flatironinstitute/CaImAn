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

class sNMF(object):
    def __init__(self, mov):
        self.mov = mov
        self.Y = np.transpose(np.asarray(self.mov),(1,2,0))

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
