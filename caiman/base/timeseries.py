# -*- coding: utf-8 -*-
"""
Class representing a time series.

    Example of usage

    Parameters
    ----------
    input_arr: np.ndarray
    start_time: time beginning movie
    fr: frame rate
    meta_data: dictionary including any custom meta data


author: Andrea Giovannucci
"""
#%%
import os
import warnings
import numpy as np
import cv2
import h5py
import pylab as plt
import cPickle as cpk
try:
    plt.ion()
except:
    1;

from scipy.io import loadmat,savemat

import tifffile
from tifffile import imsave

if tifffile.__version__ < '0.7.0':
    raise Exception('You need libtiff version >= 0.7.0, run pip install tifffile --upgrade')

#%%
class timeseries(np.ndarray):
    """
    Class representing a time series.

    Example of usage

    Parameters
    ----------
    input_arr: np.ndarray
    fr: frame rate
    start_time: time beginning movie

    meta_data: dictionary including any custom meta data

    """

    def __new__(cls, input_arr, fr=30,start_time=0,file_name=None, meta_data=None):
        #

        if fr is None:
            raise Exception('You need to specify the frame rate')


        obj = np.asarray(input_arr).view(cls)
        # add the new attribute to the created instance

        obj.start_time = np.double(start_time)
        obj.fr = np.double(fr)
        if type(file_name) is list:
            obj.file_name = file_name
        else:
            obj.file_name = [file_name];

        if type(meta_data) is list:
            obj.meta_data = meta_data
        else:
            obj.meta_data = [meta_data];

        return obj

    @property
    def time(self):
        return np.linspace(self.start_time,1/self.fr*self.shape[0],self.shape[0])

    def __array_prepare__(self, out_arr, context=None):
#        print 'In __array_prepare__:'
#        print '   self is %s' % type(self)
#        print '   out_arr is %s' % type(out_arr)
        inputs=context[1];
        frRef=None;
        startRef=None;
        for inp in inputs:
            if type(inp) is timeseries:
                if frRef is None:
                    frRef=inp.fr
                else:
                    if not (frRef-inp.fr) == 0:
                        raise ValueError('Frame rates of input vectors do not match. You cannot perform operations on time series with different frame rates.')
                if startRef is None:
                    startRef=inp.start_time
                else:
                    if not (startRef-inp.start_time) == 0:
                        warnings.warn('start_time of input vectors do not match: ignore if this is what desired.',UserWarning)

        # then just call the parent
        return np.ndarray.__array_prepare__(self, out_arr, context)

    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None: return

        self.start_time = getattr(obj, 'start_time', None)
        self.fr = getattr(obj, 'fr', None)
        self.file_name = getattr(obj, 'file_name', None)
        self.meta_data = getattr(obj, 'meta_data', None)

#    def __array_wrap__(self, out_arr, context=None):
#        print 'In __array_wrap__:'
#        print '   self is %s' % type(self)
#        print '   arr is %s' % type(out_arr)
#        print context
#        # then just call the parent
#        return np.ndarray.__array_wrap__(self, out_arr, context)
#
    def save(self,file_name):
        '''
        Save the timeseries in various formats

        parameters
        ----------
        file_name: name of file. Possible formats are tif, avi, npz and hdf5

        '''
        name,extension = os.path.splitext(file_name)[:2]
        print extension

        if extension == '.tif': # load avi file
    #            raise Exception('not implemented')

            np.clip(self,np.percentile(self,1),np.percentile(self,99.99999),self)
            minn,maxx = np.min(self),np.max(self)
            data = 65536 * (self-minn)/(maxx-minn)
            data = data.astype(np.int32)
            imsave(file_name, self.astype(np.float32))

        elif extension == '.npz':
            np.savez(file_name,input_arr=self, start_time=self.start_time,fr=self.fr,meta_data=self.meta_data,file_name=self.file_name)


        elif extension == '.avi':
            codec=cv2.cv.FOURCC('I','Y','U','V')
            np.clip(self,np.percentile(self,1),np.percentile(self,99),self)
            minn,maxx = np.min(self),np.max(self)
            data = 255 * (self-minn)/(maxx-minn)
            data = data.astype(np.uint8)
            y,x = data[0].shape
            vw = cv2.VideoWriter(file_name, codec, self.fr, (x,y), isColor=True)
            for d in data:
                vw.write(cv2.cvtColor(d, cv2.COLOR_GRAY2BGR))
            vw.release()

        elif extension == '.mat':
            if self.file_name[0] is not None:
                f_name=self.file_name
            else:
                f_name=''
            if self.meta_data[0] is None:
                savemat(file_name,{'input_arr':np.rollaxis(self,axis=0,start=3), 'start_time':self.start_time,'fr':self.fr,'meta_data':[],'file_name':f_name})
            else:
                savemat(file_name,{'input_arr':np.rollaxis(self,axis=0,start=3), 'start_time':self.start_time,'fr':self.fr,'meta_data':self.meta_data,'file_name':f_name})

        elif extension == '.hdf5':
            with h5py.File(file_name, "w") as f:
                dset=f.create_dataset("mov",data=np.asarray(self))
                dset.attrs["fr"]=self.fr
                dset.attrs["start_time"]=self.start_time
                try: 
                    dset.attrs["file_name"]=[a.encode('utf8') for a in self.file_name]
                except:
                    print 'No file name saved'

                dset.attrs["meta_data"]=cpk.dumps(self.meta_data)

        else:
            print extension
            raise Exception('Extension Unknown')


def concatenate(*args, **kwargs):
        """
        Concatenate movies

        Parameters
        ---------------------------
        mov: XMovie object
        """

        obj = []
        frRef = None
        for arg in args:
            for m in arg:
                if issubclass(type(m), timeseries):
                    if frRef is None:
                        obj = m
                        frRef = obj.fr
                    else:
                        obj.__dict__['file_name'].extend(
                                [ls for ls in m.file_name])
                        obj.__dict__['meta_data'].extend(
                                [ls for ls in m.meta_data])
                        if obj.fr != m.fr:
                            raise ValueError('Frame rates of input vectors \
                            do not match. You cannot concatenate movies with \
                            different frame rates.')

        return obj.__class__(np.concatenate(*args, **kwargs), **obj.__dict__)
