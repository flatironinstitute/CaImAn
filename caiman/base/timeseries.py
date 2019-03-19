#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Class representing a time series.

author: Andrea Giovannucci
"""

#%%
import cv2
import h5py
import logging
import numpy as np
import os
import pylab as plt
import pickle as cpk
from scipy.io import savemat
import tifffile
import warnings

try:
    cv2.setNumThreads(0)
except:
    pass

try:
    plt.ion()
except:
    pass

#%%
class timeseries(np.ndarray):
    """
    Class representing a time series.
    """

    def __new__(cls, input_arr, fr=30, start_time=0, file_name=None, meta_data=None):
        """
            Class representing a time series.

            Example of usage

            Args:
                input_arr: np.ndarray

                fr: frame rate

                start_time: time beginning movie

                meta_data: dictionary including any custom meta data

            Raises:
                Exception 'You need to specify the frame rate'
            """
        if fr is None:
            raise Exception('You need to specify the frame rate')

        obj = np.asarray(input_arr).view(cls)
        # add the new attribute to the created instance

        obj.start_time = np.double(start_time)
        obj.fr = np.double(fr)
        if type(file_name) is list:
            obj.file_name = file_name
        else:
            obj.file_name = [file_name]

        if type(meta_data) is list:
            obj.meta_data = meta_data
        else:
            obj.meta_data = [meta_data]

        return obj

    @property
    def time(self):
        return np.linspace(self.start_time, 1 / self.fr * self.shape[0], self.shape[0])

    def __array_prepare__(self, out_arr, context=None):
        # todo: todocument

        frRef = None
        startRef = None
        if context is not None:
            inputs = context[1]
            for inp in inputs:
                if type(inp) is timeseries:
                    if frRef is None:
                        frRef = inp.fr
                    else:
                        if not (frRef - inp.fr) == 0:
                            raise ValueError('Frame rates of input vectors do not match.'
                                             ' You cannot perform operations on time series with different frame rates.')
                    if startRef is None:
                        startRef = inp.start_time
                    else:
                        if not (startRef - inp.start_time) == 0:
                            warnings.warn(
                                'start_time of input vectors do not match: ignore if this is what desired.', UserWarning)

        # then just call the parent
        return np.ndarray.__array_prepare__(self, out_arr, context)

    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None:
            return

        self.start_time = getattr(obj, 'start_time', None)
        self.fr = getattr(obj, 'fr', None)
        self.file_name = getattr(obj, 'file_name', None)
        self.meta_data = getattr(obj, 'meta_data', None)

    def save(self, file_name, to32=True, order='F',imagej=False, bigtiff=True, software='CaImAn', compress=0):
        """
        Save the timeseries in various formats

        Args:
            file_name: str
                name of file. Possible formats are tif, avi, npz, mmap and hdf5

            to32: Bool
                whether to transform to 32 bits

            order: 'F' or 'C'
                C or Fortran order

        Raises:
            Exception 'Extension Unknown'

        """
        name, extension = os.path.splitext(file_name)[:2]
        extension = extension.lower()
        logging.debug("Parsing extension " + str(extension))

        if extension == '.tif':  # load avi file

            with tifffile.TiffWriter(file_name, bigtiff=bigtiff, imagej=imagej) as tif:


                for i in range(self.shape[0]):
                    if i % 200 == 0:
                        logging.debug(str(i) + ' frames saved')

                    curfr = self[i].copy()
                    if to32 and not('float32' in str(self.dtype)):
                         curfr = curfr.astype(np.float32)

                    tif.save(curfr, compress=compress)



        elif extension == '.npz':
            if to32 and not('float32' in str(self.dtype)):
                input_arr = self.astype(np.float32)
            else:
                input_arr = np.array(self)

            np.savez(file_name, input_arr=input_arr, start_time=self.start_time,
                     fr=self.fr, meta_data=self.meta_data, file_name=self.file_name)

        elif extension == '.avi':
            codec = None
            try:
                codec = cv2.FOURCC('I', 'Y', 'U', 'V')
            except AttributeError:
                codec = cv2.VideoWriter_fourcc(*'IYUV')
            np.clip(self, np.percentile(self, 1),
                    np.percentile(self, 99), self)
            minn, maxx = np.min(self), np.max(self)
            data = 255 * (self - minn) / (maxx - minn)
            data = data.astype(np.uint8)
            y, x = data[0].shape
            vw = cv2.VideoWriter(file_name, codec, self.fr,
                                 (x, y), isColor=True)
            for d in data:
                vw.write(cv2.cvtColor(d, cv2.COLOR_GRAY2BGR))
            vw.release()

        elif extension == '.mat':
            if self.file_name[0] is not None:
                f_name = self.file_name
            else:
                f_name = ''

            if to32 and not('float32' in str(self.dtype)):
                input_arr = self.astype(np.float32)
            else:
                input_arr = np.array(self)

            if self.meta_data[0] is None:
                savemat(file_name, {'input_arr': np.rollaxis(
                    input_arr, axis=0, start=3), 'start_time': self.start_time, 'fr': self.fr, 'meta_data': [], 'file_name': f_name})
            else:
                savemat(file_name, {'input_arr': np.rollaxis(
                    input_arr, axis=0, start=3), 'start_time': self.start_time, 'fr': self.fr, 'meta_data': self.meta_data, 'file_name': f_name})

        elif extension in ('.hdf5', '.h5'):
            with h5py.File(file_name, "w") as f:
                if to32 and not('float32' in str(self.dtype)):
                    input_arr = self.astype(np.float32)
                else:
                    input_arr = np.array(self)

                dset = f.create_dataset("mov", data=input_arr)
                dset.attrs["fr"] = self.fr
                dset.attrs["start_time"] = self.start_time
                try:
                    dset.attrs["file_name"] = [
                        a.encode('utf8') for a in self.file_name]
                except:
                    logging.warning('No file saved')
                if self.meta_data[0] is not None:
                    logging.debug("Metadata for saved file: " + str(self.meta_data))
                    dset.attrs["meta_data"] = cpk.dumps(self.meta_data)

        elif extension == '.mmap':
            base_name = name

            T = self.shape[0]
            dims = self.shape[1:]
            if to32 and not('float32' in str(self.dtype)):
                input_arr = self.astype(np.float32)
            else:
                input_arr = np.array(self)

            input_arr = np.transpose(input_arr, list(range(1, len(dims) + 1)) + [0])
            input_arr = np.reshape(input_arr, (np.prod(dims), T), order='F')

            fname_tot = base_name + '_d1_' + str(dims[0]) + '_d2_' + str(dims[1]) + '_d3_' + str(
                1 if len(dims) == 2 else dims[2]) + '_order_' + str(order) + '_frames_' + str(T) + '_.mmap'
            fname_tot = os.path.join(os.path.split(file_name)[0], fname_tot)
            big_mov = np.memmap(fname_tot, mode='w+', dtype=np.float32,
                                shape=(np.uint64(np.prod(dims)), np.uint64(T)), order=order)

            big_mov[:] = np.asarray(input_arr, dtype=np.float32)
            big_mov.flush()
            del big_mov, input_arr
            return fname_tot

        else:
            logging.error("Extension " + str(extension) + " unknown")
            raise Exception('Extension Unknown')


def concatenate(*args, **kwargs):
    """
    Concatenate movies

    Args:
        mov: XMovie object
    """
    # todo: todocument return

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
    try:
        return obj.__class__(np.concatenate(*args, **kwargs), **obj.__dict__)
    except:
        logging.debug('no meta information passed')
        return obj.__class__(np.concatenate(*args, **kwargs))
