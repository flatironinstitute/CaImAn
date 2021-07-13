#!/usr/bin/env python

"""
Class representing a time series.
"""

import cv2
from datetime import datetime
from dateutil.tz import tzlocal
import h5py
import logging
import numpy as np
import os
import pylab as plt
from pynwb import NWBHDF5IO, NWBFile
from pynwb.ophys import TwoPhotonSeries, OpticalChannel
from pynwb.device import Device
import pickle as cpk
from scipy.io import savemat
import tifffile
import warnings

import caiman.paths

try:
    cv2.setNumThreads(0)
except:
    pass

try:
    plt.ion()
except:
    pass


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
                            raise ValueError(
                                'Frame rates of input vectors do not match.'
                                ' You cannot perform operations on time series with different frame rates.')
                    if startRef is None:
                        startRef = inp.start_time
                    else:
                        if not (startRef - inp.start_time) == 0:
                            warnings.warn('start_time of input vectors do not match: ignore if this is desired.',
                                          UserWarning)

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

    def save(self,
             file_name,
             to32=True,
             order='F',
             imagej=False,
             bigtiff=True,
             excitation_lambda=488.0,
             compress=0,
             q_max=99.75,
             q_min=1,
             var_name_hdf5='mov',
             sess_desc='some_description',
             identifier='some identifier',
             imaging_plane_description='some imaging plane description',
             emission_lambda=520.0,
             indicator='OGB-1',
             location='brain',
             starting_time=0.,
             experimenter='Dr Who',
             lab_name=None,
             institution=None,
             experiment_description='Experiment Description',
             session_id='Session ID'):
        """
        Save the timeseries in single precision. Supported formats include
        TIFF, NPZ, AVI, MAT, HDF5/H5, MMAP, and NWB

        Args:
            file_name: str
                name of file. Possible formats are tif, avi, npz, mmap and hdf5

            to32: Bool
                whether to transform to 32 bits

            order: 'F' or 'C'
                C or Fortran order

            var_name_hdf5: str
                Name of hdf5 file subdirectory

            q_max, q_min: float in [0, 100]
                percentile for maximum/minimum clipping value if saving as avi
                (If set to None, no automatic scaling to the dynamic range [0, 255] is performed)

        Raises:
            Exception 'Extension Unknown'

        """
        file_name = caiman.paths.fn_relocated(file_name)
        name, extension = os.path.splitext(file_name)[:2] # name is only used by the memmap saver
        extension = extension.lower()
        logging.debug("Parsing extension " + str(extension))

        if extension in ['.tif', '.tiff', '.btf']:
            with tifffile.TiffWriter(file_name, bigtiff=bigtiff, imagej=imagej) as tif:
                if "%4d%02d%02d" % tuple(map(int, tifffile.__version__.split('.'))) >= '20200813':
                    def foo(i):
                        if i % 200 == 0:
                            logging.debug(str(i) + ' frames saved')
                        curfr = self[i].copy()
                        if to32 and not ('float32' in str(self.dtype)):
                            curfr = curfr.astype(np.float32)
                        return curfr             
                    tif.save([foo(i) for i in range(self.shape[0])], compress=compress)
                else:
                    for i in range(self.shape[0]):
                        if i % 200 == 0:
                            logging.debug(str(i) + ' frames saved')
                        curfr = self[i].copy()
                        if to32 and not ('float32' in str(self.dtype)):
                            curfr = curfr.astype(np.float32)
                        tif.save(curfr, compress=compress)
        elif extension == '.npz':
            if to32 and not ('float32' in str(self.dtype)):
                input_arr = self.astype(np.float32)
            else:
                input_arr = np.array(self)

            np.savez(file_name,
                     input_arr=input_arr,
                     start_time=self.start_time,
                     fr=self.fr,
                     meta_data=self.meta_data,
                     file_name=self.file_name)
        elif extension in ('.avi', '.mkv'):
            codec = None
            try:
                codec = cv2.FOURCC('I', 'Y', 'U', 'V')
            except AttributeError:
                codec = cv2.VideoWriter_fourcc(*'IYUV')
            if q_max is None or q_min is None:
                data = self.astype(np.uint8)
            else:
                if q_max < 100:
                    maxmov = np.nanpercentile(self[::max(1, len(self) // 100)], q_max)
                else:
                    maxmov = np.nanmax(self)
                if q_min > 0:
                    minmov = np.nanpercentile(self[::max(1, len(self) // 100)], q_min)
                else:
                    minmov = np.nanmin(self)
                data = 255 * (self - minmov) / (maxmov - minmov)
                np.clip(data, 0, 255, data)
                data = data.astype(np.uint8)
                
            y, x = data[0].shape
            vw = cv2.VideoWriter(file_name, codec, self.fr, (x, y), isColor=True)
            for d in data:
                vw.write(cv2.cvtColor(d, cv2.COLOR_GRAY2BGR))
            vw.release()

        elif extension == '.mat':
            if self.file_name[0] is not None:
                f_name = self.file_name
            else:
                f_name = ''

            if to32 and not ('float32' in str(self.dtype)):
                input_arr = self.astype(np.float32)
            else:
                input_arr = np.array(self)

            if self.meta_data[0] is None:
                savemat(
                    file_name, {
                        'input_arr': np.rollaxis(input_arr, axis=0, start=3),
                        'start_time': self.start_time,
                        'fr': self.fr,
                        'meta_data': [],
                        'file_name': f_name
                    })
            else:
                savemat(
                    file_name, {
                        'input_arr': np.rollaxis(input_arr, axis=0, start=3),
                        'start_time': self.start_time,
                        'fr': self.fr,
                        'meta_data': self.meta_data,
                        'file_name': f_name
                    })

        elif extension in ('.hdf5', '.h5'):
            with h5py.File(file_name, "w") as f:
                if to32 and not ('float32' in str(self.dtype)):
                    input_arr = self.astype(np.float32)
                else:
                    input_arr = np.array(self)

                dset = f.create_dataset(var_name_hdf5, data=input_arr)
                dset.attrs["fr"] = self.fr
                dset.attrs["start_time"] = self.start_time
                try:
                    dset.attrs["file_name"] = [a.encode('utf8') for a in self.file_name]
                except:
                    logging.warning('No file saved')
                if self.meta_data[0] is not None:
                    logging.debug("Metadata for saved file: " + str(self.meta_data))
                    dset.attrs["meta_data"] = cpk.dumps(self.meta_data)
        elif extension == '.mmap':
            base_name = name

            T = self.shape[0]
            dims = self.shape[1:]
            if to32 and not ('float32' in str(self.dtype)):
                input_arr = self.astype(np.float32)
            else:
                input_arr = np.array(self)

            input_arr = np.transpose(input_arr, list(range(1, len(dims) + 1)) + [0])
            input_arr = np.reshape(input_arr, (np.prod(dims), T), order='F')

            fname_tot = caiman.paths.memmap_frames_filename(base_name, dims, T, order)
            fname_tot = os.path.join(os.path.split(file_name)[0], fname_tot)
            big_mov = np.memmap(fname_tot,
                                mode='w+',
                                dtype=np.float32,
                                shape=(np.uint64(np.prod(dims)), np.uint64(T)),
                                order=order)

            big_mov[:] = np.asarray(input_arr, dtype=np.float32)
            big_mov.flush()
            del big_mov, input_arr
            return fname_tot
        elif extension == '.nwb':
            if to32 and not ('float32' in str(self.dtype)):
                input_arr = self.astype(np.float32)
            else:
                input_arr = np.array(self)
            # Create NWB file
            nwbfile = NWBFile(sess_desc,
                              identifier,
                              datetime.now(tzlocal()),
                              experimenter=experimenter,
                              lab=lab_name,
                              institution=institution,
                              experiment_description=experiment_description,
                              session_id=session_id)
            # Get the device
            device = Device('imaging_device')
            nwbfile.add_device(device)
            # OpticalChannel
            optical_channel = OpticalChannel('OpticalChannel', 'main optical channel', emission_lambda=emission_lambda)
            imaging_plane = nwbfile.create_imaging_plane(name='ImagingPlane',
                                                         optical_channel=optical_channel,
                                                         description=imaging_plane_description,
                                                         device=device,
                                                         excitation_lambda=excitation_lambda,
                                                         imaging_rate=self.fr,
                                                         indicator=indicator,
                                                         location=location)
            # Images
            image_series = TwoPhotonSeries(name=var_name_hdf5,
                                           dimension=self.shape[1:],
                                           data=input_arr,
                                           imaging_plane=imaging_plane,
                                           starting_frame=[0],
                                           starting_time=starting_time,
                                           rate=self.fr)

            nwbfile.add_acquisition(image_series)

            with NWBHDF5IO(file_name, 'w') as io:
                io.write(nwbfile)

            return file_name

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
                    obj.__dict__['file_name'].extend([ls for ls in m.file_name])
                    obj.__dict__['meta_data'].extend([ls for ls in m.meta_data])
                    if obj.fr != m.fr:
                        raise ValueError('Frame rates of input vectors \
                            do not match. You cannot concatenate movies with \
                            different frame rates.')
    try:
        return obj.__class__(np.concatenate(*args, **kwargs), **obj.__dict__)
    except:
        logging.debug('no meta information passed')
        return obj.__class__(np.concatenate(*args, **kwargs))

