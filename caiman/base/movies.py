#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Suite of functions that help manage movie data

Contains the movie class.

"""

# \package caiman/dource_ectraction/cnmf
# \version   1.0
# \copyright GNU General Public License v2.0
# \date Created on Tue Jun 30 20:56:07 2015 , Updated on Fri Aug 19 17:30:11 2016

from builtins import str
from builtins import range
from past.utils import old_div

import cv2
from functools import partial
import h5py
import logging
from matplotlib import animation
import numpy as np
import os
from PIL import Image  # $ pip install pillow
import pylab as pl
import scipy.ndimage
import scipy
from scipy.io import loadmat
from skimage.transform import warp, AffineTransform
from skimage.feature import match_template
import sklearn
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF, incremental_pca, FastICA
from sklearn.metrics.pairwise import euclidean_distances
import sys
import tifffile
from tqdm import tqdm
from typing import Any, Dict, List, Tuple, Union
import warnings
from zipfile import ZipFile

import caiman as cm

from . import timeseries

try:
    cv2.setNumThreads(0)
except:
    pass

try:
    import sima
    HAS_SIMA = True
except ImportError:
    HAS_SIMA = False

from . import timeseries as ts
from .traces import trace

from ..mmapping import load_memmap
from ..utils import visualization
from .. import summary_images as si
from ..motion_correction import apply_shift_online, motion_correct_online


class movie(ts.timeseries):
    """
    Class representing a movie. This class subclasses timeseries,
    that in turn subclasses ndarray

    movie(input_arr, fr=None,start_time=0,file_name=None, meta_data=None)

    Example of usage:
        input_arr = 3d ndarray
        fr=33; # 33 Hz
        start_time=0
        m=movie(input_arr, start_time=0,fr=33);

    See https://docs.scipy.org/doc/numpy/user/basics.subclassing.html for
    notes on objects that are descended from ndarray
    """

    def __new__(cls, input_arr, **kwargs):
        """
        Args:
            input_arr:  np.ndarray, 3D, (time,height,width)

            fr: frame rate

            start_time: time beginning movie, if None it is assumed 0

            meta_data: dictionary including any custom meta data

            file_name: name associated with the file (e.g. path to the original file)
        """
        if isinstance(input_arr, movie):
            return input_arr

        if (type(input_arr) is np.ndarray) or \
           (type(input_arr) is h5py._hl.dataset.Dataset) or \
           ('mmap' in str(type(input_arr))) or \
           ('tifffile' in str(type(input_arr))):
            return super().__new__(cls, input_arr, **kwargs)  
            #return super(movie, cls).__new__(cls, input_arr, **kwargs)
        else:
            raise Exception('Input must be an ndarray, use load instead!')

    def apply_shifts_online(self, xy_shifts, save_base_name=None):
        # This function is unused.
        # todo: todocument

        if save_base_name is None:
            return movie(apply_shift_online(self, xy_shifts, save_base_name=save_base_name), fr=self.fr)
        else:
            return apply_shift_online(self, xy_shifts, save_base_name=save_base_name)

    def calc_min(self) -> 'movie':
        # todo: todocument

        tmp = []
        bins = np.linspace(0, self.shape[0], 10).round(0)
        for i in range(9):
            tmp.append(np.nanmin(self[np.int(bins[i]):np.int(bins[i + 1]), :, :]).tolist() + 1)
        minval = np.ndarray(1)
        minval[0] = np.nanmin(tmp)
        return movie(input_arr=minval)

    def motion_correct(self,
                       max_shift_w=5,
                       max_shift_h=5,
                       num_frames_template=None,
                       template=None,
                       method: str = 'opencv',
                       remove_blanks: bool = False,
                       interpolation: str = 'cubic') -> Tuple[Any, Tuple, Any, Any]:
        """
        Extract shifts and motion corrected movie automatically,

        for more control consider the functions extract_shifts and apply_shifts
        Disclaimer, it might change the object itself.

        Args:
            max_shift_w,max_shift_h: maximum pixel shifts allowed when correcting
                                     in the width and height direction

            template: if a good template for frame by frame correlation exists
                      it can be passed. If None it is automatically computed

            method: depends on what is installed 'opencv' or 'skimage'. 'skimage'
                    is an order of magnitude slower

            num_frames_template: if only a subset of the movies needs to be loaded
                                 for efficiency/speed reasons


        Returns:
            self: motion corected movie, it might change the object itself

            shifts : tuple, contains x & y shifts and correlation with template

            xcorrs: cross correlation of the movies with the template

            template: the computed template
        """

        if template is None:   # if template is not provided it is created
            if num_frames_template is None:
                num_frames_template = old_div(10e7, (self.shape[1] * self.shape[2]))

            frames_to_skip = int(np.maximum(1, old_div(self.shape[0], num_frames_template)))

            # sometimes it is convenient to only consider a subset of the
            # movie when computing the median
            submov = self[::frames_to_skip, :].copy()
            templ = submov.bin_median()                                        # create template with portion of movie
            shifts, xcorrs = submov.extract_shifts(max_shift_w=max_shift_w,
                                                   max_shift_h=max_shift_h,
                                                   template=templ,
                                                   method=method)
            submov.apply_shifts(shifts, interpolation=interpolation, method=method)
            template = submov.bin_median()
            del submov
            m = self.copy()
            shifts, xcorrs = m.extract_shifts(max_shift_w=max_shift_w,
                                              max_shift_h=max_shift_h,
                                              template=template,
                                              method=method)
            m = m.apply_shifts(shifts, interpolation=interpolation, method=method)
            template = (m.bin_median())
            del m
        else:
            template = template - np.percentile(template, 8)

        # now use the good template to correct
        shifts, xcorrs = self.extract_shifts(max_shift_w=max_shift_w,
                                             max_shift_h=max_shift_h,
                                             template=template,
                                             method=method)
        self = self.apply_shifts(shifts, interpolation=interpolation, method=method)

        if remove_blanks:
            max_h, max_w = np.max(shifts, axis=0)
            min_h, min_w = np.min(shifts, axis=0)
            self.crop(crop_top=max_h,
                      crop_bottom=-min_h + 1,
                      crop_left=max_w,
                      crop_right=-min_w,
                      crop_begin=0,
                      crop_end=0)

        return self, shifts, xcorrs, template

    def motion_correct_3d(self,
                          max_shift_z=5,
                          max_shift_w=5,
                          max_shift_h=5,
                          num_frames_template=None,
                          template=None,
                          method='opencv',
                          remove_blanks=False,
                          interpolation='cubic'):
        """
        Extract shifts and motion corrected movie automatically,

        for more control consider the functions extract_shifts and apply_shifts
        Disclaimer, it might change the object itself.

        Args:
            max_shift,z,max_shift_w,max_shift_h: maximum pixel shifts allowed when 
                    correcting in the axial, width, and height directions

            template: if a good template for frame by frame correlation exists
                      it can be passed. If None it is automatically computed

            method: depends on what is installed 'opencv' or 'skimage'. 'skimage'
                    is an order of magnitude slower

            num_frames_template: if only a subset of the movies needs to be loaded
                                 for efficiency/speed reasons


        Returns:
            self: motion corected movie, it might change the object itself

            shifts : tuple, contains x, y, and z shifts and correlation with template

            xcorrs: cross correlation of the movies with the template

            template: the computed template
        """

        if template is None:   # if template is not provided it is created
            if num_frames_template is None:
                num_frames_template = old_div(10e7, (self.shape[1] * self.shape[2]))

            frames_to_skip = int(np.maximum(1, old_div(self.shape[0], num_frames_template)))

            # sometimes it is convenient to only consider a subset of the
            # movie when computing the median
            submov = self[::frames_to_skip, :].copy()
            templ = submov.bin_median_3d()                                     # create template with portion of movie
            shifts, xcorrs = submov.extract_shifts_3d(
                max_shift_z=max_shift_z,                                       # NOTE: extract_shifts_3d has not been implemented yet - use skimage
                max_shift_w=max_shift_w,
                max_shift_h=max_shift_h,
                template=templ,
                method=method)
            submov.apply_shifts_3d(                                            # NOTE: apply_shifts_3d has not been implemented yet
                shifts, interpolation=interpolation, method=method)
            template = submov.bin_median_3d()
            del submov
            m = self.copy()
            shifts, xcorrs = m.extract_shifts_3d(max_shift_z=max_shift_z,
                                                 max_shift_w=max_shift_w,
                                                 max_shift_h=max_shift_h,
                                                 template=template,
                                                 method=method)
            m = m.apply_shifts_3d(shifts, interpolation=interpolation, method=method)
            template = (m.bin_median_3d())
            del m
        else:
            template = template - np.percentile(template, 8)

        # now use the good template to correct
        shifts, xcorrs = self.extract_shifts_3d(max_shift_z=max_shift_z,
                                                max_shift_w=max_shift_w,
                                                max_shift_h=max_shift_h,
                                                template=template,
                                                method=method)
        self = self.apply_shifts_3d(shifts, interpolation=interpolation, method=method)

        if remove_blanks:
            max_z, max_h, max_w = np.max(shifts, axis=0)
            min_z, min_h, min_w = np.min(shifts, axis=0)
            self.crop(
                crop_top=max_z,
                crop_bottom=min_z,             # NOTE: edge boundaries for z dimension need to be tested
                crop_left=max_h,
                crop_right=-min_h + 1,
                crop_begin=max_w,
                crop_end=-min_w)

        return self, shifts, xcorrs, template

    def bin_median(self, window: int = 10) -> np.ndarray:
        """ compute median of 3D array in along axis o by binning values

        Args:
            mat: ndarray
                input 3D matrix, time along first dimension

            window: int
                number of frames in a bin

        Returns:
            img:
                median image

        """
        T, d1, d2 = np.shape(self)
        num_windows = np.int(old_div(T, window))
        num_frames = num_windows * window
        return np.nanmedian(np.nanmean(np.reshape(self[:num_frames], (window, num_windows, d1, d2)), axis=0), axis=0)

    def bin_median_3d(self, window=10):
        """ compute median of 4D array in along axis o by binning values

        Args:
            mat: ndarray
                input 4D matrix, (T, h, w, z)

            window: int
                number of frames in a bin

        Returns:
            img:
                median image

        """
        T, d1, d2, d3 = np.shape(self)
        num_windows = np.int(old_div(T, window))
        num_frames = num_windows * window
        return np.nanmedian(np.nanmean(np.reshape(self[:num_frames], (window, num_windows, d1, d2, d3)), axis=0),
                            axis=0)

    def extract_shifts(self, max_shift_w: int = 5, max_shift_h: int = 5, template=None,
                       method: str = 'opencv') -> Tuple[List, List]:
        """
        Performs motion correction using the opencv matchtemplate function. At every iteration a template is built by taking the median of all frames and then used to align the other frames.

        Args:
            max_shift_w,max_shift_h: maximum pixel shifts allowed when correcting in the width and height direction

            template: if a good template for frame by frame correlation is available it can be passed. If None it is automatically computed

            method: depends on what is installed 'opencv' or 'skimage'. 'skimage' is an order of magnitude slower

        Returns:
            shifts : tuple, contains shifts in x and y and correlation with template

            xcorrs: cross correlation of the movies with the template

        Raises:
            Exception 'Unknown motion correction method!'

        """
        min_val = np.percentile(self, 1)
        if min_val < -0.1:
            logging.debug("min_val in extract_shifts: " + str(min_val))
            logging.warning('Movie average is negative. Removing 1st percentile.')
            self = self - min_val
        else:
            min_val = 0

        if type(self[0, 0, 0]) is not np.float32:
            warnings.warn('Casting the array to float32')
            self = np.asanyarray(self, dtype=np.float32)

        _, h_i, w_i = self.shape

        ms_w = max_shift_w
        ms_h = max_shift_h

        if template is None:
            template = np.median(self, axis=0)
        else:
            if np.percentile(template, 8) < -0.1:
                logging.warning('Movie average is negative. Removing 1st percentile.')
                template = template - np.percentile(template, 1)

        template = template[ms_h:h_i - ms_h, ms_w:w_i - ms_w].astype(np.float32)

        #% run algorithm, press q to stop it
        shifts = []    # store the amount of shift in each frame
        xcorrs = []

        for i, frame in enumerate(self):
            if i % 100 == 99:
                logging.debug("Frame %i" % (i + 1))
            if method == 'opencv':
                res = cv2.matchTemplate(frame, template, cv2.TM_CCORR_NORMED)
                top_left = cv2.minMaxLoc(res)[3]
            elif method == 'skimage':
                res = match_template(frame, template)
                top_left = np.unravel_index(np.argmax(res), res.shape)
                top_left = top_left[::-1]
            else:
                raise Exception('Unknown motion correction method!')
            avg_corr = np.mean(res)
            sh_y, sh_x = top_left

            if (0 < top_left[1] < 2 * ms_h - 1) & (0 < top_left[0] < 2 * ms_w - 1):
                # if max is internal, check for subpixel shift using gaussian
                # peak registration
                log_xm1_y = np.log(res[sh_x - 1, sh_y])
                log_xp1_y = np.log(res[sh_x + 1, sh_y])
                log_x_ym1 = np.log(res[sh_x, sh_y - 1])
                log_x_yp1 = np.log(res[sh_x, sh_y + 1])
                four_log_xy = 4 * np.log(res[sh_x, sh_y])

                sh_x_n = -(sh_x - ms_h + old_div((log_xm1_y - log_xp1_y),
                                                 (2 * log_xm1_y - four_log_xy + 2 * log_xp1_y)))
                sh_y_n = -(sh_y - ms_w + old_div((log_x_ym1 - log_x_yp1),
                                                 (2 * log_x_ym1 - four_log_xy + 2 * log_x_yp1)))
            else:
                sh_x_n = -(sh_x - ms_h)
                sh_y_n = -(sh_y - ms_w)

            shifts.append([sh_x_n, sh_y_n])
            xcorrs.append([avg_corr])

        self = self + min_val

        return (shifts, xcorrs)

    def apply_shifts(self, shifts, interpolation: str = 'linear', method: str = 'opencv', remove_blanks: bool = False):
        """
        Apply precomputed shifts to a movie, using subpixels adjustment (cv2.INTER_CUBIC function)

        Args:
            shifts: array of tuples representing x and y shifts for each frame

            interpolation: 'linear', 'cubic', 'nearest' or cvs.INTER_XXX

            method: (undocumented)

            remove_blanks: (undocumented)

        Returns:
            self

        Raise:
            Exception 'Interpolation method not available'

            Exception 'Method not defined'
        """
        if type(self[0, 0, 0]) is not np.float32:
            warnings.warn('Casting the array to float32')
            self = np.asanyarray(self, dtype=np.float32)

        if interpolation == 'cubic':
            if method == 'opencv':
                interpolation = cv2.INTER_CUBIC
            else:
                interpolation = 3
            logging.debug('cubic interpolation')

        elif interpolation == 'nearest':
            if method == 'opencv':
                interpolation = cv2.INTER_NEAREST
            else:
                interpolation = 0
            logging.debug('nearest interpolation')

        elif interpolation == 'linear':
            if method == 'opencv':
                interpolation = cv2.INTER_LINEAR
            else:
                interpolation = 1
            logging.debug('linear interpolation')
        elif interpolation == 'area':
            if method == 'opencv':
                interpolation = cv2.INTER_AREA
            else:
                raise Exception('Method not defined')
            logging.debug('area interpolation')
        elif interpolation == 'lanczos4':
            if method == 'opencv':
                interpolation = cv2.INTER_LANCZOS4
            else:
                interpolation = 4
            logging.debug('lanczos/biquartic interpolation')
        else:
            raise Exception('Interpolation method not available')

        _, h, w = self.shape
        for i, frame in enumerate(self):
            if i % 100 == 99:
                logging.debug("Frame %i" % (i + 1))

            sh_x_n, sh_y_n = shifts[i]

            if method == 'opencv':
                M = np.float32([[1, 0, sh_y_n], [0, 1, sh_x_n]])
                min_, max_ = np.min(frame), np.max(frame)
                self[i] = np.clip(cv2.warpAffine(frame, M, (w, h), flags=interpolation, borderMode=cv2.BORDER_REFLECT),
                                  min_, max_)

            elif method == 'skimage':

                tform = AffineTransform(translation=(-sh_y_n, -sh_x_n))
                self[i] = warp(frame, tform, preserve_range=True, order=interpolation)

            else:
                raise Exception('Unknown shift application method')

        if remove_blanks:
            max_h, max_w = np.max(shifts, axis=0)
            min_h, min_w = np.min(shifts, axis=0)
            self.crop(crop_top=max_h,
                      crop_bottom=-min_h + 1,
                      crop_left=max_w,
                      crop_right=-min_w,
                      crop_begin=0,
                      crop_end=0)

        return self

    def debleach(self):
        """ Debleach by fiting a model to the median intensity.
        """
        #todo: todocument
        if type(self[0, 0, 0]) is not np.float32:
            warnings.warn('Casting the array to float32')
            self = np.asanyarray(self, dtype=np.float32)

        t, _, _ = self.shape
        x = np.arange(t)
        y = np.median(self.reshape(t, -1), axis=1)

        def expf(x, a, b, c):
            return a * np.exp(-b * x) + c

        def linf(x, a, b):
            return a * x + b

        try:
            p0: Tuple = (y[0] - y[-1], 1e-6, y[-1])
            popt, _ = scipy.optimize.curve_fit(expf, x, y, p0=p0)
            y_fit = expf(x, *popt)
        except:
            p0 = (old_div(float(y[-1] - y[0]), float(x[-1] - x[0])), y[0])
            popt, _ = scipy.optimize.curve_fit(linf, x, y, p0=p0)
            y_fit = linf(x, *popt)

        norm = y_fit - np.median(y[:])
        for frame in range(t):
            self[frame, :, :] = self[frame, :, :] - norm[frame]

        return self

    def crop(self, crop_top=0, crop_bottom=0, crop_left=0, crop_right=0, crop_begin=0, crop_end=0) -> None:
        """
        Crop movie (inline)

        Args:
            crop_top/crop_bottom/crop_left,crop_right: (undocumented)

            crop_begin/crop_end: (undocumented)
        """
        t, h, w = self.shape
        self[:, :, :] = self[crop_begin:t - crop_end, crop_top:h - crop_bottom, crop_left:w - crop_right]
    

           
    
    def removeBL(self, windowSize:int=100, quantilMin:int=8, in_place:bool=False, returnBL:bool=False):                   
        """
        Remove baseline from movie using percentiles over a window
        Args:
            windowSize: int
                window size over which to compute the baseline (the larger the faster the algorithm and the less granular

            quantilMin: float
                percentil to be used as baseline value
            in_place: bool
                update movie in place
            returnBL:
                return only baseline

        Return:
             movie without baseline or baseline itself (returnBL=True)
        """
        pixs = self.to2DPixelxTime()
        iter_win = rolling_window(pixs,windowSize,windowSize)
        myperc = partial(np.percentile, q=quantilMin, axis=-1)
        res = np.array(list(map(myperc,iter_win))).T
        if returnBL:
                return cm.movie(cv2.resize(res,pixs.shape[::-1]),fr=self.fr).to3DFromPixelxTime(self.shape)           
        if (not in_place):            
            return (pixs-cv2.resize(res,pixs.shape[::-1])).to3DFromPixelxTime(self.shape)
        else:
            self -= cm.movie(cv2.resize(res,pixs.shape[::-1]),fr=self.fr).to3DFromPixelxTime(self.shape) 
            return self
            
        
        
    
    def to2DPixelxTime(self, order='F'):
        """
        Transform 3D movie into 2D
        """
        return self.transpose([2,1,0]).reshape((-1,self.shape[0]),order=order)  


    def to3DFromPixelxTime(self, shape, order='F'):
        """
            Transform 2D movie into 3D
        """
        return to_3D(self,shape[::-1],order=order).transpose([2,1,0])
    

        

    def computeDFF(self, secsWindow: int = 5, quantilMin: int = 8, method: str = 'only_baseline',
                   order: str = 'F') -> Tuple[Any, Any]:
        """
        compute the DFF of the movie or remove baseline

        In order to compute the baseline frames are binned according to the window length parameter
        and then the intermediate values are interpolated.

        Args:
            secsWindow: length of the windows used to compute the quantile

            quantilMin : value of the quantile

            method='only_baseline','delta_f_over_f','delta_f_over_sqrt_f'

        Returns:
            self: DF or DF/F or DF/sqrt(F) movies

            movBL=baseline movie

        Raises:
            Exception 'Unknown method'
        """

        logging.debug("computing minimum ...")
        sys.stdout.flush()
        if np.min(self) <= 0 and method != 'only_baseline':
            raise ValueError("All pixels must be positive")

        numFrames, linePerFrame, pixPerLine = np.shape(self)
        downsampfact = int(secsWindow * self.fr)
        logging.debug(f"Downsample factor: {downsampfact}")
        elm_missing = int(np.ceil(numFrames * 1.0 / downsampfact) * downsampfact - numFrames)
        padbefore = int(np.floor(old_div(elm_missing, 2.0)))
        padafter = int(np.ceil(old_div(elm_missing, 2.0)))

        logging.debug('Initial Size Image:' + np.str(np.shape(self)))
        sys.stdout.flush()
        mov_out = movie(np.pad(self.astype(np.float32), ((padbefore, padafter), (0, 0), (0, 0)), mode='reflect'),
                        **self.__dict__)
        numFramesNew, linePerFrame, pixPerLine = np.shape(mov_out)

        #% compute baseline quickly
        logging.debug("binning data ...")
        sys.stdout.flush()
        movBL = np.reshape(mov_out.copy(),
                           (downsampfact, int(old_div(numFramesNew, downsampfact)), linePerFrame, pixPerLine),
                           order=order)
        movBL = np.percentile(movBL, quantilMin, axis=0)
        logging.debug("interpolating data ...")
        sys.stdout.flush()
        logging.debug("movBL shape is " + str(movBL.shape))
        movBL = scipy.ndimage.zoom(np.array(movBL, dtype=np.float32), [downsampfact, 1, 1],
                                   order=1,
                                   mode='constant',
                                   cval=0.0,
                                   prefilter=False)

        #% compute DF/F
        if method == 'delta_f_over_sqrt_f':
            mov_out = old_div((mov_out - movBL), np.sqrt(movBL))
        elif method == 'delta_f_over_f':
            mov_out = old_div((mov_out - movBL), movBL)
        elif method == 'only_baseline':
            mov_out = (mov_out - movBL)
        else:
            raise Exception('Unknown method')

        mov_out = mov_out[padbefore:len(movBL) - padafter, :, :]
        logging.debug('Final Size Movie:' + np.str(self.shape))
        return mov_out, movie(movBL,
                              fr=self.fr,
                              start_time=self.start_time,
                              meta_data=self.meta_data,
                              file_name=self.file_name)

    def NonnegativeMatrixFactorization(self,
                                       n_components: int = 30,
                                       init: str = 'nndsvd',
                                       beta: int = 1,
                                       tol=5e-7,
                                       sparseness: str = 'components',
                                       **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        See documentation for scikit-learn NMF
        """
        if np.min(self) < 0:
            raise ValueError("All values must be positive")

        T, h, w = self.shape
        Y = np.reshape(self, (T, h * w))
        Y = Y - np.percentile(Y, 1)
        Y = np.clip(Y, 0, np.Inf)
        estimator = NMF(n_components=n_components, init=init, tol=tol, **kwargs)
        time_components = estimator.fit_transform(Y)
        components_ = estimator.components_
        space_components = np.reshape(components_, (n_components, h, w))

        return space_components, time_components

    def online_NMF(self,
                   n_components: int = 30,
                   method: str = 'nnsc',
                   lambda1: int = 100,
                   iterations: int = -5,
                   model=None,
                   **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """ Method performing online matrix factorization and using the spams

        (http://spams-devel.gforge.inria.fr/doc-python/html/index.html) package from Inria.
        Implements bith the nmf and nnsc methods

        Args:
            n_components: int

            method: 'nnsc' or 'nmf' (see http://spams-devel.gforge.inria.fr/doc-python/html/index.html)

            lambda1: see http://spams-devel.gforge.inria.fr/doc-python/html/index.html

            iterations: see http://spams-devel.gforge.inria.fr/doc-python/html/index.html

            batchsize: see http://spams-devel.gforge.inria.fr/doc-python/html/index.html

            model: see http://spams-devel.gforge.inria.fr/doc-python/html/index.html

            **kwargs: more arguments to be passed to nmf or nnsc

        Returns:
            time_comps

            space_comps
        """
        try:
            import spams       # XXX consider moving this to the head of the file
        except:
            logging.error("You need to install the SPAMS package")
            raise

        T, d1, d2 = np.shape(self)
        d = d1 * d2
        X = np.asfortranarray(np.reshape(self, [T, d], order='F'))

        if method == 'nmf':
            (time_comps, V) = spams.nmf(X, return_lasso=True, K=n_components, numThreads=4, iter=iterations, **kwargs)

        elif method == 'nnsc':
            (time_comps, V) = spams.nnsc(X,
                                         return_lasso=True,
                                         K=n_components,
                                         lambda1=lambda1,
                                         iter=iterations,
                                         model=model,
                                         **kwargs)
        else:
            raise Exception('Method unknown')

        space_comps = []

        for _, mm in enumerate(V):
            space_comps.append(np.reshape(mm.todense(), (d1, d2), order='F'))

        return time_comps, np.array(space_comps)

    def IPCA(self, components: int = 50, batch: int = 1000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Iterative Principal Component analysis, see sklearn.decomposition.incremental_pca

        Args:
            components (default 50) = number of independent components to return

            batch (default 1000)  = number of pixels to load into memory simultaneously in IPCA. More requires more memory but leads to better fit

        Returns:
            eigenseries: principal components (pixel time series) and associated singular values

            eigenframes: eigenframes are obtained by multiplying the projected frame matrix by the projected movie (whitened frames?)

            proj_frame_vectors:the reduced version of the movie vectors using only the principal component projection
        """

        # vectorize the images
        num_frames, h, w = np.shape(self)
        frame_size = h * w
        frame_samples = np.reshape(self, (num_frames, frame_size)).T

        # run IPCA to approxiate the SVD
        ipca_f = incremental_pca(n_components=components, batch_size=batch)
        ipca_f.fit(frame_samples)

        # construct the reduced version of the movie vectors using only the
        # principal component projection

        proj_frame_vectors = ipca_f.inverse_transform(ipca_f.transform(frame_samples))

        # get the temporal principal components (pixel time series) and
        # associated singular values

        eigenseries = ipca_f.components_.T

        # the rows of eigenseries are approximately orthogonal
        # so we can approximately obtain eigenframes by multiplying the
        # projected frame matrix by this transpose on the right

        eigenframes = np.dot(proj_frame_vectors, eigenseries)

        return eigenseries, eigenframes, proj_frame_vectors

    def IPCA_stICA(self,
                   componentsPCA: int = 50,
                   componentsICA: int = 40,
                   batch: int = 1000,
                   mu: float = 1,
                   ICAfun: str = 'logcosh',
                   **kwargs) -> np.ndarray:
        """
        Compute PCA + ICA a la Mukamel 2009.

        Args:
            components (default 50) = number of independent components to return
    
            batch (default 1000) = number of pixels to load into memory simultaneously in IPCA. More requires more memory but leads to better fit
    
            mu (default 0.05) = parameter in range [0,1] for spatiotemporal ICA, higher mu puts more weight on spatial information
    
            ICAFun (default = 'logcosh') = cdf to use for ICA entropy maximization
    
            Plus all parameters from sklearn.decomposition.FastICA
    
        Returns:
            ind_frames [components, height, width] = array of independent component "eigenframes"
        """
        # FIXME: mu defaults to 1, not 0.05
        eigenseries, eigenframes, _proj = self.IPCA(componentsPCA, batch)
        # normalize the series

        frame_scale = old_div(mu, np.max(eigenframes))
        frame_mean = np.mean(eigenframes, axis=0)
        n_eigenframes = frame_scale * (eigenframes - frame_mean)

        series_scale = old_div((1 - mu), np.max(eigenframes))
        series_mean = np.mean(eigenseries, axis=0)
        n_eigenseries = series_scale * (eigenseries - series_mean)

        # build new features from the space/time data
        # and compute ICA on them

        eigenstuff = np.concatenate([n_eigenframes, n_eigenseries])

        ica = FastICA(n_components=componentsICA, fun=ICAfun, **kwargs)
        joint_ics = ica.fit_transform(eigenstuff)

        # extract the independent frames
        _, h, w = np.shape(self)
        frame_size = h * w
        ind_frames = joint_ics[:frame_size, :]
        ind_frames = np.reshape(ind_frames.T, (componentsICA, h, w))

        return ind_frames

    def IPCA_denoise(self, components: int = 50, batch: int = 1000):
        """
        Create a denoised version of the movie using only the first 'components' components
        """
        _, _, clean_vectors = self.IPCA(components, batch)
        self = self.__class__(np.reshape(np.float32(clean_vectors.T), np.shape(self)), **self.__dict__)
        return self

    def IPCA_io(self, n_components: int = 50, fun: str = 'logcosh', max_iter: int = 1000, tol=1e-20) -> np.ndarray:
        """ DO NOT USE STILL UNDER DEVELOPMENT
        """
        pca_comp = n_components
        [T, d1, d2] = self.shape
        M = np.reshape(self, (T, d1 * d2))
        [U, S, V] = scipy.sparse.linalg.svds(M, pca_comp)
        S = np.diag(S)
        #        whiteningMatrix = np.dot(scipy.linalg.inv(np.sqrt(S)),U.T)
        #        dewhiteningMatrix = np.dot(U,np.sqrt(S))
        whiteningMatrix = np.dot(scipy.linalg.inv(S), U.T)
        dewhiteningMatrix = np.dot(U, S)
        whitesig = np.dot(whiteningMatrix, M)
        wsigmask = np.reshape(whitesig.T, (d1, d2, pca_comp))
        f_ica = sklearn.decomposition.FastICA(whiten=False, fun=fun, max_iter=max_iter, tol=tol)
        S_ = f_ica.fit_transform(whitesig.T)
        A_ = f_ica.mixing_
        A = np.dot(A_, whitesig)
        mask = np.reshape(A.T, (d1, d2, pca_comp)).transpose([2, 0, 1])

        return mask

    def local_correlations(self,
                           eight_neighbours: bool = False,
                           swap_dim: bool = True,
                           frames_per_chunk: int = 1500,
                           order_mean=1) -> np.ndarray:
        """Computes the correlation image for the input dataset Y

            Args:
                self:  np.ndarray (3D or 4D)
                    Input movie data in 3D or 4D format
    
                eight_neighbours: Boolean
                    Use 8 neighbors if true, and 4 if false for 3D data (default = True)
                    Use 6 neighbors for 4D data, irrespectively
    
                swap_dim: Boolean
                    True indicates that time is listed in the last axis of Y (matlab format)
                    and moves it in the front

                frames_per_chunk: int (undocumented)

                order_mean: (undocumented)
                

            Returns:
                rho: d1 x d2 [x d3] matrix, cross-correlation with adjacent pixels

        """
        T = self.shape[0]
        Cn = np.zeros(self.shape[1:])
        if T <= 3000:
            Cn = si.local_correlations(np.array(self),
                                       eight_neighbours=eight_neighbours,
                                       swap_dim=swap_dim,
                                       order_mean=order_mean)
        else:

            n_chunks = T // frames_per_chunk
            for jj, mv in enumerate(range(n_chunks - 1)):
                logging.debug('number of chunks:' + str(jj) + ' frames: ' +
                              str([mv * frames_per_chunk, (mv + 1) * frames_per_chunk]))
                rho = si.local_correlations(np.array(self[mv * frames_per_chunk:(mv + 1) * frames_per_chunk]),
                                            eight_neighbours=eight_neighbours,
                                            swap_dim=swap_dim,
                                            order_mean=order_mean)
                Cn = np.maximum(Cn, rho)
                pl.imshow(Cn, cmap='gray')
                pl.pause(.1)

            logging.debug('number of chunks:' + str(n_chunks - 1) + ' frames: ' +
                          str([(n_chunks - 1) * frames_per_chunk, T]))
            rho = si.local_correlations(np.array(self[(n_chunks - 1) * frames_per_chunk:]),
                                        eight_neighbours=eight_neighbours,
                                        swap_dim=swap_dim,
                                        order_mean=order_mean)
            Cn = np.maximum(Cn, rho)
            pl.imshow(Cn, cmap='gray')
            pl.pause(.1)

        return Cn

    def partition_FOV_KMeans(self,
                             tradeoff_weight: float = .5,
                             fx: float = .25,
                             fy: float = .25,
                             n_clusters: int = 4,
                             max_iter: int = 500) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Partition the FOV in clusters that are grouping pixels close in space and in mutual correlation

        Args:
            tradeoff_weight:between 0 and 1 will weight the contributions of distance and correlation in the overall metric
    
            fx,fy: downsampling factor to apply to the movie
    
            n_clusters,max_iter: KMeans algorithm parameters

        Returns:
            fovs:array 2D encoding the partitions of the FOV
    
            mcoef: matrix of pairwise correlation coefficients
    
            distanceMatrix: matrix of picel distances
        """
        _, h1, w1 = self.shape
        self.resize(fx, fy)
        T, h, w = self.shape
        Y = np.reshape(self, (T, h * w))
        mcoef = np.corrcoef(Y.T)

        idxA, idxB = np.meshgrid(list(range(w)), list(range(h)))
        coordmat = np.vstack((idxA.flatten(), idxB.flatten()))
        distanceMatrix = euclidean_distances(coordmat.T)
        distanceMatrix = old_div(distanceMatrix, np.max(distanceMatrix))
        estim = KMeans(n_clusters=n_clusters, max_iter=max_iter)
        kk = estim.fit(tradeoff_weight * mcoef - (1 - tradeoff_weight) * distanceMatrix)
        labs = kk.labels_
        fovs = np.reshape(labs, (h, w))
        fovs = cv2.resize(np.uint8(fovs), (w1, h1), old_div(1., fx), old_div(1., fy), interpolation=cv2.INTER_NEAREST)
        return np.uint8(fovs), mcoef, distanceMatrix

    def extract_traces_from_masks(self, masks: np.ndarray) -> trace:
        """
        Args:
            masks: array, 3D with each 2D slice bein a mask (integer or fractional)

        Returns:
            traces: array, 2D of fluorescence traces
        """
        T, h, w = self.shape
        Y = np.reshape(self, (T, h * w))
        if masks.ndim == 2:
            masks = masks[None, :, :]

        nA, _, _ = masks.shape

        A = np.reshape(masks, (nA, h * w))

        pixelsA = np.sum(A, axis=1)
        A = old_div(A, pixelsA[:, None])       # obtain average over ROI
        traces = trace(np.dot(A, np.transpose(Y)).T, **self.__dict__)
        return traces

    def resize(self, fx=1, fy=1, fz=1, interpolation=cv2.INTER_AREA):
        """
        Resizing caiman movie into a new one. Note that the temporal
        dimension is controlled by fz and fx, fy, fz correspond to
        magnification factors. For example to downsample in time by
        a factor of 2, you need to set fz = 0.5.

        Args:
            fx (float):
                Magnification factor along x-dimension

            fy (float):
                Magnification factor along y-dimension

            fz (float):
                Magnification factor along temporal dimension

        Returns:
            self (caiman movie)
        """
        T, d1, d2 = self.shape
        d = d1 * d2
        elm = d * T
        max_els = 2**61 - 1    # the bug for sizes >= 2**31 is appears to be fixed now
        if elm > max_els:
            chunk_size = old_div((max_els), d)
            new_m: List = []
            logging.debug('Resizing in chunks because of opencv bug')
            for chunk in range(0, T, chunk_size):
                logging.debug([chunk, np.minimum(chunk + chunk_size, T)])
                m_tmp = self[chunk:np.minimum(chunk + chunk_size, T)].copy()
                m_tmp = m_tmp.resize(fx=fx, fy=fy, fz=fz, interpolation=interpolation)
                if len(new_m) == 0:
                    new_m = m_tmp
                else:
                    new_m = timeseries.concatenate([new_m, m_tmp], axis=0)

            return new_m
        else:
            if fx != 1 or fy != 1:
                logging.debug("reshaping along x and y")
                t, h, w = self.shape
                newshape = (int(w * fy), int(h * fx))
                mov = []
                logging.debug("New shape is " + str(newshape))
                for frame in self:
                    mov.append(cv2.resize(frame, newshape, fx=fx, fy=fy, interpolation=interpolation))
                self = movie(np.asarray(mov), **self.__dict__)
            if fz != 1:
                logging.debug("reshaping along z")
                t, h, w = self.shape
                self = np.reshape(self, (t, h * w))
                mov = cv2.resize(self, (h * w, int(fz * t)), fx=1, fy=fz, interpolation=interpolation)
                mov = np.reshape(mov, (np.maximum(1, int(fz * t)), h, w))
                self = movie(mov, **self.__dict__)
                self.fr = self.fr * fz

        return self

    def guided_filter_blur_2D(self, guide_filter, radius: int = 5, eps=0):
        """
        performs guided filtering on each frame. See opencv documentation of cv2.ximgproc.guidedFilter
        """
        for idx, fr in enumerate(self):
            if idx % 1000 == 0:
                logging.debug("At index: " + str(idx))
            self[idx] = cv2.ximgproc.guidedFilter(guide_filter, fr, radius=radius, eps=eps)

        return self

    def bilateral_blur_2D(self, diameter: int = 5, sigmaColor: int = 10000, sigmaSpace=0):
        """
        performs bilateral filtering on each frame. See opencv documentation of cv2.bilateralFilter
        """
        if type(self[0, 0, 0]) is not np.float32:
            warnings.warn('Casting the array to float 32')
            self = np.asanyarray(self, dtype=np.float32)

        for idx, fr in enumerate(self):
            if idx % 1000 == 0:
                logging.debug("At index: " + str(idx))
            self[idx] = cv2.bilateralFilter(fr, diameter, sigmaColor, sigmaSpace)

        return self

    def gaussian_blur_2D(self,
                         kernel_size_x=5,
                         kernel_size_y=5,
                         kernel_std_x=1,
                         kernel_std_y=1,
                         borderType=cv2.BORDER_REPLICATE):
        """
        Compute gaussian blut in 2D. Might be useful when motion correcting

        Args:
            kernel_size: double
                see opencv documentation of GaussianBlur
            kernel_std_: double
                see opencv documentation of GaussianBlur
            borderType: int
                see opencv documentation of GaussianBlur

        Returns:
            self: ndarray
                blurred movie
        """

        for idx, fr in enumerate(self):
            logging.debug(idx)
            self[idx] = cv2.GaussianBlur(fr,
                                         ksize=(kernel_size_x, kernel_size_y),
                                         sigmaX=kernel_std_x,
                                         sigmaY=kernel_std_y,
                                         borderType=borderType)

        return self

    def median_blur_2D(self, kernel_size: float = 3.0):
        """
        Compute gaussian blut in 2D. Might be useful when motion correcting

        Args:
            kernel_size: double
                see opencv documentation of GaussianBlur

            kernel_std_: double
                see opencv documentation of GaussianBlur

            borderType: int
                see opencv documentation of GaussianBlur

        Returns:
            self: ndarray
                blurred movie
        """

        for idx, fr in enumerate(self):
            logging.debug(idx)
            self[idx] = cv2.medianBlur(fr, ksize=kernel_size)

        return self

    def to_2D(self, order='F') -> np.ndarray:
        [T, d1, d2] = self.shape
        d = d1 * d2
        return np.reshape(self, (T, d), order=order)

    def zproject(self, method: str = 'mean', cmap=pl.cm.gray, aspect='auto', **kwargs) -> np.ndarray:
        """
        Compute and plot projection across time:

        Args:
            method: String
                'mean','median','std'

            **kwargs: dict
                arguments to imagesc

        Raises:
             Exception 'Method not implemented'
        """
        # TODO: make the imshow optional
        # TODO: todocument
        if method == 'mean':
            zp = np.mean(self, axis=0)
        elif method == 'median':
            zp = np.median(self, axis=0)
        elif method == 'std':
            zp = np.std(self, axis=0)
        else:
            raise Exception('Method not implemented')
        pl.imshow(zp, cmap=cmap, aspect=aspect, **kwargs)
        return zp

    def play(self,
             gain: float = 1,
             fr=None,
             magnification=1,
             offset=0,
             interpolation=cv2.INTER_LINEAR,
             backend: str = 'opencv',
             do_loop: bool = False,
             bord_px=None,
             q_max=99.75,
             q_min=1,
             plot_text: bool = False,
             save_movie: bool = False,
             opencv_codec: str = 'H264',
             movie_name: str = 'movie.avi') -> None:
        """
        Play the movie using opencv

        Args:
            gain: adjust movie brightness

            fr: framerate, playing speed if different from original (inter frame interval in seconds)

            magnification: int
                magnification factor

            offset: (undocumented)

            interpolation: (undocumented)

            backend: 'pylab' or 'opencv', the latter much faster

            do_loop: Whether to loop the video

            bord_px: int
                truncate pixels from the borders

            q_max, q_min: float in [0, 100]
                 percentile for maximum/minimum plotting value

            plot_text: bool
                show some text

            save_movie: bool
                flag to save an avi file of the movie

            opencv_codec: str
                FourCC video codec for saving movie. Check http://www.fourcc.org/codecs.php

            movie_name: str
                name of saved file

        Raises:
            Exception 'Unknown backend!'
        """
        # todo: todocument
        if backend == 'pylab':
            logging.warning('*** WARNING *** SPEED MIGHT BE LOW. USE opencv backend if available')

        gain *= 1.     # convert to float in case we were passed an int
        if q_max < 100:
            maxmov = np.nanpercentile(self[0:10], q_max)
        else:
            maxmov = np.nanmax(self)

        if q_min > 0:
            minmov = np.nanpercentile(self[0:10], q_min)
        else:
            minmov = np.nanmin(self)

        if backend == 'pylab':
            pl.ion()
            fig = pl.figure(1)
            ax = fig.add_subplot(111)
            ax.set_title("Play Movie")
            im = ax.imshow((offset + self[0] - minmov) * gain / (maxmov - minmov + offset),
                           cmap=pl.cm.gray,
                           vmin=0,
                           vmax=1,
                           interpolation='none')                                            # Blank starting image
            fig.show()
            im.axes.figure.canvas.draw()
            pl.pause(1)

        if backend == 'notebook':
            # First set up the figure, the axis, and the plot element we want to animate
            fig = pl.figure()
            im = pl.imshow(self[0], interpolation='None', cmap=pl.cm.gray)
            pl.axis('off')

            def animate(i):
                im.set_data(self[i])
                return im,

            # call the animator.  blit=True means only re-draw the parts that have changed.
            anim = animation.FuncAnimation(fig, animate, frames=self.shape[0], interval=1, blit=True)

            # call our new function to display the animation
            return visualization.display_animation(anim, fps=fr)

        if fr is None:
            fr = self.fr

        looping = True
        terminated = False
        if save_movie:
            fourcc = cv2.VideoWriter_fourcc(*opencv_codec)
            frame_in = self[0]
            if bord_px is not None and np.sum(bord_px) > 0:
                frame_in = frame_in[bord_px:-bord_px, bord_px:-bord_px]
            out = cv2.VideoWriter(movie_name, fourcc, 30.,
                                  tuple([int(magnification * s) for s in frame_in.shape[1::-1]]))
        while looping:

            for iddxx, frame in enumerate(self):
                if bord_px is not None and np.sum(bord_px) > 0:
                    frame = frame[bord_px:-bord_px, bord_px:-bord_px]

                if backend == 'opencv':
                    if magnification != 1:
                        frame = cv2.resize(frame, None, fx=magnification, fy=magnification, interpolation=interpolation)
                    frame = (offset + frame - minmov) * gain / (maxmov - minmov)

                    if plot_text == True:
                        text_width, text_height = cv2.getTextSize('Frame = ' + str(iddxx),
                                                                  fontFace=5,
                                                                  fontScale=0.8,
                                                                  thickness=1)[0]
                        cv2.putText(frame,
                                    'Frame = ' + str(iddxx),
                                    ((frame.shape[1] - text_width) // 2, frame.shape[0] - (text_height + 5)),
                                    fontFace=5,
                                    fontScale=0.8,
                                    color=(255, 255, 255),
                                    thickness=1)

                    cv2.imshow('frame', frame)
                    if save_movie:
                        if frame.ndim < 3:
                            frame = np.repeat(frame[:, :, None], 3, axis=-1)
                        frame = np.minimum((frame * 255.), 255).astype('u1')
                        out.write(frame)
                    if cv2.waitKey(int(1. / fr * 1000)) & 0xFF == ord('q'):
                        looping = False
                        terminated = True
                        break

                elif backend == 'pylab':

                    im.set_data((offset + frame) * gain / maxmov)
                    ax.set_title(str(iddxx))
                    pl.axis('off')
                    fig.canvas.draw()
                    pl.pause(1. / fr * .5)
                    ev = pl.waitforbuttonpress(1. / fr * .5)
                    if ev is not None:
                        pl.close()
                        break

                elif backend == 'notebook':
                    logging.debug('Animated via MP4')
                    break

                else:
                    raise Exception('Unknown backend!')

            if terminated:
                break

            if save_movie:
                out.release()
                save_movie = False

            if do_loop:
                looping = True
            else:
                looping = False

        if backend == 'opencv':
            cv2.waitKey(100)
            cv2.destroyAllWindows()
            for i in range(10):
                cv2.waitKey(100)


def load(file_name: Union[str, List[str]],
         fr: float = 30,
         start_time: float = 0,
         meta_data: Dict = None,
         subindices=None,
         shape: Tuple[int, int] = None,
         var_name_hdf5: str = 'mov',
         in_memory: bool = False,
         is_behavior: bool = False,
         bottom=0,
         top=0,
         left=0,
         right=0,
         channel=None,
         outtype=np.float32,
         is3D: bool = False) -> Any:
    """
    load movie from file. Supports a variety of formats. tif, hdf5, npy and memory mapped. Matlab is experimental.

    Args:
        file_name: string or List[str]
            name of file. Possible extensions are tif, avi, npy, (npz and hdf5 are usable only if saved by calblitz)

        fr: float
            frame rate

        start_time: float
            initial time for frame 1

        meta_data: dict
            dictionary containing meta information about the movie

        subindices: iterable indexes
            for loading only portion of the movie

        shape: tuple of two values
            dimension of the movie along x and y if loading from a two dimensional numpy array

        var_name_hdf5: str
            if loading from hdf5 name of the variable to load

        in_memory: (undocumented)

        is_behavior: (undocumented)

        bottom,top,left,right: (undocumented)

        channel: (undocumented)

        outtype: The data type for the movie

    Returns:
        mov: caiman.movie

    Raises:
        Exception 'Subindices not implemented'
    
        Exception 'Subindices not implemented'
    
        Exception 'sima module unavailable'
    
        Exception 'Unknown file type'
    
        Exception 'File not found!'
    """
    # case we load movie from file
    if max(top, bottom, left, right) > 0 and type(file_name) is str:
        file_name = [file_name]        # type: ignore # mypy doesn't like that this changes type

    if type(file_name) is list:
        if shape is not None:
            logging.error('shape not supported for multiple movie input')

        return load_movie_chain(file_name,
                                fr=fr,
                                start_time=start_time,
                                meta_data=meta_data,
                                subindices=subindices,
                                bottom=bottom,
                                top=top,
                                left=left,
                                right=right,
                                channel=channel,
                                outtype=outtype,
                                var_name_hdf5=var_name_hdf5,
                                is3D=is3D)

    if max(top, bottom, left, right) > 0:
        logging.error('top bottom etc... not supported for single movie input')

    if channel is not None:
        logging.error('channel not supported for single movie input')

    if os.path.exists(file_name):
        _, extension = os.path.splitext(file_name)[:2]
        extension = extension.lower()
        if extension == '.mat':
            logging.warning('Loading a *.mat file. x- and y- dimensions ' +
                            'might have been swapped.')
            byte_stream, file_opened = scipy.io.matlab.mio._open_file(file_name, appendmat=False)
            mjv, mnv = scipy.io.matlab.mio.get_matfile_version(byte_stream)
            if mjv == 2:
                extension = '.h5'

        if extension == '.tif' or extension == '.tiff':        # load avi file
            with tifffile.TiffFile(file_name) as tffl:
                multi_page = True if tffl.series[0].shape[0] > 1 else False
                if len(tffl.pages) == 1:
                    logging.warning('Your tif file is saved a single page' + 'file. Performance will be affected')
                    multi_page = False
                if subindices is not None:
                    #if isinstance(subindices, (list, tuple)): # is list or tuple:
                    if isinstance(subindices, list):  # is list or tuple:
                        if multi_page:
                            input_arr = tffl.asarray(key=subindices[0])[:, subindices[1], subindices[2]]
                        else:
                            input_arr = tffl.asarray()
                            input_arr = input_arr[subindices[0], subindices[1], subindices[2]]
                    else:
                        if multi_page:
                            input_arr = tffl.asarray(key=subindices)
                        else:
                            input_arr = tffl.asarray()
                            input_arr = input_arr[subindices]

                else:
                    input_arr = tffl.asarray()

                input_arr = np.squeeze(input_arr)

        elif extension == '.avi':      # load avi file
            cap = cv2.VideoCapture(file_name)

            try:
                length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            except:
                logging.info('Roll back to opencv 2')
                length = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
                width = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))

            cv_failed = False
            dims = [length, height, width]                     # type: ignore # a list in one block and a tuple in another
            if length == 0 or width == 0 or height == 0:       #CV failed to load
                cv_failed = True
            if subindices is not None:
                if type(subindices) is not list:
                    subindices = [subindices]
                for ind, sb in enumerate(subindices):
                    if type(sb) is range:
                        subindices[ind] = np.r_[sb]
                        dims[ind] = subindices[ind].shape[0]
                    elif type(sb) is slice:
                        if sb.start is None:
                            sb = slice(0, sb.stop, sb.step)
                        if sb.stop is None:
                            sb = slice(sb.start, dims[ind], sb.step)
                        subindices[ind] = np.r_[sb]
                        dims[ind] = subindices[ind].shape[0]
                    elif type(sb) is np.ndarray:
                        dims[ind] = sb.shape[0]

                start_frame = subindices[0][0]
            else:
                subindices = [np.r_[range(dims[0])]]
                start_frame = 0
            if not cv_failed:
                input_arr = np.zeros((dims[0], height, width), dtype=np.uint8)
                counter = 0
                cap.set(1, start_frame)
                current_frame = start_frame
                while True and counter < dims[0]:
                    # Capture frame-by-frame
                    if current_frame != subindices[0][counter]:
                        current_frame = subindices[0][counter]
                        cap.set(1, current_frame)
                    ret, frame = cap.read()
                    if not ret:
                        break
                    input_arr[counter] = frame[:, :, 0]
                    counter += 1
                    current_frame += 1

                if len(subindices) > 1:
                    input_arr = input_arr[:, subindices[1]]
                if len(subindices) > 2:
                    input_arr = input_arr[:, :, subindices[2]]
            else:      #use pims to load movie
                import pims

                def rgb2gray(rgb):
                    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

                pims_movie = pims.Video(file_name)
                length = len(pims_movie)
                height, width = pims_movie.frame_shape[0:2]    #shape is (h,w,channels)
                input_arr = np.zeros((length, height, width), dtype=np.uint8)
                for i in range(len(pims_movie)):               #iterate over frames
                    input_arr[i] = rgb2gray(pims_movie[i])

            # When everything done, release the capture
            cap.release()
            cv2.destroyAllWindows()

        elif extension == '.npy':      # load npy file
            if fr is None:
                fr = 30
            if in_memory:
                input_arr = np.load(file_name)
            else:
                input_arr = np.load(file_name, mmap_mode='r')

            if subindices is not None:
                input_arr = input_arr[subindices]

            if input_arr.ndim == 2:
                if shape is not None:
                    _, T = np.shape(input_arr)
                    d1, d2 = shape
                    input_arr = np.transpose(np.reshape(input_arr, (d1, d2, T), order='F'), (2, 0, 1))
                else:
                    input_arr = input_arr[np.newaxis, :, :]

        elif extension == '.mat':      # load npy file
            input_arr = loadmat(file_name)['data']
            input_arr = np.rollaxis(input_arr, 2, -3)
            if subindices is not None:
                input_arr = input_arr[subindices]

        elif extension == '.npz':      # load movie from saved file
            if subindices is not None:
                raise Exception('Subindices not implemented')
            with np.load(file_name) as f:
                return movie(**f).astype(outtype)

        elif extension in ('.hdf5', '.h5', '.nwb'):
            if is_behavior:
                with h5py.File(file_name, "r") as f:
                    kk = list(f.keys())
                    kk.sort(key=lambda x: np.int(x.split('_')[-1]))
                    input_arr = []
                    for trial in kk:
                        logging.info('Loading ' + trial)
                        input_arr.append(np.array(f[trial]['mov']))

                    input_arr = np.vstack(input_arr)

            else:
                with h5py.File(file_name, "r") as f:
                    fkeys = list(f.keys())
                    if len(fkeys) == 1:
                        var_name_hdf5 = fkeys[0]

                    if extension == '.nwb':
                        fgroup = f[var_name_hdf5]['data']
                    else:
                        fgroup = f[var_name_hdf5]

                    if var_name_hdf5 in f:
                        if subindices is None:
                            images = np.array(fgroup).squeeze()
                            #if images.ndim > 3:
                            #    images = images[:, 0]
                        else:
                            if type(subindices).__module__ is 'numpy':
                                subindices = subindices.tolist()
                            if len(fgroup.shape) > 3:
                                images = np.array(fgroup[subindices]).squeeze()
                            else:
                                images = np.array(fgroup[subindices]).squeeze()

                        #input_arr = images
                        return movie(images.astype(outtype))
                    else:
                        logging.debug('KEYS:' + str(f.keys()))
                        raise Exception('Key not found in hdf5 file')

        elif extension == '.mmap':

            filename = os.path.split(file_name)[-1]
            Yr, dims, T = load_memmap(
                os.path.join(                  # type: ignore # same dims typing issue as above
                    os.path.split(file_name)[0], filename))
            images = np.reshape(Yr.T, [T] + list(dims), order='F')
            if subindices is not None:
                images = images[subindices]

            if in_memory:
                logging.debug('loading mmap file in memory')
                images = np.array(images).astype(outtype)

            logging.debug('mmap')
            return movie(images, fr=fr)

        elif extension == '.sbx':
            logging.debug('sbx')
            if subindices is not None:
                return movie(sbxreadskip(file_name[:-4], subindices), fr=fr).astype(outtype)
            else:
                return movie(sbxread(file_name[:-4], k=0, n_frames=np.inf), fr=fr).astype(outtype)

        elif extension == '.sima':
            if not HAS_SIMA:
                raise Exception("sima module unavailable")

            dataset = sima.ImagingDataset.load(file_name)
            frame_step = 1000
            if subindices is None:
                input_arr = np.empty(
                    (dataset.sequences[0].shape[0], dataset.sequences[0].shape[2], dataset.sequences[0].shape[3]),
                    dtype=outtype)
                for nframe in range(0, dataset.sequences[0].shape[0], frame_step):
                    input_arr[nframe:nframe + frame_step] = np.array(
                        dataset.sequences[0][nframe:nframe + frame_step, 0, :, :, 0]).astype(outtype).squeeze()
            else:
                input_arr = np.array(dataset.sequences[0])[subindices, :, :, :, :].squeeze()

        else:
            raise Exception('Unknown file type')
    else:
        logging.error(f"File request:[{file_name}] not found!")
        raise Exception('File not found!')

    return movie(input_arr.astype(outtype),
                 fr=fr,
                 start_time=start_time,
                 file_name=os.path.split(file_name)[-1],
                 meta_data=meta_data)


def load_movie_chain(file_list: List[str],
                     fr: float = 30,
                     start_time=0,
                     meta_data=None,
                     subindices=None,
                     var_name_hdf5: str = 'mov',
                     bottom=0,
                     top=0,
                     left=0,
                     right=0,
                     z_top=0,
                     z_bottom=0,
                     is3D: bool = False,
                     channel=None,
                     outtype=np.float32) -> Any:
    """ load movies from list of file names

    Args:
        file_list: list
           file names in string format
    
        the other parameters as in load_movie except
    
        bottom, top, left, right, z_top, z_bottom : int
            to load only portion of the field of view
    
        is3D : bool
            flag for 3d data (adds a fourth dimension)

    Returns:
        movie: movie
            movie corresponding to the concatenation og the input files

    """
    mov = []
    for f in tqdm(file_list):
        m = load(f,
                 fr=fr,
                 start_time=start_time,
                 meta_data=meta_data,
                 subindices=subindices,
                 in_memory=True,
                 outtype=outtype,
                 var_name_hdf5=var_name_hdf5)
        if channel is not None:
            logging.debug(m.shape)
            m = m[channel].squeeze()
            logging.debug(f"Movie shape: {m.shape}")

        if not is3D:
            if m.ndim == 2:
                m = m[np.newaxis, :, :]

            _, h, w = np.shape(m)
            m = m[:, top:h - bottom, left:w - right]
        else:
            if m.ndim == 3:
                m = m[np.newaxis, :, :, :]

            _, h, w, d = np.shape(m)
            m = m[:, top:h - bottom, left:w - right, z_top:d - z_bottom]

        mov.append(m)
    return ts.concatenate(mov, axis=0)


####
# TODO: Consider pulling these functions that work with .mat files into a separate file


def loadmat_sbx(filename: str):
    """
    this wrapper should be called instead of directly calling spio.loadmat

    It solves the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to fix all entries
    which are still mat-objects
    """
    data_ = loadmat(filename, struct_as_record=False, squeeze_me=True)
    _check_keys(data_)
    return data_


def _check_keys(checkdict: Dict) -> None:
    """
    checks if entries in dictionary are mat-objects. If yes todict is called to change them to nested dictionaries.
    Modifies its parameter in-place.
    """

    for key in checkdict:
        if isinstance(checkdict[key], scipy.io.matlab.mio5_params.mat_struct):
            checkdict[key] = _todict(checkdict[key])


def _todict(matobj) -> Dict:
    """
    A recursive function which constructs from matobjects nested dictionaries
    """

    ret = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, scipy.io.matlab.mio5_params.mat_struct):
            ret[strg] = _todict(elem)
        else:
            ret[strg] = elem
    return ret


def sbxread(filename: str, k: int = 0, n_frames=np.inf) -> np.ndarray:
    """
    Args:
        filename: str
            filename should be full path excluding .sbx
    """
    # Check if contains .sbx and if so just truncate
    if '.sbx' in filename:
        filename = filename[:-4]

    # Load info
    info = loadmat_sbx(filename + '.mat')['info']

    # Defining number of channels/size factor
    if info['channels'] == 1:
        info['nChan'] = 2
        factor = 1
    elif info['channels'] == 2:
        info['nChan'] = 1
        factor = 2
    elif info['channels'] == 3:
        info['nChan'] = 1
        factor = 2

    # Determine number of frames in whole file
    max_idx = os.path.getsize(filename + '.sbx') / info['recordsPerBuffer'] / info['sz'][1] * factor / 4 - 1

    # Paramters
    N = max_idx + 1    # Last frame
    N = np.minimum(N, n_frames)

    nSamples = info['sz'][1] * info['recordsPerBuffer'] * 2 * info['nChan']

    # Open File
    fo = open(filename + '.sbx')

    # Note: SBX files store the values strangely, its necessary to subtract the values from the max int16 to get the correct ones
    fo.seek(k * nSamples, 0)
    ii16 = np.iinfo(np.uint16)
    x = ii16.max - np.fromfile(fo, dtype='uint16', count=int(nSamples / 2 * N))
    x = x.reshape((int(info['nChan']), int(info['sz'][1]), int(info['recordsPerBuffer']), int(N)), order='F')

    x = x[0, :, :, :]

    fo.close()

    return x.transpose([2, 1, 0])


def sbxreadskip(filename: str, subindices: slice) -> np.ndarray:
    """
    Args:
        filename: str
            filename should be full path excluding .sbx

        slice: pass a slice to slice along the last dimension
    """
    # Check if contains .sbx and if so just truncate
    if '.sbx' in filename:
        filename = filename[:-4]

    # Load info
    info = loadmat_sbx(filename + '.mat')['info']

    # Defining number of channels/size factor
    if info['channels'] == 1:
        info['nChan'] = 2
        factor = 1
    elif info['channels'] == 2:
        info['nChan'] = 1
        factor = 2
    elif info['channels'] == 3:
        info['nChan'] = 1
        factor = 2

    # Determine number of frames in whole file
    max_idx = np.int(os.path.getsize(filename + '.sbx') / info['recordsPerBuffer'] / info['sz'][1] * factor / 4 - 1)

    # Paramters
    if isinstance(subindices, slice):
        if subindices.start is None:
            start = 0
        else:
            start = subindices.start

        if subindices.stop is None:
            N = max_idx + 1    # Last frame
        else:
            N = np.minimum(subindices.stop, max_idx + 1).astype(np.int)

        if subindices.step is None:
            skip = 1
        else:
            skip = subindices.step

        iterable_elements = range(start, N, skip)

    else:

        N = len(subindices)
        iterable_elements = subindices
        skip = 0

    N_time = len(list(iterable_elements))

    nSamples = info['sz'][1] * info['recordsPerBuffer'] * 2 * info['nChan']
    assert nSamples >= 0

    # Open File
    fo = open(filename + '.sbx')

    # Note: SBX files store the values strangely, its necessary to subtract the values from the max int16 to get the correct ones

    counter = 0

    if skip == 1:
        # Note: SBX files store the values strangely, its necessary to subtract the values from the max int16 to get the correct ones
        assert start * nSamples > 0
        fo.seek(start * nSamples, 0)
        ii16 = np.iinfo(np.uint16)
        x = ii16.max - np.fromfile(fo, dtype='uint16', count=int(nSamples / 2 * (N - start)))
        x = x.reshape((int(info['nChan']), int(info['sz'][1]), int(info['recordsPerBuffer']), int(N - start)),
                      order='F')

        x = x[0, :, :, :]

    else:
        for k in iterable_elements:
            assert k >= 0
            if counter % 100 == 0:
                print(f'Reading Iteration: {k}')
            fo.seek(k * nSamples, 0)
            ii16 = np.iinfo(np.uint16)
            tmp = ii16.max - \
                np.fromfile(fo, dtype='uint16', count=int(nSamples / 2 * 1))

            tmp = tmp.reshape((int(info['nChan']), int(info['sz'][1]), int(info['recordsPerBuffer'])), order='F')
            if counter == 0:
                x = np.zeros((tmp.shape[0], tmp.shape[1], tmp.shape[2], N_time))

            x[:, :, :, counter] = tmp
            counter += 1

        x = x[0, :, :, :]
    fo.close()

    return x.transpose([2, 1, 0])


def sbxshape(filename: str) -> Tuple[int, int, int]:
    """
    Args:
        filename should be full path excluding .sbx
    """
    # TODO: Document meaning of return values

    # Check if contains .sbx and if so just truncate
    if '.sbx' in filename:
        filename = filename[:-4]

    # Load info
    info = loadmat_sbx(filename + '.mat')['info']

    # Defining number of channels/size factor
    if info['channels'] == 1:
        info['nChan'] = 2
        factor = 1
    elif info['channels'] == 2:
        info['nChan'] = 1
        factor = 2
    elif info['channels'] == 3:
        info['nChan'] = 1
        factor = 2

    # Determine number of frames in whole file
    max_idx = os.path.getsize(filename + '.sbx') / info['recordsPerBuffer'] / info['sz'][1] * factor / 4 - 1
    N = max_idx + 1    # Last frame
    x = (int(info['sz'][1]), int(info['recordsPerBuffer']), int(N))
    return x


def to_3D(mov2D: np.ndarray, shape: Tuple, order='F') -> np.ndarray:
    """
    transform a vectorized movie into a 3D shape
    """
    return np.reshape(mov2D, shape, order=order)


def from_zip_file_to_movie(zipfile_name: str, start_end: Tuple = None) -> Any:
    '''
    Read/convert a movie from a zipfile.

    Args:
        zipfile_name: Name of zipfile to read
        start_end: Start and end frame to extract
    Returns:
        movie
    '''
    mov: List = []
    print('unzipping file into movie object')
    if start_end is not None:
        num_frames = start_end[1] - start_end[0]

    counter = 0
    with ZipFile(zipfile_name) as archive:
        for idx, entry in enumerate(archive.infolist()):
            if idx >= start_end[0] and idx < start_end[1]:     # type: ignore # FIXME: default value would not work with this
                with archive.open(entry) as file:
                    if counter == 0:
                        img = np.array(Image.open(file))
                        mov = np.zeros([num_frames, *img.shape], dtype=np.float32)
                        mov[counter] = img
                    else:
                        mov[counter] = np.array(Image.open(file))

                    if counter % 100 == 0:
                        print([counter, idx])

                    counter += 1

    return cm.movie(mov[:counter])


def from_zipfiles_to_movie_lists(zipfile_name: str, max_frames_per_movie: int = 3000,
                                 binary: bool = False) -> List[str]:
    '''
    Transform zip file into set of tif movies
    Args:
        zipfile_name: Name of zipfile to read
        max_frames_per_movie: (undocumented)
        binary: (undocumented)
    Returns:
        List of filenames in the movie list
    '''
    # TODO: Filter contents of zipfile for appropriate filenames, so we can have a README in there
    with ZipFile(zipfile_name) as archive:
        num_frames_total = len(archive.infolist())

    base_file_names = os.path.split(zipfile_name)[0]
    start_frames = np.arange(0, num_frames_total, max_frames_per_movie)

    movie_list = []
    for sf in start_frames:

        mov = from_zip_file_to_movie(zipfile_name, start_end=(sf, sf + max_frames_per_movie))
        if binary:
            fname = os.path.join(base_file_names, f'movie_{sf}.mmap')
            fname = mov.save(fname, order='C')
        else:
            fname = os.path.join(base_file_names, f'movie_{sf}.tif')
            mov.save(fname)

        movie_list.append(fname)

    return movie_list


def rolling_window(ndarr, window_size, stride):   
        """
        generates efficient rolling window for running statistics
        Args:
            ndarr: ndarray
                input pixels in format pixels x time
            window_size: int
                size of the sliding window
            stride: int
                stride of the sliding window
        Returns:
                iterator with views of the input array
                
        """
        for i in range(0,ndarr.shape[-1]-window_size-stride+1,stride): 
            yield ndarr[:,i:np.minimum(i+window_size, ndarr.shape[-1])]
            
        if i+stride != ndarr.shape[-1]:
           yield ndarr[:,i+stride:]


def load_iter(file_name, subindices=None, var_name_hdf5: str = 'mov'):
    """
    load iterator over movie from file. Supports a variety of formats. tif, hdf5, avi.

    Args:
        file_name: string
            name of file. Possible extensions are tif, avi and hdf5

        subindices: iterable indexes
            for loading only a portion of the movie

    Returns:
        iter: iterator over movie

    Raises:
        Exception 'Subindices not implemented'

        Exception 'sima module unavailable'

        Exception 'Unknown file type'

        Exception 'File not found!'
    """
    if os.path.exists(file_name):
        extension = os.path.splitext(file_name)[1].lower()
        if extension in ('.tif', '.tiff'):
            Y = tifffile.TiffFile(file_name).pages
            if subindices is not None:
                if type(subindices) is range:
                    subindices = slice(subindices.start, subindices.stop, subindices.step)
                Y = Y[subindices]
            for y in Y:
                yield y.asarray()
        elif extension == '.avi':
            cap = cv2.VideoCapture(file_name)
            if subindices is None:
                while True:
                    ret, frame = cap.read()
                    if ret:
                        yield frame[..., 0]
                    else:
                        cap.release()
                        return
                        #raise StopIteration
            else:
                if type(subindices) is slice:
                    subindices = range(
                        subindices.start,
                        int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if subindices.stop is None else subindices.stop,
                        1 if subindices.step is None else subindices.step)
                t = 0
                for ind in subindices:
#                    cap.set(1, ind)
#                    ret, frame = cap.read()
#                    if ret:
#                        yield frame[..., 0]
#                    else:
#                        raise StopIteration
                    while t <= ind:
                        ret, frame = cap.read()
                        t += 1
                    if ret:
                        yield frame[..., 0]
                    else:
                        return
                        #raise StopIteration
                cap.release()
                return
                #raise StopIteration
        elif extension in ('.hdf5', '.h5'):
            with h5py.File(file_name, "r") as f:
                Y = f.get(var_name_hdf5)
                if subindices is None:
                    for y in Y:
                        yield y
                else:
                    if type(subindices) is slice:
                        subindices = range(subindices.start,
                                           len(Y) if subindices.stop is None else subindices.stop,
                                           1 if subindices.step is None else subindices.step)
                    for ind in subindices:
                        yield Y[ind]
        else:  # fall back to memory inefficient version
            for y in load(file_name, subindices=subindices):
                yield y
    else:
        logging.error(f"File request:[{file_name}] not found!")
        raise Exception('File not found!')

