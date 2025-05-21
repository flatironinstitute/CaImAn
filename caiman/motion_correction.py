#!/usr/bin/env python

"""

The functions apply_shifts_dft, register_translation, _compute_error, _compute_phasediff, and _upsampled_dft are from
SIMA (https://github.com/losonczylab/sima), licensed under the  GNU GENERAL PUBLIC LICENSE, Version 2, 1991.
These same functions were adapted from sckikit-image, licensed as follows:

Copyright (C) 2011, the scikit-image team
 All rights reserved.

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are
 met:

  1. Redistributions of source code must retain the above copyright
     notice, this list of conditions and the following disclaimer.
  2. Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer in
     the documentation and/or other materials provided with the
     distribution.
  3. Neither the name of skimage nor the names of its contributors may be
     used to endorse or promote products derived from this software without
     specific prior written permission.

 THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
 IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT,
 INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
 IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 POSSIBILITY OF SUCH DAMAGE.
"""

import collections
import cv2
import gc
import itertools
import logging
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import ifftshift
import os
import scipy
import scipy.interpolate
from skimage.transform import resize as resize_sk, rescale as rescale_sk, warp as warp_sk
import sys
import tifffile
from tqdm import tqdm
from typing import Optional, Literal, Union

import caiman
import caiman.base.movies
import caiman.mmapping
import caiman.motion_correction
import caiman.paths

try:
    cv2.setNumThreads(0)
except:
    pass

try:
    profile
except:
    def profile(a): return a

class MotionCorrect(object):
    """
        class implementing motion correction operations
       """

    def __init__(self, fname, min_mov=None, dview=None, max_shifts=(6, 6), niter_rig=1, splits_rig=14, num_splits_to_process_rig=None,
                 strides=(96, 96), overlaps=(32, 32), splits_els=14, num_splits_to_process_els=None,
                 upsample_factor_grid=4, max_deviation_rigid=3, shifts_opencv=True, nonneg_movie=True, gSig_filt=None,
                 use_cuda=False, border_nan=True, pw_rigid=False, num_frames_split=80, var_name_hdf5='mov', is3D=False,
                 indices=(slice(None), slice(None)), shifts_interpolate=False):
        """
        Constructor class for motion correction operations

        Args:
           fname: str
               path to file to motion correct

           min_mov: int16 or float32
               estimated minimum value of the movie to produce an output that is positive

           dview: ipyparallel view object list
               to perform parallel computing, if NOne will operate in single thread

           max_shifts: tuple
               maximum allow rigid shift

           niter_rig:int
               maximum number of iterations rigid motion correction, in general is 1. 0
               will quickly initialize a template with the first frames

           splits_rig: int
            for parallelization split the movies in  num_splits chunks across time

           num_splits_to_process_rig: list,
               if none all the splits are processed and the movie is saved, otherwise at each iteration
               num_splits_to_process_rig are considered

           strides: tuple
               intervals at which patches are laid out for motion correction

           overlaps: tuple
               overlap between patches (size of patch strides+overlaps)

           splits_els':list
               for parallelization split the movies in  num_splits chunks across time

           num_splits_to_process_els: UNUSED, deprecated
               Legacy parameter, does not do anything

           upsample_factor_grid:int,
               upsample factor of shifts per patches to avoid smearing when merging patches

           max_deviation_rigid:int
               maximum deviation allowed for patch with respect to rigid shift

           shifts_opencv: bool
               apply shifts fast way (but smoothing results)

           nonneg_movie: bool
               make the SAVED movie and template mostly nonnegative by removing min_mov from movie

           gSig_filt: list[int], default None
               if specified, indicates the shape of a Gaussian kernel used to perform high-pass spatial filtering of the video frames before performing motion correction. Useful for data with high background to emphasize landmarks. 

           use_cuda : bool (DEPRECATED)
               cuda is no longer supported for motion correction; this kwarg will be removed in a future version of Caiman

           border_nan : bool or string, optional
               Specifies how to deal with borders. (True, False, 'copy', 'min')
               TODO: make this just the bool, and make another variable called
                     border_strategy to hold the how

           pw_rigid: bool, default: False
               flag for performing elastic motion correction when calling motion_correct

           num_frames_split: int, default: 80
               Number of frames in each batch. Used when constructing the options
               through the params object

           var_name_hdf5: str, default: 'mov'
               If loading from hdf5, name of the variable to load

           is3D: bool, default: False
               Flag for 3D motion correction

           indices: tuple(slice), default: (slice(None), slice(None))
               Use that to apply motion correction only on a part of the FOV
            
           shifts_interpolate: bool, default: False
               use patch locations to interpolate shifts rather than just upscaling to size of image (for pw_rigid only)

       Returns:
           self

        """
        logger = logging.getLogger("caiman")
        if 'ndarray' in str(type(fname)) or isinstance(fname, caiman.base.movies.movie):
            mc_tempfile = caiman.paths.fn_relocated('tmp_mov_mot_corr.hdf5')
            if os.path.isfile(mc_tempfile):
                os.remove(mc_tempfile)
            logger.info(f"Creating file for motion correction: {mc_tempfile}")
            caiman.movie(fname).save(mc_tempfile)
            fname = [mc_tempfile]

        if not isinstance(fname, list):
            fname = [fname]

        if isinstance(gSig_filt, tuple):
            gSig_filt = list(gSig_filt) # There are some serializers down the line that choke otherwise

        self.fname = fname
        self.dview = dview
        self.max_shifts = max_shifts
        self.niter_rig = niter_rig
        self.splits_rig = splits_rig
        self.num_splits_to_process_rig = num_splits_to_process_rig
        self.strides = strides
        self.overlaps = overlaps
        self.splits_els = splits_els
        self.upsample_factor_grid = upsample_factor_grid
        self.max_deviation_rigid = max_deviation_rigid
        self.shifts_opencv = bool(shifts_opencv)
        self.min_mov = min_mov
        self.nonneg_movie = nonneg_movie
        self.gSig_filt = gSig_filt
        self.use_cuda = bool(use_cuda)
        self.border_nan = border_nan # FIXME (see comments)
        self.pw_rigid = bool(pw_rigid)
        self.var_name_hdf5 = var_name_hdf5
        self.is3D = bool(is3D)
        self.indices = indices
        self.shifts_interpolate = shifts_interpolate
        if self.use_cuda:
            logger.warning("cuda is no longer supported; this kwarg will be removed in a future version of caiman")

    def __str__(self):
        ret = f"Caiman MotionCorrect Object. Subfields:{list(self.__dict__.keys())}"
        if hasattr(self, 'fname'):
            ret += "MotionCorrect backing fname is {self.fname}"
            if 'tmp_mov_mot_corr' in self.fname:
                ret += "Target data was passed inline"
        return ret

    def __repr__(self):
        return self.__str__()

    def __getitem__(self, idx):
        return getattr(self, idx)
    # We intentionally do not provide __setitem__

    def motion_correct(self, template=None, save_movie=False):
        """general function for performing all types of motion correction. The
        function will perform either rigid or piecewise rigid motion correction
        depending on the attribute self.pw_rigid and will perform high pass
        spatial filtering for determining the motion (used in 1p data) if the
        attribute self.gSig_filt is not None. A template can be passed, and the
        output can be saved as a memory mapped file.

        Args:
            template: ndarray, default: None
                template provided by user for motion correction

            save_movie: bool, default: False
                flag for saving motion corrected file(s) as memory mapped file(s)
        """
        # TODO: Review the docs here
        if self.min_mov is None:
            if self.gSig_filt is None:
                iterator = caiman.base.movies.load_iter(self.fname[0],
                                                    var_name_hdf5=self.var_name_hdf5)
                mi = np.inf
                for _ in range(400):
                    try:
                        mi = min(mi, next(iterator).min()[()])
                    except StopIteration:
                        break
                self.min_mov = mi
            else:
                self.min_mov = np.array([high_pass_filter_space(m_, self.gSig_filt)
                    for m_ in caiman.load(self.fname[0], var_name_hdf5=self.var_name_hdf5,
                                      subindices=slice(400))]).min()

        if self.pw_rigid:
            self.motion_correct_pwrigid(template=template, save_movie=save_movie)
            if self.is3D:
                # TODO - error at this point after saving
                b0 = np.ceil(np.max([np.max(np.abs(self.x_shifts_els)),
                                     np.max(np.abs(self.y_shifts_els)),
                                     np.max(np.abs(self.z_shifts_els))]))
            else:
                b0 = np.ceil(np.maximum(np.max(np.abs(self.x_shifts_els)),
                                    np.max(np.abs(self.y_shifts_els))))
        else:
            self.motion_correct_rigid(template=template, save_movie=save_movie)
            b0 = np.ceil(np.max(np.abs(self.shifts_rig)))
        self.border_to_0 = b0.astype(int)
        self.mmap_file = self.fname_tot_els if self.pw_rigid else self.fname_tot_rig

    def motion_correct_rigid(self, template: Optional[np.ndarray] = None, save_movie=False) -> None:
        """
        Perform rigid motion correction

        Args:
            template: ndarray 2D (or 3D)
                if known, one can pass a template to register the frames to

            save_movie_rigid:Bool
                save the movies vs just get the template

        Important Fields:
            self.fname_tot_rig: name of the mmap file saved

            self.total_template_rig: template updated by iterating  over the chunks

            self.templates_rig: list of templates. one for each chunk

            self.shifts_rig: shifts in x and y (and z if 3D) per frame
        """
        logger = logging.getLogger("caiman")

        logger.debug('Entering Rigid Motion Correction')
        logger.debug(f"{self.min_mov=}")
        self.total_template_rig = template
        self.templates_rig:list = []
        self.fname_tot_rig:list = []
        self.shifts_rig:list = []

        for fname_cur in self.fname:
            _fname_tot_rig, _total_template_rig, _templates_rig, _shifts_rig = motion_correct_batch_rigid(
                fname_cur,
                self.max_shifts,
                dview=self.dview,
                splits=self.splits_rig,
                num_splits_to_process=self.num_splits_to_process_rig,
                num_iter=self.niter_rig,
                template=self.total_template_rig,
                shifts_opencv=self.shifts_opencv,
                save_movie_rigid=save_movie,
                add_to_movie=-self.min_mov,
                nonneg_movie=self.nonneg_movie,
                gSig_filt=self.gSig_filt,
                use_cuda=self.use_cuda,
                border_nan=self.border_nan,
                var_name_hdf5=self.var_name_hdf5,
                is3D=self.is3D,
                indices=self.indices,
                shifts_interpolate=self.shifts_interpolate)
            if template is None:
                self.total_template_rig = _total_template_rig

            self.templates_rig += _templates_rig
            self.fname_tot_rig += [_fname_tot_rig]
            self.shifts_rig += _shifts_rig

    def motion_correct_pwrigid(self, save_movie:bool=True, template: Optional[np.ndarray] = None, show_template:bool=False) -> None:
        """Perform pw-rigid motion correction

        Args:
            save_movie:Bool
                save the movies vs just get the template

            template: ndarray 2D (or 3D)
                if known, one can pass a template to register the frames to

            show_template: bool
                whether to show the updated template at each iteration

        Important Fields:
            self.fname_tot_els: name of the mmap file saved
            self.templates_els: template updated by iterating  over the chunks
            self.x_shifts_els: shifts in x per frame per patch
            self.y_shifts_els: shifts in y per frame per patch
            self.z_shifts_els: shifts in z per frame per patch (if 3D)
            self.coord_shifts_els: coordinates associated to the patch for
            values in x_shifts_els and y_shifts_els (and z_shifts_els if 3D)
            self.total_template_els: list of templates. one for each chunk

        Raises:
            Exception: 'Error: Template contains NaNs, Please review the parameters'
        """
        logger = logging.getLogger("caiman")

        num_iter = 1
        if template is None:
            logger.info('Generating template by rigid motion correction')
            self.motion_correct_rigid()
            self.total_template_els = self.total_template_rig.copy()
        else:
            self.total_template_els = template

        self.fname_tot_els:list = []
        self.templates_els:list = []
        self.x_shifts_els:list = []
        self.y_shifts_els:list = []
        if self.is3D:
            self.z_shifts_els:list = []

        self.coord_shifts_els:list = []
        for name_cur in self.fname:
            _fname_tot_els, new_template_els, _templates_els,\
                _x_shifts_els, _y_shifts_els, _z_shifts_els, _coord_shifts_els = motion_correct_batch_pwrigid(
                    name_cur, self.max_shifts, self.strides, self.overlaps, -self.min_mov,
                    dview=self.dview, upsample_factor_grid=self.upsample_factor_grid,
                    max_deviation_rigid=self.max_deviation_rigid, splits=self.splits_els,
                    num_splits_to_process=None, num_iter=num_iter, template=self.total_template_els,
                    shifts_opencv=self.shifts_opencv, save_movie=save_movie, nonneg_movie=self.nonneg_movie, gSig_filt=self.gSig_filt,
                    use_cuda=self.use_cuda, border_nan=self.border_nan, var_name_hdf5=self.var_name_hdf5, is3D=self.is3D,
                    indices=self.indices, shifts_interpolate=self.shifts_interpolate)
            if not self.is3D:
                if show_template:
                    plt.imshow(new_template_els)
                    plt.pause(.5)  #TODO: figure out if pausing half-second is necessary
            if np.isnan(np.sum(new_template_els)):
                raise Exception(
                    'Template contains NaNs, something went wrong. Reconsider the parameters')

            if template is None:
                self.total_template_els = new_template_els

            self.fname_tot_els += [_fname_tot_els]
            self.templates_els += _templates_els
            self.x_shifts_els += _x_shifts_els
            self.y_shifts_els += _y_shifts_els
            if self.is3D:
                self.z_shifts_els += _z_shifts_els
            self.coord_shifts_els += _coord_shifts_els

    def apply_shifts_movie(self, fname, rigid_shifts: Optional[bool] = None, save_memmap:bool=False,
                           save_base_name:str='MC', order: Literal['C', 'F'] = 'F', remove_min:bool=True):
        """
        Applies shifts found by registering one file to a different file. Useful
        for cases when shifts computed from a structural channel are applied to a
        functional channel. Currently only application of shifts through openCV is
        supported. Returns either caiman.movie or the path to a memory mapped file.

        Args:
            fname: str of list[str]
                name(s) of the movie to motion correct. It should not contain
                nans. All the loadable formats from CaImAn are acceptable

            rigid_shifts: bool (True)
                apply rigid or pw-rigid shifts (must exist in the mc object)
                deprecated (read directly from mc.pw_rigid)

            save_memmap: bool (False)
                flag for saving the resulting file in memory mapped form

            save_base_name: str ['MC']
                base name for memory mapped file name

            order: 'F' or 'C' ['F']
                order of resulting memory mapped file

            remove_min: bool (True)
                If minimum value is negative, subtract it from the data

        Returns:
            m_reg: caiman movie object
                caiman movie object with applied shifts (not memory mapped)
        """
        logger = logging.getLogger("caiman")

        Y = caiman.load(fname).astype(np.float32)
        if remove_min: 
            ymin = Y.min()
            if ymin < 0:
                Y -= Y.min()

        if rigid_shifts is not None:
            logger.warning('The rigid_shifts flag is deprecated and it is ' +
                            'being ignored. The value is read directly from' +
                            ' mc.pw_rigid and is currently set to the opposite' +
                            f' of {self.pw_rigid}')
        
        if self.pw_rigid is False:
            if self.is3D:
                m_reg = [apply_shifts_dft(img, (sh[0], sh[1], sh[2]), 0,
                                          is_freq=False, border_nan=self.border_nan)
                         for img, sh in zip(Y, self.shifts_rig)]
            elif self.shifts_opencv:
                m_reg = [apply_shift_iteration(img, shift, border_nan=self.border_nan)
                         for img, shift in zip(Y, self.shifts_rig)]
            else:
                m_reg = [apply_shifts_dft(img, (
                    sh[0], sh[1]), 0, is_freq=False, border_nan=self.border_nan) for img, sh in zip(
                    Y, self.shifts_rig)]
        else:
            # take potential upsampling into account when recreating patch grid
            dims = Y.shape[1:]
            patch_centers = get_patch_centers(dims, overlaps=self.overlaps, strides=self.strides,
                                              shifts_opencv=self.shifts_opencv, upsample_factor_grid=self.upsample_factor_grid)
            if self.is3D:
                # x_shifts_els and y_shifts_els are switched intentionally
                m_reg = [
                    apply_pw_shifts_remap_3d(img, shifts_y=-x_shifts, shifts_x=-y_shifts, shifts_z=-z_shifts,
                                             patch_centers=patch_centers, border_nan=self.border_nan,
                                             shifts_interpolate=self.shifts_interpolate)
                    for img, x_shifts, y_shifts, z_shifts in zip(Y, self.x_shifts_els, self.y_shifts_els, self.z_shifts_els)
                ]

            else:
                # x_shifts_els and y_shifts_els are switched intentionally
                m_reg = [
                    apply_pw_shifts_remap_2d(img, shifts_y=-x_shifts, shifts_x=-y_shifts, patch_centers=patch_centers,
                                             border_nan=self.border_nan, shifts_interpolate=self.shifts_interpolate)
                    for img, x_shifts, y_shifts in zip(Y, self.x_shifts_els, self.y_shifts_els)
                ]

        del Y
        m_reg = np.stack(m_reg, axis=0)
        if save_memmap:
            dims = m_reg.shape
            fname_tot = caiman.paths.memmap_frames_filename(save_base_name, dims[1:], dims[0], order)
            big_mov = np.memmap(fname_tot, mode='w+', dtype=np.float32,
                        shape=caiman.mmapping.prepare_shape((np.prod(dims[1:]), dims[0])), order=order)
            big_mov[:] = np.reshape(m_reg.transpose(1, 2, 0), (np.prod(dims[1:]), dims[0]), order='F')
            big_mov.flush()
            del big_mov
            return fname_tot
        else:
            return caiman.movie(m_reg)

def apply_shift_iteration(img, shift, border_nan=False, border_type=cv2.BORDER_REFLECT):
    # Used by MotionCorrect.apply_shifts_movie(), tile_and_correct(), and apply_shift_online()
    # Applies an xy shift to one iterable unit of a movie.

    sh_x_n, sh_y_n = shift
    w_i, h_i = img.shape
    M = np.array([[1, 0, sh_y_n], [0, 1, sh_x_n]], dtype=np.float32)
    min_, max_ = np.nanmin(img), np.nanmax(img)
    img = np.clip(cv2.warpAffine(img, M, (h_i, w_i),
                                 flags=cv2.INTER_CUBIC, borderMode=border_type), min_, max_)
    if border_nan is not False:
        max_w, max_h, min_w, min_h = 0, 0, 0, 0
        max_h, max_w = np.ceil(np.maximum(
            (max_h, max_w), shift)).astype(int)
        min_h, min_w = np.floor(np.minimum(
            (min_h, min_w), shift)).astype(int)
        if border_nan is True:
            img[:max_h, :] = np.nan
            if min_h < 0:
                img[min_h:, :] = np.nan
            img[:, :max_w] = np.nan
            if min_w < 0:
                img[:, min_w:] = np.nan
        elif border_nan == 'min':
            img[:max_h, :] = min_
            if min_h < 0:
                img[min_h:, :] = min_
            img[:, :max_w] = min_
            if min_w < 0:
                img[:, min_w:] = min_
        elif border_nan == 'copy':
            if max_h > 0:
                img[:max_h] = img[max_h]
            if min_h < 0:
                img[min_h:] = img[min_h-1]
            if max_w > 0:
                img[:, :max_w] = img[:, max_w, np.newaxis]
            if min_w < 0:
                img[:, min_w:] = img[:, min_w-1, np.newaxis]
        else:
            logging.warning(f'Unknown value of border_nan ({border_nan}); treating as False')

    return img

def apply_shift_online(movie_iterable, xy_shifts, save_base_name=None, order: Literal['C', 'F'] = 'F'):
    """
    Applies rigid shifts to a loaded movie. Useful when processing a dataset
    with CaImAn online and you want to obtain the registered movie after
    processing is finished or when operating with dual channels.
    When save_base_name is None the registered file is returned as a caiman
    movie. Otherwise a memory mapped file is saved.
    Currently only rigid shifts are supported supported.

    Args:
        movie_iterable: caiman.movie or np.array
            Movie to be registered in T x X x Y format

        xy_shifts: list
            list of shifts to be applied

        save_base_name: str or None
            prefix for memory mapped file otherwise registered file is returned
            to memory

        order: 'F' or 'C'
            order of memory mapped file

    Returns:
        name of registered memory mapped file if save_base_name is not None,
        otherwise registered file
    """
    # todo use for non rigid shifts

    if len(movie_iterable) != len(xy_shifts):
        raise Exception('Number of shifts does not match movie length!')
    count = 0
    new_mov = []
    dims = (len(movie_iterable),) + movie_iterable[0].shape  # TODO: Refactor so length is either tracked separately or is last part of tuple

    if save_base_name is not None:
        fname_tot = caiman.paths.memmap_frames_filename(save_base_name, dims[1:], dims[0], order)
        fname_tot = caiman.paths.fn_relocated(fname_tot)
        big_mov = np.memmap(fname_tot, mode='w+', dtype=np.float32,
                            shape=caiman.mmapping.prepare_shape((np.prod(dims[1:]), dims[0])), order=order)

    for page, shift in zip(movie_iterable, xy_shifts):
        if 'tifffile' in str(type(movie_iterable[0])):
            page = page.asarray()

        img = np.array(page, dtype=np.float32)
        new_img = apply_shift_iteration(img, shift)
        if save_base_name is not None:
            big_mov[:, count] = np.reshape(
                new_img, np.prod(dims[1:]), order='F')
        else:
            new_mov.append(new_img)
        count += 1

    if save_base_name is not None:
        big_mov.flush()
        del big_mov
        return fname_tot
    else:
        return np.array(new_mov)

def motion_correct_oneP_rigid(
        filename,
        gSig_filt,
        max_shifts,
        dview=None,
        splits_rig=10,
        save_movie=True,
        border_nan=True):
    '''Perform rigid motion correction on one photon imaging movies

    Args:
        filename: str
            name of the file to correct
        gSig_filt:
            size of the filter. If algorithm does not work change this parameters
        max_shifts: tuple of ints
            max shifts in x and y allowed
        dview:
            handle to cluster
        splits_rig: int
            number of chunks for parallelizing motion correction (remember that it should hold that length_movie/num_splits_to_process_rig>100)
        save_movie: bool
            whether to save the movie in memory mapped format
        border_nan : bool or string, optional
            Specifies how to deal with borders. (True, False, 'copy', 'min')        

    Returns:
        Motion correction object
    '''
    min_mov = np.array([caiman.motion_correction.high_pass_filter_space(
        m_, gSig_filt) for m_ in caiman.load(filename[0], subindices=range(400))]).min()
    new_templ = None

    # TODO: needinfo how the classes works
    mc = MotionCorrect(
        filename,
        min_mov,
        dview=dview,
        max_shifts=max_shifts,
        niter_rig=1,
        splits_rig=splits_rig,
        num_splits_to_process_rig=None,
        shifts_opencv=True,
        nonneg_movie=True,
        gSig_filt=gSig_filt,
        border_nan=border_nan,
        is3D=False)

    mc.motion_correct_rigid(save_movie=save_movie, template=new_templ)

    return mc

def motion_correct_oneP_nonrigid(
        filename,
        gSig_filt,
        max_shifts,
        strides,
        overlaps,
        splits_els,
        upsample_factor_grid,
        max_deviation_rigid,
        dview=None,
        splits_rig=10,
        save_movie=True,
        new_templ=None,
        border_nan=True):
    '''Perform rigid motion correction on one photon imaging movies

    Args:
        filename: str
            name of the file to correct
        gSig_filt:
            size of the filter. If algorithm does not work change this parameters
        max_shifts: tuple of ints
            max shifts in x and y allowed
        dview:
            handle to cluster
        splits_rig: int
            number of chunks for parallelizing motion correction (remember that it should hold that length_movie/num_splits_to_process_rig>100)
        save_movie: bool
            whether to save the movie in memory mapped format
        border_nan : bool or string, optional
            specifies how to deal with borders. (True, False, 'copy', 'min')

    Returns:
        Motion correction object
    '''

    if new_templ is None:
        min_mov = np.array([caiman.motion_correction.high_pass_filter_space(
            m_, gSig_filt) for m_ in caiman.load(filename, subindices=range(400))]).min()
    else:
        min_mov = np.min(new_templ)

    # TODO: needinfo how the classes works
    mc = MotionCorrect(
        filename,
        min_mov,
        dview=dview,
        max_shifts=max_shifts,
        niter_rig=1,
        splits_rig=splits_rig,
        num_splits_to_process_rig=None,
        shifts_opencv=True,
        nonneg_movie=True,
        gSig_filt=gSig_filt,
        strides=strides,
        overlaps=overlaps,
        splits_els=splits_els,
        upsample_factor_grid=upsample_factor_grid,
        max_deviation_rigid=max_deviation_rigid,
        border_nan=border_nan,
        is3D=False)

    mc.motion_correct_pwrigid(save_movie=True, template=new_templ)
    return mc

def motion_correct_online_multifile(list_files, add_to_movie, order='C', **kwargs):
    logger = logging.getLogger("caiman")
    # todo todocument

    kwargs['order'] = order
    all_names = []
    all_shifts = []
    all_xcorrs = []
    all_templates = []
    template = None
    kwargs_ = kwargs.copy()
    kwargs_['order'] = order
    total_frames = 0
    for file_ in list_files:
        logger.info(f'Processing: {file_}')
        kwargs_['template'] = template
        kwargs_['save_base_name'] = os.path.splitext(file_)[0]
        tffl = tifffile.TiffFile(file_)
        shifts, xcorrs, template, fname_tot = motion_correct_online(
            tffl, add_to_movie, **kwargs_)[0:4]
        all_names.append(fname_tot)
        all_shifts.append(shifts)
        all_xcorrs.append(xcorrs)
        all_templates.append(template)
        total_frames = total_frames + len(shifts)

    return all_names, all_shifts, all_xcorrs, all_templates

def motion_correct_online(movie_iterable, add_to_movie, max_shift_w=25, max_shift_h=25, save_base_name=None, order: Literal['F', 'C']='C',
                          init_frames_template=100, show_movie=False, bilateral_blur=False, template=None, min_count=1000,
                          border_to_0=0, n_iter=1, remove_blanks=False, show_template=False, return_mov=False,
                          use_median_as_template=False):
    # todo todocument
    logger = logging.getLogger("caiman")

    shifts = []  # store the amount of shift in each frame
    xcorrs = []
    if remove_blanks and n_iter == 1:
        raise Exception(
            'In order to remove blanks you need at least two iterations n_iter=2')

    if 'tifffile' in str(type(movie_iterable[0])):
        if len(movie_iterable) == 1:
            logger.warning(
                '******** WARNING ****** NEED TO LOAD IN MEMORY SINCE SHAPE OF PAGE IS THE FULL MOVIE')
            movie_iterable = movie_iterable.asarray()
            init_mov = movie_iterable[:init_frames_template]
        else:
            init_mov = [m.asarray()
                        for m in movie_iterable[:init_frames_template]]
    else:
        init_mov = movie_iterable[slice(0, init_frames_template, 1)]

    dims = (len(movie_iterable),) + movie_iterable[0].shape # TODO: Refactor so length is either tracked separately or is last part of tuple
    logger.debug(f"dimensions: {dims}")

    if use_median_as_template:
        template = bin_median(movie_iterable)

    if template is None:
        template = bin_median(init_mov)
        count = init_frames_template
        if np.percentile(template, 1) + add_to_movie < - 10:
            raise Exception(
                'Movie too negative, You need to add a larger value to the movie (add_to_movie)')
        template = np.array(template + add_to_movie, dtype=np.float32)
    else:
        if np.percentile(template, 1) < - 10:
            raise Exception(
                'Movie too negative, You need to add a larger value to the movie (add_to_movie)')
        count = min_count

    min_mov = 0
    buffer_size_frames = 100
    buffer_size_template = 100
    buffer_frames:collections.deque = collections.deque(maxlen=buffer_size_frames)
    buffer_templates:collections.deque = collections.deque(maxlen=buffer_size_template)
    max_w, max_h, min_w, min_h = 0, 0, 0, 0

    big_mov = None
    if return_mov:
        mov:Optional[list] = []
    else:
        mov = None

    for n in range(n_iter):
        if n > 0:
            count = init_frames_template

        if (save_base_name is not None) and (big_mov is None) and (n_iter == (n + 1)):

            if remove_blanks:
                dims = (dims[0], dims[1] + min_h -
                        max_h, dims[2] + min_w - max_w)

            fname_tot:Optional[str] = caiman.paths.memmap_frames_filename(save_base_name, dims[1:], dims[0], order)
            big_mov = np.memmap(fname_tot, mode='w+', dtype=np.float32,
                                shape=caiman.mmapping.prepare_shape((np.prod(dims[1:]), dims[0])), order=order)

        else:
            fname_tot = None

        shifts_tmp = []
        xcorr_tmp = []
        for idx_frame, page in enumerate(movie_iterable):

            if 'tifffile' in str(type(movie_iterable[0])):
                page = page.asarray()

            img = np.array(page, dtype=np.float32)
            img = img + add_to_movie

            new_img, template_tmp, shift, avg_corr = motion_correct_iteration(
                img, template, count, max_shift_w=max_shift_w, max_shift_h=max_shift_h, bilateral_blur=bilateral_blur)

            max_h, max_w = np.ceil(np.maximum(
                (max_h, max_w), shift)).astype(int)
            min_h, min_w = np.floor(np.minimum(
                (min_h, min_w), shift)).astype(int)

            if count < (buffer_size_frames + init_frames_template):
                template_old = template
                template = template_tmp
            else:
                template_old = template
            buffer_frames.append(new_img)

            if count % 100 == 0:
                if count >= (buffer_size_frames + init_frames_template):
                    buffer_templates.append(np.mean(buffer_frames, 0))
                    template = np.median(buffer_templates, 0)

                if show_template:
                    plt.cla()
                    plt.imshow(template, cmap='gray', vmin=250,
                              vmax=350, interpolation='none')
                    plt.pause(.001)

                logger.debug('Relative change in template:' + str(
                    np.sum(np.abs(template - template_old)) / np.sum(np.abs(template))))
                logger.debug('Iteration:' + str(count))

            if border_to_0 > 0:
                new_img[:border_to_0, :] = min_mov
                new_img[:, :border_to_0] = min_mov
                new_img[:, -border_to_0:] = min_mov
                new_img[-border_to_0:, :] = min_mov

            shifts_tmp.append(shift)
            xcorr_tmp.append(avg_corr)

            if remove_blanks and n > 0 and (n_iter == (n + 1)):

                new_img = new_img[max_h:, :]
                if min_h < 0:
                    new_img = new_img[:min_h, :]
                new_img = new_img[:, max_w:]
                if min_w < 0:
                    new_img = new_img[:, :min_w]

            if (save_base_name is not None) and (n_iter == (n + 1)):
                big_mov[:, idx_frame] = np.reshape(new_img, np.prod(dims[1:]), order='F') # type: ignore
                                                                                          # mypy cannot prove that big_mov is not still None

            if mov is not None and (n_iter == (n + 1)):
                mov.append(new_img)

            if show_movie:
                cv2.imshow('frame', new_img / 500)
                logger.info(shift)
                if not np.any(np.remainder(shift, 1) == (0, 0)):
                    cv2.waitKey(int(1. / 500 * 1000))

            count += 1
        shifts.append(shifts_tmp)
        xcorrs.append(xcorr_tmp)

    if save_base_name is not None:
        logger.debug('Flushing memory')
        big_mov.flush() # type: ignore # mypy cannot prove big_mov is not still None
        del big_mov
        gc.collect()

    if mov is not None:
        mov = np.dstack(mov).transpose([2, 0, 1])

    return shifts, xcorrs, template, fname_tot, mov

def motion_correct_iteration(img, template, frame_num, max_shift_w=25,
                             max_shift_h=25, bilateral_blur=False, diameter=10, sigmaColor=10000, sigmaSpace=0):
    # todo todocument
    h_i, w_i = template.shape
    ms_h = max_shift_h
    ms_w = max_shift_w

    if bilateral_blur:
        img = cv2.bilateralFilter(img, diameter, sigmaColor, sigmaSpace)
    templ_crop = template[max_shift_h:h_i - max_shift_h,
                          max_shift_w:w_i - max_shift_w].astype(np.float32)
    res = cv2.matchTemplate(img, templ_crop, cv2.TM_CCORR_NORMED)

    top_left = cv2.minMaxLoc(res)[3]
    avg_corr = np.max(res)
    sh_y, sh_x = top_left

    if (0 < top_left[1] < 2 * ms_h - 1) & (0 < top_left[0] < 2 * ms_w - 1):
        # if max is internal, check for subpixel shift using gaussian
        # peak registration
        log_xm1_y = np.log(res[sh_x - 1, sh_y])
        log_xp1_y = np.log(res[sh_x + 1, sh_y])
        log_x_ym1 = np.log(res[sh_x, sh_y - 1])
        log_x_yp1 = np.log(res[sh_x, sh_y + 1])
        four_log_xy = 4 * np.log(res[sh_x, sh_y])

        sh_x_n = -(sh_x - ms_h + ((log_xm1_y - log_xp1_y) /
                                   (2 * log_xm1_y - four_log_xy + 2 * log_xp1_y)))
        sh_y_n = -(sh_y - ms_w + ((log_x_ym1 - log_x_yp1) /
                                   (2 * log_x_ym1 - four_log_xy + 2 * log_x_yp1)))
    else:
        sh_x_n = -(sh_x - ms_h)
        sh_y_n = -(sh_y - ms_w)

    M = np.float32([[1, 0, sh_y_n], [0, 1, sh_x_n]])
    min_, max_ = np.min(img), np.max(img)
    new_img = np.clip(cv2.warpAffine(
        img, M, (w_i, h_i), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT), min_, max_)

    new_templ = template * frame_num / \
        (frame_num + 1) + 1. / (frame_num + 1) * new_img
    shift = [sh_x_n, sh_y_n]

    return new_img, new_templ, shift, avg_corr

@profile
def motion_correct_iteration_fast(img, template, max_shift_w=10, max_shift_h=10):
    """ For using in online realtime scenarios """
    h_i, w_i = template.shape
    ms_h = max_shift_h
    ms_w = max_shift_w

    templ_crop = template[max_shift_h:h_i - max_shift_h,
                          max_shift_w:w_i - max_shift_w].astype(np.float32)

    res = cv2.matchTemplate(img, templ_crop, cv2.TM_CCORR_NORMED)
    top_left = cv2.minMaxLoc(res)[3]

    sh_y, sh_x = top_left

    if (0 < top_left[1] < 2 * ms_h - 1) & (0 < top_left[0] < 2 * ms_w - 1):
        # if max is internal, check for subpixel shift using gaussian
        # peak registration
        log_xm1_y = np.log(res[sh_x - 1, sh_y])
        log_xp1_y = np.log(res[sh_x + 1, sh_y])
        log_x_ym1 = np.log(res[sh_x, sh_y - 1])
        log_x_yp1 = np.log(res[sh_x, sh_y + 1])
        four_log_xy = 4 * np.log(res[sh_x, sh_y])

        sh_x_n = -(sh_x - ms_h + ((log_xm1_y - log_xp1_y) /
                                   (2 * log_xm1_y - four_log_xy + 2 * log_xp1_y)))
        sh_y_n = -(sh_y - ms_w + ((log_x_ym1 - log_x_yp1) /
                                   (2 * log_x_ym1 - four_log_xy + 2 * log_x_yp1)))
    else:
        sh_x_n = -(sh_x - ms_h)
        sh_y_n = -(sh_y - ms_w)

    M = np.float32([[1, 0, sh_y_n], [0, 1, sh_x_n]])

    new_img = cv2.warpAffine(
        img, M, (w_i, h_i), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)

    shift = [sh_x_n, sh_y_n]

    return new_img, shift

def bin_median(mat, window=10, exclude_nans=True):
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

    T, d1, d2 = np.shape(mat)
    if T < window:
        window = T
    num_windows = int(T // window)
    num_frames = num_windows * window
    if exclude_nans:
        img = np.nanmedian(np.nanmean(np.reshape(
            mat[:num_frames], (window, num_windows, d1, d2)), axis=0), axis=0)
    else:
        img = np.median(np.mean(np.reshape(
            mat[:num_frames], (window, num_windows, d1, d2)), axis=0), axis=0)

    return img

def bin_median_3d(mat, window=10, exclude_nans=True):
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

    T, d1, d2, d3 = np.shape(mat)
    if T < window:
        window = T
    num_windows = int(T // window)
    num_frames = num_windows * window
    if exclude_nans:
        img = np.nanmedian(np.nanmean(np.reshape(
            mat[:num_frames], (window, num_windows, d1, d2, d3)), axis=0), axis=0)
    else:
        img = np.median(np.mean(np.reshape(
            mat[:num_frames], (window, num_windows, d1, d2, d3)), axis=0), axis=0)

    return img

def process_movie_parallel(arg_in):
    #todo: todocument
    logger = logging.getLogger("caiman")
    fname, fr, margins_out, template, max_shift_w, max_shift_h, remove_blanks, apply_smooth, save_hdf5 = arg_in

    if template is not None:
        if isinstance(template, str):
            if os.path.exists(template):
                template = caiman.load(template, fr=1)
            else:
                raise Exception('Path to template does not exist:' + template)

    type_input = str(type(fname))
    if 'movie' in type_input:
        Yr = fname

    elif 'ndarray' in type_input:
        Yr = caiman.movie(np.array(fname, dtype=np.float32), fr=fr)
    elif isinstance(fname, str):
        Yr = caiman.load(fname, fr=fr)
    else:
        raise Exception('Unknown input type:' + type_input)

    if Yr.ndim > 1:
        if apply_smooth:
            Yr = Yr.bilateral_blur_2D(
                diameter=10, sigmaColor=10000, sigmaSpace=0)

        if margins_out != 0:
            Yr = Yr[:, margins_out:-margins_out, margins_out:-
                    margins_out]  # borders create troubles

        Yr, shifts, xcorrs, template = Yr.motion_correct(max_shift_w=max_shift_w, max_shift_h=max_shift_h,
                                                         method='opencv', template=template, remove_blanks=remove_blanks)

        if ('movie' in type_input) or ('ndarray' in type_input):
            return Yr, shifts, xcorrs, template

        else:

            #            logger.debug('median computing')
            template = Yr.bin_median()
            idx_dot = len(fname.split('.')[-1])
            if save_hdf5:
                Yr.save(fname[:-idx_dot] + 'hdf5')
            np.savez(fname[:-idx_dot] + 'npz', shifts=shifts,
                     xcorrs=xcorrs, template=template)
            del Yr
            return fname[:-idx_dot]
    else:
        return None

def motion_correct_parallel(file_names, fr=10, template=None, margins_out=0,
                            max_shift_w=5, max_shift_h=5, remove_blanks=False, apply_smooth=False, dview=None, save_hdf5=True):
    """motion correct many movies usingthe ipyparallel cluster

    Args:
        file_names: list of strings
            names of he files to be motion corrected

        fr: double
            fr parameters for calcblitz movie

        margins_out: int
            number of pixels to remove from the borders

    Returns:
        base file names of the motion corrected files:list[str]
    """
    logger = logging.getLogger("caiman")
    args_in = []
    for file_idx, f in enumerate(file_names):
        if isinstance(template, list):
            args_in.append((f, fr, margins_out, template[file_idx], max_shift_w, max_shift_h,
                            remove_blanks, apply_smooth, save_hdf5))
        else:
            args_in.append((f, fr, margins_out, template, max_shift_w,
                            max_shift_h, remove_blanks, apply_smooth, save_hdf5))

    try:
        if dview is not None:
            if 'multiprocessing' in str(type(dview)):
                file_res = dview.map_async(
                    process_movie_parallel, args_in).get(4294967)
            else:
                file_res = dview.map_sync(process_movie_parallel, args_in)
                dview.results.clear()
        else:
            file_res = list(map(process_movie_parallel, args_in))

    except:
        try:
            if (dview is not None) and 'multiprocessing' not in str(type(dview)):
                dview.results.clear()

        except UnboundLocalError:
            logger.error('could not close client')

        raise

    return file_res

def _upsampled_dft(data, upsampled_region_size,
                   upsample_factor=1, axis_offsets=None):
    """
    adapted from SIMA (https://github.com/losonczylab) and the scikit-image (http://scikit-image.org/) package.

    Unless otherwise specified by LICENSE.txt files in individual
    directories, all code is

    Copyright (C) 2011, the scikit-image team
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:

     1. Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.
     2. Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in
        the documentation and/or other materials provided with the
        distribution.
     3. Neither the name of skimage nor the names of its contributors may be
        used to endorse or promote products derived from this software without
        specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
    IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT,
    INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
    HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
    STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
    IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
    POSSIBILITY OF SUCH DAMAGE.

    Upsampled DFT by matrix multiplication.

    This code is intended to provide the same result as if the following
    operations were performed:
        - Embed the array "data" in an array that is ``upsample_factor`` times
          larger in each dimension.  ifftshift to bring the center of the
          image to (1,1).
        - Take the FFT of the larger array.
        - Extract an ``[upsampled_region_size]`` region of the result, starting
          with the ``[axis_offsets+1]`` element.

    It achieves this result by computing the DFT in the output array without
    the need to zeropad. Much faster and memory efficient than the zero-padded
    FFT approach if ``upsampled_region_size`` is much smaller than
    ``data.size * upsample_factor``.

    Args:
        data : 2D ndarray
            The input data array (DFT of original data) to upsample.

        upsampled_region_size : integer or tuple of integers, optional
            The size of the region to be sampled.  If one integer is provided, it
            is duplicated up to the dimensionality of ``data``.

        upsample_factor : integer, optional
            The upsampling factor.  Defaults to 1.

        axis_offsets : tuple of integers, optional
            The offsets of the region to be sampled.  Defaults to None (uses
            image center)

    Returns:
        output : 2D ndarray
                The upsampled DFT of the specified region.
    """
    # if people pass in an integer, expand it to a list of equal-sized sections
    if not hasattr(upsampled_region_size, "__iter__"):
        upsampled_region_size = [upsampled_region_size, ] * data.ndim
    else:
        if len(upsampled_region_size) != data.ndim:
            raise ValueError("shape of upsampled region sizes must be equal to input data's number of dimensions.")

    if axis_offsets is None:
        axis_offsets = [0, ] * data.ndim
    else:
        if len(axis_offsets) != data.ndim:
            raise ValueError("number of axis offsets must be equal to input data's number of dimensions.")

    col_kernel = np.exp(
        (-1j * 2 * np.pi / (data.shape[1] * upsample_factor)) *
        (ifftshift(np.arange(data.shape[1]))[:, None] -
         np.floor(data.shape[1] // 2)).dot(
             np.arange(upsampled_region_size[1])[None, :] - axis_offsets[1])
    )
    row_kernel = np.exp(
        (-1j * 2 * np.pi / (data.shape[0] * upsample_factor)) *
        (np.arange(upsampled_region_size[0])[:, None] - axis_offsets[0]).dot(
            ifftshift(np.arange(data.shape[0]))[None, :] -
            np.floor(data.shape[0] // 2))
    )

    if data.ndim > 2:
        pln_kernel = np.exp(
        (-1j * 2 * np.pi / (data.shape[2] * upsample_factor)) *
        (np.arange(upsampled_region_size[2])[:, None] - axis_offsets[2]).dot(
                ifftshift(np.arange(data.shape[2]))[None, :] -
                np.floor(data.shape[2] // 2)))

    # output = np.tensordot(np.tensordot(row_kernel,data,axes=[1,0]),col_kernel,axes=[1,0])
    output = np.tensordot(row_kernel, data, axes = [1,0])
    output = np.tensordot(output, col_kernel, axes = [1,0])

    if data.ndim > 2:
        output = np.tensordot(output, pln_kernel, axes = [1,1])
    #output = row_kernel.dot(data).dot(col_kernel)
    return output


def _compute_phasediff(cross_correlation_max):
    """
    Compute global phase difference between the two images (should be zero if images are non-negative).

    Args:
        cross_correlation_max : complex
            The complex value of the cross correlation at its maximum point.
    """
    return np.arctan2(cross_correlation_max.imag, cross_correlation_max.real)


def _compute_error(cross_correlation_max, src_amp, target_amp):
    """
    Compute RMS error metric between ``src_image`` and ``target_image``.

    Args:
        cross_correlation_max : complex
            The complex value of the cross correlation at its maximum point.

        src_amp : float
            The normalized average image intensity of the source image

        target_amp : float
            The normalized average image intensity of the target image
    """
    error = 1.0 - cross_correlation_max * cross_correlation_max.conj() /\
        (src_amp * target_amp)
    return np.sqrt(np.abs(error))

def register_translation_3d(src_image, target_image, upsample_factor = 1,
                            space = "real", shifts_lb = None, shifts_ub = None,
                            max_shifts = [10,10,10]):

    """
    Simple script for registering translation in 3D using an FFT approach.
    
    Args:
        src_image : ndarray
            Reference image.

        target_image : ndarray
            Image to register.  Must be same dimensionality as ``src_image``.

        upsample_factor : int, optional
            Upsampling factor. Images will be registered to within
            ``1 / upsample_factor`` of a pixel. For example
            ``upsample_factor == 20`` means the images will be registered
            within 1/20th of a pixel.  Default is 1 (no upsampling)

        space : string, one of "real" or "fourier"
            Defines how the algorithm interprets input data.  "real" means data
            will be FFT'd to compute the correlation, while "fourier" data will
            bypass FFT of input data.  Case insensitive.

    Returns:
        shifts : ndarray
            Shift vector (in pixels) required to register ``target_image`` with
            ``src_image``.  Axis ordering is consistent with numpy (e.g. Z, Y, X)

        error : float
            Translation invariant normalized RMS error between ``src_image`` and
            ``target_image``.

        phasediff : float
            Global phase difference between the two images (should be
            zero if images are non-negative).

    Raises:
     NotImplementedError "Error: register_translation_3d only supports "
                                  "subpixel registration for 3D images"

     ValueError "Error: images must really be same size for "
                         "register_translation_3d"

     ValueError "Error: register_translation_3d only knows the \"real\" "
                         "and \"fourier\" values for the ``space`` argument."

    """

    # images must be the same shape
    if src_image.shape != target_image.shape:
        raise ValueError("Error: images must really be same size for register_translation_3d")

    # only 3D data makes sense right now
    if src_image.ndim != 3 and upsample_factor > 1:
        raise NotImplementedError("Error: register_translation_3d only supports subpixel registration for 3D images")

    # assume complex data is already in Fourier space
    if space.lower() == 'fourier':
        src_freq = src_image
        target_freq = target_image
    # real data needs to be fft'd.
    elif space.lower() == 'real':
        src_image_cpx = np.array(
            src_image, dtype=np.complex64, copy=False)
        target_image_cpx = np.array(
            target_image, dtype=np.complex64, copy=False)
        src_freq = np.fft.fftn(src_image_cpx)
        target_freq = np.fft.fftn(target_image_cpx)
    else:
        raise ValueError('Error: register_translation_3d only knows the "real" and "fourier" values for the ``space`` argument.')

    shape = src_freq.shape
    image_product = src_freq * target_freq.conj()
    cross_correlation = np.fft.ifftn(image_product)
    new_cross_corr = np.abs(cross_correlation)

    CCmax = cross_correlation.max()

    del cross_correlation

    if (shifts_lb is not None) or (shifts_ub is not None):
        # TODO this will fail if only one is not none - should this be an "and"?
        if (shifts_lb[0] < 0) and (shifts_ub[0] >= 0):
            new_cross_corr[shifts_ub[0]:shifts_lb[0], :, :] = 0
        else:
            new_cross_corr[:shifts_lb[0], :, :] = 0
            new_cross_corr[shifts_ub[0]:, :, :] = 0

        if (shifts_lb[1] < 0) and (shifts_ub[1] >= 0):
            new_cross_corr[:, shifts_ub[1]:shifts_lb[1], :] = 0
        else:
            new_cross_corr[:, :shifts_lb[1], :] = 0
            new_cross_corr[:, shifts_ub[1]:, :] = 0

        if (shifts_lb[2] < 0) and (shifts_ub[2] >= 0):
            new_cross_corr[:, :, shifts_ub[2]:shifts_lb[2]] = 0
        else:
            new_cross_corr[:, :, :shifts_lb[2]] = 0
            new_cross_corr[:, :, shifts_ub[2]:] = 0
    else:
        new_cross_corr[max_shifts[0]:-max_shifts[0], :, :] = 0
        new_cross_corr[:, max_shifts[1]:-max_shifts[1], :] = 0
        new_cross_corr[:, :, max_shifts[2]:-max_shifts[2]] = 0

    maxima = np.unravel_index(np.argmax(new_cross_corr), new_cross_corr.shape)
    midpoints = np.array([np.fix(axis_size//2) for axis_size in shape])

    shifts = np.array(maxima, dtype=np.float32)
    shifts[shifts > midpoints] -= np.array(shape)[shifts > midpoints]


    if upsample_factor > 1:

        shifts = np.round(shifts * upsample_factor) / upsample_factor
        upsampled_region_size = np.ceil(upsample_factor * 1.5)
        # Center of output array at dftshift + 1
        dftshift = np.fix(upsampled_region_size / 2.)
        upsample_factor = np.array(upsample_factor, dtype=np.float64)
        normalization = (src_freq.size * upsample_factor ** 2)
        # Matrix multiply DFT around the current shift estimate
        sample_region_offset = dftshift - shifts * upsample_factor

        cross_correlation = _upsampled_dft(image_product.conj(),
                                           upsampled_region_size,
                                           upsample_factor,
                                           sample_region_offset).conj()
        cross_correlation /= normalization
        # Locate maximum and map back to original pixel grid
        maxima = np.array(np.unravel_index(
            np.argmax(np.abs(cross_correlation)),
            cross_correlation.shape),
            dtype=np.float64)
        maxima -= dftshift
        shifts = shifts + (maxima / upsample_factor)
        CCmax = cross_correlation.max()

    for dim in range(src_freq.ndim):
        if shape[dim] == 1:
            shifts[dim] = 0

    return shifts, src_freq, _compute_phasediff(CCmax)

def register_translation(src_image, target_image, upsample_factor=1,
                         space="real", shifts_lb=None, shifts_ub=None, max_shifts=(10, 10),
                         use_cuda=False):
    """

    adapted from SIMA (https://github.com/losonczylab) and the
    scikit-image (http://scikit-image.org/) package.


    Unless otherwise specified by LICENSE.txt files in individual
    directories, all code is

    Copyright (C) 2011, the scikit-image team
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:

     1. Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.
     2. Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in
        the documentation and/or other materials provided with the
        distribution.
     3. Neither the name of skimage nor the names of its contributors may be
        used to endorse or promote products derived from this software without
        specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
    IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT,
    INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
    HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
    STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
    IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
    POSSIBILITY OF SUCH DAMAGE.
    Efficient subpixel image translation registration by cross-correlation.

    This code gives the same precision as the FFT upsampled cross-correlation
    in a fraction of the computation time and with reduced memory requirements.
    It obtains an initial estimate of the cross-correlation peak by an FFT and
    then refines the shift estimation by upsampling the DFT only in a small
    neighborhood of that estimate by means of a matrix-multiply DFT.

    Args:
        src_image : ndarray
            Reference image.

        target_image : ndarray
            Image to register.  Must be same dimensionality as ``src_image``.

        upsample_factor : int, optional
            Upsampling factor. Images will be registered to within
            ``1 / upsample_factor`` of a pixel. For example
            ``upsample_factor == 20`` means the images will be registered
            within 1/20th of a pixel.  Default is 1 (no upsampling)

        space : string, one of "real" or "fourier"
            Defines how the algorithm interprets input data.  "real" means data
            will be FFT'd to compute the correlation, while "fourier" data will
            bypass FFT of input data.  Case insensitive.

        use_cuda : bool (DEPRECATED)
            cuda is no longer supported for motion correction; this kwarg will be removed in a future version of Caiman

    Returns:
        shifts : ndarray
            Shift vector (in pixels) required to register ``target_image`` with
            ``src_image``.  Axis ordering is consistent with numpy (e.g. Z, Y, X)

        error : float
            Translation invariant normalized RMS error between ``src_image`` and
            ``target_image``.

        phasediff : float
            Global phase difference between the two images (should be
            zero if images are non-negative).

    Raises:
     NotImplementedError "Error: register_translation only supports "
                                  "subpixel registration for 2D images"

     ValueError "Error: images must really be same size for "
                         "register_translation"

     ValueError "Error: register_translation only knows the \"real\" "
                         "and \"fourier\" values for the ``space`` argument."

    References:
    .. [1] Manuel Guizar-Sicairos, Samuel T. Thurman, and James R. Fienup,
           "Efficient subpixel image registration algorithms,"
           Optics Letters 33, 156-158 (2008).
    """
    # images must be the same shape
    if src_image.shape != target_image.shape:
        raise ValueError("Error: images must really be same size for register_translation")

    # only 2D data makes sense right now
    if src_image.ndim != 2 and upsample_factor > 1:
        raise NotImplementedError("Error: register_translation only supports subpixel registration for 2D images")

    # assume complex data is already in Fourier space
    if space.lower() == 'fourier':
        src_freq = src_image
        target_freq = target_image
    # real data needs to be fft'd.
    elif space.lower() == 'real':
        src_freq_1 = cv2.dft(
            src_image, flags=cv2.DFT_COMPLEX_OUTPUT + cv2.DFT_SCALE)
        src_freq = src_freq_1[:, :, 0] + 1j * src_freq_1[:, :, 1]
        src_freq = np.array(src_freq, dtype=np.complex128, copy=False)
        target_freq_1 = cv2.dft(
            target_image, flags=cv2.DFT_COMPLEX_OUTPUT + cv2.DFT_SCALE)
        target_freq = target_freq_1[:, :, 0] + 1j * target_freq_1[:, :, 1]
        target_freq = np.array(
            target_freq, dtype=np.complex128, copy=False)
    else:
        raise ValueError('Error: register_translation only knows the "real" and "fourier" values for the ``space`` argument.')

    # Whole-pixel shift - Compute cross-correlation by an IFFT
    shape = src_freq.shape
    image_product = src_freq * target_freq.conj()
    image_product_cv = np.dstack([np.real(image_product), np.imag(image_product)])
    cross_correlation = cv2.dft(image_product_cv, flags=cv2.DFT_INVERSE + cv2.DFT_SCALE)
    cross_correlation = cross_correlation[:, :, 0] + 1j * cross_correlation[:, :, 1]

    # Locate maximum
    new_cross_corr = np.abs(cross_correlation)

    if (shifts_lb is not None) or (shifts_ub is not None):
        if (shifts_lb[0] < 0) and (shifts_ub[0] >= 0):
            new_cross_corr[shifts_ub[0]:shifts_lb[0], :] = 0
        else:
            new_cross_corr[:shifts_lb[0], :] = 0
            new_cross_corr[shifts_ub[0]:, :] = 0

        if (shifts_lb[1] < 0) and (shifts_ub[1] >= 0):
            new_cross_corr[:, shifts_ub[1]:shifts_lb[1]] = 0
        else:
            new_cross_corr[:, :shifts_lb[1]] = 0
            new_cross_corr[:, shifts_ub[1]:] = 0
    else:
        new_cross_corr[max_shifts[0]:-max_shifts[0], :] = 0
        new_cross_corr[:, max_shifts[1]:-max_shifts[1]] = 0

    maxima = np.unravel_index(np.argmax(new_cross_corr),
                              cross_correlation.shape)
    midpoints = np.array([np.fix(axis_size//2)
                          for axis_size in shape])

    shifts = np.array(maxima, dtype=np.float64)
    shifts[shifts > midpoints] -= np.array(shape)[shifts > midpoints]

    if upsample_factor == 1:
        src_amp = np.sum(np.abs(src_freq) ** 2) / src_freq.size
        target_amp = np.sum(np.abs(target_freq) ** 2) / target_freq.size
        CCmax = cross_correlation.max()
    # If upsampling > 1, then refine estimate with matrix multiply DFT
    else:
        # Initial shift estimate in upsampled grid
        shifts = np.round(shifts * upsample_factor) / upsample_factor
        upsampled_region_size = np.ceil(upsample_factor * 1.5)
        # Center of output array at dftshift + 1
        dftshift = np.fix(upsampled_region_size/2.)
        upsample_factor = np.array(upsample_factor, dtype=np.float64)
        normalization = (src_freq.size * upsample_factor ** 2)
        # Matrix multiply DFT around the current shift estimate
        sample_region_offset = dftshift - shifts * upsample_factor

        cross_correlation = _upsampled_dft(image_product.conj(),
                                           upsampled_region_size,
                                           upsample_factor,
                                           sample_region_offset).conj()
        cross_correlation /= normalization
        # Locate maximum and map back to original pixel grid
        maxima = np.array(np.unravel_index(
            np.argmax(np.abs(cross_correlation)),
            cross_correlation.shape),
            dtype=np.float64)
        maxima -= dftshift
        shifts = shifts + (maxima / upsample_factor)
        CCmax = cross_correlation.max()
        src_amp = _upsampled_dft(src_freq * src_freq.conj(),
                                 1, upsample_factor)[0, 0]
        src_amp /= normalization
        target_amp = _upsampled_dft(target_freq * target_freq.conj(),
                                    1, upsample_factor)[0, 0]
        target_amp /= normalization

    # If its only one row or column the shift along that dimension has no
    # effect. We set to zero.
    for dim in range(src_freq.ndim):
        if shape[dim] == 1:
            shifts[dim] = 0

    return shifts, src_freq, _compute_phasediff(CCmax)

def apply_shifts_dft(src_freq, shifts, diffphase, is_freq=True, border_nan=True):
    """
    adapted from SIMA (https://github.com/losonczylab) and the
    scikit-image (http://scikit-image.org/) package.


    Unless otherwise specified by LICENSE.txt files in individual
    directories, all code is

    Copyright (C) 2011, the scikit-image team
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:

     1. Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.
     2. Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in
        the documentation and/or other materials provided with the
        distribution.
     3. Neither the name of skimage nor the names of its contributors may be
        used to endorse or promote products derived from this software without
        specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
    IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT,
    INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
    HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
    STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
    IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
    POSSIBILITY OF SUCH DAMAGE.
    Args:
        apply shifts using inverse dft
        src_freq: ndarray
            if is_freq it is fourier transform image else original image
        shifts: shifts to apply
        diffphase: comes from the register_translation output
    """

    is3D = len(src_freq.shape) == 3
    if not is_freq:
        if is3D:
            src_freq = np.fft.fftn(src_freq)
        else:
            src_freq = np.dstack([np.real(src_freq), np.imag(src_freq)])
            src_freq = cv2.dft(src_freq, flags=cv2.DFT_COMPLEX_OUTPUT + cv2.DFT_SCALE)
            src_freq = src_freq[:, :, 0] + 1j * src_freq[:, :, 1]
            src_freq = np.array(src_freq, dtype=np.complex128, copy=False)

    if not is3D:
        nr, nc = np.shape(src_freq)
        Nr = ifftshift(np.arange(-np.fix(nr/2.), np.ceil(nr/2.)))
        Nc = ifftshift(np.arange(-np.fix(nc/2.), np.ceil(nc/2.)))
        Nc, Nr = np.meshgrid(Nc, Nr)
        Greg = src_freq * np.exp(1j * 2 * np.pi *
                                 (-shifts[0] * Nr / nr - shifts[1] * Nc / nc))
    else:
        nr, nc, nd = np.array(np.shape(src_freq), dtype=float)
        Nr = ifftshift(np.arange(-np.fix(nr / 2.), np.ceil(nr / 2.)))
        Nc = ifftshift(np.arange(-np.fix(nc / 2.), np.ceil(nc / 2.)))
        Nd = ifftshift(np.arange(-np.fix(nd / 2.), np.ceil(nd / 2.)))
        Nc, Nr, Nd = np.meshgrid(Nc, Nr, Nd)
        Greg = src_freq * np.exp(1j * 2 * np.pi *
                                 (-shifts[0] * Nr / nr - shifts[1] * Nc / nc -
                                  shifts[2] * Nd / nd))

    Greg = Greg.dot(np.exp(1j * diffphase))
    if is3D:
        new_img = np.real(np.fft.ifftn(Greg))
    else:
        Greg = np.dstack([np.real(Greg), np.imag(Greg)])
        new_img = cv2.idft(Greg)[:, :, 0]

    if border_nan is not False:
        max_w, max_h, min_w, min_h = 0, 0, 0, 0
        max_h, max_w = np.ceil(np.maximum(
            (max_h, max_w), shifts[:2])).astype(int)
        min_h, min_w = np.floor(np.minimum(
            (min_h, min_w), shifts[:2])).astype(int)
        if is3D:
            max_d = np.ceil(np.maximum(0, shifts[2])).astype(int)
            min_d = np.floor(np.minimum(0, shifts[2])).astype(int)
        if border_nan is True:
            new_img[:max_h, :] = np.nan
            if min_h < 0:
                new_img[min_h:, :] = np.nan
            new_img[:, :max_w] = np.nan
            if min_w < 0:
                new_img[:, min_w:] = np.nan
            if is3D:
                new_img[:, :, :max_d] = np.nan
                if min_d < 0:
                    new_img[:, :, min_d:] = np.nan
        elif border_nan == 'min':
            min_ = np.nanmin(new_img)
            new_img[:max_h, :] = min_
            if min_h < 0:
                new_img[min_h:, :] = min_
            new_img[:, :max_w] = min_
            if min_w < 0:
                new_img[:, min_w:] = min_
            if is3D:
                new_img[:, :, :max_d] = min_
                if min_d < 0:
                    new_img[:, :, min_d:] = min_
        elif border_nan == 'copy':
            new_img[:max_h] = new_img[max_h]
            if min_h < 0:
                new_img[min_h:] = new_img[min_h-1]
            if max_w > 0:
                new_img[:, :max_w] = new_img[:, max_w, np.newaxis]
            if min_w < 0:
                new_img[:, min_w:] = new_img[:, min_w-1, np.newaxis]
            if is3D:
                if max_d > 0:
                    new_img[:, :, :max_d] = new_img[:, :, max_d, np.newaxis]
                if min_d < 0:
                    new_img[:, :, min_d:] = new_img[:, :, min_d-1, np.newaxis]
        else:
            logging.warning(f'Unknown value of border_nan ({border_nan}); treating as False')

    return new_img

def get_patch_edges(dims: tuple[int, ...], overlaps: tuple[int, ...], strides: tuple[int, ...],
                    ) -> tuple[list[int], ...]:
    """For each dimension, return a vector of pixels along that dimension where patches start"""
    windowSizes = np.add(overlaps, strides)
    return tuple(
        list(range(0, dim - windowSize, stride)) + [dim - windowSize]
        for dim, windowSize, stride in zip(dims, windowSizes, strides)
    )


def get_patch_centers(dims: tuple[int, ...], overlaps: tuple[int, ...], strides: tuple[int, ...],
                      shifts_opencv=False, upsample_factor_grid=1) -> tuple[list[float], ...]:
    """
    For each dimension, return a vector of patch center locations for pw_rigid correction
    shifts_opencv just overrides upsample_factor_grid (forces it to 1), this is an easy way to
    get the correct values by just providing the motion correction parameters, but
    by default no extra upsampling is done.
    """
    if not shifts_opencv and upsample_factor_grid != 1:
        # account for upsampling step
        strides = tuple(np.round(np.divide(strides, upsample_factor_grid)).astype(int))
    
    patch_edges = get_patch_edges(dims, overlaps, strides)
    windowSizes = np.add(overlaps, strides)
    return tuple([edge + (sz - 1) / 2 for edge in edges] for edges, sz in zip(patch_edges, windowSizes))


def sliding_window_dims(dims: tuple[int, ...], overlaps: tuple[int, ...], strides: tuple[int, ...]):
    """ computes dimensions for a sliding window with given image dims, overlaps, and strides

    Args: 
        dims: tuple
            the dimensions of the image

        overlaps: tuple
            overlap of patches in each dimension, except that the last patch will be all the way
            at the bottom/right regardless of overlap

        strides: tuple
            stride in each dimension, except that the last patch will be all the way
            at the bottom/right regardless of stride

     Returns:
         iterator containing 3 items:
              inds: (x, y, ...) coordinates in the patch grid
              corner: (x, y, ...) location of corner of the patch in the original matrix
              size: (x, y, ...) size of patch in pixels
    """
    windowSizes = np.add(overlaps, strides)
    edge_ranges = get_patch_edges(dims, overlaps, strides)

    for patch in itertools.product(*[enumerate(r) for r in edge_ranges]):
        inds, corners = zip(*patch)
        yield (tuple(inds), tuple(corners), windowSizes)


def sliding_window(image: np.ndarray, overlaps: tuple[int, int], strides: tuple[int, int]):
    """ efficiently and lazily slides a window across the image

    Args: 
        img: ndarray 2D
            image that needs to be sliced

        overlaps: tuple
            overlap of patches in each dimension, except that the last patch will be all the way
            at the bottom/right regardless of overlap

        strides: tuple
            stride in each dimension, except that the last patch will be all the way
            at the bottom/right regardless of stride

     Returns:
         iterator containing five items
              dim_1, dim_2 coordinates in the patch grid
              x, y: bottom border of the patch in the original matrix
              patch: the patch
    """
    if image.ndim != 2:
        raise ValueError('Input to sliding_window must be 2D')

    for inds, corner, size in sliding_window_dims(image.shape, overlaps, strides):
        patch = image[corner[0]:corner[0] + size[0], corner[1]:corner[1] + size[1]]
        yield inds + corner + (patch,)


def sliding_window_3d(image: np.ndarray, overlaps: tuple[int, int, int], strides: tuple[int, int, int]):
    """ efficiently and lazily slides a window across the image

    Args: 
        img: ndarray 3D
            image that needs to be sliced

        overlaps: tuple
            overlap of patches in each dimension, except that the last patch will be all the way
            at the bottom/right regardless of overlap 

        strides: tuple
            stride in each dimension, except that the last patch will be all the way
            at the bottom/right regardless of stride

     Returns:
         iterator containing seven items
              dim_1, dim_2, dim_3 coordinates in the patch grid
              x, y, z: bottom border of the patch in the original matrix
              patch: the patch
     """
    if image.ndim != 3:
        raise ValueError('Input to sliding_window_3d must be 3D')

    for inds, corner, size in sliding_window_dims(image.shape, overlaps, strides):
        patch = image[corner[0]:corner[0] + size[0],
                      corner[1]:corner[1] + size[1],
                      corner[2]:corner[2] + size[2]]
        yield inds + corner + (patch,)


def interpolate_shifts(shifts, coords_orig: tuple, coords_new: tuple) -> np.ndarray:
    """
    Interpolate piecewise shifts onto new coordinates. Pixels outside the original coordinates will be filled with edge values.
    
    Args:
        shifts: ndarray or other array-like
            shifts to interpolate; must have the same number of elements as the outer product of coords_orig

        coords_orig: tuple of float vectors
            patch center coordinates along each dimension (e.g. outputs of get_patch_centers)
        
        coords_new: tuple of float vectors
            coordinates along each dimension at which to output interpolated shifts
    
    Returns:
        ndarray of interpolated shifts, of shape tuple(len(coords) for coords in coords_new)
    """
    # clip new coordinates to avoid extrapolation
    coords_new_clipped = [np.clip(coord, min(coord_orig), max(coord_orig)) for coord, coord_orig in zip(coords_new, coords_orig)]
    coords_new_stacked = np.stack(np.meshgrid(*coords_new_clipped, indexing='ij'), axis=-1)
    shifts_grid = np.reshape(shifts, tuple(len(coord) for coord in coords_orig))
    return scipy.interpolate.interpn(coords_orig, shifts_grid, coords_new_stacked, method="cubic")


def iqr(a):
    return np.percentile(a, 75) - np.percentile(a, 25)

def create_weight_matrix_for_blending(img, overlaps, strides):
    """ create a matrix that is used to normalize the intersection of the stiched patches

    Args:
        img: original image, ndarray

        shapes, overlaps, strides:  tuples
            shapes, overlaps and strides of the patches

    Returns:
        weight_mat: normalizing weight matrix
    """
    shapes = np.add(strides, overlaps)

    max_grid_1, max_grid_2 = np.max(
        np.array([it[:2] for it in sliding_window(img, overlaps, strides)]), 0)

    for grid_1, grid_2, _, _, _ in sliding_window(img, overlaps, strides):

        weight_mat = np.ones(shapes)

        if grid_1 > 0:
            weight_mat[:overlaps[0], :] = np.linspace(
                0, 1, overlaps[0])[:, None]
        if grid_1 < max_grid_1:
            weight_mat[-overlaps[0]:,
                       :] = np.linspace(1, 0, overlaps[0])[:, None]
        if grid_2 > 0:
            weight_mat[:, :overlaps[1]] = weight_mat[:, :overlaps[1]
                                                     ] * np.linspace(0, 1, overlaps[1])[None, :]
        if grid_2 < max_grid_2:
            weight_mat[:, -overlaps[1]:] = weight_mat[:, -
                                                      overlaps[1]:] * np.linspace(1, 0, overlaps[1])[None, :]

        yield weight_mat

def high_pass_filter_space(img_orig, gSig_filt=None, freq=None, order=None):
    """
    Function for high passing the image(s) with centered Gaussian if gSig_filt
    is specified or Butterworth filter if freq and order are specified

    Args:
        img_orig: 2-d or 3-d array
            input image/movie

        gSig_filt:
            size of the Gaussian filter 

        freq: float
            cutoff frequency of the Butterworth filter

        order: int
            order of the Butterworth filter

    Returns:
        img: 2-d array or 3-d movie
            image/movie after filtering            
    """
    if freq is None or order is None:  # Gaussian
        ksize = tuple([(3 * i) // 2 * 2 + 1 for i in gSig_filt])
        ker = cv2.getGaussianKernel(ksize[0], gSig_filt[0])
        ker2D = ker.dot(ker.T)
        nz = np.nonzero(ker2D >= ker2D[:, 0].max())
        zz = np.nonzero(ker2D < ker2D[:, 0].max())
        ker2D[nz] -= ker2D[nz].mean()
        ker2D[zz] = 0
        if img_orig.ndim == 2:  # image
            return cv2.filter2D(np.array(img_orig, dtype=np.float32),
                                -1, ker2D, borderType=cv2.BORDER_REFLECT)
        else:  # movie
            return caiman.movie(np.array([cv2.filter2D(np.array(img, dtype=np.float32),
                                -1, ker2D, borderType=cv2.BORDER_REFLECT) for img in img_orig]))     
    else:  # Butterworth
        rows, cols = img_orig.shape[-2:]
        xx, yy = np.meshgrid(np.arange(cols, dtype=np.float32) - cols / 2,
                             np.arange(rows, dtype=np.float32) - rows / 2, sparse=True)
        H = np.fft.ifftshift(1 - 1 / (1 + ((xx**2 + yy**2)/freq**2)**order))
        if img_orig.ndim == 2:  # image
            return cv2.idft(cv2.dft(img_orig, flags=cv2.DFT_COMPLEX_OUTPUT) *
                            H[..., None])[..., 0] / (rows*cols)
        else:  # movie
            return caiman.movie(np.array([cv2.idft(cv2.dft(img, flags=cv2.DFT_COMPLEX_OUTPUT) * 
                            H[..., None])[..., 0] for img in img_orig]) / (rows*cols))

def tile_and_correct(img, template, strides, overlaps, max_shifts, newoverlaps=None, newstrides=None, upsample_factor_grid=4,
                     upsample_factor_fft=10, show_movie=False, max_deviation_rigid=2, add_to_movie=0, shifts_opencv=False, gSig_filt=None,
                     use_cuda=False, border_nan=True, shifts_interpolate=False):
    """ perform piecewise rigid motion correction iteration, by
        1) dividing the FOV in patches
        2) motion correcting each patch separately
        3) upsampling the motion correction vector field
        4) stiching back together the corrected subpatches

    Args:
        img: ndaarray 2D
            image to correct

        template: ndarray
            reference image

        strides: tuple
            strides of the patches in which the FOV is subdivided

        overlaps: tuple
            amount of pixel overlapping between patches along each dimension

        max_shifts: tuple
            max shifts in x and y

        newoverlaps:tuple
            amount of pixel overlapping between patches along each dimension when upsampling the vector fields

        newstrides:tuple
            strides between patches along each dimension when upsampling the vector fields

        upsample_factor_grid: int
            if newshapes or newstrides are not specified this is inferred upsampling by a constant factor the cvector field

        upsample_factor_fft: int
            resolution of fractional shifts

        show_movie: bool
            whether to visualize the original and corrected frame during motion correction

        max_deviation_rigid: int
            maximum deviation in shifts of each patch from the rigid shift (should not be large)

        add_to_movie: int
            if movie is too negative the correction might have some issues. In this case it is good to add values so that it is non negative most of the times

        shifts_opencv: bool
            FIXME: (UNDOCUMENTED)

        gSig_filt: tuple
            standard deviation and size of gaussian filter to center filter data in case of one photon imaging data

        use_cuda: bool (DEPRECATED)
               cuda is no longer supported for motion correction; this kwarg will be removed in a future version of Caiman

        border_nan : bool or string, optional
            specifies how to deal with borders. (True, False, 'copy', 'min')
        
        shifts_interpolate: bool
            use patch locations to interpolate shifts rather than just upscaling to size of image. Default: False

    Returns:
        (new_img, total_shifts, start_step, xy_grid)
            new_img: ndarray, corrected image


    """
    logger = logging.getLogger("caiman")
    logger.debug("Starting tile and correct")
    img = img.astype(np.float64).copy()
    template = template.astype(np.float64).copy()

    if gSig_filt is not None:
        img_orig = img.copy()
        img = high_pass_filter_space(img_orig, gSig_filt)

    img = img + add_to_movie
    template = template + add_to_movie

    # compute rigid shifts
    rigid_shts, sfr_freq, diffphase = register_translation(
        img, template, upsample_factor=upsample_factor_fft, max_shifts=max_shifts, use_cuda=use_cuda)

    if max_deviation_rigid == 0:
        if shifts_opencv:
            if gSig_filt is not None:
                img = img_orig

            new_img = apply_shift_iteration(
                img, (-rigid_shts[0], -rigid_shts[1]), border_nan=border_nan)

        else:
            if gSig_filt is not None:
                raise Exception(
                    'The use of FFT and filtering options have not been tested. Set shifts_opencv=True or do not provide gSig_filt')

            new_img = apply_shifts_dft(
                sfr_freq, (-rigid_shts[0], -rigid_shts[1]), diffphase, border_nan=border_nan)

        return new_img - add_to_movie, (-rigid_shts[0], -rigid_shts[1]), None, None
    else:
        # extract patches
        logger.debug("Extracting patches")
        xy_grid: list[tuple[int, int]] = []
        templates: list[np.ndarray] = []

        for (xind, yind, _, _, patch) in sliding_window(template, overlaps=overlaps, strides=strides):
            xy_grid.append((xind, yind))
            templates.append(patch)

        imgs = [it[-1] for it in sliding_window(img, overlaps=overlaps, strides=strides)]
        dim_grid = tuple(np.add(xy_grid[-1], 1))
        num_tiles = len(xy_grid)

        if max_deviation_rigid is not None:
            lb_shifts = np.ceil(np.subtract(
                rigid_shts, max_deviation_rigid)).astype(int)
            ub_shifts = np.floor(
                np.add(rigid_shts, max_deviation_rigid)).astype(int)
        else:
            lb_shifts = None
            ub_shifts = None

        # extract shifts for each patch
        logger.debug("Extracting shifts for each patch")
        shfts_et_all = [register_translation(
            a, b, c, shifts_lb=lb_shifts, shifts_ub=ub_shifts, max_shifts=max_shifts, use_cuda=use_cuda) for a, b, c in zip(
            imgs, templates, [upsample_factor_fft] * num_tiles)]
        shfts = [sshh[0] for sshh in shfts_et_all]
        diffs_phase = [sshh[2] for sshh in shfts_et_all]
        # create a vector field
        shift_img_x = np.reshape(np.array(shfts)[:, 0], dim_grid)
        shift_img_y = np.reshape(np.array(shfts)[:, 1], dim_grid)
        diffs_phase_grid = np.reshape(np.array(diffs_phase), dim_grid)

        if shifts_opencv:
            if gSig_filt is not None:
                img = img_orig

            dims = img.shape
            patch_centers = get_patch_centers(dims, strides=strides, overlaps=overlaps)
            # shift_img_x and shift_img_y are switched intentionally
            m_reg = apply_pw_shifts_remap_2d(img, shifts_y=shift_img_x, shifts_x=shift_img_y,
                                             patch_centers=patch_centers, border_nan=border_nan,
                                             shifts_interpolate=shifts_interpolate)
            total_shifts = [
                    (-x, -y) for x, y in zip(shift_img_x.reshape(num_tiles), shift_img_y.reshape(num_tiles))]
            return m_reg - add_to_movie, total_shifts, None, None

        # create automatically upsample parameters if not passed
        if newoverlaps is None:
            newoverlaps = overlaps
        if newstrides is None:
            newstrides = tuple(
                np.round(np.divide(strides, upsample_factor_grid)).astype(int))

        newshapes = np.add(newstrides, newoverlaps)

        xy_grid: list[tuple[int, int]] = []
        start_step: list[tuple[int, int]] = []
        imgs: list[np.ndarray] = []

        for (xind, yind, xstart, ystart, patch) in sliding_window(img, overlaps=newoverlaps, strides=newstrides):
            xy_grid.append((xind, yind))
            start_step.append((xstart, ystart))
            imgs.append(patch)

        dim_new_grid = tuple(np.add(xy_grid[-1], 1))
        num_tiles = len(xy_grid)

        if shifts_interpolate:
            patch_centers_orig = get_patch_centers(img.shape, strides=strides, overlaps=overlaps)
            patch_centers_new = get_patch_centers(img.shape, strides=newstrides, overlaps=newoverlaps)
            shift_img_x = interpolate_shifts(shift_img_x, patch_centers_orig, patch_centers_new)
            shift_img_y = interpolate_shifts(shift_img_y, patch_centers_orig, patch_centers_new)
            diffs_phase_grid_us = interpolate_shifts(diffs_phase_grid, patch_centers_orig, patch_centers_new)
        else:
            shift_img_x = cv2.resize(
                shift_img_x, dim_new_grid[::-1], interpolation=cv2.INTER_CUBIC)
            shift_img_y = cv2.resize(
                shift_img_y, dim_new_grid[::-1], interpolation=cv2.INTER_CUBIC)
            diffs_phase_grid_us = cv2.resize(
                diffs_phase_grid, dim_new_grid[::-1], interpolation=cv2.INTER_CUBIC)


        max_shear = np.percentile(
            [np.max(np.abs(np.diff(ssshh, axis=xxsss))) for ssshh, xxsss in itertools.product(
                [shift_img_x, shift_img_y], [0, 1])], 75)

        total_shifts = [
            (-x, -y) for x, y in zip(shift_img_x.reshape(num_tiles), shift_img_y.reshape(num_tiles))]
        total_diffs_phase = [
            dfs for dfs in diffs_phase_grid_us.reshape(num_tiles)]

        if gSig_filt is not None:
            raise Exception(
                'The use of FFT and filtering options have not been tested. Set shifts_opencv=True')

        imgs = [apply_shifts_dft(im, (
            sh[0], sh[1]), dffphs, is_freq=False, border_nan=border_nan) for im, sh, dffphs in zip(
            imgs, total_shifts, total_diffs_phase)]

        normalizer = np.zeros_like(img) * np.nan
        new_img = np.zeros_like(img) * np.nan

        weight_matrix = create_weight_matrix_for_blending(
            img, newoverlaps, newstrides)

        if max_shear < 0.5:
            for (x, y), (_, _), im, (_, _), weight_mat in zip(start_step, xy_grid, imgs, total_shifts, weight_matrix):

                prev_val_1 = normalizer[x:x + newshapes[0], y:y + newshapes[1]]

                normalizer[x:x + newshapes[0], y:y + newshapes[1]] = np.nansum(
                    np.dstack([~np.isnan(im) * 1 * weight_mat, prev_val_1]), -1)
                prev_val = new_img[x:x + newshapes[0], y:y + newshapes[1]]
                new_img[x:x + newshapes[0], y:y + newshapes[1]
                        ] = np.nansum(np.dstack([im * weight_mat, prev_val]), -1)

            new_img = new_img / normalizer

        else:  # if the difference in shift between neighboring patches is larger than 0.5 pixels we do not interpolate in the overlapping area
            half_overlap_x = int(newoverlaps[0] / 2)
            half_overlap_y = int(newoverlaps[1] / 2)
            for (x, y), (idx_0, idx_1), im, (_, _), weight_mat in zip(start_step, xy_grid, imgs, total_shifts, weight_matrix):

                if idx_0 == 0:
                    x_start = x
                else:
                    x_start = x + half_overlap_x

                if idx_1 == 0:
                    y_start = y
                else:
                    y_start = y + half_overlap_y

                x_end = x + newshapes[0]
                y_end = y + newshapes[1]
                new_img[x_start:x_end,
                        y_start:y_end] = im[x_start - x:, y_start - y:]

        if show_movie:
            img = apply_shifts_dft(
                sfr_freq, (-rigid_shts[0], -rigid_shts[1]), diffphase, border_nan=border_nan)
            img_show = np.vstack([new_img, img])

            img_show = cv2.resize(img_show, None, fx=1, fy=1)

            cv2.imshow('frame', img_show / np.percentile(template, 99))
            cv2.waitKey(int(1. / 500 * 1000))

        else:
            try:
                cv2.destroyAllWindows()
            except:
                pass
        return new_img - add_to_movie, total_shifts, start_step, xy_grid

def tile_and_correct_3d(img:np.ndarray, template:np.ndarray, strides:tuple, overlaps:tuple, max_shifts:tuple, newoverlaps:Optional[tuple]=None, newstrides:Optional[tuple]=None, upsample_factor_grid:int=4,
                     upsample_factor_fft:int=10, show_movie:bool=False, max_deviation_rigid:int=2, add_to_movie:int=0, shifts_opencv:bool=True, gSig_filt=None,
                     use_cuda:bool=False, border_nan:bool=True, shifts_interpolate:bool=False):
    """ perform piecewise rigid motion correction iteration, by
        1) dividing the FOV in patches
        2) motion correcting each patch separately
        3) upsampling the motion correction vector field
        4) stiching back together the corrected subpatches

    Args:
        img: ndarray 3D
            image to correct

        template: ndarray
            reference image

        strides: tuple
            strides of the patches in which the FOV is subdivided

        overlaps: tuple
            amount of pixel overlapping between patches along each dimension

        max_shifts: tuple
            max shifts in x, y, and z

        newstrides:tuple
            strides between patches along each dimension when upsampling the vector fields

        newoverlaps:tuple
            amount of pixel overlapping between patches along each dimension when upsampling the vector fields

        upsample_factor_grid: int
            if newshapes or newstrides are not specified this is inferred upsampling by a constant factor the cvector field

        upsample_factor_fft: int
            resolution of fractional shifts

        show_movie: bool
            whether to visualize the original and corrected frame during motion correction

        max_deviation_rigid: int
            maximum deviation in shifts of each patch from the rigid shift (should not be large)

        add_to_movie: if movie is too negative the correction might have some issues. In this case it is good to add values so that it is non negative most of the times

        shifts_opencv: bool
            FIXME: (UNDOCUMENTED)

        gSig_filt: tuple
            standard deviation and size of gaussian filter to center filter data in case of one photon imaging data

        use_cuda : bool (DEPRECATED)
            cuda is no longer supported for motion correction; this kwarg will be removed in a future version of Caiman

        border_nan : bool or string, optional
            specifies how to deal with borders. (True, False, 'copy', 'min')
        
        shifts_interpolate: bool
            use patch locations to interpolate shifts rather than just upscaling to size of image. Default: False

    Returns:
        (new_img, total_shifts, start_step, xyz_grid)
            new_img: ndarray, corrected image


    """

    # TODO The clarity of this function is poor; it'd be good to refactor it towards being easier to read/understand
    img = img.astype(np.float64).copy()
    template = template.astype(np.float64).copy()

    if gSig_filt is not None:
        img_orig = img.copy()
        img = high_pass_filter_space(img_orig, gSig_filt)

    img = img + add_to_movie
    template = template + add_to_movie

    # compute rigid shifts
    rigid_shts, sfr_freq, diffphase = register_translation_3d(
        img, template, upsample_factor=upsample_factor_fft, max_shifts=max_shifts)

    if max_deviation_rigid == 0: # if rigid shifts only
        if gSig_filt is not None:
            raise Exception(
                'The use of FFT and filtering options has not been tested. Do not use gSig_filt with rigid shifts only')

        new_img = apply_shifts_dft( # TODO: check
            sfr_freq, (-rigid_shts[0], -rigid_shts[1], -rigid_shts[2]), diffphase, border_nan=border_nan)

        return new_img - add_to_movie, (-rigid_shts[0], -rigid_shts[1], -rigid_shts[2]), None, None
    else:
        # extract patches
        xyz_grid: list[tuple[int, int, int]] = []
        templates: list[np.ndarray] = []

        for (xind, yind, zind, _, _, _, patch) in sliding_window_3d(template, overlaps=overlaps, strides=strides):
            xyz_grid.append((xind, yind, zind))
            templates.append(patch)

        imgs = [it[-1] for it in sliding_window_3d(img, overlaps=overlaps, strides=strides)]
        dim_grid = tuple(np.add(xyz_grid[-1], 1))
        num_tiles = len(xyz_grid)

        if max_deviation_rigid is not None:
            lb_shifts = np.ceil(np.subtract(
                rigid_shts, max_deviation_rigid)).astype(int)
            ub_shifts = np.floor(
                np.add(rigid_shts, max_deviation_rigid)).astype(int)
        else:
            lb_shifts = None
            ub_shifts = None

        # extract shifts for each patch
        shfts_et_all = [register_translation_3d(
            a, b, c, shifts_lb=lb_shifts, shifts_ub=ub_shifts, max_shifts=max_shifts) for a, b, c in zip(
            imgs, templates, [upsample_factor_fft] * num_tiles)]
        shfts = [sshh[0] for sshh in shfts_et_all]
        diffs_phase = [sshh[2] for sshh in shfts_et_all]
        # create a vector field
        shift_img_x = np.reshape(np.array(shfts)[:, 0], dim_grid)
        shift_img_y = np.reshape(np.array(shfts)[:, 1], dim_grid)
        shift_img_z = np.reshape(np.array(shfts)[:, 2], dim_grid)
        diffs_phase_grid = np.reshape(np.array(diffs_phase), dim_grid)

        #  shifts_opencv doesn't make sense here- replace with shifts_skimage
        if shifts_opencv:
            if gSig_filt is not None:
                img = img_orig
            
            patch_centers = get_patch_centers(img.shape, strides=strides, overlaps=overlaps)
            # shift_img_x and shift_img_y are switched intentionally
            m_reg = apply_pw_shifts_remap_3d(
                img, shifts_y=shift_img_x, shifts_x=shift_img_y, shifts_z=shift_img_z,
                patch_centers=patch_centers, border_nan=border_nan,
                shifts_interpolate=shifts_interpolate)

            total_shifts = [
                    (-x, -y, -z) for x, y, z in zip(shift_img_x.reshape(num_tiles), shift_img_y.reshape(num_tiles), shift_img_z.reshape(num_tiles))]
            return m_reg - add_to_movie, total_shifts, None, None

        # create automatically upsampled parameters if not passed
        if newoverlaps is None:
            newoverlaps = overlaps
        if newstrides is None:
            newstrides = tuple(
                np.round(np.divide(strides, upsample_factor_grid)).astype(int))

        newshapes = np.add(newstrides, newoverlaps)

        xyz_grid: list[tuple[int, int, int]] = []
        start_step: list[tuple[int, int, int]] = []
        imgs: list[np.ndarray] = []

        for (xind, yind, zind, xstart, ystart, zstart, patch) in sliding_window_3d(
            img, overlaps=newoverlaps, strides=newstrides):
            xyz_grid.append((xind, yind, zind))
            start_step.append((xstart, ystart, zstart))
            imgs.append(patch)

        dim_new_grid = tuple(np.add(xyz_grid[-1], 1))
        num_tiles = len(xyz_grid)

        if shifts_interpolate:
            patch_centers_orig = get_patch_centers(img.shape, strides=strides, overlaps=overlaps)
            patch_centers_new = get_patch_centers(img.shape, strides=newstrides, overlaps=newoverlaps)
            shift_img_x = interpolate_shifts(shift_img_x, patch_centers_orig, patch_centers_new)
            shift_img_y = interpolate_shifts(shift_img_y, patch_centers_orig, patch_centers_new)
            shift_img_z = interpolate_shifts(shift_img_z, patch_centers_orig, patch_centers_new)
            diffs_phase_grid_us = interpolate_shifts(diffs_phase_grid, patch_centers_orig, patch_centers_new)
        else:
            shift_img_x = resize_sk(
                shift_img_x, dim_new_grid[::-1], order=3)
            shift_img_y = resize_sk(
                shift_img_y, dim_new_grid[::-1], order=3)
            shift_img_z = resize_sk(
                shift_img_z, dim_new_grid[::-1], order=3)
            diffs_phase_grid_us = resize_sk(
                diffs_phase_grid, dim_new_grid[::-1], order=3)

        # what dimension shear should be looked at? shearing for 3d point scanning happens in y and z but not for plane-scanning
        max_shear = np.percentile(
            [np.max(np.abs(np.diff(ssshh, axis=xxsss))) for ssshh, xxsss in itertools.product(
                [shift_img_x, shift_img_y], [0, 1])], 75)

        total_shifts = [
            (-x, -y, -z) for x, y, z in zip(shift_img_x.reshape(num_tiles), shift_img_y.reshape(num_tiles), shift_img_z.reshape(num_tiles))]
        total_diffs_phase = [
            dfs for dfs in diffs_phase_grid_us.reshape(num_tiles)]

        if gSig_filt is not None:
            raise Exception(
                'The use of FFT and filtering options have not been tested. Do not use gSig_filt in this mode without shifts_opencv')

        imgs = [apply_shifts_dft(im, (
            sh[0], sh[1], sh[2]), dffphs, is_freq=False, border_nan=border_nan) for im, sh, dffphs in zip(
            imgs, total_shifts, total_diffs_phase)]

        normalizer = np.zeros_like(img) * np.nan
        new_img = np.zeros_like(img) * np.nan

        weight_matrix = create_weight_matrix_for_blending(
            img, newoverlaps, newstrides)

        if max_shear < 0.5:
            for (x, y, z), (_, _, _), im, (_, _, _), weight_mat in zip(start_step, xyz_grid, imgs, total_shifts, weight_matrix):

                prev_val_1 = normalizer[x:x + newshapes[0], y:y + newshapes[1], z:z + newshapes[2]]

                normalizer[x:x + newshapes[0], y:y + newshapes[1], z:z + newshapes[2]] = np.nansum(
                    np.dstack([~np.isnan(im) * 1 * weight_mat, prev_val_1]), -1)
                prev_val = new_img[x:x + newshapes[0], y:y + newshapes[1], z:z + newshapes[2]]
                new_img[x:x + newshapes[0], y:y + newshapes[1], z:z + newshapes[2]
                        ] = np.nansum(np.dstack([im * weight_mat, prev_val]), -1)

            new_img = new_img / normalizer

        else:  # if the difference in shift between neighboring patches is larger than 0.5 pixels we do not interpolate in the overlapping area
            half_overlap_x = int(newoverlaps[0] / 2)
            half_overlap_y = int(newoverlaps[1] / 2)
            half_overlap_z = int(newoverlaps[2] / 2)
            
            for (x, y, z), (idx_0, idx_1, idx_2), im, (_, _, _), weight_mat in zip(start_step, xyz_grid, imgs, total_shifts, weight_matrix):

                if idx_0 == 0:
                    x_start = x
                else:
                    x_start = x + half_overlap_x

                if idx_1 == 0:
                    y_start = y
                else:
                    y_start = y + half_overlap_y

                if idx_2 == 0:
                    z_start = z
                else:
                    z_start = z + half_overlap_z

                x_end = x + newshapes[0]
                y_end = y + newshapes[1]
                z_end = z + newshapes[2]
                new_img[x_start:x_end,y_start:y_end,
                        z_start:z_end] = im[x_start - x:, y_start - y:, z_start -z:]

        if show_movie:
            img = apply_shifts_dft(
                sfr_freq, (-rigid_shts[0], -rigid_shts[1], -rigid_shts[2]), diffphase, border_nan=border_nan)
            img_show = np.vstack([new_img, img])

            img_show = rescale_sk(img_show, 1)  # TODO does this actually do anything??

            cv2.imshow('frame', img_show / np.percentile(template, 99))
            cv2.waitKey(int(1. / 500 * 1000))

        else:
            try:
                cv2.destroyAllWindows()
            except:
                pass
        return new_img - add_to_movie, total_shifts, start_step, xyz_grid
    
def apply_pw_shifts_remap_2d(img: np.ndarray, shifts_y: np.ndarray, shifts_x: np.ndarray,
                             patch_centers: tuple[list[float], ...], border_nan: Union[bool, Literal['copy', 'min']],
                             shifts_interpolate=False) -> np.ndarray:
    """
    Use OpenCV remap to apply 2D piecewise shifts
    Inputs:
        img: the 2D image to apply shifts to
        shifts_y: array of y shifts for each patch (C order) (this is the actual Y i.e. the first dimension of the image)
        shifts_x: array of x shifts for each patch (C order)
        patch_centers: tuple of patch locations in each dimension.
        border_nan: how to deal with borders when remapping
        shifts_interpolate: if true, uses interpn to upsample shifts based on patch centers instead of resize
    Outputs:
        img_remapped: the remapped image
    """
    # reshape shifts_y and shifts_x based on patch grid
    patch_grid = tuple(len(centers) for centers in patch_centers)
    shift_img_y = np.reshape(shifts_y, patch_grid)
    shift_img_x = np.reshape(shifts_x, patch_grid)

    # get full image shape/coordinates
    dims = img.shape
    x_coords, y_coords = [np.arange(0., dims[dim]).astype(np.float32) for dim in (1, 0)]
    x_grid, y_grid = np.meshgrid(x_coords, y_coords)

    # up-sample shifts
    if shifts_interpolate:
        shifts_y = interpolate_shifts(shift_img_y, patch_centers, (y_coords, x_coords)).astype(np.float32)
        shifts_x = interpolate_shifts(shift_img_x, patch_centers, (y_coords, x_coords)).astype(np.float32)
    else:
        shifts_y = cv2.resize(shift_img_y.astype(np.float32), dims[::-1])
        shifts_x = cv2.resize(shift_img_x.astype(np.float32), dims[::-1])
    
    # apply to image
    if border_nan is False:
        mode = cv2.BORDER_CONSTANT
        value = 0.0
    elif border_nan is True:
        mode = cv2.BORDER_CONSTANT
        value = np.nan
    elif border_nan == 'min':
        mode = cv2.BORDER_CONSTANT
        value = np.min(img)
    elif border_nan == 'copy':
        mode = cv2.BORDER_REPLICATE
        value = 0.0
    else:
        raise ValueError(f'Unknown value of border_nan ({border_nan})')
    
    return cv2.remap(img, shifts_x + x_grid, shifts_y + y_grid, cv2.INTER_CUBIC,
                     borderMode=mode, borderValue=value)

def apply_pw_shifts_remap_3d(img: np.ndarray, shifts_y: np.ndarray, shifts_x: np.ndarray, shifts_z: np.ndarray,
                             patch_centers: tuple[list[float], ...], border_nan: Union[bool, Literal['copy', 'min']],
                             shifts_interpolate=False) -> np.ndarray:
    """
    Use skimage warp to apply 3D piecewise shifts
    Inputs:
        img: the 3D image to apply shifts to
        shifts_y: array of y shifts for each patch (C order) (this is the actual Y i.e. the first dimension of the image)
        shifts_x: array of x shifts for each patch (C order)
        shifts_z: array of z shifts for each patch (C order)
        patch_centers: tuple of patch locations in each dimension.
        border_nan: how to deal with borders when remapping
        shifts_interpolate: if true, uses interpn to upsample shifts based on patch centers instead of resize
    Outputs:
        img_remapped: the remapped image
    """
    # reshape shifts based on patch grid
    patch_grid = tuple(len(centers) for centers in patch_centers)
    shift_img_y = np.reshape(shifts_y, patch_grid)
    shift_img_x = np.reshape(shifts_x, patch_grid)
    shift_img_z = np.reshape(shifts_z, patch_grid)

    # get full image shape/coordinates
    dims = img.shape
    x_coords, y_coords, z_coords = [np.arange(0., dims[dim]).astype(np.float32) for dim in (1, 0, 2)]
    x_grid, y_grid, z_grid = np.meshgrid(x_coords, y_coords, z_coords)

    # up-sample shifts
    if shifts_interpolate:
        coords_new = (y_coords, x_coords, z_coords)
        shifts_y = interpolate_shifts(shift_img_y, patch_centers, coords_new).astype(np.float32)
        shifts_x = interpolate_shifts(shift_img_x, patch_centers, coords_new).astype(np.float32)
        shifts_z = interpolate_shifts(shift_img_z, patch_centers, coords_new).astype(np.float32)
    else:
        shifts_y = resize_sk(shift_img_y.astype(np.float32), dims)
        shifts_x = resize_sk(shift_img_x.astype(np.float32), dims)
        shifts_z = resize_sk(shift_img_z.astype(np.float32), dims)
    
    shift_map = np.stack((shifts_y + y_grid, shifts_x + x_grid, shifts_z + z_grid), axis=0)
    
    # apply to image
    if border_nan is False:
        mode = 'constant'
        value = 0.0
    elif border_nan is True:
        mode = 'constant'
        value = np.nan
    elif border_nan == 'min':
        mode = 'constant'
        value = np.min(img)
    elif border_nan == 'copy':
        mode = 'edge'
        value = 0.0
    else:
        raise ValueError(f'Unknown value of border_nan ({border_nan})')

    return warp_sk(img, shift_map, order=3, mode=mode, cval=value)


def compute_flow_single_frame(frame, templ, pyr_scale=.5, levels=3, winsize=100, iterations=15, poly_n=5,
                              poly_sigma=1.2 / 5, flags=0):
    flow = cv2.calcOpticalFlowFarneback(
        templ, frame, None, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags)
    return flow

def compute_metrics_motion_correction(fname, final_size_x, final_size_y, swap_dim, pyr_scale=.5, levels=3,
                                      winsize=100, iterations=15, poly_n=5, poly_sigma=1.2 / 5, flags=0,
                                      play_flow=False, resize_fact_flow=.2, template=None,
                                      opencv=True, resize_fact_play=3, fr_play=30, max_flow=1,
                                      gSig_filt=None):
    #todo: todocument
    logger = logging.getLogger("caiman")
    if os.environ.get('ENABLE_TQDM') == 'True': # TODO Find a better way to toggle this or remove the code; this is also not documented anywhere
        disable_tqdm = False
    else:
        disable_tqdm = True
        
    vmin, vmax = -max_flow, max_flow
    m = caiman.load(fname)
    if gSig_filt is not None:
        m = high_pass_filter_space(m, gSig_filt)
    mi, ma = m.min(), m.max()
    m_min = mi + (ma - mi) / 100
    m_max = mi + (ma - mi) / 4

    max_shft_x = int(np.ceil((np.shape(m)[1] - final_size_x) / 2))
    max_shft_y = int(np.ceil((np.shape(m)[2] - final_size_y) / 2))
    max_shft_x_1 = - ((np.shape(m)[1] - max_shft_x) - (final_size_x))
    max_shft_y_1 = - ((np.shape(m)[2] - max_shft_y) - (final_size_y))
    if max_shft_x_1 == 0:
        max_shft_x_1 = None

    if max_shft_y_1 == 0:
        max_shft_y_1 = None
    logger.info([max_shft_x, max_shft_x_1, max_shft_y, max_shft_y_1])
    m = m[:, max_shft_x:max_shft_x_1, max_shft_y:max_shft_y_1]
    if np.sum(np.isnan(m)) > 0:
        logger.info(m.shape)
        logger.warning('Movie contains NaN')
        raise Exception('Movie contains NaN')

    logger.debug('Local correlations..')
    img_corr = m.local_correlations(eight_neighbours=True, swap_dim=swap_dim)
    logger.debug(m.shape)
    if template is None:
        tmpl = caiman.motion_correction.bin_median(m)
    else:
        tmpl = template

    logger.debug('Compute Smoothness.. ')
    smoothness = np.sqrt(
        np.sum(np.sum(np.array(np.gradient(np.mean(m, 0)))**2, 0)))
    smoothness_corr = np.sqrt(
        np.sum(np.sum(np.array(np.gradient(img_corr))**2, 0)))

    logger.debug('Computing correlations.. ')
    correlations = []
    count = 0
    sys.stdout.flush()
    for fr in tqdm(m, desc="Correlations", disable=disable_tqdm):
        if disable_tqdm:
            if count % 100 == 0:
                logger.debug(count)
        count += 1
        correlations.append(scipy.stats.pearsonr(
            fr.flatten(), tmpl.flatten())[0])

    logger.info('Computing optical flow.. ')

    m = m.resize(1, 1, resize_fact_flow)
    norms = []
    flows = []
    count = 0
    sys.stdout.flush()
    for fr in tqdm(m, desc="Optical flow", disable=disable_tqdm):
        if disable_tqdm:
            if count % 100 == 0:
                logger.debug(count)


        count += 1
        flow = cv2.calcOpticalFlowFarneback(
            tmpl, fr, None, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags)

        if play_flow:
            if opencv:
                dims = tuple(np.array(flow.shape[:-1]) * resize_fact_play)
                vid_frame = np.concatenate([
                    np.repeat(np.clip((cv2.resize(fr, dims)[..., None] - m_min) /
                                      (m_max - m_min), 0, 1), 3, -1),
                    np.transpose([cv2.resize(np.clip(flow[:, :, 1] / vmax, 0, 1), dims),
                                  np.zeros(dims, np.float32),
                                  cv2.resize(np.clip(flow[:, :, 1] / vmin, 0, 1), dims)],
                                  (1, 2, 0)),
                    np.transpose([cv2.resize(np.clip(flow[:, :, 0] / vmax, 0, 1), dims),
                                  np.zeros(dims, np.float32),
                                  cv2.resize(np.clip(flow[:, :, 0] / vmin, 0, 1), dims)],
                                  (1, 2, 0))], 1).astype(np.float32)
                cv2.putText(vid_frame, 'movie', (10, 20), fontFace=5, fontScale=0.8, color=(
                    0, 255, 0), thickness=1)
                cv2.putText(vid_frame, 'y_flow', (dims[0] + 10, 20), fontFace=5, fontScale=0.8, color=(
                    0, 255, 0), thickness=1)
                cv2.putText(vid_frame, 'x_flow', (2 * dims[0] + 10, 20), fontFace=5, fontScale=0.8, color=(
                    0, 255, 0), thickness=1)
                cv2.imshow('frame', vid_frame)
                cv2.waitKey(1 / fr_play) # to pause between frames
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                plt.subplot(1, 3, 1)
                plt.cla()
                plt.imshow(fr, vmin=m_min, vmax=m_max, cmap='gray')
                plt.title('movie')
                plt.subplot(1, 3, 3)
                plt.cla()
                plt.imshow(flow[:, :, 1], vmin=vmin, vmax=vmax)
                plt.title('y_flow')
                plt.subplot(1, 3, 2)
                plt.cla()
                plt.imshow(flow[:, :, 0], vmin=vmin, vmax=vmax)
                plt.title('x_flow')
                plt.pause(1 / fr_play)

        n = np.linalg.norm(flow)
        flows.append(flow)
        norms.append(n)
    if play_flow and opencv:
        cv2.destroyAllWindows()

    # FIXME: This generates a metrics dump potentially right next to the datafiles it was generated from;
    #        We will need to fix this in some future revision of the code, ideally returning the filename we used to the caller
    #        or abstracting the path handling logic into some kind of a policy-aware getpath function for specific uses.
    #        This should be fixed with future work to have separate runs have separate workdirs.
    np.savez(os.path.splitext(fname)[0] + '_metrics', flows=flows, norms=norms, correlations=correlations, smoothness=smoothness,
             tmpl=tmpl, smoothness_corr=smoothness_corr, img_corr=img_corr)
    return tmpl, correlations, flows, norms, smoothness

def motion_correct_batch_rigid(fname, max_shifts, dview=None, splits=56, num_splits_to_process=None, num_iter=1,
                               template=None, shifts_opencv=False, save_movie_rigid=False, add_to_movie=None,
                               nonneg_movie=False, gSig_filt=None, subidx=slice(None, None, 1), use_cuda=False,
                               border_nan=True, var_name_hdf5='mov', is3D=False, indices=(slice(None), slice(None)),
                               shifts_interpolate=False):
    """
    Function that perform memory efficient hyper parallelized rigid motion corrections while also saving a memory mappable file

    Args:
        fname: str
            name of the movie to motion correct. It should not contain nans. All the loadable formats from CaImAn are acceptable

        max_shifts: tuple
            x and y (and z if 3D) maximum allowed shifts

        dview: ipyparallel view
            used to perform parallel computing

        splits: int
            number of batches in which the movies is subdivided

        num_splits_to_process: int
            number of batches to process. when not None, the movie is not saved since only a random subset of batches will be processed

        num_iter: int
            number of iterations to perform. The more iteration the better will be the template.

        template: ndarray
            if a good approximation of the template to register is available, it can be used

        shifts_opencv: bool
             toggle the shifts applied with opencv, if yes faster but induces some smoothing

        save_movie_rigid: bool
             toggle save movie

        subidx: slice
            Indices to slice

        use_cuda : bool (DEPRECATED)
            cuda is no longer supported for motion correction; this kwarg will be removed in a future version of Caiman

        indices: tuple(slice), default: (slice(None), slice(None))
           Use that to apply motion correction only on a part of the FOV

    Returns:
         fname_tot_rig: str

         total_template:ndarray

         templates:list
              list of produced templates, one per batch

         shifts: list
              inferred rigid shifts to correct the movie

    Raises:
        Exception 'The movie contains nans. Nans are not allowed!'

    """
    logger = logging.getLogger("caiman")

    dims, T = caiman.base.movies.get_file_size(fname, var_name_hdf5=var_name_hdf5)
    Ts = np.arange(T)[subidx].shape[0]
    step = Ts // 10 if is3D else Ts // 50
    corrected_slicer = slice(subidx.start, subidx.stop, step + 1)
    m = caiman.load(fname, var_name_hdf5=var_name_hdf5, subindices=corrected_slicer)

    if len(m.shape) < 3:
        m = caiman.load(fname, var_name_hdf5=var_name_hdf5)
        m = m[corrected_slicer]
        logger.warning("Your original file was saved as a single page " +
                       "file. Consider saving it in multiple smaller files" +
                       "with size smaller than 4GB (if it is a .tif file)")

    if is3D:
        m = m[:, indices[0], indices[1], indices[2]]
    else:
        m = m[:, indices[0], indices[1]]

    if template is None:
        if gSig_filt is not None:
            m = caiman.movie(
                np.array([high_pass_filter_space(m_, gSig_filt) for m_ in m]))
        if is3D:     
            # TODO - motion_correct_3d needs to be implemented in movies.py
            template = caiman.motion_correction.bin_median_3d(m) # motion_correct_3d has not been implemented yet - instead initialize to just median image
        else:
            if not m.flags['WRITEABLE']:
                m = m.copy()
            template = caiman.motion_correction.bin_median(
                    m.motion_correct(max_shifts[1], max_shifts[0], template=None)[0])

    new_templ = template
    if add_to_movie is None:
        add_to_movie = -np.min(template)

    if np.isnan(add_to_movie):
        logger.error('The movie contains NaNs. NaNs are not allowed!')
        raise Exception('The movie contains NaNs. NaNs are not allowed!')
    else:
        logger.debug(f'Adding to movie {add_to_movie}')

    save_movie = False
    fname_tot_rig = None
    res_rig:list = []
    for iter_ in range(num_iter):
        logger.debug(iter_)
        old_templ = new_templ.copy()
        if iter_ == num_iter - 1:
            save_movie = save_movie_rigid
            logger.debug('saving!')


        if isinstance(fname, tuple):
            base_name=os.path.splitext(os.path.split(fname[0])[-1])[0] + '_rig_'
        else:
            base_name=os.path.splitext(os.path.split(fname)[-1])[0] + '_rig_'

        fname_tot_rig, res_rig = motion_correction_piecewise(fname, splits, strides=None, overlaps=None,
                                                             add_to_movie=add_to_movie, template=old_templ, max_shifts=max_shifts, max_deviation_rigid=0,
                                                             dview=dview, save_movie=save_movie, base_name=base_name, subidx = subidx,
                                                             num_splits=num_splits_to_process, shifts_opencv=shifts_opencv, nonneg_movie=nonneg_movie, gSig_filt=gSig_filt,
                                                             use_cuda=use_cuda, border_nan=border_nan, var_name_hdf5=var_name_hdf5, is3D=is3D,
                                                             indices=indices, shifts_interpolate=shifts_interpolate)
        if is3D:
            new_templ = np.nanmedian(np.stack([r[-1] for r in res_rig]), 0)           
        else:
            new_templ = np.nanmedian(np.dstack([r[-1] for r in res_rig]), -1)
        if gSig_filt is not None:
            new_templ = high_pass_filter_space(new_templ, gSig_filt)

    total_template = new_templ
    templates = []
    shifts:list = []
    for rr in res_rig:
        shift_info, idxs, tmpl = rr
        templates.append(tmpl)
        shifts += [sh[0] for sh in shift_info[:len(idxs)]]

    return fname_tot_rig, total_template, templates, shifts

def motion_correct_batch_pwrigid(fname, max_shifts, strides, overlaps, add_to_movie, newoverlaps=None, newstrides=None,
                                 dview=None, upsample_factor_grid=4, max_deviation_rigid=3,
                                 splits=56, num_splits_to_process=None, num_iter=1,
                                 template=None, shifts_opencv=False, save_movie=False, nonneg_movie=False, gSig_filt=None,
                                 use_cuda=False, border_nan=True, var_name_hdf5='mov', is3D=False,
                                 indices=(slice(None), slice(None)), shifts_interpolate=False):
    """
    Function that perform memory efficient hyper parallelized rigid motion corrections while also saving a memory mappable file

    Args:
        fname: str
            name of the movie to motion correct. It should not contain nans. All the loadable formats from CaImAn are acceptable

        strides: tuple
            strides of patches along x and y (and z if 3D)

        overlaps:
            overlaps of patches along x and y (and z if 3D). example: If strides = (64,64) and overlaps (32,32) patches will be (96,96)

        newstrides: tuple
            overlaps after upsampling

        newoverlaps: tuple
            strides after upsampling

        max_shifts: tuple
            x and y maximum allowed shifts (and z if 3D)

        dview: ipyparallel view
            used to perform parallel computing

        splits: int
            number of batches in which the movies is subdivided

        num_splits_to_process: int
            number of batches to process. when not None, the movie is not saved since only a random subset of batches will be processed

        num_iter: int
            number of iterations to perform. The more iteration the better will be the template.

        template: ndarray
            if a good approximation of the template to register is available, it can be used

        shifts_opencv: bool
             toggle the shifts applied with opencv, if yes faster but induces some smoothing

        save_movie_rigid: bool
             toggle save movie

        use_cuda : bool (DEPRECATED)
            cuda is no longer supported for motion correction; this kwarg will be removed in a future version of Caiman

        indices: tuple(slice), default: (slice(None), slice(None))
           Use that to apply motion correction only on a part of the FOV

    Returns:
        fname_tot_rig: str

        total_template:ndarray

        templates:list
            list of produced templates, one per batch

        shifts: list
            inferred rigid shifts to correct the movie

    Raises:
        Exception 'You need to initialize the template with a good estimate. See the motion'
                        '_correct_batch_rigid function'
    """
    logger = logging.getLogger("caiman")

    if template is None:
        raise Exception('You need to initialize the template with a good estimate. See the motion'
                        '_correct_batch_rigid function')
    else:
        new_templ = template

    if np.isnan(add_to_movie):
        logger.error('The template contains NaNs. NaNs are not allowed!')
        raise Exception('The template contains NaNs. NaNs are not allowed!')
    else:
        logger.debug(f'Adding to movie {add_to_movie}')

    for iter_ in range(num_iter):
        logger.debug(iter_)
        old_templ = new_templ.copy()

        if iter_ == num_iter - 1:
            save_movie = save_movie
            if save_movie:

                if isinstance(fname, tuple):
                    logger.debug(f'saving mmap of {fname[0]} to {fname[-1]}')
                else:
                    logger.debug(f'saving mmap of {fname}')

        if isinstance(fname, tuple): # TODO Switch to using standard path functions
            base_name=os.path.splitext(os.path.split(fname[0])[-1])[0] + '_els_'
        else:
            base_name=os.path.splitext(os.path.split(fname)[-1])[0] + '_els_'

        fname_tot_els, res_el = motion_correction_piecewise(fname, splits, strides, overlaps,
                                                            add_to_movie=add_to_movie, template=old_templ, max_shifts=max_shifts,
                                                            max_deviation_rigid=max_deviation_rigid,
                                                            newoverlaps=newoverlaps, newstrides=newstrides,
                                                            upsample_factor_grid=upsample_factor_grid, order='F', dview=dview, save_movie=save_movie,
                                                            base_name=base_name, num_splits=num_splits_to_process,
                                                            shifts_opencv=shifts_opencv, nonneg_movie=nonneg_movie, gSig_filt=gSig_filt,
                                                            use_cuda=use_cuda, border_nan=border_nan, var_name_hdf5=var_name_hdf5, is3D=is3D,
                                                            indices=indices, shifts_interpolate=shifts_interpolate)
        if is3D:
            new_templ = np.nanmedian(np.stack([r[-1] for r in res_el]), 0)
        else:
            new_templ = np.nanmedian(np.dstack([r[-1] for r in res_el]), -1)
        if gSig_filt is not None:
            new_templ = high_pass_filter_space(new_templ, gSig_filt)

    total_template = new_templ
    templates = []
    x_shifts = []
    y_shifts = []
    z_shifts = []
    coord_shifts = []
    for rr in res_el:
        shift_info_chunk, idxs_chunk, tmpl_chunk = rr
        templates.append(tmpl_chunk)
        for shift_info, _ in zip(shift_info_chunk, idxs_chunk):
            if is3D:
                total_shift, _, xyz_grid = shift_info
                x_shifts.append(np.array([sh[0] for sh in total_shift]))
                y_shifts.append(np.array([sh[1] for sh in total_shift]))
                z_shifts.append(np.array([sh[2] for sh in total_shift]))
                coord_shifts.append(xyz_grid)
            else:
                total_shift, _, xy_grid = shift_info
                x_shifts.append(np.array([sh[0] for sh in total_shift]))
                y_shifts.append(np.array([sh[1] for sh in total_shift]))
                coord_shifts.append(xy_grid)

    return fname_tot_els, total_template, templates, x_shifts, y_shifts, z_shifts, coord_shifts

def tile_and_correct_wrapper(params):
    """Does motion correction on specified image frames

    Returns:
    shift_info:
    idxs:
    mean_img: mean over all frames of corrected image (to get individ frames, use out_fname to write them to disk)

    Notes:
    Also writes corrected frames to the mmap file specified by out_fname (if not None)

    """
    # todo todocument
    logger = logging.getLogger("caiman")

    try:
        cv2.setNumThreads(0)
    except:
        pass  # 'OpenCV is naturally single threaded'

    img_name, out_fname, idxs, shape_mov, template, strides, overlaps, max_shifts,\
        add_to_movie, max_deviation_rigid, upsample_factor_grid, newoverlaps, newstrides, \
        shifts_opencv, nonneg_movie, gSig_filt, is_fiji, use_cuda, border_nan, var_name_hdf5, \
        is3D, indices, shifts_interpolate = params


    if isinstance(img_name, tuple):
        name, extension = os.path.splitext(img_name[0])[:2]
    else:
        name, extension = os.path.splitext(img_name)[:2]
    extension = extension.lower()
    shift_info = []

    imgs = caiman.load(img_name, subindices=idxs, var_name_hdf5=var_name_hdf5,is3D=is3D)
    imgs = imgs[(slice(None),) + indices]
    mc = np.zeros(imgs.shape, dtype=np.float32)
    if not imgs[0].shape == template.shape:
        template = template[indices]
    for count, img in enumerate(imgs):
        if count % 10 == 0:
            logger.debug(count)
        if is3D:
            mc[count], total_shift, start_step, xyz_grid = tile_and_correct_3d(img, template, strides, overlaps, max_shifts,
                                                                       add_to_movie=add_to_movie, newoverlaps=newoverlaps,
                                                                       newstrides=newstrides,
                                                                       upsample_factor_grid=upsample_factor_grid,
                                                                       upsample_factor_fft=10, show_movie=False,
                                                                       max_deviation_rigid=max_deviation_rigid,
                                                                       shifts_opencv=shifts_opencv, gSig_filt=gSig_filt,
                                                                       use_cuda=use_cuda, border_nan=border_nan,
                                                                       shifts_interpolate=shifts_interpolate)
            shift_info.append([total_shift, start_step, xyz_grid])
            
        else:
            mc[count], total_shift, start_step, xy_grid = tile_and_correct(img, template, strides, overlaps, max_shifts,
                                                                       add_to_movie=add_to_movie, newoverlaps=newoverlaps,
                                                                       newstrides=newstrides,
                                                                       upsample_factor_grid=upsample_factor_grid,
                                                                       upsample_factor_fft=10, show_movie=False,
                                                                       max_deviation_rigid=max_deviation_rigid,
                                                                       shifts_opencv=shifts_opencv, gSig_filt=gSig_filt,
                                                                       use_cuda=use_cuda, border_nan=border_nan,
                                                                       shifts_interpolate=shifts_interpolate)
            shift_info.append([total_shift, start_step, xy_grid])

    if out_fname is not None:
        outv = np.memmap(out_fname, mode='r+', dtype=np.float32,
                         shape=caiman.mmapping.prepare_shape(shape_mov), order='F')
        if nonneg_movie:
            bias = np.float32(add_to_movie)
        else:
            bias = 0
        outv[:, idxs] = np.reshape(
            mc.astype(np.float32), (len(imgs), -1), order='F').T + bias
    new_temp = np.nanmean(mc, 0)
    new_temp[np.isnan(new_temp)] = np.nanmin(new_temp)
    return shift_info, idxs, new_temp

def motion_correction_piecewise(fname, splits, strides, overlaps, add_to_movie=0, template=None,
                                max_shifts=(12, 12), max_deviation_rigid=3, newoverlaps=None, newstrides=None,
                                upsample_factor_grid=4, order='F', dview=None, save_movie=True,
                                base_name=None, subidx = None, num_splits=None, shifts_opencv=False, nonneg_movie=False, gSig_filt=None,
                                use_cuda=False, border_nan=True, var_name_hdf5='mov', is3D=False,
                                indices=(slice(None), slice(None)), shifts_interpolate=False):
    """

    """
    # todo todocument
    logger = logging.getLogger("caiman")
    if isinstance(fname, tuple):
        name, extension = os.path.splitext(fname[0])[:2]
    else:
        name, extension = os.path.splitext(fname)[:2]
    extension = extension.lower()
    is_fiji = False

    dims, T = caiman.base.movies.get_file_size(fname, var_name_hdf5=var_name_hdf5)
    z = np.zeros(dims)
    dims = z[indices].shape
    logger.debug(f'Number of Splits: {splits}')
    if isinstance(splits, int):
        if subidx is None:
            rng = range(T)
        else:
            rng = range(T)[subidx]

        idxs = np.array_split(list(rng), splits)

    else:
        idxs = splits
        save_movie = False
    if template is None:
        raise Exception('motion_correction_piecewise(): Templateless not implemented')
    
    shape_mov = (np.prod(dims), T)
    if num_splits is not None:
        idxs = np.array(idxs)[np.random.randint(0, len(idxs), num_splits)]
        save_movie = False
        logger.warning('**** Movie not saved because num_splits is not None ****')

    if save_movie:
        if base_name is None:
            base_name = os.path.splitext(os.path.split(fname)[1])[0]
        base_name = caiman.paths.fn_relocated(base_name)
        fname_tot:Optional[str] = caiman.paths.memmap_frames_filename(base_name, dims, T, order)

        np.memmap(fname_tot, mode='w+', dtype=np.float32,
                  shape=caiman.mmapping.prepare_shape(shape_mov), order=order)
        logger.info(f'Saving file as {fname_tot}')
    else:
        fname_tot = None

    pars = []
    for idx in idxs:
        logger.debug(f'Extracting parameters for frames: {idx}')
        pars.append([fname, fname_tot, idx, shape_mov, template, strides, overlaps, max_shifts, np.array(
            add_to_movie, dtype=np.float32), max_deviation_rigid, upsample_factor_grid,
            newoverlaps, newstrides, shifts_opencv, nonneg_movie, gSig_filt, is_fiji,
            use_cuda, border_nan, var_name_hdf5, is3D, indices, shifts_interpolate])

    if dview is not None:
        logger.info('** Starting parallel motion correction **')
        if 'multiprocessing' in str(type(dview)):
            logger.debug("entering multiprocessing tile_and_correct_wrapper")
            res = dview.map_async(tile_and_correct_wrapper, pars).get(4294967)
        else:
            res = dview.map_sync(tile_and_correct_wrapper, pars)
        logger.info('** Finished parallel motion correction **')
    else:
        res = list(map(tile_and_correct_wrapper, pars))

    return fname_tot, res

