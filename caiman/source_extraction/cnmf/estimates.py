#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 11:11:45 2018

@author: epnevmatikakis
"""

import logging
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse
from typing import List
import time

import caiman
from .utilities import detrend_df_f
from .spatial import threshold_components
from .temporal import constrained_foopsi_parallel
from .merging import merge_iteration
from ...components_evaluation import (
        evaluate_components_CNN, estimate_components_quality_auto,
        select_components_from_metrics)
from ...base.rois import (
        detect_duplicates_and_subsets, nf_match_neurons_in_binary_masks,
        nf_masks_to_neurof_dict)
from .initialization import downscale


class Estimates(object):
    """
    Class for storing and reusing the analysis results and performing basic
    processing and plotting operations.
    """
    def __init__(self, A=None, b=None, C=None, f=None, R=None, dims=None):
        """Class for storing the variables related to the estimates of spatial footprints, temporal traces,
        deconvolved neural activity, and background. Quality metrics are also stored. The class has methods
        for evaluating the quality of each component, DF/F normalization and some basic plotting.

        Args:
            A:  scipy.sparse.csc_matrix (dimensions: # of pixels x # components)
                set of spatial footprints. Each footprint is represented in a column of A, flattened with order = 'F'

            C:  np.ndarray (dimensions: # of components x # of timesteps)
                set of temporal traces (each row of C corresponds to a trace)

            f:  np.ndarray (dimensions: # of background components x # of timesteps)
                set of temporal background components

            b:  np.ndarray or scipy.sparse.csc_matrix (dimensions: # of pixels x # of background components)
                set of spatial background components, flattened with order = 'F'

            R:  np.ndarray (dimensions: # of components x # of timesteps)
                set of trace residuals

            YrA:    np.ndarray (dimensions: # of components x # of timesteps)
                set of trace residuals

            S:  np.ndarray (dimensions: # of components x # of timesteps)
                set of deconvolved neural activity traces

            F_dff:  np.ndarray (dimensions: # of components x # of timesteps)
                set of DF/F normalized activity traces (only for 2p)

            W:  scipy.sparse.coo_matrix (dimensions: # of pixels x # of pixels)
                Ring model matrix (used in 1p processing with greedy_pnr for background computation)

            b0: np.ndarray (dimensions: # of pixels)
                constant baseline for each pixel

            sn: np.ndarray (dimensions: # of pixels)
                noise std for each pixel

            g:  list (length: # of components)
                time constants for each trace

            bl: list (length: # of components)
                constant baseline for each trace

            c1: list (length: # of components)
                initial value for each trace

            neurons_sn: list (length: # of components)
                noise std for each trace

            center: list (length: # of components)
                centroid coordinate for each spatial footprint

            coordinates: list (length: # of components)
                contour plot for each spatial footprint

            idx_components: list
                indices of accepted components

            idx_components_bad: list
                indices of rejected components

            SNR_comp: np.ndarray
                trace SNR for each component

            r_values: np.ndarray
                space correlation for each component

            cnn_preds: np.ndarray
                CNN predictions for each component
        """
        # variables related to the estimates of traces, footprints, deconvolution and background
        self.A = A
        self.C = C
        self.f = f
        self.b = b
        self.R = R
        self.W = None
        self.b0 = None
        self.YrA = None

        self.S = None
        self.sn = None
        self.g = None
        self.bl = None
        self.c1 = None
        self.neurons_sn = None
        self.lam = None

        self.center = None

        self.merged_ROIs = None
        self.coordinates = None
        self.F_dff = None

        self.idx_components = None
        self.idx_components_bad = None
        self.SNR_comp = None
        self.r_values = None
        self.cnn_preds = None

        # online

        self.noisyC = None
        self.C_on = None
        self.Ab = None
        self.Cf = None
        self.OASISinstances = None
        self.CY = None
        self.CC = None
        self.Ab_dense = None
        self.Yr_buf = None
        self.mn = None
        self.vr = None
        self.ind_new = None
        self.rho_buf = None
        self.AtA = None
        self.AtY_buf = None
        self.sv = None
        self.groups = None

        self.dims = dims
        self.shifts:List = []

        self.A_thr = None
        self.discarded_components = None



    def plot_contours(self, img=None, idx=None, crd=None, thr_method='max',
                      thr='0.2', display_numbers=True, params=None):
        """view contours of all spatial footprints.

        Args:
            img :   np.ndarray
                background image for contour plotting. Default is the mean
                image of all spatial components (d1 x d2)
            idx :   list
                list of accepted components
            crd :   list
                list of coordinates (if empty they are computed)
            thr_method : str
                thresholding method for computing contours ('max', 'nrg')
            thr : float
                threshold value
            display_numbers :   bool
                flag for displaying the id number of each contour
            params : params object
                set of dictionary containing the various parameters
        """
        if 'csc_matrix' not in str(type(self.A)):
            self.A = scipy.sparse.csc_matrix(self.A)
        if img is None:
            img = np.reshape(np.array(self.A.mean(1)), self.dims, order='F')
        if self.coordinates is None:  # not hasattr(self, 'coordinates'):
            self.coordinates = caiman.utils.visualization.get_contours(self.A, img.shape, thr=thr, thr_method=thr_method)
        plt.figure()
        if params is not None:
            plt.suptitle('min_SNR=%1.2f, rval_thr=%1.2f, use_cnn=%i'
                         %(params.quality['SNR_lowest'],
                           params.quality['rval_thr'],
                           int(params.quality['use_cnn'])))
        if idx is None:
            caiman.utils.visualization.plot_contours(self.A, img, coordinates=self.coordinates,
                                                     display_numbers=display_numbers)
        else:
            if not isinstance(idx, list):
                idx = idx.tolist()
            coor_g = [self.coordinates[cr] for cr in idx]
            bad = list(set(range(self.A.shape[1])) - set(idx))
            coor_b = [self.coordinates[cr] for cr in bad]
            plt.subplot(1, 2, 1)
            caiman.utils.visualization.plot_contours(self.A[:, idx], img,
                                                     coordinates=coor_g,
                                                     display_numbers=display_numbers)
            plt.title('Accepted Components')
            bad = list(set(range(self.A.shape[1])) - set(idx))
            plt.subplot(1, 2, 2)
            caiman.utils.visualization.plot_contours(self.A[:, bad], img,
                                                     coordinates=coor_b,
                                                     display_numbers=display_numbers)
            plt.title('Rejected Components')
        return self

    def plot_contours_nb(self, img=None, idx=None, crd=None, thr_method='max',
                         thr='0.2', params=None):
        """view contours of all spatial footprints (notebook environment).

        Args:
            img :   np.ndarray
                background image for contour plotting. Default is the mean
                image of all spatial components (d1 x d2)
            idx :   list
                list of accepted components
            crd :   list
                list of coordinates (if empty they are computed)
            thr_method : str
                thresholding method for computing contours ('max', 'nrg')
            thr : float
                threshold value
            params : params object
                set of dictionary containing the various parameters
        """
        try:
            import bokeh
            if 'csc_matrix' not in str(type(self.A)):
                self.A = scipy.sparse.csc_matrix(self.A)
            if img is None:
                img = np.reshape(np.array(self.A.mean(1)), self.dims, order='F')
            if self.coordinates is None:  # not hasattr(self, 'coordinates'):
                self.coordinates = caiman.utils.visualization.get_contours(self.A,
                                        self.dims, thr=thr, thr_method=thr_method)
            if idx is None:
                p = caiman.utils.visualization.nb_plot_contour(img, self.A, self.dims[0],
                                self.dims[1], coordinates=self.coordinates,
                                thr_method=thr_method, thr=thr, show=False)
                p.title.text = 'Contour plots of found components'
                if params is not None:
                    p.xaxis.axis_label = '''\
                    min_SNR={min_SNR}, rval_thr={rval_thr}, use_cnn={use_cnn}\
                    '''.format(min_SNR=params.quality['SNR_lowest'],
                               rval_thr=params.quality['rval_thr'],
                               use_cnn=params.quality['use_cnn'])
                bokeh.plotting.show(p)
            else:
                if not isinstance(idx, list):
                    idx = idx.tolist()
                coor_g = [self.coordinates[cr] for cr in idx]
                bad = list(set(range(self.A.shape[1])) - set(idx))
                coor_b = [self.coordinates[cr] for cr in bad]
                p1 = caiman.utils.visualization.nb_plot_contour(img, self.A[:, idx],
                                self.dims[0], self.dims[1], coordinates=coor_g,
                                thr_method=thr_method, thr=thr, show=False)
                p1.plot_width = 450
                p1.plot_height = 450 * self.dims[0] // self.dims[1]
                p1.title.text = "Accepted Components"
                if params is not None:
                    p1.xaxis.axis_label = '''\
                    min_SNR={min_SNR}, rval_thr={rval_thr}, use_cnn={use_cnn}\
                    '''.format(min_SNR=params.quality['SNR_lowest'],
                               rval_thr=params.quality['rval_thr'],
                               use_cnn=params.quality['use_cnn'])
                bad = list(set(range(self.A.shape[1])) - set(idx))
                p2 = caiman.utils.visualization.nb_plot_contour(img, self.A[:, bad],
                                self.dims[0], self.dims[1], coordinates=coor_b,
                                thr_method=thr_method, thr=thr, show=False)
                p2.plot_width = 450
                p2.plot_height = 450 * self.dims[0] // self.dims[1]
                p2.title.text = 'Rejected Components'
                if params is not None:
                    p2.xaxis.axis_label = '''\
                    min_SNR={min_SNR}, rval_thr={rval_thr}, use_cnn={use_cnn}\
                    '''.format(min_SNR=params.quality['SNR_lowest'],
                               rval_thr=params.quality['rval_thr'],
                               use_cnn=params.quality['use_cnn'])
                bokeh.plotting.show(bokeh.layouts.row(p1, p2))
        except:
            print("Bokeh could not be loaded. Either it is not installed or you are not running within a notebook")
            print("Using non-interactive plot as fallback")
            self.plot_contours(img=img, idx=idx, crd=crd, thr_method=thr_method,
                         thr=thr, params=params)
        return self

    def view_components(self, Yr=None, img=None, idx=None):
        """view spatial and temporal components interactively

        Args:
            Yr :    np.ndarray
                movie in format pixels (d) x frames (T)

            img :   np.ndarray
                background image for contour plotting. Default is the mean
                image of all spatial components (d1 x d2)

            idx :   list
                list of components to be plotted
        """
        if 'csc_matrix' not in str(type(self.A)):
            self.A = scipy.sparse.csc_matrix(self.A)

        plt.ion()
        nr, T = self.C.shape
        if self.R is None:
            self.R = self.YrA
        if self.R.shape != (nr, T):
            if self.YrA is None:
                self.compute_residuals(Yr)
            else:
                self.R = self.YrA

        if img is None:
            img = np.reshape(np.array(self.A.mean(axis=1)), self.dims, order='F')

        if idx is None:
            caiman.utils.visualization.view_patches_bar(Yr, self.A, self.C,
                    self.b, self.f, self.dims[0], self.dims[1], YrA=self.R, img=img)
        else:
            caiman.utils.visualization.view_patches_bar(Yr, self.A.tocsc()[:,idx],
                                                        self.C[idx], self.b, self.f,
                                                        self.dims[0], self.dims[1], YrA=self.R[idx], img=img)
        return self

    def nb_view_components(self, Yr=None, img=None, idx=None,
                           denoised_color=None, cmap='jet', thr=0.99):
        """view spatial and temporal components interactively in a notebook

        Args:
            Yr :    np.ndarray
                movie in format pixels (d) x frames (T)

            img :   np.ndarray
                background image for contour plotting. Default is the mean
                image of all spatial components (d1 x d2)

            idx :   list
                list of components to be plotted

            thr: double
                threshold regulating the extent of the displayed patches

            denoised_color: string or None
                color name (e.g. 'red') or hex color code (e.g. '#F0027F')

            cmap: string
                name of colormap (e.g. 'viridis') used to plot image_neurons
        """
        if 'csc_matrix' not in str(type(self.A)):
            self.A = scipy.sparse.csc_matrix(self.A)

        plt.ion()
        nr, T = self.C.shape
        if self.R is None:
            self.R = self.YrA
        if self.R.shape != [nr, T]:
            if self.YrA is None:
                self.compute_residuals(Yr)
            else:
                self.R = self.YrA

        if img is None:
            img = np.reshape(np.array(self.A.mean(axis=1)), self.dims, order='F')

        if idx is None:
            caiman.utils.visualization.nb_view_patches(Yr, self.A, self.C,
                    self.b, self.f, self.dims[0], self.dims[1], YrA=self.R, image_neurons=img,
                    thr=thr, denoised_color=denoised_color, cmap=cmap)
        else:
            caiman.utils.visualization.nb_view_patches(Yr, self.A.tocsc()[:,idx],
                                                        self.C[idx], self.b, self.f,
                                                        self.dims[0], self.dims[1], YrA=self.R[idx], image_neurons=img,
                                                        thr=thr, denoised_color=denoised_color, cmap=cmap)
        return self

    def hv_view_components(self, Yr=None, img=None, idx=None,
                           denoised_color=None, cmap='viridis'):
        """view spatial and temporal components interactively in a notebook

        Args:
            Yr :    np.ndarray
                movie in format pixels (d) x frames (T)

            img :   np.ndarray
                background image for contour plotting. Default is the mean
                image of all spatial components (d1 x d2)

            idx :   list
                list of components to be plotted

            denoised_color: string or None
                color name (e.g. 'red') or hex color code (e.g. '#F0027F')

            cmap: string
                name of colormap (e.g. 'viridis') used to plot image_neurons
        """
        if 'csc_matrix' not in str(type(self.A)):
            self.A = scipy.sparse.csc_matrix(self.A)

        plt.ion()
        nr, T = self.C.shape
        if self.R is None:
            self.R = self.YrA
        if self.R.shape != [nr, T]:
            if self.YrA is None:
                self.compute_residuals(Yr)
            else:
                self.R = self.YrA

        if img is None:
            img = np.reshape(np.array(self.A.mean(axis=1)), self.dims, order='F')

        if idx is None:
            hv_plot = caiman.utils.visualization.hv_view_patches(
                Yr, self.A, self.C, self.b, self.f, self.dims[0], self.dims[1],
                YrA=self.R, image_neurons=img, denoised_color=denoised_color,
                cmap=cmap)
        else:
            hv_plot = caiman.utils.visualization.hv_view_patches(
                Yr, self.A.tocsc()[:, idx], self.C[idx], self.b, self.f,
                self.dims[0], self.dims[1], YrA=self.R[idx], image_neurons=img,
                denoised_color=denoised_color, cmap=cmap)
        return hv_plot

    def nb_view_components_3d(self, Yr=None, image_type='mean', dims=None,
                              max_projection=False, axis=0,
                              denoised_color=None, cmap='jet', thr=0.9):
        """view spatial and temporal components interactively in a notebook
        (version for 3d data)

        Args:
            Yr :    np.ndarray
                movie in format pixels (d) x frames (T) (only required to
                compute the correlation image)


            dims: tuple of ints
                dimensions of movie (x, y and z)

            image_type: 'mean'|'max'|'corr'
                image to be overlaid to neurons (average of shapes,
                maximum of shapes or nearest neigbor correlation of raw data)

            max_projection: bool
                plot max projection along specified axis if True, o/w plot layers

            axis: int (0, 1 or 2)
                axis along which max projection is performed or layers are shown

            thr: scalar between 0 and 1
                Energy threshold for computing contours

            denoised_color: string or None
                color name (e.g. 'red') or hex color code (e.g. '#F0027F')

            cmap: string
                name of colormap (e.g. 'viridis') used to plot image_neurons

        """
        if 'csc_matrix' not in str(type(self.A)):
            self.A = scipy.sparse.csc_matrix(self.A)
        if dims is None:
            dims = self.dims
        plt.ion()
        nr, T = self.C.shape
        if self.R is None:
            self.R = self.YrA
        if self.R.shape != [nr, T]:
            if self.YrA is None:
                self.compute_residuals(Yr)
            else:
                self.R = self.YrA

        caiman.utils.visualization.nb_view_patches3d(self.YrA, self.A, self.C,
                    dims=dims, image_type=image_type, Yr=Yr,
                    max_projection=max_projection, axis=axis, thr=thr,
                    denoised_color=denoised_color, cmap=cmap)

        return self

    def make_color_movie(self, imgs, q_max=99.75, q_min=2, gain_res=1,
                         magnification=1, include_bck=True,
                         frame_range=slice(None, None, None),
                         bpx=0, save_movie=False, display=True,
                         movie_name='results_movie_color.avi',
                         opencv_code='H264'):
        """
        Displays a color movie where each component is given an arbitrary
        color. Will be merged with play_movie soon. Check that function for
        arg definitions.
        """
        dims = imgs.shape[1:]
        cols_c = np.random.rand(self.C.shape[0], 1, 3)
        cols_f = np.ones((self.f.shape[0], 1, 3))/8
        Cs = np.vstack((np.expand_dims(self.C[:, frame_range], -1)*cols_c,
                        np.expand_dims(self.f[:, frame_range], -1)*cols_f))
        AC = np.tensordot(np.hstack((self.A.toarray(), self.b)), Cs, axes=(1, 0))
        AC = AC.reshape((dims) + (-1, 3)).transpose(2, 0, 1, 3)
        
        AC /= np.percentile(AC, 99.75, axis=(0, 1, 2))
        mov = caiman.movie(np.concatenate((np.repeat(np.expand_dims(imgs[frame_range]/np.percentile(imgs[:1000], 99.75), -1), 3, 3),
                                           AC), axis=2))
        if not display:
            return mov

        mov.play(q_min=q_min, q_max=q_max, magnification=magnification,
                 save_movie=save_movie, movie_name=movie_name)

        return mov


    def play_movie(self, imgs, q_max=99.75, q_min=2, gain_res=1,
                   magnification=1, include_bck=True,
                   frame_range=slice(None, None, None),
                   bpx=0, thr=0., save_movie=False,
                   movie_name='results_movie.avi',
                   display=True, opencv_codec='H264',
                   use_color=False, gain_color=4, gain_bck=0.2):
        """
        Displays a movie with three panels (original data (left panel),
        reconstructed data (middle panel), residual (right panel))

        Args:
            imgs: np.array (possibly memory mapped, t,x,y[,z])
                Imaging data

            q_max: float (values in [0, 100], default: 99.75)
                percentile for maximum plotting value

            q_min: float (values in [0, 100], default: 1)
                percentile for minimum plotting value

            gain_res: float (1)
                amplification factor for residual movie

            magnification: float (1)
                magnification factor for whole movie

            include_bck: bool (True)
                flag for including background in original and reconstructed movie

            frame_rage: range or slice or list (default: slice(None))
                display only a subset of frames

            bpx: int (deafult: 0)
                number of pixels to exclude on each border

            thr: float (values in [0, 1[) (default: 0)
                threshold value for contours, no contours if thr=0

            save_movie: bool (default: False)
                flag to save an avi file of the movie

            movie_name: str (default: 'results_movie.avi')
                name of saved file

            display: bool (deafult: True)
                flag for playing the movie (to stop the movie press 'q')

            opencv_codec: str (default: 'H264')
                FourCC video codec for saving movie. Check http://www.fourcc.org/codecs.php

            use_color: bool (default: False)
                flag for making a color movie. If True a random color will be assigned
                for each of the components

            gain_color: float (default: 4)
                amplify colors in the movie to make them brighter

            gain_bck: float (default: 0.2)
                dampen background in the movie to expose components (applicable
                only when color is used.)

        Returns:
            mov: The concatenated output movie
        """
        dims = imgs.shape[1:]
        if 'movie' not in str(type(imgs)):
            imgs = caiman.movie(imgs[frame_range])
        else:
            imgs = imgs[frame_range]

        if use_color:
            cols_c = np.random.rand(self.C.shape[0], 1, 3)*gain_color
            Cs = np.expand_dims(self.C[:, frame_range], -1)*cols_c
            #AC = np.tensordot(np.hstack((self.A.toarray(), self.b)), Cs, axes=(1, 0))
            Y_rec_color = np.tensordot(self.A.toarray(), Cs, axes=(1, 0))
            Y_rec_color = Y_rec_color.reshape((dims) + (-1, 3), order='F').transpose(2, 0, 1, 3)

        AC = self.A.dot(self.C[:, frame_range])
        Y_rec = AC.reshape(dims + (-1,), order='F')
        Y_rec = Y_rec.transpose([2, 0, 1])
        if self.W is not None:
            ssub_B = int(round(np.sqrt(np.prod(dims) / self.W.shape[0])))
            B = imgs.reshape((-1, np.prod(dims)), order='F').T - AC
            if ssub_B == 1:
                B = self.b0[:, None] + self.W.dot(B - self.b0[:, None])
            else:
                WB = self.W.dot(downscale(B.reshape(dims + (B.shape[-1],), order='F'),
                              (ssub_B, ssub_B, 1)).reshape((-1, B.shape[-1]), order='F'))
                Wb0 = self.W.dot(downscale(self.b0.reshape(dims, order='F'),
                              (ssub_B, ssub_B)).reshape((-1, 1), order='F'))
                B = self.b0.flatten('F')[:, None] + (np.repeat(np.repeat((WB - Wb0).reshape(((dims[0] - 1) // ssub_B + 1, (dims[1] - 1) // ssub_B + 1, -1), order='F'),
                                     ssub_B, 0), ssub_B, 1)[:dims[0], :dims[1]].reshape((-1, B.shape[-1]), order='F'))
            B = B.reshape(dims + (-1,), order='F').transpose([2, 0, 1])
        elif self.b is not None and self.f is not None:
            B = self.b.dot(self.f[:, frame_range])
            if 'matrix' in str(type(B)):
                B = B.toarray()
            B = B.reshape(dims + (-1,), order='F').transpose([2, 0, 1])
        else:
            B = np.zeros_like(Y_rec)
        if bpx > 0:
            B = B[:, bpx:-bpx, bpx:-bpx]
            Y_rec = Y_rec[:, bpx:-bpx, bpx:-bpx]
            imgs = imgs[:, bpx:-bpx, bpx:-bpx]

        Y_res = imgs - Y_rec - B
        if use_color:
            if bpx > 0:
                Y_rec_color = Y_rec_color[:, bpx:-bpx, bpx:-bpx]
            mov = caiman.concatenate((np.repeat(np.expand_dims(imgs - (not include_bck) * B, -1), 3, 3),
                                      Y_rec_color + include_bck * np.expand_dims(B*gain_bck, -1),
                                      np.repeat(np.expand_dims(Y_res * gain_res, -1), 3, 3)), axis=2)
        else:
            mov = caiman.concatenate((imgs[frame_range] - (not include_bck) * B,
                                      Y_rec + include_bck * B, Y_res * gain_res), axis=2)
        if not display:
            return mov

        if thr > 0:
            import cv2
            if save_movie:
                fourcc = cv2.VideoWriter_fourcc(*opencv_codec)
                out = cv2.VideoWriter(movie_name, fourcc, 30.0,
                                      tuple([int(magnification*s) for s in mov.shape[1:][::-1]]))
            contours = []
            for a in self.A.T.toarray():
                a = a.reshape(dims, order='F')
                if bpx > 0:
                    a = a[bpx:-bpx, bpx:-bpx]
                # a = cv2.GaussianBlur(a, (9, 9), .5)
                if magnification != 1:
                    a = cv2.resize(a, None, fx=magnification, fy=magnification,
                                   interpolation=cv2.INTER_LINEAR)
                ret, thresh = cv2.threshold(a, thr * np.max(a), 1., 0)
                contour, hierarchy = cv2.findContours(
                    thresh.astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                contours.append(contour)
                contours.append(list([c + np.array([[a.shape[1], 0]]) for c in contour]))
                contours.append(list([c + np.array([[2 * a.shape[1], 0]]) for c in contour]))

            maxmov = np.nanpercentile(mov[0:10], q_max) if q_max < 100 else np.nanmax(mov)
            minmov = np.nanpercentile(mov[0:10], q_min) if q_min > 0 else np.nanmin(mov)
            for iddxx, frame in enumerate(mov):
                if magnification != 1:
                    frame = cv2.resize(frame, None, fx=magnification, fy=magnification,
                                       interpolation=cv2.INTER_LINEAR)
                frame = np.clip((frame - minmov) * 255. / (maxmov - minmov), 0, 255)
                if frame.ndim < 3:
                    frame = np.repeat(frame[..., None], 3, 2)
                for contour in contours:
                    cv2.drawContours(frame, contour, -1, (0, 255, 255), 1)
                cv2.imshow('frame', frame.astype('uint8'))
                if save_movie:
                    out.write(frame.astype('uint8'))
                if cv2.waitKey(30) & 0xFF == ord('q'):
                    break
            if save_movie:
                out.release()
            cv2.destroyAllWindows()
            
        else:
            mov.play(q_min=q_min, q_max=q_max, magnification=magnification,
                     save_movie=save_movie, movie_name=movie_name)

        return mov

    def compute_residuals(self, Yr):
        """compute residual for each component (variable R)

         Args:
             Yr :    np.ndarray
                 movie in format pixels (d) x frames (T)
        """
        if len(Yr.shape) > 2:
            Yr = np.reshape(Yr.transpose(1,2,0), (-1, Yr.shape[0]), order='F')
        if 'csc_matrix' not in str(type(self.A)):
            self.A = scipy.sparse.csc_matrix(self.A)
        if 'array' not in str(type(self.b)):
            self.b = self.b.toarray()
        if 'array' not in str(type(self.C)):
            self.C = self.C.toarray()
        if 'array' not in str(type(self.f)):
            self.f = self.f.toarray()

        Ab = scipy.sparse.hstack((self.A, self.b)).tocsc()
        nA2 = np.ravel(Ab.power(2).sum(axis=0)) + np.finfo(np.float32).eps
        nA2_inv_mat = scipy.sparse.spdiags(
            1. / nA2, 0, nA2.shape[0], nA2.shape[0])
        Cf = np.vstack((self.C, self.f))
        if 'numpy.ndarray' in str(type(Yr)):
            YA = (Ab.T.dot(Yr)).T * nA2_inv_mat
        else:
            YA = caiman.mmapping.parallel_dot_product(Yr, Ab, dview=self.dview,
                        block_size=2000, transpose=True, num_blocks_per_run=5) * nA2_inv_mat

        AA = Ab.T.dot(Ab) * nA2_inv_mat
        self.R = (YA - (AA.T.dot(Cf)).T)[:, :self.A.shape[-1]].T

        return self

    def detrend_df_f(self, quantileMin=8, frames_window=500,
                     flag_auto=True, use_fast=False, use_residuals=True,
                     detrend_only=False):
        """Computes DF/F normalized fluorescence for the extracted traces. See
        caiman.source.extraction.utilities.detrend_df_f for details

        Args:
            quantile_min: float
                quantile used to estimate the baseline (values in [0,100])

            frames_window: int
                number of frames for computing running quantile

            flag_auto: bool
                flag for determining quantile automatically (different for each
                trace)

            use_fast: bool
                flag for using approximate fast percentile filtering

            use_residuals: bool
                flag for using non-deconvolved traces in DF/F calculation

            detrend_only: bool (False)
                flag for only subtracting baseline and not normalizing by it.
                Used in 1p data processing where baseline fluorescence cannot
                be determined.

        Returns:
            self: CNMF object
                self.F_dff contains the DF/F normalized traces
        """

        if self.C is None or self.C.shape[0] == 0:
            logging.warning("There are no components for DF/F extraction!")
            return self

        if use_residuals:
            if self.R is None:
                if self.YrA is None:
                    R = None
                else:
                    R = self.YrA
            else:
                R = self.R
        else:
            R = None

        self.F_dff = detrend_df_f(self.A, self.b, self.C, self.f, self.YrA,
                                  quantileMin=quantileMin,
                                  frames_window=frames_window,
                                  flag_auto=flag_auto, use_fast=use_fast,
                                  detrend_only=detrend_only)
        return self

    def normalize_components(self):
        """ Normalizes components such that spatial components have l_2 norm 1
        """
        if 'csc_matrix' not in str(type(self.A)):
            self.A = scipy.sparse.csc_matrix(self.A)
        if 'array' not in str(type(self.C)):
            self.C = self.C.toarray()
        if 'array' not in str(type(self.f)) and self.f is not None:
            self.f = self.f.toarray()

        nA = np.sqrt(np.ravel(self.A.power(2).sum(axis=0)))
        nA_mat = scipy.sparse.spdiags(nA, 0, nA.shape[0], nA.shape[0])
        nA_inv_mat = scipy.sparse.spdiags(1. / (nA + np.finfo(np.float32).eps), 0, nA.shape[0], nA.shape[0])
        self.A = self.A * nA_inv_mat
        self.C = nA_mat * self.C
        if self.YrA is not None:
            self.YrA = nA_mat * self.YrA
        if self.R is not None:
            self.R = nA_mat * self.YrA
        if self.bl is not None:
            self.bl = nA * self.bl
        if self.c1 is not None:
            self.c1 = nA * self.c1
        if self.neurons_sn is not None:
            self.neurons_sn = nA * self.neurons_sn

        if self.f is not None:  # 1p with exact ring-model
            nB = np.sqrt(np.ravel((self.b.power(2) if scipy.sparse.issparse(self.b)
                         else self.b**2).sum(axis=0)))
            nB_mat = scipy.sparse.spdiags(nB, 0, nB.shape[0], nB.shape[0])
            nB_inv_mat = scipy.sparse.spdiags(1. / (nB + np.finfo(np.float32).eps), 0, nB.shape[0], nB.shape[0])
            self.b = self.b * nB_inv_mat
            self.f = nB_mat * self.f
        return self

    def select_components(self, idx_components=None, use_object=False, save_discarded_components=True):
        """Keeps only a selected subset of components and removes the rest.
        The subset can be either user defined with the variable idx_components
        or read from the estimates object. The flag use_object determines this
        choice. If no subset is present then all components are kept.

        Args:
            idx_components: list
                indices of components to be kept

            use_object: bool
                Flag to use self.idx_components for reading the indices.

            save_discarded_components: bool
                whether to save the components from initialization so that they can be restored using the restore_discarded_components method

        Returns:
            self: Estimates object
        """
        if use_object:
            idx_components = self.idx_components
            idx_components_bad = self.idx_components_bad
        else:
            idx_components_bad = np.setdiff1d(np.arange(self.A.shape[-1]), idx_components)

        if idx_components is not None:
            if save_discarded_components:
                self.discarded_components = Estimates()

            for field in ['C', 'S', 'YrA', 'R', 'F_dff', 'g', 'bl', 'c1', 'neurons_sn', 'lam', 'cnn_preds','SNR_comp','r_values','coordinates']:
                if getattr(self, field) is not None:
                    if type(getattr(self, field)) is list:
                        setattr(self, field, np.array(getattr(self, field)))
                    if len(getattr(self, field)) == self.A.shape[-1]:
                        if save_discarded_components:
                            setattr(self.discarded_components, field, getattr(self, field)[idx_components_bad])
                        setattr(self, field, getattr(self, field)[idx_components])
                    else:
                        print('*** Variable ' + field + ' has not the same number of components as A ***')

            for field in ['A', 'A_thr']:
                if getattr(self, field) is not None:
                    if 'sparse' in str(type(getattr(self, field))):
                        if save_discarded_components:
                            setattr(self.discarded_components, field, getattr(self, field).tocsc()[:, idx_components_bad])
                        setattr(self, field, getattr(self, field).tocsc()[:, idx_components])

                    else:
                        if save_discarded_components:
                            setattr(self.discarded_components, field, getattr(self, field)[:, idx_components_bad])
                        setattr(self, field, getattr(self, field)[:, idx_components])


            self.nr = len(idx_components)

            if save_discarded_components:
                self.discarded_components.nr = len(idx_components_bad)
                self.discarded_components.dims = self.dims

            self.idx_components = None
            self.idx_components_bad = None

        return self

    def restore_discarded_components(self):
        ''' Recover components that are filtered out with the select_components method
        '''
        if self.discarded_components is not None:
            for field in ['C', 'S', 'YrA', 'R', 'F_dff', 'g', 'bl', 'c1', 'neurons_sn', 'lam', 'cnn_preds','SNR_comp','r_values','coordinates']:
                if getattr(self, field) is not None:
                    if type(getattr(self, field)) is list:
                        setattr(self, field, np.array(getattr(self, field)))
                    if len(getattr(self, field)) == self.A.shape[-1]:
                        setattr(self, field, np.concatenate([getattr(self, field), getattr(self.discarded_components, field)], axis=0))
                        setattr(self.discarded_components, field, None)
                    else:
                        logging.warning('Variable ' + field + ' could not be \
                                        restored as it does not have the same \
                                        number of components as A')

            for field in ['A', 'A_thr']:
                print(field)
                if getattr(self, field) is not None:
                    if 'sparse' in str(type(getattr(self, field))):
                        setattr(self, field, scipy.sparse.hstack([getattr(self, field).tocsc(),getattr(self.discarded_components, field).tocsc()]))
                    else:
                        setattr(self, field,np.concatenate([getattr(self, field), getattr(self.discarded_components, field)], axis=0))

                    setattr(self.discarded_components, field, None)

            self.nr = self.A.shape[-1]

    def evaluate_components_CNN(self, params, neuron_class=1):
        """Estimates the quality of inferred spatial components using a
        pretrained CNN classifier.

        Args:
            params: params object
                see .params for details
            neuron_class: int
                class label for neuron shapes
        Returns:
            self: Estimates object
                self.idx_components contains the indeced of components above
                the required treshold.
        """
        dims = params.get('data', 'dims')
        gSig = params.get('init', 'gSig')
        min_cnn_thr = params.get('quality', 'min_cnn_thr')
        predictions = evaluate_components_CNN(self.A, dims, gSig)[0]
        self.cnn_preds = predictions[:, neuron_class]
        self.idx_components = np.where(self.cnn_preds >= min_cnn_thr)[0]
        return self

    def evaluate_components(self, imgs, params, dview=None):
        """Computes the quality metrics for each component and stores the
        indices of the components that pass user specified thresholds. The
        various thresholds and parameters can be passed as inputs. If left
        empty then they are read from self.params.quality']

        Args:
            imgs: np.array (possibly memory mapped, t,x,y[,z])
                Imaging data

            params: params object
                Parameters of the algorithm. The parameters in play here are
                contained in the subdictionary params.quality:

                min_SNR: float
                    trace SNR threshold

                rval_thr: float
                    space correlation threshold

                use_cnn: bool
                    flag for using the CNN classifier

                min_cnn_thr: float
                    CNN classifier threshold

        Returns:
            self: estimates object
                self.idx_components: np.array
                    indices of accepted components
                self.idx_components_bad: np.array
                    indices of rejected components
                self.SNR_comp: np.array
                    SNR values for each temporal trace
                self.r_values: np.array
                    space correlation values for each component
                self.cnn_preds: np.array
                    CNN classifier values for each component
        """
        dims = imgs.shape[1:]
        opts = params.get_group('quality')
        idx_components, idx_components_bad, SNR_comp, r_values, cnn_preds = \
            estimate_components_quality_auto(imgs, self.A, self.C, self.b, self.f, self.YrA,
                                             params.get('data', 'fr'),
                                             params.get('data', 'decay_time'),
                                             params.get('init', 'gSig'),
                                             dims, dview=dview,
                                             min_SNR=opts['min_SNR'],
                                             r_values_min=opts['rval_thr'],
                                             use_cnn=opts['use_cnn'],
                                             thresh_cnn_min=opts['min_cnn_thr'],
                                             thresh_cnn_lowest=opts['cnn_lowest'],
                                             r_values_lowest=opts['rval_lowest'],
                                             min_SNR_reject=opts['SNR_lowest'])
        self.idx_components = idx_components.astype(int)
        self.idx_components_bad = idx_components_bad.astype(int)
        if np.any(np.isnan(r_values)):
            logging.warning('NaN values detected for space correlation in {}'.format(np.where(np.isnan(r_values))[0]) +
                            '. Changing their value to -1.')
            r_values = np.where(np.isnan(r_values), -1, r_values)
        if np.any(np.isnan(SNR_comp)):
            logging.warning('NaN values detected for trace SNR in {}'.format(np.where(np.isnan(SNR_comp))[0]) +
                            '. Changing their value to 0.')
            SNR_comp = np.where(np.isnan(SNR_comp), 0, SNR_comp)
        self.SNR_comp = SNR_comp
        self.r_values = r_values
        self.cnn_preds = cnn_preds

        return self

    def filter_components(self, imgs, params, new_dict={}, dview=None, select_mode='All'):
        """Filters components based on given thresholds without re-computing
        the quality metrics. If the quality metrics are not present then it
        calls self.evaluate components.

        Args:
            imgs: np.array (possibly memory mapped, t,x,y[,z])
                Imaging data

            params: params object
                Parameters of the algorithm

            select_mode: str
                Can be 'All' (no subselection is made, but quality filtering is performed),
                'Accepted' (subselection of accepted components, a field named self.accepted_list must exist),
                'Rejected' (subselection of rejected components, a field named self.rejected_list must exist),
                'Unassigned' (both fields above need to exist)

            new_dict: dict
                New dictionary with parameters to be called. The dictionary
                modifies the params.quality subdictionary in the following
                entries:
                    min_SNR: float
                        trace SNR threshold

                    SNR_lowest: float
                        minimum required trace SNR

                    rval_thr: float
                        space correlation threshold

                    rval_lowest: float
                        minimum required space correlation

                    use_cnn: bool
                        flag for using the CNN classifier

                    min_cnn_thr: float
                        CNN classifier threshold

                    cnn_lowest: float
                        minimum required CNN threshold

                    gSig_range: list
                        gSig scale values for CNN classifier

        Returns:
            self: estimates object
                self.idx_components: np.array
                    indices of accepted components
                self.idx_components_bad: np.array
                    indices of rejected components
                self.SNR_comp: np.array
                    SNR values for each temporal trace
                self.r_values: np.array
                    space correlation values for each component
                self.cnn_preds: np.array
                    CNN classifier values for each component
        """
        dims = imgs.shape[1:]
        params.set('quality', new_dict)

        opts = params.get_group('quality')
        flag = [a is None for a in [self.r_values, self.SNR_comp, self.cnn_preds]]

        if any(flag):
            self.evaluate_components(imgs, params, dview=dview)
        else:
            self.idx_components, self.idx_components_bad, self.cnn_preds = \
            select_components_from_metrics(self.A, dims, params.get('init', 'gSig'),
                                           self.r_values, self.SNR_comp,
                                           predictions=self.cnn_preds,
                                           r_values_min=opts['rval_thr'],
                                           r_values_lowest=opts['rval_lowest'],
                                           min_SNR=opts['min_SNR'],
                                           min_SNR_reject=opts['SNR_lowest'],
                                           thresh_cnn_min=opts['min_cnn_thr'],
                                           thresh_cnn_lowest=opts['cnn_lowest'],
                                           use_cnn=opts['use_cnn'],
                                           gSig_range=opts['gSig_range'])

        if select_mode == 'Accepted':
           self.idx_components = np.array(np.intersect1d(self.idx_components,self.accepted_list))
        elif select_mode == 'Rejected':
           self.idx_components = np.array(np.intersect1d(self.idx_components,self.rejected_list))
        elif select_mode == 'Unassigned':
           self.idx_components = np.array(np.setdiff1d(self.idx_components,np.union1d(self.rejected_list,self.accepted_list)))

        self.idx_components_bad = np.array(np.setdiff1d(range(len(self.SNR_comp)),self.idx_components))

        return self

    def deconvolve(self, params, dview=None, dff_flag=False):
        ''' performs deconvolution on the estimated traces using the parameters
        specified in params. Deconvolution on detrended and normalized (DF/F)
        traces can be performed by setting dff_flag=True. In this case the
        results of the deconvolution are stored in F_dff_dec and S_dff

        Args:
            params: params object
                Parameters of the algorithm
            dff_flag: bool (True)
                Flag for deconvolving the DF/F traces

        Returns:
            self: estimates object
        '''

        F = self.C + self.YrA
        args = dict()
        args['p'] = params.get('preprocess', 'p')
        args['method_deconvolution'] = params.get('temporal', 'method_deconvolution')
        args['bas_nonneg'] = params.get('temporal', 'bas_nonneg')
        args['noise_method'] = params.get('temporal', 'noise_method')
        args['s_min'] = params.get('temporal', 's_min')
        args['optimize_g'] = params.get('temporal', 'optimize_g')
        args['noise_range'] = params.get('temporal', 'noise_range')
        args['fudge_factor'] = params.get('temporal', 'fudge_factor')

        args_in = [(F[jj], None, jj, None, None, None, None,
                    args) for jj in range(F.shape[0])]

        if 'multiprocessing' in str(type(dview)):
            results = dview.map_async(
                constrained_foopsi_parallel, args_in).get(4294967)
        elif dview is not None:
            results = dview.map_sync(constrained_foopsi_parallel, args_in)
        else:
            results = list(map(constrained_foopsi_parallel, args_in))

        results = list(zip(*results))

        order = list(results[7])
        self.C = np.stack([results[0][i] for i in order])
        self.S = np.stack([results[1][i] for i in order])
        self.bl = [results[3][i] for i in order]
        self.c1 = [results[4][i] for i in order]
        self.g = [results[6][i] for i in order]
        self.neurons_sn = [results[5][i] for i in order]
        self.lam = [results[8][i] for i in order]
        self.YrA = F - self.C

        if dff_flag:
            if self.F_dff is None:
                logging.warning('The F_dff field is empty. Run the method' +
                                ' estimates.detrend_df_f before attempting' +
                                ' to deconvolve.')
            else:
                args_in = [(self.F_dff[jj], None, jj, 0, 0, self.g[jj], None,
                        args) for jj in range(F.shape[0])]

                if 'multiprocessing' in str(type(dview)):
                    results = dview.map_async(
                        constrained_foopsi_parallel, args_in).get(4294967)
                elif dview is not None:
                    results = dview.map_sync(constrained_foopsi_parallel,
                                             args_in)
                else:
                    results = list(map(constrained_foopsi_parallel, args_in))

                results = list(zip(*results))
                order = list(results[7])
                self.F_dff_dec = np.stack([results[0][i] for i in order])
                self.S_dff = np.stack([results[1][i] for i in order])


    def manual_merge(self, components, params):
        ''' merge a given list of components. The indices
        of components are not pythonic, i.e., they start from 1. Moreover,
        the indices refer to the absolute indices, i.e., the indices before
        spliting the components in accepted and rejected. If you want to e.g.
        merge components 1 from idx_components and 10 from idx_components_bad
        you will to set
        ```
        components = [[self.idx_components[0], self.idx_components_bad[9]]]
        ```

        Args:
            components: list
                list of components to be merged. Each element should be a
                tuple, list or np.array of the components to be merged. No
                duplicates are allowed. If you're merging only one pair (or
                set) of components, then use a list with a single (list)
                element
            params: params object

        Returns:
            self: estimates object
        '''

        ln = np.sum(np.array([len(comp) for comp in components]))
        ids = set.union(*[set(comp) for comp in components])
        if ln != len(ids):
            raise Exception('The given list contains duplicate entries')

        p = params.temporal['p']
        nbmrg = len(components)   # number of merging operations
        d = self.A.shape[0]
        T = self.C.shape[1]
        # we initialize the values
        A_merged = scipy.sparse.lil_matrix((d, nbmrg))
        C_merged = np.zeros((nbmrg, T))
        R_merged = np.zeros((nbmrg, T))
        S_merged = np.zeros((nbmrg, T))
        bl_merged = np.zeros((nbmrg, 1))
        c1_merged = np.zeros((nbmrg, 1))
        sn_merged = np.zeros((nbmrg, 1))
        g_merged = np.zeros((nbmrg, p))
        merged_ROIs = []

        for i in range(nbmrg):
            merged_ROI = list(set(components[i]))
            logging.info('Merging components {}'.format(merged_ROI))
            merged_ROIs.append(merged_ROI)

            Acsc = self.A.tocsc()[:, merged_ROI]
            Ctmp = np.array(self.C[merged_ROI]) + np.array(self.YrA[merged_ROI])

            C_to_norm = np.sqrt(np.ravel(Acsc.power(2).sum(
                axis=0)) * np.sum(Ctmp ** 2, axis=1))
            indx = np.argmax(C_to_norm)
            g_idx = [merged_ROI[indx]]
            fast_merge = True
            bm, cm, computedA, computedC, gm, \
            sm, ss, yra = merge_iteration(Acsc, C_to_norm, Ctmp, fast_merge,
                                          None, g_idx, indx, params.temporal)

            A_merged[:, i] = computedA[:, np.newaxis]
            C_merged[i, :] = computedC
            R_merged[i, :] = yra
            S_merged[i, :] = ss[:T]
            bl_merged[i] = bm
            c1_merged[i] = cm
            sn_merged[i] = sm
            g_merged[i, :] = gm

        empty = np.ravel((C_merged.sum(1) == 0) + (A_merged.sum(0) == 0))
        nbmrg -= len(empty)
        if np.any(empty):
            A_merged = A_merged[:, ~empty]
            C_merged = C_merged[~empty]
            R_merged = R_merged[~empty]
            S_merged = S_merged[~empty]
            bl_merged = bl_merged[~empty]
            c1_merged = c1_merged[~empty]
            sn_merged = sn_merged[~empty]
            g_merged = g_merged[~empty]

        neur_id = np.unique(np.hstack(merged_ROIs))
        nr = self.C.shape[0]
        good_neurons = np.setdiff1d(list(range(nr)), neur_id)
        if self.idx_components is not None:
            new_indices = list(range(len(good_neurons),
                                     len(good_neurons) + nbmrg))

            mapping_mat = np.zeros(nr)
            mapping_mat[good_neurons] = np.arange(len(good_neurons), dtype=int)
            gn_ = good_neurons.tolist()
            new_idx = [mapping_mat[i] for i in self.idx_components if i in gn_]
            new_idx_bad = [mapping_mat[i] for i in self.idx_components_bad if i in gn_]
            new_idx.sort()
            new_idx_bad.sort()
            self.idx_components = np.array(new_idx + new_indices, dtype=int)
            self.idx_components_bad = np.array(new_idx_bad, dtype=int)

        self.A = scipy.sparse.hstack((self.A.tocsc()[:, good_neurons],
                                      A_merged.tocsc()))
        self.C = np.vstack((self.C[good_neurons, :], C_merged))
        # we continue for the variables
        if self.YrA is not None:
            self.YrA = np.vstack((self.YrA[good_neurons, :], R_merged))
            self.R = self.YrA
        if self.S is not None:
            self.S = np.vstack((self.S[good_neurons, :], S_merged))
        if self.bl is not None:
            self.bl = np.hstack((self.bl[good_neurons],
                                 np.array(bl_merged).flatten()))
        if self.c1 is not None:
            self.c1 = np.hstack((self.c1[good_neurons],
                                 np.array(c1_merged).flatten()))
        if self.sn is not None:
            self.sn = np.hstack((self.sn[good_neurons],
                                 np.array(sn_merged).flatten()))
        if self.g is not None:
            self.g = np.vstack((np.vstack(self.g)[good_neurons], g_merged))
        self.nr = nr - len(neur_id) + len(C_merged)
        if self.coordinates is not None:
            self.coordinates = caiman.utils.visualization.get_contours(self.A,\
                                self.dims, thr_method='max', thr='0.2')

    def threshold_spatial_components(self, maxthr=0.25, dview=None):
        ''' threshold spatial components. See parameters of
        spatial.threshold_components

        @param medw:
        @param thr_method:
        @param maxthr:
        @param extract_cc:
        @param se:
        @param ss:
        @param dview:
        @return:
        '''

        if self.A_thr is None:
            A_thr = threshold_components(self.A, self.dims,  maxthr=maxthr, dview=dview,
                                         medw=None, thr_method='max', nrgthr=0.99,
                                         extract_cc=True, se=None, ss=None)

            self.A_thr = A_thr
        else:
            print('A_thr already computed. If you want to recompute set self.A_thr to None')

    def remove_small_large_neurons(self, min_size_neuro, max_size_neuro,
                                   select_comp=False):
        ''' remove neurons that are too large or too small

    	Args:
            min_size_neuro: int
                min size in pixels
            max_size_neuro: int
                max size in pixels
            select_comp: bool
                remove components that are too small/large from main estimates
                fields. See estimates.selecte_components() for more details.

        Returns:
            neurons_to_keep: np.array
                indeces of components with size within the acceptable range
        '''
        if self.A_thr is None:
            raise Exception('You need to compute thresolded components before calling remove_duplicates: use the threshold_components method')

        A_gt_thr_bin = self.A_thr.toarray() > 0
        size_neurons_gt = A_gt_thr_bin.sum(0)
        neurons_to_keep = np.where((size_neurons_gt > min_size_neuro) & (size_neurons_gt < max_size_neuro))[0]
#        self.select_components(idx_components=neurons_to_keep)
        if self.idx_components is None:
            self.idx_components = np.arange(self.A.shape[-1])
        self.idx_components = np.intersect1d(self.idx_components, neurons_to_keep)
        self.idx_components_bad = np.setdiff1d(np.arange(self.A.shape[-1]), self.idx_components)
        if select_comp:
            self.select_components(use_object=True)
        return neurons_to_keep



    def remove_duplicates(self, predictions=None, r_values=None, dist_thr=0.1,
                          min_dist=10, thresh_subset=0.6, plot_duplicates=False,
                          select_comp=False):
        ''' remove neurons that heavily overlap and might be duplicates. 

        Args:
            predictions
            r_values
            dist_thr
            min_dist
            thresh_subset
            plot_duplicates
        '''
        if self.A_thr is None:
            raise Exception('You need to compute thresolded components before calling remove_duplicates: use the threshold_components method')

        A_gt_thr_bin = (self.A_thr.toarray() > 0).reshape([self.dims[0], self.dims[1], -1], order='F').transpose([2, 0, 1]) * 1.

        duplicates_gt, indices_keep_gt, indices_remove_gt, D_gt, overlap_gt = detect_duplicates_and_subsets(
            A_gt_thr_bin,predictions=predictions, r_values=r_values, dist_thr=dist_thr, min_dist=min_dist,
            thresh_subset=thresh_subset)
        logging.info('Number of duplicates: {}'.format(len(duplicates_gt)))
        if len(duplicates_gt) > 0:
            if plot_duplicates:
                plt.figure()
                plt.subplot(1, 3, 1)
                plt.imshow(A_gt_thr_bin[np.array(duplicates_gt).flatten()].sum(0))
                plt.colorbar()
                plt.subplot(1, 3, 2)
                plt.imshow(A_gt_thr_bin[np.array(indices_keep_gt)[:]].sum(0))
                plt.colorbar()
                plt.subplot(1, 3, 3)
                plt.imshow(A_gt_thr_bin[np.array(indices_remove_gt)[:]].sum(0))
                plt.colorbar()
                plt.pause(1)

            components_to_keep = np.delete(np.arange(self.A.shape[-1]), indices_remove_gt)

        else:
            components_to_keep = np.arange(self.A.shape[-1])

        if self.idx_components is None:
            self.idx_components = np.arange(self.A.shape[-1])
        self.idx_components = np.intersect1d(self.idx_components, components_to_keep)
        self.idx_components_bad = np.setdiff1d(np.arange(self.A.shape[-1]), self.idx_components)
        if select_comp:
            self.select_components(use_object=True)

        return components_to_keep

    def masks_2_neurofinder(self, dataset_name):
        """Return masks to neurofinder format
        """
        if self.A_thr is None:
            raise Exception(
                'You need to compute thresolded components before calling this method: use the threshold_components method')
        bin_masks = self.A_thr.reshape([self.dims[0], self.dims[1], -1], order='F').transpose([2, 0, 1])
        return nf_masks_to_neurof_dict(bin_masks, dataset_name)

    def save_NWB(self,
                 filename,
                 imaging_plane_name=None,
                 imaging_series_name=None,
                 sess_desc='CaImAn Results',
                 exp_desc=None,
                 identifier=None,
                 imaging_rate=30.,
                 starting_time=0.,
                 session_start_time=None,
                 excitation_lambda=488.0,
                 imaging_plane_description='some imaging plane description',
                 emission_lambda=520.0,
                 indicator='OGB-1',
                 location='brain',
                 raw_data_file=None):
        """writes NWB file

        Args:
            filename: str

            imaging_plane_name: str, optional

            imaging_series_name: str, optional

            sess_desc: str, optional

            exp_desc: str, optional

            identifier: str, optional

            imaging_rate: float, optional
                default: 30 (Hz)

            starting_time: float, optional
                default: 0.0 (seconds)

            location: str, optional

            session_start_time: datetime.datetime, optional
                Only required for new files

            excitation_lambda: float

            imaging_plane_description: str

            emission_lambda: float

            indicator: str

            location: str
        """

        from pynwb import NWBHDF5IO, TimeSeries, NWBFile
        from pynwb.base import Images
        from pynwb.image import GrayscaleImage
        from pynwb.ophys import ImageSegmentation, Fluorescence, OpticalChannel, ImageSeries
        from pynwb.device import Device
        import os

        if identifier is None:
            import uuid
            identifier = uuid.uuid1().hex

        if '.nwb' != os.path.splitext(filename)[-1].lower():
            raise Exception("Wrong filename")

        if not os.path.isfile(filename):  # if the file doesn't exist create new and add the original data path
            print('filename does not exist. Creating new NWB file with only estimates output')

            nwbfile = NWBFile(sess_desc, identifier, session_start_time, experiment_description=exp_desc)
            device = Device('imaging_device')
            nwbfile.add_device(device)
            optical_channel = OpticalChannel('OpticalChannel',
                                             'main optical channel',
                                             emission_lambda=emission_lambda)
            nwbfile.create_imaging_plane(name='ImagingPlane',
                                         optical_channel=optical_channel,
                                         description=imaging_plane_description,
                                         device=device,
                                         excitation_lambda=excitation_lambda,
                                         imaging_rate=imaging_rate,
                                         indicator=indicator,
                                         location=location)
            if raw_data_file:
                nwbfile.add_acquisition(ImageSeries(name='TwoPhotonSeries',
                                                    external_file=[raw_data_file],
                                                    format='external',
                                                    rate=imaging_rate,
                                                    starting_frame=[0]))
            with NWBHDF5IO(filename, 'w') as io:
                io.write(nwbfile)

        time.sleep(4)  # ensure the file is fully closed before opening again in append mode
        logging.info('Saving the results in the NWB file...')

        with NWBHDF5IO(filename, 'r+') as io:
            nwbfile = io.read()
            # Add processing results

            # Create the module as 'ophys' unless it is taken and append 'ophysX' instead
            ophysmodules = [s[5:] for s in list(nwbfile.modules) if s.startswith('ophys')]
            if any('' in s for s in ophysmodules):
                if any([s for s in ophysmodules if s.isdigit()]):
                    nummodules = max([int(s) for s in ophysmodules if s.isdigit()])+1
                    print('ophys module previously created, writing to ophys'+str(nummodules)+' instead')
                    mod = nwbfile.create_processing_module('ophys'+str(nummodules), 'contains caiman estimates for '
                                                                                    'the main imaging plane')
                else:
                    print('ophys module previously created, writing to ophys1 instead')
                    mod = nwbfile.create_processing_module('ophys1', 'contains caiman estimates for the main '
                                                                     'imaging plane')
            else:
                mod = nwbfile.create_processing_module('ophys', 'contains caiman estimates for the main imaging plane')

            img_seg = ImageSegmentation()
            mod.add_data_interface(img_seg)
            fl = Fluorescence()
            mod.add_data_interface(fl)
#            mot_crct = MotionCorrection()
#            mod.add_data_interface(mot_crct)

            # Add the ROI-related stuff
            if imaging_plane_name is not None:
                imaging_plane = nwbfile.imaging_planes[imaging_plane_name]
            else:
                if len(nwbfile.imaging_planes) == 1:
                    imaging_plane = list(nwbfile.imaging_planes.values())[0]
                else:
                    raise Exception('There is more than one imaging plane in the file, you need to specify the name'
                                    ' via the "imaging_plane_name" parameter')

            if imaging_series_name is not None:
                image_series = nwbfile.acquisition[imaging_series_name]
            else:
                if not len(nwbfile.acquisition):
                    image_series = None
                elif len(nwbfile.acquisition) == 1:
                    image_series = list(nwbfile.acquisition.values())[0]
                else:
                    raise Exception('There is more than one imaging plane in the file, you need to specify the name'
                                    ' via the "imaging_series_name" parameter')

            ps = img_seg.create_plane_segmentation('CNMF_ROIs', imaging_plane, 'PlaneSegmentation', image_series)

            ps.add_column('r', 'description of r values')
            ps.add_column('snr', 'signal to noise ratio')
            ps.add_column('cnn', 'description of CNN')
            ps.add_column('keep', 'in idx_components')
            ps.add_column('accepted', 'in accepted list')
            ps.add_column('rejected', 'in rejected list')

            # Add ROIs
            if not hasattr(self, 'accepted_list'):
                for i, (roi, snr, r, cnn) in enumerate(zip(self.A.T, self.SNR_comp, self.r_values, self.cnn_preds)):
                    ps.add_roi(image_mask=roi.T.toarray().reshape(self.dims), r=r, snr=snr, cnn=cnn,
                               keep=i in self.idx_components, accepted=False, rejected=False)
            else:
                for i, (roi, snr, r, cnn) in enumerate(zip(self.A.T, self.SNR_comp, self.r_values, self.cnn_preds)):
                    ps.add_roi(image_mask=roi.T.toarray().reshape(self.dims), r=r, snr=snr, cnn=cnn,
                               keep=i in self.idx_components, accepted=i in self.accepted_list, rejected=i in self.rejected_list)
            
            for bg in self.b.T:  # Backgrounds
                ps.add_roi(image_mask=bg.reshape(self.dims), r=np.nan, snr=np.nan, cnn=np.nan, keep=False, accepted=False, rejected=False)
            # Add Traces
            n_rois = self.A.shape[-1]
            n_bg = len(self.f)
            rt_region_roi = ps.create_roi_table_region(
                'ROIs', region=list(range(n_rois)))

            rt_region_bg = ps.create_roi_table_region(
                'Background', region=list(range(n_rois, n_rois+n_bg)))

            timestamps = np.arange(self.f.shape[1]) / imaging_rate + starting_time

            # Neurons
            fl.create_roi_response_series(name='RoiResponseSeries', data=self.C.T, rois=rt_region_roi, unit='lumens', timestamps=timestamps)
            # Background
            fl.create_roi_response_series(name='Background_Fluorescence_Response', data=self.f.T, rois=rt_region_bg, unit='lumens', 
                                          timestamps=timestamps)

            mod.add(TimeSeries(name='residuals', description='residuals', data=self.YrA.T, timestamps=timestamps,
                               unit='NA'))
            if hasattr(self, 'Cn'):
                images = Images('summary_images')
                images.add_image(GrayscaleImage(name='local_correlations', data=self.Cn))

                # Add MotionCorreciton
    #            create_corrected_image_stack(corrected, original, xy_translation, name='CorrectedImageStack')
                io.write(nwbfile)


def compare_components(estimate_gt, estimate_cmp,  Cn=None, thresh_cost=.8, min_dist=10, print_assignment=False,
                       labels=['GT', 'CMP'], plot_results=False):
    if estimate_gt.A_thr is None:
        raise Exception(
            'You need to compute thresolded components for first argument before calling remove_duplicates: use the threshold_components method')
    if estimate_cmp.A_thr is None:
        raise Exception(
            'You need to compute thresolded components for second argument before calling remove_duplicates: use the threshold_components method')

    if plot_results:
        plt.figure(figsize=(20, 10))

    dims = estimate_gt.dims
    A_gt_thr_bin = (estimate_gt.A_thr.toarray()>0).reshape([dims[0], dims[1], -1], order='F').transpose([2, 0, 1]) * 1.
    A_thr_bin = (estimate_cmp.A_thr.toarray()>0).reshape([dims[0], dims[1], -1], order='F').transpose([2, 0, 1]) * 1.

    tp_gt, tp_comp, fn_gt, fp_comp, performance_cons_off = nf_match_neurons_in_binary_masks(
        A_gt_thr_bin, A_thr_bin, thresh_cost=thresh_cost, min_dist=min_dist, print_assignment=print_assignment,
        plot_results=plot_results, Cn=Cn, labels=labels)

    return tp_gt, tp_comp, fn_gt, fp_comp, performance_cons_off
