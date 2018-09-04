#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 11:11:45 2018

@author: epnevmatikakis
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
import caiman
import logging
from .utilities import detrend_df_f
from .spatial import threshold_components
from ...components_evaluation import (
        evaluate_components_CNN, estimate_components_quality_auto,
        select_components_from_metrics)
from ...base.rois import detect_duplicates_and_subsets, nf_match_neurons_in_binary_masks, nf_masks_to_neurof_dict
from .initialization import downscale


class Estimates(object):
    def __init__(self, A=None, b=None, C=None, f=None, R=None, dims=None):
        """Class for storing the variables related to the estimates of spatial footprints, temporal traces,
        deconvolved neural activity, and background. Quality metrics are also stored. The class has methods
        for evaluating the quality of each component, DF/F normalization and some basic plotting.

        Parameters/Attributes
        ---------------------
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
            indeces of accepted components

        idx_components_bad: list
            indeces of rejected components

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
        self.shifts = []

        self.A_thr = None



    def plot_contours(self, img=None, idx=None, crd=None, thr_method='max',
                      thr='0.2'):
        """view contours of all spatial footprints.
        Parameters:
        -----------
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
        """
        if 'csc_matrix' not in str(type(self.A)):
            self.A = scipy.sparse.csc_matrix(self.A)
        if img is None:
            img = np.reshape(np.array(self.A.mean(1)), self.dims, order='F')
        if self.coordinates is None:  # not hasattr(self, 'coordinates'):
            self.coordinates = caiman.utils.visualization.get_contours(self.A, self.dims, thr=thr, thr_method=thr_method)
        plt.figure()
        if idx is None:
            caiman.utils.visualization.plot_contours(self.A, img, coordinates=self.coordinates)
        else:
            if not isinstance(idx, list):
                idx = idx.tolist()
            coor_g = [self.coordinates[cr] for cr in idx]
            bad = list(set(range(self.A.shape[1])) - set(idx))
            coor_b = [self.coordinates[cr] for cr in bad]
            plt.subplot(1, 2, 1)
            caiman.utils.visualization.plot_contours(self.A[:, idx], img,
                                                     coordinates=coor_g)
            plt.title('Accepted Components')
            bad = list(set(range(self.A.shape[1])) - set(idx))
            plt.subplot(1, 2, 2)
            caiman.utils.visualization.plot_contours(self.A[:, bad], img,
                                                     coordinates=coor_b)
            plt.title('Rejected Components')
        return self

    def plot_contours_nb(self, img=None, idx=None, crd=None, thr_method='max',
                         thr='0.2'):
        """view contours od all spatial footprints (notebook environment).
        Parameters:
        -----------
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
        """
        if 'csc_matrix' not in str(type(self.A)):
            self.A = scipy.sparse.csc_matrix(self.A)
        if img is None:
            img = np.reshape(np.array(self.A.mean(1)), self.dims, order='F')
        if self.coordinates is None:  # not hasattr(self, 'coordinates'):
            self.coordinates = caiman.utils.visualization.get_contours(self.A,
                                    self.dims, thr=thr, thr_method=thr_method)
        if idx is None:
            caiman.utils.visualization.nb_plot_contour(img, self.A, self.dims[0],
                            self.dims[1], coordinates=self.coordinates,
                            thr_method=thr_method, thr=thr)
        else:
            if not isinstance(idx, list):
                idx = idx.tolist()
            coor_g = [self.coordinates[cr] for cr in idx]
            bad = list(set(range(self.A.shape[1])) - set(idx))
            coor_b = [self.coordinates[cr] for cr in bad]
            caiman.utils.visualization.nb_plot_contour(img, self.A[:, idx],
                            self.dims[0], self.dims[1], coordinates=coor_g,
                            thr_method=thr_method, thr=thr)
            bad = list(set(range(self.A.shape[1])) - set(idx))
            caiman.utils.visualization.nb_plot_contour(img, self.A[:, bad],
                            self.dims[0], self.dims[1], coordinates=coor_b,
                            thr_method=thr_method, thr=thr)
            plt.title('Rejected Components')
        return self

    def view_components(self, Yr=None, img=None, idx=None):
        """view spatial and temporal components interactively

        Parameters:
        -----------
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
        if self.R.shape != [nr, T]:
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

        Parameters:
        -----------
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

    def nb_view_components_3d(self, Yr=None, image_type='mean', dims=None,
                              max_projection=False, axis=0,
                              denoised_color=None, cmap='jet', thr=0.9):
        """view spatial and temporal components interactively in a notebook
        (version for 3d data)

        Parameters:
        -----------
        Yr :    np.ndarray
                movie in format pixels (d) x frames (T) (only required to
                compute the correlation image)

        img :   np.ndarray
                background image for contour plotting. Default is the mean
                image of all spatial components (d1 x d2)

        idx :   list
                list of components to be plotted

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

    def play_movie(self, imgs, q_max=99.75, q_min=2, gain_res=1,
                   magnification=1, include_bck=True,
                   frame_range=slice(None, None, None),
                   bpx=0):

        """Displays a movie with three panels (original data (left panel),
        reconstructed data (middle panel), residual (right panel))
        Parameters:
        -----------
        imgs: np.array (possibly memory mapped, t,x,y[,z])
            Imaging data

        q_max: float (values in [0, 100])
            percentile for maximum plotting value

        q_min: float (values in [0, 100])
            percentile for minimum plotting value

        gain_res: float
            amplification factor for residual movie

        magnification: float
            magnification factor for whole movie

        include_bck: bool
            flag for including background in original and reconstructed movie

        frame_rage: range or slice or list
            display only a subset of frames

        bpx: int
            number of pixels to exclude on each border

        Returns:
        --------
        self (to stop the movie press 'q')
        """
        dims = imgs.shape[1:]
        if 'movie' not in str(type(imgs)):
            imgs = caiman.movie(imgs)
        Y_rec = self.A.dot(self.C[:, frame_range])
        Y_rec = Y_rec.reshape(dims + (-1,), order='F')
        Y_rec = Y_rec.transpose([2, 0, 1])
        if self.b is not None and self.f is not None:
            B = self.b.dot(self.f[:, frame_range])
            if 'matrix' in str(type(B)):
                B = B.toarray()
            B = B.reshape(dims + (-1,), order='F').transpose([2, 0, 1])
        elif self.W is not None:
            ssub_B = int(round(np.sqrt(np.prod(dims) / self.W.shape[0])))
            B = imgs[frame_range].reshape((-1, np.prod(dims)), order='F').T - \
                self.A.dot(self.C[:, frame_range])
            if ssub_B==1:
                B = self.b0[:, None] + self.W.dot(B - self.b0[:, None])
            else:
                B = self.b0[:, None] + (np.repeat(np.repeat(self.W.dot(
                    downscale(B.reshape(dims + (B.shape[-1],), order='F'),
                              (ssub_B, ssub_B, 1)).reshape((-1, B.shape[-1]), order='F') -
                    downscale(self.b0.reshape(dims, order='F'),
                              (ssub_B, ssub_B)).reshape((-1, 1), order='F'))
                    .reshape(((dims[0] - 1) // ssub_B + 1, (dims[1] - 1) // ssub_B + 1, -1), order='F'),
                    ssub_B, 0), ssub_B, 1)[:dims[0], :dims[1]].reshape((-1, B.shape[-1]), order='F'))
            B = B.reshape(dims + (-1,), order='F').transpose([2, 0, 1])
        else:
            B = np.zeros_like(Y_rec)
        if bpx > 0:
            B = B[:, bpx:-bpx, bpx:-bpx]
            Y_rec = Y_rec[:, bpx:-bpx, bpx:-bpx]
            imgs = imgs[:, bpx:-bpx, bpx:-bpx]

        Y_res = imgs[frame_range] - Y_rec - B

        caiman.concatenate((imgs[frame_range] - (not include_bck)*B, Y_rec + include_bck*B, Y_res*gain_res), axis=2).play(q_min=q_min, q_max=q_max, magnification=magnification)

        return self

    def compute_residuals(self, Yr):
        """compute residual for each component (variable R)

         Parameters:
         -----------
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
        nA2 = np.ravel(Ab.power(2).sum(axis=0))
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
                     flag_auto=True, use_fast=False, use_residuals=True):
        """Computes DF/F normalized fluorescence for the extracted traces. See
        caiman.source.extraction.utilities.detrend_df_f for details

        Parameters:
        -----------
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

        Returns:
        --------
        self: CNMF object
            self.F_dff contains the DF/F normalized traces
        """

        if self.C is None:
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

        self.F_dff = detrend_df_f(self.A, self.b, self.C, self.f, R,
                                  quantileMin=quantileMin,
                                  frames_window=frames_window,
                                  flag_auto=flag_auto, use_fast=use_fast)
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
        nA_inv_mat = scipy.sparse.spdiags(1. / nA, 0, nA.shape[0], nA.shape[0])
        self.A = self.A * nA_inv_mat
        self.C = nA_mat * self.C
        if self.YrA is not None:
            self.YrA = nA_mat * self.YrA
        if self.R is not None:
            self.R = nA_mat * self.R
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
            nB_inv_mat = scipy.sparse.spdiags(1. / nB, 0, nB.shape[0], nB.shape[0])
            self.b = self.b * nB_inv_mat
            self.f = nB_mat * self.f
        return self

    def select_components(self, idx_components=None, use_object=False):
        """Keeps only a selected subset of components and removes the rest.
        The subset can be either user defined with the variable idx_components
        or read from the estimates object. The flag use_object determines this
        choice. If no subset is present then all components are kept.
        Parameters:
        -----------
        idx_components: list
            indeces of components to be kept

        use_object: bool
            Flag to use self.idx_components for reading the indeces.

        Returns:
        --------
        self: Estimates object
        """
        if use_object:
            idx_components = self.idx_components
        if idx_components is None:
            idx_components = range(self.A.shape[-1])
        import pdb

        for field in ['C', 'S', 'YrA', 'R', 'g', 'bl', 'c1', 'neurons_sn', 'lam', 'cnn_preds']:
            print(field)
            if getattr(self, field) is not None:
                if type(getattr(self, field)) is list:
                    setattr(self, field, np.array(getattr(self, field)))
                if len(getattr(self, field)) == self.A.shape[-1]:
                    setattr(self, field, getattr(self, field)[idx_components])
                else:
                    print('*** Variable ' + field + ' has not the same number of components as A ***')

        for field in ['A', 'A_thr']:
            print(field)
            if getattr(self, field) is not None:
                if 'sparse' in str(type(getattr(self, field))):
                    setattr(self, field, getattr(self, field).tocsc()[:, idx_components])
                else:
                    setattr(self, field, getattr(self, field)[:, idx_components])


        # self.A = self.A.tocsc()[:, idx_components]
        # self.C = self.C[idx_components]
        # self.S = self.S[idx_components]
        # self.YrA = self.YrA[idx_components]
        # self.R = self.YrA
        # self.g = self.g[idx_components]
        # self.bl = self.bl[idx_components]
        # self.c1 = self.c1[idx_components]
        # self.neurons_sn = self.neurons_sn[idx_components]
        # self.lam = self.lam[idx_components]
        #
        # if self.A_thr is not None:
        #     self.A_thr = self.A_thr.tocsc()[:, idx_components]
        self.idx_components = None
        self.idx_components_bad = None
        return self

    def evaluate_components_CNN(self, params, neuron_class=1):
        """Estimates the quality of inferred spatial components using a
        pretrained CNN classifier.
        Parameters:
        -----------
        params: params object
            see .params for details
        neuron_class: int
            class label for neuron shapes
        Returns:
        ----------
        self: Estimates object
            self.idx_components contains the indeced of components above
            the required treshold.
        """
        dims = params.get('data', 'dims')
        gSig = params.get('init', 'gSig')
        min_cnn_thr= params.get('quality', 'min_cnn_thr')
        predictions = evaluate_components_CNN(self.A, dims, gSig)[0]
        self.cnn_preds = predictions[:, neuron_class]
        self.idx_components = np.where(self.cnn_preds >= min_cnn_thr)[0]
        return self
    
    def evaluate_components(self, imgs, params, dview=None):
        """Computes the quality metrics for each component and stores the
        indeces of the components that pass user specified thresholds. The
        various thresholds and parameters can be passed as inputs. If left
        empty then they are read from self.params.quality']
        Parameters:
        -----------
        imgs: np.array (possibly memory mapped, t,x,y[,z])
            Imaging data

        fr: float
            Imaging rate

        decay_time: float
            length of decay of typical transient (in seconds)

        min_SNR: float
            trace SNR threshold

        rval_thr: float
            space correlation threshold

        use_cnn: bool
            flag for using the CNN classifier

        min_cnn_thr: float
            CNN classifier threshold

        Returns:
        --------
        self: CNMF object
            self.idx_components: np.array
                indeces of accepted components
            self.idx_components_bad: np.array
                indeces of rejected components
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
        estimate_components_quality_auto(imgs, self.A, self.C, self.b, self.f,
                                         self.YrA, 
                                         params.get('data', 'fr'),
                                         params.get('data', 'decay_time'),
                                         params.get('init', 'gSig'),
                                         dims, dview=dview,
                                         min_SNR=opts['min_SNR'],
                                         r_values_min=opts['rval_thr'],
                                         use_cnn=opts['use_cnn'],
                                         thresh_cnn_min=opts['min_cnn_thr'])
        self.idx_components = idx_components
        self.idx_components_bad = idx_components_bad
        self.SNR_comp = SNR_comp
        self.r_values = r_values
        self.cnn_preds = cnn_preds

        return self


    def filter_components(self, imgs, **kwargs):
        """Filters components based on given thresholds without re-computing
        the quality metrics. If the quality metrics are not present then it
        calls self.evaluate components.
        Parameters:
        -----------
        imgs: np.array (possibly memory mapped, t,x,y[,z])
            Imaging data

        fr: float
            Imaging rate

        decay_time: float
            length of decay of typical transient (in seconds)

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
        --------
        self: CNMF object
            self.idx_components: np.array
                indeces of accepted components
            self.idx_components_bad: np.array
                indeces of rejected components
            self.SNR_comp: np.array
                SNR values for each temporal trace
            self.r_values: np.array
                space correlation values for each component
            self.cnn_preds: np.array
                CNN classifier values for each component
        """
        dims = imgs.shape[1:]
        self.params.set('quality', kwargs)

        opts = self.params.get_group('quality')
        self.idx_components, self.idx_components_bad, self.cnn_preds = \
        select_components_from_metrics(self.A, dims, self.params.get('init', 'gSig'), self.r_values,
                                       self.SNR_comp, predictions=self.cnn_preds,
                                       r_values_min=opts['rval_thr'],
                                       r_values_lowest=opts['rval_lowest'],
                                       min_SNR=opts['min_SNR'],
                                       min_SNR_reject=opts['SNR_lowest'],
                                       thresh_cnn_min=opts['min_cnn_thr'],
                                       thresh_cnn_lowest=opts['cnn_lowest'],
                                       use_cnn=opts['use_cnn'],
                                       gSig_range=opts['gSig_range'])

        return self

    def threshold_spatial_components(self, maxthr=0.25, dview=None, **kwargs):
        ''' threshold spatial components. See parameters of spatial.threshold_components

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

            A_thr = threshold_components(self.A, self.dims,  maxthr=maxthr, dview=dview, medw=None, thr_method='max', nrgthr=0.99,
                                         extract_cc=True, se=None, ss=None, **kwargs)

            self.A_thr = A_thr
        else:
            print('A_thr already computed. If you want to recompute set self.A_thr to None')



    def remove_small_large_neurons(self, min_size_neuro, max_size_neuro):
        ''' remove neurons that are too large or too smal

        @param min_size_neuro: min size in pixels
        @param max_size_neuro: max size inpixels
        @return:
        '''
        if self.A_thr is None:
            raise Exception('You need to compute thresolded components before calling remove_duplicates: use the threshold_components method')

        A_gt_thr_bin = self.A_thr > 0
        size_neurons_gt = A_gt_thr_bin.sum(0)
        neurons_to_keep = np.where((size_neurons_gt > min_size_neuro) & (size_neurons_gt < max_size_neuro))[0]
        self.select_components(idx_components=neurons_to_keep)


    def remove_duplicates(self, predictions=None, r_values=None, dist_thr=0.1, min_dist=10, thresh_subset=0.6, plot_duplicates=False):
        ''' remove neurons that heavily overlapand might be duplicates

        @param predictions:
        @param r_values:
        @param dist_thr:
        @param min_dist:
        @param thresh_subset:
        @param plot_duplicates:
        @return:
        '''
        if self.A_thr is None:
            raise Exception('You need to compute thresolded components before calling remove_duplicates: use the threshold_components method')

        A_gt_thr_bin = (self.A_thr > 0).reshape([self.dims[0], self.dims[1], -1], order='F').transpose([2, 0, 1]) * 1.

        duplicates_gt, indeces_keep_gt, indeces_remove_gt, D_gt, overlap_gt = detect_duplicates_and_subsets(
            A_gt_thr_bin,predictions=predictions, r_values=r_values,dist_thr=dist_thr, min_dist=min_dist,
            thresh_subset=thresh_subset)

        if len(duplicates_gt) > 0:
            if plot_duplicates:
                plt.figure()
                plt.subplot(1, 3, 1)
                plt.imshow(A_gt_thr_bin[np.array(duplicates_gt).flatten()].sum(0))
                plt.colorbar()
                plt.subplot(1, 3, 2)
                plt.imshow(A_gt_thr_bin[np.array(indeces_keep_gt)[:]].sum(0))
                plt.colorbar()
                plt.subplot(1, 3, 3)
                plt.imshow(A_gt_thr_bin[np.array(indeces_remove_gt)[:]].sum(0))
                plt.colorbar()
                plt.pause(1)
            components_to_keep = np.delete(np.arange(self.A.shape[-1]), indeces_remove_gt)
            self.select_components(idx_components=components_to_keep)

        print('Duplicates gt:' + str(len(duplicates_gt)))
        return duplicates_gt, indeces_keep_gt, indeces_remove_gt, D_gt, overlap_gt

    def masks_2_neurofinder(self, dataset_name):
        if self.A_thr is None:
            raise Exception(
                'You need to compute thresolded components before calling this method: use the threshold_components method')
        bin_masks = self.A_thr.reshape([self.dims[0], self.dims[1], -1], order='F').transpose([2, 0, 1])
        return nf_masks_to_neurof_dict(bin_masks, dataset_name)




def compare_components(estimate_gt, estimate_cmp,  Cn=None, thresh_cost=.8, min_dist=10, print_assignment=False, labels=['GT', 'CMP'], plot_results=False):
    if estimate_gt.A_thr is None:
        raise Exception(
            'You need to compute thresolded components for first argument before calling remove_duplicates: use the threshold_components method')
    if estimate_cmp.A_thr is None:
        raise Exception(
            'You need to compute thresolded components for second argument before calling remove_duplicates: use the threshold_components method')


    if plot_results:
        plt.figure(figsize=(20, 10))

    dims = estimate_gt.dims
    A_gt_thr_bin = (estimate_gt.A_thr>0).reshape([dims[0], dims[1], -1], order='F').transpose([2, 0, 1]) * 1.
    A_thr_bin = (estimate_cmp.A_thr>0).reshape([dims[0], dims[1], -1], order='F').transpose([2, 0, 1]) * 1.

    tp_gt, tp_comp, fn_gt, fp_comp, performance_cons_off = nf_match_neurons_in_binary_masks(
        A_gt_thr_bin, A_thr_bin, thresh_cost=thresh_cost, min_dist=min_dist, print_assignment=print_assignment,
        plot_results=plot_results, Cn=Cn, labels=labels)

    return tp_gt, tp_comp, fn_gt, fp_comp, performance_cons_off

