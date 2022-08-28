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
from .utilities import detrend_df_f, decimation_matrix
from .spatial import threshold_components
from .temporal import constrained_foopsi_parallel
from .merging import merge_iteration, merge_components
from ...components_evaluation import (
        evaluate_components_CNN, estimate_components_quality_auto,
        select_components_from_metrics, compute_eccentricity)
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

            ecc: np.ndarray
                eccentricity values
        """
        # Sanity checks (right now these just warn, but eventually we would like to fail)
        if R is not None and not isinstance(R, np.ndarray):
            logging.warning(f"Estimates.R should be an np.ndarray but was assigned a {type(R)}")

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
        self.ecc = None

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

    def compute_background(self, Yr):
        """compute background (has big memory requirements)

         Args:
             Yr :    np.ndarray
                 movie in format pixels (d) x frames (T)
            """
        logging.warning("Computing the full background has big memory requirements!")
        if self.f is not None:  # low rank background
            return self.b.dot(self.f)
        else:  # ring model background
            ssub_B = np.round(np.sqrt(Yr.shape[0] / self.W.shape[0])).astype(int)
            if ssub_B == 1:
                return self.b0[:, None] + self.W.dot(Yr - self.A.dot(self.C) - self.b0[:, None])
            else:
                ds_mat = decimation_matrix(self.dims, ssub_B)
                B = ds_mat.dot(Yr) - ds_mat.dot(self.A).dot(self.C) - ds_mat.dot(self.b0)[:, None]
                B = self.W.dot(B).reshape(((self.dims[0] - 1) // ssub_B + 1,
                                           (self.dims[1] - 1) // ssub_B + 1, -1), order='F')
                B = self.b0[:, None] + np.repeat(np.repeat(B, ssub_B, 0), ssub_B, 1
                                                 )[:self.dims[0], :self.dims[1]].reshape(
                    (-1, B.shape[-1]), order='F')
                return B

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
        if self.S is not None:
            self.S = nA_mat * self.S
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
                whether to save the components from initialization so that they
                can be restored using the restore_discarded_components method

        Returns:
            self: Estimates object
        """
        if use_object:
            idx_components = self.idx_components
            idx_components_bad = self.idx_components_bad
        else:
            idx_components_bad = np.setdiff1d(np.arange(self.A.shape[-1]), idx_components)

        if idx_components is not None:
            if save_discarded_components and self.discarded_components is None:
                self.discarded_components = Estimates()

            for field in ['C', 'S', 'YrA', 'R', 'F_dff', 'g', 'bl', 'c1', 'neurons_sn',
                          'lam', 'cnn_preds', 'SNR_comp', 'r_values', 'coordinates']:
                if getattr(self, field) is not None:
                    if isinstance(getattr(self, field), list):
                        setattr(self, field, np.array(getattr(self, field)))
                    if len(getattr(self, field)) == self.A.shape[-1]:
                        if save_discarded_components:
                            setattr(self.discarded_components, field,
                                    getattr(self, field)[idx_components_bad]
                                    if getattr(self.discarded_components, field) is None else
                                    np.concatenate([getattr(self.discarded_components, field),
                                                    getattr(self, field)[idx_components_bad]]))
                        setattr(self, field, getattr(self, field)[idx_components])
                    else:
                        print('*** Variable ' + field +
                              ' has not the same number of components as A ***')

            for field in ['A', 'A_thr']:
                if getattr(self, field) is not None:
                    if 'sparse' in str(type(getattr(self, field))):
                        if save_discarded_components:
                            if getattr(self.discarded_components, field) is None:
                                setattr(self.discarded_components, field,
                                    getattr(self, field).tocsc()[:, idx_components_bad])
                            else:
                                caiman.source_extraction.cnmf.online_cnmf.csc_append(
                                    getattr(self.discarded_components, field),
                                    getattr(self, field).tocsc()[:, idx_components_bad])
                        setattr(self, field, getattr(self, field).tocsc()[:, idx_components])
                    else:
                        if save_discarded_components:
                            setattr(self.discarded_components, field,
                                getattr(self, field)[:, idx_components_bad]
                                    if getattr(self.discarded_components, field) is None else
                                    np.concatenate([getattr(self.discarded_components, field),
                                        getattr(self, field)[:, idx_components_bad]], axis=-1))
                        setattr(self, field, getattr(self, field)[:, idx_components])

            self.nr = len(idx_components)

            if save_discarded_components:
                if not hasattr(self.discarded_components, 'nr'):
                    self.discarded_components.nr = 0
                self.discarded_components.nr += len(idx_components_bad)
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
                    if isinstance(getattr(self, field), list):
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
        if opts['use_ecc']:
            self.ecc = compute_eccentricity(self.A, dims)
            idx_ecc = np.where(self.ecc < opts['max_ecc'])[0]
            self.idx_components_bad = np.union1d(self.idx_components_bad,
                                                 np.setdiff1d(self.idx_components,
                                                              idx_ecc))
            self.idx_components = np.intersect1d(self.idx_components, idx_ecc)
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
            if opts['use_ecc']:
                idx_ecc = np.where(self.ecc < opts['max_ecc'])[0]
                self.idx_components_bad = np.union1d(self.idx_components_bad,
                                                     np.setdiff1d(self.idx_components,
                                                                  idx_ecc))
                self.idx_components = np.intersect1d(self.idx_components, idx_ecc)

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

    def merge_components(self, Y, params, mx=50, fast_merge=True,
                         dview=None, max_merge_area=None):
            """merges components
            """
            self.A, self.C, self.nr, self.merged_ROIs, self.S, \
            self.bl, self.c1, self.neurons_sn, self.g, empty_merged, \
            self.YrA =\
                merge_components(Y, self.A, self.b, self.C, self.YrA,
                                 self.f, self.S, self.sn, params.get_group('temporal'),
                                 params.get_group('spatial'), dview=dview,
                                 bl=self.bl, c1=self.c1, sn=self.neurons_sn,
                                 g=self.g, thr=params.get('merging', 'merge_thr'), mx=mx,
                                 fast_merge=fast_merge, merge_parallel=params.get('merging', 'merge_parallel'),
                                 max_merge_area=max_merge_area)

    def manual_merge(self, components, params):
        ''' merge a given list of components. The indices
        of components are pythonic, i.e., they start from 0. Moreover,
        the indices refer to the absolute indices, i.e., the indices before
        splitting the components in accepted and rejected. If you want to e.g.
        merge components 0 from idx_components and 9 from idx_components_bad
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
        nbmrg -= sum(empty)
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
        if self.neurons_sn is not None:
            self.neurons_sn = np.hstack((self.neurons_sn[good_neurons],
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
            raise Exception('You need to compute thresholded components before calling remove_duplicates: use the threshold_components method')

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
            raise Exception('You need to compute thresholded components before calling remove_duplicates: use the threshold_components method')

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
                'You need to compute thresholded components before calling this method: use the threshold_components method')
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
            mod.add(img_seg)
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

            ps = img_seg.create_plane_segmentation(
                name='PlaneSegmentation',
                description='CNMF_ROIs',
                imaging_plane=imaging_plane,
                reference_images=image_series)

            ps.add_column('r', 'description of r values')
            ps.add_column('snr', 'signal to noise ratio')
            ps.add_column('accepted', 'in accepted list')
            ps.add_column('rejected', 'in rejected list')
            if self.cnn_preds is not None:
                ps.add_column('cnn', 'description of CNN')
            if self.idx_components is not None:
                ps.add_column('keep', 'in idx_components')

            # Add ROIs
            for i in range(self.A.shape[-1]):
                add_roi_kwargs = dict(image_mask=self.A.T[i].T.toarray().reshape(self.dims),
                                      r=self.r_values[i], snr=self.SNR_comp[i], accepted=False, rejected=False)
                if hasattr(self, 'accepted_list'):
                    add_roi_kwargs.update(accepted=i in self.accepted_list)
                if hasattr(self, 'rejected_list'):
                    add_roi_kwargs.update(rejected=i in self.rejected_list)
                if self.cnn_preds is not None:
                    add_roi_kwargs.update(cnn=self.cnn_preds[i])
                if self.idx_components is not None:
                    add_roi_kwargs.update(keep=i in self.idx_components)

                ps.add_roi(**add_roi_kwargs)

            for bg in self.b.T:  # Backgrounds
                add_bg_roi_kwargs = dict(image_mask=bg.reshape(self.dims), r=np.nan, snr=np.nan, accepted=False,
                                         rejected=False)
                if 'keep' in ps.colnames:
                    add_bg_roi_kwargs.update(keep=False)
                if 'cnn' in ps.colnames:
                    add_bg_roi_kwargs.update(cnn=np.nan)
                ps.add_roi(**add_bg_roi_kwargs)

            # Add Traces
            n_rois = self.A.shape[-1]
            n_bg = len(self.f)
            rt_region_roi = ps.create_roi_table_region(
                'ROIs', region=list(range(n_rois)))

            rt_region_bg = ps.create_roi_table_region(
                'Background', region=list(range(n_rois, n_rois+n_bg)))

            timestamps = np.arange(self.f.shape[1]) / imaging_rate + starting_time

            # Neurons
            fl.create_roi_response_series(name='RoiResponseSeries', data=self.C.T, rois=rt_region_roi, unit='lumens',
                                          timestamps=timestamps)
            # Background
            fl.create_roi_response_series(name='Background_Fluorescence_Response', data=self.f.T, rois=rt_region_bg,
                                          unit='lumens', timestamps=timestamps)

            mod.add(TimeSeries(name='residuals', description='residuals', data=self.YrA.T, timestamps=timestamps,
                               unit='NA'))
            if hasattr(self, 'Cn'):
                images = Images('summary_images')
                images.add_image(GrayscaleImage(name='local_correlations', data=self.Cn))

                # Add MotionCorrection
    #            create_corrected_image_stack(corrected, original, xy_translation, name='CorrectedImageStack')
            io.write(nwbfile)


def compare_components(estimate_gt, estimate_cmp,  Cn=None, thresh_cost=.8, min_dist=10, print_assignment=False,
                       labels=['GT', 'CMP'], plot_results=False):
    if estimate_gt.A_thr is None:
        raise Exception(
            'You need to compute thresholded components for first argument before calling remove_duplicates: use the threshold_components method')
    if estimate_cmp.A_thr is None:
        raise Exception(
            'You need to compute thresholded components for second argument before calling remove_duplicates: use the threshold_components method')

    if plot_results:
        plt.figure(figsize=(20, 10))

    dims = estimate_gt.dims
    A_gt_thr_bin = (estimate_gt.A_thr.toarray()>0).reshape([dims[0], dims[1], -1], order='F').transpose([2, 0, 1]) * 1.
    A_thr_bin = (estimate_cmp.A_thr.toarray()>0).reshape([dims[0], dims[1], -1], order='F').transpose([2, 0, 1]) * 1.

    tp_gt, tp_comp, fn_gt, fp_comp, performance_cons_off = nf_match_neurons_in_binary_masks(
        A_gt_thr_bin, A_thr_bin, thresh_cost=thresh_cost, min_dist=min_dist, print_assignment=print_assignment,
        plot_results=plot_results, Cn=Cn, labels=labels)

    return tp_gt, tp_comp, fn_gt, fp_comp, performance_cons_off
