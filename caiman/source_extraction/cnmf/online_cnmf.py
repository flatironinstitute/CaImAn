#!/usr/bin/env python

""" Online Constrained Nonnegative Matrix Factorization

The general file class which is used to analyze calcium imaging data in an
online fashion using the OnACID algorithm. The output of the algorithm
is storead in an Estimates class

More info:
------------
Giovannucci, A., Friedrich, J., Kaufman, M., Churchland, A., Chklovskii, D., 
Paninski, L., & Pnevmatikakis, E.A. (2017). OnACID: Online analysis of calcium
imaging data in real time. In Advances in Neural Information Processing Systems 
(pp. 2381-2391).
@url http://papers.nips.cc/paper/6832-onacid-online-analysis-of-calcium-imaging-data-in-real-time
"""

import cv2
import logging
from math import sqrt
from multiprocessing import cpu_count
import numpy as np
import os
from scipy.ndimage import percentile_filter
from scipy.sparse import coo_matrix, csc_matrix, spdiags, hstack
from scipy.stats import norm
from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize
import tensorflow as tf
from time import time

import caiman
import caiman.base.movies
from caiman.components_evaluation import compute_event_exceptionality
import caiman.mmapping
from caiman.motion_correction import (motion_correct_iteration_fast,
                                  tile_and_correct, tile_and_correct_3d,
                                  high_pass_filter_space, sliding_window,
                                  register_translation_3d, apply_shifts_dft)
import caiman.paths
from caiman.source_extraction.cnmf.cnmf import CNMF
from caiman.source_extraction.cnmf.estimates import Estimates
from caiman.source_extraction.cnmf.initialization import imblur, initialize_components, hals, downscale
from caiman.source_extraction.cnmf.oasis import OASIS
from caiman.source_extraction.cnmf.params import CNMFParams
from caiman.source_extraction.cnmf.pre_processing import get_noise_fft
from caiman.source_extraction.cnmf.utilities import (update_order, peak_local_max, decimation_matrix,
                        gaussian_filter, uniform_filter)
import caiman.summary_images
from caiman.utils.utils import save_dict_to_hdf5, load_dict_from_hdf5, parmap, load_graph
from caiman.utils.stats import pd_solve
from caiman.utils.nn_models import (fit_NL_model, create_LN_model, quantile_loss, rate_scheduler)

try:
    cv2.setNumThreads(0)
except():
    pass

#FIXME ???
try:
    profile
except:
    def profile(a): return a


class OnACID(object):
    """  Source extraction of streaming data using online matrix factorization.
    The class can be initialized by passing a "params" object for setting up
    the relevant parameters and an "Estimates" object for setting an initial
    state of the algorithm (optional)

    Methods:
        initialize_online: 
            Initialize the online algorithm using a provided method, and prepare
            the online object

        _prepare_object: 
            Prepare the online object given a set of estimates

        fit_next:
            Fit the algorithm on the next data frame

        fit_online:
            Run the entire online pipeline on a given list of files
    """

    def __init__(self, params=None, estimates=None, path=None, dview=None, Ain=None):
        """

        Args:
            params: CNMFParams
                CNMFParams object with parameters that are used to perform online motion correction, followed by online CNMF 

            estimates: Estimates, optional
                Estimates object to load an existing model

            path: str, optional
                path to a saved OnACID model hdf5 file on disk

            dview:
                dview instance, multiprocessing object

            Ain: csc_matrix, optional
                binary masked for seeded initialization as a Compressed Sparse Column matrix.
                To use set ``"init_method"`` to ``"seeded"``

        """
        if path is None:
            self.params = CNMFParams() if params is None else params
            self.estimates = Estimates() if estimates is None else estimates
        else:
            onacid = load_OnlineCNMF(path)
            self.params = params if params is not None else onacid.params
            self.estimates= estimates if estimates is not None else onacid.estimates
        self.dview = dview
        if Ain is not None:
            self.estimates.A = Ain
        if self.params.motion['splits_rig'] > self.params.online['init_batch']/2:
            raise Exception("In params, online.init_batch and motion.num_frames_split have incompatible values; consider increasing online.init_batch to be a small multiple of the other")
            # See issue #1483; it would actually be better to change initialisation so it ignores splits_rig (either using a default
            # value or providing an alternate parameter for that), but that's a much more intrusive change and would potentially
            # change things for code/notebooks that've worked for a long time; we should save such changes for a major rewrite
            # (if someone takes a particular interest in that).

    @profile
    def _prepare_object(self, Yr, T, new_dims=None, idx_components=None):

        logger = logging.getLogger("caiman")
        init_batch = self.params.get('online', 'init_batch')
        old_dims = self.params.get('data', 'dims')
        self.is1p = (self.params.get('init', 'method_init') == 'corr_pnr' and 
                    self.params.get('init', 'ring_size_factor') is not None)

        if idx_components is None:
            idx_components = range(self.estimates.A.shape[-1])

        self.estimates.A = self.estimates.A.astype(np.float32)
        self.estimates.C = self.estimates.C.astype(np.float32)
        if self.estimates.f is not None:
            self.estimates.f = self.estimates.f.astype(np.float32)
        if self.estimates.b is not None:
            self.estimates.b = self.estimates.b.astype(np.float32)
        self.estimates.YrA = self.estimates.YrA.astype(np.float32)
        self.estimates.select_components(idx_components=idx_components)
        self.N = self.estimates.A.shape[-1]
        self.M = self.params.get('init', 'nb') + self.N

        if not self.params.get('online', 'update_num_comps'):
            self.params.set('online', {'expected_comps': self.N})
        elif (self.params.get('online', 'expected_comps') <= 
            self.N + self.params.get('online', 'max_num_added')):
            self.params.set('online', {'expected_comps': self.N + 
                self.params.get('online', 'max_num_added') + 200})
        expected_comps = self.params.get('online', 'expected_comps')

        if Yr.shape[-1] != self.params.get('online', 'init_batch'):
            raise Exception(
                'The movie size used for initialization does not match with the minibatch size')

        if new_dims is not None:

            new_Yr = np.zeros([np.prod(new_dims), init_batch])
            for ffrr in range(init_batch):
                tmp = cv2.resize(Yr[:, ffrr].reshape(old_dims, order='F'), new_dims[::-1])
                print(tmp.shape)
                new_Yr[:, ffrr] = tmp.reshape([np.prod(new_dims)], order='F')
            Yr = new_Yr
            A_new = csc_matrix((np.prod(new_dims), self.estimates.A.shape[-1]),
                               dtype=np.float32)
            for neur in range(self.N):
                a = self.estimates.A.tocsc()[:, neur].toarray()
                a = a.reshape(old_dims, order='F')
                a = cv2.resize(a, new_dims[::-1]).reshape([-1, 1], order='F')

                A_new[:, neur] = csc_matrix(a)

            self.estimates.A = A_new
            if self.estimates.b.size:
                self.estimates.b = self.estimates.b.reshape(old_dims, order='F')
                self.estimates.b = cv2.resize(
                    self.estimates.b, new_dims[::-1]).reshape([-1, 1], order='F')
            else:
                self.estimates.b.shape = (np.prod(new_dims), 0)
            if self.is1p:
                # self.estimates.b0 is calculated below
                # ToDo, but not easy: resize self.estimates.W
                raise NotImplementedError('change of dimensions not yet implemented for CNMF-E')

            self.estimates.dims = new_dims
        else:
            self.estimates.dims = old_dims

        self.estimates.normalize_components()
        self.estimates.A = self.estimates.A.todense()
        self.estimates.noisyC = np.zeros(
            (self.params.get('init', 'nb') + expected_comps, T), dtype=np.float32)
        self.estimates.C_on = np.zeros((expected_comps, T), dtype=np.float32)

        self.estimates.noisyC[self.params.get('init', 'nb'):self.M, :self.params.get('online', 'init_batch')] = self.estimates.C + self.estimates.YrA
        self.estimates.noisyC[:self.params.get('init', 'nb'), :self.params.get('online', 'init_batch')] = self.estimates.f
        if self.params.get('preprocess', 'p'):
            # if no parameter for calculating the spike size threshold is given, then use L1 penalty
            if self.params.get('temporal', 's_min') is None:
                use_L1 = True
            else:
                use_L1 = False

            self.estimates.OASISinstances = [OASIS(
                g=gam[0], lam=0 if not use_L1 else l,
                s_min=0 if use_L1 else (self.params.get('temporal', 's_min') if self.params.get('temporal', 's_min') > 0 else
                                         (-self.params.get('temporal', 's_min') * sn * np.sqrt(1 - np.sum(gam)))),
                b=b,
                g2=0 if self.params.get('preprocess', 'p') < 2 else gam[1])
                for gam, l, b, sn in zip(self.estimates.g, self.estimates.lam, self.estimates.bl, self.estimates.neurons_sn)]

            for i, o in enumerate(self.estimates.OASISinstances):
                o.fit(self.estimates.noisyC[i + self.params.get('init', 'nb'), :init_batch])
                self.estimates.C_on[i, :init_batch] = o.c
        else:
            self.estimates.C_on[:self.N, :init_batch] = self.estimates.C
        
        if self.is1p:
            ssub_B = self.params.get('init', 'ssub_B') * self.params.get('init', 'ssub')
            X = Yr[:, :init_batch] - np.asarray(self.estimates.A.dot(self.estimates.C))
            self.estimates.b0 = X.mean(1)
            X -= self.estimates.b0[:, None]
            if ssub_B > 1:
                self.estimates.downscale_matrix = decimation_matrix(self.estimates.dims, ssub_B)
                self.estimates.upscale_matrix = self.estimates.downscale_matrix.T
                self.estimates.upscale_matrix.data = np.ones_like(self.estimates.upscale_matrix.data)
                X = self.estimates.downscale_matrix.dot(X)
            if self.params.get('online', 'full_XXt'):
                self.estimates.XXt = X.dot(X.T)
            else:
                self.XXt_mats = []
                self.XXt_vecs = []
                self.W_ind = []
                W = self.estimates.W
                for p in range(W.shape[0]):
                    index = W.indices[W.indptr[p]:W.indptr[p + 1]]
                    self.W_ind.append(index)
                    x_i = X[index]
                    self.XXt_mats.append(x_i.dot(x_i.T))
                    self.XXt_vecs.append(x_i.dot(X[p].T))

        self.estimates.Ab, self.ind_A, self.estimates.CY, self.estimates.CC = init_shapes_and_sufficient_stats(
            Yr[:, :init_batch].reshape(self.estimates.dims + (-1,), order='F'),
            self.estimates.A, self.estimates.C_on[:self.N, :init_batch],
            self.estimates.b, self.estimates.noisyC[:self.params.get('init', 'nb'), :init_batch],
            W=self.estimates.W if self.is1p else None, b0=self.estimates.b0 if self.is1p else None,
            ssub_B=self.params.get('init', 'ssub_B') * self.params.get('init', 'ssub'),
            downscale_matrix=self.estimates.downscale_matrix if (self.is1p and ssub_B > 1) else None,
            upscale_matrix=self.estimates.upscale_matrix if (self.is1p and ssub_B > 1) else None)

        self.estimates.CY = self.estimates.CY * 1. / self.params.get('online', 'init_batch')
        self.estimates.CC = 1 * self.estimates.CC / self.params.get('online', 'init_batch')

        logger.info(f'Expecting {expected_comps} components')
        self.estimates.CY.resize([expected_comps + self.params.get('init', 'nb'), self.estimates.CY.shape[-1]], refcheck=False)
        if self.params.get('online', 'use_dense'):
            self.estimates.Ab_dense = np.zeros((self.estimates.CY.shape[-1], expected_comps + self.params.get('init', 'nb')),
                                     dtype=np.float32)
            self.estimates.Ab_dense[:, :self.estimates.Ab.shape[1]] = self.estimates.Ab.toarray()
        self.estimates.C_on = np.vstack(
            [self.estimates.noisyC[:self.params.get('init', 'nb'), :], self.estimates.C_on.astype(np.float32)])

        if not self.is1p:
            self.params.set('init', {'gSiz': np.add(np.multiply(np.ceil(
                self.params.get('init', 'gSig')).astype(int), 2), 1)})

        self.estimates.Yr_buf = RingBuffer(Yr[:, self.params.get('online', 'init_batch') - self.params.get('online', 'minibatch_shape'):
                                    self.params.get('online', 'init_batch')].T.copy(), self.params.get('online', 'minibatch_shape'))
        self.estimates.Yres_buf = RingBuffer(self.estimates.Yr_buf - self.estimates.Ab.dot(
            self.estimates.C_on[:self.M, self.params.get('online', 'init_batch') - self.params.get('online', 'minibatch_shape'):
            self.params.get('online', 'init_batch')]).T, self.params.get('online', 'minibatch_shape'))

        if self.is1p:
            estim = self.estimates
            estim.Yres_buf -= estim.b0
            if ssub_B == 1:
                estim.Atb = estim.Ab.T.dot(estim.W.dot(estim.b0) - estim.b0)
                estim.AtW = estim.Ab.T.dot(estim.W)
                estim.AtWA = estim.AtW.dot(estim.Ab).toarray()
                estim.Yres_buf -= estim.W.dot(estim.Yres_buf.T).T
            else:
                A_ds = estim.downscale_matrix.dot(estim.Ab)
                estim.Atb = estim.Ab.T.dot(estim.upscale_matrix.dot(
                    estim.W.dot(estim.downscale_matrix.dot(estim.b0))) - estim.b0)
                estim.AtW = A_ds.T.dot(estim.W)
                estim.AtWA = estim.AtW.dot(A_ds).toarray()
                estim.Yres_buf -= estim.upscale_matrix.dot(estim.W.dot(
                    estim.downscale_matrix.dot(estim.Yres_buf.T))).T

        self.estimates.sn = np.array(np.std(self.estimates.Yres_buf,axis=0))
        self.estimates.vr = np.array(np.var(self.estimates.Yres_buf,axis=0))
        self.estimates.mn = self.estimates.Yres_buf.mean(0)
        self.estimates.mean_buff = self.estimates.Yres_buf.mean(0)
        self.estimates.ind_new = []
        if self.params.get('online', 'use_corr_img'):
            self.estimates.rho_buf = None
            self.estimates.sv = None
        else:
            self.estimates.rho_buf = imblur(np.maximum(self.estimates.Yres_buf.T, 0).reshape(
                self.estimates.dims + (-1,), order='F'), sig=self.params.get('init', 'gSig'),
                siz=self.params.get('init', 'gSiz'), nDimBlur=len(self.estimates.dims))**2
            self.estimates.rho_buf = np.reshape(
                self.estimates.rho_buf, (np.prod(self.estimates.dims), -1)).T
            self.estimates.rho_buf = np.ascontiguousarray(self.estimates.rho_buf)
            self.estimates.rho_buf = RingBuffer(self.estimates.rho_buf, self.params.get('online', 'minibatch_shape'))
            self.estimates.sv = np.sum(self.estimates.rho_buf.get_last_frames(
                min(self.params.get('online', 'init_batch'), self.params.get('online', 'minibatch_shape')) - 1), 0)
        self.estimates.AtA = (self.estimates.Ab.T.dot(self.estimates.Ab)).toarray()
        self.estimates.AtY_buf = self.estimates.Ab.T.dot(self.estimates.Yr_buf.T)
        self.estimates.groups = list(map(list, update_order(self.estimates.Ab)[0]))
        self.update_counter = 2**np.linspace(0, 1, self.N, dtype=np.float32)
        self.estimates.CC = np.ascontiguousarray(self.estimates.CC)
        self.estimates.CY = np.ascontiguousarray(self.estimates.CY)
        self.time_neuron_added:list = []
        for nneeuu in range(self.N):
            self.time_neuron_added.append((nneeuu, self.params.get('online', 'init_batch')))
        if self.params.get('online', 'dist_shape_update'):
            self.time_spend = 0
            self.comp_upd:list = []
        # setup per patch classifier

        if self.params.get('online', 'path_to_model') is None or self.params.get('online', 'sniper_mode') is False:
            loaded_model = None
            self.params.set('online', {'sniper_mode': False})
            self.tf_in = None
            self.tf_out = None
        else:
            try:
                from tensorflow.keras.models import model_from_json
                logger.info('Using Keras')
                use_keras = True
            except(ModuleNotFoundError):
                use_keras = False
                logger.info('Using Tensorflow')
            if use_keras:
                path = self.params.get('online', 'path_to_model').split(".")[:-1]
                json_path = ".".join(path + ["json"])
                model_path = ".".join(path + ["h5"])
                json_file = open(json_path, 'r')
                loaded_model_json = json_file.read()
                json_file.close()
                loaded_model = model_from_json(loaded_model_json)
                loaded_model.load_weights(model_path)
                self.tf_in = None
                self.tf_out = None
            else:
                path = self.params.get('online', 'path_to_model').split(".")[:-1]
                model_path = '.'.join(path + ['h5', 'pb'])
                loaded_model = load_graph(model_path)
                self.tf_in = loaded_model.get_tensor_by_name('prefix/conv2d_1_input:0')
                self.tf_out = loaded_model.get_tensor_by_name('prefix/output_node0:0')
                loaded_model = tf.Session(graph=loaded_model)
        self.loaded_model = loaded_model

        if self.is1p:
            from skimage.morphology import disk
            radius = int(round(self.params.get('init', 'ring_size_factor') *
                self.params.get('init', 'gSiz')[0] / float(ssub_B)))
            ring = disk(radius + 1)
            ring[1:-1, 1:-1] -= disk(radius)
            self._ringidx = [i - radius - 1 for i in np.nonzero(ring)]
            self._dims_B = ((self.estimates.dims[0] - 1) // ssub_B + 1,
                            (self.estimates.dims[1] - 1) // ssub_B + 1)

            def get_indices_of_pixels_on_ring(self, pixel):
                pixel = np.unravel_index(pixel, self._dims_B, order='F')
                x = pixel[0] + self._ringidx[0]
                y = pixel[1] + self._ringidx[1]
                inside = (x >= 0) * (x < self._dims_B[0]) * (y >= 0) * (y < self._dims_B[1])
                return np.ravel_multi_index((x[inside], y[inside]), self._dims_B, order='F')
            self.get_indices_of_pixels_on_ring = get_indices_of_pixels_on_ring.__get__(self)

            # generate list of indices of XX' that get accessed
            if self.params.get('online', 'full_XXt'):
                l = np.prod(self._dims_B)
                tmp = np.zeros((l, l), dtype=bool)
                for p in range(l):
                    index = self.get_indices_of_pixels_on_ring(p)
                    tmp[index[:, None], index] = True
                    tmp[index, p] = True
                self.estimates.XXt_ind = list([np.where(t)[0] for t in tmp])

        if self.params.get('online', 'use_corr_img'):
            Yres = Yr[:, :init_batch] - self.estimates.Ab.dot(
                self.estimates.C_on[:self.M, :init_batch])
            if self.is1p:
                Yres -= self.estimates.b0[:, None]
                if ssub_B == 1:
                    Yres -= self.estimates.W.dot(Yres)
                else:
                    Yres -= estim.upscale_matrix.dot(estim.W.dot(
                        estim.downscale_matrix.dot(Yres)))
            d1, d2 = self.estimates.dims
            Yres = Yres.reshape((d1, d2, -1), order='F')

            (self.estimates.first_moment, self.estimates.second_moment,
                self.estimates.crosscorr, self.estimates.col_ind, self.estimates.row_ind,
                self.estimates.num_neigbors, self.estimates.corrM, self.estimates.corr_img) = \
            caiman.summary_images.prepare_local_correlations(Yres, swap_dim=True, eight_neighbours=False)
            self.estimates.max_img = Yres.max(-1)

        self.comp_upd = []
        self.t_shapes:list = []
        self.t_detect:list = []
        self.t_motion:list = []
        self.t_stat:list = []

        return self

    @profile
    def fit_next(self, t, frame_in, num_iters_hals=3):
        """
        This method fits the next frame using the CaImAn online algorithm and
        updates the object. Does NOT perform motion correction, see ``mc_next()``

        Args
            t : int
                temporal index of the next frame to fit

            frame_in : array
                flattened array of shape (x * y [ * z],) containing the t-th image.

            num_iters_hals: int, optional
                maximal number of iterations for HALS (NNLS via blockCD)
        """
        # FIXME This whole function is overly complex; should rewrite it for legibility
        logger = logging.getLogger("caiman")
        t_start = time()

        # locally scoped variables for brevity of code and faster look up
        nb_ = self.params.get('init', 'nb')
        Ab_ = self.estimates.Ab
        mbs = self.params.get('online', 'minibatch_shape')
        ssub_B = self.params.get('init', 'ssub_B') * self.params.get('init', 'ssub')
        expected_comps = self.params.get('online', 'expected_comps')
        frame = frame_in.astype(np.float32)
        self.estimates.Yr_buf.append(frame)
        if len(self.estimates.ind_new) > 0:
            self.estimates.mean_buff = self.estimates.Yres_buf.mean(0)

        if (not self.params.get('online', 'simultaneously')) or self.params.get('preprocess', 'p') == 0:
            # get noisy fluor value via NNLS (project data on shapes & demix)
            C_in = self.estimates.noisyC[:self.M, t - 1].copy()
            if self.is1p:
                self.estimates.C_on[:self.M, t], self.estimates.noisyC[:self.M, t] = demix1p(
                    frame, self.estimates.Ab, C_in, self.estimates.AtA, Atb=self.estimates.Atb,
                    AtW=self.estimates.AtW, AtWA=self.estimates.AtWA, iters=num_iters_hals,
                    groups=self.estimates.groups, ssub_B=ssub_B, 
                    downscale_matrix=self.estimates.downscale_matrix if ssub_B > 1 else None)
            else:
                self.estimates.C_on[:self.M, t], self.estimates.noisyC[:self.M, t] = HALS4activity(
                    frame, self.estimates.Ab, C_in, self.estimates.AtA, iters=num_iters_hals, groups=self.estimates.groups)
            if self.params.get('preprocess', 'p'):
                # denoise & deconvolve
                for i, o in enumerate(self.estimates.OASISinstances):
                    o.fit_next(self.estimates.noisyC[nb_ + i, t])
                    self.estimates.C_on[nb_ + i, t - o.get_l_of_last_pool() +
                              1: t + 1] = o.get_c_of_last_pool()

        else:
            if self.is1p:
                raise NotImplementedError(
                    'simultaneous demixing and deconvolution not implemented yet for CNMF-E')
            # update buffer, initialize C with previous value
            self.estimates.C_on[:, t] = self.estimates.C_on[:, t - 1]
            self.estimates.noisyC[:, t] = self.estimates.C_on[:, t - 1]
            self.estimates.AtY_buf = np.concatenate((self.estimates.AtY_buf[:, 1:], self.estimates.Ab.T.dot(frame)[:, None]), 1) \
                if self.params.get('online', 'n_refit') else self.estimates.Ab.T.dot(frame)[:, None]
            # demix, denoise & deconvolve
            (self.estimates.C_on[:self.M, t + 1 - mbs:t + 1], self.estimates.noisyC[:self.M, t + 1 - mbs:t + 1],
                self.estimates.OASISinstances) = demix_and_deconvolve(
                self.estimates.C_on[:self.M, t + 1 - mbs:t + 1],
                self.estimates.noisyC[:self.M, t + 1 - mbs:t + 1],
                self.estimates.AtY_buf, self.estimates.AtA, self.estimates.OASISinstances, iters=num_iters_hals,
                n_refit=self.params.get('online', 'n_refit'))
            for i, o in enumerate(self.estimates.OASISinstances):
                self.estimates.C_on[nb_ + i, t - o.get_l_of_last_pool() + 1: t +
                          1] = o.get_c_of_last_pool()

        #self.estimates.mean_buff = self.estimates.Yres_buf.mean(0)
        res_frame = frame - self.estimates.Ab.dot(self.estimates.C_on[:self.M, t])
        if self.is1p:
            self.estimates.b0 = self.estimates.b0 * (t-1)/t + res_frame/t
            res_frame -= self.estimates.b0
            res_frame -= (self.estimates.W.dot(res_frame) if ssub_B == 1 else
                          self.estimates.upscale_matrix.dot(self.estimates.W.dot(
                            self.estimates.downscale_matrix.dot(res_frame))))
        mn_ = self.estimates.mn.copy()
        self.estimates.mn = (t-1)/t*self.estimates.mn + res_frame/t
        self.estimates.vr = (t-1)/t*self.estimates.vr + (res_frame - mn_)*(res_frame - self.estimates.mn)/t
        self.estimates.sn = np.sqrt(self.estimates.vr)
        
        t_new = time()
        num_added = 0
        if self.params.get('online', 'update_num_comps'):
            if self.params.get('online', 'use_corr_img'):
                corr_img_mode = 'simple'  #'exponential'  # 'cumulative'
                self.estimates.corr_img = caiman.summary_images.update_local_correlations(
                    t + 1 if corr_img_mode == 'cumulative' else mbs, 
                    res_frame.reshape((1,) + self.estimates.dims, order='F'),
                    self.estimates.first_moment, self.estimates.second_moment,
                    self.estimates.crosscorr, self.estimates.col_ind, self.estimates.row_ind,
                    self.estimates.num_neigbors, self.estimates.corrM,
                    del_frames=[self.estimates.Yres_buf[self.estimates.Yres_buf.cur]]
                    if corr_img_mode == 'simple' else None)
            self.estimates.mean_buff += (res_frame-self.estimates.Yres_buf[self.estimates.Yres_buf.cur])/self.params.get('online', 'minibatch_shape')
            self.estimates.Yres_buf.append(res_frame)

            res_frame = np.reshape(res_frame, self.estimates.dims, order='F')

            if self.params.get('online', 'use_corr_img'):
                self.estimates.max_img = np.max([self.estimates.max_img, res_frame], 0)
            else:
                rho = imblur(np.maximum(res_frame,0), sig=self.params.get('init', 'gSig'),
                             siz=self.params.get('init', 'gSiz'),
                             nDimBlur=len(self.params.get('data', 'dims')))**2
                rho = np.reshape(rho, np.prod(self.params.get('data', 'dims')))
                self.estimates.rho_buf.append(rho)

            # old_max_img = self.estimates.max_img.copy()
            if self.params.get('preprocess', 'p') == 1:
                g_est = np.mean(self.estimates.g)
            elif self.params.get('preprocess', 'p') == 2:
                g_est = np.mean(self.estimates.g, 0)
            else:
                g_est = 0
            use_corr = self.params.get('online', 'use_corr_img')
            # FIXME The next statement is really hard to read
            (self.estimates.Ab, Cf_temp, self.estimates.Yres_buf, self.estimates.rho_buf,
                self.estimates.CC, self.estimates.CY, self.ind_A, self.estimates.sv,
                self.estimates.groups, self.estimates.ind_new, self.ind_new_all,
                self.estimates.sv, self.cnn_pos) = update_num_components(
                t, self.estimates.sv, self.estimates.Ab, self.estimates.C_on[:self.M, (t - mbs + 1):(t + 1)],
                self.estimates.Yres_buf, self.estimates.Yr_buf, self.estimates.rho_buf,
                self.params.get('data', 'dims'), self.params.get('init', 'gSig'),
                self.params.get('init', 'gSiz'), self.ind_A, self.estimates.CY, self.estimates.CC,
                rval_thr=self.params.get('online', 'rval_thr'),
                thresh_fitness_delta=self.params.get('online', 'thresh_fitness_delta'),
                thresh_fitness_raw=self.params.get('online', 'thresh_fitness_raw'),
                thresh_overlap=self.params.get('online', 'thresh_overlap'), groups=self.estimates.groups,
                batch_update_suff_stat=self.params.get('online', 'batch_update_suff_stat'),
                gnb=self.params.get('init', 'nb'), sn=self.estimates.sn, 
                g=g_est, s_min=self.params.get('temporal', 's_min'),
                Ab_dense=self.estimates.Ab_dense if self.params.get('online', 'use_dense') else None,
                oases=self.estimates.OASISinstances if self.params.get('preprocess', 'p') else None,
                N_samples_exceptionality=self.params.get('online', 'N_samples_exceptionality'),
                max_num_added=self.params.get('online', 'max_num_added'),
                min_num_trial=self.params.get('online', 'min_num_trial'),
                loaded_model = self.loaded_model, test_both=self.params.get('online', 'test_both'),
                thresh_CNN_noisy = self.params.get('online', 'thresh_CNN_noisy'),
                sniper_mode=self.params.get('online', 'sniper_mode'),
                use_peak_max=self.params.get('online', 'use_peak_max'),
                mean_buff=self.estimates.mean_buff,
                tf_in=self.tf_in, tf_out=self.tf_out,
                ssub_B=ssub_B, W=self.estimates.W if self.is1p else None,
                b0=self.estimates.b0 if self.is1p else None,
                corr_img=self.estimates.corr_img if use_corr else None,
                first_moment=self.estimates.first_moment if use_corr else None,
                second_moment=self.estimates.second_moment if use_corr else None,
                crosscorr=self.estimates.crosscorr if use_corr else None,
                col_ind=self.estimates.col_ind if use_corr else None,
                row_ind=self.estimates.row_ind if use_corr else None,
                corr_img_mode=corr_img_mode if use_corr else None,
                downscale_matrix=self.estimates.downscale_matrix if
                (self.is1p and ssub_B > 1) else None,
                upscale_matrix=self.estimates.upscale_matrix if
                (self.is1p and ssub_B > 1) else None,
                max_img=self.estimates.max_img if use_corr else None)

            num_added = len(self.ind_A) - self.N

            if num_added > 0:
                self.N += num_added
                self.M += num_added
                if self.N + self.params.get('online', 'max_num_added') > expected_comps:
                    expected_comps += 200
                    self.params.set('online', {'expected_comps': expected_comps})
                    self.estimates.CY.resize(
                        [expected_comps + nb_, self.estimates.CY.shape[-1]])
                    self.estimates.C_on.resize(
                        [expected_comps + nb_, self.estimates.C_on.shape[-1]], refcheck=False)
                    self.estimates.noisyC.resize(
                        [expected_comps + nb_, self.estimates.C_on.shape[-1]])
                    if self.params.get('online', 'use_dense'):  # resize won't work due to contingency issue
                        self.estimates.Ab_dense = np.zeros((self.estimates.CY.shape[-1], expected_comps + nb_),
                                                 dtype=np.float32)
                        self.estimates.Ab_dense[:, :Ab_.shape[1]] = Ab_.toarray()
                    logger.info(f'Increasing number of expected components to: {expected_comps}')
                self.update_counter.resize(self.N, refcheck=False)

                self.estimates.noisyC[self.M - num_added:self.M, t - mbs +
                            1:t + 1] = Cf_temp[self.M - num_added:self.M]

                for _ct in range(self.M - num_added, self.M):
                    self.time_neuron_added.append((_ct - nb_, t))
                    if self.params.get('preprocess', 'p'):
                        # N.B. OASISinstances are already updated within update_num_components
                        self.estimates.C_on[_ct, t - mbs + 1: t +
                                  1] = self.estimates.OASISinstances[_ct - nb_].get_c(mbs)
                    else:
                        self.estimates.C_on[_ct, t - mbs + 1: t + 1] = np.maximum(
                            0, self.estimates.noisyC[_ct, t - mbs + 1: t + 1])
                    if self.params.get('online', 'simultaneously') and self.params.get('online', 'n_refit'):
                        self.estimates.AtY_buf = np.concatenate((
                            self.estimates.AtY_buf, [Ab_.data[Ab_.indptr[_ct]:Ab_.indptr[_ct + 1]].dot(
                                self.estimates.Yr_buf.T[Ab_.indices[Ab_.indptr[_ct]:Ab_.indptr[_ct + 1]]])]))
                    # N.B. Ab_dense is already updated within update_num_components as side effect

                AtA = self.estimates.AtA
                self.estimates.AtA = np.zeros((self.M, self.M), dtype=np.float32)
                self.estimates.AtA[:-num_added, :-num_added] = AtA
                if self.params.get('online', 'use_dense'):
                    self.estimates.AtA[:, -num_added:] = self.estimates.Ab.T.dot(
                        self.estimates.Ab_dense[:, self.M - num_added:self.M])
                else:
                    self.estimates.AtA[:, -num_added:] = self.estimates.Ab.T.dot(
                        self.estimates.Ab[:, -num_added:]).toarray()
                self.estimates.AtA[-num_added:] = self.estimates.AtA[:, -num_added:].T
                
                if self.is1p:
                    # # update XXt and W: TODO only update necessary pixels not all!

                    if ssub_B == 1:
                        # self.estimates.AtW = Ab_.T.dot(self.estimates.W)
                        # self.estimates.AtWA = self.estimates.AtW.dot(Ab_).toarray()
                        # faster incremental update of AtW and AtWA instead of above lines:
                        csr_append(self.estimates.AtW, Ab_.T[-num_added:].dot(self.estimates.W))
                        AtWA = self.estimates.AtWA
                        self.estimates.AtWA = np.zeros((self.M, self.M), dtype=np.float32)
                        self.estimates.AtWA[:-num_added, :-num_added] = AtWA
                        self.estimates.AtWA[:, -num_added:] = self.estimates.AtW.dot(
                            Ab_[:, -num_added:]).toarray()
                        self.estimates.AtWA[-num_added:] = self.estimates.AtW[-num_added:].dot(
                            Ab_).toarray()
                        self.estimates.Atb = self.estimates.AtW.dot(
                            self.estimates.b0) - Ab_.T.dot(self.estimates.b0)
                    else:
                        A_ds = self.estimates.downscale_matrix.dot(self.estimates.Ab)
                        csr_append(self.estimates.AtW, A_ds.T[-num_added:].dot(self.estimates.W))
                        AtWA = self.estimates.AtWA
                        self.estimates.AtWA = np.zeros((self.M, self.M), dtype=np.float32)
                        self.estimates.AtWA[:-num_added, :-num_added] = AtWA
                        self.estimates.AtWA[:, -num_added:] = self.estimates.AtW.dot(
                            A_ds[:, -num_added:]).toarray()
                        self.estimates.AtWA[-num_added:] = self.estimates.AtW[-num_added:].dot(
                            A_ds).toarray()
                        self.estimates.Atb = ssub_B**2 * self.estimates.AtW.dot(
                            self.estimates.downscale_matrix.dot(
                                self.estimates.b0)) - Ab_.T.dot(self.estimates.b0)

                # set the update counter to 0 for components that are overlapping the newly added
                idx_overlap = self.estimates.AtA[nb_:-num_added, -num_added:].nonzero()[0]
                self.update_counter[idx_overlap] = 0
            self.t_detect.append(time() - t_new)
        t_stat = time()
        if self.params.get('online', 'batch_update_suff_stat'):
        # faster update using minibatch of frames
            min_batch = min(self.params.get('online', 'update_freq'), mbs)
            if ((t + 1 - self.params.get('online', 'init_batch')) % min_batch == 0):

                ccf = self.estimates.C_on[:self.M, t - min_batch + 1:t + 1]
                y = self.estimates.Yr_buf.get_last_frames(min_batch)
                if self.is1p:  # subtract background
                    if ssub_B == 1:
                        x = (y - self.estimates.Ab.dot(ccf).T - self.estimates.b0).T
                        y -= self.estimates.W.dot(x).T
                    else:
                        x = self.estimates.downscale_matrix.dot(
                            y.T - self.estimates.Ab.dot(ccf) - self.estimates.b0[:, None])
                        y -= self.estimates.upscale_matrix.dot(self.estimates.W.dot(x)).T
                    y -= self.estimates.b0
                    # exploit that we only access some elements of XXt, hence update only these

                    if self.params.get('online', 'full_XXt'):
                        XXt = self.estimates.XXt  # alias for faster repeated look up in large loop
                        for i, idx in enumerate(self.estimates.XXt_ind):
                            XXt[i, idx] += (x[i].dot(x[idx].T)).flatten()
                    else:
                        XXt_mats = self.XXt_mats
                        XXt_vecs = self.XXt_vecs
                        W = self.estimates.W
                        for p in range(len(XXt_mats)):
                            # index = W.indices[W.indptr[p]:W.indptr[p + 1]]
                            x_i = x[self.W_ind[p]]
                            XXt_mats[p] += x_i.dot(x_i.T)
                            XXt_vecs[p] += x_i.dot(x[p].T)
                # much faster: exploit that we only access CY[m, ind_pixels], hence update only these
                n0 = min_batch
                t0 = 0 * self.params.get('online', 'init_batch')
                w1 = (t - n0 + t0) * 1. / (t + t0)  # (1 - 1./t)#mbs*1. / t
                w2 = 1. / (t + t0)  # 1.*mbs /t
                ccf = np.ascontiguousarray(ccf)
                y = np.asfortranarray(y)
                for m in range(self.N):
                    self.estimates.CY[m + nb_, self.ind_A[m]] *= w1
                    self.estimates.CY[m + nb_, self.ind_A[m]] += w2 * \
                        ccf[m + nb_].dot(y[:, self.ind_A[m]])

                self.estimates.CY[:nb_] = self.estimates.CY[:nb_] * w1 + \
                    w2 * ccf[:nb_].dot(y)   # background
                self.estimates.CC = self.estimates.CC * w1 + w2 * ccf.dot(ccf.T)

        else:
            ccf = self.estimates.C_on[:self.M, t - self.params.get('online', 'minibatch_suff_stat'):t -
                                      self.params.get('online', 'minibatch_suff_stat') + 1]
            y = self.estimates.Yr_buf.get_last_frames(self.params.get('online', 'minibatch_suff_stat') + 1)[:1]
            if self.is1p:  # subtract background
                if ssub_B == 1:
                    x = (y - self.estimates.Ab.dot(ccf).T - self.estimates.b0).T
                    y -= self.estimates.W.dot(x).T
                else:
                    x = self.estimates.downscale_matrix.dot(
                        y.T - self.estimates.Ab.dot(ccf) - self.estimates.b0[:, None])
                    y -= self.estimates.upscale_matrix.dot(self.estimates.W.dot(x)).T
                y -= self.estimates.b0
                # exploit that we only access some elements of XXt, hence update only these
                if self.params.get('online', 'full_XXt'):
                    XXt = self.estimates.XXt  # alias for faster repeated look up in large loop
                    for i, idx in enumerate(self.estimates.XXt_ind):
                        XXt[i, idx] += (x[i] * x[idx]).flatten()
                else:
                    XXt_mats = self.XXt_mats
                    XXt_vecs = self.XXt_vecs
                    W = self.estimates.W
                    for p in range(len(XXt_mats)):
                        x_i = x[self.W_ind[p]]
                        XXt_mats[p] += np.outer(x_i, x_i)
                        XXt_vecs[p] += x_i.dot(x[p].T)
            # much faster: exploit that we only access CY[m, ind_pixels], hence update only these
            ccf = np.ascontiguousarray(ccf)
            y = np.asfortranarray(y)
            for m in range(self.N):
                self.estimates.CY[m + nb_, self.ind_A[m]] *= (1 - 1. / t)
                self.estimates.CY[m + nb_, self.ind_A[m]] += ccf[m +
                                                       nb_].dot(y[:, self.ind_A[m]]) / t
            self.estimates.CY[:nb_] = self.estimates.CY[:nb_] * (1 - 1. / t) + ccf[:nb_].dot(y / t)
            self.estimates.CC = self.estimates.CC * (1 - 1. / t) + ccf.dot(ccf.T / t)
        self.t_stat.append(time() - t_stat)

        # update shapes
        t_sh = time()
        if not self.params.get('online', 'dist_shape_update'):  # bulk shape update
            if ((t + 1 - self.params.get('online', 'init_batch')) %
                    self.params.get('online', 'update_freq') == 0):
                logger.info('Updating Shapes')

                if self.N > self.params.get('online', 'max_comp_update_shape'):
                    indicator_components = np.where(self.update_counter <=
                                                    self.params.get('online', 'num_times_comp_updated'))[0]
                    self.update_counter[indicator_components] += 1
                else:
                    indicator_components = None

                if self.params.get('online', 'use_dense'):
                    # update dense Ab and sparse Ab simultaneously;
                    # this is faster than calling update_shapes with sparse Ab only
                    Ab_, self.ind_A, self.estimates.Ab_dense[:, :self.M] = update_shapes(
                        self.estimates.CY, self.estimates.CC, self.estimates.Ab, self.ind_A,
                        indicator_components=indicator_components,
                        Ab_dense=self.estimates.Ab_dense[:, :self.M],
                        sn=self.estimates.sn, q=0.5, iters=self.params.get('online', 'iters_shape'))
                else:
                    Ab_, self.ind_A, _ = update_shapes(
                        self.estimates.CY, self.estimates.CC, Ab_, self.ind_A,
                        indicator_components=indicator_components, sn=self.estimates.sn,
                        q=0.5, iters=self.params.get('online', 'iters_shape'))

                self.estimates.AtA = (Ab_.T.dot(Ab_)).toarray()
                if self.is1p and ((t + 1 - self.params.get('online', 'init_batch')) %
                    (self.params.get('online', 'W_update_factor') * self.params.get('online', 'update_freq')) == 0):
                    W = self.estimates.W
                    if self.params.get('online', 'full_XXt'):
                        XXt = self.estimates.XXt  # alias for considerably faster look up in large loop
                        def process_pixel(p):
                            index = self.W_ind[p]
                            tmp = XXt[index[:, None], index]
                            tmp[np.diag_indices(len(tmp))] += np.trace(tmp) * 1e-5
                            return pd_solve(tmp, XXt[index, p])
                        if False:  # current_process().name == 'MainProcess':
                            W.data = np.concatenate(parmap(process_pixel, range(W.shape[0])))
                        else:
                            W.data = np.concatenate(list(map(process_pixel, range(W.shape[0]))))
                    else:
                        XXt_mats = self.XXt_mats
                        XXt_vecs = self.XXt_vecs
                        if self.dview is None: 
                            W.data = np.concatenate(list(map(inv_mat_vec, zip(XXt_mats, XXt_vecs))))
                        elif 'multiprocessing' in str(type(self.dview)):
                            W.data = np.concatenate(list(self.dview.imap(inv_mat_vec, zip(XXt_mats, XXt_vecs), chunksize=256)))
                        else:
                            W.data = np.concatenate(list(self.dview.map_sync(inv_mat_vec, zip(XXt_mats, XXt_vecs))))
                            self.dview.results.clear()

                    if ssub_B == 1:
                        self.estimates.Atb = Ab_.T.dot(W.dot(self.estimates.b0) - self.estimates.b0)
                        self.estimates.AtW = Ab_.T.dot(W)
                        self.estimates.AtWA = self.estimates.AtW.dot(Ab_).toarray()
                    else:
                        d1, d2 = self.estimates.dims
                        A_ds = self.estimates.downscale_matrix.dot(self.estimates.Ab)
                        self.estimates.Atb = Ab_.T.dot(self.estimates.upscale_matrix.dot(W.dot(
                            self.estimates.downscale_matrix.dot(self.estimates.b0))) - self.estimates.b0)
                        self.estimates.AtW = A_ds.T.dot(W)
                        self.estimates.AtWA = self.estimates.AtW.dot(A_ds).toarray()

                ind_zero = list(np.where(self.estimates.AtA.diagonal() < 1e-10)[0])
                if len(ind_zero) > 0:
                    ind_zero.sort()
                    ind_zero = ind_zero[::-1]
                    ind_keep = list(set(range(Ab_.shape[-1])) - set(ind_zero))
                    ind_keep.sort()

                    if self.params.get('online', 'use_dense'):
                        self.estimates.Ab_dense = np.delete(
                            self.estimates.Ab_dense, ind_zero, axis=1)
                    self.estimates.AtA = np.delete(self.estimates.AtA, ind_zero, axis=0)
                    self.estimates.AtA = np.delete(self.estimates.AtA, ind_zero, axis=1)
                    self.estimates.CY = np.delete(self.estimates.CY, ind_zero, axis=0)
                    self.estimates.CC = np.delete(self.estimates.CC, ind_zero, axis=0)
                    self.estimates.CC = np.delete(self.estimates.CC, ind_zero, axis=1)
                    self.M -= len(ind_zero)
                    self.N -= len(ind_zero)
                    self.estimates.noisyC = np.delete(self.estimates.noisyC, ind_zero, axis=0)
                    for ii in ind_zero:
                        del self.estimates.OASISinstances[ii - self.params.get('init', 'nb')]

                    self.estimates.C_on = np.delete(self.estimates.C_on, ind_zero, axis=0)
                    self.estimates.AtY_buf = np.delete(self.estimates.AtY_buf, ind_zero, axis=0)
                    Ab_ = csc_matrix(Ab_[:, ind_keep])
                    self.Ab_dense_copy = self.estimates.Ab_dense
                    self.Ab_copy = Ab_
                    self.estimates.Ab = Ab_
                    self.ind_A = list(
                        [(self.estimates.Ab.indices[self.estimates.Ab.indptr[ii]:self.estimates.Ab.indptr[ii + 1]]) for ii in range(self.params.get('init', 'nb'), self.M)])
                    self.estimates.groups = list(map(list, update_order(Ab_)[0]))

                if self.params.get('online', 'n_refit'):
                    self.estimates.AtY_buf = Ab_.T.dot(self.estimates.Yr_buf.T)

        else:  # distributed shape update
            self.update_counter *= 2**(-1. / self.params.get('online', 'update_freq'))
            if (not num_added) and (time() - t_start < 2*self.time_spend / (t - self.params.get('online', 'init_batch') + 1)):
                candidates = np.where(self.update_counter <= 1)[0]
                if len(candidates):
                    indicator_components = candidates[:self.N // mbs + 1]
                    self.comp_upd.append(len(indicator_components))
                    self.update_counter[indicator_components] += 1
                    update_bkgrd = (t % mbs == 0)
                    if self.params.get('online', 'use_dense'):
                        # update dense Ab and sparse Ab simultaneously;
                        # this is faster than calling update_shapes with sparse Ab only
                        Ab_, self.ind_A, self.estimates.Ab_dense[:, :self.M] = update_shapes(
                            self.estimates.CY, self.estimates.CC, self.estimates.Ab, self.ind_A,
                            indicator_components=indicator_components, update_bkgrd=update_bkgrd,
                            Ab_dense=self.estimates.Ab_dense[:, :self.M], sn=self.estimates.sn,
                            q=0.5, iters=self.params.get('online', 'iters_shape'))
                        if update_bkgrd:
                            self.estimates.AtA = (Ab_.T.dot(Ab_)).toarray()
                        else:
                            indicator_components += nb_
                            self.estimates.AtA[indicator_components, indicator_components[:, None]] = \
                                self.estimates.Ab_dense[:, indicator_components].T.dot(
                                self.estimates.Ab_dense[:, indicator_components])
                    else:
                        Ab_, self.ind_A, _ = update_shapes(
                            self.estimates.CY, self.estimates.CC, Ab_, self.ind_A,
                            indicator_components=indicator_components, update_bkgrd=update_bkgrd,
                            q=0.5, iters=self.params.get('online', 'iters_shape'))
                        self.estimates.AtA = (Ab_.T.dot(Ab_)).toarray()
                else:
                    self.comp_upd.append(0)
                self.estimates.Ab = Ab_
            else:
                self.comp_upd.append(0)
            self.time_spend += time() - t_start
        self.t_shapes.append(time() - t_sh)

        return self

    def initialize_online(self, model_LN=None, T=None):
        logger = logging.getLogger("caiman")
        fls = self.params.get('data', 'fnames')
        opts = self.params.get_group('online')
        Y = caiman.load(fls[0], subindices=slice(0, opts['init_batch'],
                 None), var_name_hdf5=self.params.get('data', 'var_name_hdf5')).astype(np.float32)
        if model_LN is not None:
            Y = Y - caiman.movie(np.squeeze(model_LN.predict(np.expand_dims(Y, -1))))
            Y = np.maximum(Y, 0)
        # Downsample if needed
        ds_factor = np.maximum(opts['ds_factor'], 1)
        if ds_factor > 1:
            Y = Y.resize(1./ds_factor, 1./ds_factor)
        self.estimates.shifts = []  # store motion shifts here
        self.estimates.time_new_comp = []
        if self.params.get('online', 'motion_correct'):
            mc = caiman.motion_correction.MotionCorrect(Y, dview=self.dview, **self.params.get_group('motion'))
            mc.motion_correct(save_movie=True)
            fname_new = caiman.save_memmap(mc.mmap_file, base_name='memmap_', order='C', dview=self.dview)
            Y = caiman.load(fname_new, is3D=self.params.get('motion', 'is3D'))
            if self.params.get('motion', 'pw_rigid'):
                if self.params.get('motion', 'is3D'):
                    self.estimates.shifts.extend(list(map(tuple, np.transpose([x, y, z])))
                        for (x, y, z) in zip(mc.x_shifts_els, mc.y_shifts_els, mc.z_shifts_els))
                else:
                    self.estimates.shifts.extend(list(map(tuple, np.transpose([x, y])))
                        for (x, y) in zip(mc.x_shifts_els, mc.y_shifts_els))
            else:
                self.estimates.shifts.extend(mc.shifts_rig)
            self.min_mov = mc.min_mov
        img_min = Y.min()

        if self.params.get('online', 'normalize'):
            Y = Y - img_min
        img_norm = np.std(Y, axis=0)
        img_norm += np.median(img_norm)  # normalize data to equalize the FOV
        logger.info(f'Frame size: {img_norm.shape}')
        if self.params.get('online', 'normalize'):
            Y = Y/img_norm[None, :, :]
        if opts['show_movie']:
            self.bnd_Y = np.percentile(Y,(0.001,100-0.001))
        Yr = Y.to_2D().T        # convert data into 2D array
        self.img_min = img_min
        self.img_norm = img_norm
        if self.params.get('online', 'init_method') == 'bare':
            # bare: no initialization is done
            # cnmf: runs offline CNMF using offline CNMF init params on a small portion of the movie to obtain spatial footprints of neurons which are then used to seed online CNMF
            init = self.params.get_group('init').copy()
            is1p = (init['method_init'] == 'corr_pnr' and  init['ring_size_factor'] is not None)
            if is1p:
                self.estimates.sn, psx = get_noise_fft(
                    Yr, noise_range=self.params.get('preprocess', 'noise_range'),
                    noise_method=self.params.get('preprocess', 'noise_method'),
                    max_num_samples_fft=self.params.get('preprocess', 'max_num_samples_fft'))
            for key in ('K', 'nb', 'gSig', 'method_init'):
                init.pop(key, None)
            tmp = bare_initialization(
                Y.transpose(1, 2, 0), init_batch=self.params.get('online', 'init_batch'),
                k=self.params.get('init', 'K'), gnb=self.params.get('init', 'nb'),
                method_init=self.params.get('init', 'method_init'), sn=self.estimates.sn,
                gSig=self.params.get('init', 'gSig'), return_object=False,
                options_total=self.params.to_dict(), **init)
            if is1p:
                (self.estimates.A, self.estimates.b, self.estimates.C, self.estimates.f,
                 self.estimates.YrA, self.estimates.W, self.estimates.b0) = tmp
            else:
                (self.estimates.A, self.estimates.b, self.estimates.C, self.estimates.f,
                 self.estimates.YrA) = tmp
            self.estimates.S = np.zeros_like(self.estimates.C)
            nr = self.estimates.C.shape[0]
            self.estimates.g = np.array([-np.poly([0.9] * max(self.params.get('preprocess', 'p'), 1))[1:]
                               for gg in np.ones(nr)])
            self.estimates.bl = np.zeros(nr)
            self.estimates.c1 = np.zeros(nr)
            self.estimates.neurons_sn = np.std(self.estimates.YrA, axis=-1)
            self.estimates.lam = np.zeros(nr)
        elif self.params.get('online', 'init_method') == 'cnmf':
            n_processes = cpu_count() - 1 or 1
            cnm = CNMF(n_processes=n_processes, params=self.params, dview=self.dview)
            cnm.estimates.shifts = self.estimates.shifts
            if self.params.get('patch', 'rf') is None:
                cnm.dview = None
                cnm.fit(np.array(Y))
                self.estimates = cnm.estimates

            else:
                temp_init_file = os.path.join(caiman.paths.get_tempdir(), 'init_file.hdf5')
                Y.save(temp_init_file)
                f_new = caiman.mmapping.save_memmap([temp_init_file], base_name='Yr', order='C',
                                             slices=[slice(0, opts['init_batch']), None, None])

                Yrm, dims_, T_ = caiman.mmapping.load_memmap(f_new)
                Y = np.reshape(Yrm.T, [T_] + list(dims_), order='F')
                cnm.fit(Y)
                self.estimates = cnm.estimates
                if self.params.get('online', 'normalize'):
                    self.estimates.A /= self.img_norm.reshape(-1, order='F')[:, np.newaxis]
                    self.estimates.b /= self.img_norm.reshape(-1, order='F')[:, np.newaxis]
                    self.estimates.A = csc_matrix(self.estimates.A)

        elif self.params.get('online', 'init_method') == 'seeded':
            self.estimates.A, self.estimates.b, self.estimates.C, self.estimates.f, self.estimates.YrA = seeded_initialization(
                    Y.transpose(1, 2, 0), self.estimates.A, gnb=self.params.get('init', 'nb'), k=self.params.get('init', 'K'),
                    gSig=self.params.get('init', 'gSig'), return_object=False)
            self.estimates.S = np.zeros_like(self.estimates.C)
            nr = self.estimates.C.shape[0]
            self.estimates.g = np.array([-np.poly([0.9] * max(self.params.get('preprocess', 'p'), 1))[1:]
                               for gg in np.ones(nr)])
            self.estimates.bl = np.zeros(nr)
            self.estimates.c1 = np.zeros(nr)
            self.estimates.neurons_sn = np.std(self.estimates.YrA, axis=-1)
            self.estimates.lam = np.zeros(nr)
        else:
            raise Exception('Unknown initialization method!')
        dims, Ts = caiman.base.movies.get_file_size(fls, var_name_hdf5=self.params.get('data', 'var_name_hdf5'))
        dims = Y.shape[1:]
        self.params.set('data', {'dims': dims})
        T1 = np.array(Ts).sum()*self.params.get('online', 'epochs') if T is None else T
        self._prepare_object(Yr, T1)
        if opts['show_movie']:
            self.bnd_AC = np.percentile(np.ravel(self.estimates.A.dot(self.estimates.C)),
                                        (0.001, 100-0.005))
        return self

    def save(self,filename):
        """save object in hdf5 file format

        Args:
            filename: str
                path to the hdf5 file containing the saved object
        """

        if '.hdf5' in filename:
            save_dict_to_hdf5(self.__dict__, caiman.paths.fn_relocated(filename))
        else:
            raise Exception("Unsupported file extension")

    def mc_next(self, t: int, frame: np.ndarray) -> np.ndarray:
        """
        Perform online motion correction on the next frame

        Args:
            t: int
                temporal index of the next frame to motion correct

            frame: np.ndarray
                frame to fit

        Returns:
            np.ndarray
                motion corrected frame
        """
        if self.params.motion['nonneg_movie']:
            frame = frame-self.min_mov
        frame_ = frame.flatten(order='F')
        if self.is1p and self.estimates.W is not None:
            templ = self.estimates.Ab.dot(
                np.median(self.estimates.C_on[:self.M, t-50:t], 1))            
            if self.params.get('init','ssub_B') == 1:
                B = self.estimates.W.dot(frame_ - templ - self.estimates.b0) + self.estimates.b0
            else:
                bc2 = self.estimates.downscale_matrix.dot(frame_ - templ - self.estimates.b0)
                B = self.estimates.upscale_matrix.dot(self.estimates.W.dot(bc2))
                B += self.estimates.b0
            templ += B
        else:
            templ = self.estimates.Ab.dot(self.estimates.C_on[:self.M, t-1])
        templ = templ.reshape(self.params.get('data', 'dims'), order='F')
        if self.params.get('online', 'normalize'):
            templ *= self.img_norm
        if self.is1p:
            templ = high_pass_filter_space(templ, self.params.motion['gSig_filt'])
        if self.params.get('motion', 'pw_rigid'):
            tac = tile_and_correct_3d if self.params.get('motion', 'is3D') else tile_and_correct
            frame_cor, shift, _, xy_grid = tac(
                frame, templ, self.params.motion['strides'], self.params.motion['overlaps'],
                self.params.motion['max_shifts'], newoverlaps=None, newstrides=None,
                upsample_factor_grid=self.params.motion['upsample_factor_grid'],
                upsample_factor_fft=10, show_movie=False,
                max_deviation_rigid=self.params.motion['max_deviation_rigid'], add_to_movie=0,
                shifts_opencv=True, gSig_filt=None, use_cuda=False, border_nan='copy')
        else:
            if self.is1p:
                frame_orig = frame.copy()
                frame = high_pass_filter_space(frame, self.params.motion['gSig_filt'])

            if self.params.get('motion', 'is3D'):
                # TODO: write function motion_correct_iteration_fast_3d?
                shift, sfr_freq, diffphase = register_translation_3d(
                    frame, templ, upsample_factor=10, max_shifts=self.params.motion['max_shifts'])
                frame_cor = apply_shifts_dft(
                    sfr_freq, -shift, diffphase, border_nan=self.params.motion['border_nan'])
                # register_translation[_3d] returns by how much the frame is shifted w/ respect
                #     to the template, i.e. need to apply -shifts to get the corrected frame
                # tile_and_correct[_3d] and motion_correct_iteration[_fast] return the shifts needed to apply to get the corrected frame
                shift = tuple(-np.array(shift))
            else:
                frame_cor, shift = motion_correct_iteration_fast(
                        frame, templ, *(self.params.get('online', 'max_shifts_online'),)*2)
            if self.is1p:
                M = np.float32([[1, 0, shift[1]], [0, 1, shift[0]]])
                frame_cor = cv2.warpAffine(frame_orig, M, frame.shape[::-1],
                                           flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)

        self.estimates.shifts.append(shift)
        return frame_cor


    def fit_online(self, **kwargs):
        """Implements the caiman online algorithm on the list of files fls. The
        files are taken in alpha numerical order and are assumed to each have
        the same number of frames (except the last one that can be shorter).
        Caiman online is initialized using the seeded or bare initialization
        methods.

        Args:
            fls: list
                list of files to be processed

            init_batch: int
                number of frames to be processed during initialization

            epochs: int
                number of passes over the data

            motion_correct: bool
                flag for performing motion correction

            kwargs: dict
                additional parameters used to modify self.params.online']
                see options.['online'] for details

        Returns:
            self (results of caiman online)
        """
        logger = logging.getLogger("caiman")
        self.t_init = -time()
        fls = self.params.get('data', 'fnames')
        init_batch = self.params.get('online', 'init_batch')
        if self.params.get('online', 'ring_CNN'):
            logger.info('Using Ring CNN model')
            gSig = self.params.get('init', 'gSig')[0]
            width = self.params.get('ring_CNN', 'width')
            nch = self.params.get('ring_CNN', 'n_channels')
            if self.params.get('ring_CNN', 'loss_fn') == 'pct':
                loss_fn = quantile_loss(self.params.get('ring_CNN', 'pct'))
            else:
                loss_fn = self.params.get('ring_CNN', 'loss_fn')
            if self.params.get('ring_CNN', 'lr_scheduler') is None:
                sch = None
            else:
                sch = rate_scheduler(*self.params.get('ring_CNN', 'lr_scheduler'))
            Y = caiman.base.movies.load(fls[0], subindices=slice(init_batch),
                                        var_name_hdf5=self.params.get('data', 'var_name_hdf5'))
            shape = Y.shape[1:] + (1,)
            logger.info('Starting background model training.')
            model_LN = create_LN_model(Y, shape=shape, n_channels=nch,
                                       lr=self.params.get('ring_CNN', 'lr'), gSig=gSig,
                                       loss=loss_fn, width=width,
                                       use_add=self.params.get('ring_CNN', 'use_add'),
                                       use_bias=self.params.get('ring_CNN', 'use_bias'))
            if self.params.get('ring_CNN', 'reuse_model'):
                logger.info('Using existing model from {self.params.get("ring_CNN", "path_to_model")}')
                model_LN.load_weights(self.params.get('ring_CNN', 'path_to_model'))
            else:
                logger.info('Estimating model from scratch, starting training.')
                model_LN, history, path_to_model = fit_NL_model(model_LN, Y,
                                                                epochs=self.params.get('ring_CNN', 'max_epochs'),
                                                                patience=self.params.get('ring_CNN', 'patience'),
                                                                schedule=sch)
                logger.info(f'Training complete. Model saved in {path_to_model}.')
                self.params.set('ring_CNN', {'path_to_model': path_to_model})
        else:
            model_LN = None
        epochs = self.params.get('online', 'epochs')
        self.initialize_online(model_LN=model_LN)
        self.t_init += time()
        extra_files = len(fls) - 1
        init_files = 1
        t = init_batch
        self.Ab_epoch:list = []
        t_online = []
        if extra_files == 0:     # check whether there are any additional files
            process_files = fls[:init_files]     # end processing at this file
            init_batc_iter = [init_batch]         # place where to start
        else:
            process_files = fls[:init_files + extra_files]   # additional files
            # where to start reading at each file
            init_batc_iter = [init_batch] + [0]*extra_files
        if self.params.get('online', 'save_online_movie') + self.params.get('online', 'show_movie'):
            resize_fact = 2
            fourcc = cv2.VideoWriter_fourcc(*self.params.get('online', 'opencv_codec'))
            out = cv2.VideoWriter(self.params.get('online', 'movie_name_online'),
                                  fourcc, 30, tuple([int(resize_fact*2*x) for x in self.params.get('data', 'dims')]),
                                  True)
        # Iterate through the epochs
        for iter in range(epochs):
            if iter == epochs - 1 and self.params.get('online', 'stop_detection'):
                self.params.set('online', {'update_num_comps': False})
            logger.info(f"Searching for new components set to: {self.params.get('online', 'update_num_comps')}")
            if iter > 0:
                # if not on first epoch process all files from scratch
                process_files = fls[:init_files + extra_files]
                init_batc_iter = [0] * (extra_files + init_files)

            # Go through all files
            # TODO Use better variable names
            for file_count, ffll in enumerate(process_files):
                logger.warning(f'Now processing file {ffll}')
                Y_ = caiman.base.movies.load_iter(
                    ffll, var_name_hdf5=self.params.get('data', 'var_name_hdf5'),
                    subindices=slice(init_batc_iter[file_count], None, None))

                old_comps = self.N     # number of existing components
                frame_count = -1
                while True:   # process each file
                    try:
                        frame = next(Y_)
                        if model_LN is not None:
                            if self.params.get('ring_CNN', 'remove_activity'):
                                activity = self.estimates.Ab[:,:self.N].dot(self.estimates.C_on[:self.N, t-1]).reshape(self.params.get('data', 'dims'), order='F')
                                if self.params.get('online', 'normalize'):
                                    activity *= self.img_norm
                            else:
                                activity = 0.
#                                frame = frame.astype(np.float32) - activity
                            frame = frame - np.squeeze(model_LN.predict(np.expand_dims(np.expand_dims(frame.astype(np.float32) - activity, 0), -1)))
                            frame = np.maximum(frame, 0)
                        frame_count += 1
                        t_frame_start = time()
                        if np.isnan(np.sum(frame)):
                            raise Exception('Frame ' + str(frame_count) +
                                            ' contains NaN')
                        if t % 500 == 0:
                            logger.info(f'Epoch: {iter + 1}. {t}' +
                                         ' frames have been processed in total. ' +
                                         f'{self.N - old_comps} new components were added. Total # of components is '
                                         + str(self.estimates.Ab.shape[-1] - self.params.get('init', 'nb')))
                            old_comps = self.N

                        # Downsample and normalize
                        frame_ = frame.copy().astype(np.float32)
                        if self.params.get('online', 'ds_factor') > 1:
                            frame_ = cv2.resize(frame_, self.img_norm.shape[::-1])

                        if self.params.get('online', 'normalize'):
                            frame_ -= self.img_min     # make data non-negative

                        # Motion Correction
                        t_mot = time()
                        if self.params.get('online', 'motion_correct'):    # motion correct
                            frame_cor = self.mc_next(t, frame_)
                        else:
                            templ = None
                            frame_cor = frame_
                        self.t_motion.append(time() - t_mot)
                        
                        if self.params.get('online', 'normalize'):
                            frame_cor = frame_cor/self.img_norm
                        # Fit next frame
                        self.fit_next(t, frame_cor.reshape(-1, order='F'))
                        # Show
                        if self.params.get('online', 'show_movie'):
                            self.t = t
                            vid_frame = self.create_frame(frame_cor, resize_fact=resize_fact)
                            if self.params.get('online', 'save_online_movie'):
                                out.write(vid_frame)
                                for rp in range(len(self.estimates.ind_new)*2):
                                    out.write(vid_frame)

                            cv2.imshow('frame', vid_frame)
                            for rp in range(len(self.estimates.ind_new)*2):
                                cv2.imshow('frame', vid_frame)
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                break
                        t += 1
                        t_online.append(time() - t_frame_start)
                    except  (StopIteration, RuntimeError):
                        break
        
            self.Ab_epoch.append(self.estimates.Ab.copy())

        if self.params.get('online', 'normalize'):
            self.estimates.Ab = csc_matrix(self.estimates.Ab.multiply(
                self.img_norm.reshape(-1, order='F')[:, np.newaxis]))
        self.estimates.A, self.estimates.b = self.estimates.Ab[:, self.params.get('init', 'nb'):], self.estimates.Ab[:, :self.params.get('init', 'nb')].toarray()
        self.estimates.C, self.estimates.f = self.estimates.C_on[self.params.get('init', 'nb'):self.M, t - t //
                         epochs:t], self.estimates.C_on[:self.params.get('init', 'nb'), t - t // epochs:t]
        noisyC = self.estimates.noisyC[self.params.get('init', 'nb'):self.M, t - t // epochs:t]
        self.estimates.YrA = noisyC - self.estimates.C
        if self.estimates.OASISinstances is not None:
            self.estimates.bl = [osi.b for osi in self.estimates.OASISinstances]
            self.estimates.S = np.stack([osi.s for osi in self.estimates.OASISinstances])
            self.estimates.S = self.estimates.S[:, t - t // epochs:t]
        else:
            self.estimates.bl = [0] * self.estimates.C.shape[0]
            self.estimates.S = np.zeros_like(self.estimates.C)
        if self.params.get('online', 'ds_factor') > 1:
            dims = frame.shape
            self.estimates.A = hstack([coo_matrix(cv2.resize(self.estimates.A[:, i].reshape(self.estimates.dims, order='F').toarray(),
                                                            dims[::-1]).reshape(-1, order='F')[:,None]) for i in range(self.N)], format='csc')
            if self.estimates.b.shape[-1] > 0:
                self.estimates.b = np.concatenate([cv2.resize(self.estimates.b[:, i].reshape(self.estimates.dims, order='F'),
                                                              dims[::-1]).reshape(-1, order='F')[:,None] for i in range(self.params.get('init', 'nb'))], axis=1)
            else:
                self.estimates.b = np.resize(self.estimates.b, (self.estimates.A.shape[0], 0))
            if self.estimates.b0 is not None:
                b0 = self.estimates.b0.reshape(self.estimates.dims, order='F')
                b0 = cv2.resize(b0, dims[::-1])
                self.estimates.b0 = b0.reshape((-1, 1), order='F')
            self.params.set('data', {'dims': dims})
            self.estimates.dims = dims
        if self.params.get('online', 'save_online_movie'):
            out.release()
        if self.params.get('online', 'show_movie'):
            cv2.destroyAllWindows()
        self.t_online = t_online
        self.estimates.C_on = self.estimates.C_on[:self.M]
        self.estimates.noisyC = self.estimates.noisyC[:self.M]

        return self

    def create_frame(self, frame_cor, show_residuals=True, resize_fact=3, transpose=True):
        if show_residuals:
            caption = 'Corr*PSNR buffer' if self.params.get('online', 'use_corr_img') else 'Mean Residual Buffer'
        else:
            caption = 'Identified Components'
        captions = ['Raw Data', 'Inferred Activity', caption, 'Denoised Data']
        self.dims = self.estimates.dims
        self.captions = captions
        est = self.estimates
        gnb = self.M - self.N
        A, b = est.Ab[:, gnb:], est.Ab[:, :gnb].toarray()
        C, f = est.C_on[gnb:self.M, :], est.C_on[:gnb, :]
        # inferred activity due to components (no background)
        frame_plot = (frame_cor.copy() - self.bnd_Y[0])/np.diff(self.bnd_Y)
        comps_frame = A.dot(C[:, self.t - 1]).reshape(self.dims, order='F')
        if self.is1p:
            ssub_B = self.params.get('init', 'ssub_B') * self.params.get('init', 'ssub')
            if ssub_B == 1:
                B = self.estimates.W.dot((frame_cor - comps_frame).flatten(order='F') - self.estimates.b0) + self.estimates.b0
                bgkrnd_frame = B.reshape(self.dims, order='F')
            else:
                bc2 = self.estimates.downscale_matrix.dot(
                    (frame_cor - comps_frame).flatten(order='F') - self.estimates.b0)
                bgkrnd_frame = (self.estimates.b0 + self.estimates.upscale_matrix.dot(
                    self.estimates.W.dot(bc2))).reshape(self.dims, order='F')
        else:
            bgkrnd_frame = b.dot(f[:, self.t - 1]).reshape(self.dims, order='F')  # denoised frame (components + background)
        denoised_frame = comps_frame + bgkrnd_frame
        denoised_frame = (denoised_frame.copy() - self.bnd_Y[0])/np.diff(self.bnd_Y)
        comps_frame = (comps_frame.copy() - self.bnd_AC[0])/np.diff(self.bnd_AC)

        if show_residuals:
            if self.params.get('online', 'use_corr_img'):
                pnr_img = est.max_img / est.sn.reshape(est.dims, order='F')
                pnr_img[pnr_img<2] = 0
                all_comps = np.nan_to_num(est.corr_img * pnr_img)
                fac = 1. / self.params.get('init', 'min_corr') / self.params.get('init', 'min_pnr')
            else:
                #all_comps = np.reshape(self.Yres_buf.mean(0), self.dims, order='F')
                all_comps = np.reshape(est.mean_buff, self.dims, order='F')
                fac = 1. / np.percentile(est.mean_buff, 99.995)
        else:
            all_comps = np.array(A.sum(-1)).reshape(self.dims, order='F')
            fac = 2
        #all_comps = (all_comps.copy() - self.bnd_Y[0])/np.diff(self.bnd_Y)
        all_comps = np.minimum(np.maximum(all_comps, 0)*fac, 1)
                                                  # spatial shapes
        frame_comp_1 = cv2.resize(np.concatenate([frame_plot, all_comps * 1.], axis=-1),
                                  (2 * int(self.dims[1] * resize_fact), int(self.dims[0] * resize_fact)))
        frame_comp_2 = cv2.resize(np.concatenate([comps_frame, denoised_frame], axis=-1), 
                                  (2 * int(self.dims[1] * resize_fact), int(self.dims[0] * resize_fact)))
        frame_pn = np.concatenate([frame_comp_1, frame_comp_2], axis=0).T
        if transpose:
            self.dims = self.dims[::-1]
            frame_pn = frame_pn.T

        vid_frame = np.repeat(frame_pn[:, :, None], 3, axis=-1)
        vid_frame = np.minimum((vid_frame * 255.), 255).astype('u1')

        #if show_residuals and est.ind_new:
        if est.ind_new:
            add_v = int(self.dims[1-transpose]*resize_fact)
            for ind_new in est.ind_new:
                cv2.rectangle(vid_frame,(int(ind_new[transpose][1]*resize_fact) + transpose*add_v,
                                         int(ind_new[1-transpose][1]*resize_fact) + (1-transpose)*add_v),
                                             (int(ind_new[transpose][0]*resize_fact) + transpose*add_v,
                                              int(ind_new[1-transpose][0]*resize_fact)+ (1-transpose)*add_v), (255,0,255), 2)

        cv2.putText(vid_frame, captions[0], (5, 20), fontFace=5, fontScale=0.8, color=(
            0, 255, 0), thickness=1)
        cv2.putText(vid_frame, captions[1+transpose], (int(
            self.dims[0] * resize_fact) + 5, 20), fontFace=5, fontScale=0.8, color=(0, 255, 0), thickness=1)
        cv2.putText(vid_frame, captions[2-transpose], (5, int(
            self.dims[1] * resize_fact) + 20), fontFace=5, fontScale=0.8, color=(0, 255, 0), thickness=1)
        cv2.putText(vid_frame, captions[3], (int(self.dims[0] * resize_fact) + 5, int(
            self.dims[1] * resize_fact) + 20), fontFace=5, fontScale=0.8, color=(0, 255, 0), thickness=1)
        cv2.putText(vid_frame, 'Frame = ' + str(self.t), (vid_frame.shape[1] // 2 - vid_frame.shape[1] //
                                                     10, vid_frame.shape[0] - 20), fontFace=5, fontScale=0.8, color=(0, 255, 255), thickness=1)
        if transpose:
            self.dims = self.dims[::-1]
        return vid_frame

def bare_initialization(Y, init_batch=1000, k=1, method_init='greedy_roi', gnb=1,
                        gSig=[5, 5], motion_flag=False, p=1,
                        return_object=True, **kwargs):
    """
    Quick and dirty initialization for OnACID, bypassing CNMF entirely
    Args:
        Y               movie object or np.array
                        matrix of data

        init_batch      int
                        number of frames to process

        method_init     string
                        initialization method

        k               int
                        number of components to find

        gnb             int
                        number of background components

        gSig            [int,int]
                        half-size of component

        motion_flag     bool
                        also perform motion correction

    Output:
        cnm_init    object
                    caiman CNMF-like object to initialize OnACID
    """

    if Y.ndim == 4:  # 3D data
        Y = Y[:, :, :, :init_batch]
    else:
        Y = Y[:, :, :init_batch]

    try:
        Ain, Cin, b_in, f_in, center = initialize_components(
            Y, K=k, gSig=gSig, nb=gnb, method_init=method_init, **kwargs)
        Ain = coo_matrix(Ain)
        b_in = np.array(b_in)
        Yr = np.reshape(Y, (Ain.shape[0], Y.shape[-1]), order='F')
        nA = (Ain.power(2).sum(axis=0))
        nr = nA.size

        YA = spdiags(1./nA, 0, nr, nr) * \
            (Ain.T.dot(Yr) - (Ain.T.dot(b_in)).dot(f_in))
        AA = spdiags(1./nA, 0, nr, nr) * (Ain.T.dot(Ain))
        YrA = YA - AA.T.dot(Cin)
    except ValueError:
        Ain, Cin, b_in, f_in, center, extra_1p = initialize_components(
            Y, K=k, gSig=gSig, nb=gnb, method_init=method_init, **kwargs)
        Ain = coo_matrix(Ain)
        YrA, _, W, b0 = extra_1p[-4:]
    if return_object:
        cnm_init = caiman.source_extraction.cnmf.cnmf.CNMF(2, k=k, gSig=gSig, Ain=Ain, Cin=Cin, b_in=np.array(
            b_in), f_in=f_in, method_init=method_init, p=p, gnb=gnb, **kwargs)

        cnm_init.estimates.A, cnm_init.estimates.C, cnm_init.estimates.b, cnm_init.estimates.f, cnm_init.estimates.S,\
            cnm_init.estimates.YrA = Ain, Cin, b_in, f_in, np.maximum(np.atleast_2d(Cin), 0), YrA

        #cnm_init.g = np.array([-np.poly([0.9]*max(p,1))[1:] for gg in np.ones(k)])
        cnm_init.estimates.g = np.array([-np.poly([0.9, 0.5][:max(1, p)])[1:]
                               for gg in np.ones(k)])
        cnm_init.estimates.bl = np.zeros(k)
        cnm_init.estimates.c1 = np.zeros(k)
        cnm_init.estimates.neurons_sn = np.std(YrA, axis=-1)
        cnm_init.estimates.lam = np.zeros(k)
        cnm_init.dims = Y.shape[:-1]
        cnm_init.params.set('online', {'init_batch': init_batch})

        return cnm_init
    else:
        try:
            return Ain, np.array(b_in), Cin, f_in, YrA, W, b0
        except:
            return Ain, np.array(b_in), Cin, f_in, YrA

def seeded_initialization(Y, Ain, dims=None, init_batch=1000, order_init=None, gnb=1, p=1,
                          return_object=True, **kwargs):

    """
    Initialization for OnACID based on a set of user given binary masks.
    Args:
        Y               movie object or np.array
                        matrix of data

        Ain             bool np.array
                        2d np.array with binary masks

        dims            tuple
                        dimensions of FOV

        init_batch      int
                        number of frames to process

        gnb             int
                        number of background components

        order_init:     list
                        order of elements to be initialized using rank1 nmf restricted to the support of
                        each component

    Output:
        cnm_init    object
                    caiman CNMF-like object to initialize OnACID
    """

    if 'ndarray' not in str(type(Ain)):
        Ain = Ain.toarray()

    if dims is None:
        dims = Y.shape[:-1]
    px = (np.sum(Ain > 0, axis=1) > 0)
    not_px = 1 - px
    if 'matrix' in str(type(not_px)):
        not_px = np.array(not_px).flatten()
    Yr = np.reshape(Y, (Ain.shape[0], Y.shape[-1]), order='F')
    model = NMF(n_components=gnb, init='nndsvdar', max_iter=10)
    _ = model.fit_transform(np.maximum(Yr[not_px], 0)) # Done to update the model object
    f_in = model.components_.squeeze()
    f_in = np.atleast_2d(f_in)
    Y_resf = np.dot(Yr, f_in.T)
#    b_in = np.maximum(Y_resf.dot(np.linalg.inv(f_in.dot(f_in.T))), 0)
    b_in = np.maximum(np.linalg.solve(f_in.dot(f_in.T), Y_resf.T), 0).T
    # b_in = np.maximum(pd_solve(f_in.dot(f_in.T), Y_resf.T), 0).T
    Yr_no_bg = (Yr - b_in.dot(f_in)).astype(np.float32)

    Cin = np.zeros([Ain.shape[-1],Yr.shape[-1]], dtype = np.float32)
    if order_init is not None: #initialize using rank-1 nmf for each component

        model_comp = NMF(n_components=1, init='nndsvdar', max_iter=50)
        for count, idx_in in enumerate(order_init):
            if count % 10 == 0:
                print(count)
            idx_domain = np.where(Ain[:,idx_in])[0]
            Ain[idx_domain,idx_in] = model_comp.fit_transform(\
                                   np.maximum(Yr_no_bg[idx_domain], 0)).squeeze()
            Cin[idx_in] = model_comp.components_.squeeze()
            Yr_no_bg[idx_domain] -= np.outer(Ain[idx_domain, idx_in],Cin[idx_in])
    else:
        Ain = normalize(Ain.astype('float32'), axis=0, norm='l1')
        Cin = np.maximum(Ain.T.dot(Yr) - (Ain.T.dot(b_in)).dot(f_in), 0)
        Ain = HALS4shapes(Yr_no_bg, Ain, Cin, iters=5)

    Ain, Cin, b_in, f_in = hals(Yr, Ain, Cin, b_in, f_in, maxIter=8, bSiz=None)
    Ain = csc_matrix(Ain)
    nA = (Ain.power(2).sum(axis=0))
    nr = nA.size

    YA = spdiags(1./nA, 0, nr, nr) * \
        (Ain.T.dot(Yr) - (Ain.T.dot(b_in)).dot(f_in))
    AA = spdiags(1./nA, 0, nr, nr) * (Ain.T.dot(Ain))
    YrA = YA - AA.T.dot(Cin)
    if return_object:
        cnm_init = caiman.source_extraction.cnmf.cnmf.CNMF(
            2, Ain=Ain, Cin=Cin, b_in=np.array(b_in), f_in=f_in, p=1, **kwargs)
        cnm_init.estimates.A, cnm_init.estimates.C, cnm_init.estimates.b, cnm_init.estimates.f, cnm_init.estimates.S, \
                cnm_init.estimates.YrA = Ain, Cin, b_in, f_in, np.fmax(np.atleast_2d(Cin), 0), YrA
    #    cnm_init.g = np.array([[gg] for gg in np.ones(nr)*0.9])
        cnm_init.estimates.g = np.array([-np.poly([0.9] * max(p, 1))[1:]
                               for gg in np.ones(nr)])
        cnm_init.estimates.bl = np.zeros(nr)
        cnm_init.estimates.c1 = np.zeros(nr)
        cnm_init.estimates.neurons_sn = np.std(YrA, axis=-1)
        cnm_init.estimates.lam = np.zeros(nr)
        cnm_init.dims = Y.shape[:-1]
        cnm_init.params.set('online', {'init_batch': init_batch})

        return cnm_init
    else:
        return Ain, np.array(b_in), Cin, f_in, YrA


def HALS4shapes(Yr, A, C, iters=2):
    K = A.shape[-1]
    ind_A = A > 0
    U = C.dot(Yr.T)
    V = C.dot(C.T)
    V_diag = V.diagonal() + np.finfo(float).eps
    for _ in range(iters):
        for m in range(K):  # neurons
            ind_pixels = np.squeeze(ind_A[:, m])
            A[ind_pixels, m] = np.clip(A[ind_pixels, m] +
                                       ((U[m, ind_pixels] - V[m].dot(A[ind_pixels].T)) /
                                        V_diag[m]), 0, np.inf)

    return A


# definitions for demixed time series extraction and denoising/deconvolving
@profile
def HALS4activity(Yr, A, noisyC, AtA=None, iters=5, tol=1e-3, groups=None,
                  order=None):
    """Solves C = argmin_C ||Yr-AC|| using block-coordinate decent. Can use
    groups to update non-overlapping components in parallel or a specified
    order.

    Args:
        Yr : np.array (possibly memory mapped, (x,y,[,z]) x t)
            Imaging data reshaped in matrix format

        A : scipy.sparse.csc_matrix (or np.array) (x,y,[,z]) x # of components)
            Spatial components and background

        noisyC : np.array  (# of components x t)
            Temporal traces (including residuals plus background)

        AtA : np.array, optional (# of components x # of components)
            A.T.dot(A) Overlap matrix of shapes A.

        iters : int, optional
            Maximum number of iterations.

        tol : float, optional
            Change tolerance level

        groups : list of sets
            grouped components to be updated simultaneously

        order : list
            Update components in that order (used if nonempty and groups=None)

    Returns:
        C : np.array (# of components x t)
            solution of HALS

        noisyC : np.array (# of components x t)
            solution of HALS + residuals, i.e, (C + YrA)
    """

    AtY = A.T.dot(Yr)
    num_iters = 0
    C_old = np.zeros_like(noisyC)
    C = noisyC.copy()
    if AtA is None:
        AtA = A.T.dot(A)
    AtAd = AtA.diagonal() + np.finfo(np.float32).eps

    # faster than np.linalg.norm
    def norm(c): return sqrt(c.ravel().dot(c.ravel()))
    while (norm(C_old - C) >= tol * norm(C_old)) and (num_iters < iters):
        C_old[:] = C
        if groups is None:
            if order is None:
                order = list(range(AtY.shape[0]))
            for m in order:
                noisyC[m] = C[m] + (AtY[m] - AtA[m].dot(C)) / AtAd[m]
                C[m] = np.maximum(noisyC[m], 0)
        else:
            for m in groups:
                noisyC[m] = C[m] + ((AtY[m] - AtA[m].dot(C)).T/AtAd[m]).T
                C[m] = np.maximum(noisyC[m], 0)
        num_iters += 1
    return C, noisyC


def demix1p(y, A, noisyC, AtA, Atb, AtW, AtWA, iters=5, tol=1e-3,
            groups=None, downscale_matrix=None, ssub_B=1):
    """
    Solve C = argmin_C ||Yr-AC-B|| using block-coordinate decent
    where B = W(Y-AC-b0) + b0  (ring model for 1p data)
    Parameters
    ----------
    y : array of float, shape (x*y[*z],)
        flattened array of raw data frame
    A : sparse matrix of float
        neural shapes
    noisyC : ndarray of float
        Initial value of fluorescence intensities.
    AtA : ndarray of float
        Overlap matrix of shapes A.
    Atb : ndarray of float
        Projection of constant background terms on shapes A: A'(Wb0-b0)
    AtW : sparse matrix of float, shape (x*y, x*y)
        Projection of ring matrix W on shapes A
    AtWA : ndarray of float
        A'*W*A
    iters : int, optional
        Maximal number of iterations.
    tol : float, optional
        Tolerance.
    groups: list of lists
        groups of components to update in parallel
    """
    AtY = A.T.dot(y)
    AtWyb = AtW.dot(y if ssub_B == 1 else downscale_matrix.dot(y) *
                    ssub_B**2) - Atb  # Atb is A'(Wb0-b0)
    num_iters = 0
    C_old = np.zeros_like(noisyC)
    C = noisyC.copy()
    # faster than np.linalg.norm
    def norm(c): return sqrt(c.ravel().dot(c.ravel()))
    while (norm(C_old - C) >= tol * norm(C_old)) and (num_iters < iters):
        C_old[:] = C
        AtB = AtWyb - AtWA.dot(C)  # A'B = A'WY - A'WAC - A'(Wb0-b0)
        if groups is None:
            for m in range(len(AtY)):
                noisyC[m] = C[m] + (AtY[m] - AtA[m].dot(C) - AtB[m]) / (AtA[m, m] + np.finfo(C.dtype).eps)
                C[m] = max(noisyC[m], 0)
        else:
            for m in groups:
                noisyC[m] = C[m] + (AtY[m] - AtA[m].dot(C) - AtB[m]) / (AtA.diagonal()[m] + np.finfo(C.dtype).eps)
                C[m] = np.maximum(noisyC[m], 0)
        num_iters += 1
    return C, noisyC


@profile
def demix_and_deconvolve(C, noisyC, AtY, AtA, OASISinstances, iters=3, n_refit=0):
    """
    Solve C = argmin_C ||Y-AC|| subject to C following AR(p) dynamics
    using OASIS within block-coordinate decent
    Newly fits the last elements in buffers C and AtY and possibly refits
    earlier elements.

    Args:
        C : ndarray of float
            Buffer containing the denoised fluorescence intensities.
            All elements up to and excluding the last one have been denoised in
            earlier calls.
        noisyC : ndarray of float
            Buffer containing the undenoised fluorescence intensities.
        AtY : ndarray of float
            Buffer containing the projections of data Y on shapes A.
        AtA : ndarray of float
            Overlap matrix of shapes A.
        OASISinstances : list of OASIS objects
            Objects for deconvolution and denoising
        iters : int, optional
            Number of iterations.
        n_refit : int, optional
            Number of previous OASIS pools to refit
            0 fits only last pool, np.inf all pools fully (i.e. starting) within buffer
    """
    AtA += np.finfo(float).eps
    T = OASISinstances[0].t + 1
    len_buffer = C.shape[1]
    nb = AtY.shape[0] - len(OASISinstances)
    if n_refit == 0:
        for i in range(iters):
            for m in range(AtY.shape[0]):
                noisyC[m, -1] = C[m, -1] + \
                    (AtY[m, -1] - AtA[m].dot(C[:, -1])) / AtA[m, m]
                if m >= nb and i > 0:
                    n = m - nb
                    if i == iters - 1:  # commit
                        OASISinstances[n].fit_next(noisyC[m, -1])
                        l = OASISinstances[n].get_l_of_last_pool()
                        if l < len_buffer:
                            C[m, -l:] = OASISinstances[n].get_c_of_last_pool()
                        else:
                            C[m] = OASISinstances[n].get_c(len_buffer)
                    else:  # temporary non-commited update of most recent frame
                        C[m] = OASISinstances[n].fit_next_tmp(
                            noisyC[m, -1], len_buffer)
                else:
                    # no need to enforce max(c, 0) for background, is it?
                    C[m, -1] = np.maximum(noisyC[m, -1], 0)
    else:
        # !threshold .1 assumes normalized A (|A|_2=1)
        overlap = np.sum(AtA[nb:, nb:] > .1, 0) > 1

        def refit(o, c):
            # remove last pools
            tmp = 0
            while tmp < n_refit and o.t - o.get_l_of_last_pool() > T - len_buffer:
                o.remove_last_pool()
                tmp += 1
            # refit last pools
            for cc in c[o.t - T + len_buffer:-1]:
                o.fit_next(cc)
        for i in range(iters):
            for m in range(AtY.shape[0]):
                noisyC[m] = C[m] + (AtY[m] - AtA[m].dot(C)) / AtA[m, m]
                if m >= nb:
                    n = m - nb
                    if overlap[n]:
                        refit(OASISinstances[n], noisyC[m])
                    if i == iters - 1:  # commit
                        OASISinstances[n].fit_next(noisyC[m, -1])
                        C[m] = OASISinstances[n].get_c(len_buffer)
                    else:  # temporary non-commited update of most recent frame
                        C[m] = OASISinstances[n].fit_next_tmp(
                            noisyC[m, -1], len_buffer)
                else:
                    # no need to enforce max(c, 0) for background, is it?
                    C[m] = noisyC[m]
    return C, noisyC, OASISinstances


# Estimate shapes on small initial batch
def init_shapes_and_sufficient_stats(Y, A, C, b, f, W=None, b0=None, ssub_B=1, bSiz=3,
                                     downscale_matrix=None, upscale_matrix=None):
    # smooth the components
    dims, T = np.shape(Y)[:-1], np.shape(Y)[-1]
    K = A.shape[1]  # number of neurons
    if W is None:
        nb = b.shape[1]  # number of background components
    # if isinstance(bSiz, (int, float)):
    #     bSiz = [bSiz] * len(dims)
        Ab = np.hstack([b, A])
    else:
        Ab = A
        nb = 0
    # Ab = scipy.sparse.hstack([A.astype('float32'), b.astype('float32')]).tocsc()  might be faster
    # closing of shapes to not have holes in index matrix ind_A.
    # do this somehow smarter & faster, e.g. smooth only within patch !!
    A_smooth = np.transpose([gaussian_filter(np.array(a).reshape(
        dims, order='F'), 0).ravel(order='F') for a in Ab.T])
    A_smooth[A_smooth < 1e-2] = 0
    # set literal zero values of Ab to small values, s.t. ind_A and Ab.indptr match
    Ab += 1e-6 * A_smooth
    Ab = csc_matrix(Ab)
    ind_A = [Ab.indices[Ab.indptr[m]:Ab.indptr[m + 1]]
             for m in range(nb, nb + K)]
    Cf = np.r_[f.reshape(nb, -1), C] if f.size else C
    CY = Cf.dot(np.reshape(Y, (np.prod(dims), T), order='F').T)
    if W is not None:
        if ssub_B == 1:
            CY -= Cf.dot(W.dot(np.reshape(Y, (-1, T), order='F') -
                               A.dot(C) - b0[:, None]).T + b0)
        else:
            d1, d2 = dims
            B = upscale_matrix.dot(W.dot(downscale_matrix.dot(
                np.reshape(Y, (-1, T), order='F') - A.dot(C) - b0[:, None]))).T + b0
            CY -= Cf.dot(B)
    CC = Cf.dot(Cf.T)
    return Ab, ind_A, CY, CC


@profile
def update_shapes(CY, CC, Ab, ind_A, sn=None, q=0.5, indicator_components=None,
                  Ab_dense=None, update_bkgrd=True, iters=5):

    D, M = Ab.shape
    N = len(ind_A)
    nb = M - N
    if indicator_components is None:
        idx_comp = range(nb, M)
    else:
        idx_comp = np.where(indicator_components)[0] + nb
    if sn is None or q == 0.5:  # avoid costly construction of L=np.zeros((M, D), dtype=np.float32)
        for _ in range(iters):  # it's presumably better to run just 1 iter but update more neurons
            if Ab_dense is None:
                for m in idx_comp:  # neurons
                    ind_pixels = ind_A[m - nb]
                    tmp = np.maximum(Ab.data[Ab.indptr[m]:Ab.indptr[m + 1]] +
                        ((CY[m, ind_pixels] - Ab.dot(CC[m])[ind_pixels]) / (CC[m, m] + np.finfo(CC.dtype).eps)), 0)
                    # normalize
                    if tmp.dot(tmp) > 0:
                        tmp *= 1e-3 / \
                            min(1e-3, sqrt(tmp.dot(tmp)) + np.finfo(float).eps)
                        tmp = tmp / max(1, sqrt(tmp.dot(tmp)))
                        Ab.data[Ab.indptr[m]:Ab.indptr[m + 1]] = tmp
                        ind_A[m - nb] = Ab.indices[slice(Ab.indptr[m], Ab.indptr[m + 1])]
            else:
                for m in idx_comp:  # neurons
                    ind_pixels = ind_A[m - nb]
                    tmp = np.maximum(Ab_dense[ind_pixels, m] + 
                        ((CY[m, ind_pixels] - Ab_dense[ind_pixels].dot(CC[m])) / (CC[m, m] + np.finfo(CC.dtype).eps)), 0)
                    # normalize
                    if tmp.dot(tmp) > 0:
                        tmp *= 1e-3 / \
                            min(1e-3, sqrt(tmp.dot(tmp)) + np.finfo(float).eps)
                        Ab_dense[ind_pixels, m] = tmp / max(1, sqrt(tmp.dot(tmp)))
                        Ab.data[Ab.indptr[m]:Ab.indptr[m + 1]] = Ab_dense[ind_pixels, m]
                        ind_A[m - nb] = Ab.indices[slice(Ab.indptr[m], Ab.indptr[m + 1])]
            if update_bkgrd:
                for m in range(nb):  # background
                    sl = slice(Ab.indptr[m], Ab.indptr[m + 1])
                    ind_pixels = Ab.indices[sl]
                    Ab.data[sl] = np.maximum(
                        Ab.data[sl] + ((CY[m, ind_pixels] - Ab.dot(CC[m])[ind_pixels]) / (CC[m, m] + np.finfo(CC.dtype).eps)), 0)
                    if Ab_dense is not None:
                        Ab_dense[ind_pixels, m] = Ab.data[sl]
    else:
        L = norm.ppf(q)*np.outer(np.sqrt(CC.diagonal()), sn)
        L[:nb] = 0
        for _ in range(iters):  # it's presumably better to run just 1 iter but update more neurons
            if Ab_dense is None:
                for m in idx_comp:  # neurons
                    ind_pixels = ind_A[m - nb]
                    tmp = np.maximum(Ab.data[Ab.indptr[m]:Ab.indptr[m + 1]] +
                        ((CY[m, ind_pixels] - L[m, ind_pixels] - Ab.dot(CC[m])[ind_pixels]) / (CC[m, m] + np.finfo(CC.dtype).eps)), 0)

                    if tmp.dot(tmp) > 0:
                        tmp *= 1e-3 / \
                            min(1e-3, sqrt(tmp.dot(tmp)) + np.finfo(float).eps)
                        tmp = tmp / max(1, sqrt(tmp.dot(tmp)))
                        Ab.data[Ab.indptr[m]:Ab.indptr[m + 1]] = tmp

                        ind_A[m -
                              nb] = Ab.indices[slice(Ab.indptr[m], Ab.indptr[m + 1])]
                    # N.B. Ab[ind_pixels].dot(CC[m]) is slower for csc matrix due to indexing rows
            else:
                for m in idx_comp:  # neurons
                    ind_pixels = ind_A[m - nb]
                    tmp = np.maximum(Ab_dense[ind_pixels, m] + ((CY[m, ind_pixels] - L[m, ind_pixels] -
                                                                 Ab_dense[ind_pixels].dot(CC[m])) /
                                                                (CC[m, m] + np.finfo(CC.dtype).eps)), 0)
                    # normalize
                    if tmp.dot(tmp) > 0:
                        tmp *= 1e-3 / \
                            min(1e-3, sqrt(tmp.dot(tmp)) + np.finfo(float).eps)
                        Ab_dense[ind_pixels, m] = tmp / max(1, sqrt(tmp.dot(tmp)))
                        Ab.data[Ab.indptr[m]:Ab.indptr[m + 1]] = Ab_dense[ind_pixels, m]
                        ind_A[m - nb] = Ab.indices[slice(Ab.indptr[m], Ab.indptr[m + 1])]
                # Ab.data[Ab.indptr[nb]:] = np.concatenate(
                #     [Ab_dense[ind_A[m - nb], m] for m in range(nb, M)])
                # N.B. why does selecting only overlapping neurons help surprisingly little, i.e
                # Ab[ind_pixels][:, overlap[m]].dot(CC[overlap[m], m])
                # where overlap[m] are the indices of all neurons overlappping with & including m?
                # sparsify ??
            if update_bkgrd:
                for m in range(nb):  # background
                    sl = slice(Ab.indptr[m], Ab.indptr[m + 1])
                    ind_pixels = Ab.indices[sl]
                    Ab.data[sl] = np.maximum(
                        Ab.data[sl] + ((CY[m, ind_pixels] - Ab.dot(CC[m])[ind_pixels]) / (CC[m, m] + np.finfo(CC.dtype).eps)), 0)
                    if Ab_dense is not None:
                        Ab_dense[ind_pixels, m] = Ab.data[sl]

    return Ab, ind_A, Ab_dense


class RingBuffer(np.ndarray):
    """ implements ring buffer efficiently"""

    def __new__(cls, input_array, num_els):
        obj = np.asarray(input_array).view(cls)
        obj.max_ = num_els
        obj.cur = 0
        if input_array.shape[0] != num_els:
            print([input_array.shape[0], num_els])
            raise Exception('The first dimension should equal num_els')

        return obj

    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None:
            return

        self.max_ = getattr(obj, 'max_', None)
        self.cur = getattr(obj, 'cur', None)

    def append(self, x):
        self[self.cur] = x
        self.cur = (self.cur + 1) % self.max_

    def get_ordered(self):
        return np.concatenate([self[self.cur:], self[:self.cur]], axis=0)

    def get_first(self):
        return self[self.cur]

    def get_last_frames(self, num_frames):
        if self.cur >= num_frames:
            return self[self.cur - num_frames:self.cur]
        else:
            return np.concatenate([self[(self.cur - num_frames):], self[:self.cur]], axis=0)

def csc_append(a, b):
    """ Takes in 2 csc_matrices and appends the second one to the right of the first one.
    Much faster than scipy.sparse.hstack but assumes the type to be csc and overwrites
    the first matrix instead of copying it. The data, indices, and indptr still get copied."""
    a.data = np.concatenate((a.data, b.data))
    a.indices = np.concatenate((a.indices, b.indices))
    a.indptr = np.concatenate((a.indptr, (b.indptr + a.nnz)[1:]))
    a._shape = (a.shape[0], a.shape[1] + b.shape[1])


def csr_append(a, b):
    """ Takes in 2 csr_matrices and appends the second one below the first one.
    Much faster than scipy.sparse.vstack but assumes the type to be csr and overwrites
    the first matrix instead of copying it. The data, indices, and indptr still get copied."""
    a.data = np.concatenate((a.data, b.data))
    a.indices = np.concatenate((a.indices, b.indices))
    a.indptr = np.concatenate((a.indptr, (b.indptr + a.nnz)[1:]))
    a._shape = (a.shape[0] + b.shape[0], a.shape[1])


def corr(a, b):
    """
    faster correlation than np.corrcoef, especially for smaller arrays
    be aware of side effects and pass a copy if necessary!
    """
    a -= a.mean()
    b -= b.mean()
    return a.dot(b) / sqrt(a.dot(a) * b.dot(b) + np.finfo(float).eps)


def rank1nmf(Ypx, ain, iters=10):
    """
    perform a fast rank 1 NMF
    """
    # cin_old = -1
    eps = np.finfo(np.float32).eps
    for t in range(iters):
        cin_res = ain.dot(Ypx)  # / ain.dot(ain)
        cin = np.maximum(cin_res, 0)
        ain = np.maximum(Ypx.dot(cin), 0)
        # ain /= (sqrt(ain.dot(ain)) + np.finfo(np.float32).eps)
        if t in (0, iters-1):
            ain /= (sqrt(ain.dot(ain)) + eps)
        elif t % 2 == 0:  # division by squared norm every 2nd iter is faster yet numerically stable
            ain /= (ain.dot(ain) + eps)
        # nc = cin.dot(cin)
        # ain = np.maximum(Ypx.dot(cin.T) / nc, 0)
        # tmp = cin - cin_old
        # if tmp.dot(tmp) < 1e-6 * nc:
        #     break
        # cin_old = cin.copy()
    cin_res = ain.dot(Ypx)  # / ain.dot(ain)
    cin = np.maximum(cin_res, 0)
    return ain, cin, cin_res

@profile
def get_candidate_components(sv, dims, Yres_buf, min_num_trial=3, gSig=(5, 5),
                             gHalf=(5, 5), sniper_mode=True, rval_thr=0.85,
                             patch_size=50, loaded_model=None, test_both=False,
                             thresh_CNN_noisy=0.5, use_peak_max=False,
                             thresh_std_peak_resid = 1, mean_buff=None,
                             tf_in=None, tf_out=None):
    """
    Extract new candidate components from the residual buffer and test them
    using space correlation or the CNN classifier. The function runs the CNN
    classifier in batch mode which can bring speed improvements when
    multiple components are considered in each timestep.
    """
    Ain = []
    Ain_cnn = []
    Cin = []
    Cin_res = []
    idx = []
    all_indices = []
    ijsig_all = []
    cnn_pos:list = []
    local_maxima:list = []
    Y_patch = []
    ksize = tuple([int(3 * i / 2) * 2 + 1 for i in gSig])
    compute_corr = test_both

    if use_peak_max:

        img_select_peaks = sv.reshape(dims).copy()
        img_select_peaks = gaussian_filter(img_select_peaks, gSig, truncate=3/2, mode='nearest') \
                        - uniform_filter(img_select_peaks, ksize, mode='nearest')
        thresh_img_sel = 0
        local_maxima = peak_local_max(img_select_peaks,
                                      min_distance=np.max(np.array(gSig)).astype(int),
                                      num_peaks=min_num_trial,threshold_abs=thresh_img_sel, exclude_border = False)
        min_num_trial = np.minimum(len(local_maxima),min_num_trial)


    for i in range(min_num_trial):
        if use_peak_max:
            ij = local_maxima[i]
        else:
            ind = np.argmax(sv)
            ij = np.unravel_index(ind, dims, order='C')
            local_maxima.append(ij)

        ij = [min(max(ij_val, g_val), dim_val-g_val-1)
              for ij_val, g_val, dim_val in zip(ij, gHalf, dims)]
        ind = np.ravel_multi_index(ij, dims, order='C')
        ijSig = [[max(i - g, 0), min(i+g+1, d)] for i, g, d in zip(ij, gHalf, dims)]
        ijsig_all.append(ijSig)
        indices = np.ravel_multi_index(np.ix_(*[np.arange(ij[0], ij[1])
                                                for ij in ijSig]), dims, order='F').ravel(order='C')

        ain = np.maximum(mean_buff[indices], 0)

        if sniper_mode:
            half_crop_cnn = tuple([int(np.minimum(gs*2, patch_size/2)) for gs in gSig])
            ij_cnn = [min(max(ij_val,g_val),dim_val-g_val-1) for ij_val, g_val, dim_val in zip(ij,half_crop_cnn,dims)]
            ijSig_cnn = [[max(i - g, 0), min(i+g+1,d)] for i, g, d in zip(ij_cnn, half_crop_cnn, dims)]
            indices_cnn = np.ravel_multi_index(np.ix_(*[np.arange(ij[0], ij[1])
                            for ij in ijSig_cnn]), dims, order='F').ravel(order = 'C')
            ain_cnn = mean_buff[indices_cnn]

        else:
            compute_corr = True  # determine when to compute corr coef

        na = ain.dot(ain)
        # sv[indices_] /= 1  # 0
        if na:
            ain /= sqrt(na)
            Ain.append(ain)
            if compute_corr:
                Y_patch.append(Yres_buf.T[indices, :])
            else:
                all_indices.append(indices)
            idx.append(ind)
            if sniper_mode:
                Ain_cnn.append(ain_cnn)

    if sniper_mode & (len(Ain_cnn) > 0):
        Ain_cnn = np.stack(Ain_cnn)
        Ain2 = Ain_cnn.copy()
        Ain2 -= np.median(Ain2,axis=1)[:,None]
        Ain2 /= np.std(Ain2,axis=1)[:,None]
        Ain2 = np.reshape(Ain2,(-1,) + tuple(np.diff(ijSig_cnn).squeeze()),order= 'F')
        Ain2 = np.stack([cv2.resize(ain,(patch_size ,patch_size)) for ain in Ain2])
        if tf_in is None:
            predictions = loaded_model.predict(Ain2[:,:,:,np.newaxis], batch_size=min_num_trial, verbose=0)
        else:
            predictions = loaded_model.run(tf_out, feed_dict={tf_in: Ain2[:, :, :, np.newaxis]})
        keep_cnn = list(np.where(predictions[:, 0] > thresh_CNN_noisy)[0])
        cnn_pos = Ain2[keep_cnn]
    else:
        keep_cnn = []  # list(range(len(Ain_cnn)))

    if compute_corr:
        keep_corr = []
        for i, (ain, Ypx) in enumerate(zip(Ain, Y_patch)):
            ain, cin, cin_res = rank1nmf(Ypx, ain)
            Ain[i] = ain
            Cin.append(cin)
            Cin_res.append(cin_res)
            rval = corr(ain.copy(), np.mean(Ypx, -1))
            if rval > rval_thr:
                keep_corr.append(i)
        keep_final:list = list(set().union(keep_cnn, keep_corr))
        if len(keep_final) > 0:
            Ain = np.stack(Ain)[keep_final]
        else:
            Ain = []
        Cin = [Cin[kp] for kp in keep_final]
        Cin_res = [Cin_res[kp] for kp in keep_final]
        idx = list(np.array(idx)[keep_final])
    else:
        Ain = [Ain[kp] for kp in keep_cnn]
        Y_patch = [Yres_buf.T[all_indices[kp]] for kp in keep_cnn]
        idx = list(np.array(idx)[keep_cnn])
        for i, (ain, Ypx) in enumerate(zip(Ain, Y_patch)):
            ain, cin, cin_res = rank1nmf(Ypx, ain)
            Ain[i] = ain
            Cin.append(cin)
            Cin_res.append(cin_res)

    return Ain, Cin, Cin_res, idx, ijsig_all, cnn_pos, local_maxima

@profile
def update_num_components(t, sv, Ab, Cf, Yres_buf, Y_buf, rho_buf,
                          dims, gSig, gSiz, ind_A, CY, CC, groups, oases, gnb=1,
                          rval_thr=0.875, bSiz=3, robust_std=False,
                          N_samples_exceptionality=5, remove_baseline=True,
                          thresh_fitness_delta=-80, thresh_fitness_raw=-20,
                          thresh_overlap=0.25, batch_update_suff_stat=False,
                          sn=None, g=None, thresh_s_min=None, s_min=None,
                          Ab_dense=None, max_num_added=1, min_num_trial=1,
                          loaded_model=None, thresh_CNN_noisy=0.99,
                          sniper_mode=False, use_peak_max=False, test_both=False,
                          mean_buff=None, ssub_B=1, W=None, b0=None,
                          corr_img=None, first_moment=None, second_moment=None,
                          crosscorr=None, col_ind=None, row_ind=None, corr_img_mode=None,
                          max_img=None, downscale_matrix=None, upscale_matrix=None,
                          tf_in=None, tf_out=None):
    """
    Checks for new components in the residual buffer and incorporates them if they pass the acceptance tests
    """

    ind_new = []
    gHalf = np.array(gSiz) // 2

    # number of total components (including background)
    M = np.shape(Ab)[-1]
    N = M - gnb                 # number of components (without background)

    if corr_img is None:
        sv -= rho_buf.get_first()
        # update variance of residual buffer
        sv += rho_buf.get_last_frames(1).squeeze()
        sv = np.maximum(sv, 0)

    if max_img is not None:
        # pnr_img = max_img.ravel(order='F') / np.sqrt(second_moment - first_moment**2)
        # pnr_img = max_img / np.sqrt(second_moment - first_moment**2).reshape(dims, order='F')
        pnr_img = max_img / sn.reshape(dims, order='F')
        pnr_img[pnr_img<2] = 0

    Ains, Cins, Cins_res, inds, ijsig_all, cnn_pos, local_max = get_candidate_components(
        sv if corr_img is None else corr_img*pnr_img, dims, Yres_buf=Yres_buf,
        min_num_trial=min_num_trial, gSig=gSig, gHalf=gHalf,
        sniper_mode=sniper_mode, rval_thr=rval_thr, patch_size=50,
        loaded_model=loaded_model, thresh_CNN_noisy=thresh_CNN_noisy,
        use_peak_max=use_peak_max, test_both=test_both, mean_buff=mean_buff,
        tf_in=tf_in, tf_out=tf_out)

    ind_new_all = ijsig_all

    num_added = 0  # len(inds)
    cnt = 0
    for ind, ain, cin, cin_res in zip(inds, Ains, Cins, Cins_res):
        cnt += 1
        ij = np.unravel_index(ind, dims)

        ijSig = [[max(i - temp_g, 0), min(i + temp_g + 1, d)] for i, temp_g, d in zip(ij, gHalf, dims)]

        indices = np.ravel_multi_index(
                np.ix_(*[np.arange(ij[0], ij[1])
                       for ij in ijSig]), dims, order='F').ravel()

        cin_circ = cin.get_ordered()
        useOASIS = False  # whether to use faster OASIS for cell detection
        accepted = True   # flag indicating new component has not been rejected yet

        if Ab_dense is None:
            Ain = np.zeros((np.prod(dims), 1), dtype=np.float32)
            Ain[indices, :] = ain[:, None]
            ff = np.where((Ab.T.dot(Ain).T > thresh_overlap)
                          [:, gnb:])[1] + gnb
        else:
            ff = np.where(Ab_dense[indices, gnb:M].T.dot(
                ain).T > thresh_overlap)[0] + gnb

        if ff.size > 0:
#                accepted = False
            cc = [corr(cin_circ.copy(), cins) for cins in Cf[ff, :]]
            if np.any(np.array(cc) > .25) and accepted:
                accepted = False         # reject component as duplicate

        if s_min is None:
            s_min = 0
        # use s_min * noise estimate * sqrt(1-sum(gamma))
        elif s_min < 0:
            # the formula has been obtained by running OASIS with s_min=0 and lambda=0 on Gaussin noise.
            # e.g. 1 * sigma * sqrt(1-sum(gamma)) corresponds roughly to the root mean square (non-zero) spike size, sqrt(<s^2>)
            #      2 * sigma * sqrt(1-sum(gamma)) corresponds roughly to the 95% percentile of (non-zero) spike sizes
            #      3 * sigma * sqrt(1-sum(gamma)) corresponds roughly to the 99.7% percentile of (non-zero) spike sizes
            s_min = -s_min * sqrt((ain**2).dot(sn[indices]**2)) * sqrt(1 - np.sum(g))

        cin_res = cin_res.get_ordered()
        if accepted:
            if useOASIS:
                oas = OASIS(g=g, s_min=s_min,
                            num_empty_samples=t + 1 - len(cin_res))
                for yt in cin_res:
                    oas.fit_next(yt)
                accepted = oas.get_l_of_last_pool() <= t
            else:
                fitness_delta, erfc_delta, std_rr, _ = compute_event_exceptionality(
                    np.diff(cin_res)[None, :], robust_std=robust_std, N=N_samples_exceptionality)
                if remove_baseline:
                    num_samps_bl = min(len(cin_res) // 5, 800)
                    bl = percentile_filter(cin_res, 8, size=num_samps_bl)
                else:
                    bl = 0
                fitness_raw, erfc_raw, std_rr, _ = compute_event_exceptionality(
                    (cin_res - bl)[None, :], robust_std=robust_std,
                    N=N_samples_exceptionality)
                accepted = (fitness_delta < thresh_fitness_delta) or (
                    fitness_raw < thresh_fitness_raw)

        if accepted:
            # print('adding component' + str(N + 1) + ' at timestep ' + str(t))
            num_added += 1
            ind_new.append(ijSig)

            if oases is not None:
                if not useOASIS:
                    # lambda from Selesnick's 3*sigma*|K| rule
                    # use noise estimate from init batch or use std_rr?
                    #                    sn_ = sqrt((ain**2).dot(sn[indices]**2)) / sqrt(1 - g**2)
                    sn_ = std_rr
                    oas = OASIS(np.ravel(g)[0], 3 * sn_ /
                                (sqrt(1 - g**2) if np.size(g) == 1 else
                                 sqrt((1 + g[1]) * ((1 - g[1])**2 - g[0]**2) / (1 - g[1])))
                                      if s_min == 0 else 0,
                                      s_min, num_empty_samples=t +
                                      1 - len(cin_res),
                                      g2=0 if np.size(g) == 1 else g[1])
                    for yt in cin_res:
                        oas.fit_next(yt)

                oases.append(oas)

            Ain_csc = csc_matrix((ain, (indices, [0] * len(indices))), (np.prod(dims), 1), dtype=np.float32)
            if Ab_dense is None:
                groups = update_order(Ab, Ain, groups)[0]
            else:
                groups = update_order(Ab_dense[indices, :M], ain, groups)[0]
                Ab_dense[indices, M] = ain

            # faster version of scipy.sparse.hstack
            csc_append(Ab, Ain_csc)
            ind_A.append(Ab.indices[Ab.indptr[M]:Ab.indptr[M + 1]])

            tt = t * 1.

            # preallocate memory for speed up?
            CC1 = np.hstack([CC, Cf.dot(cin_circ / tt)[:, None]])
            CC2 = np.hstack(
                [(Cf.dot(cin_circ)).T, cin_circ.dot(cin_circ)]) / tt
            CC = np.vstack([CC1, CC2])
            Cf = np.vstack([Cf, cin_circ])

            if W is not None:  # 1p data, subtract background
                y = Y_buf.get_ordered()
                if ssub_B == 1:
                    x = y.T - Ab.dot(Cf) - b0[:, None]
                    y = y[:, indices] - W[indices].dot(x).T - b0[indices]
                else:
                    d1, d2 = dims
                    x = downscale_matrix.dot(y.T - Ab.dot(Cf) - b0[:, None])
                    y = y[:, indices] - upscale_matrix.tocsr()[indices].dot(
                        W).dot(x).T - b0[indices]
                CY[M, indices] = cin_circ.dot(y) / tt
            else:
                CY[M, indices] = cin.dot(Y_buf[:, indices]) / tt

            N = N + 1
            M = M + 1

            if crosscorr is not None:
                # TODO: restrict to indices where component is located
                if Ab_dense is None:
                    Ain = np.zeros(np.prod(dims), dtype=np.float32)
                    Ain[indices] = ain
                else:
                    Ain = Ab_dense[:, M - 1]
                if corr_img_mode == 'cumulative':
                    # first_moment[indices] = 0
                    # second_moment[indices] = 0
                    first_moment[Ain > 0] = 0
                    second_moment[Ain > 0] = 0
                    crosscorr *= (Ain[row_ind]==0) * (Ain[col_ind]==0)
                else:
                    div = t if corr_img_mode == 'cumulative' else len(cin)
                    first_moment[indices] -= cin.sum() / div * ain
                    # (Y-ac')^2 = Y.^2 + (ac'.^2 - 2Y.ac)
                    second_moment[indices] += (ain**2 * cin.dot(cin) -
                                               2 * cin.dot(Yres_buf[:, indices]) * ain) / div
                    crosscorr += (Ain[row_ind] * Ain[col_ind] * cin.dot(cin) -
                                  cin.dot(Yres_buf[:, row_ind]) * Ain[col_ind] -
                                  cin.dot(Yres_buf[:, col_ind]) * Ain[row_ind]) / div
                max_img[Ain.reshape(dims, order='F') > 0] = 0
                # # max_img[[slice(*i) for i in ijSig]] = first_moment[indices].reshape(
                # #     np.diff(ijSig).ravel(), order='F')

                
            Yres_buf[:, indices] -= np.outer(cin, ain)

            
            if corr_img is None:
                # restrict blurring to region where component is located
                # update bigger region than neural patch to avoid boundary effects
                slices_update = tuple(slice(max(0, ijs[0] - sg // 2), min(d, ijs[1] + sg // 2))
                                      for ijs, sg, d in zip(ijSig, gSiz, dims))
                # filter even bigger region to avoid boundary effects
                slices_filter = tuple(slice(max(0, ijs[0] - sg), min(d, ijs[1] + sg))
                                      for ijs, sg, d in zip(ijSig, gSiz, dims))

                ind_vb = np.ravel_multi_index(
                    np.ix_(*[np.arange(sl.start, sl.stop)
                             for sl in slices_update]), dims, order='C').ravel()

                if len(dims) == 3:
                    rho_buf[:, ind_vb] = np.stack([imblur(
                        vb.reshape(dims, order='F')[slices_filter], sig=gSig, siz=gSiz,
                        nDimBlur=len(dims))[tuple([slice(
                            slices_update[i].start - slices_filter[i].start,
                            slices_update[i].stop - slices_filter[i].start)
                            for i in range(len(dims))])].ravel() for vb in Yres_buf])**2
                else:
                    # faster than looping over frames:
                    # transform all frames into one, blur all simultaneously, transform back
                    Y_filter = Yres_buf.reshape((-1,) + dims, order='F'
                                                )[:, slices_filter[0], slices_filter[1]]
                    T, d0, d1 = Y_filter.shape
                    tmp = np.concatenate((Y_filter, np.zeros((T, gHalf[0], d1), dtype=np.float32)),
                                         axis=1).reshape(-1, d1)
                    cv2.GaussianBlur(tmp, tuple(gSiz), gSig[0], tmp, gSig[1], cv2.BORDER_CONSTANT)
                    slices = tuple([slice(slices_update[i].start - slices_filter[i].start,
                                          slices_update[i].stop - slices_filter[i].start)
                                    for i in range(len(dims))])
                    rho_buf[:, ind_vb] = tmp.reshape(T, -1, d1)[
                        (slice(None),) + slices].reshape(T, -1)**2

                sv[ind_vb] = np.sum(rho_buf[:, ind_vb], 0)

    return Ab, Cf, Yres_buf, rho_buf, CC, CY, ind_A, sv, groups, ind_new, ind_new_all, sv, cnn_pos


# remove components online
def remove_components_online(ind_rem, gnb, Ab, use_dense, Ab_dense, AtA, CY,
                             CC, M, N, noisyC, OASISinstances, C_on, exp_comps):

    """
    Remove components indexed by ind_r (indexing starts at zero)

    Args:
        ind_rem list
            indices of components to be removed (starting from zero)
        gnb int
            number of global background components
        Ab  csc_matrix
            matrix of components + background
        use_dense bool
            use dense representation
        Ab_dense ndarray
    """

    ind_rem.sort()
    ind_rem = [ind + gnb for ind in ind_rem[::-1]]
    ind_keep = list(set(range(Ab.shape[-1])) - set(ind_rem))
    ind_keep.sort()

    if use_dense:
        Ab_dense = np.delete(Ab_dense, ind_rem, axis=1)
    else:
        Ab_dense = []
    AtA = np.delete(AtA, ind_rem, axis=0)
    AtA = np.delete(AtA, ind_rem, axis=1)
    CY = np.delete(CY, ind_rem, axis=0)
    CC = np.delete(CC, ind_rem, axis=0)
    CC = np.delete(CC, ind_rem, axis=1)
    M -= len(ind_rem)
    N -= len(ind_rem)
    exp_comps -= len(ind_rem)
    noisyC = np.delete(noisyC, ind_rem, axis=0)
    for ii in ind_rem:
        del OASISinstances[ii - gnb]

    C_on = np.delete(C_on, ind_rem, axis=0)
    Ab = csc_matrix(Ab[:, ind_keep])
    ind_A = list(
        [(Ab.indices[Ab.indptr[ii]:Ab.indptr[ii+1]]) for ii in range(gnb, M)])
    groups = list(map(list, update_order(Ab)[0]))

    return Ab, Ab_dense, CC, CY, M, N, noisyC, OASISinstances, C_on, exp_comps, ind_A, groups, AtA

def initialize_movie_online(Y, K, gSig, rf, stride, base_name,
                            p=1, merge_thresh=0.95, rval_thr_online=0.9, thresh_fitness_delta_online=-30, thresh_fitness_raw_online=-50,
                            rval_thr_init=.5, thresh_fitness_delta_init=-20, thresh_fitness_raw_init=-20,
                            rval_thr_refine=0.95, thresh_fitness_delta_refine=-100, thresh_fitness_raw_refine=-100,
                            final_frate=10, Npeaks=10, single_thread=True, dview=None, n_processes=None):
    """
    Initialize movie using CNMF on minibatch. See CNMF parameters
    """

    Yr = Y.to_2D().T
    # merging threshold, max correlation allowed
    # order of the autoregressive system
    base_name = base_name + '.mmap'
    fname_new = Y.save(caiman.paths.fn_relocated(base_name), order='C')
    Yr, dims, T = caiman.load_memmap(fname_new)
    d1, d2 = dims
    images = np.reshape(Yr.T, [T] + list(dims), order='F')
    Y = np.reshape(Yr, dims + (T,), order='F')
    Cn2 = caiman.local_correlations(Y)
    # RUN ALGORITHM ON PATCHES
    cnm_init = caiman.source_extraction.cnmf.CNMF(n_processes, method_init='greedy_roi', k=K, gSig=gSig, merge_thresh=merge_thresh,
                                              p=0, dview=dview, Ain=None, rf=rf, stride=stride, method_deconvolution='oasis', skip_refinement=False,
                                              normalize_init=False, options_local_NMF=None,
                                              minibatch_shape=100, minibatch_suff_stat=5,
                                              update_num_comps=True, rval_thr=rval_thr_online, thresh_fitness_delta=thresh_fitness_delta_online, thresh_fitness_raw=thresh_fitness_raw_online,
                                              batch_update_suff_stat=True, max_comp_update_shape=5)

    cnm_init.fit(images)
    A_tot = cnm_init.A
    C_tot = cnm_init.C
    YrA_tot = cnm_init.YrA
    b_tot = cnm_init.b
    f_tot = cnm_init.f

    print(f"Number of components: {A_tot.shape[-1]}")

    traces = C_tot + YrA_tot
    fitness_raw, fitness_delta, erfc_raw, erfc_delta, r_values, significant_samples = caiman.components_evaluation.evaluate_components(
        Y, traces, A_tot, C_tot, b_tot, f_tot, final_frate, remove_baseline=True, N=5, robust_std=False, Athresh=0.1, Npeaks=Npeaks,  thresh_C=0.3)

    idx_components_r = np.where(r_values >= rval_thr_init)[0]
    idx_components_raw = np.where(fitness_raw < thresh_fitness_raw_init)[0]
    idx_components_delta = np.where(
        fitness_delta < thresh_fitness_delta_init)[0]

    idx_components = np.union1d(idx_components_r, idx_components_raw)
    idx_components = np.union1d(idx_components, idx_components_delta)
    idx_components_bad = np.setdiff1d(list(range(len(traces))), idx_components)

    print(('Keeping ' + str(len(idx_components)) +
           ' and discarding  ' + str(len(idx_components_bad))))

    A_tot = A_tot.tocsc()[:, idx_components]
    C_tot = C_tot[idx_components]

    cnm_refine = caiman.source_extraction.cnmf.CNMF(n_processes, method_init='greedy_roi', k=A_tot.shape, gSig=gSig, merge_thresh=merge_thresh, rf=None, stride=None,
                                                p=p, dview=dview, Ain=A_tot, Cin=C_tot, f_in=f_tot, method_deconvolution='oasis', skip_refinement=True,
                                                normalize_init=False, options_local_NMF=None,
                                                minibatch_shape=100, minibatch_suff_stat=5,
                                                update_num_comps=True, rval_thr=rval_thr_refine, thresh_fitness_delta=thresh_fitness_delta_refine, thresh_fitness_raw=thresh_fitness_raw_refine,
                                                batch_update_suff_stat=True, max_comp_update_shape=5)

    cnm_refine.fit(images)

    A, C, b, f, YrA = cnm_refine.A, cnm_refine.C, cnm_refine.b, cnm_refine.f, cnm_refine.YrA

    final_frate = 10
    Npeaks = 10
    traces = C + YrA

    fitness_raw, fitness_delta, erfc_raw, erfc_delta, r_values, significant_samples = \
        caiman.components_evaluation.evaluate_components(Y, traces, A, C, b, f, final_frate, remove_baseline=True,
                                                     N=5, robust_std=False, Athresh=0.1, Npeaks=Npeaks,  thresh_C=0.3)

    idx_components_r = np.where(r_values >= rval_thr_refine)[0]
    idx_components_raw = np.where(fitness_raw < thresh_fitness_raw_refine)[0]
    idx_components_delta = np.where(
        fitness_delta < thresh_fitness_delta_refine)[0]

    idx_components = np.union1d(idx_components_r, idx_components_raw)
    idx_components = np.union1d(idx_components, idx_components_delta)
    idx_components_bad = np.setdiff1d(list(range(len(traces))), idx_components)

    print(' ***** ')
    print((len(traces)))
    print((len(idx_components)))

    cnm_refine.sn = sn # FIXME: There is no sn in scope here
    cnm_refine.idx_components = idx_components
    cnm_refine.idx_components_bad = idx_components_bad
    cnm_refine.r_values = r_values
    cnm_refine.fitness_raw = fitness_raw
    cnm_refine.fitness_delta = fitness_delta
    cnm_refine.Cn2 = Cn2

    return cnm_refine, Cn2, fname_new

def load_OnlineCNMF(filename, dview = None):
    """load object saved with the CNMF save method

    Args:
        filename: str
            hdf5 file name containing the saved object
        dview: multiprocessing or ipyparallel object
            useful to set up parllelization in the objects
    """

    filename = caiman.paths.fn_relocated(filename)
    for key, val in load_dict_from_hdf5(filename).items():
        if key == 'params':
            prms = CNMFParams()
            for subdict in val.keys():
                prms.set(subdict, val[subdict])

    new_obj = OnACID(params=prms)

    for key, val in load_dict_from_hdf5(filename).items():
        if key == 'dview':
            setattr(new_obj, key, dview)
        elif key == 'estimates':
            estim = Estimates()
            for key_est, val_est in val.items():
                setattr(estim, key_est, val_est)
            new_obj.estimates = estim
        else:
            if key not in ['params', 'estimates']:
                setattr(new_obj, key, val)

    return new_obj

def inv_mat_vec(A):
    return np.linalg.solve(A[0], A[1])
