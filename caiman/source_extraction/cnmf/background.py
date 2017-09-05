# -*- coding: utf-8 -*-

"""
all functions for estimating the background in calcium imaging data.
supported models:
1. ring model

author: Pengcheng Zhou
email: zhoupc1988@gmail.com
created: 8/30/17
last edited: 
"""

from __future__ import division
from __future__ import print_function
from builtins import range
import numpy as np
from skimage.morphology import disk
import scipy.sparse as spr
import sys
from sklearn.decomposition import NMF

if sys.version_info >= (3, 0):
    def xrange(*args, **kwargs):
        return iter(range(*args, **kwargs))
"""
-------------------------------FUNCTIONS-------------------------------
"""

# def estimate_bg(Y, model='nmf', nb=1)
#
#
#
# def bg_ring(Y, ring_radius=20):
#     ring_radius = int(ring_radius)
#     dims, total_frames = np.shape(Y)[:-1], np.shape(Y)[-1]
#     B = Y.reshape((-1, total_frames), order='F')
#     W, b0 = compute_W(Y.reshape((-1, total_frames), order='F'), dims, ring_radius)
#     B = b0[:, None] + W.dot(B - b0[:, None])
#
#
# def bg_nmf(B, nb=1):
#     model = NMF(n_components=nb)  # , init='random', random_state=0)
#     b = model.fit_transform(np.maximum(B, 0))
#     f = model.components_.squeeze()
#     return b, f


# #
# def ring_filter_data(Y, W, b0=None):
#     """
#     filter data using ring model Yhat=W*(Y-b0*1') + b0*1'
#     Args:
#         Y: np.ndarray (2D)
#             movie, raw data in 2D (pixels X time)
#         W: np.ndarray (2D)
#             d*d weight matrix, where d is the number of pixels.
#
#     Returns:
#         Yhat: np.ndarray (2D)
#             filtered matrix with the same dimension as Y
#     """
#     if b0 is None:
#         b0 = np.array(Y.mean(1))
#     temp = b0 - W.dot(b0)
#     Yhat = W.dot(Y) - temp[:, None]
#     return Yhat
#
#
# def ring_fit(Y, dims, radius):
#     """compute background according to ring model
#
#     solves the problem
#         min_{W,b0} ||X-W*X|| with X = Y - b0*1'
#     subject to
#         W(i,j) = 0 for each pixel j that is not in ring around pixel i
#     Problem parallelizes over pixels i
#     Fluctuating background activity is W*X, constant baselines b0.
#
#     Parameters:
#     ----------
#     Y: np.ndarray (2D or 3D)
#         movie, raw data in 2D or 3D (pixels x time).
#
#     dims: tuple
#         x, y[, z] movie dimensions
#
#     radius: int
#         radius of ring
#
#     Returns:
#     --------
#     W: scipy.sparse.csr_matrix (pixels x pixels)
#         estimate of weight matrix for fluctuating background
#
#     b0: np.ndarray (pixels,)
#         estimate of constant background baselines
#     """
#     ring = disk(radius+1, dtype=bool)
#     ring[1:-1, 1:-1] -= disk(radius, dtype=bool)
#     ringidx = [i - radius - 1 for i in np.nonzero(ring)]
#
#     def get_indices_of_pixels_on_ring(pixel):
#         pixel = np.unravel_index(pixel, dims)
#         x = pixel[0] + ringidx[0]
#         y = pixel[1] + ringidx[1]
#         inside = (x >= 0) * (x < dims[0]) * (y >= 0) * (y < dims[1])
#         return np.ravel_multi_index((x[inside], y[inside]), dims)
#
#     b0 = np.array(Y.mean(1))
#
#     indices = []
#     data = []
#     indptr = [0]
#     for p in xrange(np.prod(dims)):
#         index = get_indices_of_pixels_on_ring(p)
#         indices += list(index)
#         B = Y[index] - b0[index, None]
#         data += list(np.linalg.inv(np.array(B.dot(B.T)) + 1e-9 * np.eye(len(index))).
#                      dot(B.dot(Y[p] - b0[p])))
#         # np.linalg.lstsq seems less robust but scipy version would be (robust but for the problem size slower) alternative
#         # data += list(scipy.linalg.lstsq(B.T, Y[p] - A[p].dot(C) - b0[p], check_finite=False)[0])
#         indptr.append(len(indices))
#     return spr.csr_matrix((data, indices, indptr), dtype='float32'), b0
#

def compute_W(Y, A, C, dims, radius):
    """compute background according to ring model
    solves the problem
        min_{W,b0} ||X-W*X|| with X = Y - A*C - b0*1'
    subject to
        W(i,j) = 0 for each pixel j that is not in ring around pixel i
    Problem parallelizes over pixels i
    Fluctuating background activity is W*X, constant baselines b0.
    Parameters:
    ----------
    Y: np.ndarray (2D or 3D)
        movie, raw data in 2D or 3D (pixels x time).
    A: np.ndarray or sparse matrix
        spatial footprint of each neuron.
    C: np.ndarray
        calcium activity of each neuron.
    dims: tuple
        x, y[, z] movie dimensions
    radius: int
        radius of ring
    Returns:
    --------
    W: scipy.sparse.csr_matrix (pixels x pixels)
        estimate of weight matrix for fluctuating background
    b0: np.ndarray (pixels,)
        estimate of constant background baselines
    """
    # ring = (disk(radius + 1) -
    #         np.concatenate([np.zeros((1, 2 * radius + 3), dtype=bool),
    #                         np.concatenate([np.zeros((2 * radius + 1, 1), dtype=bool),
    #                                         disk(radius),
    #                                         np.zeros((2 * radius + 1, 1), dtype=bool)], 1),
    #                         np.zeros((1, 2 * radius + 3), dtype=bool)]))
    # ringidx = [i - radius - 1 for i in np.nonzero(ring)]

    ring = disk(radius+1, dtype=bool)
    ring[1:-1, 1:-1] -= disk(radius, dtype=bool)
    ringidx = [i - radius - 1 for i in np.nonzero(ring)]

    def get_indices_of_pixels_on_ring(pixel):
        pixel = np.unravel_index(pixel, dims, order='F')
        x = pixel[0] + ringidx[0]
        y = pixel[1] + ringidx[1]
        inside = (x >= 0) * (x < dims[0]) * (y >= 0) * (y < dims[1])
        return np.ravel_multi_index((x[inside], y[inside]), dims, order='F')

    b0 = np.array(Y.mean(1)) - A.dot(C.mean(1))

    indices = []
    data = []
    indptr = [0]
    for p in xrange(np.prod(dims)):
        index = get_indices_of_pixels_on_ring(p)
        indices += list(index)
        B = Y[index] - A[index].dot(C) - b0[index, None]
        data += list(np.linalg.inv(np.array(B.dot(B.T)) + 1e-9 * np.eye(len(index))).
                     dot(B.dot(Y[p] - A[p].dot(C).ravel() - b0[p])))
        # np.linalg.lstsq seems less robust but scipy version would be (robust but for the problem size slower) alternative
        # data += list(scipy.linalg.lstsq(B.T, Y[p] - A[p].dot(C) - b0[p], check_finite=False)[0])
        indptr.append(len(indices))
    return spr.csr_matrix((data, indices, indptr), dtype='float32'), b0
