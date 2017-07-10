#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 11:58:18 2017

@author: johannes
"""

import numpy as np
from skimage.morphology import disk
from scipy.sparse import csr_matrix


def compute_W(Y, A, C, dims, radius):
    ring = (disk(radius + 1) -
            np.concatenate([np.zeros((1, 2 * radius + 3), dtype=bool),
                            np.concatenate([np.zeros((2 * radius + 1, 1), dtype=bool),
                                            disk(radius),
                                            np.zeros((2 * radius + 1, 1), dtype=bool)], 1),
                            np.zeros((1, 2 * radius + 3), dtype=bool)]))
    ringidx = [i - radius - 1 for i in np.nonzero(ring)]

    def get_indices_of_pixels_on_ring(pixel):
        pixel = np.unravel_index(pixel, dims, order='F')
        x = pixel[0] + ringidx[0]
        y = pixel[1] + ringidx[1]
        inside = (x >= 0) * (x < dims[0]) * (y >= 0) * (y < dims[1])
        return np.ravel_multi_index((x[inside], y[inside]), dims, order='F')

    b0 = Y.mean(1) - A.dot(C.mean(1))

    indices = []
    data = []
    indptr = [0]
    for p in xrange(np.prod(dims)):
        index = get_indices_of_pixels_on_ring(p)
        indices += list(index)
        B = Y[index] - A[index].dot(C) - b0[index, None]
        data += list(np.linalg.inv(B.dot(B.T) + 1e-12 * np.eye(len(index))).
                     dot(B.dot(Y[p] - A[p].dot(C).ravel() - b0[p])))
        # np.linalg.lstsq seems less robust but scipy version would be (slower) alternative
        # data += list(scipy.linalg.lstsq(B.T, Y[p] - A[p].dot(C) - b0[p], check_finite=False)[0])
        indptr.append(len(indices))
    return csr_matrix((data, indices, indptr), dtype='float32'), b0


def compute_W_parallel(Y, A, C, dims, radius, dview=None, n_pixels_per_process=128,
                       Y_name=None, data_fits_in_memory=False):
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

    dview: [optional] view on ipyparallel client
        you need to create an ipyparallel client and pass a view on the processors
        (client = Client(), dview=client[:])

    n_pixels_per_process: [optional] int
        number of pixels to be processed by each thread

    Y_name: [optional] string
            name of memmap Y file, only required if using ipyparallel

    data_fits_in_memory: [optional] bool
            If true, use faster but more memory consuming computation

    Returns:
    --------
    W: scipy.sparse.csr_matrix (pixels x pixels)
        estimate of weight matrix for fluctuating background

    b0: np.ndarray (pixels,)
        estimate of constant background baselines
    """

    if dview is not None and Y_name is None:
        raise Exception('must provide filename of memmapped file if dview is not None')

    ring = (disk(radius + 1) -
            np.concatenate([np.zeros((1, 2 * radius + 3), dtype=bool),
                            np.concatenate([np.zeros((2 * radius + 1, 1), dtype=bool),
                                            disk(radius),
                                            np.zeros((2 * radius + 1, 1), dtype=bool)], 1),
                            np.zeros((1, 2 * radius + 3), dtype=bool)]))
    ringidx = [i - radius - 1 for i in np.nonzero(ring)]

    b0 = Y.mean(1) - A.dot(C.mean(1))
    X = Y - A.dot(C) - b0[:, None] if data_fits_in_memory else None
    if data_fits_in_memory and dview is not None:
        # save and later each parallel process loads it as a memory map file
        import tempfile
        import os
        tmp_dir = tempfile.mkdtemp()
        X_name = os.path.join(tmp_dir, 'X_temp.npy')
        np.save(X_name, X)

    tmp = np.prod(dims) // n_pixels_per_process * n_pixels_per_process
    pixel_groups = np.arange(tmp).reshape(-1, n_pixels_per_process).tolist()
    if tmp < np.prod(dims):
        pixel_groups.append(list(range(tmp, np.prod(dims))))

    def regress(pars):
        import numpy as np
        import caiman as cm
        pixels, Y_name, A, C, b0, dims, ringidx, X_name = pars
        if isinstance(Y_name, basestring):
            Y, _, _ = cm.load_memmap(Y_name)
        else:
            Y = Y_name
        if isinstance(X_name, basestring):
            X = np.load(X_name, mmap_mode='r')
        else:
            X = X_name

        def get_indices_of_pixels_on_ring(pixel):
            pixel = np.unravel_index(pixel, dims, order='F')
            x = pixel[0] + ringidx[0]
            y = pixel[1] + ringidx[1]
            inside = (x >= 0) * (x < dims[0]) * (y >= 0) * (y < dims[1])
            return np.ravel_multi_index((x[inside], y[inside]), dims, order='F')

        indices = []
        data = []
        indptr = []
        for p in pixels:
            index = get_indices_of_pixels_on_ring(p)
            indices += list(index)
            B = Y[index] - A[index].dot(C) - b0[index, None] if X is None else X[index]
            data += list(np.linalg.inv(B.dot(B.T) + 1e-12 * np.eye(len(index))).
                         dot(B.dot(Y[p] - A[p].dot(C).ravel() - b0[p] if X is None else X[p])))
            # np.linalg.lstsq seems less robust but scipy version would be (slower) alternative
            # data += list(scipy.linalg.lstsq(B.T, Y[p] - A[p].dot(C) - b0[p])[0])
            indptr.append(len(indices))
        return data, indices, indptr

    if dview is not None:
        parallel_result = dview.map_sync(regress, [(pixels, Y_name, A, C, b0, dims, ringidx, X)
                                                   for pixels in pixel_groups])
        dview.results.clear()
    else:
        parallel_result = map(lambda pixels: regress((pixels, Y, A, C, b0, dims, ringidx, X)),
                              pixel_groups)

    indices = []
    data = []
    indptr = np.zeros(len(Y) + 1, dtype='int32')
    for i, r in enumerate(parallel_result):
        data += r[0]
        indptr[1 + i * n_pixels_per_process:1 +
               (i + 1) * n_pixels_per_process] = np.array(r[2]) + len(indices)
        indices += r[1]

    if data_fits_in_memory and dview is not None:
        import shutil
        shutil.rmtree(tmp_dir)
    return csr_matrix((data, indices, indptr), dtype='float32'), b0
