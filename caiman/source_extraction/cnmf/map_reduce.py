#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Function for implementing parallel scalable segmentation of two photon imaging data

..image::docs/img/cnmf1.png


@author: agiovann
"""
#\package caiman/dource_ectraction/cnmf
#\version   1.0
#\copyright GNU General Public License v2.0
#\date Created on Wed Feb 17 14:58:26 2016

from __future__ import division
from __future__ import print_function
from builtins import zip
from builtins import str
from builtins import map
from builtins import range

from past.utils import old_div

from copy import copy, deepcopy
import logger
import numpy as np
import os
import scipy
import time

from ...mmapping import load_memmap
from ...cluster import extract_patch_coordinates

#%%
def cnmf_patches(args_in):
    """Function that is run for each patches

         Will be called

        Parameters:
        ----------
        file_name: string
            full path to an npy file (2D, pixels x time) containing the movie

        shape: tuple of thre elements
            dimensions of the original movie across y, x, and time

        params:
            CNMFParms object containing all the parameters for the various algorithms

        rf: int
            half-size of the square patch in pixel

        stride: int
            amount of overlap between patches

        gnb: int
            number of global background components

        backend: string
            'ipyparallel' or 'single_thread' or SLURM

        n_processes: int
            nuber of cores to be used (should be less than the number of cores started with ipyparallel)

        memory_fact: double
            unitless number accounting how much memory should be used.
            It represents the fration of patch processed in a single thread.
             You will need to try different values to see which one would work

        low_rank_background: bool
            if True the background is approximated with gnb components. If false every patch keeps its background (overlaps are randomly assigned to one spatial component only)

        Returns:
        -------
        A_tot: matrix containing all the componenents from all the patches

        C_tot: matrix containing the calcium traces corresponding to A_tot

        sn_tot: per pixel noise estimate

        optional_outputs: set of outputs related to the result of CNMF ALGORITHM ON EACH patch

        Raise:
        -----

        Empty Exception
        """

    import logging
    from . import cnmf
    file_name, idx_, shapes, params = args_in

    logger = logging.getLogger(__name__)
    name_log = os.path.basename(
        file_name[:-5]) + '_LOG_ ' + str(idx_[0]) + '_' + str(idx_[-1])
    #logger = logging.getLogger(name_log)
    #hdlr = logging.FileHandler('./' + name_log)
    #formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    #hdlr.setFormatter(formatter)
    #logger.addHandler(hdlr)
    #logger.setLevel(logging.INFO)

    logger.debug(name_log + 'START')

    logger.debug(name_log + 'Read file')
    Yr, dims, timesteps = load_memmap(file_name)

    # slicing array (takes the min and max index in n-dimensional space and cuts the box they define)
    # for 2d a rectangle/square, for 3d a rectangular cuboid/cube, etc.
    upper_left_corner = min(idx_)
    lower_right_corner = max(idx_)
    indices = np.unravel_index([upper_left_corner, lower_right_corner],
                               dims, order='F')  # indices as tuples
    slices = [slice(min_dim, max_dim + 1) for min_dim, max_dim in indices]
    # insert slice for timesteps, equivalent to :
    slices.insert(0, slice(timesteps))

    images = np.reshape(Yr.T, [timesteps] + list(dims), order='F')
    if params.get('patch', 'in_memory'):
        images = np.array(images[slices],dtype=np.float32)
    else:
        images = images[slices]

    logger.debug(name_log+'file loaded')

    if (np.sum(np.abs(np.diff(images.reshape(timesteps, -1).T)))) > 0.1:

        opts = copy(params)
        opts.set('patch', {'n_processes': 1, 'rf': None, 'stride': None})
        for group in ('init', 'temporal', 'spatial'):
            opts.set(group, {'nb': params.get('patch', 'nb_patch')})

        cnm = cnmf.CNMF(n_processes=1, params=opts)

        cnm = cnm.fit(images)
        return [idx_, shapes, scipy.sparse.coo_matrix(cnm.estimates.A),
                cnm.estimates.b, cnm.estimates.C, cnm.estimates.f, cnm.estimates.S, cnm.estimates.bl, cnm.estimates.c1,
                cnm.estimates.neurons_sn, cnm.estimates.g, cnm.estimates.sn, cnm.params.to_dict(), cnm.estimates.YrA]
    else:
        return None


#%%
def run_CNMF_patches(file_name, shape, params, gnb=1, dview=None, memory_fact=1,
                     border_pix=0, low_rank_background=True, del_duplicates=False,
                     indeces=[slice(None)]*3):
    """Function that runs CNMF in patches

     Either in parallel or sequentially, and return the result for each.
     It requires that ipyparallel is running

     Will basically initialize everything in order to compute on patches then call a function in parallel that will
     recreate the cnmf object and fit the values.
     It will then recreate the full frame by listing all the fitted values together

    Parameters:
    ----------
    file_name: string
        full path to an npy file (2D, pixels x time) containing the movie

    shape: tuple of three elements
        dimensions of the original movie across y, x, and time

    params:
        CNMFParms object containing all the parameters for the various algorithms

    gnb: int
        number of global background components

    backend: string
        'ipyparallel' or 'single_thread' or SLURM

    n_processes: int
        nuber of cores to be used (should be less than the number of cores started with ipyparallel)

    memory_fact: double
        unitless number accounting how much memory should be used.
        It represents the fration of patch processed in a single thread.
         You will need to try different values to see which one would work

    low_rank_background: bool
        if True the background is approximated with gnb components. If false every patch keeps its background (overlaps are randomly assigned to one spatial component only)

    del_duplicates: bool
        if True keeps only neurons in each patch that are well centered within the patch.
        I.e. neurons that are closer to the center of another patch are removed to
        avoid duplicates, cause the other patch should already account for them.

    Returns:
    -------
    A_tot: matrix containing all the components from all the patches

    C_tot: matrix containing the calcium traces corresponding to A_tot

    sn_tot: per pixel noise estimate

    optional_outputs: set of outputs related to the result of CNMF ALGORITHM ON EACH patch

    Raise:
    -----

    Empty Exception
    """
    dims = shape[:-1]
    d = np.prod(dims)
    T = shape[-1]

    rf = params.get('patch', 'rf')
    if rf is None:
        rf = 16
    if np.isscalar(rf):
        rfs = [rf] * len(dims)
    else:
        rfs = rf

    stride = params.get('patch', 'stride')
    if stride is None:
        stride = 4
    if np.isscalar(stride):
        strides = [stride] * len(dims)
    else:
        strides = stride

    params_copy = deepcopy(params)

    npx_per_proc = np.int(old_div(np.prod(rfs), memory_fact))
    params_copy.set('preprocess', {'n_pixels_per_process': npx_per_proc})
    params_copy.set('spatial', {'n_pixels_per_process': npx_per_proc})
    params_copy.set('temporal', {'n_pixels_per_process': npx_per_proc})

    idx_flat, idx_2d = extract_patch_coordinates(
        dims, rfs, strides, border_pix=border_pix, indeces=indeces[1:])
    args_in = []
    patch_centers = []
    for id_f, id_2d in zip(idx_flat, idx_2d):
        #        print(id_2d)
        args_in.append((file_name, id_f, id_2d, params_copy))
        if del_duplicates:
            foo = np.zeros(d, dtype=bool)
            foo[id_f] = 1
            patch_centers.append(scipy.ndimage.center_of_mass(
                foo.reshape(dims, order='F')))
    print(id_2d)
    st = time.time()
    if dview is not None:
        if 'multiprocessing' in str(type(dview)):
            file_res = dview.map_async(cnmf_patches, args_in).get(4294967)
        else:
            try:
                file_res = dview.map_sync(cnmf_patches, args_in)
                dview.results.clear()
            except:
                print('Something went wrong')
                raise
            finally:
                print('You may think that it went well but reality is harsh')

    else:
        file_res = list(map(cnmf_patches, args_in))

    print((time.time() - st))
    # count components
    count = 0
    count_bgr = 0
    patch_id = 0
    num_patches = len(file_res)
    for jj, fff in enumerate(file_res):
        if fff is not None:
            idx_, shapes, A, b, C, f, S, bl, c1, neurons_sn, g, sn, _, YrA = fff
            for _ in range(np.shape(b)[-1]):
                count_bgr += 1

            A = A.tocsc()
            if del_duplicates:
                keep = []
                for ii in range(np.shape(A)[-1]):
                    neuron_center = (np.array(scipy.ndimage.center_of_mass(
                        A[:, ii].toarray().reshape(shapes, order='F'))) -
                        np.array(shapes) / 2. + np.array(patch_centers[jj]))
                    if np.argmin([np.linalg.norm(neuron_center - p) for p in
                                  np.array(patch_centers)]) == jj:
                        keep.append(ii)
                A = A[:, keep]
                file_res[jj][2] = A
                file_res[jj][4] = C[keep]
                if S is not None:
                    file_res[jj][6] = S[keep]
                    file_res[jj][7] = bl[keep]
                    file_res[jj][8] = c1[keep]
                    file_res[jj][9] = neurons_sn[keep]
                    file_res[jj][10] = g[keep]
                file_res[jj][-1] = YrA[keep]

            # for ii in range(np.shape(A)[-1]):
            #     new_comp = A[:, ii] / np.sqrt(A[:, ii].power(2).sum())
            #     if new_comp.sum() > 0:
            #         count += 1
            count += np.sum(A.sum(0) > 0)

            patch_id += 1

    # INITIALIZING
    nb_patch = params.get('patch', 'nb_patch')
    C_tot = np.zeros((count, T), dtype=np.float32)
    if params.get('init', 'center_psf'):
        S_tot = np.zeros((count, T), dtype=np.float32)
    else:
        S_tot = None
    YrA_tot = np.zeros((count, T), dtype=np.float32)
    F_tot = np.zeros((max(0, num_patches * nb_patch), T), dtype=np.float32)
    mask = np.zeros(d, dtype=np.uint8)
    sn_tot = np.zeros((d))

    f_tot, bl_tot, c1_tot, neurons_sn_tot, g_tot, idx_tot, id_patch_tot, shapes_tot = [
    ], [], [], [], [], [], [], []
    patch_id, empty, count_bgr, count = 0, 0, 0, 0
    idx_tot_B, idx_tot_A, a_tot, b_tot = [], [], [], []
    idx_ptr_B, idx_ptr_A = [0], [0]

    # instead of filling in the matrices, construct lists with their non-zero
    # entries and coordinates
    print('Transforming patches into full matrix')
    for fff in file_res:
        if fff is not None:

            idx_, shapes, A, b, C, f, S, bl, c1, neurons_sn, g, sn, _, YrA = fff
            A = A.tocsc()

            sn_tot[idx_] = sn
            f_tot.append(f)
            bl_tot.append(bl)
            c1_tot.append(c1)
            neurons_sn_tot.append(neurons_sn)
            g_tot.append(g)
            idx_tot.append(idx_)
            shapes_tot.append(shapes)
            mask[idx_] += 1

            if scipy.sparse.issparse(b):
                b = scipy.sparse.csc_matrix(b)
                b_tot.append(b.data)
                idx_ptr_B += list(b.indptr[1:] - b.indptr[:-1])
                idx_tot_B.append(idx_[b.indices])
            else:
                for ii in range(np.shape(b)[-1]):
                    b_tot.append(b[:, ii])
                    idx_tot_B.append(idx_)
                    idx_ptr_B.append(len(idx_))
                    # F_tot[patch_id, :] = f[ii, :]
            count_bgr += b.shape[-1]
            if nb_patch >= 0:
                F_tot[patch_id * nb_patch:(patch_id + 1) * nb_patch] = f
            else:  # full background per patch
                F_tot = np.concatenate([F_tot, f])

            for ii in range(np.shape(A)[-1]):
                new_comp = A[:, ii]  # / np.sqrt(A[:, ii].power(2).sum())
                if new_comp.sum() > 0:
                    a_tot.append(new_comp.toarray().flatten())
                    idx_tot_A.append(idx_)
                    idx_ptr_A.append(len(idx_))
                    C_tot[count, :] = C[ii, :]
                    if params.get('init', 'center_psf'):
                        S_tot[count, :] = S[ii, :]
                    YrA_tot[count, :] = YrA[ii, :]
                    id_patch_tot.append(patch_id)
                    count += 1

            patch_id += 1
        else:
            empty += 1

    print('Skipped %d Empty Patch', empty)
    if count_bgr > 0:
        idx_tot_B = np.concatenate(idx_tot_B)
        b_tot = np.concatenate(b_tot)
        idx_ptr_B = np.cumsum(np.array(idx_ptr_B))
        B_tot = scipy.sparse.csc_matrix(
            (b_tot, idx_tot_B, idx_ptr_B), shape=(d, count_bgr))
    else:
        B_tot = scipy.sparse.csc_matrix((d, count_bgr), dtype=np.float32)

    if len(idx_tot_A):
        idx_tot_A = np.concatenate(idx_tot_A)
        a_tot = np.concatenate(a_tot)
        idx_ptr_A = np.cumsum(np.array(idx_ptr_A))
    A_tot = scipy.sparse.csc_matrix(
        (a_tot, idx_tot_A, idx_ptr_A), shape=(d, count), dtype=np.float32)

    C_tot = C_tot[:count, :]
    YrA_tot = YrA_tot[:count, :]
    F_tot = F_tot[:count_bgr]

    optional_outputs = dict()
    optional_outputs['b_tot'] = b_tot
    optional_outputs['f_tot'] = f_tot
    optional_outputs['bl_tot'] = bl_tot
    optional_outputs['c1_tot'] = c1_tot
    optional_outputs['neurons_sn_tot'] = neurons_sn_tot
    optional_outputs['g_tot'] = g_tot
    optional_outputs['S_tot'] = S_tot
    optional_outputs['idx_tot'] = idx_tot
    optional_outputs['shapes_tot'] = shapes_tot
    optional_outputs['id_patch_tot'] = id_patch_tot
    optional_outputs['B'] = B_tot
    optional_outputs['F'] = F_tot
    optional_outputs['mask'] = mask

    print("Generating background")

    Im = scipy.sparse.csr_matrix(
        (1. / mask, (np.arange(d), np.arange(d))), dtype=np.float32)

    if not del_duplicates:
        A_tot = Im.dot(A_tot)

    if count_bgr == 0:
        b = None
        f = None
    elif low_rank_background is None:
        b = Im.dot(B_tot)
        f = F_tot
        print("Leaving background components intact")
    elif low_rank_background:
        print("Compressing background components with a low rank NMF")
        B_tot = Im.dot(B_tot)
        Bm = (B_tot)
        f = np.r_[np.atleast_2d(np.mean(F_tot, axis=0)),
                  np.random.rand(gnb - 1, T)]

        for _ in range(100):
            f /= np.sqrt((f**2).sum(1)[:, None])
            try:
                b = np.fmax(Bm.dot(F_tot.dot(f.T)).dot(
                    np.linalg.inv(f.dot(f.T))), 0)
            except np.linalg.LinAlgError:  # singular matrix
                b = np.fmax(Bm.dot(scipy.linalg.lstsq(f.T, F_tot.T)[0].T), 0)
            try:
                f = np.linalg.inv(b.T.dot(b)).dot((Bm.T.dot(b)).T.dot(F_tot))
            except np.linalg.LinAlgError:  # singular matrix
                f = scipy.linalg.lstsq(b, Bm.toarray())[0].dot(F_tot)

        nB = np.ravel(np.sqrt((b**2).sum(0)))
        b /= nB
        b = np.array(b, dtype=np.float32)
#        B_tot = scipy.sparse.coo_matrix(B_tot)
        f *= nB[:, None]
    else:
        print('Removing overlapping background components from different patches')
        nA = np.ravel(np.sqrt(A_tot.power(2).sum(0)))
        A_tot /= nA
        A_tot = scipy.sparse.coo_matrix(A_tot)
        C_tot *= nA[:, None]
        YrA_tot *= nA[:, None]
        nB = np.ravel(np.sqrt(B_tot.power(2).sum(0)))
        B_tot /= nB
        B_tot = np.array(B_tot, dtype=np.float32)
#        B_tot = scipy.sparse.coo_matrix(B_tot)
        F_tot *= nB[:, None]

        processed_idx = set([])
        # needed if a patch has more than 1 background component
        processed_idx_prev = set([])
        for _b in np.arange(B_tot.shape[-1]):
            idx_mask = np.where(B_tot[:, _b])[0]
            idx_mask_repeat = processed_idx.intersection(idx_mask)
            if len(idx_mask_repeat) < len(idx_mask):
                processed_idx_prev = processed_idx
            else:
                idx_mask_repeat = processed_idx_prev.intersection(idx_mask)
            processed_idx = processed_idx.union(idx_mask)
            if len(idx_mask_repeat) > 0:
                B_tot[np.array(list(idx_mask_repeat), dtype=np.int), _b] = 0

        b = B_tot
        f = F_tot

        print('******** USING ONE BACKGROUND PER PATCH ******')

    print("Generating background DONE")

    return A_tot, C_tot, YrA_tot, b, f, sn_tot, optional_outputs
