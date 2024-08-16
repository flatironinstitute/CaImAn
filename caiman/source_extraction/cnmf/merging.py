#!/usr/bin/env python

"""
Merging of spatially overlapping components that are temporally correlated
"""

import logging
import numpy as np
import scipy
from scipy.sparse import csgraph, csc_matrix, lil_matrix, csr_matrix

from caiman.source_extraction.cnmf.deconvolution import constrained_foopsi


def merge_components(Y, A, b, C, R, f, S, sn_pix, temporal_params,
                     spatial_params, dview=None, thr=0.85, fast_merge=True,
                     mx=1000, bl=None, c1=None, sn=None, g=None,
                     merge_parallel=False) -> tuple[scipy.sparse.csc_matrix, np.ndarray, int, list, np.ndarray, float, float, float, float, list, np.ndarray]:

    """ Merging of spatially overlapping components that have highly correlated temporal activity

    The correlation threshold for merging overlapping components is user specified in thr

    Args:
        Y: np.ndarray
            residual movie after subtracting all found components
            (Y_res = Y - A*C - b*f) (d x T)

        A: sparse matrix
            matrix of spatial components (d x K)

        b: np.ndarray
             spatial background (vector of length d)

        C: np.ndarray
             matrix of temporal components (K x T)

        R: np.ndarray
             array of residuals (K x T)

        f:     np.ndarray
             temporal background (vector of length T)

        S:     np.ndarray
             matrix of deconvolved activity (spikes) (K x T)

        sn_pix: ndarray
             noise standard deviation for each pixel

        temporal_params: dictionary
             all the parameters that can be passed to the
             update_temporal_components function

        spatial_params: dictionary
             all the parameters that can be passed to the
             update_spatial_components function

        thr:   scalar between 0 and 1
             correlation threshold for merging (default 0.85)

        mx:    int
             maximum number of merging operations (default 50)

        sn_pix:    nd.array
             noise level for each pixel (vector of length d)

        fast_merge: bool
            if true perform rank 1 merging, otherwise takes best neuron

        bl:
             baseline for fluorescence trace for each row in C
        c1:
             initial concentration for each row in C
        g:
             discrete time constant for each row in C
        sn:
             noise level for each row in C

        merge_parallel: bool
             perform merging in parallel

    Returns:
        A:     sparse matrix
                matrix of merged spatial components (d x K)

        C:     np.ndarray
                matrix of merged temporal components (K x T)

        nr:    int
            number of components after merging

        merged_ROIs: list
            index of components that have been merged

        S:     np.ndarray
                matrix of merged deconvolved activity (spikes) (K x T)

        bl: float
            baseline for fluorescence trace

        c1: float
            initial concentration

        sn: float
            noise level

        g:  float
            discrete time constant

        empty: list
            indices of neurons that were removed, as they were merged with other neurons.

        R:  np.ndarray
            residuals
    Raises:
        Exception "The number of elements of bl, c1, g, sn must match the number of components"
    """

    logger = logging.getLogger("caiman")
    # tests and initialization
    nr = A.shape[1]
    A = csc_matrix(A)
    if bl is not None and len(bl) != nr:
        raise Exception(
            "The number of elements of bl must match the number of components")
    if c1 is not None and len(c1) != nr:
        raise Exception(
            "The number of elements of c1 must match the number of components")
    if sn is not None and len(sn) != nr:
        raise Exception(
            "The number of elements of sn must match the number of components")
    if g is not None and len(g) != nr:
        raise Exception(
            "The number of elements of g must match the number of components")
    if R is None:
        R = np.zeros_like(C)

    [d, t] = np.shape(Y)

    # find graph of overlapping spatial components
    A_corr = scipy.sparse.triu(A.T * A)
    A_corr.setdiag(0)
    A_corr = A_corr.tocsc()
    FF2 = A_corr > 0
    C_corr = scipy.sparse.lil_matrix(A_corr.shape)
    for ii in range(nr):
        overlap_indices = A_corr[ii, :].nonzero()[1]
        if len(overlap_indices) > 0:
            # we chesk the correlation of the calcium traces for each overlapping components
            corr_values = [scipy.stats.pearsonr(C[ii, :], C[jj, :])[
                0] for jj in overlap_indices]
            C_corr[ii, overlap_indices] = corr_values

    FF1 = (C_corr + C_corr.T) > thr
    FF3 = FF1.multiply(FF2)

    nb, connected_comp = csgraph.connected_components(
        FF3)  # % extract connected components

    p = temporal_params['p']
    list_conxcomp_initial = []
    for i in range(nb):  # we list them
        if np.sum(connected_comp == i) > 1:
            list_conxcomp_initial.append((connected_comp == i).T)
    list_conxcomp = np.asarray(list_conxcomp_initial).T

    if list_conxcomp.ndim > 1:
        cor = np.zeros((np.shape(list_conxcomp)[1], 1))
        for i in range(np.size(cor)):
            fm = np.where(list_conxcomp[:, i])[0]
            for j1 in range(np.size(fm)):
                for j2 in range(j1 + 1, np.size(fm)):
                    cor[i] = cor[i] + C_corr[fm[j1], fm[j2]]
        if np.size(cor) > 1:
            # we get the size (indices)
            ind = np.argsort(np.squeeze(cor))[::-1]
        else:
            ind = [0]

        nbmrg = min((np.size(ind), mx))   # number of merging operations

        if merge_parallel:
            merged_ROIs = [np.where(list_conxcomp[:, ind[i]])[0] for i in range(nbmrg)]
            Acsc_mats = [csc_matrix(A[:, merged_ROI]) for merged_ROI in merged_ROIs]
            Ctmp_mats = [C[merged_ROI] + R[merged_ROI] for merged_ROI in merged_ROIs]
            C_to_norms = [np.sqrt(np.ravel(Acsc.power(2).sum(
                    axis=0)) * np.sum(Ctmp ** 2, axis=1)) for (Acsc, Ctmp) in zip(Acsc_mats, Ctmp_mats)]
            indxs = [np.argmax(C_to_norm) for C_to_norm in C_to_norms]
            g_idxs = [merged_ROI[indx] for (merged_ROI, indx) in zip(merged_ROIs, indxs)]
            fms = [fast_merge]*nbmrg
            tps = [temporal_params]*nbmrg
            gs = [g]*nbmrg
            
            if dview is None:
               merge_res = list(map(merge_iter, zip(Acsc_mats, C_to_norms, Ctmp_mats, fms, gs, g_idxs, indxs, tps)))
            elif 'multiprocessin' in str(type(dview)):
               merge_res = list(dview.map(merge_iter, zip(Acsc_mats, C_to_norms, Ctmp_mats, fms, gs, g_idxs, indxs, tps)))
            else:
               merge_res = list(dview.map_sync(merge_iter, zip(Acsc_mats, C_to_norms, Ctmp_mats, fms, gs, g_idxs, indxs, tps)))
               dview.results.clear()        
            bl_merged = np.array([res[0] for res in merge_res])
            c1_merged = np.array([res[1] for res in merge_res])
            A_merged = csc_matrix(scipy.sparse.vstack([csc_matrix(res[2]) for res in merge_res]).T)
            C_merged = np.vstack([res[3] for res in merge_res])
            g_merged = np.vstack([res[4] for res in merge_res])
            sn_merged = np.array([res[5] for res in merge_res])
            S_merged = np.vstack([res[6] for res in merge_res])
            R_merged = np.vstack([res[7] for res in merge_res])
        else:
            # we initialize the values
            A_merged = lil_matrix((d, nbmrg))
            C_merged = np.zeros((nbmrg, t))
            R_merged = np.zeros((nbmrg, t))
            S_merged = np.zeros((nbmrg, t))
            bl_merged = np.zeros((nbmrg, 1))
            c1_merged = np.zeros((nbmrg, 1))
            sn_merged = np.zeros((nbmrg, 1))
            g_merged = np.zeros((nbmrg, p))
            merged_ROIs = []
            for i in range(nbmrg):
                merged_ROI = np.where(list_conxcomp[:, ind[i]])[0]
                logger.info(f'Merging components {merged_ROI}')
                merged_ROIs.append(merged_ROI)
                Acsc = A.tocsc()[:, merged_ROI]
                Ctmp = np.array(C)[merged_ROI, :] + np.array(R)[merged_ROI, :]
                C_to_norm = np.sqrt(np.ravel(Acsc.power(2).sum(
                    axis=0)) * np.sum(Ctmp ** 2, axis=1))
                indx = np.argmax(C_to_norm)
                g_idx = [merged_ROI[indx]]
                bm, cm, computedA, computedC, gm, sm, ss, yra = merge_iteration(Acsc, C_to_norm, Ctmp, fast_merge, g, g_idx,
                                                                                indx, temporal_params)

                A_merged[:, i] = csr_matrix(computedA).T
                C_merged[i, :] = computedC
                R_merged[i, :] = yra
                S_merged[i, :] = ss[:t]
                bl_merged[i] = bm
                c1_merged[i] = cm
                sn_merged[i] = sm
                g_merged[i, :] = gm

        empty = np.ravel((C_merged.sum(1) == 0) + (A_merged.sum(0) == 0))
        if np.any(empty):
            A_merged = A_merged[:, ~empty]
            C_merged = C_merged[~empty]
            R_merged = R_merged[~empty]
            S_merged = S_merged[~empty]
            bl_merged = bl_merged[~empty]
            c1_merged = c1_merged[~empty]
            sn_merged = sn_merged[~empty]
            g_merged = g_merged[~empty]

        if len(merged_ROIs) > 0:
            # we want to remove merged neuron from the initial part and replace them with merged ones
            neur_id = np.unique(np.hstack(merged_ROIs))
            good_neurons = np.setdiff1d(list(range(nr)), neur_id)
            A = scipy.sparse.hstack((A.tocsc()[:, good_neurons], A_merged.tocsc()))
            C = np.vstack((C[good_neurons, :], C_merged))
            # we continue for the variables
            if S is not None:
                S = np.vstack((S[good_neurons, :], S_merged))
            if R is not None:
                R = np.vstack((R[good_neurons, :], R_merged))
            if bl is not None:
                bl = np.hstack((bl[good_neurons], np.array(bl_merged).flatten()))
            if c1 is not None:
                c1 = np.hstack((c1[good_neurons], np.array(c1_merged).flatten()))
            if sn is not None:
                sn = np.hstack((sn[good_neurons], np.array(sn_merged).flatten()))
            if g is not None:
                g = np.vstack(g)[good_neurons]
                if g.shape[1] == 0:
                    g = np.zeros((len(good_neurons), g_merged.shape[1]))
                g = np.vstack((g, g_merged))

            nr = nr - len(neur_id) + len(C_merged)

    else:
        logger.info('No more components merged!')
        merged_ROIs = []
        empty = []

    return A, C, nr, merged_ROIs, S, bl, c1, sn, g, empty, R

def merge_iter(a):
    Acsc, C_to_norm, Ctmp, fast_merge, g, g_idx, indx, temporal_params = a
    res = merge_iteration(Acsc, C_to_norm, Ctmp, fast_merge, g, g_idx,
                          indx, temporal_params)
    return res

def merge_iteration(Acsc, C_to_norm, Ctmp, fast_merge, g, g_idx, indx, temporal_params):
    logger = logging.getLogger("caiman")
    if fast_merge:
        # we normalize the values of different A's to be able to compare them efficiently. we then sum them

        computedA = Acsc.dot(C_to_norm)
        for _ in range(10):
            computedC = np.maximum((Acsc.T.dot(computedA)).dot(Ctmp) /
                                   (computedA.T.dot(computedA)), 0)
            nc = computedC.T.dot(computedC)
            if nc == 0:
                break
            computedA = np.maximum(Acsc.dot(Ctmp.dot(computedC.T)) / nc, 0)
    else:
        logger.info('Simple merging ny taking best neuron')
        computedC = Ctmp[indx]
        computedA = Acsc[:, indx]
    # then we de-normalize them using A_to_norm
    A_to_norm = np.sqrt(computedA.T.dot(computedA)) #/Acsc.power(2).sum(0).max())
    computedA /= A_to_norm
    computedC *= A_to_norm
    r = ((Acsc.T.dot(computedA)).dot(Ctmp))/(computedA.T.dot(computedA)) - computedC
    # we then compute the traces ( deconvolution ) to have a clean c and noise in the background
    c_in =  np.array(computedC+r).squeeze()
    if g is not None:
        deconvC, bm, cm, gm, sm, ss, lam_ = constrained_foopsi(
            c_in, g=g_idx, **temporal_params)
    else:
        deconvC, bm, cm, gm, sm, ss, lam_ = constrained_foopsi(
            c_in, g=None, **temporal_params)
    return bm, cm, computedA, deconvC, gm, sm, ss, c_in - deconvC
