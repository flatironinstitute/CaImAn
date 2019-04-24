#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Merging of spatially overlapping components that are temporally correlated
Created on Tue Sep  8 16:23:57 2015

@author: agiovann
"""
#\package caiman/source_extraction/cnmf
#\version   1.0
#\copyright GNU General Public License v2.0

from builtins import range
import numpy as np
import logging
from past.utils import old_div
import scipy
from scipy.sparse import coo_matrix, csgraph, csc_matrix, lil_matrix

from .spatial import update_spatial_components, threshold_components
from .temporal import update_temporal_components
from .deconvolution import constrained_foopsi
from .utilities import update_order_greedy



def merge_components(Y, A, b, C, f, S, sn_pix, temporal_params, spatial_params, dview=None, thr=0.85, fast_merge=True, mx=1000, bl=None, c1=None, sn=None, g=None):
    """ Merging of spatially overlapping components that have highly correlated temporal activity

    The correlation threshold for merging overlapping components is user specified in thr

    Args:
        Y: np.ndarray
            residual movie after subtracting all found components (Y_res = Y - A*C - b*f) (d x T)

        A: sparse matrix
            matrix of spatial components (d x K)

        b: np.ndarray
             spatial background (vector of length d)
        
        C: np.ndarray
             matrix of temporal components (K x T)
        
        f:     np.ndarray
             temporal background (vector of length T)
        
        S:     np.ndarray
             matrix of deconvolved activity (spikes) (K x T)
        
        sn_pix: ndarray
             noise standard deviation for each pixel
        
        temporal_params: dictionary
             all the parameters that can be passed to the update_temporal_components function
        
        spatial_params: dictionary
             all the parameters that can be passed to the update_spatial_components function
        
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
        
        g:  float
            discrete time constant
        
        sn: float
            noise level

    Raises:
        Exception "The number of elements of bl\c1\g\sn must match the number of components"
    """

    #tests and initialization
    nr = A.shape[1]
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

    [d, t] = np.shape(Y)

    # % find graph of overlapping spatial components
    A_corr = scipy.sparse.triu(A.T * A)
    A_corr.setdiag(0)
    A_corr = A_corr.tocsc()
    FF2 = A_corr > 0
    C_corr = scipy.sparse.lil_matrix(A_corr.shape)
    for ii in range(nr):
        overlap_indeces = A_corr[ii, :].nonzero()[1]
        if len(overlap_indeces) > 0:
            # we chesk the correlation of the calcium traces for eahc overlapping components
            corr_values = [scipy.stats.pearsonr(C[ii, :], C[jj, :])[
                0] for jj in overlap_indeces]
            C_corr[ii, overlap_indeces] = corr_values

    FF1 = (C_corr + C_corr.T) > thr
    FF3 = FF1.multiply(FF2)

    nb, connected_comp = csgraph.connected_components(
        FF3)  # % extract connected components

    p = temporal_params['p']
    list_conxcomp = []
    for i in range(nb):  # we list them
        if np.sum(connected_comp == i) > 1:
            list_conxcomp.append((connected_comp == i).T)
    list_conxcomp = np.asarray(list_conxcomp).T

    if list_conxcomp.ndim > 1:
        cor = np.zeros((np.shape(list_conxcomp)[1], 1))
        for i in range(np.size(cor)):
            fm = np.where(list_conxcomp[:, i])[0]
            for j1 in range(np.size(fm)):
                for j2 in range(j1 + 1, np.size(fm)):
                    cor[i] = cor[i] + C_corr[fm[j1], fm[j2]]

#        if not fast_merge:
#            Y_res = Y - A.dot(C) #residuals=background=noise
        if np.size(cor) > 1:
            # we get the size (indeces)
            ind = np.argsort(np.squeeze(cor))[::-1]
        else:
            ind = [0]

        nbmrg = min((np.size(ind), mx))   # number of merging operations

        # we initialize the values
        A_merged = lil_matrix((d, nbmrg))
        C_merged = np.zeros((nbmrg, t))
        S_merged = np.zeros((nbmrg, t))
        bl_merged = np.zeros((nbmrg, 1))
        c1_merged = np.zeros((nbmrg, 1))
        sn_merged = np.zeros((nbmrg, 1))
        g_merged = np.zeros((nbmrg, p))
        merged_ROIs = []

        for i in range(nbmrg):
            merged_ROI = np.where(list_conxcomp[:, ind[i]])[0]
            logging.info('Merging components {}'.format(merged_ROI))
            merged_ROIs.append(merged_ROI)

            Acsc = A.tocsc()[:, merged_ROI]
            Ctmp = np.array(C)[merged_ROI, :]


            # # we l2 the traces to have normalization values
            # C_to_norm = np.sqrt([computedC.dot(computedC)
            #                      for computedC in C[merged_ROI]])
#            fast_merge = False

            # from here we are computing initial values for C and A


            # this is a  big normalization value that for every one of the merged neuron
            C_to_norm = np.sqrt(np.ravel(Acsc.power(2).sum(
                axis=0)) * np.sum(Ctmp ** 2, axis=1))
            indx = np.argmax(C_to_norm)
            g_idx = [merged_ROI[indx]]

            bm, cm, computedA, computedC, gm, sm, ss = merge_iteration(Acsc, C_to_norm, Ctmp, fast_merge, g, g_idx,
                                                                       indx, temporal_params)

            A_merged[:, i] = computedA
            C_merged[i, :] = computedC
            S_merged[i, :] = ss[:t]
            bl_merged[i] = bm
            c1_merged[i] = cm
            sn_merged[i] = sm
            g_merged[i, :] = gm

        empty = np.ravel((C_merged.sum(1) == 0) + (A_merged.sum(0) == 0))
        if np.any(empty):
            A_merged = A_merged[:, ~empty]
            C_merged = C_merged[~empty]
            S_merged = S_merged[~empty]
            bl_merged = bl_merged[~empty]
            c1_merged = c1_merged[~empty]
            sn_merged = sn_merged[~empty]
            g_merged = g_merged[~empty]

        # we want to remove merged neuron from the initial part and replace them with merged ones
        neur_id = np.unique(np.hstack(merged_ROIs))
        good_neurons = np.setdiff1d(list(range(nr)), neur_id)
        A = scipy.sparse.hstack((A.tocsc()[:, good_neurons], A_merged.tocsc()))
        C = np.vstack((C[good_neurons, :], C_merged))
        # we continue for the variables
        if S is not None:
            S = np.vstack((S[good_neurons, :], S_merged))
        if bl is not None:
            bl = np.hstack((bl[good_neurons], np.array(bl_merged).flatten()))
        if c1 is not None:
            c1 = np.hstack((c1[good_neurons], np.array(c1_merged).flatten()))
        if sn is not None:
            sn = np.hstack((sn[good_neurons], np.array(sn_merged).flatten()))
        if g is not None:
            g = np.vstack((np.vstack(g)[good_neurons], g_merged))
        nr = nr - len(neur_id) + len(C_merged)

    else:
        logging.info('No more components merged!')
        merged_ROIs = []

    return A, C, nr, merged_ROIs, S, bl, c1, sn, g


def merge_iteration(Acsc, C_to_norm, Ctmp, fast_merge, g, g_idx, indx, temporal_params):
    if fast_merge:
        # we normalize the values of different A's to be able to compare them efficiently. we then sum them
        computedA = Acsc.dot(scipy.sparse.diags(
            C_to_norm, 0, (len(C_to_norm), len(C_to_norm)))).sum(axis=1)

        # we operate a rank one NMF, refining it multiple times (see cnmf demos )
        for _ in range(10):
            computedC = np.maximum(Acsc.T.dot(computedA).T.dot(
                Ctmp) / (computedA.T * computedA), 0)
            if computedC * computedC.T == 0:
                break
            computedA = np.maximum(
                Acsc.dot(Ctmp.dot(computedC.T)) / (computedC * computedC.T), 0)
    else:
        logging.info('Simple merging ny taking best neuron')
        computedC = Ctmp[indx]
        computedA = Acsc[:, indx]
    # then we de-normalize them using A_to_norm
    A_to_norm = np.sqrt(computedA.T.dot(computedA)[
                            0, 0] / Acsc.power(2).sum(0).max())
    computedA /= A_to_norm
    computedC *= A_to_norm
    # we then compute the traces ( deconvolution ) to have a clean c and noise in the background
    if g is not None:
        computedC, bm, cm, gm, sm, ss, lam_ = constrained_foopsi(
            np.array(computedC).squeeze(), g=g_idx, **temporal_params)
    else:
        computedC, bm, cm, gm, sm, ss, lam_ = constrained_foopsi(
            np.array(computedC).squeeze(), g=None, **temporal_params)
    return bm, cm, computedA, computedC, gm, sm, ss
