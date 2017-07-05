# -*- coding: utf-8 -*-
"""Merging of spatially overlapping components that are temporally correlated
Created on Tue Sep  8 16:23:57 2015

@author: agiovann
"""
#\package caiman/source_extraction/cnmf
#\version   1.0
#\copyright GNU General Public License v2.0

from __future__ import division
from __future__ import print_function
from builtins import range
from scipy.sparse import coo_matrix, csgraph, csc_matrix, lil_matrix
import scipy
import numpy as np
from .spatial import update_spatial_components
from .temporal import update_temporal_components
from .deconvolution import constrained_foopsi

#%%


def merge_components(Y, A, b, C, f, S, sn_pix, temporal_params, spatial_params, dview=None, thr=0.85,
                     fast_merge=True, mx=1000, bl=None, c1=None, sn=None, g=None):
    """ Merging of spatially overlapping components that have highly correlated temporal activity

    The correlation threshold for merging overlapping components is user specified in thr

Parameters:
-----------     

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

bl:        
     baseline for fluorescence trace for each row in C
c1:        
     initial concentration for each row in C
g:         
     discrete time constant for each row in C
sn:        
     noise level for each row in C

Returns:
--------

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

    Raise:
    -----
    Exception("The number of elements of bl\c1\g\sn must match the number of components")


    See Also:
    --------
    """
    #tests and initialization
    nr = A.shape[1]
    if bl is not None and len(bl) != nr:
        raise Exception("The number of elements of bl must match the number of components")
    if c1 is not None and len(c1) != nr:
        raise Exception("The number of elements of c1 must match the number of components")
    if sn is not None and len(sn) != nr:
        raise Exception("The number of elements of sn must match the number of components")
    if g is not None and len(g) != nr:
        raise Exception("The number of elements of g must match the number of components")

    [d, t] = np.shape(Y)

    # % find graph of overlapping spatial components
    A_corr = scipy.sparse.triu(A.T * A)
    A_corr.setdiag(0)
    A_corr = A_corr.tocsc()
    FF2 = A_corr > 0
    C_corr = scipy.sparse.csc_matrix(A_corr.shape)
    for ii in range(nr):
        overlap_indeces = A_corr[ii,:].nonzero()[1]
        if len(overlap_indeces)>0:
            #we chesk the correlation of the calcium traces for eahc overlapping components
            corr_values = [scipy.stats.pearsonr(C[ii,:],C[jj,:])[0] for jj in overlap_indeces]
            C_corr[ii,overlap_indeces] = corr_values
    
    FF1 = (C_corr+C_corr.T) > thr
    FF3 = FF1.multiply(FF2)

    nb, connected_comp = csgraph.connected_components(FF3)  # % extract connected components

    p = temporal_params['p']
    list_conxcomp = []
    for i in range(nb): # we list them
        if np.sum(connected_comp == i) > 1:
            list_conxcomp.append((connected_comp == i).T)
    list_conxcomp = np.asarray(list_conxcomp).T

    if list_conxcomp.ndim > 1:
        cor = np.zeros((np.shape(list_conxcomp)[1], 1))
        for i in range(np.size(cor)): #
            fm = np.where(list_conxcomp[:, i])[0]
            for j1 in range(np.size(fm)):
                for j2 in range(j1 + 1, np.size(fm)):
                    cor[i] = cor[i] + C_corr[fm[j1], fm[j2]]

        if not fast_merge:
            Y_res = Y - A.dot(C) #residuals=background=noise
        if np.size(cor) > 1:
            ind = np.argsort(np.squeeze(cor))[::-1]  #we get the size (indeces)
        else:
            ind = [0]

        nbmrg = min((np.size(ind), mx))   # number of merging operations

        #we initialize the values
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
            merged_ROIs.append(merged_ROI)

            #we l2 the traces to have normalization values
            C_to_norm = np.sqrt([computedC.dot(computedC) for computedC in C[merged_ROI]])
            if fast_merge:
                # from here we are computing initial values for C and A
                Acsc = A.tocsc()[:, merged_ROI]
                Ctmp = np.array(C)[merged_ROI, :]
                print((merged_ROI.T))

                #we normalize the values of different A's to be able to compare them efficiently. we then sum them
                computedA = Acsc.dot(scipy.sparse.diags(C_to_norm, 0, (len(C_to_norm), len(C_to_norm)))).sum(axis=1)

                for _ in range(10): # we operate a rank one NMF, refining it multiple times (see cnmf demos )
                    computedC = Acsc.T.dot(computedA).T.dot(Ctmp) / (computedA.T * computedA)
                    computedA = Acsc.dot(Ctmp.dot(computedC.T)) / (computedC * computedC.T)

                # then we re-normalize them using A_to_norm
                A_to_norm = np.sqrt(computedA.T.dot(computedA)[0, 0] / Acsc.power(2).sum(0).max())
                computedA /= A_to_norm
                computedC *= A_to_norm

                #this is a  big normalization value that for every one of the merged neuron
                C_to_norm = np.sqrt(np.ravel(Acsc.power(2).sum(axis=0)) * np.sum(Ctmp ** 2, axis=1))
                indx = np.argmax(C_to_norm)
                # we then compute the traces ( deconvolution ) to have a clean c and noise in the background
                if g is not None:
                    computedC, bm, cm, gm, sm, ss = constrained_foopsi(
                        np.array(computedC).squeeze(), g=g[merged_ROI[indx]], **temporal_params)
                else:
                    computedC, bm, cm, gm, sm, ss = constrained_foopsi(
                        np.array(computedC).squeeze(), g=None, **temporal_params)

                A_merged[:, i] = computedA
                C_merged[i, :] = computedC
                S_merged[i, :] = ss[:t]
                bl_merged[i] = bm
                c1_merged[i] = cm
                sn_merged[i] = sm
                g_merged[i, :] = gm
            else:
                A_merged[:, i] = lil_matrix((A.tocsc()[:, merged_ROI].dot(
                    scipy.sparse.diags(C_to_norm, 0, (len(C_to_norm), len(C_to_norm))))).sum(axis=1))

                Y_res += A.tocsc()[:, merged_ROI].dot(C[merged_ROI, :])

                computedA_1 = scipy.sparse.linalg.spsolve(scipy.sparse.diags(
                    C_to_norm, 0, (len(C_to_norm), len(C_to_norm))), csc_matrix(C[merged_ROI, :]))

                computedA_2 = (computedA_1).mean(axis=0)
                ff = np.nonzero(A_merged[:, i])[0]

                computedC, _, _, __, _, bl__, c1__, sn__, g__, YrA = update_temporal_components(
                    np.asarray(Y_res[ff, :]), A_merged[ff, i], b[ff],
                    computedA_2, f, dview=dview, bl=None, c1=None, sn=None, g=None, **temporal_params)

                computedA, bb, computedC, f = update_spatial_components(
                    np.asarray(Y_res), computedC, f, A_merged[:, i],sn=sn_pix, dview=dview, **spatial_params)

                A_merged[:, i] = computedA.tocsr()

                computedC, _, _, _, _, ss, bl__, c1__, sn__, g__, YrA = update_temporal_components(
                    Y_res[ff, :], A_merged[ff, i], bb[ff], computedC, f, dview=dview, bl=bl__,
                    c1=c1__, sn=sn__, g=g__, **temporal_params)

                C_merged[i, :] = computedC
                S_merged[i, :] = ss
                bl_merged[i] = bl__[0]
                c1_merged[i] = c1__[0]
                sn_merged[i] = sn__[0]
                g_merged[i, :] = g__[0]
                if i + 1 < nbmrg:
                    Y_res[ff, :] = Y_res[ff, :] - A_merged[ff, i] * computedC
        #we want to remove merged neuron from the initial part and replace them with merged ones
        neur_id = np.unique(np.hstack(merged_ROIs))
        good_neurons = np.setdiff1d(list(range(nr)), neur_id)
        A = scipy.sparse.hstack((A.tocsc()[:, good_neurons], A_merged.tocsc()))
        C = np.vstack((C[good_neurons, :], C_merged))
        #we continue for the variables
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
        nr = nr - len(neur_id) + nbmrg

    else:
        print('No neurons merged!')
        merged_ROIs = []

    return A, C, nr, merged_ROIs, S, bl, c1, sn, g
