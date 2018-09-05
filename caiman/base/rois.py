#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on Thu Oct 22 13:22:26 2015

@author: agiovann
"""

from builtins import map
from builtins import zip
from builtins import str
from builtins import range

import cv2
import json
import logging
import matplotlib.pyplot as pl
import matplotlib.patches as mpatches
import numpy as np
import os
from past.utils import old_div
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import label, center_of_mass
from skimage.morphology import remove_small_objects, remove_small_holes, dilation
import scipy
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment
import shutil
from skimage.filters import sobel
from skimage.morphology import watershed
from skimage.draw import polygon
import tempfile
import time
import zipfile

from ..motion_correction import tile_and_correct

try:
    cv2.setNumThreads(0)
except:
    pass

def com(A, d1, d2, d3=None):
    """Calculation of the center of mass for spatial components

     Args:
         A:   np.ndarray
              matrix of spatial components (d x K)

         d1:  int
              number of pixels in x-direction

         d2:  int
              number of pixels in y-direction

         d3:  int
              number of pixels in z-direction

     Returns:
         cm:  np.ndarray
              center of mass for spatial components (K x 2 or 3)
    """

    if 'csc_matrix' not in str(type(A)):
        A = scipy.sparse.csc_matrix(A)

    if d3 is None:
        Coor = np.matrix([np.outer(np.ones(d2), np.arange(d1)).ravel(),
                          np.outer(np.arange(d2), np.ones(d1)).ravel()], dtype=A.dtype)
    else:
        Coor = np.matrix([
            np.outer(np.ones(d3), np.outer(np.ones(d2), np.arange(d1)).ravel()).ravel(),
            np.outer(np.ones(d3), np.outer(np.arange(d2), np.ones(d1)).ravel()).ravel(),
            np.outer(np.arange(d3), np.outer(np.ones(d2), np.ones(d1)).ravel()).ravel()],
            dtype=A.dtype)
    cm = (Coor * A / A.sum(axis=0)).T
    return np.array(cm)

def extract_binary_masks_from_structural_channel(Y, min_area_size=30, min_hole_size=15, gSig=5, expand_method='closing', selem=np.ones((3, 3))):
    """Extract binary masks by using adaptive thresholding on a structural channel

    Args:
        Y:                  caiman movie object
                            movie of the structural channel (assumed motion corrected)

        min_area_size:      int
                            ignore components with smaller size

        min_hole_size:      int
                            fill in holes up to that size (donuts)

        gSig:               int
                            average radius of cell

        expand_method:      string
                            method to expand binary masks (morphological closing or dilation)

        selem:              np.array
                            morphological element with which to expand binary masks

    Returns:
        A:                  sparse column format matrix
                            matrix of binary masks to be used for CNMF seeding

        mR:                 np.array
                            mean image used to detect cell boundaries
    """

    mR = Y.mean(axis=0)
    img = cv2.blur(mR, (gSig, gSig))
    img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255.
    img = img.astype(np.uint8)

    th = cv2.adaptiveThreshold(img, np.max(
        img), cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, gSig, 0)
    th = remove_small_holes(th > 0, min_size=min_hole_size)
    th = remove_small_objects(th, min_size=min_area_size)
    areas = label(th)

    A = np.zeros((np.prod(th.shape), areas[1]), dtype=bool)

    for i in range(areas[1]):
        temp = (areas[0] == i + 1)
        if expand_method == 'dilation':
            temp = dilation(temp, selem=selem)
        elif expand_method == 'closing':
            temp = dilation(temp, selem=selem)

        A[:, i] = temp.flatten('F')

    return A, mR

def mask_to_2d(mask):
    # todo todocument
    if mask.ndim > 2:
        _, d1, d2 = np.shape(mask)
        dims = d1, d2
        return scipy.sparse.coo_matrix(np.reshape(mask[:].transpose([1, 2, 0]), (np.prod(dims), -1,), order='F'))
    else:
        dims = np.shape(mask)
        return scipy.sparse.coo_matrix(np.reshape(mask, (np.prod(dims), -1,), order='F'))

def get_distance_from_A(masks_gt, masks_comp, min_dist=10):
    # todo todocument

    _, d1, d2 = np.shape(masks_gt)
    dims = d1, d2
    A_ben = scipy.sparse.csc_matrix(np.reshape(
        masks_gt[:].transpose([1, 2, 0]), (np.prod(dims), -1,), order='F'))
    A_cnmf = scipy.sparse.csc_matrix(np.reshape(
        masks_comp[:].transpose([1, 2, 0]), (np.prod(dims), -1,), order='F'))

    cm_ben = [scipy.ndimage.center_of_mass(mm) for mm in masks_gt]
    cm_cnmf = [scipy.ndimage.center_of_mass(mm) for mm in masks_comp]

    return distance_masks([A_ben, A_cnmf], [cm_ben, cm_cnmf], min_dist)

def nf_match_neurons_in_binary_masks(masks_gt, masks_comp, thresh_cost=.7, min_dist=10, print_assignment=False,
                                     plot_results=False, Cn=None, labels=['Session 1','Session 2'], cmap='viridis', D=None, enclosed_thr=None):
    """
    Match neurons expressed as binary masks. Uses Hungarian matching algorithm

    Args:
        masks_gt: bool ndarray  components x d1 x d2
            ground truth masks

        masks_comp: bool ndarray  components x d1 x d2
            mask to compare to

        thresh_cost: double
            max cost accepted

        min_dist: min distance between cm

        print_assignment:
            for hungarian algorithm

        plot_results: bool

        Cn:
            correlation image or median

        D: list of ndarrays
            list of distances matrices

        enclosed_thr: float
            if not None set distance to at most the specified value when ground truth is a subset of inferred

    Returns:
        idx_tp_1:
            indeces true pos ground truth mask

        idx_tp_2:
            indeces true pos comp

        idx_fn_1:
            indeces false neg

        idx_fp_2:
            indeces false pos

    """

    _, d1, d2 = np.shape(masks_gt)
    dims = d1, d2

    # transpose to have a sparse list of components, then reshaping it to have a 1D matrix red in the Fortran style
    A_ben = scipy.sparse.csc_matrix(np.reshape(
        masks_gt[:].transpose([1, 2, 0]), (np.prod(dims), -1,), order='F'))
    A_cnmf = scipy.sparse.csc_matrix(np.reshape(
        masks_comp[:].transpose([1, 2, 0]), (np.prod(dims), -1,), order='F'))

    # have the center of mass of each element of the two masks
    cm_ben = [scipy.ndimage.center_of_mass(mm) for mm in masks_gt]
    cm_cnmf = [scipy.ndimage.center_of_mass(mm) for mm in masks_comp]

    if D is None:
        #% find distances and matches
        # find the distance between each masks
        D = distance_masks([A_ben, A_cnmf], [cm_ben, cm_cnmf],
                           min_dist, enclosed_thr=enclosed_thr)
        level = 0.98
    else:
        level = .98

    matches, costs = find_matches(D, print_assignment=print_assignment)
    matches = matches[0]
    costs = costs[0]

    #%% compute precision and recall
    TP = np.sum(np.array(costs) < thresh_cost) * 1.
    FN = np.shape(masks_gt)[0] - TP
    FP = np.shape(masks_comp)[0] - TP
    TN = 0

    performance = dict()
    performance['recall'] = old_div(TP, (TP + FN))
    performance['precision'] = old_div(TP, (TP + FP))
    performance['accuracy'] = old_div((TP + TN), (TP + FP + FN + TN))
    performance['f1_score'] = 2 * TP / (2 * TP + FP + FN)
    logging.debug(performance)
    #%%
    idx_tp = np.where(np.array(costs) < thresh_cost)[0]
    idx_tp_ben = matches[0][idx_tp]    # ground truth
    idx_tp_cnmf = matches[1][idx_tp]   # algorithm - comp

    idx_fn = np.setdiff1d(
        list(range(np.shape(masks_gt)[0])), matches[0][idx_tp])

    idx_fp = np.setdiff1d(
        list(range(np.shape(masks_comp)[0])), matches[1][idx_tp])

    idx_fp_cnmf = idx_fp

    idx_tp_gt, idx_tp_comp, idx_fn_gt, idx_fp_comp = idx_tp_ben, idx_tp_cnmf, idx_fn, idx_fp_cnmf

    if plot_results:
        try:  # Plotting function
            pl.rcParams['pdf.fonttype'] = 42
            font = {'family': 'Myriad Pro',
                    'weight': 'regular',
                    'size': 10}
            pl.rc('font', **font)
            lp, hp = np.nanpercentile(Cn, [5, 95])
            ses_1 = mpatches.Patch(color='red', label=labels[0])
            ses_2 = mpatches.Patch(color='white', label=labels[1])
            pl.subplot(1, 2, 1)
            pl.imshow(Cn, vmin=lp, vmax=hp, cmap=cmap)
            [pl.contour(norm_nrg(mm), levels=[level], colors='w', linewidths=1)
             for mm in masks_comp[idx_tp_comp]]
            [pl.contour(norm_nrg(mm), levels=[level], colors='r',
                        linewidths=1) for mm in masks_gt[idx_tp_gt]]
            if labels is None:
                pl.title('MATCHES')
            else:
                pl.title('MATCHES: ' + labels[1] + '(w), ' + labels[0] + '(r)')
            pl.legend(handles=[ses_1, ses_2])
            pl.show()
            pl.axis('off')
            pl.subplot(1, 2, 2)
            pl.imshow(Cn, vmin=lp, vmax=hp, cmap=cmap)
            [pl.contour(norm_nrg(mm), levels=[level], colors='w', linewidths=1)
             for mm in masks_comp[idx_fp_comp]]
            [pl.contour(norm_nrg(mm), levels=[level], colors='r',
                        linewidths=1) for mm in masks_gt[idx_fn_gt]]
            if labels is None:
                pl.title('FALSE POSITIVE (w), FALSE NEGATIVE (r)')
            else:
                pl.title(labels[1] + '(w), ' + labels[0] + '(r)')
            pl.legend(handles=[ses_1, ses_2])
            pl.show()
            pl.axis('off')
        except Exception as e:
            logging.warning("not able to plot precision recall: graphics failure")
            logging.warning(e)
    return idx_tp_gt, idx_tp_comp, idx_fn_gt, idx_fp_comp, performance

def register_ROIs(A1, A2, dims, template1=None, template2=None, align_flag=True, 
                  D=None, max_thr = 0, use_opt_flow = True, thresh_cost=.7, 
                  max_dist=10, enclosed_thr=None, print_assignment=False, 
                  plot_results=False, Cn=None, cmap='viridis'):
    """
    Register ROIs across different sessions using an intersection over union 
    metric and the Hungarian algorithm for optimal matching

    Args:
        A1: ndarray or csc_matrix  # pixels x # of components
            ROIs from session 1

        A2: ndarray or csc_matrix  # pixels x # of components
            ROIs from session 2

        dims: list or tuple
            dimensionality of the FOV

        template1: ndarray dims
            template from session 1

        template2: ndarray dims
            template from session 2

        align_flag: bool
            align the templates before matching

        D: ndarray
            matrix of distances in the event they are pre-computed

        max_thr: scalar
            max threshold parameter before binarization    

        use_opt_flow: bool
            use dense optical flow to align templates

        thresh_cost: scalar
            maximum distance considered

        max_dist: scalar
            max distance between centroids

        enclosed_thr: float
            if not None set distance to at most the specified value when ground 
            truth is a subset of inferred

        print_assignment: bool
            print pairs of matched ROIs

        plot_results: bool
            create a plot of matches and mismatches

        Cn: ndarray
            background image for plotting purposes

        cmap: string
            colormap for background image

    Returns:
        matched_ROIs1: list
            indeces of matched ROIs from session 1

        matched_ROIs2: list
            indeces of matched ROIs from session 2

        non_matched1: list
            indeces of non-matched ROIs from session 1

        non_matched2: list
            indeces of non-matched ROIs from session 2

        performance:  list
            (precision, recall, accuracy, f_1 score) with A1 taken as ground truth

        A2: csc_matrix  # pixels x # of components
            ROIs from session 2 aligned to session 1

    """

#    if 'csc_matrix' not in str(type(A1)):
#        A1 = scipy.sparse.csc_matrix(A1)
#    if 'csc_matrix' not in str(type(A2)):
#        A2 = scipy.sparse.csc_matrix(A2)

    if 'ndarray' not in str(type(A1)):
        A1 = A1.toarray()
    if 'ndarray' not in str(type(A2)):
        A2 = A2.toarray()

    if template1 is None or template2 is None:
        align_flag = False

    x_grid, y_grid = np.meshgrid(np.arange(0., dims[0]).astype(
                np.float32), np.arange(0., dims[1]).astype(np.float32))

    if align_flag:  # first align ROIs from session 2 to the template from session 1
        template1 -= template1.min()
        template1 /= template1.max()
        template2 -= template2.min()
        template2 /= template2.max()

        if use_opt_flow:
            template1_norm = np.uint8(template1*(template1 > 0)*255)
            template2_norm = np.uint8(template2*(template2 > 0)*255)
            flow = cv2.calcOpticalFlowFarneback(np.uint8(template1_norm*255),
                                                np.uint8(template2_norm*255),
                                                None,0.5,3,128,3,7,1.5,0)
            x_remap = (flow[:,:,0] + x_grid).astype(np.float32) 
            y_remap = (flow[:,:,1] + y_grid).astype(np.float32)

        else:
            template2, shifts, _, xy_grid = tile_and_correct(template2, template1 - template1.min(),
                                                             [int(
                                                                 dims[0] / 4), int(dims[1] / 4)], [16, 16], [10, 10],
                                                             add_to_movie=template2.min(), shifts_opencv=True)

            dims_grid = tuple(np.max(np.stack(xy_grid, axis=0), axis=0) -
                              np.min(np.stack(xy_grid, axis=0), axis=0) + 1)
            _sh_ = np.stack(shifts, axis=0)
            shifts_x = np.reshape(_sh_[:, 1], dims_grid,
                                  order='C').astype(np.float32)
            shifts_y = np.reshape(_sh_[:, 0], dims_grid,
                                  order='C').astype(np.float32)

            x_remap = (-np.resize(shifts_x, dims) + x_grid).astype(np.float32)
            y_remap = (-np.resize(shifts_y, dims) + y_grid).astype(np.float32)

        A_2t = np.reshape(A2, dims + (-1,), order='F').transpose(2, 0, 1)
        A2 = np.stack([cv2.remap(img.astype(np.float32), x_remap,
                                 y_remap, cv2.INTER_NEAREST) for img in A_2t], axis=0)
        A2 = np.reshape(A2.transpose(1, 2, 0),
                        (A1.shape[0], A_2t.shape[0]), order='F')

    A1 = np.stack([a*(a>max_thr*a.max()) for a in A1.T]).T 
    A2 = np.stack([a*(a>max_thr*a.max()) for a in A2.T]).T 

    if D is None:
        if 'csc_matrix' not in str(type(A1)):
            A1 = scipy.sparse.csc_matrix(A1)
        if 'csc_matrix' not in str(type(A2)):
            A2 = scipy.sparse.csc_matrix(A2)

        cm_1 = com(A1, dims[0], dims[1])
        cm_2 = com(A2, dims[0], dims[1])
        A1_tr = (A1 > 0).astype(float)
        A2_tr = (A2 > 0).astype(float)
        D = distance_masks([A1_tr, A2_tr], [cm_1, cm_2],
                           max_dist, enclosed_thr=enclosed_thr)

    matches, costs = find_matches(D, print_assignment=print_assignment)
    matches = matches[0]
    costs = costs[0]

    #%% store indeces

    idx_tp = np.where(np.array(costs) < thresh_cost)[0]
    if len(idx_tp) > 0:
        matched_ROIs1 = matches[0][idx_tp]    # ground truth
        matched_ROIs2 = matches[1][idx_tp]   # algorithm - comp
        non_matched1 = np.setdiff1d(
            list(range(D[0].shape[0])), matches[0][idx_tp])
        non_matched2 = np.setdiff1d(
            list(range(D[0].shape[1])), matches[1][idx_tp])
        TP = np.sum(np.array(costs) < thresh_cost) * 1.
    else:
        TP = 0.
        plot_results = False
        matched_ROIs1 = []
        matched_ROIs2 = []
        non_matched1 = list(range(D[0].shape[0]))
        non_matched2 = list(range(D[0].shape[1]))

    #%% compute precision and recall

    FN = D[0].shape[0] - TP
    FP = D[0].shape[1] - TP
    TN = 0

    performance = dict()
    performance['recall'] = old_div(TP, (TP + FN))
    performance['precision'] = old_div(TP, (TP + FP))
    performance['accuracy'] = old_div((TP + TN), (TP + FP + FN + TN))
    performance['f1_score'] = 2 * TP / (2 * TP + FP + FN)
    logging.info(performance)

    if plot_results:
        if Cn is None:
            if template1 is not None:
                Cn = template1
            elif template2 is not None:
                Cn = template2
            else:
                Cn = np.reshape(A1.sum(1) + A2.sum(1), dims, order='F')

        masks_1 = np.reshape(A1.toarray(), dims + (-1,),
                             order='F').transpose(2, 0, 1)
        masks_2 = np.reshape(A2.toarray(), dims + (-1,),
                             order='F').transpose(2, 0, 1)
#        try : #Plotting function
        level = 0.98
        pl.rcParams['pdf.fonttype'] = 42
        font = {'family': 'Myriad Pro',
                'weight': 'regular',
                'size': 10}
        pl.rc('font', **font)
        lp, hp = np.nanpercentile(Cn, [5, 95])
        pl.subplot(1, 2, 1)
        pl.imshow(Cn, vmin=lp, vmax=hp, cmap=cmap)
        [pl.contour(norm_nrg(mm), levels=[level], colors='w', linewidths=1)
         for mm in masks_1[matched_ROIs1]]
        [pl.contour(norm_nrg(mm), levels=[level], colors='r', linewidths=1)
         for mm in masks_2[matched_ROIs2]]
        pl.title('Matches')
        pl.axis('off')
        pl.subplot(1, 2, 2)
        pl.imshow(Cn, vmin=lp, vmax=hp, cmap=cmap)
        [pl.contour(norm_nrg(mm), levels=[level], colors='w', linewidths=1)
         for mm in masks_1[non_matched1]]
        [pl.contour(norm_nrg(mm), levels=[level], colors='r', linewidths=1)
         for mm in masks_2[non_matched2]]
        pl.title('Mismatches')
        pl.axis('off')
#        except Exception as e:
#            logging.warning("not able to plot precision recall usually because we are on travis")
#            logging.warning(e)

    return matched_ROIs1, matched_ROIs2, non_matched1, non_matched2, performance, A2

def register_multisession(A, dims, templates = [None], align_flag=True, 
                          max_thr = 0, use_opt_flow = True, thresh_cost=.7, 
                          max_dist=10, enclosed_thr=None):
    """
    Register ROIs across multiple sessions using an intersection over union metric
    and the Hungarian algorithm for optimal matching. Registration occurs by 
    aligning session 1 to session 2, keeping the union of the matched and 
    non-matched components to register with session 3 and so on.

    Args:
        A: list of ndarray or csc_matrix matrices # pixels x # of components
           ROIs from each session

        dims: list or tuple
            dimensionality of the FOV

        template: list of ndarray matrices of size dims
            templates from each session

        align_flag: bool
            align the templates before matching

        max_thr: scalar
            max threshold parameter before binarization    

        use_opt_flow: bool
            use dense optical flow to align templates

        thresh_cost: scalar
            maximum distance considered

        max_dist: scalar
            max distance between centroids

        enclosed_thr: float
            if not None set distance to at most the specified value when ground 
            truth is a subset of inferred

    Returns:
        A_union: csc_matrix # pixels x # of total distinct components
            union of all kept ROIs 

        assignments: ndarray int of size # of total distinct components x # sessions
            element [i,j] = k if component k from session j is mapped to component
            i in the A_union matrix. If there is no much the value is NaN

        matchings: list of lists
            matchings[i][j] = k means that component j from session i is represented
            by component k in A_union

    """

    n_sessions = len(A)
    templates = list(templates)
    if len(templates) == 1:
        templates = n_sessions*templates

    if n_sessions <= 1:
        raise Exception('number of sessions must be greater than 1')

    A = [a.toarray() if 'ndarray' not in str(type(a)) else a for a in A]

    A_union = A[0].copy()
    matchings = []
    matchings.append(list(range(A_union.shape[-1])))

    for sess in range(1,n_sessions):
        reg_results = register_ROIs(A[sess], A_union, dims, 
                                    template1 = templates[sess], 
                                    template2 = templates[sess-1], align_flag=True, 
                                    max_thr = max_thr, use_opt_flow = use_opt_flow,
                                    thresh_cost = thresh_cost, max_dist = max_dist, 
                                    enclosed_thr = enclosed_thr)

        mat_sess, mat_un, nm_sess, nm_un, _, A2 = reg_results
        logging.info(len(mat_sess))
        A_union = A2.copy()
        A_union[:,mat_un] = A[sess][:,mat_sess]
        A_union = np.concatenate((A_union.toarray(), A[sess][:,nm_sess]), axis = 1)
        new_match = np.zeros(A[sess].shape[-1], dtype = int)
        new_match[mat_sess] = mat_un
        new_match[nm_sess] = range(A2.shape[-1],A_union.shape[-1])
        matchings.append(new_match.tolist())

    assignments = np.empty((A_union.shape[-1], n_sessions))*np.nan
    for sess in range(n_sessions):
        assignments[matchings[sess],sess] = range(len(matchings[sess]))

    return A_union, assignments, matchings

def extract_active_components(assignments, indeces, only = True):
    """
    Computes the indeces of components that were active in a specified set of 
    sessions. 

    Args:
        assignments: ndarray # of components X # of sessions
            assignments matrix returned by function register_multisession

        indeces: list int
            set of sessions to look for active neurons. Session 1 corresponds to a
            pythonic index 0 etc

        only: bool
            If True return components that were active ONLY in these sessions and
            where inactive in all the others. If False components can be active
            in other sessions as well

    Returns:
        components: list int
            indeces of components 

    """

    components = np.where(np.isnan(assignments[:,indeces]).sum(-1) == 0)[0]

    if only:
        not_inds = list(np.setdiff1d(range(assignments.shape[-1]), indeces))
        not_comps = np.where(np.isnan(assignments[:,not_inds]).sum(-1) == 
                             len(not_inds))[0]
        components = np.intersect1d(components, not_comps)

    return components

def norm_nrg(a_):

    a = a_.copy()
    dims = a.shape
    a = a.reshape(-1, order='F')
    indx = np.argsort(a, axis=None)[::-1]
    cumEn = np.cumsum(a.flatten()[indx]**2)
    cumEn /= cumEn[-1]
    a = np.zeros(np.prod(dims))
    a[indx] = cumEn
    return a.reshape(dims, order='F')

def distance_masks(M_s, cm_s, max_dist, enclosed_thr=None):
    """
    Compute distance matrix based on an intersection over union metric. Matrix are compared in order,
    with matrix i compared with matrix i+1

    Args:
        M_s: tuples of 1-D arrays
            The thresholded A matrices (masks) to compare, output of threshold_components

        cm_s: list of list of 2-ples
            the centroids of the components in each M_s

        max_dist: float
            maximum distance among centroids allowed between components. This corresponds to a distance
            at which two components are surely disjoined

        enclosed_thr: float
            if not None set distance to at most the specified value when ground truth is a subset of inferred

    Returns:
        D_s: list of matrix distances

    Raises:
        Exception: 'Nan value produced. Error in inputs'

    """
    D_s = []

    for gt_comp, test_comp, cmgt_comp, cmtest_comp in zip(M_s[:-1], M_s[1:], cm_s[:-1], cm_s[1:]):

        # todo : better with a function that calls itself
        logging.info('New Pair **')
        # not to interfer with M_s
        gt_comp = gt_comp.copy()[:, :]
        test_comp = test_comp.copy()[:, :]

        # the number of components for each
        nb_gt = np.shape(gt_comp)[-1]
        nb_test = np.shape(test_comp)[-1]
        D = np.ones((nb_gt, nb_test))

        cmgt_comp = np.array(cmgt_comp)
        cmtest_comp = np.array(cmtest_comp)
        if enclosed_thr is not None:
            gt_val = gt_comp.T.dot(gt_comp).diagonal()
        for i in range(nb_gt):
            # for each components of gt
            k = gt_comp[:, np.repeat(i, nb_test)] + test_comp
            # k is correlation matrix of this neuron to every other of the test
            for j in range(nb_test):  # for each components on the tests
                dist = np.linalg.norm(cmgt_comp[i] - cmtest_comp[j])
                # we compute the distance of this one to the other ones
                if dist < max_dist:
                    # union matrix of the i-th neuron to the jth one
                    union = k[:, j].sum()
                    # we could have used OR for union and AND for intersection while converting
                    # the matrice into real boolean before

                    # product of the two elements' matrices
                    # we multiply the boolean values from the jth omponent to the ith
                    intersection = np.array(gt_comp[:, i].T.dot(
                        test_comp[:, j]).todense()).squeeze()

                    # if we don't have even a union this is pointless
                    if union > 0:

                        # intersection is removed from union since union contains twice the overlaping area
                        # having the values in this format 0-1 is helpfull for the hungarian algorithm that follows
                        D[i, j] = 1 - 1. * intersection / \
                            (union - intersection)
                        if enclosed_thr is not None:
                            if intersection == gt_val[j] or intersection == gt_val[i]:
                                D[i, j] = min(D[i, j], 0.5)
                    else:
                        D[i, j] = 1.

                    if np.isnan(D[i, j]):
                        raise Exception('Nan value produced. Error in inputs')
                else:
                    D[i, j] = 1

        D_s.append(D)
    return D_s

def find_matches(D_s, print_assignment=False):
    # todo todocument

    matches = []
    costs = []
    t_start = time.time()
    for ii, D in enumerate(D_s):
        # we make a copy not to set changes in the original
        DD = D.copy()
        if np.sum(np.where(np.isnan(DD))) > 0:
            logging.error('Exception: Distance Matrix contains NaN, not allowed!')
            raise Exception('Distance Matrix contains NaN, not allowed!')

        # we do the hungarian
        indexes = linear_sum_assignment(DD)
        indexes2 = [(ind1, ind2) for ind1, ind2 in zip(indexes[0], indexes[1])]
        matches.append(indexes)
        DD = D.copy()
        total = []
        # we want to extract those informations from the hungarian algo
        for row, column in indexes2:
            value = DD[row, column]
            if print_assignment:
                logging.debug(('(%d, %d) -> %f' % (row, column, value)))
            total.append(value)
        logging.debug(('FOV: %d, shape: %d,%d total cost: %f' %
               (ii, DD.shape[0], DD.shape[1], np.sum(total))))
        logging.debug((time.time() - t_start))
        costs.append(total)
        # send back the results in the format we want
    return matches, costs

def link_neurons(matches, costs, max_cost=0.6, min_FOV_present=None):
    """
    Link neurons from different FOVs given matches and costs obtained from the hungarian algorithm

    Args:
        matches: lists of list of tuple
            output of the find_matches function

        costs: list of lists of scalars
            cost associated to each match in matches

        max_cost: float
            maximum allowed value of the 1- intersection over union metric

        min_FOV_present: int
            number of FOVs that must consequently contain the neuron starting from 0. If none
            the neuro must be present in each FOV

    Returns:
        neurons: list of arrays representing the indices of neurons in each FOV

    """
    if min_FOV_present is None:
        min_FOV_present = len(matches)

    neurons = []
    num_neurons = 0
    num_chunks = len(matches) + 1
    for idx in range(len(matches[0][0])):
        neuron = []
        neuron.append(idx)
        for match, cost, _ in zip(matches, costs, list(range(1, num_chunks))):
            rows, cols = match
            m_neur = np.where(rows == neuron[-1])[0].squeeze()
            if m_neur.size > 0:
                if cost[m_neur] <= max_cost:
                    neuron.append(cols[m_neur])
                else:
                    break
            else:
                break
        if len(neuron) > min_FOV_present:
            num_neurons += 1
            neurons.append(neuron)

    neurons = np.array(neurons).T
    logging.info(('num_neurons:' + str(num_neurons)))
    return neurons

def nf_load_masks(file_name, dims):
    # todo todocument

    # load the regions (training data only)
    with open(file_name) as f:
        regions = json.load(f)

    def tomask(coords):
        mask = np.zeros(dims)
        mask[list(zip(*coords))] = 1
        return mask

    masks = np.array([tomask(s['coordinates']) for s in regions])
    return masks

def nf_masks_to_json(binary_masks, json_filename):
    """
    Take as input a tensor of binary mask and produces json format for neurofinder

    Args:
        binary_masks: 3d ndarray (components x dimension 1  x dimension 2)

        json_filename: str

    Returns:
        regions: list of dict
            regions in neurofinder format

    """
    regions = []
    for m in binary_masks:
        coords = [[int(x), int(y)] for x, y in zip(*np.where(m))]
        regions.append({"coordinates": coords})

    with open(json_filename, 'w') as f:
        f.write(json.dumps(regions))

    return regions

#%%
def nf_masks_to_neurof_dict(binary_masks, dataset_name):
    """
    Take as input a tensor of binary mask and produces dict format for neurofinder

    Args:
        binary_masks: 3d ndarray (components x dimension 1  x dimension 2)
        dataset_filename: name of the dataset

    Returns:
        dset: dict
            dataset in neurofinder format to be saved in json
    """

    regions = []
    for m in binary_masks:
        coords = [[int(x), int(y)] for x, y in zip(*np.where(m))]
        regions.append({"coordinates": coords})

    dset = {"regions": regions, "dataset": dataset_name}

    return dset

#%%



def nf_read_roi(fileobj):
    '''
    points = read_roi(fileobj)
    Read ImageJ's ROI format

    Adapted from https://gist.github.com/luispedro/3437255
    '''
# This is based on:
# http://rsbweb.nih.gov/ij/developer/source/ij/io/RoiDecoder.java.html
# http://rsbweb.nih.gov/ij/developer/source/ij/io/RoiEncoder.java.html

    # TODO: Use an enum
    #SPLINE_FIT = 1
    #DOUBLE_HEADED = 2
    #OUTLINE = 4
    #OVERLAY_LABELS = 8
    #OVERLAY_NAMES = 16
    #OVERLAY_BACKGROUNDS = 32
    #OVERLAY_BOLD = 64
    SUB_PIXEL_RESOLUTION = 128
    #DRAW_OFFSET = 256

    pos = [4]

    def get8():
        pos[0] += 1
        s = fileobj.read(1)
        if not s:
            raise IOError('readroi: Unexpected EOF')
        return ord(s)

    def get16():
        b0 = get8()
        b1 = get8()
        return (b0 << 8) | b1

    def get32():
        s0 = get16()
        s1 = get16()
        return (s0 << 16) | s1

    def getfloat():
        v = np.int32(get32())
        return v.view(np.float32)

    magic = fileobj.read(4)
    if magic != 'Iout':
        #        raise IOError('Magic number not found')
        logging.warning('Magic number not found')
    version = get16()

    # It seems that the roi type field occupies 2 Bytes, but only one is used

    roi_type = get8()
    # Discard second Byte:
    get8()

#    if not (0 <= roi_type < 11):
#        logging.error(('roireader: ROI type %s not supported' % roi_type))
#
#    if roi_type != 7:
#
#        logging.error(('roireader: ROI type %s not supported (!= 7)' % roi_type))

    top = get16()
    left = get16()
    bottom = get16()
    right = get16()
    n_coordinates = get16()

    x1 = getfloat()
    y1 = getfloat()
    x2 = getfloat()
    y2 = getfloat()
    stroke_width = get16()
    shape_roi_size = get32()
    stroke_color = get32()
    fill_color = get32()
    subtype = get16()
    if subtype != 0:
        raise ValueError(
            'roireader: ROI subtype %s not supported (!= 0)' % subtype)
    options = get16()
    arrow_style = get8()
    arrow_head_size = get8()
    rect_arc_size = get16()
    position = get32()
    header2offset = get32()

    if options & SUB_PIXEL_RESOLUTION:
        getc = getfloat
        points = np.empty((n_coordinates, 2), dtype=np.float32)
    else:
        getc = get16
        points = np.empty((n_coordinates, 2), dtype=np.int16)
    points[:, 1] = [getc() for i in range(n_coordinates)]
    points[:, 0] = [getc() for i in range(n_coordinates)]
    points[:, 1] += left
    points[:, 0] += top
    points -= 1

    return points

def nf_read_roi_zip(fname, dims, return_names=False):
    # todo todocument

    with zipfile.ZipFile(fname) as zf:
        names = zf.namelist()
        coords = [nf_read_roi(zf.open(n))
                  for n in names]

    def tomask(coords):
        mask = np.zeros(dims)
        coords = np.array(coords)
        rr, cc = polygon(coords[:, 0] + 1, coords[:, 1] + 1)
        mask[rr, cc] = 1

        return mask

    masks = np.array([tomask(s - 1) for s in coords])
    if return_names:
        return masks, names
    else:
        return masks

def nf_merge_roi_zip(fnames, idx_to_keep, new_fold):
    """
    Create a zip file containing ROIs for ImageJ by combining elements from a list of ROI zip files

    Args:
        fnames: str
            list of zip files containing ImageJ rois

        idx_to_keep:   list of lists
            for each zip file index of elements to keep

        new_fold: str
            name of the output zip file (without .zip extension)

    """
    folders_rois = []
    files_to_keep = []
    # unzip the files and keep only the ones that are requested
    for fn, idx in zip(fnames, idx_to_keep):
        dirpath = tempfile.mkdtemp()
        folders_rois.append(dirpath)
        with zipfile.ZipFile(fn) as zf:
            name_rois = zf.namelist()
            logging.debug(len(name_rois))
        zip_ref = zipfile.ZipFile(fn, 'r')
        zip_ref.extractall(dirpath)
        files_to_keep.append([os.path.join(dirpath, ff)
                              for ff in np.array(name_rois)[idx]])
        zip_ref.close()

    os.makedirs(new_fold)
    for fls in files_to_keep:
        for fl in fls:
            shutil.move(fl, new_fold)
    shutil.make_archive(new_fold, 'zip', new_fold)
    shutil.rmtree(new_fold)

def extract_binary_masks_blob(A, neuron_radius, dims, num_std_threshold=1, minCircularity=0.5,
                              minInertiaRatio=0.2, minConvexity=.8):
    """
    Function to extract masks from data. It will also perform a preliminary selectino of good masks based on criteria like shape and size

    Args:
        A: scipy.sparse matris
            contains the components as outputed from the CNMF algorithm

        neuron_radius: float
            neuronal radius employed in the CNMF settings (gSiz)

        num_std_threshold: int
            number of times above iqr/1.349 (std estimator) the median to be considered as threshold for the component

        minCircularity: float
            parameter from cv2.SimpleBlobDetector

        minInertiaRatio: float
            parameter from cv2.SimpleBlobDetector

        minConvexity: float
            parameter from cv2.SimpleBlobDetector

    Returns:
        masks: np.array

        pos_examples:

        neg_examples:

    """
    params = cv2.SimpleBlobDetector_Params()
    params.minCircularity = minCircularity
    params.minInertiaRatio = minInertiaRatio
    params.minConvexity = minConvexity

    # Change thresholds
    params.blobColor = 255

    params.minThreshold = 0
    params.maxThreshold = 255
    params.thresholdStep = 3

    params.minArea = np.pi * ((neuron_radius * .75)**2)

    params.filterByColor = True
    params.filterByArea = True
    params.filterByCircularity = True
    params.filterByConvexity = True
    params.filterByInertia = True

    detector = cv2.SimpleBlobDetector_create(params)

    masks_ws = []
    pos_examples = []
    neg_examples = []

    for count, comp in enumerate(A.tocsc()[:].T):
        logging.debug(count)
        comp_d = np.array(comp.todense())
        gray_image = np.reshape(comp_d, dims, order='F')
        gray_image = (gray_image - np.min(gray_image)) / \
            (np.max(gray_image) - np.min(gray_image)) * 255
        gray_image = gray_image.astype(np.uint8)

        # segment using watershed
        markers = np.zeros_like(gray_image)
        elevation_map = sobel(gray_image)
        thr_1 = np.percentile(gray_image[gray_image > 0], 50)
        iqr = np.diff(np.percentile(gray_image[gray_image > 0], (25, 75)))
        thr_2 = thr_1 + num_std_threshold * iqr / 1.35
        markers[gray_image < thr_1] = 1
        markers[gray_image > thr_2] = 2
        edges = watershed(elevation_map, markers) - 1
        # only keep largest object
        label_objects, _ = ndi.label(edges)
        sizes = np.bincount(label_objects.ravel())

        if len(sizes) > 1:
            idx_largest = np.argmax(sizes[1:])
            edges = (label_objects == (1 + idx_largest))
            edges = ndi.binary_fill_holes(edges)
        else:
            logging.warning('empty component')
            edges = np.zeros_like(edges)

        masks_ws.append(edges)
        keypoints = detector.detect((edges * 200.).astype(np.uint8))

        if len(keypoints) > 0:
            pos_examples.append(count)
        else:
            neg_examples.append(count)

    return np.array(masks_ws), np.array(pos_examples), np.array(neg_examples)


def extract_binary_masks_blob_parallel(A, neuron_radius, dims, num_std_threshold=1, minCircularity=0.5,
                                       minInertiaRatio=0.2, minConvexity=.8, dview=None):
    # todo todocument

    pars = []
    for a in range(A.shape[-1]):
        pars.append([A[:, a], neuron_radius, dims, num_std_threshold,
                     minCircularity, minInertiaRatio, minConvexity])
    if dview is not None:
        res = dview.map_sync(
            extract_binary_masks_blob_parallel_place_holder, pars)
    else:
        res = list(map(extract_binary_masks_blob_parallel_place_holder, pars))

    masks = []
    is_pos = []
    is_neg = []
    for r in res:
        masks.append(np.squeeze(r[0]))
        is_pos.append(r[1])
        is_neg.append(r[2])

    masks = np.dstack(masks).transpose([2, 0, 1])
    return masks, is_pos, is_neg


def extract_binary_masks_blob_parallel_place_holder(pars):
    A, neuron_radius, dims, num_std_threshold, _, minInertiaRatio, minConvexity = pars
    masks_ws, pos_examples, neg_examples = extract_binary_masks_blob(A, neuron_radius,
                                                                     dims, num_std_threshold=num_std_threshold, minCircularity=0.5,
                                                                     minInertiaRatio=minInertiaRatio, minConvexity=minConvexity)
    return masks_ws, len(pos_examples), len(neg_examples)



def extractROIsFromPCAICA(spcomps, numSTD=4, gaussiansigmax=2, gaussiansigmay=2, thresh=None):
    """
    Given the spatial components output of the IPCA_stICA function extract possible regions of interest

    The algorithm estimates the significance of a components by thresholding the components after gaussian smoothing

    Args:
        spcompomps, 3d array containing the spatial components

        numSTD: number of standard deviation above the mean of the spatial component to be considered signiificant
    """

    numcomps, _, _ = spcomps.shape

    allMasks = []
    maskgrouped = []
    for k in range(0, numcomps):
        comp = spcomps[k]
        comp = gaussian_filter(comp, [gaussiansigmay, gaussiansigmax])

        q75, q25 = np.percentile(comp, [75, 25])
        iqr = q75 - q25
        minCompValuePos = np.median(comp) + numSTD * iqr / 1.35
        minCompValueNeg = np.median(comp) - numSTD * iqr / 1.35

        # got both positive and negative large magnitude pixels
        if thresh is None:
            compabspos = comp * (comp > minCompValuePos) - \
                comp * (comp < minCompValueNeg)
        else:
            compabspos = comp * (comp > thresh) - comp * (comp < -thresh)

        labeledpos, n = label(compabspos > 0, np.ones((3, 3)))
        maskgrouped.append(labeledpos)
        for jj in range(1, n + 1):
            tmp_mask = np.asarray(labeledpos == jj)
            allMasks.append(tmp_mask)

    return allMasks, maskgrouped

def detect_duplicates_and_subsets(binary_masks, predictions=None, r_values=None, dist_thr=0.1, min_dist=10,
                                  thresh_subset=0.8):

    cm = [scipy.ndimage.center_of_mass(mm) for mm in binary_masks]
    sp_rois = scipy.sparse.csc_matrix(
        np.reshape(binary_masks, (binary_masks.shape[0], -1)).T)
    D = distance_masks([sp_rois, sp_rois], [cm, cm], min_dist)[0]
    np.fill_diagonal(D, 1)
    overlap = sp_rois.T.dot(sp_rois).toarray()
    sz = np.array(sp_rois.sum(0))
    logging.info(sz.shape)
    overlap = overlap/sz.T
    np.fill_diagonal(overlap, 0)
    # pairs of duplicate indeces

    indeces_orig = np.where((D < dist_thr) | ((overlap) >= thresh_subset))
    indeces_orig = [(a, b) for a, b in zip(indeces_orig[0], indeces_orig[1])]

    use_max_area = False
    if predictions is not None:
        metric = predictions.squeeze()
    elif r_values is not None:
        metric = r_values.squeeze()
    else:
        metric = sz.squeeze()
        logging.debug('***** USING MAX AREA BY DEFAULT')

    overlap_tmp = overlap.copy() >= thresh_subset
    overlap_tmp = overlap_tmp*metric[:, None]

    max_idx = np.argmax(overlap_tmp)
    one, two = np.unravel_index(max_idx, overlap_tmp.shape)
    max_val = overlap_tmp[one, two]

    indeces_to_keep = []
    indeces_to_remove = []
    while max_val > 0:
        one, two = np.unravel_index(max_idx, overlap_tmp.shape)
        if metric[one] > metric[two]:
            #indeces_to_keep.append(one)
            overlap_tmp[:,two] = 0
            overlap_tmp[two,:] = 0
            indeces_to_remove.append(two)
            #if two in indeces_to_keep:
            #    indeces_to_keep.remove(two)
        else:
            overlap_tmp[:,one] = 0
            overlap_tmp[one,:] = 0
            indeces_to_remove.append(one)
            #indeces_to_keep.append(two)
            #if one in indeces_to_keep:
            #    indeces_to_keep.remove(one)

        max_idx = np.argmax(overlap_tmp)
        one, two = np.unravel_index(max_idx, overlap_tmp.shape)
        max_val = overlap_tmp[one, two]

    #indeces_to_remove = np.setdiff1d(np.unique(indeces_orig),indeces_to_keep)
    indeces_to_keep = np.setdiff1d(np.unique(indeces_orig),indeces_to_remove)

#    if len(indeces) > 0:
#        if use_max_area:
#            # if is to  deal with tie breaks in case of same area
#            indeces_keep = np.argmax([[overlap[sec, frst], overlap[frst, sec]]
#                    for frst, sec in indeces], 1)
#            indeces_remove = np.argmin([[overlap[sec, frst], overlap[frst, sec]]
#                    for frst, sec in indeces], 1)
#
#
#        else: #use CNN
#            indeces_keep = np.argmin([[metric[sec], metric[frst]]
#                for frst, sec in indeces], 1)
#            indeces_remove = np.argmax([[metric[sec], metric[frst]]
#                for frst, sec in indeces], 1)
#
#        indeces_keep = np.unique([elms[ik] for ik, elms in
#                                      zip(indeces_keep, indeces)])
#        indeces_remove = np.unique([elms[ik] for ik, elms in
#                                       zip(indeces_remove, indeces)])
#
#        multiple_appearance = np.intersect1d(indeces_keep,indeces_remove)
#        for mapp in multiple_appearance:
#            indeces_remove.remove(mapp)
#    else:
#        indeces_keep = []
#        indeces_remove = []
#        indeces_keep = []
#        indeces_remove = []


    return indeces_orig, indeces_to_keep, indeces_to_remove, D, overlap

def detect_duplicates(file_name, dist_thr=0.1, FOV=(512, 512)):
    """
    Removes duplicate ROIs from file file_name

    Args:
        file_name:  .zip file with all rois

        dist_thr:   distance threshold for duplicate detection

        FOV:        dimensions of the FOV

    Returns:
        duplicates  : list of indeces with duplicate entries

        ind_keep    : list of kept indeces

    """
    rois = nf_read_roi_zip(file_name, FOV)
    cm = [scipy.ndimage.center_of_mass(mm) for mm in rois]
    sp_rois = scipy.sparse.csc_matrix(
        np.reshape(rois, (rois.shape[0], np.prod(FOV))).T)
    D = distance_masks([sp_rois, sp_rois], [cm, cm], 10)[0]
    np.fill_diagonal(D, 1)
    indeces = np.where(D < dist_thr)      # pairs of duplicate indeces

    ind = list(np.unique(indeces[1][indeces[1] > indeces[0]]))
    ind_keep = list(set(range(D.shape[0])) - set(ind))
    duplicates = list(np.unique(np.concatenate((indeces[0], indeces[1]))))

    return duplicates, ind_keep
