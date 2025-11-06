#!/usr/bin/env python
"""
Mask R-CNN
Common utility functions and classes.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
Revised by Eric Thompson, Changjia Cai, and Manuel Paez 
"""

import colorsys
import logging
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.optimize import linear_sum_assignment
from skimage.draw import polygon2mask
import time 
import torch
from torchvision.transforms.v2 import functional as F
import torchvision.transforms.v2 as T
from typing import Any, Optional

class ScaleImage:
    """
    Scale image so it is between 0-1: works on floats only
    """
    def __call__(self, img):
        min_val, max_val = img.min(), img.max()
        range = max_val - min_val
        return F.normalize_image(img, mean=[min_val, min_val, min_val], std=[range, range, range])  

def bounding_box(mask_coords):
    """
    Calculates the bounding box from a list of mask coordinates.
    Note: Assumes input coordinates are in (y, x) format.

    Args:
        mask_coords (np.ndarray): A NumPy array of shape (N, 2)
                                  where each row is a [y, x] coordinate.

    Returns:
        A list containing two tuples for the top-left (xmin, ymin)
        and bottom-right (xmax, ymax) corners of the bounding box.
    """
    x_vals = mask_coords[:,1]
    y_vals = mask_coords[:,0]
    return [(min(x_vals), min(y_vals)), (max(x_vals), max(y_vals))]

def box_area(bbox):
    """
    Calculates the area of a bounding box.

    Args:
        bbox: A list or tuple representing the bounding box in
              (xmin, ymin, xmax, ymax) format.

    Returns:
        The area of the bounding box.
    """
    return (bbox[3]-bbox[1])*(bbox[2]-bbox[0])

def collate_fn(batch: list[tuple[Any, Any]]):
    """
    Custom collate function for a DataLoader.

    When the dataset returns samples that cannot be automatically stacked (e.g.,
    images of different sizes or targets as dictionaries), this function
    prevents the DataLoader from trying to batch them together. Instead, it
    groups the images and targets into separate tuples.

    Args:
        batch (List[Tuple[Any, Any]]): A list of samples from the dataset,
                                       where each sample is a tuple (e.g., image, target).

    Returns:
        A tuple where the first element is a tuple of all images from the batch,
        and the second element is a tuple of all targets.
    """
    return tuple(zip(*batch))

def create_mask(shape, mask_dict):
    """
    Creates a 2D boolean mask from a dictionary of polygon coordinates.

    This function is designed to work with mask dictionaries that store
    polygon vertices in separate keys for x and y coordinates.

    Args:
        shape (Tuple[int, int]): The desired output shape of the mask (rows, columns).
        mask_dict (Dict): A dictionary containing the keys 'all_points_x' and
                          'all_points_y', which hold the polygon's vertex coordinates.

    Returns:
        np.ndarray: A 2D boolean array of the specified shape, where pixels
                    inside the polygon are True and pixels outside are False.
    """
    x_points, y_points = mask_dict['all_points_x'], mask_dict['all_points_y']
    mask_coords = np.stack([y_points, x_points]).T
    return polygon2mask(shape, mask_coords)

def data_transform(train: bool = False):
    """
    Defines data augmentation and transformation pipelines for object detection.

    Args:
        train (bool): If True, creates a pipeline with data augmentation
                      for training. Otherwise, creates a basic pipeline
                      for validation or testing.

    Returns:
        T.Compose: A composed torchvision transform object.
    """
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(p=0.5))
        transforms.append(T.RandomVerticalFlip(p=0.5))
        transforms.append(T.RandomApply([T.RandomRotation(degrees=(-5, 5))], p=0.5))
        transforms.append(T.ColorJitter(brightness=0.5,
                                        contrast=0.5,
                                        saturation=0.5,
                                        hue=0))
        transforms.append(T.GaussianBlur(kernel_size=(5, 5), sigma=(0.001, 0.3)))
        transforms.append(T.SanitizeBoundingBoxes(min_size=2))

    transforms.append(T.ToDtype(torch.float32, scale=False))
    return T.Compose(transforms)


def distance_masks(M_s:list, cm_s:list[list], max_dist: float, enclosed_thr:Optional[float] = None) -> list:
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
        # not to interfere with M_s
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
            for j in range(nb_test):   # for each components on the tests
                dist = np.linalg.norm(cmgt_comp[i] - cmtest_comp[j])
                                       # we compute the distance of this one to the other ones
                if dist < max_dist:
                                       # union matrix of the i-th neuron to the jth one
                    union = k[:, j].sum()
                                       # we could have used OR for union and AND for intersection while converting
                                       # the matrice into real boolean before

                    # product of the two elements' matrices
                    # we multiply the boolean values from the jth omponent to the ith
                    intersection = np.array(gt_comp[:, i].T.dot(test_comp[:, j]).todense()).squeeze()

                    # if we don't have even a union this is pointless
                    if union > 0:

                    # intersection is removed from union since union contains twice the overlapping area
                        # having the values in this format 0-1 is helpful for the hungarian algorithm that follows
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

def find_matches(D_s, print_assignment: bool = False) -> tuple[list, list]:
    """
    Finds the optimal assignments for a series of cost matrices using the
    Hungarian algorithm (linear sum assignment).

    This function iterates through a list of distance/cost matrices. For each
    matrix, it computes the assignment of rows to columns that minimizes the
    total cost.

    Args:
        D_s (List[np.ndarray]): A list of 2D NumPy arrays, where each array is a
                                cost matrix. `D_s[i][j, k]` represents the cost
                                of assigning row `j` to column `k` in the i-th matrix.
        print_assignment (bool): If True, prints the individual row-column
                                 assignments and their costs for each matrix.

    Returns:
        A tuple containing two lists:
        matches (List[Tuple[np.ndarray, np.ndarray]]): A list where each element
          is a tuple of two arrays `(row_ind, col_ind)`. These arrays contain the
          indices of the optimal assignments for the corresponding cost matrix.
        costs (List[List[float]]): A list where each element is a list of costs
          for the matched pairs in the corresponding assignment.
    """

    matches = []
    costs = []
    t_start = time.time()
    for ii, D in enumerate(D_s):
        # we make a copy not to set changes in the original
        DD = D.copy()
        if np.sum(np.where(np.isnan(DD))) > 0:
            logging.error('Exception: Distance Matrix contains invalid value NaN')
            raise Exception('Distance Matrix contains invalid value NaN')

        # we do the hungarian
        indexes = linear_sum_assignment(DD)
        indexes2 = [(ind1, ind2) for ind1, ind2 in zip(indexes[0], indexes[1])]
        matches.append(indexes)
        DD = D.copy()
        total = []
        # we want to extract those information from the hungarian algo
        for row, column in indexes2:
            value = DD[row, column]
            if print_assignment:
                logging.debug(('(%d, %d) -> %f' % (row, column, value)))
            total.append(value)
        logging.debug(('FOV: %d, shape: %d,%d total cost: %f' % (ii, DD.shape[0], DD.shape[1], np.sum(total))))
        logging.debug((time.time() - t_start))
        costs.append(total)
        # send back the results in the format we want
    return matches, costs

def f1_score(gt_masks, pred_masks, iou_threshold=0.5):
    """
    Calculates the F1 score for a set of ground truth and predicted masks.

    Args:
        gt_masks (np.ndarray): A boolean or integer array of ground truth masks,
                               shaped (num_gt_masks, height, width).
        pred_masks (np.ndarray): A boolean or integer array of predicted masks,
                                 shaped (num_pred_masks, height, width).
        iou_threshold (float): The IoU threshold to consider a predicted mask
                               as a true positive. Default is 0.5.

    Returns:
        float: The calculated F1 score, a value between 0.0 and 1.0.
    """
    if pred_masks.shape[0] == 0 or gt_masks.shape[0] == 0:
        return 0.0
    
    iou_matrix = np.zeros((gt_masks.shape[0], pred_masks.shape[0]))
    for i, gt_mask in enumerate(gt_masks):
        for j, pred_mask in enumerate(pred_masks):
            intersection = np.logical_and(gt_mask, pred_mask).sum()
            union = np.logical_or(gt_mask, pred_mask).sum()
            if union > 0:
                iou_matrix[i, j] = intersection / union

    true_positives = 0
    matched_preds = set()
    for i in range(gt_masks.shape[0]):
        if iou_matrix.shape[1] > 0:
            best_match_idx = np.argmax(iou_matrix[i, :])
            if iou_matrix[i, best_match_idx] > iou_threshold:
                if best_match_idx not in matched_preds:
                    true_positives += 1
                    matched_preds.add(best_match_idx)
    
    false_positives = pred_masks.shape[0] - len(matched_preds)
    false_negatives = gt_masks.shape[0] - true_positives
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0


def nf_match_neurons_in_binary_masks(masks_gt,
                                     masks_comp,
                                     thresh_cost=.7,
                                     min_dist=10,
                                     print_assignment=False,
                                     plot_results=False,
                                     Cn=None,
                                     labels=['Session 1', 'Session 2'],
                                     cmap='gray',
                                     D=None,
                                     enclosed_thr=None,
                                     colors=['red', 'white']):
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
            indices true pos ground truth mask

        idx_tp_2:
            indices true pos comp

        idx_fn_1:
            indices false neg

        idx_fp_2:
            indices false pos

    """

    _, d1, d2 = np.shape(masks_gt)
    dims = d1, d2

    # transpose to have a sparse list of components, then reshaping it to have a 1D matrix red in the Fortran style
    A_ben = scipy.sparse.csc_matrix(np.reshape(masks_gt[:].transpose([1, 2, 0]), (
        np.prod(dims),
        -1,
    ), order='F'))
    A_cnmf = scipy.sparse.csc_matrix(np.reshape(masks_comp[:].transpose([1, 2, 0]), (
        np.prod(dims),
        -1,
    ), order='F'))

    # have the center of mass of each element of the two masks
    cm_ben  = [scipy.ndimage.center_of_mass(mm) for mm in masks_gt]
    cm_cnmf = [scipy.ndimage.center_of_mass(mm) for mm in masks_comp]

    if D is None:
        # find distances and matches
        # find the distance between each masks
        D = distance_masks([A_ben, A_cnmf], [cm_ben, cm_cnmf], min_dist, enclosed_thr=enclosed_thr)

    level = 0.98

    matches, costs = find_matches(D, print_assignment=print_assignment)
    matches = matches[0]
    costs = costs[0]

    # compute precision and recall
    TP = np.sum(np.array(costs) < thresh_cost) * 1.
    FN = np.shape(masks_gt)[0] - TP
    FP = np.shape(masks_comp)[0] - TP
    TN = 0

    performance = dict()
    performance['recall'] = TP / (TP + FN)
    performance['precision'] = TP / (TP + FP)
    performance['accuracy'] = (TP + TN) / (TP + FP + FN + TN)
    performance['f1_score'] = 2 * TP / (2 * TP + FP + FN)
    logging.debug(performance)

    idx_tp = np.where(np.array(costs) < thresh_cost)[0]
    idx_tp_ben = matches[0][idx_tp]    # ground truth
    idx_tp_cnmf = matches[1][idx_tp]   # algorithm - comp

    idx_fn = np.setdiff1d(list(range(np.shape(masks_gt)[0])), matches[0][idx_tp])

    idx_fp = np.setdiff1d(list(range(np.shape(masks_comp)[0])), matches[1][idx_tp])

    idx_fp_cnmf = idx_fp

    idx_tp_gt, idx_tp_comp, idx_fn_gt, idx_fp_comp = idx_tp_ben, idx_tp_cnmf, idx_fn, idx_fp_cnmf

    if plot_results:
        #try:   # Plotting function
        plt.rcParams['pdf.fonttype'] = 42
        #font = {'family': 'Myriad Pro', 'weight': 'regular', 'size': 10}
        #pl.rc('font', **font)
        lp, hp = np.nanpercentile(Cn, [5, 95])
        ses_1 = matplotlib.patches.Patch(color=colors[0], label=labels[0])
        ses_2 = matplotlib.patches.Patch(color=colors[1], label=labels[1])
        plt.subplot(1, 2, 1)
        plt.imshow(Cn, vmin=lp, vmax=hp, cmap=cmap)
        #import pdb
        #pdb.set_trace()
        [plt.contour(norm_nrg(mm), levels=[level], colors=colors[1], linewidths=1) for mm in masks_comp[idx_tp_comp]]
        [plt.contour(norm_nrg(mm), levels=[level], colors=colors[0], linewidths=1) for mm in masks_gt[idx_tp_gt]]
        if labels is None:
            plt.title('MATCHES')
        else:
            plt.title('MATCHES: ' + labels[1] + f'({colors[1][0]}), ' + labels[0] + f'({colors[0][0]})')
        plt.legend(handles=[ses_1, ses_2])
        #pl.show()
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(Cn, vmin=lp, vmax=hp, cmap=cmap)
        
        
        [plt.contour(norm_nrg(mm), levels=[level], colors=colors[1], linewidths=1) for mm in masks_comp[idx_fp_comp]]
        [plt.contour(norm_nrg(mm), levels=[level], colors=colors[0], linewidths=1) for mm in masks_gt[idx_fn_gt]]
        if labels is None:
            plt.title(f'FALSE POSITIVE ({colors[1][0]}), FALSE NEGATIVE ({colors[0][0]})')
        else:
            plt.title(labels[1] + f'({colors[1][0]}), ' + labels[0] + f'({colors[0][0]})')
        #pl.legend(handles=[ses_1, ses_2])
        plt.axis('off')
        plt.show()
        plt.tight_layout()
        #except Exception as e:
        #    logging.warning("not able to plot precision recall: graphics failure")
        #    logging.warning(e)
    return idx_tp_gt, idx_tp_comp, idx_fn_gt, idx_fp_comp, performance

def normalize_image(image):
    """
    Normalizes a grayscale image to have values between 0.0 and 1.0
    and ensures the data type is float32.

    Args:
        image (np.ndarray): The input grayscale image as a NumPy array.

    Returns:
        np.ndarray: The normalized image as a float32 NumPy array.
    """
    image_shifted = image - image.min()
    image_normed = image_shifted/image_shifted.max()
    return np.array(image_normed, dtype=np.float32)

def norm_nrg(a_):
    """
    Calculates the normalized cumulative energy map of an array.

    The function flattens the input array, sorts its elements in descending
    order, and computes the normalized cumulative energy. The resulting energy
    values are then placed back into an array with the original shape at the
    locations corresponding to the original element values.

    Args:
        a_ (np.ndarray): The input NumPy array.

    Returns:
        np.ndarray: An array of the same shape as the input, containing the
                    normalized cumulative energy values.
    """
    a = a_.copy()
    dims = a.shape
    a = a.reshape(-1, order='F')
    indx = np.argsort(a, axis=None)[::-1]
    cumEn = np.cumsum(a.flatten()[indx]**2)
    cumEn /= cumEn[-1]
    a = np.zeros(np.prod(dims))
    a[indx] = cumEn
    return a.reshape(dims, order='F')