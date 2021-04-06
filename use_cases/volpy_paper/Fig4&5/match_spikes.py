#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 13:42:25 2020
Matching spikes based on Victor-Purpura distance. We provide two ways to solve the problem:
we may solve it as a linear sum assignment problem, or we use a greedy algorithm to match spikes.
Two methods provide the same result in terms of total number of spikes matched
(might be different pairs of spikes matching). The linear sum assignment method provides more accurate
matches in some cases. The greedy method is much faster. 
@author: caichangjia
"""
import matplotlib.pyplot as plt
import numpy as np
import random
from scipy.optimize import linear_sum_assignment

def compute_distances(s1, s2, max_dist):
    """
    Define a distance matrix of spikes.
    Distances greater than maximum distance are assigned one.

    Parameters
    ----------
    s1,s2 : ndarray
        Spikes time of two methods
    max_dist : int
        Maximum distance allowed between two matched spikes

    Returns
    -------
    D : ndarray
        Distance matrix between two spikes
    """
    D = np.ones((len(s1), len(s2)))
    for i in range(len(s1)):
        for j in range(len(s2)):
            if np.abs(s1[i] - s2[j]) > max_dist:
                D[i, j] = 1
            else:
                # 1.01 is to avoid two pairs of matches 'cross' each other
                D[i, j] = (np.abs(s1[i] - s2[j]))/5/max_dist ** 1.01 
    return D

def match_spikes_linear_sum(D):
    """
    Find matches among spikes by solving linear sum assigment problem.
    Delete matches where their distances are greater than the maximum distance.
    Parameters
    ----------
    D : ndarray
        Distance matrix between two spikes
        
    Returns
    -------
    idx1, idx2 : ndarray
        Matched spikes indexes

    """
    idx1, idx2 = linear_sum_assignment(D)
    del_list = []
    for i in range(len(idx1)):
        if D[idx1[i], idx2[i]] == 1:
            del_list.append(i)
    idx1 = np.delete(idx1, del_list)
    idx2 = np.delete(idx2, del_list)
    return idx1, idx2

def match_spikes_greedy(s1, s2, max_dist):
    """
    Match spikes using the greedy algorithm. Spikes greater than the maximum distance
    are never matched.
    Parameters
    ----------
    s1,s2 : ndarray
        Spike time of two methods
    max_dist : int
        Maximum distance allowed between two matched spikes

    Returns
    -------
    idx1, idx2 : ndarray
        Matched spikes indexes with respect to s1 and s2

    """
    l1 = list(s1.copy())
    l2 = list(s2.copy())
    idx1 = []
    idx2 = []
    temp1 = 0
    temp2 = 0
    while len(l1) * len(l2) > 0:
        #print(np.abs(l1[0] - l2[0]))
        if np.abs(l1[0] - l2[0]) <= max_dist:
            idx1.append(temp1)
            idx2.append(temp2)
            temp1 += 1
            temp2 += 1
            del l1[0]
            del l2[0]
        elif l1[0] < l2[0]:
            temp1 += 1
            del l1[0]
        elif l1[0] > l2[0]:
            temp2 += 1
            del l2[0]
    return idx1, idx2

def compute_F1(s1, s2, idx1, idx2):
    """
    Compute F1 scores, precision and recall.

    Parameters
    ----------
    s1,s2 : ndarray
        Spike time of two methods. Note we assume s1 as ground truth spikes.
    
    idx1, idx2 : ndarray
        Matched spikes indexes with respect to s1 and s2

    Returns
    -------
    F1 : float
        Measures of how well spikes are matched with ground truth spikes. 
        The higher F1 score, the better.
        F1 = 2 * (precision * recall) / (precision + recall)
    precision, recall : float
        Precision and recall rate of spikes matching.
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)

    """
    TP = len(idx1)
    FP = len(s2) - TP
    FN = len(s1) - TP
    
    if len(s1) == 0:
        F1 = np.nan
        precision = np.nan
        recall = np.nan
    else:
        try:    
            precision = TP / (TP + FP)
        except ZeroDivisionError:
            precision = 0
        recall = TP / (TP + FN)
        try:
            F1 = 2 * (precision * recall) / (precision + recall) 
        except ZeroDivisionError:
            F1 = 0
            
    return F1, precision, recall
    
#%% small test
# Note here we assume s1 as our ground truth spikes !!
if __name__ == "__main__":
    random.seed(2020)
    s1 =  np.array(sorted(random.sample(range(5000), 400)))
    s2 =  np.array(sorted(random.sample(range(5000), 1200)))
    D = compute_distances(s1, s2, max_dist=3)
    idx1_linear, idx2_linear = match_spikes_linear_sum(D)
    idx1_greedy, idx2_greedy = match_spikes_greedy(s1, s2, max_dist=3)
    
    height=1
    plt.figure()
    plt.plot(s1, 1*height*np.ones(s1.shape),color='b', marker='.', fillstyle='full', linestyle='none')
    plt.plot(s2, 1.5*height*np.ones(len(s2)),color='orange', marker='.', fillstyle='full', linestyle='none')
    for j in range(len(idx1_linear)):
        plt.plot((s1[idx1_linear[j]], s2[idx2_linear[j]]),(1.1*height, 1.4*height), color='red',alpha=0.5, linewidth=1)
    for j in range(len(idx1_greedy)):
        plt.plot((s1[idx1_greedy[j]], s2[idx2_greedy[j]]),(1.11*height, 1.41*height), color='blue',alpha=0.5, linewidth=1)
    
    # two methods provide the same number of pairs
    print(f'Same number of pairs? {len(idx1_linear) == len(idx1_greedy)}')
    
    # Compute measures
    F1, precision, recall = compute_F1(s1, s2, idx1_greedy, idx2_greedy)
    print(f'F1:{round(F1, 3)}, precision:{round(precision,3)}, recall:{round(recall, 3)}')
    




