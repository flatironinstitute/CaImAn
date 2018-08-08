#!/usr/bin/env python

""" test the principal functions of CaImAn

use for nosetests and continuous integration development.

See Also
------------
caiman/tests/comparison/comparison.py


"""
#%%
#\package None
#\version   1.0
#\copyright GNU General Public License v2.0
#\date Created on june 2017
#\author: Jremie KALFON


from __future__ import division
from __future__ import print_function
from builtins import str
from builtins import range

import copy
import cv2
import glob
import logging
import matplotlib
import numpy as np
import os
import time

try:
    cv2.setNumThreads(0)
except:
    pass

try:
    if __IPYTHON__:
        # this is used for debugging purposes only. allows to reload classes
        # when changed
        get_ipython().magic('load_ext autoreload')
        get_ipython().magic('autoreload 2')
except NameError:
    pass


import caiman as cm
from caiman.components_evaluation import estimate_components_quality
from caiman.motion_correction import MotionCorrect
from caiman.source_extraction.cnmf import cnmf as cnmf
from caiman.tests.comparison import comparison
from caiman.utils.utils import download_demo

# Set up the logger; change this if you like.
# You can log to a file using the filename parameter, or make the output more or less
# verbose by setting level to logging.DEBUG, logging.INFO, logging.WARNING, or logging.ERROR

logging.basicConfig(format=
                          "%(relativeCreated)12d [%(filename)s:%(funcName)20s():%(lineno)s] [%(process)d] %(message)s",
                    # filename="/tmp/caiman.log",
                    level=logging.DEBUG)

# GLOBAL VAR
params_movie = {'fname': ['Sue_2x_3000_40_-46.tif'],
                'niter_rig': 1,
                'max_shifts': (3, 3),  # maximum allow rigid shift
                'splits_rig': 20,  # for parallelization split the movies in  num_splits chuncks across time
                # if none all the splits are processed and the movie is saved
                'num_splits_to_process_rig': None,
                # intervals at which patches are laid out for motion correction
                'p': 1,  # order of the autoregressive system
                'merge_thresh': 0.8,  # merging threshold, max correlation allowed
                'rf': 15,  # half-size of the patches in pixels. rf=25, patches are 50x50
                'stride_cnmf': 6,  # amounpl.it of overlap between the patches in pixels
                'K': 4,  # number of components per patch
                # if dendritic. In this case you need to set init_method to
                # sparse_nmf
                'is_dendrites': False,
                'init_method': 'greedy_roi',
                'gSig': [4, 4],  # expected half size of neurons
                'alpha_snmf': None,  # this controls sparsity
                'final_frate': 30,
                'r_values_min_patch': .7,  # threshold on space consistency
                'fitness_min_patch': -40,  # threshold on time variability
                # threshold on time variability (if nonsparse activity)
                'fitness_delta_min_patch': -40,
                'Npeaks': 10,
                'r_values_min_full': .85,
                'fitness_min_full': - 50,
                'fitness_delta_min_full': - 50,
                'only_init_patch': True,
                'gnb': 1,
                'memory_fact': 1,
                'n_chunks': 10
                }
params_display = {
    'downsample_ratio': .2,
    'thr_plot': 0.9
}

# params_movie = {'fname': [u'./example_movies/demoMovieJ.tif'],
#                 'max_shifts': (2, 2),  # maximum allow rigid shift (2,2)
#                 'niter_rig': 1,
#                 'splits_rig': 14,  # for parallelization split the movies in  num_splits chuncks across time
#                 'num_splits_to_process_rig': None,  # if none all the splits are processed and the movie is saved
#                 'p': 1,  # order of the autoregressive system
#                 'merge_thresh': 0.8,  # merging threshold, max correlation allow
#                 'rf': 20,  # half-size of the patches in pixels. rf=25, patches are 50x50    20
#                 'stride_cnmf': 5,  # amounpl.it of overlap between the patches in pixels
#                 'K': 6,  # number of components per patch
#                 'is_dendrites': False,  # if dendritic. In this case you need to set init_method to sparse_nmf
#                 'init_method': 'greedy_roi',
#                 'gSig': [6, 6],  # expected half size of neurons
#                 'alpha_snmf': None,  # this controls sparsity
#                 'final_frate': 10,
#                 'r_values_min_patch': .7,  # threshold on space consistency
#                 'fitness_min_patch': -40,  # threshold on time variability
#                 # threshold on time variability (if nonsparse activity)
#                 'fitness_delta_min_patch': -40,
#                 'Npeaks': 10,
#                 'r_values_min_full': .85,
#                 'fitness_min_full': - 50,
#                 'fitness_delta_min_full': - 50,
#                 'only_init_patch': True,
#                 'gnb': 1,
#                 'memory_fact': 1,
#                 'n_chunks': 10
#
#                 }


def test_general():
    """  General Test of pipeline with comparison against ground truth
    A shorter version than the demo pipeline that calls comparison for the real test work



        Raises:
      ---------
        params_movie

        params_cnmf

        rig correction

        cnmf on patch

        cnmf full frame

        not able to read the file

        no groundtruth


    """
#\bug
#\warning

    global params_movie
    global params_diplay
    fname = params_movie['fname']
    niter_rig = params_movie['niter_rig']
    max_shifts = params_movie['max_shifts']
    splits_rig = params_movie['splits_rig']
    num_splits_to_process_rig = params_movie['num_splits_to_process_rig']

    cwd = os.getcwd()
    fname = download_demo(fname[0])
    m_orig = cm.load(fname)
    min_mov = m_orig[:400].min()
    comp = comparison.Comparison()
    comp.dims = np.shape(m_orig)[1:]


################ RIG CORRECTION #################
    t1 = time.time()
    mc = MotionCorrect(fname, min_mov,
                       max_shifts=max_shifts, niter_rig=niter_rig, splits_rig=splits_rig,
                       num_splits_to_process_rig=num_splits_to_process_rig,
                       shifts_opencv=True, nonneg_movie=True)
    mc.motion_correct_rigid(save_movie=True)
    m_rig = cm.load(mc.fname_tot_rig)
    bord_px_rig = np.ceil(np.max(mc.shifts_rig)).astype(np.int)
    comp.comparison['rig_shifts']['timer'] = time.time() - t1
    comp.comparison['rig_shifts']['ourdata'] = mc.shifts_rig
###########################################

    if 'max_shifts' not in params_movie:
        fnames = params_movie['fname']
        border_to_0 = 0
    else:  # elif not params_movie.has_key('overlaps'):
        fnames = mc.fname_tot_rig
        border_to_0 = bord_px_rig
        m_els = m_rig

    idx_xy = None
    add_to_movie = -np.nanmin(m_els) + 1  # movie must be positive
    remove_init = 0
    downsample_factor = 1
    base_name = fname[0].split('/')[-1][:-4]
    name_new = cm.save_memmap_each(fnames, base_name=base_name, resize_fact=(
        1, 1, downsample_factor), remove_init=remove_init,
        idx_xy=idx_xy, add_to_movie=add_to_movie, border_to_0=border_to_0)
    name_new.sort()

    if len(name_new) > 1:
        fname_new = cm.save_memmap_join(
            name_new, base_name='Yr', n_chunks=params_movie['n_chunks'], dview=None)
    else:
        logging.warning('One file only, not saving!')
        fname_new = name_new[0]

    Yr, dims, T = cm.load_memmap(fname_new)
    images = np.reshape(Yr.T, [T] + list(dims), order='F')
    Y = np.reshape(Yr, dims + (T,), order='F')

    if np.min(images) < 0:
        # TODO: should do this in an automatic fashion with a while loop at the 367 line
        raise Exception('Movie too negative, add_to_movie should be larger')
    if np.sum(np.isnan(images)) > 0:
        # TODO: same here
        raise Exception(
            'Movie contains nan! You did not remove enough borders')

    Cn = cm.local_correlations(Y)
    Cn[np.isnan(Cn)] = 0
    p = params_movie['p']
    merge_thresh = params_movie['merge_thresh']
    rf = params_movie['rf']
    stride_cnmf = params_movie['stride_cnmf']
    K = params_movie['K']
    init_method = params_movie['init_method']
    gSig = params_movie['gSig']
    alpha_snmf = params_movie['alpha_snmf']

    if params_movie['is_dendrites'] == True:
        if params_movie['init_method'] is not 'sparse_nmf':
            raise Exception('dendritic requires sparse_nmf')
        if params_movie['alpha_snmf'] is None:
            raise Exception('need to set a value for alpha_snmf')


################ CNMF PART PATCH #################
    t1 = time.time()
    cnm = cnmf.CNMF(n_processes=1, k=K, gSig=gSig, merge_thresh=params_movie['merge_thresh'], p=params_movie['p'],
                    dview=None, rf=rf, stride=stride_cnmf, memory_fact=params_movie['memory_fact'],
                    method_init=init_method, alpha_snmf=alpha_snmf, only_init_patch=params_movie[
                        'only_init_patch'],
                    gnb=params_movie['gnb'], method_deconvolution='oasis')
    comp.cnmpatch = copy.copy(cnm)
    comp.cnmpatch.estimates = None
    cnm = cnm.fit(images)
    A_tot = cnm.estimates.A
    C_tot = cnm.estimates.C
    YrA_tot = cnm.estimates.YrA
    b_tot = cnm.estimates.b
    f_tot = cnm.estimates.f
    # DISCARDING
    logging.info(('Number of components:' + str(A_tot.shape[-1])))
    final_frate = params_movie['final_frate']
    # threshold on space consistency
    r_values_min = params_movie['r_values_min_patch']
    # threshold on time variability
    fitness_min = params_movie['fitness_delta_min_patch']
    fitness_delta_min = params_movie['fitness_delta_min_patch']
    Npeaks = params_movie['Npeaks']
    traces = C_tot + YrA_tot
    idx_components, idx_components_bad = estimate_components_quality(
        traces, Y, A_tot, C_tot, b_tot, f_tot, final_frate=final_frate,
        Npeaks=Npeaks, r_values_min=r_values_min, fitness_min=fitness_min,
        fitness_delta_min=fitness_delta_min)
    #######
    A_tot = A_tot.tocsc()[:, idx_components]
    C_tot = C_tot[idx_components]
    comp.comparison['cnmf_on_patch']['timer'] = time.time() - t1
    comp.comparison['cnmf_on_patch']['ourdata'] = [A_tot.copy(), C_tot.copy()]
#################### ########################


################ CNMF PART FULL #################
    t1 = time.time()
    cnm = cnmf.CNMF(n_processes=1, k=A_tot.shape, gSig=gSig, merge_thresh=merge_thresh, p=p, Ain=A_tot, Cin=C_tot,
                    f_in=f_tot, rf=None, stride=None, method_deconvolution='oasis')
    cnm = cnm.fit(images)
    # DISCARDING
    A, C, b, f, YrA, sn = cnm.estimates.A, cnm.estimates.C, cnm.estimates.b, cnm.estimates.f, cnm.estimates.YrA, cnm.estimates.sn
    final_frate = params_movie['final_frate']
    # threshold on space consistency
    r_values_min = params_movie['r_values_min_full']
    # threshold on time variability
    fitness_min = params_movie['fitness_delta_min_full']
    fitness_delta_min = params_movie['fitness_delta_min_full']
    Npeaks = params_movie['Npeaks']
    traces = C + YrA
    idx_components, idx_components_bad, fitness_raw, fitness_delta, r_values = estimate_components_quality(
        traces, Y, A, C, b, f, final_frate=final_frate, Npeaks=Npeaks, r_values_min=r_values_min,
        fitness_min=fitness_min,
        fitness_delta_min=fitness_delta_min, return_all=True)
    ##########
    A_tot_full = A_tot.tocsc()[:, idx_components]
    C_tot_full = C_tot[idx_components]
    comp.comparison['cnmf_full_frame']['timer'] = time.time() - t1
    comp.comparison['cnmf_full_frame']['ourdata'] = [
        A_tot_full.copy(), C_tot_full.copy()]
#################### ########################
    comp.save_with_compare(istruth=False, params=params_movie, Cn=Cn)
    log_files = glob.glob('*_LOG_*')
    try:
        for log_file in log_files:
            os.remove(log_file)
    except:
        logging.warning('Cannot remove log files')
############ assertions ##################
    pb = False
    if (comp.information['differences']['params_movie']):
        logging.error("you need to set the same movie parameters than the ground truth to have a real comparison (use the comp.see() function to explore it)")
        pb = True
    if (comp.information['differences']['params_cnm']):
        logging.error("you need to set the same cnmf parameters than the ground truth to have a real comparison (use the comp.see() function to explore it)")
        pb = True
    if (comp.information['diff']['rig']['isdifferent']):
        logging.error("the rigid shifts are different from the groundtruth ")
        pb = True
    if (comp.information['diff']['cnmpatch']['isdifferent']):
        logging.error("the cnmf on patch produces different results than the groundtruth ")
        pb = True
    if (comp.information['diff']['cnmfull']['isdifferent']):
        logging.error("the cnmf full frame produces different  results than the groundtruth ")
        pb = True

    assert (not pb)
