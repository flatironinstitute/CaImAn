#!/usr/bin/env python
"""
This file is used for performing offline initialization to provide spatial footprints
for FIOLA online analysis.
@author: @caichangjia
"""
import cv2
import glob
import logging
import numpy as np
import os

try:
    cv2.setNumThreads(0)
except:
    pass

import caiman as cm
from caiman.motion_correction import MotionCorrect
from caiman.source_extraction.cnmf import cnmf as cnmf
from caiman.source_extraction.cnmf import params as params
from caiman.summary_images import local_correlations_movie_offline

def run_caiman_init(mov, fnames, mc_dict, opts_dict, quality_dict, save_movie):
    """
    run caiman initialization for FIOLA

    Parameters
    ----------
    mov : ndarray
        input movie
    mc_dict : dict
        parameters for motion correction
    opts_dict : dict
        parameters for source extraction
    quality_dict : dict
        parameters for quality control
    save_movie : bool
        whether to save input movie or not. The default is True.

    Returns
    -------
    cnm2.estimates
        estimates object that contains masks and traces extracted by CaImAn

    """

    # Select file(s) to be processed (download if not present)
    if save_movie:
        m = cm.movie(mov)    
        m.save(fnames)    
        logging.info(f'saving movie of shape {m.shape} into {fnames}')
    else:
        logging.info('not saving movie')

    # First setup some parameters for data and motion correction
    logging.info('beginning motion correction')
    opts = params.CNMFParams(params_dict=mc_dict)
    
    # start a cluster for parallel processing
    c, dview, n_processes = cm.cluster.setup_cluster(
        backend='local', n_processes=None, single_thread=False)

    # % MOTION CORRECTION
    # first we create a motion correction object with the specified parameters
    mc = MotionCorrect(fnames, dview=dview, **opts.get_group('motion'))
    # note that the file is not loaded in memory

    #  Run (piecewise-rigid motion) correction using NoRMCorre
    mc.motion_correct(save_movie=True)

    #  MEMORY MAPPING
    logging.info('beginning memory mapping')
    border_to_0 = 0 if mc.border_nan == 'copy' else mc.border_to_0
    # you can include the boundaries of the FOV if you used the 'copy' option
    # during motion correction, although be careful about the components near
    # the boundaries

    # memory map the file in order 'C'
    fname_new = cm.save_memmap(mc.mmap_file, base_name='memmap_', order='C',
                               border_to_0=border_to_0)  # exclude borders


    # now load the file
    Yr, dims, T = cm.load_memmap(fname_new)
    images = np.reshape(Yr.T, [T] + list(dims), order='F')
    # load frames in python format (T x X x Y)

    #  restart cluster to clean up memory
    cm.stop_server(dview=dview)
    c, dview, n_processes = cm.cluster.setup_cluster(backend='local')

    #   parameters for source extraction and deconvolution
    logging.info('beginning cnmf')
    opts.change_params(params_dict=opts_dict);
    #  RUN CNMF ON PATCHES
    # First extract spatial and temporal components on patches and combine them
    # for this step deconvolution is turned off (p=0). If you want to have
    # deconvolution within each patch change params.patch['p_patch'] to a
    # nonzero value

    opts.change_params({'p': 0})
    cnm = cnmf.CNMF(n_processes, params=opts, dview=dview)
    cnm = cnm.fit(images)

    #  plot contours of found components
    Cns = local_correlations_movie_offline(mc.mmap_file[0],
                                           remove_baseline=True, window=1000, stride=1000,
                                           winSize_baseline=100, quantil_min_baseline=10,
                                           dview=dview)
    Cn = Cns.max(axis=0)
    Cn[np.isnan(Cn)] = 0

    # save results
    cnm.estimates.Cn = Cn
    cnm.save(fname_new[:-5]+'_init.hdf5')

    #  RE-RUN seeded CNMF on accepted patches to refine and perform deconvolution
    logging.info('beginning cnmf refit')
    cnm2 = cnm.refit(images, dview=dview)

    #  COMPONENT EVALUATION
    logging.info('beginning component evaluation')
    cnm2.params.set('quality', quality_dict)
    cnm2.estimates.evaluate_components(images, cnm2.params, dview=dview)
    # PLOT COMPONENTS
    # if display_images:
    #     cnm2.estimates.plot_contours(img=Cn, idx=cnm2.estimates.idx_components)

    # VIEW TRACES (accepted and rejected)
    # if display_images:
    #     cnm2.estimates.view_components(images, img=Cn,
    #                                   idx=cnm2.estimates.idx_components)
    #     cnm2.estimates.view_components(images, img=Cn,
    #                                   idx=cnm2.estimates.idx_components_bad)
    # update object with selected components
    cnm2.estimates.select_components(use_object=True)
    # Extract DF/F values
    cnm2.estimates.detrend_df_f(quantileMin=8, frames_window=250)

    # Show final traces
    # cnm2.estimates.view_components(img=Cn)
    #
    cnm2.estimates.Cn = Cn
    cnm2.estimates.fname_new = fname_new
    cnm2.save(cnm2.mmap_file[:-4] + 'hdf5')
    
    # reconstruct denoised movie (press q to exit)
    #cnm2.estimates.play_movie(images, q_max=99.9, gain_res=2,
    #                              magnification=1,
    #                              bpx=border_to_0,
    #                              include_bck=False)  # background not shown

    # STOP CLUSTER and clean up log files
    cm.stop_server(dview=dview)
    log_files = glob.glob('*_LOG_*')
    for log_file in log_files:
        os.remove(log_file)

    return cnm2.estimates
