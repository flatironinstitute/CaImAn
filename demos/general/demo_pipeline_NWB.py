#!/usr/bin/env python

"""
This script follows closely the demo_pipeline.py script but uses the
Neurodata Without Borders (NWB) file format for loading the input and saving
the output. It is meant as an example on how to use NWB files with CaImAn.

If you provide a filename in the config json, it must be an nwb file.
"""

import argparse
import cv2
from datetime import datetime
from dateutil.tz import tzlocal
import glob
import logging
import matplotlib.pyplot as plt
import numpy as np
import os

try:
    cv2.setNumThreads(0)
except:
    pass

import caiman
from caiman.motion_correction import MotionCorrect
from caiman.paths import caiman_datadir
from caiman.source_extraction.cnmf import cnmf as cnmf
from caiman.source_extraction.cnmf import params as params
from caiman.utils.utils import download_demo

# TODO: Various code here isn't careful enough with paths and may produce files in
#       places alongside their sources, rather than in temporary directories. Come back
#       later and fix this once we've worked out where everything should go.

def main():
    cfg = handle_args()

    if cfg.logfile:
        logging.basicConfig(format=
            "[%(filename)s:%(funcName)20s():%(lineno)s] %(message)s",
            level=logging.INFO,
            filename=cfg.logfile)
        # You can make the output more or less verbose by setting level to logging.DEBUG, logging.INFO, logging.WARNING, or logging.ERROR
    else:
        logging.basicConfig(format=
            "[%(filename)s:%(funcName)20s():%(lineno)s] %(message)s",
            level=logging.INFO)

    opts = params.CNMFParams(params_from_file=cfg.configfile)
    if cfg.input is not None:
        opts.change_params({"data": {"fnames": cfg.input}})
        # TODO throw error if it is not an nwb file
    if not opts.data['fnames']: # Set neither by CLI arg nor through JSON, so use default data
        # This demo will either accept a nwb file or it'll convert an existing movie into one
        # default path, no input was provide, so we'll take the "Sue" sample data, convert it to a nwb, and then ingest it.
        # The convert target is the original fn, with the extension swapped to nwb.
        fr = float(15)      # imaging rate in frames per second
        pre_convert    = download_demo('Sue_2x_3000_40_-46.tif')
        convert_target = os.path.join(caiman_datadir(), 'Sue_2x_3000_40_-46.nwb')

        orig_movie = caiman.load(pre_convert, fr=fr)
        # save file in NWB format with various additional info
        orig_movie.save(convert_target,
                        sess_desc="test",
                        identifier="demo 1",
                        imaging_plane_description='single plane',
                        emission_lambda=520.0, indicator='GCAMP6f',
                        location='parietal cortex',
                        experimenter='Sue Ann Koay', lab_name='Tank Lab',
                        institution='Princeton U',
                        experiment_description='Experiment Description',
                        session_id='Session 1',
                        var_name_hdf5='TwoPhotonSeries')
        fnames = [convert_target]
        opts.change_params({'data': {'fnames': fnames, 'var_name_hdf5': 'TwoPhotonSeries'}})
    elif cfg.input[0].endswith('.nwb'):
        if os.path.isfile(cfg.input[0]):
            # We were handed at least one nwb file and it exists either right here or as an absolute path
            for fn in cfg.input:
                if not os.path.isfile(fn):
                    raise Exception(f"Could not find input file {fn}")
                if not fn.endswith('.nwb'):
                    raise Exception(f"Cannot mix nwb and other file formats in input with this demo")
            fnames = cfg.input
        elif os.path.isfile(os.path.join(caiman_datadir(), cfg.input[0])):
            # We were handed at least one nwb file and it exists, but in the caiman_datadir()
            # So repeat the logic above except look in caiman_datadir()
            for fn in cfg.input:
                if not os.path.isfile(os.path.join(caiman_datadir(), fn)):
                    raise Exception(f"Could not find input file {fn}")
                if not fn.endswith('.nwb'):
                    raise Exception(f"Cannot mix nwb and other file formats in input with this demo")
            fnames = list(map(lambda n: os.path.join(caiman_datadir, n), cfg.input))

        else: # Someone mentioned an nwb file but we can't find it!
            raise Exception(f"Could not find referenced input file {cfg.input[0]}")
    else:
        raise Exception("If you're providing your own data to this demo, you must provide it in nwb format or pre-convert it yourself")

    # estimates save path can be same or different from raw data path
    save_path = os.path.splitext(fnames[0])[0] + '_CNMF_estimates.nwb'

    # We're done with all the file input parts

    m_orig = caiman.load_movie_chain(fnames, var_name_hdf5=opts.data['var_name_hdf5'])

    # play the movie (optional)
    # playing the movie using opencv. It requires loading the movie in memory.
    # To close the video press q

    if not cfg.no_play:
        ds_ratio = 0.2
        moviehandle = m_orig.resize(1, 1, ds_ratio)
        moviehandle.play(q_max=99.5, fr=60, magnification=2)

    # start a cluster for parallel processing
    c, dview, n_processes = caiman.cluster.setup_cluster(backend=cfg.cluster_backend, n_processes=cfg.cluster_nproc)

    # Motion Correction
    # first we create a motion correction object with the specified parameters
    mc = MotionCorrect(fnames, dview=dview, var_name_hdf5=opts.data['var_name_hdf5'], **opts.get_group('motion'))
    # note that the file is not loaded in memory

    # Run (piecewise-rigid motion) correction using NoRMCorre
    mc.motion_correct(save_movie=True)

    # compare with original movie
    if not cfg.no_play:
        m_orig = caiman.load_movie_chain(fnames, var_name_hdf5=opts.data['var_name_hdf5'])
        m_els = caiman.load(mc.mmap_file)
        ds_ratio = 0.2
        moviehandle = caiman.concatenate([m_orig.resize(1, 1, ds_ratio) - mc.min_mov*mc.nonneg_movie,
                                      m_els.resize(1, 1, ds_ratio)], axis=2)
        moviehandle.play(fr=60, q_max=99.5, magnification=2)  # press q to exit

    # MEMORY MAPPING
    if mc.border_nan == 'copy':
        border_to_0 = 0
    else:
        border_to_0 = mc.border_to_0
    # you can include the boundaries of the FOV if you used the 'copy' option
    # during motion correction, although be careful about the components near
    # the boundaries

    # memory map the file in order 'C'
    fname_new = caiman.save_memmap(mc.mmap_file, base_name='memmap_', order='C',
                               border_to_0=border_to_0)  # exclude borders

    # now load the file
    Yr, dims, T = caiman.load_memmap(fname_new)
    images = np.reshape(Yr.T, [T] + list(dims), order='F')
    # load frames in python format (T x X x Y)

    # restart cluster to clean up memory
    caiman.stop_server(dview=dview)
    c, dview, n_processes = caiman.cluster.setup_cluster(backend=cfg.cluster_backend, n_processes=cfg.cluster_nproc)

    # RUN CNMF ON PATCHES
    # First extract spatial and temporal components on patches and combine them
    # for this step deconvolution is turned off (p=0)

    saved_p = opts.preprocess['p'] # Save the deconvolution parameter for later restoration
    opts.change_params({'preprocess': {'p': 0}, 'temporal': {'p': 0}})
    cnm = cnmf.CNMF(n_processes, params=opts, dview=dview)
    cnm = cnm.fit(images)

    # ALTERNATE WAY TO RUN THE PIPELINE AT ONCE (optional)
    
    #   you can also perform the motion correction plus cnmf fitting steps
    #   simultaneously after defining your parameters object using
    #  cnm1 = cnmf.CNMF(n_processes, params=opts, dview=dview)
    #  cnm1.fit_file(motion_correct=True)

    # plot contours of found components
    Cn = caiman.summary_images.local_correlations(images, swap_dim=False)
    Cn[np.isnan(Cn)] = 0
    if not cfg.no_play:
        cnm.estimates.plot_contours(img=Cn)
    plt.title('Contour plots of found components')

    # save results in a separate file (just for demonstration purposes)
    cnm.estimates.Cn = Cn
    cnm.save(fname_new[:-4] + 'hdf5') # FIXME

    # RE-RUN seeded CNMF on accepted patches to refine and perform deconvolution
    cnm.params.change_params({'preprocess': {'p': saved_p}, 'temporal': {'p': saved_p}}) # Restore deconvolution
    cnm2 = cnm.refit(images, dview=dview)

    cnm2.estimates.evaluate_components(images, cnm2.params, dview=dview)

    #  Save new estimates object (after adding correlation image)
    cnm2.estimates.Cn = Cn
    cnm2.save(fname_new[:-4] + 'hdf5')

    if not cfg.no_play:
        # Plot Components
        cnm2.estimates.plot_contours(img=Cn, idx=cnm2.estimates.idx_components)

        # View Traces (accepted and rejected)
        cnm2.estimates.view_components(images, img=Cn,
                                      idx=cnm2.estimates.idx_components)
        cnm2.estimates.view_components(images, img=Cn,
                                      idx=cnm2.estimates.idx_components_bad)

    # update object with selected components (optional)
    # cnm2.estimates.select_components(use_object=True)

    # Extract DF/F values
    cnm2.estimates.detrend_df_f(quantileMin=8, frames_window=250)

    # Show final traces
    if not cfg.no_play:
        cnm2.estimates.view_components(img=Cn)

    # reconstruct denoised movie (press q to exit)
    if not cfg.no_play:
        cnm2.estimates.play_movie(images, q_max=99.9, gain_res=2,
                                  magnification=2,
                                  bpx=border_to_0,
                                  include_bck=False)  # background not shown

    # Stop the cluster and clean up log files
    caiman.stop_server(dview=dview)

    if not cfg.keep_logs:
        log_files = glob.glob('*_LOG_*')
        for log_file in log_files:
            os.remove(log_file)

    # save the results in the original NWB file
    cnm2.estimates.save_NWB(save_path, imaging_rate=fr, session_start_time=datetime.now(tzlocal()),
                            raw_data_file=fnames[0])

def handle_args():
    parser = argparse.ArgumentParser(description="Demonstrate 2P Pipeline using batch algorithm, with nwb files")
    parser.add_argument("--configfile", default=os.path.join(caiman.paths.caiman_datadir(), 'demos', 'general', 'params_demo_pipeline_NWB.json'), help="JSON Configfile for Caiman parameters")
    parser.add_argument("--keep_logs",  action="store_true", help="Keep temporary logfiles")
    parser.add_argument("--no_play",    action="store_true", help="Do not display results")
    parser.add_argument("--cluster_backend", default="multiprocessing", help="Specify multiprocessing, ipyparallel, or single to pick an engine")
    parser.add_argument("--cluster_nproc", type=int, default=None, help="Override automatic selection of number of workers to use")
    parser.add_argument("--input", action="append", help="File(s) to work on, provide multiple times for more files")
    parser.add_argument("--logfile",    help="If specified, log to the named file")
    return parser.parse_args()

########
main()

