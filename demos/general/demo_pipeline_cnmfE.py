#!/usr/bin/env python

"""
Complete pipeline for motion correction, source extraction, and deconvolution
of one photon microendoscopic calcium imaging data using the CaImAn package.
The demo demonstrates how to use the params, MotionCorrect and cnmf objects
for processing 1p microendoscopic data. The analysis pipeline is similar as in
the case of 2p data processing with core difference being the usage of the
CNMF-E algorithm for source extraction (as opposed to plain CNMF). Check
the companion paper for more details.

See also the jupyter notebook demo_pipeline_cnmfE.ipynb
"""

import argparse
import glob
import logging
import matplotlib.pyplot as plt
import numpy as np
import os

import caiman
from caiman.motion_correction import MotionCorrect
from caiman.source_extraction import cnmf
from caiman.source_extraction.cnmf import params as params
from caiman.utils.utils import download_demo
from caiman.utils.visualization import inspect_correlation_pnr



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
        # If no input is specified, use sample data, downloading if necessary
        opts.change_params({"data": {"fnames": cfg.input}})
    if not opts.data['fnames']: # Set neither by CLI arg nor through JSON, so use default data
        fnames = [download_demo('data_endoscope.tif')]
        opts.change_params({"data": {"fnames": fnames}})

    # First set up some parameters for data and motion correction
    # dataset dependent parameters

    c, dview, n_processes = caiman.cluster.setup_cluster(backend=cfg.cluster_backend, n_processes=cfg.cluster_nproc)
    # Motion Correction
    #  The motion:pw_rigid parameter determines where to use rigid or pw-rigid
    #  motion correction
    if not cfg.no_motion_correct:
        # do motion correction rigid
        mc = MotionCorrect(opts.data['fnames'], dview=dview, **opts.get_group('motion'))
        mc.motion_correct(save_movie=True)
        fname_mc = mc.fname_tot_els if opts.motion['pw_rigid'] else mc.fname_tot_rig
        if opts.motion['pw_rigid']:
            bord_px = np.ceil(np.maximum(np.max(np.abs(mc.x_shifts_els)),
                                         np.max(np.abs(mc.y_shifts_els)))).astype(int)
        else:
            bord_px = np.ceil(np.max(np.abs(mc.shifts_rig))).astype(int)
            plt.subplot(1, 2, 1); plt.imshow(mc.total_template_rig)  # plot template
            plt.subplot(1, 2, 2); plt.plot(mc.shifts_rig)  # plot rigid shifts
            plt.legend(['x shifts', 'y shifts'])
            plt.xlabel('frames')
            plt.ylabel('pixels')

        bord_px = 0 if opts.motion['border_nan'] == 'copy' else bord_px
        fname_new = caiman.save_memmap(fname_mc, base_name='memmap_', order='C',
                                   border_to_0=bord_px)
    else:  # if no motion correction just memory map the file
        fname_new = caiman.save_memmap(opts.data['fnames'], base_name='memmap_',
                                   order='C', border_to_0=0, dview=dview)

    # load memory mappable file
    Yr, dims, T = caiman.load_memmap(fname_new)
    images = Yr.T.reshape((T,) + dims, order='F')

    # Parameters for source extraction and deconvolution (CNMF-E algorithm)
    Ain = None          # possibility to seed with predetermined binary masks

    opts.change_params(params_dict={'dims': dims,                          # we rework the source files
                                    'border_pix': bord_px})                # number of pixels to not consider in the borders)

    # compute some summary images (correlation and peak to noise)
    # change swap dim if output looks weird, it is a problem with tiffile
    cn_filter, pnr = caiman.summary_images.correlation_pnr(images[::1], gSig=opts.init['gSig'][0], swap_dim=False)
    # if your images file is too long this computation will take unnecessarily
    # long time and consume a lot of memory. Consider changing images[::1] to
    # images[::5] or something similar to compute on a subset of the data

    # inspect the summary images and set the parameters
    inspect_correlation_pnr(cn_filter, pnr)
    print(f"Minimum correlation: {opts.init['min_corr']}") # min correlation of peak (from correlation image)
    print(f"Minimum peak to noise ratio: {opts.init['min_pnr']}")  # min peak to noise ratio

    # Run CMNF in patches
    cnm = cnmf.CNMF(n_processes=n_processes, dview=dview, Ain=Ain, params=opts)
    cnm.fit(images)

    # Quality Control: DISCARD LOW QUALITY COMPONENTS
    cnm.estimates.evaluate_components(images, cnm.params, dview=dview)

    print(' ***** ')
    print(f"Number of total components: {len(cnm.estimates.C)}")
    print(f"Number of accepted components: {len(cnm.estimates.idx_components)}")

    # Play result movies
    if not cfg.no_play:
        cnm.dims = dims
        cnm.estimates.plot_contours(img=cn_filter, idx=cnm.estimates.idx_components)
        cnm.estimates.view_components(images, idx=cnm.estimates.idx_components)
        # fully reconstructed movie
        cnm.estimates.play_movie(images, q_max=99.5, magnification=2,
                                 include_bck=True, gain_res=10, bpx=bord_px)
        # movie without background
        cnm.estimates.play_movie(images, q_max=99.9, magnification=2,
                                 include_bck=False, gain_res=4, bpx=bord_px)

    # Stop the cluster and clean up log files
    caiman.stop_server(dview=dview)

    if not cfg.keep_logs:
        log_files = glob.glob('*_LOG_*')
        for log_file in log_files:
            os.remove(log_file)

def handle_args():
    parser = argparse.ArgumentParser(description="Demonstrate CNMFE Pipeline")
    parser.add_argument("--configfile", default=os.path.join(caiman.paths.caiman_datadir(), 'demos', 'general', 'params_demo_pipeline_cnmfE.json'), help="JSON Configfile for Caiman parameters")
    parser.add_argument("--no_motion_correct", action='store_true', help="Set to disable motion correction")
    parser.add_argument("--keep_logs",  action="store_true", help="Keep temporary logfiles")
    parser.add_argument("--no_play",    action="store_true", help="Do not display results")
    parser.add_argument("--cluster_backend", default="multiprocessing", help="Specify multiprocessing, ipyparallel, or single to pick an engine")
    parser.add_argument("--cluster_nproc", type=int, default=None, help="Override automatic selection of number of workers to use")
    parser.add_argument("--input", action="append", help="File(s) to work on, provide multiple times for more files")
    parser.add_argument("--logfile",    help="If specified, log to the named file")
    return parser.parse_args()

########
main()

