#!/usr/bin/env python

"""
Complete pipeline for online processing using CaImAn Online (OnACID).
The demo demonstrates the analysis of a sequence of files using the CaImAn online
algorithm. The steps include i) motion correction, ii) tracking current 
components, iii) detecting new components, iv) updating of spatial footprints.
The script demonstrates how to construct and use the params and online_cnmf
objects required for the analysis, and presents the various parameters that
can be passed as options. A plot of the processing time for the various steps
of the algorithm is also included.

Thanks to Andreas Tolias and his lab at Baylor College of Medicine
for sharing the data used in this demo.
"""

import argparse
import glob
import logging
import matplotlib
import numpy as np
import os
import matplotlib.pyplot as plt

try:
    cv2.setNumThreads(0)
except:
    pass

import caiman as cm
from caiman.paths import caiman_datadir
from caiman.source_extraction import cnmf as cnmf
from caiman.utils.utils import download_demo


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

    opts = cnmf.params.CNMFParams(params_from_file=cfg.configfile)

    if cfg.input is not None and len(cfg.input) != 0: # CLI arg can override all other settings for fnames, although other data-centric commands still must match source data
        opts.change_params({'data': {'fnames': cfg.input}})

    if not opts.data['fnames']: # Set neither by CLI arg nor through JSON, so use default data
        fld_name = 'Mesoscope'
        print("Downloading demos...")
        fnames = [download_demo('Tolias_mesoscope_1.hdf5', fld_name),
                  download_demo('Tolias_mesoscope_2.hdf5', fld_name),
                  download_demo('Tolias_mesoscope_3.hdf5', fld_name)]
        opts.change_params({'data': {'fnames': fnames}})

    opts.change_params({'online': {'show_movie': cfg.show_movie}}) # Override from CLI

    print(f"Params: {opts.to_json()}")
    # fit online
    cnm = cnmf.online_cnmf.OnACID(params=opts)
    cnm.fit_online()

    # plot contours (this may take time)
    logging.info(f"Number of components: {cnm.estimates.A.shape[-1]}")
    images = cm.load(fnames)
    Cn = images.local_correlations(swap_dim=False, frames_per_chunk=500)
    cnm.estimates.plot_contours(img=Cn, display_numbers=False)

    # view components
    cnm.estimates.view_components(img=Cn)

    # plot timing performance (if a movie is generated during processing, timing
    # will be severely over-estimated)

    T_motion = 1e3*np.array(cnm.t_motion)
    T_detect = 1e3*np.array(cnm.t_detect)
    T_shapes = 1e3*np.array(cnm.t_shapes)
    T_track  = 1e3*np.array(cnm.t_online) - T_motion - T_detect - T_shapes
    plt.figure()
    plt.stackplot(np.arange(len(T_motion)), T_motion, T_track, T_detect, T_shapes)
    plt.legend(labels=['motion', 'tracking', 'detect', 'shapes'], loc=2)
    plt.title('Processing time allocation')
    plt.xlabel('Frame #')
    plt.ylabel('Processing time [ms]')

    # Prepare result visualisations (might take time)
    c, dview, n_processes = cm.cluster.setup_cluster(backend=cfg.cluster_backend, n_processes=cfg.cluster_nproc)
    if opts.online['motion_correct']:
        shifts = cnm.estimates.shifts[-cnm.estimates.C.shape[-1]:]
        if not opts.motion['pw_rigid']:
            memmap_file = cm.motion_correction.apply_shift_online(images, shifts,
                                                        save_base_name='MC')
        else:
            mc = cm.motion_correction.MotionCorrect(fnames, dview=dview,
                                                    **opts.get_group('motion'))

            mc.y_shifts_els = [[sx[0] for sx in sh] for sh in shifts]
            mc.x_shifts_els = [[sx[1] for sx in sh] for sh in shifts]
            memmap_file = mc.apply_shifts_movie(fnames, rigid_shifts=False,
                                                save_memmap=True,
                                                save_base_name='MC')
    else:  # To do: apply non-rigid shifts on the fly
        memmap_file = images.save(fnames[0][:-4] + 'mmap')
    cnm.mmap_file = memmap_file
    Yr, dims, T = cm.load_memmap(memmap_file)

    images = np.reshape(Yr.T, [T] + list(dims), order='F')

    # evaluate_components uses parameters from the 'quality' category
    cnm.estimates.evaluate_components(images, cnm.params, dview=dview)
    cnm.estimates.Cn = Cn
    cnm.save(os.path.splitext(fnames[0])[0] + '_results.hdf5')

    dview.terminate()
    if not cfg.no_play:
        matplotlib.pyplot.show(block=True)

def handle_args():
    parser = argparse.ArgumentParser(description="Full OnACID Caiman demo")
    parser.add_argument("--configfile", default=os.path.join(caiman_datadir(), 'demos', 'general', 'params_demo_OnACID_mesoscope.json'), help="JSON Configfile for Caiman parameters")
    parser.add_argument("--show_movie",    action="store_true", help="Display results as movie")
    parser.add_argument("--cluster_backend", default="multiprocessing", help="Specify multiprocessing, ipyparallel, or single to pick an engine")
    parser.add_argument("--cluster_nproc", type=int, default=None, help="Override automatic selection of number of workers to use")
    parser.add_argument("--input", action="append", help="File(s) to work on, provide multiple times for more files")
    parser.add_argument("--no_play",    action="store_true", help="Do not display results")
    parser.add_argument("--logfile",    help="If specified, log to the named file")
    return parser.parse_args()

########
main()