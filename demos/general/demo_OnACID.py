#!/usr/bin/env python

"""
Basic demo for the CaImAn Online algorithm (OnACID) using CNMF initialization.
It demonstrates the construction of the params and online_cnmf objects and
the fit function that is used to run the algorithm.
For a more complete demo check the script demo_OnACID_mesoscope.py
"""

import argparse
#import code
import logging
import numpy as np
import os

try:
    cv2.setNumThreads(0)
except:
    pass

import caiman as cm
from caiman.source_extraction import cnmf as cnmf
from caiman.paths import caiman_datadir


def main():
    cfg = handle_args()

    if cfg.logfile:
        logging.basicConfig(format=
            "%(relativeCreated)12d [%(filename)s:%(funcName)20s():%(lineno)s][%(process)d] %(message)s",
            level=logging.WARNING,
            filename=cfg.logfile)
        # You can make the output more or less verbose by setting level to logging.DEBUG, logging.INFO, logging.WARNING, or logging.ERROR
    else:
        logging.basicConfig(format=
            "%(relativeCreated)12d [%(filename)s:%(funcName)20s():%(lineno)s][%(process)d] %(message)s",
            level=logging.WARNING)

    opts = cnmf.params.CNMFParams(params_from_file=cfg.configfile)

    if cfg.input is not None:
        opts.change_params({"data": {"fnames": cfg.input}})

    if not opts.data['fnames']: # Set neither by CLI arg nor through JSON, so use default data
        fnames = [os.path.join(caiman_datadir(), 'example_movies', 'demoMovie.tif')]

    # If you want to break into an interactive console session, move and uncomment this wherever you want in the code
    # (and uncomment the code import at the top)
    #code.interact(local=dict(globals(), **locals()) )

    # fit with online object
    cnm = cnmf.online_cnmf.OnACID(params=opts)
    cnm.fit_online()

    # plot contours
    logging.info(f"Number of components: {cnm.estimates.A.shape[-1]}")
    Cn = cm.load(fnames[0], subindices=slice(0,500)).local_correlations(swap_dim=False)
    cnm.estimates.plot_contours(img=Cn)

    # pass through the CNN classifier with a low threshold (keeps clearer neuron shapes and excludes processes)
    use_CNN = True
    if use_CNN:
        # threshold for CNN classifier
        opts.set('quality', {'min_cnn_thr': 0.05})
        cnm.estimates.evaluate_components_CNN(opts)
        cnm.estimates.plot_contours(img=Cn, idx=cnm.estimates.idx_components)
    
    # plot results
    cnm.estimates.view_components(img=Cn, idx=cnm.estimates.idx_components)

def handle_args():
    parser = argparse.ArgumentParser(description="Demonstrate basic Caiman Online functionality with CNMF initialization")
    parser.add_argument("--configfile", default=os.path.join(caiman_datadir(), 'demos', 'general', 'params_demo_OnACID.json'), help="JSON Configfile for Caiman parameters")
    parser.add_argument("--input", action="append", help="File(s) to work on, provide multiple times for more files")
    parser.add_argument("--logfile",    help="If specified, log to the named file")
    return parser.parse_args()

########
main()
