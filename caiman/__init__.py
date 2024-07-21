#!/usr/bin/env python

# Find keras, depending on tensorflow version
import os
try:
    import tensorflow.keras as keras
except ModuleNotFoundError:
    try:
        # workaround to continue using Keras 2 with tensorflow >= 2.16
        os.environ["TF_USE_LEGACY_KERAS"] = "1"
        import tf_keras as keras
    except ModuleNotFoundError:
        keras = None

import pkg_resources
from caiman.base.movies import movie, load, load_movie_chain, _load_behavior, play_movie
from caiman.base.timeseries import concatenate
from caiman.cluster import start_server, stop_server
from caiman.mmapping import load_memmap, save_memmap, save_memmap_each, save_memmap_join
from caiman.summary_images import local_correlations

__version__ = pkg_resources.get_distribution('caiman').version
