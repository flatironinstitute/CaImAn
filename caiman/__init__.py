#!/usr/bin/env python

import pkg_resources
from caiman.base.movies import movie, load, load_movie_chain, _load_behavior, play_movie
from caiman.base.timeseries import concatenate
from caiman.cluster import start_server, stop_server, setup_cluster 
from caiman.mmapping import load_memmap, save_memmap, save_memmap_each, save_memmap_join
from caiman.summary_images import local_correlations

__version__ = pkg_resources.get_distribution('caiman').version
