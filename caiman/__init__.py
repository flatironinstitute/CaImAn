#!/usr/bin/env python

import pkg_resources
from .base.movies import movie, load, load_movie_chain, _load_behavior
from .base.timeseries import concatenate
from .cluster import start_server, stop_server
from .mmapping import load_memmap, save_memmap, save_memmap_each, save_memmap_join
from .summary_images import local_correlations
#from .source_extraction import cnmf

__version__ = pkg_resources.get_distribution('caiman').version
