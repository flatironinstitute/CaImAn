#!/usr/bin/env python

""" Utilities for retrieving paths around the caiman package and its datadirs

"""

import os
from typing import Tuple

#######
# datadir
def caiman_datadir() -> str:
	"""
	The datadir is a user-configurable place which holds a user-modifiable copy of
	data the Caiman libraries need to function, alongside code demos and other things.
	This is meant to be separate from the library install of Caiman, which may be installed
	into the global python library path (or into a conda path or somewhere else messy).
	"""
	if "CAIMAN_DATA" in os.environ:
		return os.environ["CAIMAN_DATA"]
	else:
		return os.path.join(os.path.expanduser("~"), "caiman_data")

def caiman_datadir_exists() -> bool:
	return os.path.isdir(caiman_datadir())

######
# memmap files
#
# Right now these are usually stored in the cwd of the script, although the basename could change that
# In the future we may consistently store these somewhere under the caiman_datadir

def memmap_frames_filename(basename:str, dims:Tuple, frames:int, order:str='F') -> str:
	# Some functions calling this have the first part of *their* dims Tuple be the number of frames.
	# They *must* pass a slice to this so dims is only X, Y, and optionally Z. Frames is passed separately.
	dimfield_0 = dims[0]
	dimfield_1 = dims[1]
	if len(dims) == 3:
		dimfield_2 = dims[2]
	else:
		dimfield_2 = 1
	return f"{basename}_d1_{dimfield_0}_d2_{dimfield_1}_d3_{dimfield_2}_order_{order}_frames_{frames}_.mmap"

