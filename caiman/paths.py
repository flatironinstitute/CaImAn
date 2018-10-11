#!/usr/bin/env python

""" Utilities for retrieving paths around the caiman package and its datadirs

"""

import os

def caiman_datadir():
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

def caiman_datadir_exists():
	return os.path.isdir(caiman_datadir())
