#!/usr/bin/env python
""" Utilities for retrieving paths around the caiman package and its datadirs

"""

import os
import logging
from typing import Dict, Tuple


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
# tempdir

def get_tempdir() -> str:
    """ Returns where CaImAn can store temporary files, such as memmap files. Controlled mainly by environment variables """
    # CAIMAN_TEMP is used to control where temporary files live. 
    # If unset, uses default of a temp folder under caiman_datadir()
    # To get the old "store it where I am" behaviour, set CAIMAN_TEMP to a single dot.
    # If you prefer to store it somewhere different, provide a full path to that location.
    if 'CAIMAN_TEMP' in os.environ:
        if os.path.isdir(os.environ['CAIMAN_TEMP']):
            return os.environ['CAIMAN_TEMP']
        else:
            logging.warning(f"CAIMAN_TEMP is set to nonexistent directory {os.environment['CAIMAN_TEMP']}. Ignoring")
    temp_under_data = os.path.join(caiman_datadir(), "temp")
    if not os.path.isdir(temp_under_data):
        logging.warning(f"Default temporary dir {temp_under_data} does not exist, creating")
        os.makedirs(temp_under_data)
    return temp_under_data

def fn_relocated(fn:str) -> str:
    """ If the provided filename does not contain any path elements, this returns what would be its absolute pathname
        as located in get_tempdir(). Otherwise it just returns what it is passed.

        The intent behind this is to ease having functions that explicitly mention pathnames have them go where they want,
        but if all they think about is filenames, they go under CaImAn's notion of its temporary dir. This is under the
        principle of "sensible defaults, but users can override them".
    """
    if not 'CAIMAN_NEW_TEMPFILE' in os.environ: # XXX We will ungate this in a future version of caiman
        return fn
    if str(os.path.basename(fn)) == str(fn): # No path stuff
        return os.path.join(get_tempdir(), fn)
    else:
        return fn

######
# memmap files
#
# Right now these are usually stored in the cwd of the script, although the basename could change that
# In the future we may consistently store these somewhere under the caiman_datadir


def memmap_frames_filename(basename: str, dims: Tuple, frames: int, order: str = 'F') -> str:
    # Some functions calling this have the first part of *their* dims Tuple be the number of frames.
    # They *must* pass a slice to this so dims is only X, Y, and optionally Z. Frames is passed separately.
    dimfield_0 = dims[0]
    dimfield_1 = dims[1]
    if len(dims) == 3:
        dimfield_2 = dims[2]
    else:
        dimfield_2 = 1
    return f"{basename}_d1_{dimfield_0}_d2_{dimfield_1}_d3_{dimfield_2}_order_{order}_frames_{frames}.mmap"

def fname_derived_presuffix(basename:str, addition:str, swapsuffix:str = None) -> str:
    # Given a filename with an extension, extend the pre-extension part of the filename with the
    # desired addition, adding an underscore first if needed
    fn_base, fn_ext = os.path.splitext(basename)
    if not addition.startswith('_') and not basename.endswith('_'):
        addition = '_' + addition
    if swapsuffix is not None:
        if not swapsuffix.startswith('.'):
            swapsuffix = '.' + swapsuffix
        return fn_base + addition + swapsuffix
    else:
        return fn_base + addition + fn_ext

def decode_mmap_filename_dict(basename:str) -> Dict:
    # For a mmap file we (presumably) made, return a dict with the information encoded in its
    # filename. This will usually be params like d1, d2, T, and order.
    # This function is not general; it knows the fields it wants to extract.
    print(f"Decode mmap filename {basename}")
    ret = {}
    _, fn = os.path.split(basename)
    fn_base, _ = os.path.splitext(fn)
    fpart = fn_base.split('_')[1:] # First part will (probably) reference the datasets
    for field in ['d1', 'd2', 'd3', 'order', 'frames']:
        # look for the last index of fpart and look at the next index for the value, saving into ret
        for i in range(len(fpart)-1, -1, -1): # Step backwards through the list; defensive programming
            if field == fpart[i]:
                if field == 'order': # a string
                    ret[field] = fpart[i + 1] # Assume no filenames will be constructed to end with a key and not a value
                else: # numeric
                    ret[field] = int(fpart[i + 1]) # Assume no filenames will be constructed to end with a key and not a value
    if fpart[-1] != '':
        ret['T'] = int(fpart[-1]) # XXX Unclear how 'T' differs from frames; my sample datasets have T and frames have the same value
    if 'T' in ret and 'frames' in ret and ret['T'] != ret['frames']:
        print(f"D: The value of 'T' {ret['T']} differs from 'frames' {ret['frames']}")
    if 'T' not in ret and 'frames' in ret:
        ret['T'] = ret['frames']
    return ret
