#!/usr/bin/env python

"""
Utility functions for Neurolabware Scanbox files (.sbx)
"""

import os
import logging
from typing import Iterable

import numpy as np
import scipy
import tifffile

Subindices = Iterable[int] | slice

def loadmat_sbx(filename: str) -> dict:
    """
    this wrapper should be called instead of directly calling spio.loadmat

    It solves the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to fix all entries
    which are still mat-objects
    """
    data_ = scipy.io.loadmat(filename, struct_as_record=False, squeeze_me=True)
    _check_keys(data_)
    return data_


def _check_keys(checkdict: dict) -> None:
    """
    checks if entries in dictionary are mat-objects. If yes todict is called to change them to nested dictionaries.
    Modifies its parameter in-place.
    """

    for key in checkdict:
        if isinstance(checkdict[key], scipy.io.matlab.mio5_params.mat_struct):
            checkdict[key] = _todict(checkdict[key])


def _todict(matobj) -> dict:
    """
    A recursive function which constructs from matobjects nested dictionaries
    """

    ret = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, scipy.io.matlab.mio5_params.mat_struct):
            ret[strg] = _todict(elem)
        else:
            ret[strg] = elem
    return ret


def sbxread(filename: str, subindices: Subindices = slice(None), channel: int | None = None) -> np.ndarray:
    """
    Load frames of an .sbx file into a new NumPy array

    Args:
        filename: str
            filename should be full path excluding .sbx

        subindices: slice | array-like
            which frames to read (defaults to all)

        channel: int | None
            which channel to save (required if data has >1 channel)
    """
    return _sbxread_helper(filename, subindices=subindices, channel=channel, chunk_size=None)


def sbx_to_tif(filename: str, fileout: str | None = None, subindices: Subindices = slice(None),
               channel: int | None = None, chunk_size: int = 1000):
    """
    Convert a single .sbx file to .tif format

    Args:
        filename: str
            filename should be full path excluding .sbx

        fileout: str | None
            filename to save (defaults to `filename` with .sbx replaced with .tif)

        subindices: slice | array-like
            which frames to read (defaults to all)

        channel: int | None
            which channel to save (required if data has >1 channel)

        chunk_size: int | None
            how many frames to load into memory at once (None = load the whole thing)
    """
    # Check filenames
    if '.sbx' in filename:
        filename = filename[:-4]

    if fileout is None:
        fileout = filename + '.tif'
    else:
        # Add a '.tif' extension if not already present
        extension = os.path.splitext(fileout)[1].lower()
        if extension not in ['.tif', '.tiff', '.btf']:
            fileout = fileout + '.tif'

    # Find the shape of the file we need
    data_shape = sbx_shape(filename)
    n_chans, n_x, n_y, n_planes, n_frames = data_shape
    is3D = n_planes > 1
    iterable_elements = _interpret_subindices(subindices, n_frames)[0]
    N = len(iterable_elements)
    save_shape = (N, n_y, n_x, n_planes) if is3D else (N, n_y, n_x)

    # Open tif as a memmap and copy to it
    memmap_tif = tifffile.memmap(fileout, shape=save_shape, dtype='uint16')
    _sbxread_helper(filename, subindices=subindices, channel=channel, out=memmap_tif, chunk_size=chunk_size)


def sbx_chain_to_tif(filenames: list[str], fileout: str, subindices: list[Subindices] | Subindices = slice(None),
                     bigtiff: bool | None = True, imagej: bool = False, to32: bool = False,
                     channel: int | None = None, chunk_size: int = 1000) -> None:
    """
    Concatenate a list of sbx files into one tif file.
    Args:
        filenames: list[str]
            each filename should be full path excluding .sbx

        fileout: str
            filename to save, including the .npy suffix
        
        subindices: list[Iterable[int] | slice] | Iterable[int] | slice
            if a list, each entry is interpreted as in sbx_to_npy for the corresponding file.
            otherwise, the single argument is broadcast across all files.
        
        to32: bool
            whether to save in float32 format (default is to keep as uint16)

        channel, chunk_size: see sbx_to_tif
    """
    # Validate aggressively to avoid failing after waiting to copy a lot of data
    if isinstance(subindices, slice) or np.isscalar(subindices[0]):
        subindices = [subindices for _ in filenames]
    elif len(subindices) != len(filenames):
        raise Exception('Length of subindices does not match length of file list')        

    # Get the total size of the file
    all_shapes = np.stack([sbx_shape(file) for file in filenames])

    # Check that X, Y, and Z are consistent
    for kdim, dim in enumerate(['X', 'Y', 'Z']):
        if np.any(np.diff(all_shapes[:, kdim + 1]) != 0):
            raise Exception(f'Given files have inconsistent shapes in the {dim} dimension')
    
    # Check that all files have the requested channel
    if channel is None:
        if np.any(all_shapes[:, 0] > 1):
            raise Exception('At least one file has multiple channels; must specify channel')
        channel = 0
    elif np.any(all_shapes[:, 0] <= channel):
        raise Exception('Not all files have the requested channel')
    
    # Allocate empty tif file with the final shape (do this first to ensure any existing file is overwritten)
    n_x, n_y, n_planes = map(int, all_shapes[0, 1:4])
    is3D = n_planes > 1
    Ns = [len(_interpret_subindices(subind, file_N)[0]) for (subind, file_N) in zip(subindices, all_shapes[:, 4])]
    N = sum(Ns)
    save_shape = (N, n_y, n_x, n_planes) if is3D else (N, n_y, n_x)

    extension = os.path.splitext(fileout)[1].lower()
    if extension not in ['.tif', '.tiff', '.btf']:
        fileout = fileout + '.tif'

    dtype = np.float32 if to32 else np.uint16
    tifffile.imwrite(fileout, data=None, mode='w', shape=save_shape, bigtiff=bigtiff, imagej=imagej, dtype=dtype)

    # Now convert each file
    tif_memmap = tifffile.memmap(fileout, series=0)
    offset = 0
    for filename, subind, file_N in zip(filenames, subindices, Ns):
        _sbxread_helper(filename, subindices=subind, channel=channel, out=tif_memmap[offset:offset+file_N], chunk_size=chunk_size)
        offset += file_N


def sbx_shape(filename: str, info: dict | None = None) -> tuple[int, int, int, int, int]:
    """
    Args:
        filename: str
            filename should be full path excluding .sbx

        info: dict | None
            info struct for sbx file (to avoid re-loading)

    Output: tuple (chans, X, Y, Z, frames) representing shape of scanbox data
    """
    # Check if contains .sbx and if so just truncate
    if '.sbx' in filename:
        filename = filename[:-4]

    # Load info
    if info is None:
        info = loadmat_sbx(filename + '.mat')['info']

    # Image size
    if 'sz' not in info:
        info['sz'] = np.array([512, 796])
    
    # Scan mode (0 indicates bidirectional)
    if 'scanmode' in info and info['scanmode'] == 0:
        info['recordsPerBuffer'] *= 2

    # Fold lines (multiple subframes per scan) - basically means the frames are smaller and
    # there are more of them than is reflected in the info file
    if 'fold_lines' in info and info['fold_lines'] > 0:
        if info['recordsPerBuffer'] % info['fold_lines'] != 0:
            raise Exception('Non-integer folds per frame not supported')
        n_folds = round(info['recordsPerBuffer'] / info['fold_lines'])
        info['recordsPerBuffer'] = info['fold_lines']
        info['sz'][0] = info['fold_lines']
        if 'bytesPerBuffer' in info:
            info['bytesPerBuffer'] /= n_folds
    else:
        n_folds = 1   

    # Defining number of channels/size factor
    if 'chan' in info:
        info['nChan'] = info['chan']['nchan']
        factor = 1  # should not be used
    else:
        if info['channels'] == 1:
            info['nChan'] = 2
            factor = 1
        elif info['channels'] == 2:
            info['nChan'] = 1
            factor = 2
        elif info['channels'] == 3:
            info['nChan'] = 1
            factor = 2

    # Determine number of frames in whole file
    filesize = os.path.getsize(filename + '.sbx')
    if 'scanbox_version' in info:
        if info['scanbox_version'] == 2:
            info['max_idx'] = filesize / info['recordsPerBuffer'] / info['sz'][1] * factor / 4 - 1
        elif info['scanbox_version'] == 3:
            info['max_idx'] = filesize / np.prod(info['sz']) / info['nChan'] / 2 - 1
        else:
            raise Exception('Invalid Scanbox version')
    else:
        info['max_idx'] = filesize / info['bytesPerBuffer'] * factor - 1

    N = info['max_idx'] + 1    # Last frame

    # Determine whether we are looking at a z-stack
    # Only consider optotune z-stacks - knobby schedules have too many possibilities and
    # can't determine whether it was actually armed from the saved info.
    if info['volscan']:
        n_planes = info['otparam'][2]
    else:
        n_planes = 1
    N //= n_planes

    x = (int(info['nChan']), int(info['sz'][1]), int(info['recordsPerBuffer']), int(n_planes), int(N))
    return x


def sbx_meta_data(filename: str):
    """
    Get metadata for an .sbx file
    Thanks to sbxreader for much of this: https://github.com/jcouto/sbxreader
    Field names and values are equivalent to sbxreader as much as possible

    Args:
        filename: str
            filename should be full path excluding .sbx
    """
    if '.sbx' in filename:
        filename = filename[:-4]
    
    info = loadmat_sbx(filename + '.mat')['info']

    meta_data = dict()
    n_chan, n_x, n_y, n_planes, n_frames = sbx_shape(filename, info)
    
    # Frame rate
    # use uncorrected recordsPerBuffer here b/c we want the actual # of resonant scans per frame
    meta_data['frame_rate'] = info['resfreq'] / info['recordsPerBuffer'] / n_planes

    # Spatial resolution
    magidx = info['config']['magnification'] - 1
    if 'dycal' in info and 'dxcal' in info:
        meta_data['um_per_pixel_x'] = info['dxcal']
        meta_data['um_per_pixel_y'] = info['dycal']
    else:
        try:
            meta_data['um_per_pixel_x'] = info['calibration'][magidx]['x']
            meta_data['um_per_pixel_y'] = info['calibration'][magidx]['y']
        except (KeyError, TypeError):
            pass
    
    # Optotune depths
    if n_planes == 1:
        meta_data['etl_pos'] = []
    else:
        if 'otwave' in info and not isinstance(info['otwave'], int) and len(info['otwave']):
            meta_data['etl_pos'] = [a for a in info['otwave']]
        
        if 'etl_table' in info:
            meta_data['etl_pos'] = [a[0] for a in info['etl_table']]

    meta_data['scanning_mode'] = 'bidirectional' if info['scanmode'] == 0 else 'unidirectional'
    meta_data['num_frames'] = n_frames
    meta_data['num_channels'] = n_chan
    meta_data['num_planes'] = n_planes
    meta_data['frame_size'] = info['sz']
    meta_data['num_target_frames'] = info['config']['frames']
    meta_data['num_stored_frames'] = info['max_idx'] + 1
    meta_data['stage_pos'] = [info['config']['knobby']['pos']['x'],
                              info['config']['knobby']['pos']['y'],
                              info['config']['knobby']['pos']['z']]
    meta_data['stage_angle'] = info['config']['knobby']['pos']['a']
    meta_data['filename'] = os.path.basename(filename + '.sbx')
    meta_data['resonant_freq'] = info['resfreq']
    meta_data['scanbox_version'] = info['scanbox_version']
    meta_data['records_per_buffer'] = info['recordsPerBuffer']
    meta_data['magnification'] = float(info['config']['magnification_list'][magidx])
    meta_data['objective'] = info['objective']

    for i in range(4):
        if f'pmt{i}_gain' in info['config']:
            meta_data[f'pmt{i}_gain'] = info['config'][f'pmt{i}_gain']

    possible_fields = ['messages', 'event_id', 'usernotes', 'ballmotion',
                       ('frame', 'event_frame'), ('line', 'event_line')]
    
    for field in possible_fields:
        if isinstance(field, tuple):
            field, fieldout = field
        else:
            fieldout = field

        if field in info:
            meta_data[fieldout] = info[field]

    return meta_data


def _sbxread_helper(filename: str, subindices: Subindices = slice(None), channel: int | None = None,
                     out: np.memmap | None = None, chunk_size: int | None = 1000) -> np.ndarray:
    """
    Load frames of an .sbx file into a new NumPy array, or into the given memory-mapped file.

    Args:
        filename: str
            filename should be full path excluding .sbx

        subindices: slice | array-like
            which frames to read (defaults to all)

        channel: int | None
            which channel to save (required if data has >1 channel)

        out: np.memmap | None
            existing memory-mapped file to write into (in which case fileout is ignored)

        chunk_size: int | None
            how many frames to load into memory at once (None = load the whole thing)
    """

    # Check if contains .sbx and if so just truncate
    if '.sbx' in filename:
        filename = filename[:-4]

    # Load info
    info = loadmat_sbx(filename + '.mat')['info']

    # Get shape (and update info)
    data_shape = sbx_shape(filename, info)  # (chans, X, Y, Z, frames)
    n_chans, n_x, n_y, n_planes, n_frames = data_shape
    is3D = n_planes > 1

    if channel is None:
        if n_chans > 1:
            raise Exception('Channel input required for multi-chanel data')
        channel = 0
    elif channel >= n_chans:
        raise Exception(f'Channel input out of range (data has {n_chans} channels)')

    if 'scanbox_version' in info and info['scanbox_version'] == 3:
        frame_size = round(np.prod(info['sz']) * info['nChan'] * 2 * n_planes)
    else:
        frame_size = round(info['sz'][1] * info['recordsPerBuffer'] * 2 * info['nChan'] * n_planes)
    if frame_size <= 0:
        raise Exception('Invalid scanbox metadata')

    iterable_elements, skip = _interpret_subindices(subindices, n_frames)

    if any([ind < 0 or ind >= n_frames for ind in iterable_elements]):
        raise Exception(f'Loading {filename}: requested frames out of range')
    N = len(list(iterable_elements))

    save_shape = (N, n_y, n_x, n_planes) if is3D else (N, n_y, n_x)
    if out is not None and out.shape != save_shape:
        raise Exception('Existing memmap is the wrong shape to hold loaded data')

    # Open File
    with open(filename + '.sbx') as sbx_file:
        def get_chunk(n: int):
            # Note: SBX files store the values strangely, it's necessary to invert each uint16 value to get the correct ones
            chunk = np.invert(np.fromfile(sbx_file, dtype='uint16', count=frame_size//2 * n))
            chunk = np.reshape(chunk, data_shape[:4] + (N,), order='F')  # (chans, X, Y, Z, frames)
            chunk = np.transpose(chunk, (0, 4, 2, 1, 3))[channel]  # to (frames, Y, X, Z)
            if not is3D:
                chunk = np.squeeze(chunk, axis=3)
            return chunk

        if skip == 1:
            sbx_file.seek(iterable_elements[0] * frame_size, 0)

            if chunk_size is None:
                # load a contiguous block all at once
                if out is None:
                    # avoid allocating an extra array just to copy into it
                    out = get_chunk(N)
                else:
                    out[:] = get_chunk(N)
            else:
                # load a contiguous block of data, but chunk it to not use too much memory at once
                if out is None:
                    out = np.empty(save_shape, dtype=np.uint16)

                n_remaining = N
                offset = 0
                while n_remaining > 0:
                    this_chunk_size = min(n_remaining, chunk_size)
                    chunk = get_chunk(this_chunk_size)
                    out[offset:offset+this_chunk_size] = chunk
                    n_remaining -= this_chunk_size
                    offset += this_chunk_size

        else:
            # load one frame at a time, sorting indices for fastest access
            if out is None:
                out = np.empty(save_shape, dtype=np.uint16)

            for counter, (k, ind) in enumerate(sorted(zip(iterable_elements, range(N)))):
                if counter % 100 == 0:
                    logging.debug(f'Reading iteration: {k}')
                sbx_file.seek(k * frame_size, 0)
                out[ind] = get_chunk(1)

        if isinstance(out, np.memmap):
            out.flush()
        
    return out


def _interpret_subindices(subindices: Subindices, n_frames: int) -> tuple[Iterable[int], int]:
    """
    Given the number of frames in the corresponding recording, obtain an iterable over subindices 
    and the step size (or 0 if the step size is not uniform).
    """
    if isinstance(subindices, slice):
        start = 0 if subindices.start is None else subindices.start
        skip = 1 if subindices.step is None else subindices.step

        if subindices.stop is None or not np.isfinite(subindices.stop):
            stop = n_frames
        elif subindices.stop > n_frames:
            logging.warning(f'Only {n_frames} frames available to load ' +
                            f'(requested up to {subindices.stop})')
            stop = n_frames
        else:
            stop = subindices.stop

        iterable_elements = range(start, stop, skip)
    else:
        iterable_elements = subindices
        skip = 0

    return iterable_elements, skip
