#!/usr/bin/env python

"""
Utility functions for Neurolabware Scanbox files (.sbx)
"""

import os
import logging
from typing import Iterable

import numpy as np
from numpy.lib.format import open_memmap
import scipy

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


def sbx_to_npy(filename: str, fileout: str | None = None, k: int = 0, n_frames=np.inf, channel=None) -> None:
    """
    Convert a contiguous section of an .sbx file to .npy format (defaulting to the whole file)

    Args:
        filename: str
            filename should be full path excluding .sbx

        fileout: str | None
            filename to save (defaults to `filename` with .sbx replaced with .npy)

        k: int
            the frame to start reading
        
        n_frames: int | float
            how many frames to read (inf = read to the end)

        channel: int | None
            which channel to save (required if data has >1 channel)
    """
    return sbx_to_npy_skip(filename, slice(k, k + n_frames), fileout=fileout, channel=channel)


def sbx_to_npy_skip(filename: str, subindices: Subindices = slice(None), fileout: str | None = None,
                     channel=None, chunk_size=1000, out: np.ndarray | None = None) -> None:
    """
    Convert an arbitrary subset of frames of an .sbx file to .npy format

    Args:
        filename: str
            filename should be full path excluding .sbx
        
        subindices: slice | array-like
            which frames to read (defaults to all)
        
        fileout: str | None
            filename to save (defaults to `filename` with .sbx replaced with .npy)

        channel: int | None
            which channel to save (required if data has >1 channel)

        chunk_size: int
            how many frames to load into memory at once

        out: np.memmap | None
            existing array or memory-mapped file to write into (in which case fileout is ignored)
    """

    # Check if contains .sbx and if so just truncate
    if '.sbx' in filename:
        filename = filename[:-4]

    if fileout is None:
        fileout = filename + '.npy'

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
        if out is None:
            npy_file = open_memmap(fileout, mode='w+', dtype=np.float32, shape=save_shape)
        else:
            npy_file = out

        def convert_chunk(n: int, offset: int):
            # Note: SBX files store the values strangely, it's necessary to invert each uint16 value to get the correct ones
            chunk = np.invert(np.fromfile(sbx_file, dtype='unit16', count=frame_size//2 * n))
            chunk = np.reshape(chunk, data_shape[:4] + (N,), order='F')  # (chans, X, Y, Z, frames)
            chunk = np.transpose(chunk, (0, 4, 2, 1, 3))[channel]  # to (frames, Y, X, Z)
            if not is3D:
                chunk = np.squeeze(chunk, axis=3)
            npy_file[offset:offset+this_chunk] = chunk

        offset = 0

        if skip == 1:
            # copy a contiguous block of data, but chunk it to not use too much memory at once
            sbx_file.seek(iterable_elements[0] * frame_size, 0)
            n_remaining = N

            while n_remaining > 0:
                this_chunk = min(n_remaining, chunk_size)
                convert_chunk(this_chunk, offset)
                n_remaining -= this_chunk
                offset += this_chunk

        else:
            # sort indices for fastest access
            for counter, (k, ind) in enumerate(sorted(zip(iterable_elements, range(N)))):
                if counter % 100 == 0:
                    logging.debug(f'Reading iteration: {k}')
                sbx_file.seek(k * frame_size, 0)
                convert_chunk(1, ind)

        if isinstance(npy_file, np.memmap):
            npy_file.flush()


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


def sbxread(filename: str, subindices: Subindices | None, channel=None) -> np.ndarray:
    """
    Loads the given sbx file into a numpy array.
    Not recommended. To memory-map a large file that does not fit into memory, use sbx_to_npy.
    Args:
        filename: str
            filename should be full path excluding .sbx
        
        subindices: slice | array-like | None
            which frames to read (defaults to all)
    """
    if subindices is None:
        subindices = slice(None)

    shape_in = sbx_shape(filename)
    n_chans, n_x, n_y, n_planes, n_frames = shape_in
    is3D = n_planes > 1
    N = len(_interpret_subindices(subindices, n_frames)[0])
    
    # memory warning
    n_bytes = N * n_x * n_y * n_planes * 4
    if n_bytes > 16_000_000_000:
        logging.warning(f'Loading {n_bytes/1_000_000_000:.2f} GB of data into memory; you may want to use sbx_to_npy to convert and then memory-map the npy file.')  
    
    # write the data and return
    data_shape = (N, n_y, n_x, n_planes) if is3D else (N, n_y, n_x)
    data_out = np.empty(data_shape, dtype=np.float32)
    sbx_to_npy_skip(filename, subindices, channel=channel, out=data_out)
    return data_out


def sbx_chain_to_npy(filenames: list[str], fileout: str, subindices: list[Subindices] | Subindices = slice(None),
                     channel=None, chunk_size=1000) -> None:
    """
    Concatenate a list of sbx files into one npy file.
    Args:
        filenames: list[str]
            each filename should be full path excluding .sbx

        fileout: str
            filename to save, including the .npy suffix
        
        subindices: list[Iterable[int] | slice] | Iterable[int] | slice
            if a list, each entry is interpreted as in sbx_to_npy for the corresponding file.
            otherwise, the single argument is broadcast across all files.

        channel, chunk_size: see sbx_to_npy
    """
    # Validate
    if subindices is not None and len(subindices) != len(filenames):
        raise Exception('Length of subindices does not match length of file list')
    else:
        subindices = [subindices for _ in len(filenames)]

    # Get the total size of the file
    all_shapes = np.stack([sbx_shape(file) for file in filenames])

    # Check that X, Y, and Z are consistent
    for kdim, dim in enumerate(['X', 'Y', 'Z']):
        if np.any(np.diff(all_shapes[:, kdim + 1]) != 0):
            raise Exception(f'Given files have inconsistent shapes in the {dim} dimension')
    
    n_x, n_y, n_planes = all_shapes[0, 1:4]
    is3D = n_planes > 1
    
    # Check that all files have the requested channel
    if channel is None:
        if np.any(all_shapes[:, 0] > 1):
            raise Exception('At least one file has multiple channels; must specify channel')
        channel = 0
    elif np.any(all_shapes[:, 0] <= channel):
        raise Exception('Not all files have the requested channel')

    # Get total N
    Ns = [len(_interpret_subindices(subind, file_N)[0]) for (subind, file_N) in zip(subindices, all_shapes[:, 4])]
    N = sum(Ns)

    # Make memmap to npy file for concatenated data
    save_shape = (N, n_y, n_x, n_planes) if is3D else (N, n_y, n_x)
    npy_file = open_memmap(fileout, mode='w+', dtype=np.float32, shape=save_shape)

    # Now for each file, pass a view of a subset of the whole file to sbx_to_npy_skip
    frame_size = n_x * n_y * n_planes * 4
    offset_frames = 0

    for sbx_filename, subind, file_N in zip(filenames, subindices, Ns):
        file_shape = (file_N, n_y, n_x, n_planes) if is3D else (file_N, n_y, n_x)
        file_memmap = np.memmap(fileout, mode='r+', dtype=np.float32, offset=npy_file.offset + frame_size * offset_frames,  shape=file_shape)
        sbx_to_npy_skip(sbx_filename, subind, channel=channel, chunk_size=chunk_size, out=file_memmap)
        offset_frames += file_N


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