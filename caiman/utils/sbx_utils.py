#!/usr/bin/env python

"""
Utility functions for Neurolabware Scanbox files (.sbx)
"""

import logging
import numpy as np
import os
import scipy
import tifffile
from typing import Any, Sequence, Union, Optional, cast

DimSubindices = Union[Sequence[int], slice]
FileSubindices = Union[DimSubindices, Sequence[DimSubindices]]  # can have inds for just frames or also for y, x, z
ChainSubindices = Union[FileSubindices, Sequence[FileSubindices]]  # one to apply to each file, or separate for each file

def loadmat_sbx(filename: str) -> dict[str, Any]:
    """
    this wrapper should be called instead of directly calling spio.loadmat

    It solves the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to fix all entries
    which are still mat-objects
    """
    data_ = scipy.io.loadmat(filename, struct_as_record=False, squeeze_me=True)
    _check_keys(data_)
    return data_['info']


def _check_keys(checkdict: dict) -> None:
    """
    checks if entries in dictionary are mat-objects. If yes todict is called to change them to nested dictionaries.
    Modifies its parameter in-place.
    """

    for key in checkdict:
        if isinstance(checkdict[key], scipy.io.matlab.mat_struct):
            checkdict[key] = _todict(checkdict[key])


def _todict(matobj) -> dict:
    """
    A recursive function which constructs from matobjects nested dictionaries
    """

    ret = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, scipy.io.matlab.mat_struct):
            ret[strg] = _todict(elem)
        else:
            ret[strg] = elem
    return ret


def sbxread(filename: str, subindices: Optional[FileSubindices] = slice(None), channel: Optional[int] = None,
            plane: Optional[int] = None) -> np.ndarray:
    """
    Load frames of an .sbx file into a new NumPy array

    Args:
        filename: str
            filename should be full path excluding .sbx

        subindices: slice | array-like | tuple[slice | array-like, ...]
            which frames to read (defaults to all)
            if a tuple of non-scalars, specifies slices of up to 4 dimensions in the order (frame, Y, X, Z).

        channel: int | None
            which channel to load (required if data has >1 channel)

        plane: int | None
            set to an int to load only the given plane (converts from 3D to 2D data)
            in the case that len(subindices) == 4, subindices are applied first, then the plane is selected.
    """
    if subindices is None:
        subindices = slice(None)
    return _sbxread_helper(filename, subindices=subindices, channel=channel, plane=plane, chunk_size=None)


def sbx_to_tif(filename: str, fileout: Optional[str] = None, subindices: Optional[FileSubindices] = slice(None),
               bigtiff: Optional[bool] = True, imagej: bool = False, to32: bool = False,
               channel: Optional[int] = None, plane: Optional[int] = None, chunk_size: int = 1000):
    """
    Convert a single .sbx file to .tif format

    Args:
        filename: str
            filename should be full path excluding .sbx

        fileout: str | None
            filename to save (defaults to `filename` with .sbx replaced with .tif)

        subindices: slice | array-like | tuple[slice | array-like, ...]
            which frames to read (defaults to all)
            if a tuple of non-scalars, specifies slices of up to 4 dimensions in the order (frame, Y, X, Z).

        to32: bool
            whether to save in float32 format (default is to keep as uint16)

        channel: int | None
            which channel to save (required if data has >1 channel)

        plane: int | None
            set to an int to save only the given plane (converts from 3D to 2D data)
            in the case that len(subindices) == 4, subindices are applied first, then the plane is selected.

        chunk_size: int | None
            how many frames to load into memory at once (None = load the whole thing)
    """
    # Check filenames
    if fileout is None:
        basename, ext = os.path.splitext(filename)
        if ext == '.sbx':
            filename = basename
        fileout = filename + '.tif'

    if subindices is None:
        subindices = slice(None)

    sbx_chain_to_tif([filename], fileout, [subindices], bigtiff=bigtiff, imagej=imagej, to32=to32,
                     channel=channel, plane=plane, chunk_size=chunk_size)


def sbx_chain_to_tif(filenames: list[str], fileout: str, subindices: Optional[ChainSubindices] = slice(None),
                     bigtiff: Optional[bool] = True, imagej: bool = False, to32: bool = False,
                     channel: Optional[int] = None, plane: Optional[int] = None, chunk_size: int = 1000) -> None:
    """
    Concatenate a list of sbx files into one tif file.
    Args:
        filenames: list[str]
            each filename should be full path excluding .sbx

        fileout: str
            filename to save, including the .tif suffix
        
        subindices:  Sequence[int] | slice | Sequence[Sequence[int] | slice | tuple[Sequence[int] | slice, ...]]
            see subindices for sbx_to_tif
            can specify separate subindices for each file if nested 2 levels deep; 
            X, Y, and Z sizes must match for all files after indexing.

        to32, channel, plane, chunk_size: see sbx_to_tif
    """
    if subindices is None:
        subindices = slice(None)

    # Validate aggressively to avoid failing after waiting to copy a lot of data
    if isinstance(subindices, slice) or isinstance(subindices[0], int) or np.isscalar(subindices[0]):
        # One set of subindices over time to repeat for each file
        _subindices = [(cast(DimSubindices, subindices),) for _ in filenames]

    elif isinstance(subindices[0], slice) or isinstance(subindices[0][0], int) or np.isscalar(subindices[0][0]):
        # Interpret this as being an iterable over dimensions to repeat for each file
        _subindices = [cast(FileSubindices, subindices) for _ in filenames]

    elif len(subindices) != len(filenames):
        # Must be a separate subindices for each file; must match number of files
        raise Exception('Length of subindices does not match length of file list')
    
    else:
        _subindices = cast(Sequence[FileSubindices], subindices)
    del subindices  # ensure _subindices replaces subindices from here

    # Get the total size of the file
    all_shapes = [sbx_shape(file) for file in filenames]
    all_shapes_out = np.stack([_get_output_shape(file, subind)[0] for (file, subind) in zip(filenames, _subindices)])

    # Check that X, Y, and Z are consistent
    for dimname, shapes in zip(('Y', 'X', 'Z'), all_shapes_out.T[1:]):
        if np.any(np.diff(shapes) != 0):
            raise Exception(f'Given files have inconsistent shapes in the {dimname} dimension')
    
    # Check that all files have the requested channel
    if channel is None:
        if any(shape[0] > 1 for shape in all_shapes):
            raise Exception('At least one file has multiple channels; must specify channel')
        channel = 0
    elif any(shape[0] <= channel for shape in all_shapes):
        raise Exception('Not all files have the requested channel')

    # Allocate empty tif file with the final shape (do this first to ensure any existing file is overwritten)
    common_shape = tuple(map(int, all_shapes_out[0, 1:]))
    all_n_frames_out = list(map(int, all_shapes_out[:, 0]))
    n_frames_out = sum(all_n_frames_out)
    save_shape = (n_frames_out,) + common_shape

    if plane is not None:
        if len(save_shape) < 4:
            raise Exception('Plane cannot be specified for 2D data')
        save_shape = save_shape[:3]

    # Add a '.tif' extension if not already present
    extension = os.path.splitext(fileout)[1].lower()
    if extension not in ['.tif', '.tiff', '.btf']:
        fileout = fileout + '.tif'

    dtype = np.float32 if to32 else np.uint16
    # Make the file first so we can pass in bigtiff and imagej options; otherwise could create using tifffile.memmap directly
    tifffile.imwrite(fileout, data=None, shape=save_shape, bigtiff=bigtiff, imagej=imagej,
                     dtype=dtype, photometric='MINISBLACK', align=tifffile.TIFF.ALLOCATIONGRANULARITY)

    # Now convert each file
    tif_memmap = tifffile.memmap(fileout, series=0)
    offset = 0
    for filename, subind, file_N in zip(filenames, _subindices, all_n_frames_out):
        this_memmap = cast(np.memmap, tif_memmap[offset:offset+file_N])
        _sbxread_helper(filename, subindices=subind, channel=channel, out=this_memmap, plane=plane, chunk_size=chunk_size)
        offset += file_N

    del tif_memmap  # important to make sure file is closed (on Windows)


def sbx_shape(filename: str, info: Optional[dict[str, Any]] = None) -> tuple[int, int, int, int, int]:
    """
    Args:
        filename: str
            filename should be full path excluding .sbx

        info: dict | None
            info struct for sbx file (to avoid re-loading)

    Output: tuple (chans, X, Y, Z, frames) representing shape of scanbox data
    """
    basename, ext = os.path.splitext(filename)
    if ext == '.sbx':
        filename = basename    

    # Load info
    if info is None:
        info = loadmat_sbx(filename + '.mat')

    # Image size
    if 'sz' not in info:
        info['sz'] = np.array([512, 796])

    # Fold lines (multiple subframes per scan) - basically means the frames are smaller and
    # there are more of them than is reflected in the info file
    if 'fold_lines' in info and info['fold_lines'] > 0:
        if info['sz'][0] % info['fold_lines'] != 0:
            raise Exception('Non-integer folds per frame not supported')

        info['sz'][0] = info['fold_lines']
        if 'bytesPerBuffer' in info:
            n_folds = round(info['sz'][0] / info['fold_lines'])
            info['bytesPerBuffer'] /= n_folds


    # Defining number of channels/size factor
    if 'chan' in info:
        info['nChan'] = info['chan']['nchan']
    elif info['channels'] == 1:
        info['nChan'] = 2
    else:
        info['nChan'] = 1

    # Determine number of frames in whole file
    filesize = os.path.getsize(filename + '.sbx')
    if 'scanbox_version' in info:
        if info['scanbox_version'] in [2, 3]:
            info['max_idx'] = filesize / np.prod(info['sz']) / info['nChan'] / 2 - 1
        else:
            raise Exception('Invalid Scanbox version')
    else:
        info['max_idx'] = filesize / info['bytesPerBuffer'] * (2 // info['nChan']) - 1

    n_frames = info['max_idx'] + 1    # Last frame

    # Determine whether we are looking at a z-stack
    # Only consider optotune z-stacks - knobby schedules have too many possibilities and
    # can't determine whether it was actually armed from the saved info.
    if info['volscan']:
        n_planes = info['otparam'][2]
    else:
        n_planes = 1
    n_frames //= n_planes

    x = (int(info['nChan']), int(info['sz'][1]), int(info['sz'][0]), int(n_planes), int(n_frames))
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
    basename, ext = os.path.splitext(filename)
    if ext == '.sbx':
        filename = basename
    
    info = loadmat_sbx(filename + '.mat')

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


def _sbxread_helper(filename: str, subindices: FileSubindices = slice(None), channel: Optional[int] = None,
                    plane: Optional[int] = None, out: Optional[np.memmap] = None, chunk_size: Optional[int] = 1000) -> np.ndarray:
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
            existing memory-mapped file to write into

        plane: int | None
            set to an int to load only the given plane (converts from 3D to 2D data)
            in the case that len(subindices) == 4, subindices are applied first, then the plane is selected.

        chunk_size: int | None
            how many frames to load into memory at once (None = load the whole thing)
    """
    basename, ext = os.path.splitext(filename)
    if ext == '.sbx':
        filename = basename

    # Normalize so subindices is a list over dimensions
    if isinstance(subindices, slice) or isinstance(subindices[0], int) or np.isscalar(subindices[0]):
        _subindices = [cast(DimSubindices, subindices)]
    else:
        _subindices = list(cast(Sequence[DimSubindices], subindices))
    del subindices  # ensure _subindices replaces subindices from here

    # Load info
    info = loadmat_sbx(filename + '.mat')

    # Get shape (and update info)
    data_shape = sbx_shape(filename, info)  # (chans, X, Y, Z, frames)
    n_chans, n_x, n_y, n_planes, n_frames = data_shape
    is3D = n_planes > 1

    # Fill in missing dimensions in subindices
    _subindices += [slice(None) for _ in range(max(0, 3 + is3D - len(_subindices)))]

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

    save_shape, _subindices = _get_output_shape(data_shape, _subindices)
    n_frames_out = save_shape[0]
    if plane is not None:
        if len(save_shape) < 4:
            raise Exception('Plane cannot be specified for 2D data')
        save_shape = save_shape[:3]

    if out is not None and out.shape != save_shape:
        raise Exception('Existing memmap is the wrong shape to hold loaded data')

    # Read from .sbx file, using memmap to avoid loading until necessary
    sbx_mmap = np.memmap(filename + '.sbx', mode='r', dtype='uint16', shape=data_shape, order='F')
    sbx_mmap = np.transpose(sbx_mmap, (0, 4, 2, 1, 3))  # to (chans, frames, Y, X, Z)
    sbx_mmap = sbx_mmap[channel]
    if not is3D:  # squeeze out singleton plane dim
        sbx_mmap = sbx_mmap[..., 0]
    elif plane is not None:  # select plane relative to subindices
        sbx_mmap = sbx_mmap[..., _subindices[-1][plane]]
        _subindices = _subindices[:-1]
    inds = np.ix_(*_subindices)

    out_arr: Optional[np.ndarray] = out  # widen type
    del out  # ensure out_arr replaces out from here
    if chunk_size is None:
        # load a contiguous block all at once
        chunk_size = n_frames_out
    elif out_arr is None:
        # Pre-allocate destination when loading in chunks
        out_arr = np.empty(save_shape, dtype=np.uint16)

    n_remaining = n_frames_out
    offset = 0
    while n_remaining > 0:
        this_chunk_size = min(n_remaining, chunk_size)
        # Note: important to copy the data here instead of making a view,
        # so the memmap can be closed (achieved by advanced indexing)
        chunk = sbx_mmap[(inds[0][offset:offset+this_chunk_size],) + inds[1:]]
        # Note: SBX files store the values strangely, it's necessary to invert each uint16 value to get the correct ones
        np.invert(chunk, out=chunk)  # avoid copying, may be large

        if out_arr is None:
            out_arr = chunk  # avoid copying when loading all data
        else:
            out_arr[offset:offset+this_chunk_size] = chunk
        n_remaining -= this_chunk_size
        offset += this_chunk_size

    if out_arr is None:
        raise RuntimeError('Nothing loaded - no frames selected?')

    del sbx_mmap  # Important to close file (on Windows)

    if isinstance(out_arr, np.memmap):
        out_arr.flush()    
    return out_arr


def _interpret_subindices(subindices: DimSubindices, dim_extent: int) -> tuple[Sequence[int], int]:
    """
    Given the extent of a dimension in the corresponding recording, obtain a sequence over subindices 
    and the step size (or 0 if the step size is not uniform).
    """
    logger = logging.getLogger("caiman")

    if isinstance(subindices, slice):
        iterable_elements = range(dim_extent)[subindices]
        skip = iterable_elements.step

        if subindices.stop is not None and np.isfinite(subindices.stop) and subindices.stop > dim_extent:
            logger.warning(f'Only {dim_extent} frames or pixels available to load ' +
                            f'(requested up to {subindices.stop})')
    else:
        iterable_elements = subindices
        if isinstance(subindices, range):
            skip = subindices.step
        else:
            skip = 0

    return iterable_elements, skip


def _get_output_shape(filename_or_shape: Union[str, tuple[int, ...]], subindices: FileSubindices
                      ) -> tuple[tuple[int, ...], tuple[Sequence[int], ...]]:
    """
    Helper to determine what shape will be loaded/saved given subindices
    Also returns back the subindices with slices transformed to ranges, for convenience
    """
    if isinstance(subindices, slice) or np.isscalar(subindices[0]):
        _subindices = (cast(DimSubindices, subindices),)
    else:
        _subindices = cast(Sequence[DimSubindices], subindices)
    del subindices  # ensure _subindices replaces subindices from here
    
    n_inds = len(_subindices)  # number of dimensions that are indexed

    if isinstance(filename_or_shape, str):
        data_shape = sbx_shape(filename_or_shape)
    else:
        data_shape = filename_or_shape
    
    n_x, n_y, n_planes, n_frames = data_shape[1:]
    is3D = n_planes > 1
    if n_inds > 3 + is3D:
        raise Exception('Too many dimensions in subdindices')
    
    shape_out = [n_frames, n_y, n_x, n_planes] if is3D else [n_frames, n_y, n_x]
    subinds_out = []
    for i, (dim, subind) in enumerate(zip(shape_out, _subindices)):
        iterable_elements = _interpret_subindices(subind, dim)[0]
        shape_out[i] = len(iterable_elements)
        subinds_out.append(iterable_elements)

    return tuple(shape_out), tuple(subinds_out)
