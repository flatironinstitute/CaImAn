#!/usr/bin/env python

import ipyparallel as parallel
from itertools import chain
import logging
import numpy as np
import os
import pathlib
import pickle
import sys
import tifffile
from typing import Any, Optional, Union

import caiman
import caiman.paths

def prepare_shape(mytuple:tuple) -> tuple:
    """ This promotes the elements inside a shape into np.uint64. It is intended to prevent overflows
        with some numpy operations that are sensitive to it, e.g. np.memmap """
    if not isinstance(mytuple, tuple):
        raise Exception("Internal error: prepare_shape() passed a non-tuple")
    return tuple(map(lambda x: np.uint64(x), mytuple))

def load_memmap(filename: str, mode: str = 'r') -> tuple[Any, tuple, int]:
    """ Load a memory mapped file created by the function save_memmap

    Args:
        filename: str
            path of the file to be loaded
        mode: str
            One of 'r', 'r+', 'w+'. How to interact with files

    Returns:
        Yr:
            memory mapped variable

        dims: tuple
            frame dimensions

        T: int
            number of frames


    Raises:
        ValueError "Unknown file extension"

    """
    logger = logging.getLogger("caiman")
    allowed_extensions = {'.mmap', '.npy'}
    
    extension = pathlib.Path(filename).suffix
    if extension not in allowed_extensions:
        logger.error(f"Unknown extension for file {filename}")
        raise ValueError(f'Unknown file extension for file {filename} (should be .mmap or .npy)')
    # Strip path components and use CAIMAN_DATA/example_movies
    # TODO: Eventually get the code to save these in a different dir
    #fn_without_path = os.path.split(filename)[-1]
    #fpart = fn_without_path.split('_')[1:]  # The filename encodes the structure of the map
    decoded_fn = caiman.paths.decode_mmap_filename_dict(filename)
    d1		= decoded_fn['d1']
    d2		= decoded_fn['d2']
    d3		= decoded_fn['d3']
    T		= decoded_fn['T']
    order  	= decoded_fn['order']

    #d1, d2, d3, T, order = int(fpart[-9]), int(fpart[-7]), int(fpart[-5]), int(fpart[-1]), fpart[-3]

    filename = caiman.paths.fn_relocated(filename)
    shape = prepare_shape((d1 * d2 * d3, T))
    if extension == '.mmap':
        Yr = np.memmap(filename, mode=mode, shape=shape, dtype=np.float32, order=order)
    elif extension == '.npy':
        Yr = np.load(filename, mmap_mode=mode)
        if Yr.shape != shape:
            raise ValueError(f"Data in npy file was an unexpected shape: {Yr.shape}, expected: {shape}")
        if Yr.dtype != np.float32:
            raise ValueError(f"Data in npy file was an unexpected dtype: {Yr.dtype}, expected: np.float32")
        if order == 'C' and not Yr.flags['C_CONTIGUOUS']:
            raise ValueError("Data in npy file is not in C-contiguous order as expected.")
        elif order == 'F' and not Yr.flags['F_CONTIGUOUS']:
            raise ValueError("Data in npy file is not in Fortran-contiguous order as expected.")
    
    

    if d3 == 1:
        return (Yr, (d1, d2), T)
    else:
        return (Yr, (d1, d2, d3), T)

def save_memmap_each(fnames: list[str],
                     dview=None,
                     base_name: str = None,
                     resize_fact=(1, 1, 1),
                     remove_init: int = 0,
                     idx_xy=None,
                     var_name_hdf5='mov',
                     xy_shifts=None,
                     is_3D=False,
                     add_to_movie: float = 0,
                     border_to_0: int = 0,
                     order: str = 'C',
                     slices=None) -> list[str]:
    """
    Create several memory mapped files using parallel processing

    Args:
        fnames: list of str
            list of path to the filenames

        dview: ipyparallel dview
            used to perform computation in parallel. If none it will be single thread

        base_name str
            BaseName for the file to be creates. If not given the file itself is used

        resize_fact: tuple
            resampling factors for each dimension x,y,time. .1 = downsample 10X

        remove_init: int
            number of samples to remove from the beginning of each chunk

        idx_xy: slice operator
            used to perform slicing of the movie (to select a subportion of the movie)

        xy_shifts: list
            x and y shifts computed by a motion correction algorithm to be applied before memory mapping

        is_3D: boolean
            whether it is 3D data

        add_to_movie: float
            if movie too negative will make it positive

        border_to_0: int
            number of pixels on the border to set to the minimum of the movie

        order: (undocumented)

        slices: (undocumented)

    Returns:
        fnames_tot: list
            paths to the created memory map files

    """

    pars = []
    if xy_shifts is None:
        xy_shifts = [None] * len(fnames)

    if not isinstance(resize_fact, list):
        resize_fact = [resize_fact] * len(fnames)

    for idx, f in enumerate(fnames):
        if base_name is not None:
            pars.append([
                caiman.paths.fn_relocated(f),
                base_name + '{:04d}'.format(idx), resize_fact[idx], remove_init, idx_xy, order,
                var_name_hdf5, xy_shifts[idx], is_3D, add_to_movie, border_to_0, slices
            ])
        else:
            pars.append([
                caiman.paths.fn_relocated(f),
                os.path.splitext(f)[0], resize_fact[idx], remove_init, idx_xy, order, var_name_hdf5,
                xy_shifts[idx], is_3D, add_to_movie, border_to_0, slices
            ])

    # Perform the job using whatever computing framework we're set to use
    if dview is not None:
        if 'multiprocessing' in str(type(dview)):
            fnames_new = dview.map_async(save_place_holder, pars).get(4294967)
        else:
            fnames_new = my_map(dview, save_place_holder, pars)
    else:
        fnames_new = list(map(save_place_holder, pars))

    return fnames_new

def save_memmap_join(mmap_fnames:list[str], base_name: str = None, n_chunks: int = 20, dview=None,
                     add_to_mov=0) -> str:
    """
    Makes a large file memmap from a number of smaller files

    Args:
        mmap_fnames: list of memory mapped files

        base_name: string, will be the first portion of name to be solved

        n_chunks: number of chunks in which to subdivide when saving, smaller requires more memory

        dview: cluster handle

        add_to_mov: (undocumented)

    """
    logger = logging.getLogger("caiman")

    tot_frames = 0
    order = 'C'
    for f in mmap_fnames:
        cleaner_f = caiman.paths.fn_relocated(f)
        Yr, dims, T = load_memmap(cleaner_f)
        logger.debug(f"save_memmap_join (loading data): {cleaner_f} {T}")
        tot_frames += T
        del Yr

    d = np.prod(dims)

    if base_name is None:
        base_name = mmap_fnames[0]
        base_name = base_name[:base_name.find('_d1_')] + f'-#-{len(mmap_fnames)}'

    fname_tot = caiman.paths.memmap_frames_filename(base_name, dims, tot_frames, order)
    fname_tot = os.path.join(os.path.split(mmap_fnames[0])[0], fname_tot)
    fname_tot = caiman.paths.fn_relocated(fname_tot)
    logger.info(f"Memmap file for fname_tot: {fname_tot}")

    big_mov = np.memmap(fname_tot, mode='w+', dtype=np.float32, shape=prepare_shape((d, tot_frames)), order='C')

    step = int(d // n_chunks)
    pars = []
    for ref in range(0, d - step + 1, step):
        pars.append([fname_tot, d, tot_frames, mmap_fnames, ref, ref + step, add_to_mov])

    if len(pars[-1]) != 7:
        raise Exception(
            'You cannot change the number of element in list without changing the statement below (pars[]..)')
    else:
        # last batch should include the leftover pixels
        pars[-1][-2] = d

    if dview is not None:
        if 'multiprocessing' in str(type(dview)):
            dview.map_async(save_portion, pars).get(4294967)
        else:
            my_map(dview, save_portion, pars)

    else:
        list(map(save_portion, pars))

    np.savez(caiman.paths.fn_relocated(base_name + '.npz'), mmap_fnames=mmap_fnames, fname_tot=fname_tot)

    logger.info('Deleting big mov')
    del big_mov
    sys.stdout.flush()
    return fname_tot


def my_map(dv, func, args) -> list:
    logger = logging.getLogger("caiman")

    v = dv
    rc = v.client
    # scatter 'id', so id=0,1,2 on engines 0,1,2
    dv.scatter('id', rc.ids, flatten=True)
    logger.debug(dv['id'])
    amr = v.map(func, args)

    pending = set(amr.msg_ids)
    results_all:dict = dict()
    counter = 0
    while pending:
        try:
            rc.wait(pending, 1e0)
        except parallel.TimeoutError:
            # ignore timeouterrors, since it means at least one isn't done
            pass
        # finished is the set of msg_ids that are complete
        finished = pending.difference(rc.outstanding)
        # update pending to exclude those that just finished
        pending = pending.difference(finished)
        if counter % 10 == 0:
            logger.debug(amr.progress)
        for msg_id in finished:
            # we know these are done, so don't worry about blocking
            ar = rc.get_result(msg_id)
            logger.debug(f"job id {msg_id} finished on engine {ar.engine_id}") # TODO: Abstract out the ugly bits
            logger.debug("with stdout:")
            logger.debug('    ' + ar.stdout.replace('\n', '\n    ').rstrip())
            logger.debug("and errors:")
            logger.debug('    ' + ar.stderr.replace('\n', '\n    ').rstrip())
                                                                                      # note that each job in a map always returns a list of length chunksize
                                                                                      # even if chunksize == 1
            results_all.update(ar.get_dict())
        counter += 1

    result_ordered = list(chain.from_iterable([results_all[k] for k in sorted(results_all.keys())]))
    del results_all
    return result_ordered


def save_portion(pars) -> int:
    # todo: todocument
    logger = logging.getLogger("caiman")

    use_mmap_save = False
    big_mov_fn, d, tot_frames, fnames, idx_start, idx_end, add_to_mov = pars
    big_mov_fn = caiman.paths.fn_relocated(big_mov_fn)

    Ttot = 0
    Yr_tot = np.zeros((idx_end - idx_start, tot_frames), dtype=np.float32)
    logger.debug(f"Shape of Yr_tot is {Yr_tot.shape}")
    for f in fnames:
        full_f = caiman.paths.fn_relocated(f)
        logger.debug(f"Saving portion to {full_f}")
        Yr, _, T = load_memmap(full_f)
        Yr_tot[:, Ttot:Ttot +
               T] = np.ascontiguousarray(Yr[idx_start:idx_end], dtype=np.float32) + np.float32(add_to_mov)
        Ttot = Ttot + T
        del Yr

    logger.debug(f"Index start and end are {idx_start} and {idx_end}")

    if use_mmap_save:
        big_mov = np.memmap(big_mov_fn, mode='r+', dtype=np.float32, shape=prepare_shape((d, tot_frames)), order='C')
        big_mov[idx_start:idx_end, :] = Yr_tot
        del big_mov
    else:
        with open(big_mov_fn, 'r+b') as f:
            idx_start = np.uint64(idx_start)
            tot_frames = np.uint64(tot_frames)
            f.seek(np.uint64(idx_start * np.uint64(Yr_tot.dtype.itemsize) * tot_frames))
            f.write(Yr_tot)
            computed_position = np.uint64(idx_end * np.uint64(Yr_tot.dtype.itemsize) * tot_frames)
            if f.tell() != computed_position:
                logger.critical(f"Error in mmap portion write: at position {f.tell()}")
                logger.critical(
                    f"But should be at position {idx_end} * {Yr_tot.dtype.itemsize} * {tot_frames} = {computed_position}"
                )
                f.close()
                raise Exception('Internal error in mmapping: Actual position does not match computed position')

    del Yr_tot
    logger.debug('done')
    return Ttot

def save_place_holder(pars:list) -> str:
    """ To use map reduce
    """
    # todo: todocument

    (f, base_name, resize_fact, remove_init, idx_xy, order, var_name_hdf5, xy_shifts, is_3D, add_to_movie, border_to_0, slices) = pars

    return save_memmap([f],
                       base_name=base_name,
                       resize_fact=resize_fact,
                       remove_init=remove_init,
                       idx_xy=idx_xy,
                       order=order,
                       var_name_hdf5=var_name_hdf5,
                       xy_shifts=xy_shifts,
                       is_3D=is_3D,
                       add_to_movie=add_to_movie,
                       border_to_0=border_to_0,
                       slices=slices)

def save_memmap(filenames:list[str],
                base_name:str = 'Yr',
                resize_fact:tuple = (1, 1, 1),
                remove_init:int = 0,
                idx_xy:tuple = None,
                order: str = 'F',
                var_name_hdf5: str = 'mov',
                xy_shifts: Optional[list] = None,
                is_3D: bool = False,
                add_to_movie: float = 0,
                border_to_0=0,
                dview=None,
                n_chunks: int = 100,
                slices=None) -> str:
    """ Efficiently write data from a list of tif files into a memory mappable file

    Args:
        filenames: list
            list of tif files or list of numpy arrays

        base_name: str
            the base used to build the file name. WARNING: Names containing underscores may collide with internal semantics.

        resize_fact: tuple
            x,y, and z downsampling factors (0.5 means downsampled by a factor 2)

        remove_init: int
            number of frames to remove at the beginning of each tif file
            (used for resonant scanning images if laser in rutned on trial by trial)

        idx_xy: tuple size 2 [or 3 for 3D data]
            for selecting slices of the original FOV, for instance
            idx_xy = (slice(150,350,None), slice(150,350,None))

        order: string
            whether to save the file in 'C' or 'F' order

        xy_shifts: list
            x and y shifts computed by a motion correction algorithm to be applied before memory mapping

        is_3D: boolean
            whether it is 3D data

        add_to_movie: floating-point
            value to add to each image point, typically to keep negative values out.

        border_to_0: (undocumented)

        dview:       (undocumented)

        n_chunks:    (undocumented)

        slices: slice object or list of slice objects
            slice can be used to select portion of the movies in time and x,y
            directions. For instance
            slices = [slice(0,200),slice(0,100),slice(0,100)] will take
            the first 200 frames and the 100 pixels along x and y dimensions.
    Returns:
        fname_new: the name of the mapped file, the format is such that
            the name will contain the frame dimensions and the number of frames

    """
    logger = logging.getLogger("caiman")

    if not isinstance(filenames, list):
        raise Exception('save_memmap: input should be a list of filenames')

    if slices is not None:
        slices = [slice(0, None) if sl is None else sl for sl in slices]

    if len(filenames) > 1:
        recompute_each_memmap = False
        for file__ in filenames:
            if ('order_' + order not in file__) or ('.mmap' not in file__):
                recompute_each_memmap = True


        if recompute_each_memmap or (remove_init > 0) or (idx_xy is not None)\
                or (xy_shifts is not None) or (add_to_movie != 0) or (border_to_0>0)\
                or slices is not None:

            logger.debug('Distributing memory map over many files')
            # Here we make a bunch of memmap files in the right order. Same parameters
            fname_parts = caiman.save_memmap_each(filenames,
                                              base_name=base_name,
                                              order=order,
                                              border_to_0=border_to_0,
                                              dview=dview,
                                              var_name_hdf5=var_name_hdf5,
                                              resize_fact=resize_fact,
                                              remove_init=remove_init,
                                              idx_xy=idx_xy,
                                              xy_shifts=xy_shifts,
                                              is_3D=is_3D,
                                              slices=slices,
                                              add_to_movie=add_to_movie)
        else:
            fname_parts = filenames

        # The goal is to make a single large memmap file, which we do here
        if order == 'F':
            raise Exception('You cannot merge files in F order, they must be in C order')

        fname_new = caiman.save_memmap_join(fname_parts, base_name=base_name,
                                        dview=dview, n_chunks=n_chunks)

    else:
        # TODO: can be done online
        Ttot = 0
        for idx, f in enumerate(filenames):
            if isinstance(f, str):     # Might not always be filenames.
                logger.debug(f)

            if is_3D:
                Yr = f if not (isinstance(f, str)) else tifffile.imread(f)
                if Yr.ndim == 3:
                    Yr = Yr[None, ...]
                if slices is not None:
                    Yr = Yr[tuple(slices)]
                else:
                    if idx_xy is None:         #todo remove if not used, superseded by the slices parameter
                        Yr = Yr[remove_init:]
                    elif len(idx_xy) == 2:     #todo remove if not used, superseded by the slices parameter
                        Yr = Yr[remove_init:, idx_xy[0], idx_xy[1]]
                    else:                      #todo remove if not used, superseded by the slices parameter
                        Yr = Yr[remove_init:, idx_xy[0], idx_xy[1], idx_xy[2]]

            else:
                if isinstance(f, (str, list)):
                    Yr = caiman.load(caiman.paths.fn_relocated(f), fr=1, in_memory=True, var_name_hdf5=var_name_hdf5)
                else:
                    Yr = caiman.movie(f)
                if xy_shifts is not None:
                    Yr = Yr.apply_shifts(xy_shifts, interpolation='cubic', remove_blanks=False)

                if slices is not None:
                    Yr = Yr[tuple(slices)]
                else:
                    if idx_xy is None:
                        if remove_init > 0:
                            Yr = Yr[remove_init:]
                    elif len(idx_xy) == 2:
                        Yr = Yr[remove_init:, idx_xy[0], idx_xy[1]]
                    else:
                        raise Exception('You need to set is_3D=True for 3D data)')
                        Yr = np.array(Yr)[remove_init:, idx_xy[0], idx_xy[1], idx_xy[2]]

            if border_to_0 > 0:
                if slices is not None:
                    if isinstance(slices, list):
                        raise Exception(
                            'You cannot slice in x and y and then use add_to_movie: if you only want to slice in time do not pass in a list but just a slice object'
                        )

                min_mov = Yr.calc_min()
                Yr[:, :border_to_0, :] = min_mov
                Yr[:, :, :border_to_0] = min_mov
                Yr[:, :, -border_to_0:] = min_mov
                Yr[:, -border_to_0:, :] = min_mov

            fx, fy, fz = resize_fact
            if fx != 1 or fy != 1 or fz != 1:
                if 'movie' not in str(type(Yr)):
                    Yr = caiman.movie(Yr, fr=1)
                Yr = Yr.resize(fx=fx, fy=fy, fz=fz)

            T, dims = Yr.shape[0], Yr.shape[1:]
            Yr = np.transpose(Yr, list(range(1, len(dims) + 1)) + [0])
            Yr = np.reshape(Yr, (np.prod(dims), T), order='F')
            Yr = np.ascontiguousarray(Yr, dtype=np.float32) + np.float32(0.0001) + np.float32(add_to_movie)

            if idx == 0:
                fname_tot = caiman.paths.generate_fname_tot(base_name, dims, order)
                if isinstance(f, str):
                    fname_tot = caiman.paths.fn_relocated(os.path.join(os.path.split(f)[0], fname_tot))
                if len(filenames) > 1:
                    big_mov = np.memmap(caiman.paths.fn_relocated(fname_tot),
                                        mode='w+',
                                        dtype=np.float32,
                                        shape=prepare_shape((np.prod(dims), T)),
                                        order=order)
                    big_mov[:, Ttot:Ttot + T] = Yr
                    del big_mov
                else:
                    logger.debug('SAVING WITH numpy.tofile()')
                    Yr.tofile(fname_tot)
            else:
                big_mov = np.memmap(fname_tot,
                                    dtype=np.float32,
                                    mode='r+',
                                    shape=prepare_shape((np.prod(dims), Ttot + T)),
                                    order=order)

                big_mov[:, Ttot:Ttot + T] = Yr
                del big_mov

            sys.stdout.flush()
            Ttot = Ttot + T

        fname_new = caiman.paths.fn_relocated(f'{fname_tot}_frames_{Ttot}.mmap')
        try:
            # need to explicitly remove destination on windows
            os.unlink(fname_new)
        except OSError:
            pass
        os.rename(fname_tot, fname_new)

    return fname_new

def parallel_dot_product(A: np.ndarray, b, block_size: int = 5000, dview=None, transpose=False,
                         num_blocks_per_run=20) -> np.ndarray:
    # todo: todocument
    """ Chunk matrix product between matrix and column vectors

    Args:
        A: memory mapped ndarray
            pixels x time

        b: time x comps
    """
    logger = logging.getLogger("caiman")

    pars = []
    d1, d2 = A.shape
    b = pickle.dumps(b)
    logger.debug(f'parallel dot product block size: {block_size}')

    if block_size < d1:
        for idx in range(0, d1 - block_size, block_size):
            idx_to_pass = list(range(idx, idx + block_size))
            pars.append([A.filename, idx_to_pass, b, transpose])

        if (idx + block_size) < d1:
            idx_to_pass = list(range(idx + block_size, d1))
            pars.append([A.filename, idx_to_pass, b, transpose])

    else:
        idx_to_pass = list(range(d1))
        pars.append([A.filename, idx_to_pass, b, transpose])

    logger.debug('Start product')
    b = pickle.loads(b)

    if transpose:
        output = np.zeros((d2, b.shape[-1]), dtype=np.float32)
    else:
        output = np.zeros((d1, b.shape[-1]), dtype=np.float32)

    if dview is None:
        if transpose:
            #            b = pickle.loads(b)
            logger.debug('Transposing')
            for _, pr in enumerate(pars):
                iddx, rs = dot_place_holder(pr)
                output = output + rs
        else:
            for _, pr in enumerate(pars):
                iddx, rs = dot_place_holder(pr)
                output[iddx] = rs

    else:
        for itera in range(0, len(pars), num_blocks_per_run):

            if 'multiprocessing' in str(type(dview)):
                results = dview.map_async(dot_place_holder, pars[itera:itera + num_blocks_per_run]).get(4294967)
            else:
                results = dview.map_sync(dot_place_holder, pars[itera:itera + num_blocks_per_run])

            logger.debug('Processed:' + str([itera, itera + len(results)]))

            if transpose:
                logger.debug('Transposing')

                for _, res in enumerate(results):
                    output += res[1]

            else:
                logger.debug('Filling')
                for res in results:
                    output[res[0]] = res[1]

            if 'multiprocessing' not in str(type(dview)):
                dview.clear()

    return output

def dot_place_holder(par:list) -> tuple:
    # todo: todocument
    logger = logging.getLogger("caiman")

    A_name, idx_to_pass, b_, transpose = par
    A_, _, _ = load_memmap(A_name)
    b_ = pickle.loads(b_).astype(np.float32)

    logger.debug((idx_to_pass[-1]))
    if 'sparse' in str(type(b_)):
        if transpose:
            #            outp = (b_.tocsr()[idx_to_pass].T.dot(
            #                A_[idx_to_pass])).T.astype(np.float32)
            outp = (b_.T.tocsc()[:, idx_to_pass].dot(A_[idx_to_pass])).T.astype(np.float32)
        else:
            outp = (b_.T.dot(A_[idx_to_pass].T)).T.astype(np.float32)
    else:
        if transpose:
            outp = A_[idx_to_pass].T.dot(b_[idx_to_pass]).astype(np.float32)
        else:
            outp = A_[idx_to_pass].dot(b_).astype(np.float32)

    del b_, A_
    return idx_to_pass, outp

def save_tif_to_mmap_online(movie_iterable, save_base_name='YrOL_', order='C', add_to_movie=0, border_to_0=0) -> str:
    # todo: todocument
    logger = logging.getLogger("caiman")

    if isinstance(movie_iterable, str):         # Allow specifying a filename rather than its data rep
        with tifffile.TiffFile(movie_iterable) as tf:  # And load it if that happens
            movie_iterable = caiman.movie(tf)

    count = 0
    new_mov = []

    dims = (len(movie_iterable),) + movie_iterable[0].shape    # TODO: Don't pack length into dims

    fname_tot = caiman.paths.fn_relocated(caiman.paths.memmap_frames_filename(save_base_name, dims[1:], dims[0], order))

    big_mov = np.memmap(fname_tot,
                        mode='w+',
                        dtype=np.float32,
                        shape=prepare_shape((np.prod(dims[1:]), dims[0])),
                        order=order)

    for page in movie_iterable:
        if count % 100 == 0:
            logger.debug(count)

        if 'tifffile' in str(type(movie_iterable[0])):
            page = page.asarray()

        img = np.array(page, dtype=np.float32)
        new_img = img
        if save_base_name is not None:
            big_mov[:, count] = np.reshape(new_img, np.prod(dims[1:]), order='F')
        else:
            new_mov.append(new_img)

        if border_to_0 > 0:
            img[:border_to_0, :] = 0
            img[:, :border_to_0] = 0
            img[:, -border_to_0:] = 0
            img[-border_to_0:, :] = 0

        big_mov[:, count] = np.reshape(img + add_to_movie, np.prod(dims[1:]), order='F')

        count += 1
    big_mov.flush()
    del big_mov
    return fname_tot
