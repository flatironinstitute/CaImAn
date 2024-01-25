#!/usr/bin/env python

""" functions related to the creation and management of the "cluster",
meaning the framework for distributed computation.

We put arrays on disk as raw bytes, extending along the first dimension.
Alongside each array x we ensure the value x.dtype which stores the data type.
"""

import glob
import ipyparallel
from ipyparallel import Client
import logging
import multiprocessing
from multiprocessing import Pool
import numpy as np
import os
import platform
import psutil
import shlex
import shutil
import subprocess
import sys
import time
from typing import Any, Optional, Union

from .mmapping import load_memmap

logger = logging.getLogger(__name__)


def extract_patch_coordinates(dims: tuple,
                              rf: Union[list, tuple],
                              stride: Union[list[int], tuple],
                              border_pix: int = 0,
                              indices=[slice(None)] * 2) -> tuple[list, list]:
    """
    Partition the FOV in patches
    and return the indexed in 2D and 1D (flatten, order='F') formats

    Args:
        dims: tuple of int
            dimensions of the original matrix that will be divided in patches

        rf: tuple of int
            radius of receptive field, corresponds to half the size of the square patch

        stride: tuple of int
            degree of overlap of the patches
    """

    # TODO: Find a new home for this function
    sl_start = [0 if sl.start is None else sl.start for sl in indices]
    sl_stop = [dim if sl.stop is None else sl.stop for (sl, dim) in zip(indices, dims)]
    sl_step = [1 for sl in indices]    # not used
    dims_large = dims
    dims = np.minimum(np.array(dims) - border_pix, sl_stop) - np.maximum(border_pix, sl_start)

    coords_flat = []
    shapes = []
    iters = [list(range(rf[i], dims[i] - rf[i], 2 * rf[i] - stride[i])) + [dims[i] - rf[i]] for i in range(len(dims))]

    coords = np.empty(list(map(len, iters)) + [len(dims)], dtype=object)
    for count_0, xx in enumerate(iters[0]):
        coords_x = np.arange(xx - rf[0], xx + rf[0] + 1)
        coords_x = coords_x[(coords_x >= 0) & (coords_x < dims[0])]
        coords_x += border_pix * 0 + np.maximum(sl_start[0], border_pix)

        for count_1, yy in enumerate(iters[1]):
            coords_y = np.arange(yy - rf[1], yy + rf[1] + 1)
            coords_y = coords_y[(coords_y >= 0) & (coords_y < dims[1])]
            coords_y += border_pix * 0 + np.maximum(sl_start[1], border_pix)

            if len(dims) == 2:
                idxs = np.meshgrid(coords_x, coords_y)

                coords[count_0, count_1] = idxs
                shapes.append(idxs[0].shape[::-1])

                coords_ = np.ravel_multi_index(idxs, dims_large, order='F')
                coords_flat.append(coords_.flatten())
            else:      # 3D data

                if border_pix > 0:
                    raise Exception(
                        'The parameter border pix must be set to 0 for 3D data since border removal is not implemented')

                for count_2, zz in enumerate(iters[2]):
                    coords_z = np.arange(zz - rf[2], zz + rf[2] + 1)
                    coords_z = coords_z[(coords_z >= 0) & (coords_z < dims[2])]
                    idxs = np.meshgrid(coords_x, coords_y, coords_z)
                    shps = idxs[0].shape
                    shapes.append([shps[1], shps[0], shps[2]])
                    coords[count_0, count_1, count_2] = idxs
                    coords_ = np.ravel_multi_index(idxs, dims, order='F')
                    coords_flat.append(coords_.flatten())

    for i, c in enumerate(coords_flat):
        assert len(c) == np.prod(shapes[i])

    return list(map(np.sort, coords_flat)), shapes

def start_server(ipcluster: str = "ipcluster", ncpus: int = None) -> None:
    """
    programmatically start the ipyparallel server

    Args:
        ncpus
            number of processors

        ipcluster
            ipcluster binary file name; requires 4 path separators on Windows. ipcluster="C:\\\\Anaconda3\\\\Scripts\\\\ipcluster.exe"
            Default: "ipcluster"
    """
    logger.info("Starting cluster...")
    if ncpus is None:
        ncpus = psutil.cpu_count()

    if ipcluster == "ipcluster":
        subprocess.Popen(f"ipcluster start -n {ncpus}", shell=True, close_fds=(os.name != 'nt'))
    else:
        subprocess.Popen(shlex.split(f"{ipcluster} start -n {ncpus}"),
                         shell=True,
                         close_fds=(os.name != 'nt'))
    time.sleep(1.5)
    # Check that all processes have started
    client = ipyparallel.Client()
    time.sleep(1.5)
    while len(client) < ncpus:
        sys.stdout.write(".")                              # Give some visual feedback of things starting
        sys.stdout.flush()                                 # (de-buffered)
        time.sleep(0.5)
    logger.debug('Making sure everything is up and running')
    client.direct_view().execute('__a=1', block=True)      # when done on all, we're set to go

def stop_server(ipcluster: str = 'ipcluster', pdir: str = None, profile: str = None, dview=None) -> None:
    """
    programmatically stops the ipyparallel server

    Args:
        ipcluster : str
            ipcluster binary file name; requires 4 path separators on Windows
            Default: "ipcluster"a

        pdir : Undocumented
        profile: Undocumented
        dview: Undocumented

    """
    if 'multiprocessing' in str(type(dview)):
        dview.terminate()
    else:
        logger.info("Stopping cluster...")

        if ipcluster == "ipcluster":
            proc = subprocess.Popen("ipcluster stop",
                                    shell=True,
                                    stderr=subprocess.PIPE,
                                    close_fds=(os.name != 'nt'))
        else:
            proc = subprocess.Popen(shlex.split(ipcluster + " stop"),
                                    shell=True,
                                    stderr=subprocess.PIPE,
                                    close_fds=(os.name != 'nt'))

        line_out = proc.stderr.readline()
        if b'CRITICAL' in line_out:
            logger.info("No cluster to stop...")
        elif b'Stopping' in line_out:
            st = time.time()
            logger.debug('Waiting for cluster to stop...')
            while (time.time() - st) < 4:
                sys.stdout.write('.')
                sys.stdout.flush()
                time.sleep(1)
        else:
            logger.error(line_out)
            logger.error('**** Unrecognized syntax in ipcluster output, waiting for server to stop anyways ****')

        proc.stderr.close()

    logger.info("stop_cluster(): done")

def setup_cluster(backend:str = 'multiprocessing',
                  n_processes:Optional[int] = None,
                  single_thread:bool = False,
                  ignore_preexisting:bool = False,
                  maxtasksperchild:int = None) -> tuple[Any, Any, Optional[int]]:
    """
    Setup and/or restart a parallel cluster.
    Args:
        backend
            One of:
                'multiprocessing' - Use multiprocessing library
                'ipyparallel' - Use ipyparallel instead (better on Windows?)
                'single' - Don't be parallel (good for debugging, slow)
            Most backends will try, by default, to stop a running cluster if
            it is running before setting up a new one, or throw an error if
            they find one.
        n_processes
            Sets number of processes to use. If None, is set automatically. 
        single_thread
            Deprecated alias for the 'single' backend.
        ignore_preexisting
            If True, ignores the existence of an already running multiprocessing
            pool (which usually indicates a previously-started CaImAn cluster)
        maxtasksperchild
            Only used for multiprocessing, default None (number of tasks a worker process can 
            complete before it will exit and be replaced with a fresh worker process).
            
    Returns:
        c: ipyparallel.Client object; only used for ipyparallel backends, else None
        dview: multicore processing engine that is used for parallel processing. 
            If backend is 'multiprocessing' then dview is Pool object.
            If backend is 'ipyparallel' then dview is a DirectView object. 
        n_processes: number of workers in dview. None means single core mode in use. 
    """

    sys.stdout.flush() # XXX Unsure why we do this
    if n_processes is None:
        n_processes = np.maximum(int(psutil.cpu_count() - 1), 1)

    if backend == 'multiprocessing' or backend == 'local':
        if backend == 'local':
            logger.warn('The local backend is an alias for the multiprocessing backend, and the alias may be removed in some future version of Caiman')
        if len(multiprocessing.active_children()) > 0:
            if ignore_preexisting:
                logger.warn('Found an existing multiprocessing pool. '
                            'This is often indicative of an already-running CaImAn cluster. '
                            'You have configured the cluster setup to not raise an exception.')
            else:
                raise Exception(
                    'A cluster is already running. Terminate with dview.terminate() if you want to restart.')
        if platform.system() == 'Darwin':
            try:
                if 'kernel' in get_ipython().trait_names():        # type: ignore
                                                                   # If you're on OSX and you're running under Jupyter or Spyder,
                                                                   # which already run the code in a forkserver-friendly way, this
                                                                   # can eliminate some setup and make this a reasonable approach.
                                                                   # Otherwise, setting VECLIB_MAXIMUM_THREADS=1 or using a different
                                                                   # blas/lapack is the way to avoid the issues.
                                                                   # See https://github.com/flatironinstitute/CaImAn/issues/206 for more
                                                                   # info on why we're doing this (for now).
                    multiprocessing.set_start_method('forkserver', force=True)
            except:                                                # If we're not running under ipython, don't do anything.
                pass
        c = None
        dview = Pool(n_processes, maxtasksperchild=maxtasksperchild)

    elif backend == 'ipyparallel':
        stop_server()
        start_server(ncpus=n_processes)
        c = Client()
        logger.info(f'Started ipyparallel cluster: Using {len(c)} processes')
        dview = c[:len(c)]

    elif backend == "single" or single_thread:
        if single_thread:
            logger.warn('The single_thread flag to setup_cluster() is deprecated and may be removed in the future')
        dview = None
        c = None
        n_processes = 1

    else:
        raise Exception('Unknown Backend')

    return c, dview, n_processes
