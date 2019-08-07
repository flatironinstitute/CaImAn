#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" functions related to the creation and management of the cluster

We put arrays on disk as raw bytes, extending along the first dimension.
Alongside each array x we ensure the value x.dtype which stores the data type.

@author andrea giovannucci
"""

# \package caiman
# \version   1.0
# \copyright GNU General Public License v2.0
# \date Created on Thu Oct 20 12:07:09 2016

from builtins import zip
from builtins import str
from builtins import map
from builtins import range

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
from typing import Any, Dict, List, Optional, Tuple, Union

from .mmapping import load_memmap

logger = logging.getLogger(__name__)


def extract_patch_coordinates(dims: Tuple,
                              rf: Union[List, Tuple],
                              stride: Union[List[int], Tuple],
                              border_pix: int = 0,
                              indices=[slice(None)] * 2) -> Tuple[List, List]:
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

    sl_start = [0 if sl.start is None else sl.start for sl in indices]
    sl_stop = [dim if sl.stop is None else sl.stop for (sl, dim) in zip(indices, dims)]
    sl_step = [1 for sl in indices]    # not used
    dims_large = dims
    dims = np.minimum(np.array(dims) - border_pix, sl_stop) - np.maximum(border_pix, sl_start)

    coords_flat = []
    shapes = []
    iters = [list(range(rf[i], dims[i] - rf[i], 2 * rf[i] - stride[i])) + [dims[i] - rf[i]] for i in range(len(dims))]

    coords = np.empty(list(map(len, iters)) + [len(dims)], dtype=np.object)
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


#%%
def apply_to_patch(mmap_file, shape: Tuple[Any, Any, Any], dview, rf, stride, function, *args,
                   **kwargs) -> Tuple[List, Any, Tuple]:
    """
    apply function to patches in parallel or not

    Args:
        mmap_file: Memory-mapped variable
            Variable handle, first thing returned by load_memmap()

        shape: tuple of three elements
            dimensions of the original movie across y, x, and time

        dview: ipyparallel view on client
            if None

        rf: int
            half-size of the square patch in pixel

        stride: int
            amount of overlap between patches

    Returns:
        results

    Raises:
        Exception 'Something went wrong'

    """
    # This function is (currently) only used in some demos and use_cases code. Unclear if it's still necessary,
    # and it's been broken for awhile.
    (_, d1, d2) = shape

    if not np.isscalar(rf):
        rf1, rf2 = rf
    else:
        rf1 = rf
        rf2 = rf

    if not np.isscalar(stride):
        stride1, stride2 = stride
    else:
        stride1 = stride
        stride2 = stride

    idx_flat, idx_2d = extract_patch_coordinates((d1, d2), rf=(rf1, rf2), stride=(stride1, stride2))

    shape_grid = tuple(np.ceil((d1 * 1. / (rf1 * 2 - stride1), d2 * 1. / (rf2 * 2 - stride2))).astype(np.int))
    if d1 <= rf1 * 2:
        shape_grid = (1, shape_grid[1])
    if d2 <= rf2 * 2:
        shape_grid = (shape_grid[0], 1)

    logger.debug("Shape of grid is " + str(shape_grid))

    args_in = []

    for id_f, id_2d in zip(idx_flat[:], idx_2d[:]):
        args_in.append((mmap_file.filename, id_f, id_2d, function, args, kwargs))

    logger.debug("Flat index is of length " + str(len(idx_flat)))
    if dview is not None:
        try:
            file_res = dview.map_sync(function_place_holder, args_in)
            dview.results.clear()

        except:
            raise Exception('Something went wrong')
        finally:
            logger.warn('You may think that it went well but reality is harsh') # TODO Figure out a better message
    else:

        file_res = list(map(function_place_holder, args_in))
    return file_res, idx_flat, shape_grid


#%%


def function_place_holder(args_in: Tuple) -> np.ndarray:
    #todo: todocument

    file_name, idx_, shapes, function, args, kwargs = args_in
    Yr, _, _ = load_memmap(file_name)
    Yr = Yr[idx_, :]
    Yr.filename = file_name
    _, T = Yr.shape
    Y = np.reshape(Yr, (shapes[1], shapes[0], T), order='F').transpose([2, 0, 1])
    [T, d1, d2] = Y.shape

    res_fun = function(Y, *args, **kwargs)
    if type(res_fun) is not tuple:

        if res_fun.shape == (d1, d2):
            logger.debug('** reshaping form 2D to 1D')
            res_fun = np.reshape(res_fun, d1 * d2, order='F')

    return res_fun


#%%


def start_server(slurm_script: str = None, ipcluster: str = "ipcluster", ncpus: int = None) -> None:
    """
    programmatically start the ipyparallel server

    Args:
        ncpus: int
            number of processors

        ipcluster : str
            ipcluster binary file name; requires 4 path separators on Windows. ipcluster="C:\\\\Anaconda3\\\\Scripts\\\\ipcluster.exe"
            Default: "ipcluster"
    """
    logger.info("Starting cluster...")
    if ncpus is None:
        ncpus = psutil.cpu_count()

    if slurm_script is None:

        if ipcluster == "ipcluster":
            subprocess.Popen("ipcluster start -n {0}".format(ncpus), shell=True, close_fds=(os.name != 'nt'))
        else:
            subprocess.Popen(shlex.split("{0} start -n {1}".format(ipcluster, ncpus)),
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
    else:
        shell_source(slurm_script)
        pdir, profile = os.environ['IPPPDIR'], os.environ['IPPPROFILE']
        logger.debug([pdir, profile])
        c = Client(ipython_dir=pdir, profile=profile)
        ee = c[:]
        ne = len(ee)
        logger.info(('Running on %d engines.' % (ne)))
        c.close()
        sys.stdout.write("start_server: done\n")


def shell_source(script: str) -> None:
    """ Run a source-style bash script, copy resulting env vars to current process. """
    # XXX This function is weird and maybe not a good idea. People easily might expect
    #     it to handle conditionals. Maybe just make them provide a key-value file
    #introduce echo to indicate the end of the output
    pipe = subprocess.Popen(". %s; env; echo 'FINISHED_CLUSTER'" % script, stdout=subprocess.PIPE, shell=True)

    env = dict()
    while True:
        line = pipe.stdout.readline().decode('utf-8').rstrip()
        if 'FINISHED_CLUSTER' in line:         # find the keyword set above to determine the end of the output stream
            break
        logger.debug("shell_source parsing line[" + str(line) + "]")
        lsp = str(line).split("=", 1)
        if len(lsp) > 1:
            env[lsp[0]] = lsp[1]

    os.environ.update(env)
    pipe.stdout.close()


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
        try:
            pdir, profile = os.environ['IPPPDIR'], os.environ['IPPPROFILE']
            is_slurm = True
        except:
            logger.debug('stop_server: not a slurm cluster')
            is_slurm = False

        if is_slurm:
            if pdir is None and profile is None:
                pdir, profile = os.environ['IPPPDIR'], os.environ['IPPPROFILE']
            c = Client(ipython_dir=pdir, profile=profile)
            ee = c[:]
            ne = len(ee)
            logger.info(('Shutting down %d engines.' % (ne)))
            c.close()
            c.shutdown(hub=True)
            shutil.rmtree('profile_' + str(profile))
            try:
                shutil.rmtree('./log/')
            except:
                logger.info('creating log folder')     # FIXME Not what this means

            files = glob.glob('*.log')
            os.mkdir('./log')

            for fl in files:
                shutil.move(fl, './log/')

        else:
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


#%%


def setup_cluster(backend: str = 'multiprocessing',
                  n_processes: int = None,
                  single_thread: bool = False,
                  ignore_preexisting: bool = False) -> Tuple[Any, Any, Optional[int]]:
    """Setup and/or restart a parallel cluster.
    Args:
        backend: str
            'multiprocessing' [alias 'local'], 'ipyparallel', and 'SLURM'
            ipyparallel and SLURM backends try to restart if cluster running.
            backend='multiprocessing' raises an exception if a cluster is running.
        ignore_preexisting: bool
            If True, ignores the existence of an already running multiprocessing
            pool, which is usually indicative of a previously-started CaImAn cluster

    Returns:
        c: ipyparallel.Client object; only used for ipyparallel and SLURM backends, else None
        dview: ipyparallel dview object, or for multiprocessing: Pool object
        n_processes: number of workers in dview. None means guess at number of machine cores.
    """

    if n_processes is None:
        if backend == 'SLURM':
            n_processes = np.int(os.environ.get('SLURM_NPROCS'))
        else:
            # roughly number of cores on your machine minus 1
            n_processes = np.maximum(np.int(psutil.cpu_count()), 1)

    if single_thread:
        dview = None
        c = None
    else:
        sys.stdout.flush()

        if backend == 'SLURM':
            try:
                stop_server()
            except:
                logger.debug('Nothing to stop')
            slurm_script = '/mnt/home/agiovann/SOFTWARE/CaImAn/SLURM/slurmStart.sh' # FIXME: Make this a documented environment variable
            logger.info([str(n_processes), slurm_script])
            start_server(slurm_script=slurm_script, ncpus=n_processes)
            pdir, profile = os.environ['IPPPDIR'], os.environ['IPPPROFILE']
            logger.info([pdir, profile])
            c = Client(ipython_dir=pdir, profile=profile)
            dview = c[:]
        elif backend == 'ipyparallel':
            stop_server()
            start_server(ncpus=n_processes)
            c = Client()
            logger.info(f'Started ipyparallel cluster: Using {len(c)} processes')
            dview = c[:len(c)]

        elif (backend == 'multiprocessing') or (backend == 'local'):
            if len(multiprocessing.active_children()) > 0:
                if ignore_preexisting:
                    logger.warn('Found an existing multiprocessing pool. '
                                'This is often indicative of an already-running CaImAn cluster. '
                                'You have configured the cluster setup to not raise an exception.')
                else:
                    raise Exception(
                        'A cluster is already runnning. Terminate with dview.terminate() if you want to restart.')
            if (platform.system() == 'Darwin') and (sys.version_info > (3, 0)):
                try:
                    if 'kernel' in get_ipython().trait_names():        # type: ignore
                                                                       # If you're on OSX and you're running under Jupyter or Spyder,
                                                                       # which already run the code in a forkserver-friendly way, this
                                                                       # can eliminate some setup and make this a reasonable approach.
                                                                       # Otherwise, seting VECLIB_MAXIMUM_THREADS=1 or using a different
                                                                       # blas/lapack is the way to avoid the issues.
                                                                       # See https://github.com/flatironinstitute/CaImAn/issues/206 for more
                                                                       # info on why we're doing this (for now).
                        multiprocessing.set_start_method('forkserver', force=True)
                except:                                                # If we're not running under ipython, don't do anything.
                    pass
            c = None

            dview = Pool(n_processes)
        else:
            raise Exception('Unknown Backend')

    return c, dview, n_processes
