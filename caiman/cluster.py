#!/usr/bin/env python

"""
Functions related to the creation and management of the "cluster",
meaning the framework for distributed computation.

We put arrays on disk as raw bytes, extending along the first dimension.
Alongside each array x we ensure the value x.dtype which stores the data type.
"""

import ipyparallel
import logging
import multiprocessing
import multiprocessing.pool
import numpy as np
import os
import platform
import psutil
import shlex
import subprocess
import sys
import time
from typing import Any, Optional, Union


def start_server(ipcluster: str = "ipcluster", ncpus: Optional[int] = None) -> None:
    """
    programmatically start the ipyparallel server

    Args:
        ncpus
            number of processors

        ipcluster
            ipcluster binary file name; requires 4 path separators on Windows. ipcluster="C:\\\\Anaconda3\\\\Scripts\\\\ipcluster.exe"
            Default: "ipcluster"
    """
    logger = logging.getLogger("caiman")
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

def stop_server(ipcluster: str = 'ipcluster', pdir: Optional[str] = None, profile: Optional[str] = None, dview=None) -> None:
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
    logger = logging.getLogger("caiman")
    if isinstance(dview, multiprocessing.pool.Pool):
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
        assert proc.stderr is not None, 'Should have stderr when using PIPE'

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
                  maxtasksperchild:Optional[int] = None) -> tuple[Any, Any, Optional[int]]:
    """
    Setup and/or restart a parallel cluster.

    Args:
        backend:
            One of:
                'multiprocessing' - Use multiprocessing library
                'ipyparallel' - Use ipyparallel instead (better on Windows?)
                'single' - Don't be parallel (good for debugging, slow)

            Most backends will try, by default, to stop a running cluster if
            it is running before setting up a new one, or throw an error if
            they find one.
        n_processes:
            Sets number of processes to use. If None, is set automatically. 
        single_thread:
            Deprecated alias for the 'single' backend.
        ignore_preexisting:
            If True, ignores the existence of an already running multiprocessing
            pool (which usually indicates a previously-started CaImAn cluster)
        maxtasksperchild:
            Only used for multiprocessing, default None (number of tasks a worker process can 
            complete before it will exit and be replaced with a fresh worker process).
            
    Returns:
        c:
            ipyparallel.Client object; only used for ipyparallel backends, else None
        dview:
            multicore processing engine that is used for parallel processing. 
            If backend is 'multiprocessing' then dview is Pool object.
            If backend is 'ipyparallel' then dview is a DirectView object. 
        n_processes:
            number of workers in dview. None means single core mode in use. 
    """

    logger = logging.getLogger("caiman")
    sys.stdout.flush() # XXX Unsure why we do this
    if n_processes is None:
        n_processes = np.maximum(int(psutil.cpu_count() - 1), 1)

    if backend == 'multiprocessing' or backend == 'local':
        if backend == 'local':
            logger.warning('The local backend is an alias for the multiprocessing backend, and the alias may be removed in some future version of Caiman')
        if len(multiprocessing.active_children()) > 0:
            if ignore_preexisting:
                logger.warning('Found an existing multiprocessing pool. '
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
        dview = multiprocessing.Pool(n_processes, maxtasksperchild=maxtasksperchild)

    elif backend == 'ipyparallel':
        stop_server()
        start_server(ncpus=n_processes)
        c = ipyparallel.Client()
        logger.info(f'Started ipyparallel cluster: Using {len(c)} processes')
        dview = c[:len(c)]

    elif backend == "single" or single_thread:
        if single_thread:
            logger.warning('The single_thread flag to setup_cluster() is deprecated and may be removed in the future')
        dview = None
        c = None
        n_processes = 1

    else:
        raise Exception('Unknown Backend')

    return c, dview, n_processes
