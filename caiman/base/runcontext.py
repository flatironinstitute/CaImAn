#!/usr/bin/env python

""" Runcontext class

The runcontext represents a run of Caiman and its associated setup,
computational resources, and (eventually) results. It is unrelated to
Python's contexts.
"""

import glob
import ipyparallel
import logging
import multiprocessing
import os
import platform
import psutil
import shlex
import shutil
import subprocess
import sys
import time
from typing import Any, Optional, Union

try: # Sigh
    import IPython
except:
    pass

import caiman

class RunContext():
    """
    Class representing a RunContext.
    This encapsulates:
        1) If and how parallelism should happen for this run
        2) The filesystem location of temporary and output files (and policy related to that)

    If the caller doesn't provide overrides, this will create a unique directory for a given run to
    happen, so that different runs don't step on each others feet or overwrite each others data.
    """

    def __init__(self,
                 temp_path:Optional[str],                  # Consulted by other functions to figure out where to store temporary files
                 output_path:Optional[str],                # Consulted by other functions to decide where to construct outputs of a Caiman run
                 name:Optional[str],                       # Entirely optional name to give a Caiman run.
                 parallel_engine:str = 'multiprocessing',  # What kind of parallelisation engine we want to use for the run.
                                                           #   One of:
                                                           #     'multiprocessing' - Use multiprocessing library
                                                           #     'ipyparallel' - Use ipyparallel instead (better on Windows?)
                                                           #     'single' - Don't be parallel (good for debugging, slow)
                                                           #     'SLURM' - Try to use SLURM batch system (untested, involved).
                 pe_allow_reuse:bool = False,              # Whether we should attempt to stop any existing Caiman parallel engine before starting one
                 pe_extra:Optional[Dict] = None,           # Any extra engine-specific options
                 temp_post_cleanup:bool = False,           # Whether to cleanup temporary files when they're no longer needed
                 log_to_temp:Optional[str] = None          # If not none, enables logging to files, and sets a log level for that logging
                ):
        self._temp_path         = temp_path
        self._output_path       = output_path
        self._temp_post_cleanup = temp_post_cleanup
        self._log_to_temp       = self.log_to_temp
        self._name              = name
        # Parallelism engine stuff
        self._parallel_engine   = parallel_engine
        self._pe_allow_reuse    = pe_allow_reuse
        self._pe_extra          = pe_extra
        self._pe_state          = {running: False}                       # This stores the internal state of the parallelism engine, e.g. dview

        if 'ipcluster_binary' not in self._pe_extra:
            self._pe_extra['ipcluster_binary'] = 'ipcluster' # ipyparallel binary name, used by the ipyparallel backend, ignored by other backends.
                                                             #     If you're on windows, requires the path to be escaped as so:
                                                             #     "C:\\\\Anaconda3\\\\Scripts\\\\ipcluster.exe" (this path is not likely valid)
        # TODO: If the paths do not exist, make a subdirectory of caiman_data()/auto/ with either epochtime or UUID as a basis so they end up distinct,
        #       then subdirectories of that for logs and results
	# TODO: Ensure that temp_path and output_path exist, creating any needed structire in either

    def parallel_start(self) -> None:
        # This brings up the parallelism engine (whatever is selected), or connects to it if it's not the sort to be brought up or down
        logger = logging.getLogger()
        if 'n_processes' not in self._pe_extra or self._pe_extra['n_processes'] is None:
            self._pe_extra['n_processes'] = max( int(psutil.cpu_count() - 1), 1) # Take all CPUs but 1

        if 'maxtasksperchild' not in self._pe_extra:
            self._pe_extra['maxtasksperchild'] = None

        if self._parallel_engine == 'multiprocessing':
            # ----- Multiprocessing Backend -----
            if len(multiprocessing.active_children()) > 0:
                if self._pe_allow_reuse:
                    logger.warn('An already active multiprocessing pool was found, but reuse is allowed so we will let it be')
                else:
                    raise Exception('An already active multiprocessing pool was found, and you asked to set up a new one. Stop the existing one first or allow reuse')
            if platform.system() == 'Darwin':
                try:
                    if 'kernel' in IPython.get_ipython().trait_names():        # type: ignore
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
            self._pe_state['dview'] = multiprocessing.Pool(self._pe_extra['n_processes'], self._pe_extra['maxtasksperchild'])
            self._pe_state['running'] = True
            logger.info(f"Started multiprocessing parallelism")
        elif self._parallel_engine == 'ipyparallel':
            # ----- ipyparallel Backend -----
            # It looks like ipyparallel cannot have in the same process space, access to distinct clients because it uses global methods?
            # Or are there other methods that allow that that our initial implementation didn't use?
            self.parallel_stop() # FIXME better to use the non-duplication logic that multiprocessing has

            # This part of the code starts the cluster
            subprocess.Popen(shlex.split(f"{self._pe_extra['ipcluster_binary']} start -n {self._pe_extra['n_processes']}"), shell=True, close_fds=(os.name != 'nt'))
            time.sleep(1.5) # XXX
            client = ipyparallel.Client()
            time.sleep(1.5) # XXX
            while(len(client) < ncpus): # Wait for the worker processes to start
                sys.stdout.write('.') # XXX Reconsider this output format
                sys.stdout.flush()
                time.sleep(0.5)
            logger.debug('ipyparallel backend reports ready, testing')
            client.direct_view().execute('__a=1', block=True)      # when done on all, we're set to go
            logger.debug('ipyparallel backend up')
            # Now that it's up, handle the paperwork
            self._pe_state['ipyparallel_c'] = client
            self._pe_state['dview'] = self._pe_state['ipyparallel_c'][:len(self._pe_state['ipyparallel_c'])]
            self._pe_state['running'] = True
            logger.info(f"Started ipyparallel parallelism, using {len(self._pe_state['ipyparallel_c'])} processes")
        elif self._parallel_engine == 'single':
            # ----- single Backend -----
            # By design, there is no dview for this; calling code should check to see what kind of parallelism is being used instead
            self._pe_state['running'] = True
            logger.info(f"Started single parallelism (no-op)")
        else:
            raise Exception("Unknown parallelism backend") # Ideally we'll catch this in object initialisation though!

    def parallel_stop(self) -> None:
        # This brings down the parallelism engine (whatever is selected), or disconnects from it if it's not the sort to be brought up or down
        logger = logging.getLogger()
        self._pe_state['running'] = False

        if self._parallel_engine == 'multiprocessing':
            self._pe_state['dview'].terminate()
            del self._pe_state['dview'] # Make sure the old handle won't be reused

        elif self._parallel_engine == 'ipyparallel':
            logger.info("Stopping ipyparallel parallelism")
            stophandle = subprocess.Popen(shlex.split(f"{self._pe_extra['ipcluster_binary']} stop"), shell=True, stderr=subprocess.PIPE, close_fds=(os.name != 'nt'))
            stop_command_output = stophandle.stderr.readline()
            if 'CRITICAL' in stop_command_output: # FIXME This is not specific enough
                logger.info("No ipyparallel backend found to stop")
            elif 'Stopping' in stop_command_output:
                st = time.time()
                logger.debug('Waiting for ipyparallel backend to stop...')
                while (time.time() - st) < 4: # Is there no better way to poll?
                    sys.stdout.write('.')
                    sys.stdout.flush()
                    time.sleep(1)
            else:
                logger.error(stop_command_output)
                logger.error("*** Failed to parse output of ipyparallel stop command in stopping backend, please report this as a bug ***")
            stophandle.stderr.close()
            del self._pe_state['dview'] # Make sure the old handle won't be reused
            del self._pe_state['ipyparallel_c']
            logger.info("ipyparallel parallelism stopped")

        elif self._parallel_engine == 'single':
            print("The single backend is stopped")

        else:
            raise Exception("Unknown parallelism backend") # Ideally we'll catch this in object initialisation though!

    def parallel_restart(self) -> None:
        # Efficiently restarts the chosen parallelisation engine (if applicable). Often used to save memory
        # Initial implementation just does the simple thing, but later on we may adjust this to be more efficient in a backend-specific way
        self.parallel_stop()
        self.parallel_start()

    def parallel_mode(self) -> str:
        # Returns the parallelisation engine that's active
        return self._parallel_engine

    def parallel_dview(self):
        # Returns the dview associated with the current context (meaning and type of this may differ depending on what engine is being used)
        # We should prefer to eventually rewrite code not to use this, but it will initially be a compatibility measure as we convert code over
        # to be RunContext aware (when it is, it will take a RunContext rather than a dview as an argument)
        logger = logging.getLogger()

        if 'dview' in self._pe_state:
            return self._pe_state['dview']
        else:
            logger.warn("Was asked for the dview of a RunContext without one, returning None")
            return None

    def tempdir_purge(self) -> None:
        # Remove all files from the temporary path
        pass # TODO

    def tempdir_size(self) -> int:
        # Return size in megabytes of the temporary path
        pass #TODO

    def tempdir_path(self) -> str:
        return self._temp_path

    def outputdir_path(self) -> str:
        return self._output_path


