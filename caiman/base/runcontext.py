#!/usr/bin/env python

""" Runcontext class

The runcontext represents a run of Caiman and its associated setup,
computational resources, and (eventually) results. It is unrelated to
Python's contexts.
"""

import logging
import numpy as np
import os
import pathlib
import sys
from typing import Any, Optional, Union

import caiman

class RunContext():
    """
    Class representing a RunContext. This significantly replaces setup_cluster() in
    earlier versions of the codebase

    """

    def __init__(self,
                 temp_path:Optional[str],                  # Consulted by other functions to figure out where to store temporary files
                 output_path:Optional[str],                # Consulted by other functions to decide where to construct outputs of a Caiman run
                 name:Optional[str],                       # Entirely optional name to give a Caiman run
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
        self._name              = name
        self._parallel_engine   = parallel_engine
        self._pe_allow_reuse    = pe_allow_reuse
        self._pe_extra          = pe_extra
        self._temp_post_cleanup = temp_post_cleanup
        self._log_to_temp       = self.log_to_temp
        self._steps             = None # Fill in as a dict noting when we've done MotionCorrection, Fit, Refit, ...

        # TODO: Actually setup the chosen PE, saving the handle to the PE inside the object.
        # TODO: Handle if the paths don't exist

    def pe_reset(self) -> None:
        # Closes and reopens the chosen parallelisation engine (if applicable). 
        pass # TODO

    def pe_mode(self) -> str:
        # Returns the parallelisation engine that's active
        return self._parallel_engine

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


