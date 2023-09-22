#!/usr/bin/env python

###################
# build helper for pyproject.toml to do the
# cython parts of the build.
#
# This was ported from logic originally in the
# old-stype setup.py, in order to comply with the
# new PEP-517/PEP-518 build style.

from Cython.Build import cythonize
import numpy as np
import os
import sys
from setuptools import setup, find_packages
from setuptools.command.build_py import build_py as _build_py
import setuptools.extension

class build_py(_build_py):
    def run(self):
        self.run_command("build_ext")
        return super().run()

    def initialize_options(self):
        super().initialize_options()
        if sys.platform == 'darwin':
            # TODO: Verify still needed; requirement was
            #       added w/ OSX 10.9
            # See also:
            # https://github.com/pandas-dev/pandas/issues/23424
            extra_compiler_args = ['-stdlib=libc++']
        else:
            extra_compiler_args = []

        if self.distribution.ext_modules is None:
            self.distribution.ext_modules = []

        self.distribution.ext_modules.append(
            setuptools.extension.Extension(
                "caiman.source_extraction.cnmf.oasis",
                sources=["caiman/source_extraction/cnmf/oasis.pyx"],
                include_dirs=[np.get_include()],
                language="c++",
                extra_compile_args = extra_compiler_args,
                extra_link_args = extra_compiler_args,
            )
        )
