#!/usr/bin/env python

""" Constrained Nonnegative Matrix Factorization

The general file class which is used to produce a factorization of the Y matrix being the video
it computes it using all the files inside of cnmf folder.
Its architecture is similar to the one of scikit-learn calling the function fit to run everything which is part
 of the structure of the class

 it is calling everyfunction from the cnmf folder
 you can find out more at how the functions are called and how they are laid out at the ipython notebook

See Also:
    http://www.cell.com/neuron/fulltext/S0896-6273(15)01084-3
"""

from copy import deepcopy
import cv2
import logging
import numpy as np
import os
import pathlib
import psutil
import pynwb
import scipy
import time
from typing import Optional

import caiman
from caiman.components_evaluation import estimate_components_quality
import caiman.mmapping
from caiman.motion_correction import MotionCorrect
import caiman.paths
from caiman.source_extraction.cnmf.estimates import Estimates
from caiman.source_extraction.cnmf.initialization import initialize_components, compute_W
from caiman.source_extraction.cnmf.map_reduce import run_CNMF_patches
from caiman.source_extraction.cnmf.merging import merge_components
from caiman.source_extraction.cnmf.params import CNMFParams
from caiman.source_extraction.cnmf.pre_processing import preprocess_data
from caiman.source_extraction.cnmf.spatial import update_spatial_components
from caiman.source_extraction.cnmf.temporal import update_temporal_components, constrained_foopsi_parallel
from caiman.source_extraction.cnmf.utilities import all_same, estimate_n_pixels_per_process
from caiman.utils.utils import save_dict_to_hdf5, load_dict_from_hdf5, hdf5_runmode


try:
    cv2.setNumThreads(0)
except:
    pass

class CNMF(object):
    """  Source extraction using constrained non-negative matrix factorization.

    The general class which is used to produce a factorization of the Y matrix being the video
    it computes it using all the files inside of cnmf folder.
    Its architecture is similar to the one of scikit-learn calling the function fit to run everything which is part
    of the structure of the class

    See Also:
    @url http://www.cell.com/neuron/fulltext/S0896-6273(15)01084-3
    .. image:: docs/img/quickintro.png
    """ 
    def __init__(self, n_processes, dview=None, Ain=None, Cin=None, b_in=None, f_in=None,
                 params: Optional[CNMFParams] = None, **param_kwargs):
        """
        Constructor of CNMF objects

        Below are arguments that are independent of the Params object, and which should be used with this
        constructor; passing other arguments is an older API and may be removed in some (likely distant) future update

        Args:
            n_processes: int
               number of processes used (if in parallel this controls memory usage)

            dview: Direct View object
                for parallelization purposes when using ipyparallel

            Ain: np.ndarray
                if known, it is the initial estimate of spatial filters.
                Array must be of type `bool` in 'F' order of shape: [n_pixels, n_components].
                Used to build estimates

            Cin - Used to build estimates

            b_in - Used to build estimates

            f_in - Used to build estimates

            params: CNMFParams
                Existing parameters to use (otherwise, they can be set afterwards with cnmf.params.change_params(...))
        """
        logger = logging.getLogger('caiman')

        self.runmode = "CNMF" # Single field to query to determine where an hdf5 file comes from
        self.dview = dview # longer-term this should be removed from the CNMF object and moved to a RunContext

        # these are movie properties that will be refactored into the Movie object
        self.dims = None
        self.empty_merged = None

        self.provenance = [] # This will provide a rough record of the history of the object, largely with the intent of it
                             # being useful in the serialized file form. The formatting for this will be a list of dicts,
                             # the dicts all having at least the fields "event":str, "time":int, and "text":str
                             # Other fields are permitted. Time is Unix epochtime. The semantics used here will be mirrored in
                             # the OnACID class
        self.provenance.append({'event': 'create', 'time': int(time.time()), 'description': 'CNMF Object created'})

        if params is None:
            self.params = CNMFParams(**param_kwargs)
        else:
            self.params = params
            params.set('patch', {'n_processes': n_processes}, warn=False)
            if param_kwargs:
                logger.warning(
                    'Ignoring extra parameters passed to CNMF constructor because a params object was passed. '
                    'If you want to update the params object, use params.change_params first.')

        self.estimates = Estimates(A=Ain, C=Cin, b=b_in, f=f_in,
                                   dims=self.params.data['dims'])

    @property
    def skip_refinement(self) -> bool:
        return self.params.patch.skip_refinement        

    @property
    def remove_very_bad_comps(self) -> bool:
        return self.params.patch.remove_very_bad_comps

    def __str__(self):
        ret = f"Caiman CNMF Object. subfields:{list(self.__dict__.keys()) }"
        if hasattr(self.estimates, 'A') and self.estimates.A is not None:
            ret += f" A.shape={self.estimates.A.shape}"
        if hasattr(self.estimates, 'b') and self.estimates.b is not None:
            ret += f" b.shape={self.estimates.b.shape}"
        if hasattr(self.estimates, 'C') and self.estimates.C is not None:
            ret += f" C.shape={self.estimates.C.shape}"
        return ret
    
    def __repr__(self):
        ret = f"Caiman CNMF Object"
        if hasattr(self.estimates, 'A') and self.estimates.A is not None:
            ret += f" A.shape={self.estimates.A.shape}"
        if hasattr(self.estimates, 'b') and self.estimates.b is not None and len(self.estimates.b.shape) > 1:
            ret += f" bg components={self.estimates.b.shape[1]}"
        if hasattr(self.estimates, 'C') and self.estimates.C is not None:
            ret += f" C.shape={self.estimates.C.shape}"
        ret += " Use str() for more details"
        return ret

    def __getitem__(self, idx):
        return getattr(self, idx)
    # We want subscripting to be read-only so we do not define a __setitem__ method

    def fit_file(self, motion_correct=False, indices=None) -> None:
        """
        Packages the analysis pipeline (motion correction, memory
        mapping, patch based CNMF processing and component evaluation) in a
        single method that can be called on a specific (sequence of) file(s).
        It is assumed that the CNMF object already contains a params object
        where the location of the files and all the relevant parameters have
        been specified. This method does not perform component evaluation.

        Args:
            motion_correct (bool)
                flag for performing motion correction
            indices (list of slice objects)
                perform analysis only on a part of the FOV
        Returns:
            cnmf object with the current estimates
        """

        logger = logging.getLogger("caiman")
        if indices is None:
            indices = (slice(None), slice(None))

        fnames = self.params.data.fnames
        if fnames is None:
            raise RuntimeError('No files were provided to fit (set on params.data.fnames)')

        if os.path.exists(fnames[0]):
            _, extension = os.path.splitext(fnames[0])[:2]
            extension = extension.lower()
        else:
            logger.error(f"Error: File not found, with file list:\n{fnames[0]}")
            raise Exception('File not found!')

        self.provenance.append({'event': 'fit_file', 'time': int(time.time()), 'description': f'Ran fit_file', 'data_target': str(fnames)})

        base_name = pathlib.Path(fnames[0]).stem + "_memmap_"
        if extension == '.mmap':
            fname_new = fnames[0]
            Yr, dims, T = caiman.mmapping.load_memmap(fnames[0])
            if np.isfortran(Yr):
                raise Exception('The file should be in C order (see save_memmap function)')
        else:
            data_set_name = self.params.data.var_name_hdf5
            if motion_correct:
                mc = MotionCorrect(fnames, dview=self.dview, **self.params.motion)
                mc.motion_correct(save_movie=True)
                fname_mc = mc.fname_tot_els if self.params.motion['pw_rigid'] else mc.fname_tot_rig
                if self.params.motion.pw_rigid:
                    b0 = np.ceil(np.maximum(np.max(np.abs(mc.x_shifts_els)),
                                            np.max(np.abs(mc.y_shifts_els)))).astype(int)
                    if self.params.motion.is3D:
                        self.estimates.shifts = [mc.x_shifts_els, mc.y_shifts_els, mc.z_shifts_els]
                    else:
                        self.estimates.shifts = [mc.x_shifts_els, mc.y_shifts_els]
                else:
                    b0 = np.ceil(np.max(np.abs(mc.shifts_rig))).astype(int)
                    self.estimates.shifts = mc.shifts_rig
                # TODO - b0 is currently direction inspecific, which can cause
                # sub-optimal behavior. See
                # https://github.com/flatironinstitute/CaImAn/pull/618#discussion_r313960370
                # for further details.
                # b0 = 0 if self.params.motion.border_nan == 'copy' else 0
                b0 = 0
                fname_new = caiman.mmapping.save_memmap(fname_mc, base_name=base_name, order='C',
                                                 var_name_hdf5=data_set_name, border_to_0=b0)
            else:
                fname_new = caiman.mmapping.save_memmap(fnames, base_name=base_name, var_name_hdf5=data_set_name, order='C')
            Yr, dims, T = caiman.mmapping.load_memmap(fname_new)

        images = np.reshape(Yr.T, [T] + list(dims), order='F')
        self.mmap_file = fname_new
        self.fit(images, indices=indices)

    def refit(self, images, dview=None):
        """
        Refit data using CNMF initialized from a previous iteration

        Args:
            images
            dview
        Returns:
            cnm
                A new CNMF object
        """
        cnm = CNMF(self.params.patch['n_processes'], params=self.params, dview=dview) # New object; this call does NOT modify self

        cnm.provenance += self.provenance
        cnm.provenance.append({'event': 'refit', 'time': int(time.time()), 'description': f'Ran refit, history imported from old object', 'data_target': str(images)})
        # We add the "history imported" note because the datestamp from the init of the new CNMF will be later than imported history (meaning right now)
        # so if you parse in list order you'll see a time-oddity here
        
        cnm.params.set('patch', {'rf': None, 'only_init': False}, warn=False)
        estimates = deepcopy(self.estimates)
        estimates.select_components(use_object=True)
        estimates.coordinates = None
        cnm.estimates = estimates
        cnm.mmap_file = self.mmap_file
        cnm.fit(images)
        return cnm

    def fit(self, images, indices=(slice(None), slice(None))) -> None:
        """
        This method uses the cnmf algorithm to find sources in data.
        After it finishes, the C, A, S, b, and f fields will be populated.

        Args:
            images : mapped np.ndarray of shape (t,x,y[,z]) containing the images that vary over time.

            indices: list of slice objects along dimensions (x,y[,z]) for processing only part of the FOV

        http://www.cell.com/neuron/fulltext/S0896-6273(15)01084-3

        """
        logger = logging.getLogger("caiman")
        self.provenance.append({'event': 'fit', 'time': int(time.time()), 'description': f'Ran fit'})
        if isinstance(indices, slice):
            indices = [indices]

        if isinstance(indices, tuple):
            indices = list(indices)

        indices = [slice(None)] + indices
        if len(indices) < len(images.shape):
            indices = indices + [slice(None)]*(len(images.shape) - len(indices))

        dims_orig = images.shape[1:]
        dims_sliced = images[tuple(indices)].shape[1:]
        is_sliced = (dims_orig != dims_sliced)
        if self.params.patch.rf is None and (is_sliced or 'ndarray' in str(type(images))):
            images = images[tuple(indices)]
            self.dview = None
            logger.info("Parallel processing in a single patch is not available for data that is in memory or sliced")

        T = images.shape[0]
        self.params.set('online', {'init_batch': T}, warn=False)
        self.dims = images.shape[1:]
        self.estimates.dims = self.dims
        Y = np.transpose(images, list(range(1, len(self.dims) + 1)) + [0])
        Yr = np.transpose(np.reshape(images, (T, -1), order='F'))
        if np.isfortran(Yr):
            raise Exception('The file is in F order, it should be in C order (see save_memmap function)')

        logger.info((T,) + self.dims)

        # Make sure filename is correctly set (numpy sets it to None sometimes)
        try:
            Y.filename = images.filename
            Yr.filename = images.filename
            self.mmap_file = images.filename
        except AttributeError:  # if no memmapping because we're working with small data
            pass

        logger.info(f"Using {self.params.patch.n_processes} processes")

        npx_per_proc = self.params.spatial.n_pixels_per_process
        if npx_per_proc is None:
            npx_per_proc = estimate_n_pixels_per_process(
                n_processes=self.params.patch.n_processes, T=T, dims=self.dims
            )

        logger.info(f'using {npx_per_proc} pixels per process')
        logger.info(f'using {self.params.temporal.block_size_temp} block_size_temp')

        if self.params.patch.rf is None:  # no patches
            logger.info('preprocessing ...')
            Yr = self.preprocess(Yr, n_pixels_per_process=npx_per_proc)
            if self.estimates.A is None:
                logger.info('initializing ...')
                self.initialize(Y)

            if self.params.patch.only_init:  # only return values after initialization
                if not (self.params.init.method_init == 'corr_pnr' and
                        self.params.init.ring_size_factor is not None):
                    self.compute_residuals(Yr)
                    self.estimates.bl = None
                    self.estimates.c1 = None
                    self.estimates.neurons_sn = None


                if self.remove_very_bad_comps:
                    logger.info('Removing bad components')
                    final_frate = 10
                    r_values_min = 0.5  # threshold on space consistency
                    fitness_min = -15  # threshold on time variability
                    fitness_delta_min = -15
                    Npeaks = 10
                    traces = np.array(self.C)
                    logger.info('Estimating component qualities...')
                    idx_components, idx_components_bad, fitness_raw,\
                        fitness_delta, r_values = estimate_components_quality(
                            traces, Y, self.estimates.A, self.estimates.C, self.estimates.b, self.estimates.f,
                            final_frate=final_frate, Npeaks=Npeaks, r_values_min=r_values_min,
                            fitness_min=fitness_min, fitness_delta_min=fitness_delta_min, return_all=True, N=5)

                    logger.info(f"Keeping {len(idx_components)} components and discarding {len(idx_components_bad)} components")
                    self.estimates.C = self.estimates.C[idx_components]
                    self.estimates.A = self.estimates.A[:, idx_components] # type: ignore # not provable that self.initialise provides a value
                    self.estimates.YrA = self.estimates.YrA[idx_components]

                self.estimates.normalize_components()
                return

            logger.info('update spatial ...')
            self.update_spatial(Yr, n_pixels_per_process=npx_per_proc)

            logger.info('update temporal ...')
            if self.skip_refinement:
                logger.info('deconvolution...')
                self.update_temporal(Yr)
            else:
                # first use do_deconv=False for fast updating without deconvolution
                self.update_temporal(Yr, do_deconv=False)

                logger.info('refinement...')
                if self.params.merging.do_merge:
                    logger.info('merging components ...')
                    self.merge_comps(Yr, mx=50, fast_merge=True)

                logger.info('Updating spatial ...')
                self.update_spatial(Yr, n_pixels_per_process=npx_per_proc)

                logger.info('update temporal ...')
                self.update_temporal(Yr)

            # embed in the whole FOV
            if is_sliced:
                FOV = np.zeros(dims_orig, order='C')
                FOV[indices[1:]] = 1
                FOV = FOV.flatten(order='F')
                ind_nz = np.where(FOV>0)[0].tolist()
                self.estimates.A = self.estimates.A.tocsc()
                A_data = self.estimates.A.data
                A_ind = np.array(ind_nz)[self.estimates.A.indices]
                A_ptr = self.estimates.A.indptr
                A_FOV = scipy.sparse.csc_matrix((A_data, A_ind, A_ptr),
                                                shape=(FOV.shape[0], self.estimates.A.shape[-1]))
                b_FOV = np.zeros((FOV.shape[0], self.estimates.b.shape[-1]))
                b_FOV[ind_nz] = self.estimates.b
                self.estimates.A = A_FOV
                self.estimates.b = b_FOV

        else:  # use patches
            if self.params.patch.stride is None:
                logger.info('Setting the stride to 10% of 2*rf automatically')  # see rf2stride argument to run_CNMF_patches

            if not isinstance(images, np.memmap):
                raise Exception(
                    'You need to provide a memory mapped file as input if you use patches!!')

            # We sometimes need to investigate what changes between before run_CNMF_patches and after that/before we update
            # components. This code block here is ready to uncomment for such debugging.
            #print("D: About to run run_CNMF_patches(), entering a shell. Will open a shell again afterwards")
            #import code
            #code.interact(local=dict(globals(), **locals()) )

            self.estimates.A, self.estimates.C, self.estimates.YrA, self.estimates.b, self.estimates.f, \
                self.estimates.sn, self.estimates.optional_outputs = run_CNMF_patches(
                    images.filename, self.dims + (T,), self.params,
                    dview=self.dview, memory_fact=self.params.patch.memory_fact,
                    gnb=self.params.init.nb, border_pix=self.params.patch.border_pix,
                    low_rank_background=self.params.patch.low_rank_background,
                    del_duplicates=self.params.patch.del_duplicates,
                    indices=indices, rf2stride=lambda rf: rf * 2 * .1)

            #print("D: Finished with run_CNMF_patches(), self.estimates.* are populated. Next step would be update_temporal() but first: Entering a shell.")
            #code.interact(local=dict(globals(), **locals()) )

            if (self.estimates.b is not None) and (len(list(np.where(~self.estimates.b.any(axis=0))[0])) > 0): # If any of the background ended up completely empty
                raise Exception("After run_CNMF_patches(), one or more of the background components is empty. Please restart analysis with init/nb set to a lower value")

            self.estimates.bl, self.estimates.c1, self.estimates.g, self.estimates.neurons_sn = None, None, None, None
            logger.info("Merging")
            self.estimates.merged_ROIs = [0]


            if self.params.init.center_psf:  # merge taking best neuron
                if self.params.patch.nb_patch > 0:

                    while len(self.estimates.merged_ROIs) > 0:
                        self.merge_comps(Yr, mx=np.inf, fast_merge=True)

                    logger.info("update temporal")
                    self.update_temporal(Yr)

                    logger.info('update spatial ...')
                    # skip binary closing on original-resolution data
                    self.update_spatial(Yr, n_pixels_per_process=npx_per_proc, skip_closing=True)

                    logger.info("update temporal")
                    self.update_temporal(Yr)
                else:
                    while len(self.estimates.merged_ROIs) > 0:
                        self.merge_comps(Yr, mx=np.inf, fast_merge=True)

                    if self.params.init.nb == 0:
                        self.estimates.W, self.estimates.b0 = compute_W(
                            Yr, self.estimates.A.toarray(), self.estimates.C, self.dims,
                            self.params.init.ring_size_factor *
                            self.params.init.gSiz[0],
                            ssub=self.params.init.ssub_B)

                    if len(self.estimates.C):
                        self.deconvolve()
                        self.estimates.C = self.estimates.C.astype(np.float32)
                    else:
                        self.estimates.S = self.estimates.C
            else:
                while len(self.estimates.merged_ROIs) > 0:
                    self.merge_comps(Yr, mx=np.inf)

                logger.info("Updating temporal components")
                self.update_temporal(Yr)

        self.estimates.normalize_components()

    def save(self, filename):
        '''save object in hdf5 file format

        Args:
            filename: str
                path to the hdf5 file containing the saved object
        '''

        if '.hdf5' in filename:
            filename = caiman.paths.fn_relocated(filename)
            save_dict_to_hdf5(self.__dict__, filename)
        else:
            raise Exception("File extension not supported for cnmf.save")

    def remove_components(self, ind_rm):
        """
        Remove a specified list of components from the CNMF object.

        Args:
            ind_rm :    list
                        indices of components to be removed
        """

        self.provenance.append({'event': 'remove_components', 'time': int(time.time()), 'description': f'Removed named components', 'indices_removed': str(ind_rm)})
        self.estimates.Ab, self.estimates.Ab_dense, self.estimates.CC, self.estimates.CY, self.M,\
            self.N, self.estimates.noisyC, self.estimates.OASISinstances, self.estimates.C_on,\
            expected_comps, self.ind_A,\
            self.estimates.groups, self.estimates.AtA = remove_components_online(
                ind_rm, self.params.init.nb, self.estimates.Ab,
                self.params.online.use_dense, self.estimates.Ab_dense,
                self.estimates.AtA, self.estimates.CY, self.estimates.CC, self.M, self.N,
                self.estimates.noisyC, self.estimates.OASISinstances, self.estimates.C_on,
                self.params.online.expected_comps)
        self.params.set('online', {'expected_comps': expected_comps}, warn=False)

    def compute_residuals(self, Yr) -> None:
        """
        Compute residual trace for each component (variable YrA).
        WARNING: At the moment this method is valid only for the 2p processing
        pipeline

         Args:
             Yr :    np.ndarray
                     movie in format pixels (d) x frames (T)
        """
        self.provenance.append({'event': 'compute_residuals', 'time': int(time.time()), 'description': f'Populated YrA with Computed/stored residuals'})

        block_size, num_blocks_per_run = self.params.temporal.block_size_temp, self.params.temporal.num_blocks_per_run_temp
        if 'csc_matrix' not in str(type(self.estimates.A)):
            self.estimates.A = scipy.sparse.csc_matrix(self.estimates.A)
        if 'array' not in str(type(self.estimates.b)):
            self.estimates.b = self.estimates.b.toarray()
        if 'array' not in str(type(self.estimates.C)):
            self.estimates.C = self.estimates.C.estimates.toarray()
        if 'array' not in str(type(self.estimates.f)):
            self.estimates.f = self.estimates.f.toarray()

        Ab = scipy.sparse.hstack((self.estimates.A, self.estimates.b)).tocsc()
        nA2 = np.ravel(Ab.power(2).sum(axis=0))
        nA2_inv_mat = scipy.sparse.spdiags(
            1. / (nA2 + np.finfo(np.float32).eps), 0, nA2.shape[0], nA2.shape[0])
        Cf = np.vstack((self.estimates.C, self.estimates.f))
        if 'numpy.ndarray' in str(type(Yr)):
            YA = (Ab.T.dot(Yr)).T * nA2_inv_mat
        else:
            YA = caiman.mmapping.parallel_dot_product(Yr, Ab, dview=self.dview, block_size=block_size,
                                           transpose=True, num_blocks_per_run=num_blocks_per_run) * nA2_inv_mat

        AA = Ab.T.dot(Ab) * nA2_inv_mat
        self.estimates.YrA = (YA - (AA.T.dot(Cf)).T)[:, :self.estimates.A.shape[-1]].T
        self.estimates.R = self.estimates.YrA

    def deconvolve(self, p=None, method_deconvolution=None, bas_nonneg=None,
                   noise_method=None, optimize_g=0, s_min=None, **kwargs) -> None:
        """Performs deconvolution on already extracted traces using
        constrained foopsi.
        """

        p = self.params.preprocess.p if p is None else p
        method_deconvolution = (self.params.temporal.method_deconvolution
                if method_deconvolution is None else method_deconvolution)
        bas_nonneg = (self.params.temporal.bas_nonneg
                      if bas_nonneg is None else bas_nonneg)
        noise_method = (self.params.temporal.noise_method
                        if noise_method is None else noise_method)
        s_min = self.params.temporal.s_min if s_min is None else s_min

        F = self.estimates.C + self.estimates.YrA
        args = dict()
        args['p'] = p
        args['method_deconvolution'] = method_deconvolution
        args['bas_nonneg'] = bas_nonneg
        args['noise_method'] = noise_method
        args['s_min'] = s_min
        args['optimize_g'] = optimize_g
        args['noise_range'] = self.params.temporal.noise_range
        args['fudge_factor'] = self.params.temporal.fudge_factor

        args_in = [(F[jj], None, jj, None, None, None, None,
                    args) for jj in range(F.shape[0])]

        self.provenance.append({'event': 'deconvolve', 'time': int(time.time()), 'description': f'Deconvolved on traces', 'method_deconvolution': str(method_deconvolution)})

        if 'multiprocessing' in str(type(self.dview)):
            results = self.dview.map_async(
                constrained_foopsi_parallel, args_in).get(4294967)
        elif self.dview is not None:
            results = self.dview.map_sync(constrained_foopsi_parallel, args_in)
        else:
            results = list(map(constrained_foopsi_parallel, args_in))

        results = list(zip(*results))

        order = list(results[7])
        self.estimates.C = np.stack([results[0][i] for i in order])
        self.estimates.S = np.stack([results[1][i] for i in order])
        self.estimates.bl = [results[3][i] for i in order]
        self.estimates.c1 = [results[4][i] for i in order]
        self.estimates.g = [results[6][i] for i in order]
        self.estimates.neurons_sn = [results[5][i] for i in order]
        self.estimates.lam = [results[8][i] for i in order]
        self.estimates.YrA = F - self.estimates.C

    def update_temporal(self, Y, use_init=None, do_deconv=True, **kwargs) -> None:
        """Updates temporal components

        Args:
            Y:  np.array (d1*d2) x T
                input data
        """
        logger = logging.getLogger('caiman')
        
        if use_init is not None:
            logger.warning('The use_init parameter is deprecated and has no effect.')

        if kwargs:
            self.params.change_params({'temporal': kwargs})

        # if not do_deconv, override p temporarily
        temporal_params = self.params.temporal
        if not do_deconv:
            temporal_params = {**temporal_params, 'p': 0}

        self.provenance.append({'event': 'update_temporal', 'time': int(time.time()), 'description': f'Updated temporal components based on provided Y'})

        self.estimates.C, self.estimates.A, self.estimates.b, self.estimates.f, self.estimates.S, \
        self.estimates.bl, self.estimates.c1, self.estimates.neurons_sn, \
        self.estimates.g, self.estimates.YrA, self.estimates.lam = update_temporal_components(
                Y, self.estimates.A, self.estimates.b, self.estimates.C, self.estimates.f, dview=self.dview,
                **temporal_params)
        self.estimates.R = self.estimates.YrA

    def update_spatial(self, Y, use_init=None, n_pixels_per_process: Optional[int] = None, skip_closing=False, **kwargs) -> None:
        """Updates spatial components
        modifies values self.estimates.A, self.estimates.b possibly self.estimates.C, self.estimates.f

        Args:
            Y:  np.array (d1*d2) x T
                input data
        """
        logger = logging.getLogger('caiman')
        
        if use_init is not None:
            logger.warning('The use_init parameter is deprecated and has no effect.')
            
        if kwargs:
            self.params.change_params({'spatial': kwargs})
            for key in kwargs:
                if hasattr(self, key):
                    setattr(self, key, kwargs[key])
        
        spatial_params = self.params.spatial
        if n_pixels_per_process is not None:
            spatial_params = {**spatial_params, 'n_pixels_per_process': n_pixels_per_process}

        if skip_closing:
            spatial_params = {**spatial_params, 'se': np.ones((1, 1), dtype=np.uint8)}

        self.provenance.append({'event': 'update_spatial', 'time': int(time.time()), 'description': f'Updated spatial components based on provided Y'})

        self.estimates.A, self.estimates.b, self.estimates.C, self.estimates.f =\
            update_spatial_components(Y, C=self.estimates.C, f=self.estimates.f, A_in=self.estimates.A,
                                      b_in=self.estimates.b, dview=self.dview,
                                      sn=self.estimates.sn, dims=self.dims, **spatial_params)

    def merge_comps(self, Y, mx=50, fast_merge=True) -> None:
        """merges components
        """
        self.provenance.append({'event': 'merge_comps', 'time': int(time.time()), 'description': f'Merged components based on provided Y'})

        self.estimates.A, self.estimates.C, self.estimates.nr, self.estimates.merged_ROIs, self.estimates.S, \
        self.estimates.bl, self.estimates.c1, self.estimates.neurons_sn, self.estimates.g, self.empty_merged, \
        self.estimates.YrA =\
            merge_components(Y, self.estimates.A, self.estimates.b, self.estimates.C, self.estimates.YrA,
                             self.estimates.f, self.estimates.S, self.estimates.sn, self.params.get_group('temporal'),
                             self.params.get_group('spatial'), dview=self.dview,
                             bl=self.estimates.bl, c1=self.estimates.c1, sn=self.estimates.neurons_sn,
                             g=self.estimates.g, thr=self.params.merging.merge_thr, mx=mx,
                             fast_merge=fast_merge, merge_parallel=self.params.merging.merge_parallel)

    def initialize(self, Y, **kwargs) -> None:
        """Component initialization
        """
        self.params.set('init', kwargs, warn=False)
        estim = self.estimates
        if (self.params.init.method_init == 'corr_pnr' and
                self.params.init.ring_size_factor is not None):
            estim.A, estim.C, estim.b, estim.f, estim.center, \
                extra_1p = initialize_components(
                    Y, sn=estim.sn, options_total=self.params,
                    **self.params.get_group('init'))
            try:
                estim.S, estim.bl, estim.c1, estim.neurons_sn, \
                    estim.g, estim.YrA, estim.lam = extra_1p
            except:
                estim.S, estim.bl, estim.c1, estim.neurons_sn, \
                    estim.g, estim.YrA, estim.lam, estim.W, estim.b0 = extra_1p
        else:
            estim.A, estim.C, estim.b, estim.f, estim.center =\
                initialize_components(Y, sn=estim.sn, options_total=self.params,
                                      **self.params.get_group('init'))

        self.estimates = estim

    def preprocess(self, Yr, n_pixels_per_process: Optional[int] = None):
        """
        Examines data to remove corrupted pixels and computes the noise level
        estimate for each pixel.

        Args:
            Yr: np.array (or memmap.array)
                2d array of data (pixels x timesteps) typically in memory
                mapped form
        """
        # TODO Weird that this returns Yr

        preproc_params = self.params.preprocess
        if n_pixels_per_process is not None:
            preproc_params = {**preproc_params, 'n_pixels_per_process': n_pixels_per_process}

        self.provenance.append({'event': 'preprocess', 'time': int(time.time()), 'description': f'Removed bad pixels and computed per-pixel noise based on provided Yr'})

        Yr, self.estimates.sn, self.estimates.g, self.estimates.psx = preprocess_data(
            Yr, dview=self.dview, **preproc_params)
        return Yr


def load_CNMF(filename:str, n_processes=1, dview=None):
    '''load object saved with the CNMF save method

    Args:
        filename:
            hdf5 (or nwb) file name containing the saved object
        dview: multiprocessing or ipyparallel object
            used to set up parallelization, default None
    '''

    logger = logging.getLogger("caiman")
    new_obj = CNMF(n_processes)
    file_extension = os.path.splitext(filename)[1].lower()

    if file_extension in ('.hdf5', '.h5'):
        filename = caiman.paths.fn_relocated(filename)
        runmode = hdf5_runmode(filename)
        if runmode != 'CNMF':
            logger.warning(f'Datafile {filename} not marked as cnmf')

        for key, val in load_dict_from_hdf5(filename).items():
            if key == 'params':
                prms = CNMFParams(**val)
                setattr(new_obj, key, prms)
            elif key == 'dview':
                setattr(new_obj, key, dview)
            elif key == 'estimates':
                estims = Estimates()
                for kk, vv in val.items():
                    if kk == 'discarded_components':
                        if vv is not None and not all_same(vv, b'NoneType'):
                            discarded_components = Estimates()
                            for kk__, vv__ in vv.items():
                                setattr(discarded_components, kk__, vv__)
                            setattr(estims, kk, discarded_components)
                    else:
                        setattr(estims, kk, vv)

                setattr(new_obj, key, estims)
            else:
                setattr(new_obj, key, val)
        if new_obj.estimates.dims is None or all_same(new_obj.estimates.dims, b'NoneType'):
            new_obj.estimates.dims = new_obj.dims
    elif file_extension == '.nwb':
        with pynwb.NWBHDF5IO(filename, 'r') as io:
            nwb = io.read()
            ophys = nwb.processing['ophys']
            rrs_group = ophys.data_interfaces['Fluorescence'].roi_response_series
            rrs = rrs_group['RoiResponseSeries']
            C = rrs.data[:].T
            rois = rrs.rois
            roi_indices = rois.data
            A = rois.table['image_mask'][roi_indices, ...]
            dims = A.shape[1:]
            A = A.reshape((A.shape[0], -1)).T
            A = scipy.sparse.csc_matrix(A)
            if 'Background_Fluorescence_Response' in rrs_group:
                brs = rrs_group['Background_Fluorescence_Response']
                f = brs.data[:].T
                brois = brs.rois
                broi_indices = brois.data
                b = brois.table['image_mask'][broi_indices, ...]
                b = b.reshape((b.shape[0], -1)).T
            else:
                b = None #np.zeros(mov.shape[1:])
                f = None

            estims = Estimates(A=A, b=b, C=C, f=f)
            estims.YrA = ophys.data_interfaces['residuals'].data[:].T

            frame_rate = ophys.data_interfaces['ImageSegmentation'].plane_segmentations['PlaneSegmentation']. \
                imaging_plane.imaging_rate

            if 'r' in rois.table:
                estims.r_values = rois.table['r'][roi_indices]
            if 'snr' in rois.table:
                estims.SNR_comp = rois.table['snr'][roi_indices]
            if 'cnn' in rois.table:
                estims.cnn_preds = rois.table['cnn'][roi_indices]
            if 'keep' in rois.table:
                keep = rois.table['keep'][roi_indices]
                estims.idx_components = np.where(keep)[0]
            if 'accepted' in rois.table:
                accepted = rois.table['accepted'][roi_indices]
                estims.accepted_list = np.where(accepted)[0]
            if 'rejected' in rois.table:
                rejected = rois.table['rejected'][roi_indices]
                estims.rejected_list = np.where(rejected)[0]                
                print(estims.rejected_list)
            estims.nr = len(roi_indices)

            if 'summary_images' in ophys.data_interfaces:
                if 'Cn' in ophys.data_interfaces['summary_images']:
                    estims.Cn = ophys.data_interfaces['summary_images']['Cn']
            if hasattr(nwb.acquisition['TwoPhotonSeries'], 'external_file'):
                setattr(new_obj, 'mmap_file', nwb.acquisition['TwoPhotonSeries'].external_file[0])
            else:
                setattr(new_obj, 'mmap_file', filename)

            estims.dims = dims
            prms = CNMFParams(dims=dims)
            prms.set('data', {'fr': frame_rate})

            setattr(new_obj, 'params', prms)
            setattr(new_obj, 'dview', dview)
            setattr(new_obj, 'estimates', estims)

    else:
        raise NotImplementedError(f'Unsupported file extension {file_extension}')

    return new_obj

