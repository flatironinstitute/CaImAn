# -*- coding: utf-8 -*-
""" Constrained Nonnegative Matrix Factorization

The general file class which is used to produce a factorization of the Y matrix being the video
it computes it using all the files inside of cnmf folder.
Its architecture is similar to the one of scikit-learn calling the function fit to run everything which is part
 of the structure of the class

 it is calling everyfunction from the cnmf folder
 you can find out more at how the functions are called and how they are laid out at the ipython notebook

See Also:
------------

@url http://www.cell.com/neuron/fulltext/S0896-6273(15)01084-3
.. mage:: docs/img/quickintro.png
@author andrea giovannucci
"""
#\package Caiman/utils
#\version   1.0
#\copyright GNU General Public License v2.0
#\date Created on Fri Aug 26 15:44:32 2016

from __future__ import division
from __future__ import print_function

from builtins import str
from builtins import object
from past.utils import old_div
import numpy as np
from .utilities import  CNMFSetParms
from .pre_processing import preprocess_data
from .initialization import initialize_components
from .merging import merge_components
from .spatial import update_spatial_components
from .temporal import update_temporal_components
from .map_reduce import run_CNMF_patches
from caiman import components_evaluation
import scipy


class CNMF(object):
    """  Source extraction using constrained non-negative matrix factorization.


    The general class which is used to produce a factorization of the Y matrix being the video
    it computes it using all the files inside of cnmf folder.
    Its architecture is similar to the one of scikit-learn calling the function fit to run everything which is part
     of the structure of the class

    it is calling everyfunction from the cnmf folder
    you can find out more at how the functions are called and how they are laid out at the ipython notebook

    See Also:
    ------------
    @url http://www.cell.com/neuron/fulltext/S0896-6273(15)01084-3
    .. image:: docs/img/quickintro.png
    @author andrea giovannucci
    """

    def __init__(self, n_processes, k=5, gSig=[4,4], merge_thresh=0.8 , p=2, dview=None,
                 Ain=None, Cin=None, b_in = None, f_in=None,do_merge=True,
                 ssub=2, tsub=2,p_ssub=1, p_tsub=1, method_init= 'greedy_roi',alpha_snmf=None,
                 rf=None,stride=None, memory_fact=1, gnb = 1, only_init_patch=False,
                 method_deconvolution = 'oasis', n_pixels_per_process = 4000, block_size = 20000,
                 check_nan = True, skip_refinement = False, normalize_init=True, options_local_NMF = None,
                                        remove_very_bad_comps = False, border_pix = 0, low_rank_background = True, update_background_components = True):
        """ 
        Constructor of the CNMF method

        Parameters:
        -----------

        n_processes: int
           number of processed used (if in parallel this controls memory usage)

        k: int
           number of neurons expected per FOV (or per patch if patches_pars is  None)

        gSig: tuple
            expected half size of neurons

        merge_thresh: float
            merging threshold, max correlation allowed

        dview: Direct View object
            for parallelization pruposes when using ipyparallel

        p: int
            order of the autoregressive process used to estimate deconvolution

        Ain: ndarray
            if know, it is the initial estimate of spatial filters

        ssub: int
            downsampleing factor in space

        tsub: int 
             downsampling factor in time

        p_ssub: int
            downsampling factor in space for patches 

        method_init: str
           can be greedy_roi or sparse_nmf

        alpha_snmf: float
            weight of the sparsity regularization

        p_tsub: int 
             downsampling factor in time for patches     

        rf: int
            half-size of the patches in pixels. rf=25, patches are 50x50

        gnb: int
            number of global background components

        stride: int
            amount of overlap between the patches in pixels

        memory_fact: float
            unitless number accounting how much memory should be used. You will
             need to try different values to see which one would work the default is OK for a 16 GB system

        N_samples_fitness: int 
            number of samples over which exceptional events are computed (See utilities.evaluate_components)

        only_init_patch= boolean
            only run initialization on patches
        
        method_deconvolution = 'oasis' or 'cvxpy'
            method used for deconvolution. Suggested 'oasis' see  
            Friedrich J, Zhou P, Paninski L. Fast Online Deconvolution of Calcium Imaging Data.
            PLoS Comput Biol. 2017; 13(3):e1005423.
        
        n_pixels_per_process: int. 
            Number of pixels to be processed in parallel per core (no patch mode). Decrease if memory problems
        
        block_size: int. 
            Number of pixels to be used to perform residual computation in blocks. Decrease if memory problems
        
        check_nan: Boolean. 
            Check if file contains NaNs (costly for very large files so could be turned off)
        
        skip_refinement: 
            Bool. If true it only performs one iteration of update spatial update temporal instead of two
        
        normalize_init=Bool. 
            Differences in intensities on the FOV might caus troubles in the initialization when patches are not used,
             so each pixels can be normalized by its median intensity
        
        options_local_NMF: 
            experimental, not to be used

        remove_very_bad_comps:Bool
            whether to remove components with very low values of component quality directly on the patch.
             This might create some minor imprecisions.
            Howeverm benefits can be considerable if done because if many components (>2000) are created
            and joined together, operation that causes a bottleneck
                        
        border_pix:int    
            number of pixels to not consider in the borders
        
        low_rank_background:bool
            if True the background is approximated with gnb components. If false every patch keeps its background (overlaps are randomly assigned to one spatial component only)
             In the False case all the nonzero elements of the background components are updated using hals (to be used with one background per patch)
            
        update_background_components:bool
            whether to update the background components during the spatial phase           
        
        Returns:
        --------
        self
        """

        self.k = k #number of neurons expected per FOV (or per patch if patches_pars is not None
        self.gSig=gSig # expected half size of neurons
        self.merge_thresh=merge_thresh # merging threshold, max correlation allowed
        self.p=p #order of the autoregressive system
        self.dview=dview
        self.Ain=Ain
        self.Cin=Cin
        self.f_in=f_in

        self.ssub=ssub
        self.tsub=tsub
        self.p_ssub=p_ssub
        self.p_tsub=p_tsub
        self.method_init= method_init
        self.n_processes=n_processes
        self.rf=rf # half-size of the patches in pixels. rf=25, patches are 50x50
        self.stride=stride #amount of overlap between the patches in pixels   
        # unitless number accounting how much memory should be used.
        #  You will need to try different values to see which one would work the default is OK for a 16 GB system
        self.memory_fact = memory_fact
        self.gnb = gnb                        
        self.do_merge=do_merge
        self.alpha_snmf=alpha_snmf
        self.only_init=only_init_patch
        self.method_deconvolution=method_deconvolution
        self.n_pixels_per_process = n_pixels_per_process
        self.block_size = block_size
        self.check_nan = check_nan
        self.skip_refinement = skip_refinement
        self.normalize_init = normalize_init
        self.options_local_NMF = options_local_NMF
        self.b_in = b_in
        self.A=None
        self.C=None
        self.S=None
        self.b=None
        self.f=None
        self.sn = None
        self.g = None
        self.remove_very_bad_comps = remove_very_bad_comps
        self.border_pix = border_pix
        self.low_rank_background = low_rank_background 
        self.update_background_components = update_background_components 


    def fit(self, images):
        """
        This method uses the cnmf algorithm to find sources in data.

        it is calling everyfunction from the cnmf folder
        you can find out more at how the functions are called and how they are laid out at the ipython notebook

        Parameters:
        ----------
        images : mapped np.ndarray of shape (t,x,y[,z]) containing the images that vary over time.

        Returns:
        --------
        self: updated using the cnmf algorithm with C,A,S,b,f computed according to the given initial values

        Raise:
        ------
        raise Exception('You need to provide a memory mapped file as input if you use patches!!')

        See Also:
        --------

        ..image::docs/img/quickintro.png

        http://www.cell.com/neuron/fulltext/S0896-6273(15)01084-3

        """
        #Todo : to compartiment
        T = images.shape[0]
        dims = images.shape[1:]
        Y = np.transpose(images, list(range(1, len(dims) + 1)) + [0])
        Yr = np.transpose(np.reshape(images, (T, -1), order='F'))
        print((T,) + dims)

        # Make sure filename is pointed correctly (numpy sets it to None sometimes)
        Y.filename = images.filename
        Yr.filename = images.filename

        options = CNMFSetParms(Y, self.n_processes, p=self.p, gSig=self.gSig, K=self.k, ssub=self.ssub, tsub=self.tsub,
                               p_ssub=self.p_ssub, p_tsub=self.p_tsub, method_init=self.method_init,
                               n_pixels_per_process=self.n_pixels_per_process, block_size=self.block_size,
                               check_nan=self.check_nan, nb=self.gnb, normalize_init = self.normalize_init,
                               options_local_NMF = self.options_local_NMF,
                               remove_very_bad_comps = self.remove_very_bad_comps, low_rank_background = self.low_rank_background, 
                               update_background_components = self.update_background_components)

        self.options = options
        
        if self.rf is None:  # no patches
            print('preprocessing ...')
            Yr, sn, g, psx = preprocess_data(Yr, dview=self.dview, **options['preprocess_params'])
            
            if self.Ain is None:
                print('initializing ...')
                if self.alpha_snmf is not None:
                    options['init_params']['alpha_snmf'] = self.alpha_snmf

                self.Ain, self.Cin, self.b_in, self.f_in, center = initialize_components(
                    Y, **options['init_params'])

            if self.only_init: # only return values after initialization
                
                nA = np.squeeze(np.array(np.sum(np.square(self.Ain),axis=0)))
                nr=nA.size
                Cin=scipy.sparse.coo_matrix(self.Cin)
                YA = (self.Ain.T.dot(Yr).T)*scipy.sparse.spdiags(old_div(1.,nA),0,nr,nr)
                AA = ((self.Ain.T.dot(self.Ain))*scipy.sparse.spdiags(old_div(1.,nA),0,nr,nr))
                
                self.YrA = YA - Cin.T.dot(AA)      
                self.A = self.Ain 
                self.C = Cin.todense() 
                
                if self.remove_very_bad_comps:
                    print('removing bad components : ')
                    final_frate = 10
                    r_values_min = 0.5  # threshold on space consistency
                    fitness_min = -15  # threshold on time variability
                    fitness_delta_min = -15
                    Npeaks = 10
                    traces = np.array(self.C)
                    print('estimating the quality...')
                    idx_components, idx_components_bad, fitness_raw,\
                    fitness_delta, r_values = components_evaluation.estimate_components_quality(
                        traces, Y, self.A, np.array(self.C), self.b_in, self.f_in,
                        final_frate = final_frate, Npeaks=Npeaks, r_values_min=r_values_min,
                        fitness_min=fitness_min, fitness_delta_min=fitness_delta_min, return_all = True, N = 5)
    
                    print(('Keeping ' + str(len(idx_components)) +
                           ' and discarding  ' + str(len(idx_components_bad))))
                    self.C = self.C[idx_components]                    
                    self.A = self.A[:,idx_components]                                  
                    self.YrA = self.YrA[:,idx_components]
                
                self.sn = sn                    
                self.b = self.b_in
                self.f = self.f_in    
                self.g = g    
                self.bl = None
                self.c1 = None
                self.neurons_sn = None
                
                return self

            print('update spatial ...')
            A, b, Cin, self.f_in = update_spatial_components(Yr, C = self.Cin, f = self.f_in, b_in = self.b_in, A_in = self.Ain,
                                                             sn=sn, dview=self.dview, **options['spatial_params'])

            print('update temporal ...')
            if not self.skip_refinement:
                # set this to zero for fast updating without deconvolution
                options['temporal_params']['p'] = 0
            else:
                options['temporal_params']['p'] = self.p
            print('deconvolution ...')
            options['temporal_params']['method'] = self.method_deconvolution

            C, A, b, f, S, bl, c1, neurons_sn, g, YrA = update_temporal_components(
                Yr, A, b, Cin, self.f_in, dview=self.dview, **options['temporal_params'])

            if not self.skip_refinement:
                print('refinement...')
                if self.do_merge:
                    print('merge components ...')
                    A, C, nr, merged_ROIs, S, bl, c1, sn1, g1 = merge_components(
                        Yr, A, b, C, f, S, sn, options['temporal_params'], options['spatial_params'],
                        dview=self.dview, bl=bl, c1=c1, sn=neurons_sn, g=g, thr=self.merge_thresh,
                        mx=50, fast_merge=True)
                print((A.shape))
                print('update spatial ...')
                A, b, C, f = update_spatial_components(
                    Yr, C = C, f = f, A_in = A, sn=sn, b_in = b, dview=self.dview, **options['spatial_params'])
                # set it back to original value to perform full deconvolution
                options['temporal_params']['p'] = self.p
                print('update temporal ...')
                C, A, b, f, S, bl, c1, neurons_sn, g1, YrA = update_temporal_components(
                    Yr, A, b, C, f, dview=self.dview, bl=None, c1=None, sn=None, g=None, **options['temporal_params'])
            else:
                    g1 = g
                    # todo : ask for those..
                    C, f, S, bl, c1, neurons_sn, Yra = C, f, S, bl, c1, neurons_sn, YrA

        else:  # use patches
            if self.stride is None:
                self.stride = np.int(self.rf * 2 * .1)
                print(('**** Setting the stride to 10% of 2*rf automatically:' + str(self.stride)))

            if type(images) is np.ndarray:
                raise Exception(
                    'You need to provide a memory mapped file as input if you use patches!!')

            if self.only_init:
                options['patch_params']['only_init'] = True

            if self.alpha_snmf is not None:
                options['init_params']['alpha_snmf'] = self.alpha_snmf

            
            A, C, YrA, b, f, sn, optional_outputs = run_CNMF_patches(images.filename, dims + (T,),
                                                                     options, rf=self.rf, stride=self.stride,
                                                                     dview=self.dview, memory_fact=self.memory_fact,
                                                                     gnb=self.gnb, border_pix = self.border_pix, low_rank_background = self.low_rank_background)

            options = CNMFSetParms(Y, self.n_processes, p=self.p, gSig=self.gSig, K=A.shape[
                                   -1], thr=self.merge_thresh, n_pixels_per_process=self.n_pixels_per_process,
                                   block_size=self.block_size, check_nan=self.check_nan)

            options['temporal_params']['method'] = self.method_deconvolution

            print("merging")
            merged_ROIs = [0]
            while len(merged_ROIs) > 0:
                A, C, nr, merged_ROIs, S, bl, c1, sn_n, g = merge_components(Yr, A, [], np.array(C), [], np.array(
                    C), [], options['temporal_params'], options['spatial_params'], dview=self.dview,
                                                                             thr=self.merge_thresh, mx=np.Inf)
            
#            print('update spatial ...')
#            A, b, C, f = update_spatial_components(
#                    Yr, C = C, f = f, A_in = A, sn=sn, b_in = b, dview=self.dview, **options['spatial_params'])            

            print("update temporal")
            C, A, b, f, S, bl, c1, neurons_sn, g1, YrA = update_temporal_components(
                Yr, A, b, C, f, dview=self.dview, bl=None, c1=None, sn=None, g=None, **options['temporal_params'])
            
            

        self.A=A
        self.C=C
        self.b=b
        self.f=f
        self.S = S
        self.YrA=YrA
        self.sn=sn
        self.g = g1
        self.bl = bl
        self.c1 = c1
        self.neurons_sn = neurons_sn

        return self
