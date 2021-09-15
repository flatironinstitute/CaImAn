#!/usr/bin/env python
"""
Config.py file is used to load parameters for FIOLA and CaImAn objects
@author: @caichangjia
"""

def load_fiola_config(fnames, mode='voltage', mask=None):
    if mode == 'voltage':
        # setting params
        # dataset dependent parameters
        fr = 400                        # sample rate of the movie
        ROIs = mask                     # a 3D matrix contains all region of interests

        mode = 'voltage'                # 'voltage' or 'calcium 'fluorescence indicator
        init_method = 'binary_masks'    # initialization method 'caiman', 'weighted_masks' or 'binary_masks'. Needs to provide masks or using gui to draw masks if choosing 'masks'
        num_frames_init =  10000        # number of frames used for initialization
        num_frames_total =  20000       # estimated total number of frames for processing, this is used for generating matrix to store data
        offline_batch_size = 200        # number of frames for one batch to perform offline motion correction
        batch_size = 1                  # number of frames processing at the same time using gpu 
        flip = True                     # whether to flip signal to find spikes   
        ms = [10, 10]                   # maximum shift in x and y axis respectively. Will not perform motion correction if None.
        update_bg = True                # update background components for spatial footprints
        filt_window = 15                # window size for removing the subthreshold activities 
        minimal_thresh = 3              # minimal of the threshold 
        template_window = 2             # half window size of the template; will not perform template matching if window size equals 0

        options = {
            'fnames': fnames,
            'fr': fr,
            'ROIs': ROIs,
            'mode': mode,
            'init_method':init_method,
            'num_frames_init': num_frames_init, 
            'num_frames_total':num_frames_total,
            'offline_batch_size': offline_batch_size,
            'batch_size':batch_size,
            'flip': flip,
            'ms': ms,
            'update_bg': update_bg,
            'filt_window': filt_window,
            'minimal_thresh': minimal_thresh,
            'template_window':template_window}
        
    elif mode == 'calcium':
        fr = 30                         # sample rate of the movie
        ROIs = mask                     # a 3D matrix contains all region of interests

        mode = 'calcium'                # 'voltage' or 'calcium 'fluorescence indicator
        init_method = 'caiman'          # initialization method 'caiman', 'weighted_masks' or 'binary_masks'. Needs to provide masks or using gui to draw masks if choosing 'masks'
        num_frames_init =  1500         # number of frames used for initialization
        num_frames_total =  3000        # estimated total number of frames for processing, this is used for generating matrix to store data
        offline_batch_size = 5          # number of frames for one batch to perform offline motion correction
        batch_size= 1                   # number of frames processing at the same time using gpu 
        flip = False                    # whether to flip signal to find spikes   
        ms = [5, 5]                     # maximum shift in x and y axis respectively. Will not perform motion correction if None.
        center_dims = None              # template dimensions for motion correction. If None, the input will the the shape of the FOV
        hals_movie = 'hp_thresh'        # apply hals on the movie high-pass filtered and thresholded with 0 (hp_thresh); movie only high-pass filtered (hp); 
                                        # original movie (orig); no HALS needed if the input is from CaImAn (when init_method is 'caiman' or 'weighted_masks')
        options = {
            'fnames': fnames,
            'fr': fr,
            'ROIs': ROIs,
            'mode': mode, 
            'num_frames_init': num_frames_init, 
            'num_frames_total':num_frames_total,
            'init_method':init_method,
            'offline_batch_size': offline_batch_size,
            'batch_size':batch_size,
            'flip': flip,
            'ms': ms,
            'hals_movie': hals_movie,
            'center_dims':center_dims}
    
    else:
        raise ValueError('mode must be "calcium" or "voltage"')
    
    return options

def load_caiman_config(fnames):
    # params for caiman init
    fr = 30             # imaging rate in frames per second
    decay_time = 0.4    # length of a typical transient in seconds
    dxy = (2., 2.)      # spatial resolution in x and y in (um per pixel)
    # note the lower than usual spatial resolution here
    max_shift_um = (12., 12.)       # maximum shift in um
    patch_motion_um = (100., 100.)  # patch size for non-rigid correction in um
    
    # motion correction parameters
    pw_rigid = False       # flag to select rigid vs pw_rigid motion correction
    # maximum allowed rigid shift in pixels
    max_shifts = [int(a/b) for a, b in zip(max_shift_um, dxy)]
    # start a new patch for pw-rigid motion correction every x pixels
    strides = tuple([int(a/b) for a, b in zip(patch_motion_um, dxy)])
    # overlap between pathes (size of patch in pixels: strides+overlaps)
    overlaps = (24, 24)
    # maximum deviation allowed for patch with respect to rigid shifts
    max_deviation_rigid = 3
    
    mc_dict = {
        'fnames': fnames,
        'fr': fr,
        'decay_time': decay_time,
        'dxy': dxy,
        'pw_rigid': pw_rigid,
        'max_shifts': max_shifts,
        'strides': strides,
        'overlaps': overlaps,
        'max_deviation_rigid': max_deviation_rigid,
        'border_nan': 'copy'
    }
    
    p = 1                    # order of the autoregressive system
    gnb = 2                  # number of global background components
    merge_thr = 0.85         # merging threshold, max correlation allowed
    rf = 15
    # half-size of the patches in pixels. e.g., if rf=25, patches are 50x50
    stride_cnmf = 6          # amount of overlap between the patches in pixels
    K = 4                    # number of components per patch
    gSig = [6, 6]            # expected half size of neurons in pixels
    # initialization method (if analyzing dendritic data using 'sparse_nmf')
    method_init = 'greedy_roi'
    ssub = 2                     # spatial subsampling during initialization
    tsub = 2                     # temporal subsampling during intialization
    n_processes = None
    
    # parameters for component evaluation
    opts_dict = {'fnames': fnames,
                 'p': p,
                 'fr': fr,
                 'nb': gnb,
                 'rf': rf,
                 'K': K,
                 'gSig': gSig,
                 'stride': stride_cnmf,
                 'method_init': method_init,
                 'rolling_sum': True,
                 'merge_thr': merge_thr,
                 'n_processes': n_processes,
                 'only_init': True,
                 'ssub': ssub,
                 'tsub': tsub}
    
    # the components are evaluated in three ways:
    #   a) the shape of each component must be correlated with the data
    #   b) a minimum peak SNR is required over the length of a transient
    #   c) each shape passes a CNN based classifier
    min_SNR = 5  # signal to noise ratio for accepting a component
    rval_thr = 0.85  # space correlation threshold for accepting a component
    cnn_thr = 0.99  # threshold for CNN based classifier
    cnn_lowest = 0.1 # neurons with cnn probability lower than this value are rejected
    
    quality_dict = {'decay_time': decay_time,
                    'min_SNR': min_SNR,
                    'rval_thr': rval_thr,
                    'use_cnn': True,
                    'min_cnn_thr': cnn_thr,
                    'cnn_lowest': cnn_lowest}
    
    return mc_dict, opts_dict, quality_dict
    
    