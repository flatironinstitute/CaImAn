try:
    from pathlib2 import Path  # py2.7 version
except ModuleNotFoundError:
    from pathlib import Path  # py3 version

from demos_detailed.demo_pipeline import main_pipeline

foldername = Path(r'.')  # Enter path to folder containing all files
all_files = [str(filename) for filename in foldername.rglob('*.tif')]  # Filter out unwanted files in the .rglob parameter
print("The following files were found: \n{}".format(all_files))

params_movie = {'fname': all_files,  # Don't edit
                'niter_rig': 1,
                'max_shifts': (3, 3),  # maximum allow rigid shift
                'splits_rig': 20,  # for parallelization split the movies in  num_splits chuncks across time
                # if none all the splits are processed and the movie is saved
                'num_splits_to_process_rig': None,
                # intervals at which patches are laid out for motion correction
                'strides': (48, 48),
                # overlap between pathes (size of patch strides+overlaps)
                'overlaps': (24, 24),
                'splits_els': 28,  # for parallelization split the movies in  num_splits chuncks across time
                # if none all the splits are processed and the movie is saved
                'num_splits_to_process_els': [14, None],
                'upsample_factor_grid': 4,  # upsample factor to avoid smearing when merging patches
                # maximum deviation allowed for patch with respect to rigid
                # shift
                'max_deviation_rigid': 2,
                'p': 1,  # order of the autoregressive system
                'merge_thresh': 0.8,  # merging threshold, max correlation allowed
                'rf': 15,  # half-size of the patches in pixels. rf=25, patches are 50x50
                'stride_cnmf': 6,  # amounpl.it of overlap between the patches in pixels
                'K': 4,  # number of components per patch
                # if dendritic. In this case you need to set init_method to
                # sparse_nmf
                'is_dendrites': False,
                'init_method': 'greedy_roi',
                'gSig': [4, 4],  # expected half size of neurons
                'alpha_snmf': None,  # this controls sparsity
                'final_frate': 30,
                'r_values_min_patch': .7,  # threshold on space consistency
                'fitness_min_patch': -20,  # threshold on time variability
                # threshold on time variability (if nonsparse activity)
                'fitness_delta_min_patch': -20,
                'Npeaks': 10,
                'r_values_min_full': .8,
                'fitness_min_full': - 40,
                'fitness_delta_min_full': - 40,
                'only_init_patch': True,
                'gnb': 2,
                'memory_fact': 1,
                'n_chunks': 10,
                'update_background_components': True,
                # whether to update the background components in the spatial phase
                'low_rank_background': True,
                # whether to update the using a low rank approximation. In the False case all the nonzero elements of the background components are updated using hals
                # (to be used with one background per patch)
                'num_of_channels': 1,  # If stack is interleaved with more than one data channel ---
                'channel_of_neurons': 1,  # --- Specify the relevant channel for CaImAn to process
                'var_name_hdf5': 'mov',
                }

params_display = {
    'downsample_ratio': .2,
    'thr_plot': 0.8,
    'play_movie': False
}

for file in all_files:
    main_pipeline(params_movie, params_display)
