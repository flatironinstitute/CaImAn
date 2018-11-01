import caiman as cm
import glob
import os
from caiman.cluster import setup_cluster
from caiman.source_extraction.cnmf.cnmf import load_CNMF
import numpy as np
import sys

base_folder = '/mnt/ceph/neuro/DataForPublications/DATA_PAPER_ELIFE/caiman_paper_test_neurofinder'
try:
    if __IPYTHON__:
        # this is used for debugging purposes only. allows to reload classes
        # when changed
        get_ipython().magic('load_ext autoreload')
        get_ipython().magic('autoreload 2')
except NameError:
    pass

try:
    if 'pydevconsole' in sys.argv[0]:
        raise Exception()
    ID = sys.argv[1]
    ID = str(np.int(ID) - 1)
    print('Processing ID:' + str(ID))
    ID = [np.int(ID)]
except:
    ID = [6]#np.arange(9)
    print('ID NOT PASSED')

preprocess = False
# %%
folders = list(glob.glob(os.path.join(base_folder, 'neurofinder*')))
#%%
if preprocess:
    #%%
    def preprocess_neurofinder(folder):
        import caiman as cm
        import os
        fls = glob.glob(os.path.join(folder, 'images/*.tiff'))
        fls.sort()
        print(fls[:5])
        m = cm.load_movie_chain(fls)
        m.save(os.path.join(folder, 'movie_total.hdf5'))

    #%%
    def motion_correct_file(folder, dview):
        import caiman as cm
        from caiman.motion_correction import MotionCorrect
        from caiman.source_extraction.cnmf import params as params
        print(folder)

        mc_dict = {
            'fnames': [os.path.join(folder, 'movie_total.hdf5')],
            'fr': 7,
            'decay_time': 1.5,
            'dxy': [1.15, 1.15],
            'pw_rigid': False,
            'max_shifts': [5,5],
            'strides': None,
            'overlaps': None,
            'max_deviation_rigid': None,
            'border_nan': 'copy'
        }

        opts = params.CNMFParams(params_dict=mc_dict)
        mc = MotionCorrect(mc_dict['fnames'], dview=dview, **opts.get_group('motion'))
        mc.motion_correct(save_movie=True)

    # %% start cluster
    try:
        cm.stop_server()
        dview.terminate()
    except:
        print('No clusters to stop')

    c, dview, n_processes = setup_cluster(
        backend='multiprocessing', n_processes=5, single_thread=False)

    #%% save from images
    res = dview.map(preprocess_neurofinder, folders)
    #%% motion correct
    for fold in folders:
        motion_correct_file(fold, dview)
    #%% check data
    folders = list(glob.glob(os.path.join(base_folder, 'neurofinder*')))
    for fold in folders:
        m_path = glob.glob(os.path.join(fold, '*.mmap'))[0]
        m = cm.load(m_path)
        m.resize(1,1,.5).play(fr=100,gain=3.)
    #%% save in C order
    folders = list(glob.glob(os.path.join(base_folder, 'neurofinder*')))
    allfiles = []
    for fold in folders:
        print(fold)
        m_path = glob.glob(os.path.join(fold, '*.mmap'))[0]
        cm.save_memmap([m_path], order='C')
    #%%
    def cnmf_neurofinder(params_dict):
        import numpy as np
        from caiman.source_extraction.cnmf import cnmf
        from caiman.source_extraction.cnmf import params

        fname_new = params_dict['fnames'][0]
        print(fname_new)
        Yr, dims, T = cm.load_memmap(fname_new)
        images = np.reshape(Yr.T, [T] + list(dims), order='F')
        opts = params.CNMFParams(params_dict=params_dict)
        dview = params_dict['dview']
        print('Starting CNMF')
        opts.set('temporal', {'p': 0})
        cnm = cnmf.CNMF(n_processes, params=opts, dview=dview)
        cnm = cnm.fit(images)
        cnm.params.change_params({'update_background_components': True,
                                  'skip_refinement': False,
                                  'n_pixels_per_process': 4000, 'dview': dview})
        opts.set('temporal', {'p': params_dict['p']})
        cnm2 = cnm.refit(images, dview=dview)
        cnm2.save(fname_new[:-5] + '_cnmf.hdf5')

        return cnm2

#%% select components neurofinder
def create_nf_dataset(cnm2):
    fname_new = cnm2.params.get('data', 'fnames')[0]
    print(fname_new)
    Yr, dims, T = cm.load_memmap(fname_new)
    images = np.reshape(Yr.T, [T] + list(dims), order='F')
    cnm2.params.set('quality', {'min_SNR': 2,
                                'rval_thr': 0.80,
                                'rval_lowest': 0,
                                'use_cnn': True,
                                'min_cnn_thr': 0.95,
                                'cnn_lowest': 0.1,
                                # 'thresh_fitness_delta': global_params['max_fitness_delta_accepted'],
                                'gSig_range': None})


    dview = cnm2.dview
    cnm2.estimates.evaluate_components(images, cnm2.params, dview=dview)
    cnm2.estimates.select_components(use_object=True)
    cnm2.estimates.threshold_spatial_components(maxthr=0.2, dview=dview)
    min_size_neuro = 3 * 2 * np.pi
    max_size_neuro = (2 * cnm2.params.init['gSig'][0]) ** 2 * np.pi
    cnm2.estimates.remove_small_large_neurons(min_size_neuro, max_size_neuro)
    _ = cnm2.estimates.remove_duplicates(r_values=None, dist_thr=0.1, min_dist=10, thresh_subset=0.6)
    dset = cnm2.estimates.masks_2_neurofinder('.'.join(fname_new.split('/')[-2].split('.')[1:]))
    return dset
#%%
import json
folders = list(glob.glob(os.path.join(base_folder, 'neurofinder*')))
allfiles = []
recompute_results = False
datasets = []
for fold in np.array(folders)[ID]:
    m_path = glob.glob(os.path.join(fold, '*.mmap'))[0]
    print(m_path)
    try:
        cm.stop_server()
        dview.terminate()
    except:
        print('No clusters to stop')
    c, dview, n_processes = setup_cluster(
        backend='multiprocessing', n_processes=20, single_thread=False)


    info_path = os.path.join(fold, 'info.json')
    with open(info_path) as js:
        info = json.load(js)
        merge_thresh = 0.8
        if '01.01' in m_path:
            merge_thresh = 0.9

        print([info['radius'],])
        params_dict = {'fnames': [m_path],
                       'fr': info['rate-hz'],
                       'decay_time': info['decay_time'],
                       'rf': info['radius']*4,
                       'stride': info['radius']*2,
                       'K': 7,
                       'gSig': [info['radius'], info['radius']],
                       'merge_thr': merge_thresh,
                       'p': 1,
                       'nb': 2,
                       'only_init_patch': True,
                       'dview': dview,
                       'method_deconvolution': 'oasis',
                       'border_pix': 3,
                       'low_rank_background': True,
                       'rolling_sum': True,
                       'nb_patch': 1,
                       'check_nan': False,
                       'block_size': 4000,
                       'num_blocks_per_run': 10,
                       'dview': dview
                       }
        if recompute_results:
            rests = cnmf_neurofinder(params_dict)
        else:
            rests = load_CNMF(m_path[:-6] + '__cnmf.hdf5')
            rests.params.data['fnames'] = [m_path]
            rests.dview = dview
            datasets.append(create_nf_dataset(rests))

#%%
if len(ID) > 1:
    with open('/mnt/home/agiovann/Dropbox/DATA_PAPER_ELIFE/caiman_paper_test_neurofinder/results.json', 'w') as f:
        f.write(json.dumps(datasets))



