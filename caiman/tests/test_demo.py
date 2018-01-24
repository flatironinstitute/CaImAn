#!/usr/bin/env python
#%%
import numpy.testing as npt
import numpy as np
import os
import caiman as cm
from caiman.source_extraction import cnmf

def demo(parallel=False):

    p = 2  # order of the AR model (in general 1 or 2)
    if parallel:
        c, dview, n_processes = cm.cluster.setup_cluster(
            backend='local', n_processes=None, single_thread=False)
    else:
        n_processes, dview = 2, None

    # LOAD MOVIE AND MEMORYMAP
    fname_new = cm.save_memmap([os.path.abspath(cm.__path__[0][:-7]) +
                                '/example_movies/demoMovie.tif'], base_name='Yr',
                                order = 'C')
    Yr, dims, T = cm.load_memmap(fname_new)
    # INIT
    cnm = cnmf.CNMF(n_processes, method_init='greedy_roi', k=30, gSig=[4, 4], merge_thresh=.8,
                    p=p, dview=dview, Ain=None, method_deconvolution='oasis')
    # FIT
    images = np.reshape(Yr.T, [T] + list(dims), order='F')
    cnm = cnm.fit(images)
    if parallel:
        cm.cluster.stop_server(dview=dview)

    # verifying the spatial components
    npt.assert_allclose(cnm.A.sum(), 281.1, 1e-2)
    # verifying the temporal components
    npt.assert_allclose(cnm.C.sum(), 66271668, 1e-2)
    try:
        dview.terminate()
    except:
        pass

def test_single_thread():
    demo()


def test_parallel():
    demo(True)
