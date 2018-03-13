from __future__ import division
from __future__ import print_function
# example python script for loading neurofinder data
#
# for more info see:
#
# - http://neurofinder.codeneuro.org
# - https://github.com/codeneuro/neurofinder
#
# requires three python packages
#
# - numpy
# - scipy
# - matplotlib
#
#%%
from builtins import zip
from builtins import map
from builtins import str
from past.utils import old_div
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')
import matplotlib.pyplot as plt
import ca_source_extraction as cse
import calblitz as cb
from scipy.misc import imread
from glob import glob
import numpy as np
import matplotlib as mpl
mpl.use('TKAgg')
from matplotlib import pyplot as plt
import sys
import numpy as np
import ca_source_extraction as cse
from time import time
from scipy.sparse import coo_matrix
import tifffile
import subprocess
import time as tm
from time import time
import pylab as pl
import psutil
from glob import glob
import os
import scipy
from ipyparallel import Client

#%%
params = [
    #['/mnt/ceph/neuro/VariousLabeling/non_labelled_neurofinder/neurofinder.00.00.test/',7],
    #['/mnt/ceph/neuro/VariousLabeling/non_labelled_neurofinder/neurofinder.00.01.test/',7],
    #['/mnt/ceph/neuro/VariousLabeling/non_labelled_neurofinder/neurofinder.01.00.test/',7.5],
    #['/mnt/ceph/neuro/VariousLabeling/non_labelled_neurofinder/neurofinder.01.01.test/',7.5],
    #['/mnt/ceph/neuro/VariousLabeling/non_labelled_neurofinder/neurofinder.02.00.test/',8],
    #['/mnt/ceph/neuro/VariousLabeling/non_labelled_neurofinder/neurofinder.02.01.test/',8],
    #['/mnt/ceph/neuro/VariousLabeling/non_labelled_neurofinder/neurofinder.03.00.test/',7.5],
    #['/mnt/ceph/neuro/VariousLabeling/non_labelled_neurofinder/neurofinder.04.00.test/',6.75],
    #['/mnt/ceph/neuro/VariousLabeling/non_labelled_neurofinder/neurofinder.04.01.test/',3]
    ['/mnt/ceph/neuro/labeling/neurofinder.01.01/', 7.5],

]
#%%
# for ft in folders_in:
#    print ft
#    with open(os.path.join(ft,'info.json')) as ld:
#        a=json.load(ld)
#        print(a['rate-hz'])
f_rates = np.array([el[1] for el in params])
folders = np.array([el[0] for el in params])
#%%
backend = 'local'
if backend == 'SLURM':
    n_processes = np.int(os.environ.get('SLURM_NPROCS'))
else:
    # roughly number of cores on your machine minus 1
    n_processes = np.maximum(np.int(psutil.cpu_count()), 1)
print(('using ' + str(n_processes) + ' processes'))
single_thread = False

if single_thread:
    dview = None
else:
    try:
        c.close()
    except:
        print('C was not existing, creating one')
    print("Stopping  cluster to avoid unnencessary use of memory....")
    sys.stdout.flush()
    if backend == 'SLURM':
        try:
            cse.utilities.stop_server(is_slurm=True)
        except:
            print('Nothing to stop')
        slurm_script = '/mnt/xfs1/home/agiovann/SOFTWARE/Constrained_NMF/SLURM/slurmStart.sh'
        cse.utilities.start_server(slurm_script=slurm_script)
        pdir, profile = os.environ['IPPPDIR'], os.environ['IPPPROFILE']
        c = Client(ipython_dir=pdir, profile=profile)
    else:
        cse.utilities.stop_server()
        cse.utilities.start_server()
        c = Client()

    print(('Using ' + str(len(c)) + ' processes'))
    dview = c[:len(c)]
#%%
pars = []
for folder_in, f_rate in zip(folders, f_rates):
    print((folder_in, f_rate))
    pars.append([folder_in, f_rate])
#%% '/mnt/ceph/users/agiovann/ImagingData/LABELLING/NEUROFINDER/neurofinder.00.00.test/']

#%%
fls = c[:].map_sync(processor_placeholder, pars)
#%%
fls = list(map(processor_placeholder, pars))

#%%


def processor_placeholder(pars):
    import os
    import calblitz as cb
    from glob import glob
    folder_in, f_rate = pars
    fname_mov = os.path.join(os.path.split(folder_in)[
                             0], os.path.split(folder_in)[-1] + 'MOV.hdf5')
    print(fname_mov)
    files = sorted(
        glob(os.path.join(os.path.split(folder_in)[0], 'images/*.tif')))
    print(files)
    #% LOAD MOVIE HERE USE YOUR METHOD, Movie is frames x dim2 x dim2
    m = cb.load_movie_chain(files, fr=f_rate)
    m.file_name = [os.path.basename(ttt) for ttt in m.file_name]
    m.save(fname_mov)
    del m
    return fname_mov


#%%
import os
fls = []
for ffll in folders:
    print((os.path.join(os.path.dirname(ffll), 'MOV.hfd5')))
    fls.append(os.path.join(os.path.dirname(ffll), 'MOV.hdf5'))
#%%
res = list(map(create_images_for_labeling, fls))
#%%


def create_images_for_labeling(pars):
    import scipy.stats as st
    import os
    import numpy as np
    import calblitz as cb
    from glob import glob

    try:
        f_name = pars
        cdir = os.path.dirname(f_name)

        print('loading')
        m = cb.load(f_name)

        print('corr image')
        img = m.local_correlations(eight_neighbours=True)
        im = cb.movie(img, fr=1)
        im.save(os.path.join(cdir, 'correlation_image.tif'))

        print('std image')
        img = np.std(m, 0)
        im = cb.movie(np.array(img), fr=1)
        im.save(os.path.join(cdir, 'std_projection.tif'))

        m1 = m.resize(1, 1, old_div(1., m.fr))

        print('median image')
        img = np.median(m1, 0)
        im = cb.movie(np.array(img), fr=1)
        im.save(os.path.join(cdir, 'median_projection.tif'))

        print('save BL')
        m1 = m1 - img
        m1.save(os.path.join(cdir, 'MOV_BL.tif'))
        m1 = m1.bilateral_blur_2D()
        m1.save(os.path.join(cdir, 'MOV_BL_BIL.tif'))
        m = np.array(m1)

        print('max image')
        img = np.max(m, 0)
        im = cb.movie(np.array(img), fr=1)
        im.save(os.path.join(cdir, 'max_projection.tif'))

        print('skew image')
        img = st.skew(m, 0)
        im = cb.movie(img, fr=1)
        im.save(os.path.join(cdir, 'skew_projection.tif'))
        del m
        del m1
    except Exception as e:

        return e

    return f_name
