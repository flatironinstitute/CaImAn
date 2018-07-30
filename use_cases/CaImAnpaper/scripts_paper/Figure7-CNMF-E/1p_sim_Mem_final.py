import caiman as cm
from caiman.source_extraction import cnmf
import pickle
from memory_profiler import memory_usage
from multiprocessing import Pool, cpu_count
import numpy as np
from time import time, sleep
import matplotlib.pyplot as plt
from scipy.io import loadmat
import os
# set MKL_NUM_THREADS and OPENBLAS_NUM_THREADS to 1 outside via export!
# takes about 170 mins for all runs

fname = 'test_sim.mat'
dims = (253, 316)
Yr = loadmat(fname)['Y']
Y = Yr.T.reshape((-1,) + dims, order='F')
cm.save_memmap([Y], base_name='Yr', order='C')


def main(n_processes=None, patches=True, rf=64):

    t = -time()

    Yr, dims, T = cm.load_memmap(os.path.abspath('./Yr_d1_253_d2_316_d3_1_order_C_frames_2000_.mmap'))
    Y = Yr.T.reshape((T,) + dims, order='F')

    # c, dview, n_processes = cm.cluster.setup_cluster(backend='local', n_processes=n_processes)
    # above line doesn't work cause memory_profiler creates some multiprocessing object itself
    if n_processes is None:
        n_processes = cpu_count()
    dview = Pool(n_processes) if patches else None
    print('{0} processes'.format(n_processes))

    patch_args = dict(nb_patch=0, del_duplicates=True, rf=(rf, rf), stride=(16, 16)) \
        if patches else {}

    cnm = cnmf.CNMF(n_processes=n_processes, method_init='corr_pnr', k=None, dview=dview,
                    gSig=(3, 3), gSiz=(10, 10), merge_thresh=.8, p=1, tsub=2, ssub=1,
                    only_init_patch=True, gnb=0, min_corr=.9, min_pnr=15, normalize_init=False,
                    ring_size_factor=1.5, center_psf=True, ssub_B=2, init_iter=1, **patch_args)
    cnm.fit(Y)
    if patches:
        dview.terminate()
    t += time()
    sleep(1)  # just in case Pool takes some time to terminate
    return t


try:
    dview.terminate()
except:
    pass


results = {'32': {}, '48': {}, '64': {}, 'noPatches': {}}
n_procs = [1, 2, 4, 6, 8, 12, 16, 24]
runs = 5

for n_proc in n_procs:
    for rf in [32, 48, 64]:
        results[str(rf)]['%dprocess' % n_proc] = [memory_usage(
            proc=lambda: main(n_processes=n_proc, rf=rf), include_children=True, retval=True)
            for run in range(runs)]
results['noPatches'] = [memory_usage(
    proc=lambda: main(patches=False), include_children=True, retval=True)
    for run in range(runs)]

with open('memory.pkl', 'wb') as fp:  # save results
    pickle.dump(results, fp)


#%% PLOT RESULTS

"""change some defaults for plotting"""
plt.rc('figure', facecolor='white', frameon=False)
plt.rc('lines', lw=2)
plt.rc('legend', **{'fontsize': 16, 'frameon': False, 'labelspacing': .3, 'handletextpad': .3})
plt.rc('axes', linewidth=2)
plt.rc('xtick.major', size=10, width=1.5)
plt.rc('ytick.major', size=10, width=1.5)
plt.rc('font', **{'family': 'Myriad Pro', 'weight': 'regular', 'size': 24})
plt.rcParams['pdf.fonttype'] = 42


def get_max_mem(rf='64'):
    patch = []
    for proc in n_procs:
        tmp = results[rf]['%dprocess' % proc]
        t = np.array(map(lambda a: a[1], tmp))
        m = np.array(map(lambda a: max(a[0]), tmp))
        patch.append([t, m])
    return np.transpose(patch)


patch = {}
for rf in ('64', '48', '32'):
    patch[rf] = get_max_mem(rf)
nopatch = np.array([map(lambda a: a[1], results['noPatches']),
                    map(lambda a: max(a[0]), results['noPatches'])])

max_time = max([patch[rf][:, 0].max() for rf in ('64', '48', '32')]) / 60
max_mem = max([patch[rf][:, 1].max() for rf in ('64', '48', '32')]) / 1024

plt.figure()
for rf in ('64', '48', '32'):
    size = int(rf) * 2
    plt.errorbar(patch[rf].mean(0)[0], patch[rf].mean(0)[1],
                 xerr=patch[rf].std(0)[0], yerr=patch[rf].std(0)[1],
                 ls='None', capsize=5, capthick=2,
                 label=(('w/ patches ' + '  ' * (size < 100) + '{0}x{0}'.format(size))))
plt.errorbar(nopatch[0].mean(), nopatch[1].mean(),
             xerr=nopatch[0].std(), yerr=nopatch[1].std(),
             ls='None', capsize=5, capthick=2, label='w/o patches')
plt.legend()
plt.xticks(60 * np.arange(0, max_time), range(int(max_time + 1)))
plt.yticks(1024 * np.arange(0, max_mem, 5), range(0, int(max_mem + 1), 5))
plt.xlabel('Time [min]')
plt.ylabel('Peak memory [GB]')
plt.tight_layout(0)
plt.show()
