# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 18:39:45 2016
@author: Andrea Giovannucci
For explanation consult at https://github.com/agiovann/Constrained_NMF/releases/download/v0.4-alpha/Patch_demo.zip
and https://github.com/agiovann/Constrained_NMF
"""
from __future__ import print_function
#%%
from builtins import str
from builtins import range
try:
    if __IPYTHON__:
        # this is used for debugging purposes only. allows to reload classes when changed
        get_ipython().magic('load_ext autoreload')
        get_ipython().magic('autoreload 2')
except NameError:
    print('Not IPYTHON')
    pass

import sys
import numpy as np
from time import time
from scipy.sparse import coo_matrix
import psutil
import glob
import os
import scipy
from ipyparallel import Client
import matplotlib as mpl
# mpl.use('Qt5Agg')
import pylab as pl
pl.ion()
#%%
from skimage.external import tifffile
import caiman as cm
from caiman.source_extraction import cnmf as cnmf
from caiman.components_evaluation import evaluate_components
from caiman.utils.visualization import plot_contours
from caiman.base.rois import extract_binary_masks_blob
#%%
fname = '/mnt/ceph/neuro/labeling/k37_20160109_AM_150um_65mW_zoom2p2_00001_1-16/images/k37_20160109_AM_150um_65mW_zoom2p2_00001_00001.tif'
# fname='/run/media/agiovann/ANDREA/k26_v1_176um_target_pursuit_002_013.tif'
fname = './example_movies/demoMovie.tif'

with tifffile.TiffFile(fname) as tf:
    m = cm.movie(tf, fr=30)
    mh = cm.movie(tf.asarray(), fr=30)
#(mh-np.percentile(mh,8)).play(backend='opencv',fr=100,gain=10.,magnification=5)
# mh=mh.motion_correct(5,5)[0]
#%%
from sklearn.metrics.pairwise import euclidean_distances
# mh=mh[:,30:60,30:60]
T, h, w = np.shape(mh)
idxA, idxB = np.meshgrid(list(range(w)), list(range(h)))
coordmat = np.vstack((idxA.flatten(), idxB.flatten()))
distanceMatrix = euclidean_distances(coordmat.T)
distanceMatrix = np.logical_and(
    distanceMatrix < 2, distanceMatrix > 0) * distanceMatrix
img = np.corrcoef(mh.to_2D(order='C').T)
img = img * distanceMatrix
img = scipy.sparse.coo_matrix(img)
from sklearn.cluster import DBSCAN, KMeans, SpectralClustering
cl = SpectralClustering(n_clusters=3).fit(img)

#%%
pl.imshow(cl.labels_.reshape((h, w)))
# pl.imshow(mh.local_correlations(eight_neighbours=True,swap_dim=False),alpha=.5)
#%%
pl.imshow(img, interpolation='none')
#%%
cc = scipy.sparse.csgraph.connected_components(img, directed=False)
# distanceMatrix=distanceMatrix/np.max(distanceMatrix)
#%%
m = load(fname, fr=30).resize(1, 1, .2)
#%%
shifts_, xcorrs_, template_ = m.motion_correction_online(
    init_frames_template=100, max_shift_h=5, max_shift_w=5)
#%%
shifts, xcorrs, template = m.motion_correction_online(
    template=template_, min_count=len(m), max_shift_h=5, max_shift_w=5)
#%%
mov_path = m.apply_shifts_online(shifts, save_base_name='/tmp/test')
#%%
m = load(mov_path, fr=30 * .2)
m1 = cm.concatenate([m.resize(1, 1, 1), mh.resize(1, 1, 1)], axis=1)

(m1 - np.percentile(m1, 8)).play(backend='opencv', fr=100, gain=3., magnification=5)

#%%
# mc=m.motion_correct(5,5)
#%%
# m1=mc[0].resize(1,1,.2)
#%%
(m1 - np.percentile(m1, 8)).play(backend='opencv', fr=100, gain=10., magnification=5)
#%%
final_frate = 15
is_patches = True
is_dendrites = False


if is_dendrites == True:
    # THIS METHOd CAN GIVE POSSIBLY INCONSISTENT RESULTS ON SOMAS WHEN NOT USED WITH PATCHES
    init_method = 'sparse_nmf'
    alpha_snmf = None  # this controls sparsity
else:
    init_method = 'greedy_roi'
    alpha_snmf = None

#%%
# backend='SLURM'
backend = 'local'
if backend == 'SLURM':
    n_processes = np.int(os.environ.get('SLURM_NPROCS'))
else:
    # roughly number of cores on your machine minus 1
    n_processes = np.maximum(np.int(psutil.cpu_count()), 1)
print(('using ' + str(n_processes) + ' processes'))
#%% start cluster for efficient computation
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
            stop_server(is_slurm=True)
        except:

            print('Nothing to stop')
        slurm_script = '/mnt/xfs1/home/agiovann/SOFTWARE/Constrained_NMF/SLURM/slurmStart.sh'
        start_server(slurm_script=slurm_script)

        pdir, profile = os.environ['IPPPDIR'], os.environ['IPPPROFILE']
        c = Client(ipython_dir=pdir, profile=profile)
    else:

        stop_server()
        start_server()
        c = Client()

    print(('Using ' + str(len(c)) + ' processes'))
    dview = c[:len(c)]
#%% FOR LOADING ALL TIFF FILES IN A FILE AND SAVING THEM ON A SINGLE MEMORY MAPPABLE FILE
fnames = []
base_folder = '/run/media/agiovann/ANDREA/'  # folder containing the demo files
base_folder = '/run/media/agiovann/ANDREA/'
base_folder = './example_movies'
for file in glob.glob(os.path.join(base_folder, '*.tif')):

    if file.endswith("ie.tif"):
        fnames.append(os.path.abspath(file))
fnames.sort()
if len(fnames) == 0:
    raise Exception("Could not find any tiff file")

print(fnames)
fnames = fnames
#%%
# idx_x=slice(12,500,None)
# idx_y=slice(12,500,None)
# idx_xy=(idx_x,idx_y)

add_to_movie = 300
downsample_factor = 1.  # use .2 or .1 if file is large and you want a quick answer
final_frate = final_frate * downsample_factor
idx_xy = None
base_name = 'Yr'
name_new = save_memmap_each(fnames, dview=dview, base_name=base_name, resize_fact=(
    1, 1, downsample_factor), remove_init=0, idx_xy=idx_xy, add_to_movie=add_to_movie)

name_new.sort()
print(name_new)
#%%

fname_new = save_memmap_join(
    name_new, base_name='Yr', n_chunks=12, dview=dview)
#%%
# fname_new='Yr_d1_501_d2_398_d3_1_order_F_frames_369_.mmap'
# fname_new=m1
Yr, dims, T = load_memmap(fname_new)
d1, d2 = dims
images = np.reshape(Yr.T, [T] + list(dims), order='F')
Y = np.reshape(Yr, dims + (T,), order='F')

#%%
Cn = cnmf.utilities.local_correlations(Y[:, :, :3000])
pl.imshow(Cn, cmap='gray')

#%%
if not is_patches:
    #%%

    K = 35  # number of neurons expected per patch
    gSig = [7, 7]  # expected half size of neurons
    merge_thresh = 0.8  # merging threshold, max correlation allowed
    p = 2  # order of the autoregressive system
    cnm = cnmf.CNMF(n_processes, method_init=init_method, k=K, gSig=gSig, merge_thresh=merge_thresh,
                    p=p, dview=dview, Ain=None)
    cnm = cnm.fit(images)


#%%
else:
    #%%
    rf = 15  # half-size of the patches in pixels. rf=25, patches are 50x50
    stride = 4  # amounpl.it of overlap between the patches in pixels
    K = 6  # number of neurons expected per patch
    gSig = [7, 7]  # expected half size of neurons
    merge_thresh = 0.8  # merging threshold, max correlation allowed
    p = 2  # order of the autoregressive system
    memory_fact = 1  # unitless number accounting how much memory should be used. You will need to try different values to see which one would work the default is OK for a 16 GB system
    save_results = False
    #%% RUN ALGORITHM ON PATCHES

    cnm = cnmf.CNMF(n_processes, k=K, gSig=gSig, merge_thresh=0.8, p=0, dview=c[:], Ain=None, rf=rf, stride=stride, memory_fact=memory_fact,
                    method_init=init_method, alpha_snmf=alpha_snmf, only_init_patch=True, gnb=1)
    cnm = cnm.fit(images)

    A_tot = cnm.A
    C_tot = cnm.C
    YrA_tot = cnm.YrA
    b_tot = cnm.b
    f_tot = cnm.f
    sn_tot = cnm.sn

    print(('Number of components:' + str(A_tot.shape[-1])))

    #%%
    tB = np.minimum(-2, np.floor(-5. / 30 * final_frate))
    tA = np.maximum(5, np.ceil(25. / 30 * final_frate))
    Npeaks = 10
    traces = C_tot + YrA_tot
    #        traces_a=traces-scipy.ndimage.percentile_filter(traces,8,size=[1,np.shape(traces)[-1]/5])
    #        traces_b=np.diff(traces,axis=1)
    fitness_raw, fitness_delta, erfc_raw, erfc_delta, r_values, significant_samples = evaluate_components(
        Y, traces, A_tot, C_tot, b_tot, f_tot, remove_baseline=True, N=5, robust_std=False, Athresh=0.1, Npeaks=Npeaks, tB=tB, tA=tA, thresh_C=0.3)

    idx_components_r = np.where(r_values >= .4)[0]
    idx_components_raw = np.where(fitness_raw < -20)[0]
    idx_components_delta = np.where(fitness_delta < -10)[0]

    idx_components = np.union1d(idx_components_r, idx_components_raw)
    idx_components = np.union1d(idx_components, idx_components_delta)
    idx_components_bad = np.setdiff1d(list(range(len(traces))), idx_components)

    print(('Keeping ' + str(len(idx_components)) +
           ' and discarding  ' + str(len(idx_components_bad))))
    #%%
    pl.figure()
    crd = plot_contours(A_tot.tocsc()[:, idx_components], Cn, thr=0.9)

    #%%
    A_tot = A_tot.tocsc()[:, idx_components]
    C_tot = C_tot[idx_components]
    #%%
    save_results = True
    if save_results:
        np.savez('results_analysis_patch.npz', A_tot=A_tot, C_tot=C_tot,
                 YrA_tot=YrA_tot, sn_tot=sn_tot, d1=d1, d2=d2, b_tot=b_tot, f=f_tot)
    #%% if you have many components this might take long!
    pl.figure()
    crd = plot_contours(A_tot, Cn, thr=0.9)
    #%%
    cnm = cnmf.CNMF(n_processes, k=A_tot.shape, gSig=gSig, merge_thresh=merge_thresh, p=p, dview=dview, Ain=A_tot, Cin=C_tot,
                    f_in=f_tot, rf=None, stride=None)

#    images_1=np.array(images-np.min(images))
    cnm = cnm.fit(images)

#%%
A, C, b, f, YrA, sn = cnm.A, cnm.C, cnm.b, cnm.f, cnm.YrA, cnm.sn

#%%
tB = np.minimum(-2, np.floor(-5. / 30 * final_frate))
tA = np.maximum(5, np.ceil(25. / 30 * final_frate))
Npeaks = 10
traces = C + YrA
#        traces_a=traces-scipy.ndimage.percentile_filter(traces,8,size=[1,np.shape(traces)[-1]/5])
#        traces_b=np.diff(traces,axis=1)
fitness_raw, fitness_delta, erfc_raw, erfc_delta, r_values, significant_samples = \
    evaluate_components(Y, traces, A, C, b, f, remove_baseline=True,
                        N=5, robust_std=False, Athresh=0.1, Npeaks=Npeaks, tB=tB, tA=tA, thresh_C=0.3)


idx_components_r = np.where(r_values >= .5)[0]
idx_components_raw = np.where(fitness_raw < -40)[0]
idx_components_delta = np.where(fitness_delta < -20)[0]


min_radius = gSig[0] - 2
masks_ws, idx_blobs, idx_non_blobs = extract_binary_masks_blob(
    A.tocsc(), min_radius, dims, num_std_threshold=1,
    minCircularity=0.5, minInertiaRatio=0.2, minConvexity=.8)

idx_components = np.union1d(idx_components_r, idx_components_raw)
idx_components = np.union1d(idx_components, idx_components_delta)
idx_blobs = np.intersect1d(idx_components, idx_blobs)
idx_components_bad = np.setdiff1d(list(range(len(traces))), idx_components)


print(' ***** ')
print((len(traces)))
print((len(idx_components)))
print((len(idx_blobs)))
#%%
save_results = True
if save_results:
    np.savez(os.path.join(os.path.split(fname_new)[0], 'results_analysis.npz'), Cn=Cn, A=A.todense(
    ), C=C, b=b, f=f, YrA=YrA, sn=sn, d1=d1, d2=d2, idx_components=idx_components, idx_components_bad=idx_components_bad)

#%% visualize components
# pl.figure();
pl.subplot(1, 3, 1)
crd = plot_contours(A.tocsc()[:, idx_components], Cn, thr=0.9)
pl.subplot(1, 3, 2)
crd = plot_contours(A.tocsc()[:, idx_blobs], Cn, thr=0.9)
pl.subplot(1, 3, 3)
crd = plot_contours(A.tocsc()[:, idx_components_bad], Cn, thr=0.9)
#%%
cnmf.utilities.view_patches_bar(Yr, scipy.sparse.coo_matrix(A.tocsc()[
                                :, idx_components]), C[idx_components, :], b, f, dims[0], dims[1], YrA=YrA[idx_components, :], img=Cn)
#%%
cnmf.utilities.view_patches_bar(Yr, scipy.sparse.coo_matrix(A.tocsc()[
                                :, idx_components_bad]), C[idx_components_bad, :], b, f, dims[0], dims[1], YrA=YrA[idx_components_bad, :], img=Cn)

#%% STOP CLUSTER
pl.close()
if not single_thread:
    c.close()
    stop_server()
#%%
stop_server(is_slurm=(backend == 'SLURM'))
log_files = glob.glob('Yr*_LOG_*')

for log_file in log_files:
    os.remove(log_file)
