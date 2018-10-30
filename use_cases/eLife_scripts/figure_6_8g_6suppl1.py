#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script reproduces the results for Figure 6, Figure 6-figure supplement 1
and Figure 8g, pertaining to the online analysis of whole brain zebrafish data
The script will load the results and produce the figures. For a script that
analyzes the data from scratch check the script 
/preprocessing_files/preprocess_zebrafish_paper.py

More info can be found in the companion paper
"""
import os
import sys

import numpy as np
try:
    if __IPYTHON__:
        print('Detected iPython')
        # this is used for debugging purposes only. allows to reload classes when changed
        get_ipython().magic('load_ext autoreload')
        get_ipython().magic('autoreload 2')
except NameError:
    pass


from time import time
import caiman as cm
from caiman.utils.visualization import view_patches_bar
from caiman.utils.utils import download_demo, load_object, save_object
import pylab as pl
import scipy
from caiman.motion_correction import motion_correct_iteration_fast
import cv2
from caiman.utils.visualization import plot_contours
import glob
from caiman.source_extraction.cnmf.online_cnmf import bare_initialization
from copy import deepcopy
from caiman.paths import caiman_datadir
#%%
try:
    import sys
    if 'pydevconsole' in sys.argv[0]:
        raise Exception()
    ID = sys.argv[1]
    ID = str(np.int(ID)+1)
    print('Processing ID:'+ str(ID))
    ploton = False
    save_results = True
    save_init = True     # flag for saving initialization object. Useful if you want to check OnACID with different parameters but same initialization
except:
    ID = 11 # process plane 12
    print('ID NOT PASSED')
    ploton = False
    save_results = False
    save_init = False # flag for saving initialization object. Useful if you want to check OnACID with different parameters but same initialization

reload = True
plot_figures = True
base_folder = '/mnt/ceph/neuro/DataForPublications/DATA_PAPER_ELIFE/WEBSITE/'
#%%
decay_time = 1.5
gSig = (6,6)
rval_thr = 1
epochs = 1

fls = [os.path.join(base_folder,'Zebrafish/Plane' + str(ID) + '.stack.hdf5')];
K = 100
min_num_trial = 50

mmm = cm.load(fls,subindices = 0)
dims = mmm.shape
K = np.maximum(K,np.round(600/1602720*np.prod(mmm.shape)).astype(np.int))
min_num_trial = np.maximum(min_num_trial,np.round(200/1602720*np.prod(mmm.shape)).astype(np.int))
# your list of files should look something like this
print(fls)

print([K,min_num_trial])
# number of passes over the data
#%%   Set up some parameters
# frame rate (Hz)
fr = 2
#fr = 15

# approximate length of transient event in seconds

#decay_time = 0.5

# expected half size of neurons
#gSig = (2.5, 2.5)

# order of AR indicator dynamics
p = 1
# minimum SNR for accepting new components
min_SNR = 2.5
# correlation threshold for new component inclusion

#rval_thr = 0.85

# spatial downsampling factor (increases speed but may lose some fine structure)
ds_factor = 2
# number of background components
gnb = 3
# recompute gSig if downsampling is involved
gSig = tuple((np.array(gSig) / ds_factor))#.astype('int'))
# flag for online motion correction
mot_corr = True
# maximum allowed shift during motion correction
max_shift = np.ceil(10. / ds_factor).astype('int')

# set up some additional supporting parameters needed for the algorithm (these are default values but change according to dataset characteristics)

# number of shapes to be updated each time (put this to a finite small value to increase speed)
max_comp_update_shape = np.inf
# number of files used for initialization
init_files = 1
# number of files used for online
online_files = len(fls) - 1
# number of frames for initialization (presumably from the first file)
initbatch = 200
# maximum number of expected components used for memory pre-allocation (exaggerate here)
expected_comps = 600
# initial number of components
# number of timesteps to consider when testing new neuron candidates
N_samples = np.ceil(fr * decay_time)
# exceptionality threshold
thresh_fitness_raw = scipy.special.log_ndtr(-min_SNR) * N_samples

# upper bound for number of frames in each file (used right below)
len_file = 1885#1885 1815
# total length of all files (if not known use a large number, then truncate at the end)
T1 = len(fls) * len_file * epochs
#%%
compute_corr = False
if compute_corr:
    m = cm.load(fls)
    mc = m.motion_correct(10,10)[0]
    mp = (mc.computeDFF(3))
    Cn = cv2.resize(mp[0].local_correlations(eight_neighbours=True, swap_dim=False),dims[::-1][:-1])
    np.save(os.path.join(base_folder,'Zebrafish/results_analysis_online_Plane_CN_' + str(ID) + '.npy'), Cn)
else:
    Cn = np.load(os.path.join(base_folder,'Zebrafish/results_analysis_online_Plane_CN_' + str(ID) + '.npy'))
#%%
create_avg_movie = False
if create_avg_movie:
    big_mov = cm.load(fls[0])
    mean_mov = big_mov.mean(0)
    median_mov = np.median(big_mov,0)
    pl.imshow(median_mov[::-1,::-1].T)
    pl.colorbar()
#%%    Initialize movie
# load only the first initbatch frames and possibly downsample them
if ds_factor > 1:
    Y = cm.load(fls[0], subindices=slice(0, initbatch, None)).astype(
        np.float32).resize(1. / ds_factor, 1. / ds_factor)
else:
    Y = cm.load(fls[0], subindices=slice(
        0, initbatch, None)).astype(np.float32)

if mot_corr:                                        # perform motion correction on the first initbatch frames
    mc = Y.motion_correct(max_shift, max_shift)
    Y = mc[0].astype(np.float32)
    borders = np.max(mc[1])
else:
    Y = Y.astype(np.float32)

# minimum value of movie. Subtract it to make the data non-negative
img_min = Y.min()
Y -= img_min
img_norm = np.std(Y, axis=0)
# normalizing factor to equalize the FOV
img_norm += np.median(img_norm)
Y = Y / img_norm[None, :, :]                        # normalize data

_, d1, d2 = Y.shape
dims = (d1, d2)                                     # dimensions of FOV
Yr = Y.to_2D().T                                    # convert data into 2D array

Cn_init = Y.local_correlations(swap_dim=False)    # compute correlation image
if ploton:
    pl.imshow(Cn_init)
    pl.title('Correlation Image on initial batch')
    pl.colorbar()

#%% reload dataset
with np.load(os.path.join(base_folder,'Zebrafish/results_analysis_online_1EPOCH_gSig6_equalized_Plane_' + str(ID) + '.npz')) as ld:
    locals().update(ld)
    print(ld.keys())
    Ab = Ab[()]
    A, b = Ab[:, gnb:], Ab[:, :gnb].toarray()
    C, f = Cf[gnb:Ab.shape[-1],:T1], Cf[:gnb, :T1]
    noisyC = noisyC[:, :T1]

#        m = cm.movie((A.dot(C)+b.dot(f))[:].reshape(list(dims)+[-1],order='F')).transpose([2,0,1])*img_norm[None,:,:]
    m = cm.movie((A.dot(C))[:].reshape(list(dims)+[-1],order='F')).transpose([2,0,1])*img_norm[None,:,:]
    mean_img = np.median(m,0)

#%%
recompute_CN = False
if recompute_CN:
    m = cm.load(fls)
    m = m.motion_correct(10,10)[0]
    mp = (m.computeDFF(3))
    Cn_ = mp[0].local_correlations(eight_neighbours=True, swap_dim=False)
    Cn_ = cv2.resize(Cn_,dims[::-1])
else:
    Cn_ = Cn
#%% FIGURE 7a TOP
if ploton:
    crd = cm.utils.visualization.plot_contours(A, Cn_, thr=0.9, vmax = 0.75)
    view_patches_bar(Yr, scipy.sparse.coo_matrix(A.tocsc()[:, :]), C[:, :], b, f,
                     dims[0], dims[1], YrA=noisyC[gnb:A.shape[-1]+gnb] - C, img=Cn_)


#%% START CLUSTER
try:
#        cm.stop_server()
    dview.terminate()
except:
    print('No clusters to stop')

c, dview, n_processes = cm.cluster.setup_cluster(
        backend='local', n_processes=24)
#%% FIGURE 6 Supplement 1 (print_masks = True or False to print the right or left sides), including Figure 6d (plane 11)
print('THIS MIGHT TAKE A LONG TIME IF YOU WANT BOTH MASKS AND TRACES!')
print_masks = False
IDDS = range(1,46)
for plot_traces, nm in zip([True, False],['Traces','Masks']):
    from sklearn.preprocessing import normalize
    num_neur = []
    #tott = np.zeros_like(tottime)
    update_comps_time = []
    tott = []
    totneursum = 0
    time_per_neuron = []
    pl.figure("Figure 11 " + nm)
    for ID in IDDS:
#        try:
            with np.load(os.path.join(base_folder,'Zebrafish/results_analysis_online_1EPOCH_gSig6_equalized_Plane_' + str(ID) + '.npz')) as ld:
                locals().update(ld)
                print(np.sum(ld['tottime'])+ld['time_init'])
                tottime = ld['tottime']
                print(ld.keys())
                totneursum += ld['Cf'].shape[0]-3
                pl.subplot(5,9,ID)
#                img = normalize(Ab[()][:,3:],'l1',axis=0).mean(-1).reshape(dims,order = 'F').T
                Cn_ = np.load(os.path.join(base_folder,'Zebrafish/results_analysis_online_Plane_CN_'+str(ID)+ '.npy'))
                if plot_traces :
                    pl.imshow(ld['Cf'][3:],aspect = 'auto', vmax = 10)
                    pl.ylim([0,1950])
                    pl.axis('off')
                    pl.pause(0.1)
                else:

#                pl.figure();crd = cm.utils.visualization.plot_contours(
#                        Ab[()][:,3:].toarray().reshape(tuple(dims)+(-1,), order = 'F').transpose([1,0,2]).\
#                        reshape((dims[1]*dims[0],-1),order = 'F'), cv2.resize(Cn_,tuple(dims[::-1])).T, thr=0.9, vmax = 0.75,
#                        display_numbers=False)
                    A_thr = cm.source_extraction.cnmf.spatial.threshold_components(ld['Ab'][()].tocsc()[:,gnb:].toarray(), dims, medw=None, thr_method='nrg',
                                                                  maxthr=0.3, nrgthr=0.95, extract_cc=True,
                                 se=None, ss=None, dview=dview)
#                np.save('/mnt/ceph/neuro/zebra/05292014Fish1-4/thresholded_components' + str(ID) + '.npy',A_thr)
#                A_thr = np.load('Zebrafish/thresholded_components' + str(ID) + '.npy')
#                img = normalize(Ab[()][:,gnb:].multiply(A_thr),'l1',axis=0).mean(-1).reshape(dims,order = 'F').T
#                img = Ab[()][:,gnb:].multiply(A_thr).mean(-1).reshape(dims,order = 'F').T
                    Ab_thr = ld['Ab'][()][:,gnb:].multiply(A_thr)
                    img = (Ab_thr.dot(scipy.sparse.spdiags(np.minimum(1.0/np.max(Ab_thr,0).toarray(),100),0,Ab_thr.shape[-1],Ab_thr.shape[-1]))).mean(-1).reshape(dims,order = 'F').T
                    xx,yy = np.subtract((560,860),img.shape)//2+1

                    pl.imshow(cv2.copyMakeBorder(img,xx,xx,yy,yy, cv2.BORDER_CONSTANT,0),vmin=np.percentile(img,5),vmax=np.percentile(img,99.99),cmap = 'gray')

#                A_thr = A_thr > 0

#                pl.imshow(((A_thr*np.random.randint(1,10,A_thr.shape[-1])[None,:]).sum(-1).reshape(dims,order='F')).T, cmap = 'hot', vmin = 0.9, vmax=20)
                    pl.axis('off')
                    pl.pause(0.05)

                num_neur.append(num_comps[1884-201])
                tottime = tottime[:1885-201]
                num_comps = num_comps[:1885-201]
                update_comps_time.append((np.array(num_comps)[99::100],tottime[99::100].copy()))
                tottime[99::100] = np.nan
                tottime[0] = np.nan
                [(np.where(np.diff([0]+list(num_comps))==cc)[0], tottime[np.where(np.diff([0]+list(num_comps))==cc)[0]]) for cc in range(6)]
                tott.append(tottime)

#        except:
            print(ID)
    if not print_masks:
        break
    pl.tight_layout()
#%% FIGURE 8 g
pl.rcParams['pdf.fonttype'] = 42
font = {'family' : 'Arial',
'weight' : 'regular',
'size'   : 20}

pl.rc('font', **font)
pl.figure("Figure 8c")
pl.plot(np.arange(1885-201)*1,np.max(tott,0))
pl.plot([0,(1885-201)*1],[1,1],'k--')
pl.ylabel('time (s)')
pl.xlabel('time (s)')
pl.title('neural activity tracking')
#%% FIGURE 6 e
pl.figure("Figure 6e")
pl.plot(num_neur,'r.')

#%% FIGURE 6a,b,c PREPARATION
print('THIS WILL TAKE SOME TIME!!')
from sklearn.preprocessing import normalize
num_neur = []
#tott = np.zeros_like(tottime)
update_comps_time = []
tott = []
time_per_neuron = []
pl.figure()
for ID in range(11,12):
#        try:
        with np.load(os.path.join(base_folder,'Zebrafish/results_analysis_online_1EPOCH_gSig6_equalized_Plane_' + str(ID) + '.npz')) as ld:
            locals().update(ld)
            print(ld.keys())
            pl.subplot(5,9,ID)
#                img = normalize(Ab[()][:,3:],'l1',axis=0).mean(-1).reshape(dims,order = 'F').T
            Cn_ = np.load(os.path.join(base_folder,'Zebrafish/results_analysis_online_Plane_CN_'+str(ID)+ '.npy'))
#
#                pl.imshow(Cf[3:],aspect = 'auto', vmax = 10)
            pl.figure();crd = cm.utils.visualization.plot_contours(
                    Ab[()][:,3:].toarray().reshape(tuple(dims)+(-1,), order = 'F').transpose([1,0,2]).\
                    reshape((dims[1]*dims[0],-1),order = 'F'), cv2.resize(Cn_,tuple(dims[::-1])).T, thr=0.9, vmax = 0.75,
                    display_numbers=False)
            A_thr = cm.source_extraction.cnmf.spatial.threshold_components(Ab[()].tocsc()[:,gnb:].toarray(), dims, medw=None, thr_method='nrg',
                                                              maxthr=0.3, nrgthr=0.95, extract_cc=True,
                             se=None, ss=None, dview=dview)
            Ab_thr = Ab[()][:,gnb:].multiply(A_thr)
            img = (Ab_thr.dot(scipy.sparse.spdiags(np.minimum(1.0/np.max(Ab_thr,0).toarray(),100),0,Ab_thr.shape[-1],Ab_thr.shape[-1]))).mean(-1).reshape(dims,order = 'F').T
            pl.imshow(img,vmin=np.percentile(img,5),vmax=np.percentile(img,99.99),cmap = 'hot')

#                A_thr = A_thr > 0

#                pl.imshow(((A_thr*np.random.randint(1,10,A_thr.shape[-1])[None,:]).sum(-1).reshape(dims,order='F')).T, cmap = 'hot', vmin = 0.9, vmax=20)
            pl.axis('off')
            pl.pause(0.05)


        print(ID)
pl.tight_layout()
#% close the unused figures
pl.close()
pl.close()

#%% predictions for Plan 11 to choose nicely looking neurons
from  skimage.util._montage import  montage2d
predictions, final_crops = cm.components_evaluation.evaluate_components_CNN(Ab[()][:,gnb:], dims, np.array(gSig).astype(np.int), model_name=os.path.join(caiman_datadir(), 'model', 'cnn_model'), patch_size=50, loaded_model=None, isGPU=False)
#%% FIGURE 6 a,b
idx = np.argsort(predictions[:,0])[:10]#[[0,1,2,3,5,9]]
Ab_part = Ab[()][:,gnb:][:,idx]
pl.imshow(montage2d(final_crops[idx]))
pl.close()
pl.figure("Figure 6a base");crd = cm.utils.visualization.plot_contours(
                    Ab_part.toarray().reshape(tuple(dims)+(-1,), order = 'F').transpose([1,0,2]).\
                    reshape((dims[1]*dims[0],-1),order = 'F'), cv2.resize(Cn_,tuple(dims[::-1])).T, thr=0.9, vmax = 0.95,
                    display_numbers=True)
pl.colorbar()
pl.figure("Figure 6a overlay");crd = cm.utils.visualization.plot_contours(
                    Ab_part.toarray().reshape(tuple(dims)+(-1,), order = 'F').transpose([1,0,2]).\
                    reshape((dims[1]*dims[0],-1),order = 'F'), img, thr=0.9, display_numbers=True, vmax = .001)
pl.colorbar()
#%% FIGURE 6c
pl.figure("Figure 6c")
count = 0
for cf,sp_c in zip(Cf[idx+gnb], final_crops[idx]):
    pl.subplot(10,2,2*count+1)
    pl.imshow(sp_c[10:-10,10:-10][::-1][::-1].T)
    pl.axis('off')
    pl.subplot(10,2,2*count+2)
    pl.plot(cf/cf.max())
    pl.ylim([0,1])
    count+=1
    pl.axis('off')
