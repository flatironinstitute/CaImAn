#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 17:56:14 2016

@author: agiovann
"""
from __future__ import print_function

#%%
from builtins import zip
from builtins import str
from builtins import range
try:
    get_ipython().magic('load_ext autoreload')
    get_ipython().magic('autoreload 2')
    print((1))
except:
    print('Not launched under iPython')

import matplotlib as mpl
mpl.use('TKAgg')
from matplotlib import pyplot as plt
# plt.ion()

import sys
import numpy as np
import ca_source_extraction as cse

# sys.path.append('../SPGL1_python_port')
#%
from time import time
from scipy.sparse import coo_matrix
import tifffile
import subprocess
import time as tm
from time import time
import pylab as pl
import psutil
import calblitz as cb
import shutil
import glob
from ipyparallel import Client
import os
import glob
import h5py
import re

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
    dview = c[:]

#%% get all the right folders
params = [
    #['Jan25_2015_07_13',30,False,False,False], # fname, frate, do_rotate_template, do_self_motion_correct, do_motion_correct
    #['Jan40_exp2_001',30,False,False,False],
    #['Jan42_exp4_001',30,False,False,False],
    #['Jan-AMG1_exp2_new_001',30,False,False,False],
    #['Jan-AMG_exp3_001',30,False,False,False],
    #['Yi.data.001',30,False,True,True],
    #['Yi.data.002',30,False,True,True],
    #['FN.151102_001',30,False,True,True],
    #['J115_2015-12-09_L01',30,False,False,False],
    #['J123_2015-11-20_L01_0',30,False,False,False],
    ['k26_v1_176um_target_pursuit_002_013', 30, False, True, True],
    ['k31_20151223_AM_150um_65mW_zoom2p2', 30, False, True, True],
    ['k31_20160104_MMA_150um_65mW_zoom2p2', 30, False, True, True],
    ['k31_20160106_MMA_400um_118mW_zoom2p2_00001_1-19', 30, True, True, True],
    ['k31_20160107_MMP_150um_65mW_zoom2p2_00001_1-15', 30, True, True, True],
    ['k36_20151229_MMA_200um_65mW_zoom2p2_00001_1-17', 30, True, True, True],
    ['k36_20160115_RSA_400um_118mW_zoom2p2_00001_20-38', 30, True, True, True],
    ['k36_20160127_RL_150um_65mW_zoom2p2_00002_22-41', 30, True, True, True],
    ['k37_20160109_AM_150um_65mW_zoom2p2_00001_1-16', 30, True, True, True],
    #['neurofinder.00.00',7,False,False,False],
    #['neurofinder.00.01',7,False,False,False],
    #['neurofinder.00.02',7,False,False,False],
    #['neurofinder.00.03',7,False,False,False],
    #['neurofinder.00.04',7,False,False,False],
    #['neurofinder.01.00',7.5,False,False,False],
    #['neurofinder.01.01',7.5,False,False,False],
    #['neurofinder.02.00',8,False,False,False],
    #['neurofinder.04.00',6.75,False,False,False],
    #['packer.001',15,False,False,False],
    #['yuste.Single_150u',10,False,False,False]
]
# params=params[11:13]
f_rates = np.array([el[1] for el in params])
base_folders = [os.path.join('/mnt/ceph/neuro/labeling', el[0])
                for el in params]
do_rotate_template = np.array([el[2] for el in params])
do_self_motion_correct = np.array([el[3] for el in params])
do_motion_correct = np.array([el[4] for el in params])
#%% FOR EFTY
# dir_efty='/mnt/ceph/neuro/labeling/neurofinder.01.01'
# median=cb.load(dir_efty+'/projections/median_projection.tif',fr=1)
# c_img=cb.load(dir_efty+'/projections/correlation_image.tif',fr=1)
#
# masks_ben=cse.utilities.nf_read_roi_zip(dir_efty+'/regions/ben_regions.zip',median.shape)
# masks_nf=cse.utilities.nf_load_masks(dir_efty+'/regions/regions.json',median.shape)
# masks_beal=cse.utilities.nf_read_roi_zip(dir_efty+'/regions/beale_regions.zip',median.shape)
# masks_sabr=cse.utilities.nf_read_roi_zip(dir_efty+'/regions/sabrina_regions.zip',median.shape)
#
# lq,hq=np.percentile(c_img,[5,95])
# lq_t,hq_t=np.percentile(median,[5,99])
#
# pl.subplot(2,3,1)
# pl.imshow(c_img,cmap='gray',vmin=lq,vmax=hq)
# pl.axis('off')
#pl.title('Correlation Image')
# pl.subplot(2,3,2)
# pl.imshow(c_img,cmap='gray',vmin=lq,vmax=hq)
# pl.imshow(np.sum(masks_ben,0),cmap='hot',alpha=.3)
# pl.axis('off')
# pl.title('ben')
# pl.subplot(2,3,3)
# pl.imshow(c_img,cmap='gray',vmin=lq,vmax=hq)
# pl.imshow(np.sum(masks_nf,0),cmap='Greens',alpha=.3)
# pl.title('neurofinder')
# pl.axis('off')

# pl.subplot(2,3,4)
# pl.imshow(c_img,cmap='gray',vmin=lq,vmax=hq)
# pl.imshow(np.sum(masks_beal,0),cmap='Greens',alpha=.3)
# pl.axis('off')
# pl.title('beal')
# pl.subplot(2,3,5)
# pl.imshow(c_img,cmap='gray',vmin=lq,vmax=hq)
# pl.imshow(np.sum(masks_ben,0),cmap='hot',alpha=.3)
# pl.imshow(np.sum(masks_nf,0),cmap='Greens',alpha=.3)
#pl.title('ben + nf + CI')
# pl.axis('off')
# pl.subplot(2,3,6)
# pl.imshow(np.sum(masks_ben,0)>0,cmap='hot')
# pl.imshow(np.sum(masks_nf,0)>0,cmap='Greens',alpha=.5)
#pl.title('ben + nf')
# pl.pause(.1)

#%%
final_f_rate = 5.0
#%%
counter = 0
images = [os.path.join(el, 'images') for el in base_folders]
regions = [os.path.join(el, 'regions') for el in base_folders]
projections = [os.path.join(el, 'projections') for el in base_folders]

counter = 0
masks_all = []
masks_all_nf = []
templates = []
templates_path = []
corr_images = []
for reg, img, proj, rot, self_mot, do_mot in zip(regions, images, projections, do_rotate_template, do_self_motion_correct, do_motion_correct):

    print(counter)
    counter += 1
    m = cb.load(proj + '/median_projection.tif', fr=1)
    templates_path.append(proj + '/median_projection.tif')
    m1 = cb.load(proj + '/correlation_image.tif', fr=1)

    masks = cse.utilities.nf_read_roi_zip(reg + '/ben_regions.zip', m.shape)
    masks_nf = None
#    masks_nf=cse.utilities.nf_load_masks(reg+'/regions.json',m.shape)
    if rot:
        m = m.T
        m1 = m1.T
        masks = np.transpose(masks, [0, 2, 1])
#        masks_nf=np.transpose(masks_nf,[0,2,1])
    if self_mot and do_mot:
        m = None
        m1 = None

    masks_all.append(masks)
    masks_all_nf.append(masks_nf)
    templates.append(m)
    corr_images.append(m1)
#%% visualize averages and masks
# counter=0
# pl.close('all')
# for reg,img,proj,masks,template,c_img,masks_nf in zip(regions,images,projections,masks_all,templates,corr_images,masks_all_nf):
#
#
#    pl.figure()
#    counter+=1
#
#    template[np.isnan(c_img)]=0
#
#    lq,hq=np.percentile(c_img,[5,95])
#    lq_t,hq_t=np.percentile(template,[5,99])
#    pl.subplot(2,3,1)
#    pl.imshow(c_img,cmap='gray',vmin=lq,vmax=hq)
#    pl.axis('off')
#    pl.title('Correlation Image')
#    pl.subplot(2,3,2)
#    pl.imshow(c_img,cmap='gray',vmin=lq,vmax=hq)
#    pl.imshow(np.sum(masks,0),cmap='hot',alpha=.3)
#    pl.axis('off')
#    pl.title('ben')
#    pl.subplot(2,3,3)
#    pl.imshow(c_img,cmap='gray',vmin=lq,vmax=hq)
#    pl.imshow(np.sum(masks_nf,0),cmap='Greens',alpha=.3)
#    pl.title('neurofinder')
#    pl.axis('off')
#    pl.subplot(2,3,4)
#    pl.imshow(template,cmap='gray',vmin=lq_t,vmax=hq_t)
#    pl.axis('off')
#    pl.subplot(2,3,5)
#    pl.imshow(c_img,cmap='gray',vmin=lq,vmax=hq)
#    pl.imshow(np.sum(masks,0),cmap='hot',alpha=.3)
#    pl.imshow(np.sum(masks_nf,0),cmap='Greens',alpha=.3)
#    pl.title('both + CI')
#    pl.axis('off')
#    pl.subplot(2,3,6)
#    pl.imshow(np.sum(masks,0)>0,cmap='hot')
#    pl.imshow(np.sum(masks_nf,0)>0,cmap='Greens',alpha=.5)
#    pl.title('both')
#    pl.pause(.1)
#    break

#%% visualize averages and masks
counter = 0
for reg, img, proj, masks, template, c_img, masks_nf in zip(regions, images, projections, masks_all, templates, corr_images, masks_all_nf):
    pl.subplot(5, 6, counter + 1)

    print(counter)

    counter += 1

    template[np.isnan(c_img)] = 0

    lq, hq = np.percentile(c_img, [5, 95])

    pl.imshow(c_img, cmap='gray', vmin=lq, vmax=hq)
    pl.imshow(np.sum(masks, 0), cmap='hot', alpha=.3)
    pl.imshow(np.sum(masks_nf, 0), cmap='Greens', alpha=.3)

    pl.axis('off')
    pl.title(img.split('/')[-2])
    pl.pause(.1)
#%% HERE GOES TO MOTION CORRECTION
xy_shifts = []
for fl, tmpl in zip(fls, tmpls):
    if os.path.exists(fl[:-3] + 'npz'):
        print((fl[:-3] + 'npz'))
        with np.load(fl[:-3] + 'npz') as ld:
            xy_shifts.append(ld['shifts'])
    else:
        raise Exception('*********************** ERROR, FILE NOT EXISTING!!!')
#        with np.load(fl[:-3]+'npz') as ld:
#%%
name_new = cse.utilities.save_memmap_each(
    fls, dview=c[::3], base_name=None, resize_fact=resize_facts, remove_init=0, xy_shifts=xy_shifts)
#%%
pars = []
import re

for bf in base_folders:
    fls = glob.glob(os.path.join(bf, 'images/*.mmap'))
    try:
        fls.sort(key=lambda fn: np.int(
            re.findall('_[0-9]{1,5}_d1_', fn)[0][1:-4]))
    except:
        fls.sort()
        print(fls)

    base_name_ = 'TOTAL_'
    n_chunks_ = 6
    dview_ = None
    pars.append([fls, base_name_, n_chunks_, dview_])
#%%
name_new = []


def memmap_place_holder(par):
    import ca_source_extraction as cse
    fls, base_name_, n_chunks_, dview_ = par
    return cse.utilities.save_memmap_join(fls, base_name=base_name_, n_chunks=n_chunks_, dview=dview_)


#%%
dview = c[::3]
names_map = dview.map_sync(memmap_place_holder, pars)
#%% extract file names
# tmpls=[]
# fls=[]
# frates=[]
# resize_facts=[]
# mmap_files=[]
# for reg,img,proj,template,f_rate in zip(regions,images,projections,templates_path,f_rates):
#    fl=glob.glob(img+'/*.tif')
#    fl.sort()
#    for ff in fl:
#        if len(glob.glob(ff[:-4]+'*.mmap'))==0 or 1:
#            print 'adding ' + ff
#            fls=fls+[ff]
#            tmpls=tmpls+[template]
#            frates=frates+[f_rate]
#            resize_facts=resize_facts+[(1,1,final_f_rate/f_rate)]
#        else:
#            print 'skipping ' + ff
#            mmap_files.append(glob.glob(ff[:-4]+'*.mmap')[0])
#
# %% FOR MOTION CORRECTION LOOK AT MOTIONN_CORRECTION.py
# %% create memmap file for each tif file in C order
#name_new=cse.utilities.save_memmap_each(fls, dview=c[::4],base_name=None, resize_fact=resize_facts, remove_init=0)
#%% create avg image for each chunk


def create_average_image(fname):
    import os
    if not os.path.exists(fname[:-5] + '_avg_image.npy'):
        import ca_source_extraction as cse
        import numpy as np

        Yr, dims, T = cse.utilities.load_memmap(fname)
        img = np.mean(Yr, -1)
        img = np.reshape(img, dims, order='F')
        np.save(fname[:-5] + '_avg_image.npy', np.array(img))
        return img

    return None
#%%


#%% create average images so that one could look at them
b_dview = c.load_balanced_view(targets=list(range(1, len(c), 3)))
images = b_dview.map_sync(create_average_image, names_map)
np.save('all_averages.npy', np.array(images))
#%% in order to maximally parallelize, we pass portions of work to differet workers
pars = []
import re

for bf in base_folders:
    fls = glob.glob(os.path.join(bf, 'images/*.mmap'))
    try:
        fls.sort(key=lambda fn: np.int(
            re.findall('_[0-9]{1,5}_d1_', fn)[0][1:-4]))
    except:
        fls.sort()
        print(fls)

    base_name_ = 'TOTAL_'
    n_chunks_ = 6
    dview_ = None
    pars.append([fls, base_name_, n_chunks_, dview_])
#%%
name_new = []


def memmap_place_holder(par):
    import ca_source_extraction as cse
    fls, base_name_, n_chunks_, dview_ = par
    return cse.utilities.save_memmap_join(fls, base_name=base_name_, n_chunks=n_chunks_, dview=dview_)


#%%
names_map = b_dview.map_sync(memmap_place_holder, pars)
#%%
fname_new = cse.utilities.save_memmap_join(
    fls, base_name='TOTAL_', n_chunks=6, dview=c[::3])
#%%

#%%
fnames_mmap = []
for reg, img, proj, masks, template in zip(regions, images, projections, masks_all, templates):
    if len(glob.glob(os.path.join(img, 'TOTAL_*.mmap'))) == 1:
        fnames_mmap.append(glob.glob(os.path.join(img, 'TOTAL_*.mmap'))[0])
    else:
        raise Exception('Number of files not as expected!')
#%%
imgs_avg_mmap = []
counter = 0
pl.close('all')
for nm, tmpl, masks in zip(fnames_mmap, templates, masks_all):
    pl.subplot(5, 6, counter + 1)
    print(nm)
    Yr, dims, T = cse.utilities.load_memmap(nm)
    d1, d2 = dims
    Y = np.reshape(Yr, dims + (T,), order='F')
    img = np.mean(Y, -1)
    counter += 1
    imgs_avg_mmap.append(img)
    pl.imshow(img, cmap='gray')
#%%
counter = 0
for nm, tmpl, masks, avg_img in zip(fnames_mmap, templates, masks_all, imgs_avg_mmap):
    pl.subplot(5, 6, counter + 1)
    counter += 1
    pl.imshow(avg_img, cmap='gray')
    pl.title(nm.split('/')[-3])
#%% check averages
counter = 0
for reg, img, proj, masks, template in zip(regions, images, projections, masks_all, templates):
    pl.subplot(5, 6, counter + 1)
    print(counter)
    movie_files = glob.glob(img + '/*.mmap')
    m = cb.load(movie_files[0], fr=6)
    template = np.mean(m, 0)
    lq, hq = np.percentile(template, [10, 99])
    pl.imshow(template, cmap='gray', vmin=lq, vmax=hq)
    pl.pause(.1)
    counter += 1
    pl.title(img.split('/')[-2])
#%% process files sequntially in case of failure
if 0:
    fnames1 = []
    for f in fnames:
        if os.path.isfile(f[:-3] + 'hdf5'):
            1
        else:
            print((1))
            fnames1.append(f)
    #%% motion correct
    t1 = time()
    file_res = cb.motion_correct_parallel(
        fnames1, fr=30, template=None, margins_out=0, max_shift_w=45, max_shift_h=45, dview=None, apply_smooth=True)
    t2 = time() - t1
    print(t2)
#%% LOGIN TO MASTER NODE
# TYPE salloc -n n_nodes --exclusive
# source activate environment_name

#%%#%%
backend = 'local'
if backend == 'slurm':
    slurm_script = '/mnt/xfs1/home/agiovann/SOFTWARE/Constrained_NMF/SLURM/slurmStart.sh'
    cse.utilities.start_server(slurm_script=slurm_script)
    # n_processes = 27#np.maximum(psutil.cpu_count() - 2,1) # roughly number of cores on your machine minus 1
    pdir, profile = os.environ['IPPPDIR'], os.environ['IPPPROFILE']
    client_ = Client(ipython_dir=pdir, profile=profile)
else:
    cse.utilities.stop_server()
    cse.utilities.start_server()
    client_ = Client()
print(('Using ' + str(len(client_)) + ' processes'))

#%% motion correct
t1 = time()
file_res = cb.motion_correct_parallel(fnames, fr=30, template=None, margins_out=0,
                                      max_shift_w=45, max_shift_h=45, dview=client_[::2], apply_smooth=True)
t2 = time() - t1
print(t2)

#%%
all_movs = []
counter = 0
for f in fls:
    print(f)
    with np.load(f[:-3] + 'npz') as fl:
        pl.subplot(6, 5, counter + 1)
#        pl.imshow(fl['template'],cmap=pl.cm.gray)
#        pl.subplot(1,2,2)
        pl.plot(fl['shifts'])
        counter += 1
#        all_movs.append(fl['template'][np.newaxis,:,:])
#        pl.pause(.1)
#        pl.cla()
#%%
all_movs = cb.movie(np.concatenate(all_movs, axis=0), fr=10)
all_movs, shifts, corss, _ = all_movs.motion_correct(
    template=all_movs[1], max_shift_w=45, max_shift_h=45)
#%%
template = np.median(all_movs[:], axis=0)
np.save(base_folder + 'template_total', template)
pl.imshow(template, cmap=pl.cm.gray, vmax=120)
#%%
all_movs.play(backend='opencv', gain=5, fr=30)
#%%
t1 = time()
file_res = cb.motion_correct_parallel(fnames, 30, template=template, margins_out=0,
                                      max_shift_w=45, max_shift_h=45, dview=client_[::2], remove_blanks=False)
t2 = time() - t1
print(t2)
#%%
fnames = []
for file in glob.glob(base_folder + 'k31_20160107_MMP_150um_65mW_zoom2p2_000*[0-9].hdf5'):
    fnames.append(file)
fnames.sort()
print(fnames)
#%%
file_res = cb.utils.pre_preprocess_movie_labeling(client_[::2], fnames, median_filter_size=(2, 1, 1),
                                                  resize_factors=[.2, .1666666666], diameter_bilateral_blur=4)

#%%
client_.close()
cse.utilities.stop_server(is_slurm=True)

#%%

#%%
fold = os.path.split(os.path.split(fnames[0])[-2])[-1]
os.mkdir(fold)
#%%
files = glob.glob(fnames[0][:-20] + '*BL_compress_.tif')
files.sort()
print(files)
#%%
m = cb.load_movie_chain(files, fr=3)
m.play(backend='opencv', gain=10, fr=40)
#%%
m.save(files[0][:-20] + '_All_BL.tif')
#%%
files = glob.glob(fnames[0][:-20] + '*[0-9]._compress_.tif')
files.sort()
print(files)
#%%
m = cb.load_movie_chain(files, fr=3)
m.play(backend='opencv', gain=3, fr=40)
#%%
m.save(files[0][:-20] + '_All.tif')
