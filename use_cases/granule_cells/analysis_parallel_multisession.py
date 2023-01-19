#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on Mon Aug  8 18:02:10 2016

@author: agiovann
"""

from __future__ import division
from __future__ import print_function

# analysis parallel
from builtins import next
from builtins import filter
from builtins import str
from builtins import zip
from builtins import range
from past.utils import old_div
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')
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
import caiman as cm
import sys
import numpy as np
import pickle
import use_cases
from use_cases.granule_cells.utils_granule import load_data_from_stored_results, process_eyelid_traces, process_wheel_traces, process_fast_process_day, process_wheel_traces_talmo
import pandas as pd
import re
from caiman.components_evaluation import compute_event_exceptionality
from statsmodels.robust.scale import mad

#%%
if False:
    backend = 'local'
    if backend == 'SLURM':
        n_processes = int(os.environ.get('SLURM_NPROCS'))
    else:
        # roughly number of cores on your machine minus 1
        n_processes = np.maximum(int(psutil.cpu_count()), 1)
    print(('using ' + str(n_processes) + ' processes'))

    #% start cluster for efficient computation
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

        dview = c[::4]
        print(('Using ' + str(len(dview)) + ' processes'))

#%%
base_folders = [
    '/mnt/ceph/users/agiovann/ImagingData/eyeblink/b37/20160627154015/',
    '/mnt/ceph/users/agiovann/ImagingData/eyeblink/b37/20160624105838/',
    '/mnt/ceph/users/agiovann/ImagingData/eyeblink/b37/20160625132042/',
    '/mnt/ceph/users/agiovann/ImagingData/eyeblink/b37/20160626175708/',
    '/mnt/ceph/users/agiovann/ImagingData/eyeblink/b37/20160627110747/',
    '/mnt/ceph/users/agiovann/ImagingData/eyeblink/b37/20160628100247/',
    '/mnt/ceph/users/agiovann/ImagingData/eyeblink/b37/20160705103903/',

    '/mnt/ceph/users/agiovann/ImagingData/eyeblink/b37/20160628162522/',
    '/mnt/ceph/users/agiovann/ImagingData/eyeblink/b37/20160629123648/',
    '/mnt/ceph/users/agiovann/ImagingData/eyeblink/b37/20160630120544/',
    '/mnt/ceph/users/agiovann/ImagingData/eyeblink/b37/20160701113525/',
    '/mnt/ceph/users/agiovann/ImagingData/eyeblink/b37/20160702152950/',
    '/mnt/ceph/users/agiovann/ImagingData/eyeblink/b37/20160703173620/',
    '/mnt/ceph/users/agiovann/ImagingData/eyeblink/b37/20160704130454/',
    #              ]
    # error:               '/mnt/ceph/users/agiovann/ImagingData/eyeblink/b35/20160711104450/',
    #              '/mnt/ceph/users/agiovann/ImagingData/eyeblink/b35/20160712105933/',
    # base_folders=[
    '/mnt/ceph/users/agiovann/ImagingData/eyeblink/b35/20160710134627/',
    '/mnt/ceph/users/agiovann/ImagingData/eyeblink/b35/20160710193544/',
    '/mnt/ceph/users/agiovann/ImagingData/eyeblink/b35/20160711164154/',
    '/mnt/ceph/users/agiovann/ImagingData/eyeblink/b35/20160711212316/',
    '/mnt/ceph/users/agiovann/ImagingData/eyeblink/b35/20160712101950/',
    '/mnt/ceph/users/agiovann/ImagingData/eyeblink/b35/20160712173043/',
    '/mnt/ceph/users/agiovann/ImagingData/eyeblink/b35/20160713100916/',
    '/mnt/ceph/users/agiovann/ImagingData/eyeblink/b35/20160713171246/',
    '/mnt/ceph/users/agiovann/ImagingData/eyeblink/b35/20160714094320/',
    '/mnt/ceph/users/agiovann/ImagingData/eyeblink/b35/20160714143248/'
]

base_folders.sort()
print(base_folders)
#%%
if False:
    results = dview.map_sync(
        cm.use_cases.granule_cells.utils_granule.fast_process_day, base_folders)

    #% if this does not work look below
    triggers_chunk_fluo, eyelid_chunk, wheel_chunk, triggers_chunk_bh, tm_behav, names_chunks, fluo_chunk, pos_examples_chunks, A_chunks = process_fast_process_day(
        base_folders, save_name='eyeblink_35_37_sorted.npz')
#%%
#import re
#triggers_chunk_fluo = []
#eyelid_chunk = []
#wheel_chunk = []
#triggers_chunk_bh = []
# tm_behav=[]
# names_chunks=[]
# fluo_chunk=[]
# pos_examples_chunks=[]
#
# A_chunks=[]
# for base_folder in base_folders:
#    try:
#        print (base_folder)
#        with np.load(os.path.join(base_folder,'all_triggers.npz')) as ld:
#            triggers=ld['triggers']
#            trigger_names=ld['trigger_names']
#
#        with np.load(glob(os.path.join(base_folder,'*-template_total.npz'))[0]) as ld:
#            movie_names=ld['movie_names']
#            template_each=ld['template_each']
#
#
#        idx_chunks=[]
#        for name_chunk in movie_names:
#            idx_chunks.append([int(re.search('_00[0-9][0-9][0-9]_0',nm).group(0)[2:6])-1 for nm in name_chunk])
#
#
#
#        with np.load(base_folder+'behavioral_traces.npz') as ld:
#            res_bt = dict(**ld)
#            tm=res_bt['time']
#            f_rate_bh=1/np.median(np.diff(tm))
#            ISI=np.median([rs[3]-rs[2] for rs in res_bt['trial_info'][res_bt['idx_CS_US']]])
#            trig_int=np.hstack([((res_bt['trial_info'][:,2:4]-res_bt['trial_info'][:,0][:,None])*f_rate_bh),res_bt['trial_info'][:,-1][:,np.newaxis]]).astype(int)
#            trig_int[trig_int<0]=-1
#            trig_int=np.hstack([trig_int,len(tm)+trig_int[:,:1]*0])
#            trig_US=np.argmin(np.abs(tm))
#            trig_CS=np.argmin(np.abs(tm+ISI))
#            trig_int[res_bt['idx_CS_US'],0]=trig_CS
#            trig_int[res_bt['idx_CS_US'],1]=trig_US
#            trig_int[res_bt['idx_US'],1]=trig_US
#            trig_int[res_bt['idx_CS'],0]=trig_CS
#            eye_traces=np.array(res_bt['eyelid'])
#            wheel_traces=np.array(res_bt['wheel'])
#
#
#
#
#        fls=glob(os.path.join(base_folder,'*.results_analysis_traces.pk'))
#        fls.sort()
#        fls_m=glob(os.path.join(base_folder,'*.results_analysis_masks.npz'))
#        fls_m.sort()
#
#
#        for indxs,name_chunk,fl,fl_m in zip(idx_chunks,movie_names,fls,fls_m):
#            if np.all([nmc[:-4] for nmc in name_chunk] == trigger_names[indxs]):
#                triggers_chunk_fluo.append(triggers[indxs,:])
#                eyelid_chunk.append(eye_traces[indxs,:])
#                wheel_chunk.append(wheel_traces[indxs,:])
#                triggers_chunk_bh.append(trig_int[indxs,:])
#                tm_behav.append(tm)
#                names_chunks.append(fl)
#                with open(fl,'r') as f:
#                    tr_dict=pickle.load(f)
#                    print(fl)
#                    fluo_chunk.append(tr_dict['traces_DFF'])
#                with np.load(fl_m) as ld:
#                    A_chunks.append(scipy.sparse.coo_matrix(ld['A']))
#                    pos_examples_chunks.append(ld['pos_examples'])
#            else:
#                raise Exception('Names of triggers not matching!')
#    except:
#        print("ERROR in:"+base_folder)

#%%
with np.load('/mnt/ceph/users/agiovann/ImagingData/eyeblink/eyeblink_35_37.npz', encoding='latin1') as ld:
    print((list(ld.keys())))
    locals().update(ld)

idx_sorted = names_chunks.argsort()
names_chunks = names_chunks[idx_sorted]  # [:27]
fluo_chunk = fluo_chunk[idx_sorted]  # [:27]
eyelid_chunk = eyelid_chunk[idx_sorted]  # [:27]
triggers_chunk_bh = triggers_chunk_bh[idx_sorted]  # [:27]
tm_behav = tm_behav[idx_sorted]  # [:27]
A_chunks = A_chunks[idx_sorted]  # [:27]
wheel_chunk = wheel_chunk[idx_sorted]  # [:27]
triggers_chunk_fluo = triggers_chunk_fluo[idx_sorted]  # [:27]
pos_examples_chunks = pos_examples_chunks[idx_sorted]  # [:27]

#%%
# names_trials=[]
# last_dir=''
# for nms in names_chunks:
#    new_dir = os.path.dirname(nms)
#    if last_dir != new_dir:
#        print new_dir
#        last_dir = new_dir
#        templ_file=glob.glob(os.path.join())

#%%
talmo_file_name = ['/mnt/ceph/users/agiovann/ImagingData/eyeblink/b35.mat',
                   '/mnt/ceph/users/agiovann/ImagingData/eyeblink/b37.mat']


def unroll_Talmo_data(matr_list, is_nose=False):
    files = []
    for xx in matr_list:
        trials = []
        for xxx in xx[0]:
            if is_nose:

                trials.append(np.nanmean(xxx[0], 0))
                print((np.shape(xxx[0])))
            else:

                trials.append(xxx[0][0])

        files.append(trials)

    return files


exptNames_TM = []
trials_TM = []
timestamps_TM = []
wheel_mms_TM = []
nose_TM = []
nose_vel_TM = []
animal_TM = []
for tfile in talmo_file_name:
    ld = scipy.io.loadmat(tfile)
    print((list(ld.keys())))
    exptNames_TM = exptNames_TM + [xxp[0][0] for xxp in ld['exptNames']]
    animal_TM = animal_TM + [os.path.split(tfile)[-1][:-4]] * len(exptNames_TM)
    trials_TM = trials_TM + [xxp[0] for xxp in ld['trials']]
    timestamps_TM = timestamps_TM + unroll_Talmo_data(ld['timestamps'])
    wheel_mms_TM = wheel_mms_TM + unroll_Talmo_data(ld['wheel_mms'])
    nose_TM = nose_TM + unroll_Talmo_data(ld['nose'], is_nose=True)
    nose_vel_TM = nose_vel_TM + unroll_Talmo_data(ld['nose_vel'], is_nose=True)
#%%
tiff_names_chunks = []
sess_name_chunks = []
idx_trials_chunks = []
animal_chunks = []
timestamps_TM_chunk = []
wheel_mms_TM_chunk = []
nose_vel_TM_chunk = []
conversion_to_cm_sec = 10
scale_eye = .14
for tr_fl, tr_bh, eye, whe, tm, fl, nm, pos_examples, A in zip(triggers_chunk_fluo, triggers_chunk_bh, eyelid_chunk, wheel_chunk, tm_behav, fluo_chunk, names_chunks, pos_examples_chunks, A_chunks):
    print(nm)
    init, delta = (int(nm.split('/')[-1].split('#')[0][-12:-7]),
                   int(nm.split('/')[-1].split('#')[1][1:4].replace('_', '')))
    idx_names = np.arange(init, init + delta)
    fnames_behavior_talmo_tmp = []
    sess_name_chunks.append(nm.split('#')[0].split('/')[-2])
    animal_chunks.append(nm.split('#')[0].split('/')[-3])

    idx_sess = np.where(
        [sess_name_chunks[-1] == xp for xp in exptNames_TM])[0].squeeze()
    wheel_tmp = wheel_mms_TM[idx_sess]
    nose_tmp = nose_vel_TM[idx_sess]
    time_TM_tmp = timestamps_TM[idx_sess]
    tm_ = []
    wh_ = []
    ns_ = []
    for idx_name in idx_names:
        nm_tmp = nm.split('#')[0][:-10] + str(idx_name).zfill(3)
        target_file = glob(nm_tmp + '*.tif')
        if len(target_file) != 1:
            #               print len(target_file)
            raise Exception('Zero or too many matches')
        fnames_behavior_talmo_tmp.append(target_file)
        wh_.append(old_div(wheel_tmp[idx_name - 1], conversion_to_cm_sec))
#           print len(nose_tmp[idx_name-1])
        ns_.append(nose_tmp[idx_name - 1] / np.median(
            np.diff(time_TM_tmp[idx_name - 1])) * scale_eye / conversion_to_cm_sec)
        tm_.append(time_TM_tmp[idx_name - 1])
#           timestamps_TM_chunk_TMP.append()
#           wheel_mms_TM_chunk_TMP.append()

    if len(fnames_behavior_talmo_tmp) != len(tr_bh):
        raise Exception('Triggers not matching')

    tiff_names_chunks.append(fnames_behavior_talmo_tmp)
    idx_trials_chunks.append(idx_names)
    timestamps_TM_chunk.append(np.array(tm_))
    wheel_mms_TM_chunk.append(np.array(wh_))
    nose_vel_TM_chunk.append(np.array(ns_))
#%% define zones with big neurons
if False:
    min_pix = 20
    max_pix = 492
    good_neurons_chunk = []
    for fl, nm, pos_examples, A, tiff_names in zip(fluo_chunk, names_chunks, pos_examples_chunks, A_chunks, tiff_names_chunks):
        print(nm)
        idx_large_neurons = []
        with np.load(glob(os.path.join(os.path.dirname(nm), '*template*.npz'))[0]) as ld:
            pl.cla()
            img = np.median(ld['template_each'], 0)
        masks = cse.utilities.nf_read_roi_zip(os.path.join(
            os.path.dirname(nm), 'RoiSet.zip'), (512, 512))
        for mask in masks:
            mask_flat = np.reshape(mask, (512 * 512), order='F')
            idx_large_neurons = idx_large_neurons + \
                list(np.where(A.T.dot(mask_flat))[0])

        idx_large_neurons = np.unique(idx_large_neurons)

        f_perc = np.array([np.percentile(f, 90, 1) for f in fl])
        too_active_neurons = np.where(np.mean(f_perc, 0) >= 4)[0]
    #        cb.movie(img,fr=1).save(os.path.join(os.path.dirname(nm),'template_large_neurons.tif'))
    #    lq,hq=np.percentile(img,[1, 99])
    #    pl.imshow(img,cmap='gray',vmin=lq,vmax=hq)
    #    pl.imshow(np.sum(masks,0),cmap='hot',alpha=.3,vmax=3)
        fl_tot = np.concatenate(fl, 1)
        fitness, erfc = cse.utilities.compute_event_exceptionality(fl_tot)
        good_traces = np.where(fitness < -10)[0]

        coms = cse.utilities.com(A.todense(), 512, 512)
        border_neurons = np.unique(np.concatenate([np.where(coms[:, 0] > max_pix)[0],
                                                   np.where(coms[:, 0] < min_pix)[
            0],
            np.where(coms[:, 1] > max_pix)[
            0],
            np.where(coms[:, 1] < min_pix)[0]]))
        final_examples = np.setdiff1d(pos_examples, border_neurons)
        final_examples = np.setdiff1d(final_examples, idx_large_neurons)
        final_examples = np.setdiff1d(final_examples, too_active_neurons)
        final_examples = np.intersect1d(final_examples, good_traces)
        good_neurons_chunk.append(final_examples)
        print((len(final_examples), np.shape(A)[-1]))
        img = A.tocsc()[:, final_examples].sum(
            1).reshape((512, 512), order='F')
        lq, hq = np.percentile(img, [1, 99])
        pl.imshow(img, cmap='gray', alpha=1, vmin=lq, vmax=hq)
        pl.pause(.01)

    np.savez('good_neurons_chunk.npz', good_neurons_chunk=good_neurons_chunk)
else:
    with np.load('/mnt/ceph/users/agiovann/ImagingData/eyeblink/good_neurons_chunk.npz') as ld:
        good_neurons_chunk = ld['good_neurons_chunk']
#%%
#triggers_chunk_fluo=  triggers_chunk_fluo[idx_sorted]
#triggers_chunk_bh=  triggers_chunk_bh[idx_sorted]
#eyelid_chunk=  eyelid_chunk[idx_sorted]
#wheel_chunk=  wheel_chunk[idx_sorted]
#tm_behav=  tm_behav[idx_sorted]
#fluo_chunk=  fluo_chunk[idx_sorted]
#pos_examples_chunks=  pos_examples_chunks[idx_sorted]
#A_chunks=  A_chunks[idx_sorted]
#%% ** PARAMS ****
thresh_middle = .15
thresh_advanced = .4
thresh_late = .8
time_CR_on = -.1
time_US_on = .035
thresh_mov = 2
# thresh_MOV_iqr=100
time_CS_on_MOV = -.25
time_US_on_MOV = 0
thresh_CR = 0.1,
thresh_CR_whisk = 0.15
threshold_responsiveness = 0.1
time_bef = 2.9
time_aft = 4.5
f_rate_fluo = old_div(1, 30.0)
ISI = .25
min_trials = 8  # was 8

thresh_wheel = 5
thresh_nose = 1
thresh_eye = .1
thresh_fluo_log = -10
#%% compute all binned behavior
all_nose_binned = []
all_wheel_binned = []
all_binned_eye = []
all_binned_UR = []
all_binned_CR = []
mouse_now = ''

for tr_fl, tr_bh, eye, whe, tm, fl, nm, pos_examples, A, tiff_names, timestamps_TM_, wheel_mms_TM_, nose_vel_TM_ in\
        zip(triggers_chunk_fluo, triggers_chunk_bh, eyelid_chunk, wheel_chunk, tm_behav, fluo_chunk, names_chunks, good_neurons_chunk, A_chunks, tiff_names_chunks, timestamps_TM_chunk, wheel_mms_TM_chunk, nose_vel_TM_chunk):

    session = nm.split('/')[8]
    day = nm.split('/')[8][:8]
    print(nm)
    mouse = nm.split('/')[7]
    if mouse != mouse_now:
        cell_counter = 0
        mouse_now = mouse
        session_id = 0
        session_now = ''
        learning_phase = 0
        print('early')
    else:
        if day != session_now:
            session_id += 1
            session_now = day

    chunk = re.search('_00[0-9][0-9][0-9]_', nm.split('/')[9]).group(0)[3:-1]

    idx_CS_US = np.where(tr_bh[:, -2] == 2)[0]
    idx_US = np.where(tr_bh[:, -2] == 1)[0]
    idx_CS = np.where(tr_bh[:, -2] == 0)[0]
    idx_ALL = np.sort(np.hstack([idx_CS_US, idx_US, idx_CS]))
    eye_traces, amplitudes_at_US, trig_CRs = process_eyelid_traces(
        eye, tm, idx_CS_US, idx_US, idx_CS, thresh_CR=thresh_CR, time_CR_on=time_CR_on, time_US_on=time_US_on)
    idxCSUSCR = trig_CRs['idxCSUSCR']
    idxCSUSNOCR = trig_CRs['idxCSUSNOCR']
    idxCSCR = trig_CRs['idxCSCR']
    idxCSNOCR = trig_CRs['idxCSNOCR']
    idxNOCR = trig_CRs['idxNOCR']
    idxCR = trig_CRs['idxCR']
    idxUS = trig_CRs['idxUS']
    idxCSCSUS = np.concatenate([idx_CS, idx_CS_US])

    if 0:
        # ANDREA'S WHEEL
        wheel_traces, movement_at_CS, trigs_mov = process_wheel_traces(np.array(
            whe), tm, thresh_MOV_iqr=thresh_MOV_iqr, time_CS_on=time_CS_on_MOV, time_US_on=time_US_on_MOV)

    else:
        # TALMO'S WHEEL
        wheel_traces, movement_at_CS, trigs_mov = process_wheel_traces_talmo(wheel_mms_TM_, timestamps_TM_.copy(
        ), tm, thresh_MOV=thresh_mov, time_CS_on=time_CS_on_MOV, time_US_on=time_US_on_MOV)
        nose_traces, movement_at_CS_nose, trigs_mov_nose = process_wheel_traces_talmo(
            nose_vel_TM_, timestamps_TM_.copy(), tm, thresh_MOV=0, time_CS_on=time_CS_on_MOV, time_US_on=time_US_on_MOV)

        n_samples_ISI = int(old_div(ISI, np.median(np.diff(tm))))
        # FIX FOR US SHIFT IN DATA
        wheel_traces[idxUS] = np.concatenate([wheel_traces[idxUS, :n_samples_ISI].copy(
        ), wheel_traces[idxUS, :-n_samples_ISI].copy()], 1)
        nose_traces[idxUS] = np.concatenate([nose_traces[idxUS, :n_samples_ISI].copy(
        ), nose_traces[idxUS, :-n_samples_ISI].copy()], 1)

    print(('fraction with movement:' + str(len(trigs_mov['idxMOV']) * 1. / (
        len(trigs_mov['idxNO_MOV']) + len(trigs_mov['idxNO_MOV'])))))

    mn_idx_CS_US = np.intersect1d(idx_CS_US, trigs_mov['idxNO_MOV'])
    nm_idx_US = idx_US

    nm_idx_CS = np.intersect1d(idx_CS, trigs_mov['idxNO_MOV'])
    nm_idxCSUSCR = np.intersect1d(idxCSUSCR, trigs_mov['idxNO_MOV'])
    nm_idxCSUSNOCR = np.intersect1d(idxCSUSNOCR, trigs_mov['idxNO_MOV'])
    nm_idxCSCR = np.intersect1d(idxCSCR, trigs_mov['idxNO_MOV'])
    nm_idxCSNOCR = np.intersect1d(idxCSNOCR, trigs_mov['idxNO_MOV'])
    nm_idxNOCR = np.intersect1d(idxNOCR, trigs_mov['idxNO_MOV'])
    nm_idxCR = np.intersect1d(idxCR, trigs_mov['idxNO_MOV'])
    nm_idxCSCSUS = np.intersect1d(idxCSCSUS, trigs_mov['idxNO_MOV'])

    trial_names = [''] * wheel_traces.shape[0]

    for CSUSNoCR in nm_idxCSUSNOCR:
        trial_names[CSUSNoCR] = 'CSUSNoCR'
    for CSUSwCR in nm_idxCSUSCR:
        trial_names[CSUSwCR] = 'CSUSwCR'
    for US in nm_idx_US:
        trial_names[US] = 'US'
    for CSwCR in nm_idxCSCR:
        trial_names[CSwCR] = 'CSwCR'
    for CSNoCR in nm_idxCSNOCR:
        trial_names[CSNoCR] = 'CSNoCR'

    len_min = np.min([np.array(f).shape for f in fl]
                     )  # minimal length of trial

    def selct(cs, us): return int(cs) if np.isnan(us) else int(us)
    # index of US
    trigs_US = [selct(cs, us) for cs, us in zip(tr_fl[:, 0], tr_fl[:, 1])]

    # for binning
    samplbef = int(old_div(time_bef, f_rate_fluo))
    samplaft = int(old_div(time_aft, f_rate_fluo))

    # create fluorescence matrix
    f_flat = np.concatenate([f[:, tr - samplbef:samplaft + tr]
                             for tr, f in zip(trigs_US, fl)], 1)
    f_mat = np.concatenate([f[:, tr - samplbef:samplaft + tr]
                            [np.newaxis, :] for tr, f in zip(trigs_US, fl)], 0)
    time_fl = np.arange(-samplbef, samplaft) * f_rate_fluo
    # remove baseline
    f_mat_bl = f_mat - np.median(f_mat[:, :, np.logical_and(
        time_fl > -1, time_fl < -ISI)], axis=(2))[:, :, np.newaxis]

    amplitudes_responses = np.mean(
        f_mat_bl[:, :, np.logical_and(time_fl > -.03, time_fl < .04)], -1)

    cell_responsiveness = np.median(amplitudes_responses[nm_idxCSCSUS], axis=0)
    idx_responsive = np.where(cell_responsiveness >
                              threshold_responsiveness)[0]
    fraction_responsive = len(np.where(cell_responsiveness > threshold_responsiveness)[
                              0]) * 1. / np.shape(f_mat_bl)[1]
#    a=pd.DataFrame(data=f_mat[0,idx_components[:10],:],columns=np.arange(-30,30)*.033,index=idx_components[:10])

    # only take the preselected components
    idx_components_final = pos_examples
    idx_responsive = np.intersect1d(idx_responsive, idx_components_final)

    cell_counter = cell_counter + len(idx_components_final)
    print(cell_counter)
    f_mat_bl_part = f_mat[:, idx_components_final, :].copy()

    # compute periods of activity for each neuron
    f_mat_bl_erfc = f_mat_bl_part.transpose([1, 0, 2]).reshape(
        (-1, np.shape(f_mat_bl_part)[0] * np.shape(f_mat_bl_part)[-1]))
    f_mat_bl_erfc[np.isnan(f_mat_bl_erfc)] = 0
    fitness, f_mat_bl_erfc, _, _ = compute_event_exceptionality(f_mat_bl_erfc)
    f_mat_bl_erfc = f_mat_bl_erfc.reshape(
        [-1, np.shape(f_mat_bl_part)[0], np.shape(f_mat_bl_part)[-1]]).transpose([1, 0, 2])

    bin_edge = np.arange(-3, 4.5, .1)

    bins = pd.cut(time_fl, bins=bin_edge)
    bins_tm = pd.cut(tm, bins=bin_edge)

    time_bef_edge = -0.25
    time_aft_edge = 1.5
    min_r = -1.1  # 0
    min_confidence = 100  # .01

    idx_good_samples = np.where(np.logical_or(
        time_fl <= time_bef_edge, time_fl >= time_aft_edge))[0]
    idx_good_samples_tm = np.where(np.logical_or(
        tm <= time_bef_edge, tm >= time_aft_edge))[0]

    time_ds_idx = np.where(np.logical_or(
        bin_edge <= time_bef_edge, bin_edge >= time_aft_edge))[0][1:] - 1

    dfs = [pd.DataFrame(f_mat_bl_part[:, ii, :].T, index=time_fl)
           for ii in range(np.shape(f_mat_bl_part)[1])]
    binned_fluo = np.array([df.groupby(bins).mean().values.T for df in dfs])[
        :, :, time_ds_idx].squeeze()
    binned_fluo[np.isnan(binned_fluo)] = 0

    dfs = [pd.DataFrame(wheel_traces[ii], index=tm)
           for ii in range(np.shape(wheel_traces)[0])]
    binned_wheel = np.array(
        [df.groupby(bins_tm).mean().values.squeeze() for df in dfs])[:, time_ds_idx]
    binned_wheel[np.isnan(binned_wheel)] = 0

    dfs = [pd.DataFrame(eye_traces[ii], index=tm)
           for ii in range(np.shape(eye_traces)[0])]
    binned_eye = np.array(
        [df.groupby(bins_tm).mean().values.squeeze() for df in dfs])[:, time_ds_idx]
    binned_eye[np.isnan(binned_eye)] = 0

    dfs = [pd.DataFrame(nose_traces[ii], index=tm)
           for ii in range(np.shape(nose_traces)[0])]
    binned_nose = np.array(
        [df.groupby(bins_tm).mean().values.squeeze() for df in dfs])[:, time_ds_idx]
    binned_nose[np.isnan(binned_nose)] = 0

    dfs = [pd.DataFrame(f_mat_bl_erfc[:, ii, :].T, index=time_fl)
           for ii in range(np.shape(f_mat_bl_erfc)[1])]
    binned_fluo_erfc = np.array([df.groupby(bins).mean().values.T for df in dfs])[
        :, :, time_ds_idx].squeeze()
    binned_fluo_erfc[np.isnan(binned_fluo_erfc)] = 0

    all_nose_binned.append(binned_nose)
    all_wheel_binned.append(binned_wheel)
    all_binned_eye.append(binned_eye)

    amplitudes_responses_US_fl = f_mat_bl[idx_US, :, :, ][:, idx_components_final, :, ][:, :, np.logical_and(
        time_fl > 0.03, time_fl < .75)]
    ampl_UR_eye = scipy.signal.resample(eye_traces[idx_US, :][:, np.logical_and(
        tm > .03, tm < .75)], np.shape(amplitudes_responses_US_fl)[-1], axis=1)

    all_binned_UR.append(ampl_UR_eye)

    amplitudes_responses_CR_fl = f_mat_bl[idxCSCSUS, :, :, ][:, idx_components_final, :, ][:, :, np.logical_and(
        time_fl > -0.5, time_fl < 0.03)]
    ampl_CR_eye = scipy.signal.resample(eye_traces[idxCSCSUS, :][:, np.logical_and(
        tm > -.5, tm < .03)], np.shape(amplitudes_responses_CR_fl)[-1], axis=1)

    all_binned_CR.append(ampl_CR_eye)

#%%

mat_all_nose = np.concatenate(all_nose_binned, 0)
mat_all_wheel = np.concatenate(all_wheel_binned, 0)
mat_all_eye = np.concatenate(all_binned_eye, 0)
mat_all_UR = np.concatenate(all_binned_UR, 0)
mat_all_CR = np.concatenate(all_binned_CR, 0)

pl.plot(np.mean(mat_all_nose, 0))
pl.plot(np.mean(mat_all_wheel, 0))
pl.plot(np.mean(mat_all_eye, 0))
pl.plot(np.mean(mat_all_UR, 0))
pl.plot(np.mean(mat_all_CR, 0))

#%% define functions to perform evaluation
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score, ShuffleSplit
from sklearn import linear_model


def my_custom_loss_func(ground_truth, predictions):
    r = scipy.stats.pearsonr(ground_truth, predictions)
    return r[0]


score_ = make_scorer(my_custom_loss_func, greater_is_better=True)

lr = linear_model.LinearRegression()
# lr.fit(X,Y)
# scipy.stats.pearsonr(Y.squeeze(),lr.predict(X).squeeze())
#scipy.stats.pearsonr(Y.squeeze(),cross_val_predict(lr, X, Y, cv=None).squeeze())
#predicted = cross_val_predict(lr, X, Y, cv=20)
#%% **********


def func_avg_fluo(X, *args, **kwargs):
    return np.nanmedian(X, *args, **kwargs)


#%%
mouse_now = ''
session_now = ''
session_id = 0
compute_redundancy = False
check_timing = False
redundancy = []
max_fluo_range = np.inf
#single_session = True

cr_ampl = pd.DataFrame()
bh_correl = pd.DataFrame()
all_nose = []
all_wheel = []
session_nice_trials = ['/mnt/ceph/users/agiovann/ImagingData/eyeblink/b35/20160714143248/20160714143248_00061_00001-#-51_d1_512_d2_512_d3_1_order_C_frames_3535_.results_analysis_traces.pk',
                       '/mnt/ceph/users/agiovann/ImagingData/eyeblink/b37/20160629123648/20160629123648_00061_00001-#-40_d1_512_d2_512_d3_1_order_C_frames_2780_.results_analysis_traces.pk',
                       '/mnt/ceph/users/agiovann/ImagingData/eyeblink/b37/20160630120544/20160630120544_00121_00001-#-58_d1_512_d2_512_d3_1_order_C_frames_4029_.results_analysis_traces.pk',
                       '/mnt/ceph/users/agiovann/ImagingData/eyeblink/b37/20160701113525/20160701113525_00151_00001-#-39_d1_512_d2_512_d3_1_order_C_frames_2713_.results_analysis_traces.pk',
                       '/mnt/ceph/users/agiovann/ImagingData/eyeblink/b37/20160702152950/20160702152950_00151_00001-#-45_d1_512_d2_512_d3_1_order_C_frames_3125_.results_analysis_traces.pk',
                       '/mnt/ceph/users/agiovann/ImagingData/eyeblink/b37/20160703173620/20160703173620_00121_00001-#-50_d1_512_d2_512_d3_1_order_C_frames_3479_.results_analysis_traces.pk',
                       '/mnt/ceph/users/agiovann/ImagingData/eyeblink/b37/20160704130454/20160704130454_00151_00001-#-40_d1_512_d2_512_d3_1_order_C_frames_2791_.results_analysis_traces.pk',
                       '/mnt/ceph/users/agiovann/ImagingData/eyeblink/b37/20160705103903/20160705103903_00121_00001-#-57_d1_512_d2_512_d3_1_order_C_frames_3977_.results_analysis_traces.pk']
# session_nice_trials=[]
cell_counter = []
for tr_fl, tr_bh, eye, whe, tm, fl, nm, pos_examples, A, tiff_names, timestamps_TM_, wheel_mms_TM_, nose_vel_TM_ in\
    zip(triggers_chunk_fluo, triggers_chunk_bh, eyelid_chunk, wheel_chunk,
        tm_behav, fluo_chunk, names_chunks, good_neurons_chunk, A_chunks, tiff_names_chunks,
        timestamps_TM_chunk, wheel_mms_TM_chunk, nose_vel_TM_chunk):
    if nm not in session_nice_trials[-1]:
        print(nm)
        continue
#        1

    if len(whe) < 20:
        print('skipping small files')
        continue

    session = nm.split('/')[8]
    day = nm.split('/')[8][:8]
    print(nm)
    mouse = nm.split('/')[7]
    if mouse != mouse_now:
        cell_counter = 0
        mouse_now = mouse
        session_id = 0
        session_now = ''
        learning_phase = 0
        print('early')
    else:
        if day != session_now:
            session_id += 1
            session_now = day

    chunk = re.search('_00[0-9][0-9][0-9]_', nm.split('/')[9]).group(0)[3:-1]

    idx_CS_US = np.where(tr_bh[:, -2] == 2)[0]
    idx_US = np.where(tr_bh[:, -2] == 1)[0]
    idx_CS = np.where(tr_bh[:, -2] == 0)[0]
    idx_ALL = np.sort(np.hstack([idx_CS_US, idx_US, idx_CS]))
    eye_traces, amplitudes_at_US, trig_CRs = process_eyelid_traces(
        eye, tm, idx_CS_US, idx_US, idx_CS, thresh_CR=thresh_CR, time_CR_on=time_CR_on, time_US_on=time_US_on)
    idxCSUSCR = trig_CRs['idxCSUSCR']
    idxCSUSNOCR = trig_CRs['idxCSUSNOCR']
    idxCSCR = trig_CRs['idxCSCR']
    idxCSNOCR = trig_CRs['idxCSNOCR']
    idxNOCR = trig_CRs['idxNOCR']
    idxCR = trig_CRs['idxCR']
    idxUS = trig_CRs['idxUS']
    idxCSCSUS = np.concatenate([idx_CS, idx_CS_US])

    if 0:
        # ANDREA'S WHEEL
        wheel_traces, movement_at_CS, trigs_mov = process_wheel_traces(np.array(
            whe), tm, thresh_MOV_iqr=thresh_MOV_iqr, time_CS_on=time_CS_on_MOV, time_US_on=time_US_on_MOV)

    else:
        # TALMO'S WHEEL
        wheel_traces, movement_at_CS, trigs_mov = process_wheel_traces_talmo(wheel_mms_TM_, timestamps_TM_.copy(
        ), tm, thresh_MOV=thresh_mov, time_CS_on=time_CS_on_MOV, time_US_on=time_US_on_MOV)
        nose_traces, movement_at_CS_nose, trigs_mov_nose = process_wheel_traces_talmo(
            nose_vel_TM_, timestamps_TM_.copy(), tm, thresh_MOV=0, time_CS_on=time_CS_on_MOV, time_US_on=time_US_on_MOV)

        n_samples_ISI = int(old_div(ISI, np.median(np.diff(tm))))
        # FIX FOR US SHIFT IN DATA
        wheel_traces[idxUS] = np.concatenate([wheel_traces[idxUS, :n_samples_ISI].copy(
        ), wheel_traces[idxUS, :-n_samples_ISI].copy()], 1)
        nose_traces[idxUS] = np.concatenate([nose_traces[idxUS, :n_samples_ISI].copy(
        ), nose_traces[idxUS, :-n_samples_ISI].copy()], 1)

    print(('fraction with movement:' + str(len(trigs_mov['idxMOV']) * 1. / (
        len(trigs_mov['idxNO_MOV']) + len(trigs_mov['idxNO_MOV'])))))

    fraction_movement = len(
        trigs_mov['idxMOV']) * 1. / (len(trigs_mov['idxNO_MOV']) + len(trigs_mov['idxNO_MOV']))

    mn_idx_CS_US = np.intersect1d(idx_CS_US, trigs_mov['idxNO_MOV'])
    nm_idx_US = idx_US

    nm_idx_CS = np.intersect1d(idx_CS, trigs_mov['idxNO_MOV'])
    nm_idxCSUSCR = np.intersect1d(idxCSUSCR, trigs_mov['idxNO_MOV'])
    nm_idxCSUSNOCR = np.intersect1d(idxCSUSNOCR, trigs_mov['idxNO_MOV'])
    nm_idxCSCR = np.intersect1d(idxCSCR, trigs_mov['idxNO_MOV'])
    nm_idxCSNOCR = np.intersect1d(idxCSNOCR, trigs_mov['idxNO_MOV'])
    nm_idxNOCR = np.intersect1d(idxNOCR, trigs_mov['idxNO_MOV'])
    nm_idxCR = np.intersect1d(idxCR, trigs_mov['idxNO_MOV'])
    nm_idxCSCSUS = np.intersect1d(idxCSCSUS, trigs_mov['idxNO_MOV'])

    trial_names = [''] * wheel_traces.shape[0]

    for CSUSNoCR in nm_idxCSUSNOCR:
        trial_names[CSUSNoCR] = 'CSUSNoCR'
    for CSUSwCR in nm_idxCSUSCR:
        trial_names[CSUSwCR] = 'CSUSwCR'
    for US in nm_idx_US:
        trial_names[US] = 'US'
    for CSwCR in nm_idxCSCR:
        trial_names[CSwCR] = 'CSwCR'
    for CSNoCR in nm_idxCSNOCR:
        trial_names[CSNoCR] = 'CSNoCR'

    len_min = np.min([np.array(f).shape for f in fl]
                     )  # minimal length of trial

    def selct(cs, us): return int(cs) if np.isnan(us) else int(us)
    # index of US
    trigs_US = [selct(cs, us) for cs, us in zip(tr_fl[:, 0], tr_fl[:, 1])]

    # for binning
    samplbef = int(old_div(time_bef, f_rate_fluo))
    samplaft = int(old_div(time_aft, f_rate_fluo))

    # create fluorescence matrix
    f_flat = np.concatenate([f[:, tr - samplbef:samplaft + tr]
                             for tr, f in zip(trigs_US, fl)], 1)
    f_mat = np.concatenate([f[:, tr - samplbef:samplaft + tr]
                            [np.newaxis, :] for tr, f in zip(trigs_US, fl)], 0)
    time_fl = np.arange(-samplbef, samplaft) * f_rate_fluo
    # remove baseline
    f_mat_bl = f_mat - np.median(f_mat[:, :, np.logical_and(
        time_fl > -1, time_fl < -ISI)], axis=(2))[:, :, np.newaxis]

    amplitudes_responses = func_avg_fluo(
        f_mat_bl[:, :, np.logical_and(time_fl > -.03, time_fl < .04)], -1)

#    if  session=='20160628162522':
#        raise Exception('right trial')
#    else:
#        continue
    cell_responsiveness = np.median(amplitudes_responses[nm_idxCSCSUS], axis=0)
    idx_responsive = np.where(cell_responsiveness >
                              threshold_responsiveness)[0]
    fraction_responsive = len(np.where(cell_responsiveness > threshold_responsiveness)[
                              0]) * 1. / np.shape(f_mat_bl)[1]
#    a=pd.DataFrame(data=f_mat[0,idx_components[:10],:],columns=np.arange(-30,30)*.033,index=idx_components[:10])
    ampl_CR = pd.DataFrame()
    bh_correl_tmp = pd.DataFrame()

    # only take the preselected components
#    idx_components_final=pos_examples
    idx_components_final = np.intersect1d(pos_examples, np.where(
        (np.max(f_flat, 1) - np.min(f_flat, 1)) < max_fluo_range)[0])

    idx_responsive = np.intersect1d(idx_responsive, idx_components_final)

    bh_correl_tmp['neuron_id'] = cell_counter + \
        np.arange(len(idx_components_final))
    cell_counter = cell_counter + len(idx_components_final)
    print(cell_counter)

    f_mat_bl_part = f_mat[:, idx_components_final, :].copy()

    # compute periods of activity for each neuron
    f_mat_bl_erfc = f_mat_bl_part.transpose([1, 0, 2]).reshape(
        (-1, np.shape(f_mat_bl_part)[0] * np.shape(f_mat_bl_part)[-1]))
    f_mat_bl_erfc[np.isnan(f_mat_bl_erfc)] = 0
    fitness, f_mat_bl_erfc, _, _ = compute_event_exceptionality(f_mat_bl_erfc)
    f_mat_bl_erfc = f_mat_bl_erfc.reshape(
        [-1, np.shape(f_mat_bl_part)[0], np.shape(f_mat_bl_part)[-1]]).transpose([1, 0, 2])

    if check_timing:  # compute timing lag granule eyelid

        bin_edge = np.arange(-3, 4.5, .034)

        bins = pd.cut(time_fl, bins=bin_edge)
        bins_tm = pd.cut(tm, bins=bin_edge)

        time_bef_edge = 0
        time_aft_edge = 0

    else:

        bin_edge = np.arange(-3, 4.5, .1)

        bins = pd.cut(time_fl, bins=bin_edge)
        bins_tm = pd.cut(tm, bins=bin_edge)

        time_bef_edge = -0.25
        time_aft_edge = 1.5

    min_r = -1.1  # 0
    min_confidence = 100  # .01

    idx_good_samples = np.where(np.logical_or(
        time_fl <= time_bef_edge, time_fl >= time_aft_edge))[0]
    idx_good_samples_tm = np.where(np.logical_or(
        tm <= time_bef_edge, tm >= time_aft_edge))[0]

    time_ds_idx = np.where(np.logical_or(
        bin_edge <= time_bef_edge, bin_edge >= time_aft_edge))[0][1:] - 1

    dfs = [pd.DataFrame(f_mat_bl_part[:, ii, :].T, index=time_fl)
           for ii in range(np.shape(f_mat_bl_part)[1])]
    binned_fluo = np.array([df.groupby(bins).mean().values.T for df in dfs])[
        :, :, time_ds_idx].squeeze()
    binned_fluo[np.isnan(binned_fluo)] = 0

    dfs = [pd.DataFrame(wheel_traces[ii], index=tm)
           for ii in range(np.shape(wheel_traces)[0])]
    binned_wheel = np.array(
        [df.groupby(bins_tm).mean().values.squeeze() for df in dfs])[:, time_ds_idx]
    binned_wheel[np.isnan(binned_wheel)] = 0

    dfs = [pd.DataFrame(eye_traces[ii], index=tm)
           for ii in range(np.shape(eye_traces)[0])]
    binned_eye = np.array(
        [df.groupby(bins_tm).mean().values.squeeze() for df in dfs])[:, time_ds_idx]
    binned_eye[np.isnan(binned_eye)] = 0

    dfs = [pd.DataFrame(nose_traces[ii], index=tm)
           for ii in range(np.shape(nose_traces)[0])]
    binned_nose = np.array(
        [df.groupby(bins_tm).mean().values.squeeze() for df in dfs])[:, time_ds_idx]
    binned_nose[np.isnan(binned_nose)] = 0

    dfs = [pd.DataFrame(f_mat_bl_erfc[:, ii, :].T, index=time_fl)
           for ii in range(np.shape(f_mat_bl_erfc)[1])]
    binned_fluo_erfc = np.array([df.groupby(bins).mean().values.T for df in dfs])[
        :, :, time_ds_idx].squeeze()
    binned_fluo_erfc[np.isnan(binned_fluo_erfc)] = 0

    bh_correl_tmp['active_during_nose'] = np.sum((binned_nose > thresh_nose) * (
        binned_fluo_erfc[:, :, :] < thresh_fluo_log), (1, 2)) * 1. / np.sum(binned_nose > thresh_nose)
    bh_correl_tmp['active_during_wheel'] = np.sum((binned_wheel > thresh_wheel) * (
        binned_fluo_erfc[:, :, :] < thresh_fluo_log), (1, 2)) * 1. / np.sum(binned_wheel > thresh_wheel)
    bh_correl_tmp['active_during_eye'] = np.sum((binned_eye > thresh_eye) * (
        binned_fluo_erfc[:, :, :] < thresh_fluo_log), (1, 2)) * 1. / np.sum(binned_eye > thresh_eye)

    bh_correl_tmp['avg_activity_during_locomotion'] = func_avg_fluo(
        binned_fluo[:, binned_nose > thresh_nose], 1)

    r_nose_fluo = [scipy.stats.pearsonr(
        binned_nose.flatten(), bf.flatten()) for bf in binned_fluo]
    r_nose_fluo = np.array(
        [rnf[0] if rnf[1] < min_confidence and rnf[0] > min_r else None for rnf in r_nose_fluo])

    r_wheel_fluo = [scipy.stats.pearsonr(
        binned_wheel.flatten(), bf.flatten()) for bf in binned_fluo]
    r_wheel_fluo = np.array(
        [rnf[0] if rnf[1] < min_confidence and rnf[0] > min_r else None for rnf in r_wheel_fluo])

    r_eye_fluo = [scipy.stats.pearsonr(
        binned_eye.flatten(), bf.flatten()) for bf in binned_fluo]
    r_eye_fluo = np.array(
        [rnf[0] if rnf[1] < min_confidence and rnf[0] > min_r else None for rnf in r_eye_fluo])

    bh_correl_tmp['r_nose_fluo'] = r_nose_fluo
    bh_correl_tmp['r_wheel_fluo'] = r_wheel_fluo
    bh_correl_tmp['r_eye_fluo'] = r_eye_fluo


#    if not check_timing:
    mat_rnd = mat_all_nose[np.random.permutation(np.shape(mat_all_nose)[0])[
        :np.shape(binned_nose)[1]]]
#        r_nose_fluo_rnd=[scipy.stats.pearsonr(binned_nose[np.random.permutation(np.shape(binned_nose)[0])].flatten(),bf.flatten()) for bf in binned_fluo]
    r_nose_fluo_rnd = [scipy.stats.pearsonr(
        mat_rnd.flatten(), bf.flatten()) for bf in binned_fluo]
    r_nose_fluo_rnd = np.array(
        [rnf[0] if rnf[1] < min_confidence and rnf[0] > min_r else None for rnf in r_nose_fluo_rnd])

    mat_rnd = mat_all_wheel[np.random.permutation(np.shape(mat_all_wheel)[0])[
        :np.shape(binned_wheel)[1]]]
#        r_nose_fluo_rnd=[scipy.stats.pearsonr(binned_nose[np.random.permutation(np.shape(binned_nose)[0])].flatten(),bf.flatten()) for bf in binned_fluo]
    r_wheel_fluo_rnd = [scipy.stats.pearsonr(
        mat_rnd.flatten(), bf.flatten()) for bf in binned_fluo]
    r_wheel_fluo_rnd = np.array(
        [rnf[0] if rnf[1] < min_confidence and rnf[0] > min_r else None for rnf in r_wheel_fluo_rnd])

    mat_rnd = mat_all_eye[np.random.permutation(np.shape(mat_all_eye)[0])[
        :np.shape(binned_eye)[1]]]
#        r_nose_fluo_rnd=[scipy.stats.pearsonr(binned_nose[np.random.permutation(np.shape(binned_nose)[0])].flatten(),bf.flatten()) for bf in binned_fluo]
    r_eye_fluo_rnd = [scipy.stats.pearsonr(
        mat_rnd.flatten(), bf.flatten()) for bf in binned_fluo]
    r_eye_fluo_rnd = np.array(
        [rnf[0] if rnf[1] < min_confidence and rnf[0] > min_r else None for rnf in r_eye_fluo_rnd])

    bh_correl_tmp['r_wheel_fluo_rnd'] = r_wheel_fluo_rnd
    bh_correl_tmp['r_nose_fluo_rnd'] = r_nose_fluo_rnd

    bh_correl_tmp['r_eye_fluo_rnd'] = r_eye_fluo_rnd

    fluo_crpl = func_avg_fluo(
        amplitudes_responses[idxCR, :][:, idx_responsive], 0)

    fluo_crmn = func_avg_fluo(
        amplitudes_responses[idxNOCR, :][:, idx_responsive], 0)

    amplitudes_responses_US_fl = f_mat_bl[idx_US, :, :, ][:, idx_components_final, :, ][:, :, np.logical_and(
        time_fl > 0.03, time_fl < .75)]
    ampl_UR_eye = scipy.signal.resample(eye_traces[idx_US, :][:, np.logical_and(
        tm > .03, tm < .75)], np.shape(amplitudes_responses_US_fl)[-1], axis=1)
    r_UR_fluo = [scipy.stats.pearsonr(bf.flatten(), ampl_UR_eye.flatten(
    )) for bf in amplitudes_responses_US_fl.transpose([1, 0, 2])]
    r_UR_fluo = np.array(
        [rnf[0] if rnf[1] < min_confidence and rnf[0] > min_r else None for rnf in r_UR_fluo])

    mat_rnd = mat_all_UR[np.random.permutation(np.shape(mat_all_UR)[0])[
        :np.shape(amplitudes_responses_US_fl)[0]]]

    r_UR_fluo_rnd = [scipy.stats.pearsonr(bf.flatten(), mat_rnd.flatten(
    )) for bf in amplitudes_responses_US_fl.transpose([1, 0, 2])]
#        r_UR_fluo_rnd=[scipy.stats.pearsonr(bf.flatten(),ampl_UR_eye.flatten()[np.random.permutation(np.shape(ampl_UR_eye.flatten())[0])]) for bf in amplitudes_responses_US_fl.transpose([1,0,2])]
    r_UR_fluo_rnd = np.array(
        [rnf[0] if rnf[1] < min_confidence and rnf[0] > min_r else None for rnf in r_UR_fluo_rnd])

    bh_correl_tmp['r_UR_eye_fluo'] = r_UR_fluo
    bh_correl_tmp['r_UR_eye_fluo_rnd'] = r_UR_fluo_rnd

#        if np.any([r<-.6 and r is not None for r in r_wheel_fluo]):
#            raise Exception

    amplitudes_responses_CR_fl = f_mat_bl[idxCSCSUS, :, :, ][:, idx_components_final, :, ][:, :, np.logical_and(
        time_fl > -0.5, time_fl < 0.03)]
    ampl_CR_eye = scipy.signal.resample(eye_traces[idxCSCSUS, :][:, np.logical_and(
        tm > -.5, tm < .03)], np.shape(amplitudes_responses_CR_fl)[-1], axis=1)
    CR_eye_fluo = [scipy.stats.pearsonr(bf.flatten(), ampl_CR_eye.flatten(
    )) for bf in amplitudes_responses_CR_fl.transpose([1, 0, 2])]
    CR_eye_fluo = np.array(
        [rnf[0] if rnf[1] < min_confidence and rnf[0] > min_r else None for rnf in CR_eye_fluo])
    mat_rnd = mat_all_CR[np.random.permutation(np.shape(mat_all_CR)[0])[
        :np.shape(amplitudes_responses_CR_fl)[0]]]
    CR_eye_fluo_rnd = np.array([scipy.stats.pearsonr(bf.flatten(), mat_rnd.flatten(
    )) for bf in amplitudes_responses_CR_fl.transpose([1, 0, 2])])
#        CR_eye_fluo_rnd=[scipy.stats.pearsonr(bf.flatten(),ampl_CR_eye.flatten()[np.random.permutation(np.shape(ampl_CR_eye.flatten())[0])]) for bf in amplitudes_responses_CR_fl.transpose([1,0,2])]
    CR_eye_fluo_rnd = [rnf[0] if rnf[1] < min_confidence and rnf[0]
                       > min_r else None for rnf in CR_eye_fluo_rnd]

    if check_timing:  # plot nice delay binned
        #            pl.plot(bin_edge[idx_good_samples[1:-1]],binned_fluo[100].mean(0)-.2)
        min_r_CR = .3
        time_binned_data = bin_edge[idx_good_samples[1:-1]]
        binned_fluo_norm = old_div((binned_fluo - np.percentile(binned_fluo[..., 5:50], 50, (1, 2))[
                                   :, np.newaxis, np.newaxis]), np.std(binned_fluo[..., 5:50], (1, 2))[:, np.newaxis, np.newaxis])
        binned_fluo_norm = np.mean(binned_fluo_norm[:, idxCR, :], 1)
        binned_fluo_norm = old_div((binned_fluo_norm - np.percentile(binned_fluo_norm[..., 5:50], 50, (-1))[
                                   :, np.newaxis]), np.std(binned_fluo_norm[..., 5:50], (-1))[:, np.newaxis])
        binned_eye_norm = old_div((binned_eye - np.percentile(binned_eye[..., 5:50], 50, (-1))[
                                  :, np.newaxis]), np.std(binned_eye[..., 5:50], (-1))[:, np.newaxis])
        binned_eye_norm = binned_eye_norm[idxCR].mean(0)
        binned_eye_norm = binned_eye_norm - np.percentile(binned_eye_norm[np.logical_and(
            time_binned_data[:] > -1, time_binned_data[:] < -.25)], 50)

        binned_nose_norm = old_div((binned_nose - np.percentile(binned_nose[..., 5:50], 50, (-1))[
                                   :, np.newaxis]), np.std(binned_nose[..., 5:50], (-1))[:, np.newaxis])
        binned_nose_norm = binned_nose_norm[idxCR].mean(0)
        binned_nose_norm = binned_nose_norm - \
            np.percentile(binned_nose_norm[np.logical_and(
                time_binned_data[:] > -1, time_binned_data[:] < -.25)], 50)

        idx_during_cs = np.where(
            (time_binned_data > -0.22) & (time_binned_data < 0.03))[0]
        idx_max_eye = np.where(binned_eye_norm[idx_during_cs] > 3)[0][0]
        idx_max_nose = np.where(binned_nose_norm[idx_during_cs] > 2)[0][0]

        pl.subplot(1, 2, 1)
        ax = pl.gca()
        l1 = pl.plot(
            time_binned_data[:], binned_fluo_norm[CR_eye_fluo > min_r_CR][:, :].T, color='lightgray')
        l2 = pl.plot(time_binned_data[:], binned_eye_norm[:],
                     'k', linewidth=2, label='Inline label')
        l3 = pl.plot(time_binned_data[:], binned_nose_norm[:],
                     'r', linewidth=2, label='Inline label')
        l1[0].set_label('Granule cells with r_CR>0.3')
        l2[0].set_label('Eyelid')
        l3[0].set_label('Nose')
        pl.legend()
        pl.xlabel('Time to US (s)')
        pl.ylabel('z-score')
        pl.xlim([-.4, 0.03])
        pl.ylim([-3, 30])
        # find fr each trial and neuron the first index larger than 1 std
        bins_sign_fluo = np.array([np.where(binned_fluo_norm[iid_neuro][idx_during_cs] > 3)[0][0] if np.max(
            binned_fluo_norm[iid_neuro][idx_during_cs]) > 3 else np.nan for iid_neuro in range(np.shape(binned_fluo_norm)[0])])

#            bins_sign_fluo=[bisect.bisect(binned_fluo_norm[iid_neuro][idx_during_cs],3) for iid_neuro in range(np.shape(binned_fluo_norm)[0])]
        bins_hist = time_binned_data[np.logical_and(
            time_binned_data[:] > -.25, time_binned_data[:] < .25)]
        pl.subplot(1, 2, 2)
        pl.hist(0.034 * (idx_max_eye - np.array(bins_sign_fluo)[np.logical_and(
            CR_eye_fluo > min_r_CR, ~np.isnan(bins_sign_fluo))]), bins=bins_hist)
        pl.hist(0.034 * (idx_max_nose - np.array(bins_sign_fluo)[np.logical_and(
            CR_eye_fluo > min_r_CR, ~np.isnan(bins_sign_fluo))]), bins=bins_hist)

        pl.xlabel('time lag  (s)')
        pl.ylabel('Cell count N=' + str(np.sum([CR_eye_fluo > min_r_CR])))
        pl.xlim([-.1, .25])
        pl.legend(['granule - eyelid (>3 std)', 'granule - snout (>2 std)'])

        # pl.ylim([3,8])

    else:

        print("skipping lags")

    if compute_redundancy:

        _trials_, _n_nr, _timesteps_ = amplitudes_responses_CR_fl.shape
        X = np.reshape(amplitudes_responses_CR_fl.transpose(
            [1, 0, 2]), [_n_nr, _trials_ * _timesteps_]).T
        Y = ampl_CR_eye.flatten()
        CR_eye_fluo = np.array(CR_eye_fluo, dtype=float)
        idx_significant = np.where(
            CR_eye_fluo > np.percentile(CR_eye_fluo_rnd, 95))[0]
        ls = sklearn.linear_model.LassoCV(n_alphas=100, eps=1e-3)

#            np.nanmean(cross_val_score(ls, X, Y, cv=ShuffleSplit(10,test_size=.5),scoring=score_))
#            scores = cross_val_score(lr, X, Y, cv=ShuffleSplit(30,test_size=.5),scoring=score_)

        # [idxHighCorr][::-1]
        r_neurons = np.array(CR_eye_fluo[idx_significant], dtype=float)
        X = X[:, idx_significant]
        ls.fit(X, Y)

        print((np.sum(ls.coef_ > 0)))

        print((len(idx_significant)))
        print((np.mean(CR_eye_fluo[idx_significant])))
        print((np.std(CR_eye_fluo[idx_significant])))
        print((np.max(CR_eye_fluo[idx_significant])))
        print((np.median(CR_eye_fluo[idx_significant])))
        print('*')
        print((np.mean(CR_eye_fluo)))
        print((np.std(CR_eye_fluo)))
        print((np.max(CR_eye_fluo)))
        print((np.median(CR_eye_fluo)))

        r_classif_naive = cross_val_score(
            lr, X, Y, cv=ShuffleSplit(100, test_size=.5), scoring=score_)
        r_classif = np.mean(r_classif_naive)

        print((np.mean(r_classif)))

        r_classif_lasso = cross_val_score(
            lr, X[:, ls.coef_ > 0], Y, cv=ShuffleSplit(100, test_size=.5), scoring=score_)
        r_classif = np.mean(r_classif_lasso)
        r_neurons = r_neurons[ls.coef_ > 0]

#            r_classif=[np.mean(cross_val_score(lr, X[:,:iidd+1], Y, cv=ShuffleSplit(30,test_size=.5),scoring=score_)) for iidd in range(_n_nr)][::-1]

        print((np.mean(r_classif)))

        # r_classif=np.array(r_classif).T
        idx_good = np.where((r_neurons != np.array(None))
                            & (~np.isnan(r_neurons)))[0]
        # r_classif=r_classif[idx_good]
        r_neurons = r_neurons[idx_good]

        N = len(r_neurons)
        I_neurons = -0.5 * np.log(1 - r_neurons**2) / np.log(2)
        I_classif = -0.5 * np.log(1 - r_classif**2) / np.log(2)
        Icum_neurons = np.sum(I_neurons)
#            I_classif=I_classif[::-1]
        redundancy.append(old_div(np.max(Icum_neurons), np.max(I_classif)))
        print(('Redundancy:' + str(old_div(np.max(Icum_neurons), np.max(I_classif)))))
        print((len(I_neurons)))

    else:

        print('SKIPPED REDUNDANCY')

    if 0:  # for plotting  example redundancy use Ag0515

        scores = cross_val_score(lr, X, Y, cv=ShuffleSplit(
            100, test_size=.5), scoring=score_)
        vls, bns, _ = pl.hist(CR_eye_fluo, bins=np.arange(-.1, 1, .05))
        pl.cla()
        vl_distr = scipy.signal.savgol_filter(vls, 1, 0)
        vl_distr = old_div(vl_distr, np.sum(vl_distr))
        pl.plot(bns[1:], vl_distr, 'r')
        lq, hq = np.percentile(CR_eye_fluo_rnd, [5, 95])
        pl.fill_between([lq, hq], [.35, .35], alpha=.3, color='c')
        lq, hq = np.percentile(scores, [5, 95])
        pl.fill_between([lq, hq], [.35, .35], alpha=.3, color=[.8, .8, .8])
        pl.plot([np.mean(scores), np.mean(scores)], [0, .35], 'k--')
        pl.xlabel('Person\'s r')
        pl.ylabel('Probability')

        pl.figure()

        pl.plot(Icum_neurons, '-r')
        pl.plot(I_classif, '-k')

        pl.xlabel('Number of neurons')
        pl.legend(['sum of individual GrCs', 'classifier'])
        pl.ylabel('Cumulative information (bits)')
        pl.xlim([-1, N - 1])
        pl.ylim([0, 1.5])
        pl.title('Redundancy calculation')

    bh_correl_tmp['r_CR_eye_fluo'] = CR_eye_fluo
    bh_correl_tmp['r_CR_eye_fluo_rnd'] = CR_eye_fluo_rnd

    if len(idxUS) > 4:
        bh_correl_tmp['active_during_UR'] = np.mean(np.min(
            f_mat_bl_erfc[idxUS, :, :][:, :, np.logical_and(time_fl > .15, time_fl < .3)], -1) < thresh_fluo_log, 0)
    else:
        print('** NOT ENOUGH TRIALS **')
        bh_correl_tmp['active_during_UR'] = np.nan

    all_idx_neg = np.union1d(idxUS, idxNOCR)
    if len(all_idx_neg) > min_trials:
        bh_correl_tmp['active_during_UR_NOCR'] = np.mean(np.min(
            f_mat_bl_erfc[all_idx_neg, :, :][:, :, np.logical_and(time_fl > .15, time_fl < .3)], -1) < thresh_fluo_log, 0)
    else:
        print('** NOT ENOUGH TRIALS **')
        bh_correl_tmp['active_during_UR_NOCR'] = np.nan

    bh_correl_tmp['active_during_CS'] = np.mean(np.min(f_mat_bl_erfc[idxNOCR, :, :][:, :, np.logical_and(
        time_fl > -.05, time_fl < -.03)], -1) < thresh_fluo_log, 0)

    if len(idxCR) > min_trials:
        bh_correl_tmp['active_during_CR'] = np.mean(np.min(f_mat_bl_erfc[idxCR, :, :][:, :, np.logical_and(
            time_fl > -.05, time_fl < -.03)], -1) < thresh_fluo_log, 0)
    else:
        bh_correl_tmp['active_during_CR'] = np.nan

    if learning_phase > 1 and binned_nose.shape[0] > 30 and False:
        #        session_nice_trials.append(nm)

        ax1 = pl.subplot(2, 1, 1)
        pl.cla()
        ax1.plot(scipy.signal.savgol_filter(
            binned_nose.flatten() * 5, 3, 1) - .25)
        ax1.plot(scipy.signal.savgol_filter(
            old_div(binned_wheel.flatten(), 10) + 2, 5, 1))
        ax1.plot(scipy.signal.savgol_filter(binned_eye.flatten() + 4, 5, 1))

        pl.legend(['nose', 'wheel', 'eye'])
        ax2 = pl.subplot(2, 1, 2, sharex=ax1)
        pl.cla()

        ax2.plot(scipy.signal.savgol_filter(binned_fluo[np.argsort(
            r_nose_fluo)[-6:-1:2]].reshape((3, -1)).T, 7, 1, axis=0))
        ax2.plot(scipy.signal.savgol_filter(binned_fluo[np.argsort(
            r_wheel_fluo)[-6:-1:2]].reshape((3, -1)).T + 2, 7, 1, axis=0))
        ax2.plot(scipy.signal.savgol_filter(binned_fluo[np.argsort(
            r_eye_fluo)[-6:-1:2]].reshape((3, -1)).T + 4, 7, 1, axis=0))
        ax2.plot(scipy.signal.savgol_filter(binned_fluo[np.argsort(
            CR_eye_fluo)[-6:-1:2]].reshape((3, -1)).T + 6, 7, 1, axis=0))
        ax2.plot(scipy.signal.savgol_filter(binned_fluo[np.argsort(
            r_UR_fluo)[-6:-1:2]].reshape((3, -1)).T + 8, 7, 1, axis=0))

        pl.pause(10)

    fluo_crpl_all = func_avg_fluo(
        amplitudes_responses[idxCR, :][:, idx_components_final], 0)
    fluo_crmn_all = func_avg_fluo(
        amplitudes_responses[idxNOCR, :][:, idx_components_final], 0)
    fluo_crpl = func_avg_fluo(
        amplitudes_responses[idxCR, :][:, idx_responsive], 0)
    fluo_crmn = func_avg_fluo(
        amplitudes_responses[idxNOCR, :][:, idx_responsive], 0)

#    ampl_no_CR=pd.DataFrame(np.median(amplitudes_responses[idxNOCR,:][:,idx_components_final],0))
    if len(nm_idxCR) > min_trials:

        ampl_CR['fluo_plus'] = fluo_crpl
        ampl_CR['ampl_eyelid_CR'] = func_avg_fluo(amplitudes_at_US[nm_idxCR])
        bh_correl_tmp['fluo_plus'] = fluo_crpl_all
        bh_correl_tmp['ampl_eyelid_CR'] = func_avg_fluo(
            amplitudes_at_US[nm_idxCR])
    else:

        ampl_CR['fluo_plus'] = np.nan
        ampl_CR['ampl_eyelid_CR'] = np.nan
        bh_correl_tmp['fluo_plus'] = np.nan
        bh_correl_tmp['ampl_eyelid_CR'] = np.nan

    if len(nm_idxNOCR) > min_trials:
        ampl_CR['fluo_minus'] = fluo_crmn
        bh_correl_tmp['fluo_minus'] = fluo_crmn_all
    else:
        ampl_CR['fluo_minus'] = np.nan
        bh_correl_tmp['fluo_minus'] = np.nan

    ampl_CR['session'] = session
    ampl_CR['mouse'] = mouse
    ampl_CR['chunk'] = chunk

    bh_correl_tmp['session'] = session
    bh_correl_tmp['mouse'] = mouse
    bh_correl_tmp['chunk'] = chunk

    bh_correl_tmp['num_trials'] = len(whe)
    ampl_CR['num_trials'] = len(whe)

    bh_correl_tmp['ampl_FOV'] = 512
    bh_correl_tmp['fluo_range'] = (
        np.max(f_flat, 1) - np.min(f_flat, 1))[idx_components_final]
    bh_correl_tmp['idx_components_final'] = idx_components_final
    bh_correl_tmp['perc_CR'] = len(nm_idxCR) * 1. / len(nm_idxCSCSUS)
    bh_correl_tmp['fraction_movement'] = fraction_movement

    ampl_CR['ampl_FOV'] = 500 * 500
    ampl_CR['perc_CR'] = len(nm_idxCR) * 1. / len(nm_idxCSCSUS)

    ampl_CR['fluo_range_responsive'] = (
        np.max(f_flat, 1) - np.min(f_flat, 1))[idx_responsive]
#    ampl_CR['fluo_range_final']=(np.max(f_flat,1)-np.min(f_flat,1))[idx_components_final]
    ampl_CR['idx_component'] = idx_responsive
#    ampl_CR['idx_components_final']=idx_components_final

    if len(nm_idxCR) * 1. / len(nm_idxCSCSUS) > thresh_middle and learning_phase == 0:
        learning_phase = 1
        print('middle')
    elif len(nm_idxCR) * 1. / len(nm_idxCSCSUS) > thresh_advanced and learning_phase == 1:
        learning_phase = 2
        print('advanced')
    elif len(nm_idxCR) * 1. / len(nm_idxCSCSUS) > thresh_late and learning_phase == 2:
        learning_phase = 3
        print('late')

    ampl_CR['learning_phase'] = learning_phase
    ampl_CR['ampl_eyelid_CSCSUS'] = func_avg_fluo(
        amplitudes_at_US[nm_idxCSCSUS])
    ampl_CR['session_id'] = session_id
    cr_ampl = pd.concat([cr_ampl, ampl_CR])

    bh_correl_tmp['learning_phase'] = learning_phase
    bh_correl_tmp['ampl_eyelid_CSCSUS'] = func_avg_fluo(
        amplitudes_at_US[nm_idxCSCSUS])
    bh_correl_tmp['session_id'] = session_id
    bh_correl = pd.concat([bh_correl, bh_correl_tmp])

#

#%


if False:
    thresh_middle = .05
    thresh_advanced = .35
    thresh_late = .9
    time_CR_on = -.1
    time_US_on = .035
    thresh_mov = 2
    # thresh_MOV_iqr=100
    time_CS_on_MOV = -.25
    time_US_on_MOV = 0
    thresh_CR = 0.1,
    threshold_responsiveness = 0.1
    time_bef = 2.9
    time_aft = 4.5
    f_rate_fluo = old_div(1, 30.0)
    ISI = .25
    min_trials = 8

    mouse_now = ''
    session_now = ''
    session_id = 0

    thresh_wheel = 5
    thresh_nose = 1
    thresh_eye = .1
    thresh_fluo_log = -10

    #single_session = True

    cr_ampl = pd.DataFrame()
    bh_correl = pd.DataFrame()

bh_correl_old = bh_correl.copy()
cr_ampl_old = cr_ampl.copy()
#%%
last_session = False
#%
mat_summaries = [
    '/mnt/xfs1/home/agiovann/imaging/eyeblink/MAT_SUMMARIES/gc-AGGC6f-031213-03/python_out.mat',
    '/mnt/xfs1/home/agiovann/imaging/eyeblink/MAT_SUMMARIES/gc-AG052014-02/python_out.mat',
    '/mnt/xfs1/home/agiovann/imaging/eyeblink/MAT_SUMMARIES/AG052014-01/python_out.mat',
    '/mnt/xfs1/home/agiovann/imaging/eyeblink/MAT_SUMMARIES/AG051514-01/python_out.mat']
for mat_summary in mat_summaries[:]:
    ld = scipy.io.loadmat(mat_summary)
    cr_ampl_dic = dict()
    cr_ampl_dic['trials'] = np.array([a[0][0][0] for a in ld['python_trials']])
    cr_ampl_dic['trialsTypeOrig'] = [css[0][0]
                                     for css in ld['python_trialsTypeOrig']]
    cr_ampl_dic['trialName'] = [css[0][0] for css in ld['python_trialName']]
    cr_ampl_dic['session'] = [css[0][0] for css in ld['python_session']]
    cr_ampl_dic['animal'] = [css[0][0] for css in ld['python_animal']]
    cr_ampl_dic['day'] = [css[0][0] for css in ld['python_day']]
    cr_ampl_dic['realDay'] = [css[0][0] for css in ld['python_realDay']]

    mat_time = np.array([css[0] for css in ld['python_time']])
    mat_wheel = np.array([css for css in ld['python_wheel']])
    mat_eyelid = np.array([css for css in ld['python_eyelid']])

    if 'python_trials_talmo' in ld:

        cr_ampl_dic['trials_talmo'] = [css[0][0]
                                       for css in ld['python_trials_talmo']]
        cr_ampl_dic['nose_talmo'] = [css[0][0]
                                     for css in ld['python_nose_talmo']]
        cr_ampl_dic['timestamps_talmo'] = [css[0][0]
                                           for css in ld['python_timestamps_talmo']]
        cr_ampl_dic['exptNames_talmo'] = [css[0][0]
                                          for css in ld['python_exptNames_talmo']]
        mat_wheel_talmo = np.array(
            [css for css in ld['python_wheel_cms_talmo']])
        mat_wheel_talmo = np.abs(mat_wheel_talmo)
        mat_nose_talmo = np.array([css for css in ld['python_nose_vel_talmo']])
        wheel_ampl_while_CS = np.nanmax(
            mat_wheel_talmo[np.logical_and(mat_time > -ISI, mat_time < time_US_on), :], 0)
        mat_nose_talmo = mat_nose_talmo * .12 / conversion_to_cm_sec
        print((np.nanmean(mat_nose_talmo)))
    else:
        print('********   Talmo Behavior Not found *******')
        cr_ampl_dic['nose_talmo'] = None
        cr_ampl_dic['timestamps_talmo'] = None
        cr_ampl_dic['exptNames_talmo'] = None
        mat_wheel_talmo = None
        mat_wheel_talmo = None
        mat_nose_talmo = None
        wheel_ampl_while_CS = np.nanmax(mat_wheel[np.logical_and(
            mat_time > -ISI, mat_time < time_US_on), :], 0) * 0

    mat_ampl_at_US = np.nanmedian(mat_eyelid[np.logical_and(
        mat_time > -.05, mat_time < time_US_on), :], 0)

    mat_fluo = np.concatenate([np.atleast_3d(css)
                               for css in ld['python_fluo_traces']], -1)
    if (mouse_now == 'AG052014-01') or (mouse_now == 'gc-AG052014-02'):
        mat_idxCR_orig = np.where([np.logical_and(t in ['CSUS', 'CS'], ampl >= thresh_CR_whisk)
                                   for t, ampl in zip(cr_ampl_dic['trialsTypeOrig'], mat_ampl_at_US)])[0]
        mat_idxNOCR_orig = np.where([np.logical_and(t in ['CSUS', 'CS'], ampl < thresh_CR_whisk)
                                     for t, ampl in zip(cr_ampl_dic['trialsTypeOrig'], mat_ampl_at_US)])[0]

    else:

        mat_idxCR_orig = np.where([np.logical_and(t in ['CSUS', 'CS'], ampl >= thresh_CR)
                                   for t, ampl in zip(cr_ampl_dic['trialsTypeOrig'], mat_ampl_at_US)])[0]
        mat_idxNOCR_orig = np.where([np.logical_and(t in ['CSUS', 'CS'], ampl < thresh_CR)
                                     for t, ampl in zip(cr_ampl_dic['trialsTypeOrig'], mat_ampl_at_US)])[0]

    mat_idxCSCSUS_orig = np.union1d(mat_idxCR_orig, mat_idxNOCR_orig)
    mat_idxUS_orig = np.where([t in ['US']
                               for t in cr_ampl_dic['trialsTypeOrig']])[0]
    idx_no_mov = np.where(wheel_ampl_while_CS <= thresh_mov)[0]

    print(('******************************** Fraction with movement:' +
           str(1 - len(idx_no_mov) * 1. / len(wheel_ampl_while_CS))))
    fraction_movement = str(1 - len(idx_no_mov) * 1. /
                            len(wheel_ampl_while_CS))

    mat_idxCR = np.intersect1d(mat_idxCR_orig, idx_no_mov)
    mat_idxNOCR = np.intersect1d(mat_idxNOCR_orig, idx_no_mov)
    mat_idxUS = np.intersect1d(mat_idxUS_orig, idx_no_mov)
    mat_idxCSCSUS = np.intersect1d(mat_idxCSCSUS_orig, idx_no_mov)

    mouse_now = ''
    session_now = ''
    sess_, order_, _, _ = np.unique(
        cr_ampl_dic['session'], return_index=True, return_inverse=True, return_counts=True)
    sess_ = sess_[np.argsort(order_)]
    idx_neurons = np.arange(mat_fluo.shape[1])
    mouse = cr_ampl_dic['animal'][0]
    # cr_ampl=pd.DataFrame()
    if ~last_session:

        idx_CR = mat_idxCR_orig
        idx_NOCR = mat_idxNOCR_orig
        idx_US = mat_idxUS_orig
        idxCSCSUS = mat_idxCSCSUS_orig
        idx_sess = list(range(mat_fluo.shape[0]))

    else:

        ss = cr_ampl_dic['realDay'][-1]
        idx_sess = np.where([item == ss for item in cr_ampl_dic['realDay']])[0]
        print(ss)

        session = cr_ampl_dic['day'][idx_sess[0]]
        day = cr_ampl_dic['realDay'][idx_sess[0]]

        session_id = len(np.unique(cr_ampl_dic['realDay']))

        learning_phase = np.nan

        idx_CR = np.intersect1d(idx_sess, mat_idxCR_orig)
        idx_NOCR = np.intersect1d(idx_sess, mat_idxNOCR_orig)
        idx_US = np.intersect1d(idx_sess, mat_idxUS_orig)
        idxCSCSUS = np.intersect1d(idx_sess, mat_idxCSCSUS_orig)

    bh_correl_tmp = pd.DataFrame()
    bh_correl_tmp['neuron_id'] = np.arange(len(idx_neurons))
    if 1:
        f_mat_bl_part = mat_fluo[idx_sess].copy()

        f_mat_bl_erfc = mat_fluo.transpose([1, 0, 2]).reshape(
            (-1, np.shape(mat_fluo)[0] * np.shape(mat_fluo)[-1]))
        f_mat_bl_erfc[np.isnan(f_mat_bl_erfc)] = 0
        fitness, f_mat_bl_erfc, _, _ = compute_event_exceptionality(
            f_mat_bl_erfc)
        f_mat_bl_erfc = f_mat_bl_erfc.reshape(-1, np.shape(
            mat_fluo)[0], np.shape(mat_fluo)[-1]).transpose([1, 0, 2])

        if check_timing:

            bin_edge = np.arange(mat_time[0], mat_time[-1], .064)

            bins = pd.cut(mat_time, bins=bin_edge)

            time_bef_edge = 0
            time_aft_edge = 0

        else:

            bin_edge = np.arange(mat_time[0], mat_time[-1], .1)

            bins = pd.cut(mat_time, bins=bin_edge)

            time_bef_edge = -0.4
            time_aft_edge = 1.7

        mat_eyelid_part = mat_eyelid[:, idx_sess]
        # choose samples
        idx_good_samples = np.where(np.logical_or(
            mat_time <= time_bef_edge, mat_time >= time_aft_edge))[0]

        time_ds_idx = np.where(np.logical_or(
            bin_edge <= time_bef_edge, bin_edge >= time_aft_edge))[0][1:] - 1

        dfs = [pd.DataFrame(f_mat_bl_part[:, ii, :].T, index=mat_time)
               for ii in range(np.shape(f_mat_bl_part)[1])]
        binned_fluo = np.array([df.groupby(bins).mean().values.T for df in dfs])[
            :, :, time_ds_idx].squeeze()
        binned_fluo[np.isnan(binned_fluo)] = 0

        dfs = [pd.DataFrame(f_mat_bl_erfc[:, ii, :].T, index=mat_time)
               for ii in range(np.shape(f_mat_bl_erfc)[1])]
        binned_fluo_erfc = np.array([df.groupby(bins).mean().values.T for df in dfs])[
            :, :, time_ds_idx].squeeze()

        dfs = [pd.DataFrame(mat_eyelid_part.T[ii], index=mat_time)
               for ii in range(np.shape(mat_eyelid_part.T)[0])]
        binned_eye = np.array([df.groupby(bins).mean().values.squeeze() for df in dfs])[
            :, time_ds_idx]
        binned_eye[np.isnan(binned_eye)] = 0
        r_eye_fluo = [scipy.stats.pearsonr(
            binned_eye.flatten(), bf.flatten()) for bf in binned_fluo]
        r_eye_fluo = [rnf[0] if rnf[1] < min_confidence and rnf[0]
                      > min_r else None for rnf in r_eye_fluo]
        bh_correl_tmp['r_eye_fluo'] = r_eye_fluo

        r_eye_fluo_rnd = [scipy.stats.pearsonr(binned_eye[np.random.permutation(
            np.shape(binned_eye)[0])].flatten(), bf.flatten()) for bf in binned_fluo]
        r_eye_fluo_rnd = [rnf[0] if rnf[1] < min_confidence and rnf[0]
                          > min_r else None for rnf in r_eye_fluo_rnd]
        bh_correl_tmp['r_eye_fluo_rnd'] = r_eye_fluo_rnd

        if mat_nose_talmo is not None:
            mat_nose_talmo_part = mat_nose_talmo[:, idx_sess]
            dfs = [pd.DataFrame(mat_nose_talmo_part.T[ii], index=mat_time)
                   for ii in range(np.shape(mat_nose_talmo_part.T)[0])]
            binned_nose = np.array([df.groupby(bins).mean().values.squeeze() for df in dfs])[
                :, time_ds_idx]
            binned_nose[np.isnan(binned_nose)] = 0
            r_nose_fluo = [scipy.stats.pearsonr(
                binned_nose.flatten(), bf.flatten()) for bf in binned_fluo]
            r_nose_fluo = [rnf[0] if rnf[1] < min_confidence and rnf[0]
                           > min_r else None for rnf in r_nose_fluo]
            bh_correl_tmp['r_nose_fluo'] = r_nose_fluo

            r_nose_fluo_rnd = [scipy.stats.pearsonr(binned_nose[np.random.permutation(
                np.shape(binned_nose)[0])].flatten(), bf.flatten()) for bf in binned_fluo]
            r_nose_fluo_rnd = [rnf[0] if rnf[1] < min_confidence and rnf[0]
                               > min_r else None for rnf in r_nose_fluo_rnd]
            bh_correl_tmp['r_nose_fluo_rnd'] = r_nose_fluo_rnd

            bh_correl_tmp['active_during_nose'] = np.sum((binned_nose > thresh_nose) * (
                binned_fluo_erfc[:, idx_sess, :] < thresh_fluo_log), (1, 2)) * 1. / np.sum(binned_nose > thresh_nose)

        else:

            bh_correl_tmp['r_nose_fluo'] = np.nan
            bh_correl_tmp['r_nose_fluo_rnd'] = np.nan
            bh_correl_tmp['active_during_nose'] = np.nan

        if mat_wheel_talmo is not None:
            mat_wheel_talmo_part = mat_wheel_talmo[:, idx_sess]
            dfs = [pd.DataFrame(mat_wheel_talmo_part.T[ii], index=mat_time)
                   for ii in range(np.shape(mat_wheel_talmo_part.T)[0])]

            binned_wheel = np.array(
                [df.groupby(bins).mean().values.squeeze() for df in dfs])[:, time_ds_idx]
            binned_wheel[np.isnan(binned_wheel)] = 0
            r_wheel_fluo = [scipy.stats.pearsonr(
                binned_wheel.flatten(), bf.flatten()) for bf in binned_fluo]
            r_wheel_fluo = [rnf[0] if rnf[1] < min_confidence and rnf[0]
                            > min_r else None for rnf in r_wheel_fluo]
            bh_correl_tmp['r_wheel_fluo'] = r_wheel_fluo

            r_wheel_fluo_rnd = [scipy.stats.pearsonr(binned_wheel[np.random.permutation(
                np.shape(binned_wheel)[0])].flatten(), bf.flatten()) for bf in binned_fluo]
            r_wheel_fluo_rnd = [rnf[0] if rnf[1] < min_confidence and rnf[0]
                                > min_r else None for rnf in r_wheel_fluo_rnd]
            bh_correl_tmp['r_wheel_fluo_rnd'] = r_wheel_fluo_rnd

            bh_correl_tmp['active_during_wheel'] = np.sum((binned_wheel > thresh_wheel) * (
                binned_fluo_erfc[:, idx_sess, :] < thresh_fluo_log), (1, 2)) * 1. / np.sum(binned_wheel > thresh_wheel)
            bh_correl_tmp['avg_activity_during_locomotion'] = func_avg_fluo(
                binned_fluo[:, binned_nose > thresh_nose], 1)

            if np.any([r < -.5 and r is not None for r in r_wheel_fluo]):
                raise Exception

        else:

            bh_correl_tmp['r_wheel_fluo'] = np.nan
            bh_correl_tmp['r_wheel_fluo_rnd'] = np.nan
            bh_correl_tmp['active_during_wheel'] = np.nan

        bh_correl_tmp['active_during_eye'] = np.sum((binned_eye > thresh_eye) * (
            binned_fluo_erfc[:, idx_sess, :] < thresh_fluo_log), (1, 2)) * 1. / np.sum(binned_eye > thresh_eye)

        amplitudes_responses_US_fl = mat_fluo[idxUS, :, :, ][:, :, :, ][:, :, np.logical_and(
            mat_time > 0.03, mat_time < .75)]
        ampl_UR_eye = mat_eyelid.T[idxUS, :][:, np.logical_and(
            mat_time > .03, mat_time < .75)]
        r_UR_fluo = [scipy.stats.pearsonr(bf.flatten(), ampl_UR_eye.flatten(
        )) for bf in amplitudes_responses_US_fl.transpose([1, 0, 2])]
        r_UR_fluo = np.array(
            [rnf[0] if rnf[1] < min_confidence and rnf[0] > min_r else None for rnf in r_UR_fluo])
        r_UR_fluo_rnd = [scipy.stats.pearsonr(bf.flatten(), ampl_UR_eye.flatten()[np.random.permutation(
            np.shape(ampl_UR_eye.flatten())[0])]) for bf in amplitudes_responses_US_fl.transpose([1, 0, 2])]
        r_UR_fluo_rnd = np.array(
            [rnf[0] if rnf[1] < min_confidence and rnf[0] > min_r else None for rnf in r_UR_fluo_rnd])

        bh_correl_tmp['r_UR_eye_fluo'] = r_UR_fluo
        bh_correl_tmp['r_UR_eye_fluo_rnd'] = r_UR_fluo_rnd

        amplitudes_responses_CR_fl = mat_fluo[idxCSCSUS, :, :, ][:, :, :, ][:, :, np.logical_and(
            mat_time > -0.5, mat_time < 0.03)]
        ampl_CR_eye = mat_eyelid.T[idxCSCSUS, :][:,
                                                 np.logical_and(mat_time > -.5, mat_time < .03)]
        CR_eye_fluo = [scipy.stats.pearsonr(bf.flatten(), ampl_CR_eye.flatten(
        )) for bf in amplitudes_responses_CR_fl.transpose([1, 0, 2])]
        CR_eye_fluo = np.array(
            [rnf[0] if rnf[1] < min_confidence and rnf[0] > min_r else None for rnf in CR_eye_fluo])
        CR_eye_fluo_rnd = [scipy.stats.pearsonr(bf.flatten(), ampl_CR_eye.flatten()[np.random.permutation(
            np.shape(ampl_CR_eye.flatten())[0])]) for bf in amplitudes_responses_CR_fl.transpose([1, 0, 2])]
        CR_eye_fluo_rnd = np.array(
            [rnf[0] if rnf[1] < min_confidence and rnf[0] > min_r else None for rnf in CR_eye_fluo_rnd])

        if check_timing:  # plot nic edelay binned
            #            pl.plot(bin_edge[idx_good_samples[1:-1]],binned_fluo[100].mean(0)-.2)
            #            pl.plot(bin_edge[idx_good_samples[1:-1]],binned_eye.mean(0))
            min_r_CR = 0.3
            # [(bin_edge>time_bef_edge) & (bin_edge<time_aft_edge)]
            time_binned_data = bin_edge[1:]

#                time_bl_idx=np.where((time_binned_data>-.2) & (time_binned_data<-.45)
            binned_fluo_norm = old_div((binned_fluo - np.percentile(binned_fluo[..., 1:20], 50, (1, 2))[
                                       :, np.newaxis, np.newaxis]), np.std(binned_fluo[..., 1:20], (1, 2))[:, np.newaxis, np.newaxis])
            binned_fluo_norm = np.mean(binned_fluo_norm[:, idx_CR, :], 1)
            binned_fluo_norm = old_div((binned_fluo_norm - np.percentile(binned_fluo_norm[..., 1:20], 50, (-1))[
                                       :, np.newaxis]), np.std(binned_fluo_norm[..., 1:20], (-1))[:, np.newaxis])
            binned_eye_norm = old_div((binned_eye - np.percentile(binned_eye[..., 1:20], 50, (-1))[
                                      :, np.newaxis]), np.std(binned_eye[..., 1:20], (-1))[:, np.newaxis])
            binned_eye_norm = binned_eye_norm[idx_CR].mean(0)
            binned_nose_norm = old_div((binned_nose - np.percentile(binned_nose[..., 1:20], 50, (-1))[
                                       :, np.newaxis]), np.std(binned_nose[..., 1:20], (-1))[:, np.newaxis])
            binned_nose_norm = binned_nose_norm[idx_CR].mean(0)

            idx_during_cs = np.where(
                (time_binned_data > -ISI) & (time_binned_data < 0.03))[0]
            idx_max_eye = np.where(binned_eye_norm[idx_during_cs] > 3)[0][0]
            idx_max_nose = np.where(binned_nose_norm[idx_during_cs] > 3)[0][0]

            pl.subplot(1, 2, 1)
            ax = pl.gca()
            l1 = pl.plot(
                time_binned_data[:], binned_fluo_norm[CR_eye_fluo > min_r_CR, :].T, color='lightgray')
            l2 = pl.plot(
                time_binned_data[:], binned_eye_norm[:], 'k', linewidth=2, label='Inline label')
            l3 = pl.plot(
                time_binned_data[:], binned_nose_norm[:], 'r', linewidth=2, label='Inline label')
            l1[0].set_label('Granule cells with r_CR>0.3')
            l2[0].set_label('Eyelid')
            l2[0].set_label('Snout')
            pl.legend()
            pl.xlabel('Time to US (s)')
            pl.ylabel('z-score')
            pl.xlim([-.5, 0.03])
            pl.ylim([-3, 10])
            # find fr each trial and neuron the first index larger than 1 std
            bins_sign_fluo = np.array([np.where(binned_fluo_norm[iid_neuro][idx_during_cs] > 3)[0][0] if np.max(
                binned_fluo_norm[iid_neuro][idx_during_cs]) > 3 else np.nan for iid_neuro in range(np.shape(binned_fluo_norm)[0])])


#            bins_sign_fluo=[bisect.bisect(binned_fluo_norm[iid_neuro][idx_during_cs],3) for iid_neuro in range(np.shape(binned_fluo_norm)[0])]

            pl.subplot(1, 2, 2)
            pl.hist(0.064 * (idx_max_eye - np.array(bins_sign_fluo)
                             [CR_eye_fluo > min_r_CR]), bins=10)
            pl.hist(0.064 * (idx_max_nose - np.array(bins_sign_fluo)
                             [CR_eye_fluo > min_r_CR]), bins=10)
            pl.xlabel('time lag granule - eyelid (s)')
            pl.ylabel('Cell count N=' + str(np.sum([CR_eye_fluo > min_r_CR])))
            pl.xlim([-.1, .2])

            if False:
                break
            # pl.ylim([3,8])
            all_neurons_lag = np.concatenate([np.load(a) for a in ['/mnt/ceph/users/agiovann/ImagingData/eyeblink/timing_lag_AG051501_sec.npy', '/mnt/ceph/users/agiovann/ImagingData/eyeblink/timing_lag_b35_sec.npy', '/mnt/ceph/users/agiovann/ImagingData/eyeblink/timing_lag_b37_sec.npy',
                                                                   '/mnt/ceph/users/agiovann/ImagingData/eyeblink/timing_lag_gc-AGGC6f-031213-03_sec.npy']])
            all_nose_lag = np.concatenate([np.load(a)['arr_0'] for a in [
                                          '/mnt/ceph/users/agiovann/ImagingData/eyeblink/lags_AG051515-01.npy.npz', '/mnt/ceph/users/agiovann/ImagingData/eyeblink/lags_b37.npy.npz']])

            pl.hist(all_neurons_lag, bins=np.arange(-.3, .3, .032))
            pl.hist(all_nose_lag, bins=np.arange(-.3, .3, .032))
            pl.xlabel('time lag granule - eyelid (s)')
            pl.ylabel('Cell count')
            pl.legend(['eyelid - GrC', 'nose Grc'])
#            idxHighCorr=np.argsort(CR_eye_fluo)[::-1]
        # redundancy calculation
#
        if compute_redundancy:

            _trials_, _n_nr, _timesteps_ = amplitudes_responses_CR_fl.shape
            X = np.reshape(amplitudes_responses_CR_fl.transpose(
                [1, 0, 2]), [_n_nr, _trials_ * _timesteps_]).T
            Y = ampl_CR_eye.flatten()
            CR_eye_fluo = np.array(CR_eye_fluo, dtype=float)
            idx_significant = np.where(
                CR_eye_fluo > np.percentile(CR_eye_fluo_rnd, 95))[0]
            ls = sklearn.linear_model.LassoCV(n_alphas=100, eps=1e-3)

#            np.nanmean(cross_val_score(ls, X, Y, cv=ShuffleSplit(10,test_size=.5),scoring=score_))
#            scores = cross_val_score(lr, X, Y, cv=ShuffleSplit(30,test_size=.5),scoring=score_)

            # [idxHighCorr][::-1]
            r_neurons = np.array(CR_eye_fluo[idx_significant], dtype=float)
            X = X[:, idx_significant]
            ls.fit(X, Y)

            print((np.sum(ls.coef_ > 0)))

            print((len(idx_significant)))
            print((np.mean(CR_eye_fluo[idx_significant])))
            print((np.std(CR_eye_fluo[idx_significant])))
            print((np.max(CR_eye_fluo[idx_significant])))
            print((np.median(CR_eye_fluo[idx_significant])))
            print('*')
            print((np.nanmean(CR_eye_fluo)))
            print((np.nanstd(CR_eye_fluo)))
            print((np.nanmax(CR_eye_fluo)))
            print((np.nanmedian(CR_eye_fluo)))

            r_classif_naive = cross_val_score(
                lr, X, Y, cv=ShuffleSplit(100, test_size=.5), scoring=score_)
            r_classif = np.mean(r_classif_naive)

            print((np.mean(r_classif)))

            r_classif_lasso = cross_val_score(
                lr, X[:, ls.coef_ > 0], Y, cv=ShuffleSplit(100, test_size=.5), scoring=score_)
            r_classif = np.mean(r_classif_lasso)
            r_neurons = r_neurons[ls.coef_ > 0]

#            r_classif=[np.mean(cross_val_score(lr, X[:,:iidd+1], Y, cv=ShuffleSplit(30,test_size=.5),scoring=score_)) for iidd in range(_n_nr)][::-1]

            print((np.mean(r_classif)))

            # r_classif=np.array(r_classif).T
            idx_good = np.where((r_neurons != np.array(None))
                                & (~np.isnan(r_neurons)))[0]
            # r_classif=r_classif[idx_good]
            r_neurons = r_neurons[idx_good]

            N = len(r_neurons)
            I_neurons = -0.5 * np.log(1 - r_neurons**2) / np.log(2)
            I_classif = -0.5 * np.log(1 - r_classif**2) / np.log(2)
            Icum_neurons = np.sum(I_neurons)
#            I_classif=I_classif[::-1]
            redundancy.append(old_div(np.max(Icum_neurons), np.max(I_classif)))
            print(('Redundancy:' + str(old_div(np.max(Icum_neurons), np.max(I_classif)))))
            print((len(I_neurons)))

        else:
            print('SKIPPED REDUNDANCY')

        if 0:  # for examples images use Ag0515
            scores = cross_val_score(lr, X, Y, cv=ShuffleSplit(
                100, test_size=.5), scoring=score_)
            vls, bns, _ = pl.hist(CR_eye_fluo, bins=np.arange(-.1, 1, .05))
            pl.cla()
            vl_distr = scipy.signal.savgol_filter(vls, 1, 0)
            vl_distr = old_div(vl_distr, np.sum(vl_distr))
            pl.plot(bns[1:], vl_distr, 'r')
            lq, hq = np.percentile(CR_eye_fluo_rnd, [5, 95])
            pl.fill_between([lq, hq], [.35, .35], alpha=.3, color='c')
            lq, hq = np.percentile(scores, [5, 95])
            pl.fill_between([lq, hq], [.35, .35], alpha=.3, color=[.8, .8, .8])
            pl.plot([np.mean(scores), np.mean(scores)], [0, .35], 'k--')
            pl.xlabel('Person\'s r')
            pl.ylabel('Probability')

            pl.figure()
            r_each = [np.mean(cross_val_score(lr, X[:, iidd:iidd + 1], Y, cv=ShuffleSplit(
                100, test_size=.5), scoring=score_)) for iidd in range(_n_nr)][::-1]
#                r_neurons=np.array(r_each)
#                I_neurons = -0.5*np.log(1-r_neurons**2)/np.log(2)
#                I_classif = -0.5*np.log(1-r_classif**2)/np.log(2)
#                Icum_neurons = np.cumsum(I_neurons)[::-1]
            pl.plot(Icum_neurons, '-r')
            pl.plot(I_classif, '-k')

            pl.xlabel('Number of neurons')
            pl.legend(['sum of individual GrCs', 'classifier'])
            pl.ylabel('Cumulative information (bits)')
            pl.xlim([-1, N - 1])
            pl.ylim([0, 1.5])
            pl.title('Redundancy calculation')

        bh_correl_tmp['r_CR_eye_fluo'] = CR_eye_fluo
        bh_correl_tmp['r_CR_eye_fluo_rnd'] = CR_eye_fluo_rnd

        if len(idx_US) > 2:
            bh_correl_tmp['active_during_UR'] = np.mean(np.min(f_mat_bl_erfc[idx_US, :, :][:, :, np.logical_and(
                mat_time > .15, mat_time < .3)], -1) < thresh_fluo_log, 0)
        else:
            print('** NOT ENOUGH TRIALS **')
            bh_correl_tmp['active_during_UR'] = np.nan

        all_idx_neg = np.union1d(idx_US, idx_NOCR)
        if len(all_idx_neg) > min_trials:
            bh_correl_tmp['active_during_UR_NOCR'] = np.mean(np.min(
                f_mat_bl_erfc[all_idx_neg, :, :][:, :, np.logical_and(mat_time > .15, mat_time < .3)], -1) < thresh_fluo_log, 0)
        else:
            print('** NOT ENOUGH TRIALS **')
            bh_correl_tmp['active_during_UR_NOCR'] = np.nan

        bh_correl_tmp['active_during_CS'] = np.mean(np.min(f_mat_bl_erfc[idx_NOCR, :, :][:, :, np.logical_and(
            mat_time > -.05, mat_time < -.03)], -1) < thresh_fluo_log, 0)

        if len(idx_CR) > min_trials:
            bh_correl_tmp['active_during_CR'] = np.mean(np.min(f_mat_bl_erfc[idx_CR, :, :][:, :, np.logical_and(
                mat_time > -.05, mat_time < -.03)], -1) < thresh_fluo_log, 0)
        else:
            bh_correl_tmp['active_during_CR'] = np.nan

        bh_correl_tmp['mouse'] = mouse
        print(mouse)

        bh_correl_tmp['idx_component'] = idx_neurons

        bh_correl_tmp['perc_CR'] = len(
            idx_CR) * 1. / (len(idx_NOCR) + len(idx_CR))

        bh_correl_tmp['ampl_eyelid_CSCSUS'] = func_avg_fluo(
            mat_ampl_at_US[np.union1d(idx_NOCR, idx_CR)])

        bh_correl_tmp['num_trials'] = len(idxCSCSUS) + len(idx_US)
        bh_correl_tmp['fraction_movement'] = fraction_movement
        bh_correl_tmp['ampl_FOV'] = 130 * 50
        bh_correl_tmp['fluo_range'] = 3
        bh_correl_tmp['idx_components_final'] = idx_neurons

    for ss in sess_:

        idx_sess = np.where([item == ss for item in cr_ampl_dic['session']])[0]
        print(ss)

        session = cr_ampl_dic['day'][idx_sess[0]]
        day = cr_ampl_dic['realDay'][idx_sess[0]]
        if mouse != mouse_now:
            mouse_now = mouse
            session_id = 0
            session_now = ''
            learning_phase = 0
            print('early')
        else:
            if day != session_now:
                session_id += 1
                session_now = day

        idx_CR = np.intersect1d(idx_sess, mat_idxCR)
        idx_NOCR = np.intersect1d(idx_sess, mat_idxNOCR)
        idx_US = np.intersect1d(idx_sess, mat_idxUS)
        idxCSCSUS = np.intersect1d(idx_sess, mat_idxCSCSUS)
#        if len(idxCSCSUS)+len(idx_US)  < 10:
#            ksks

        ampl_CR = pd.DataFrame()

        if len(idx_CR) > min_trials:
            fluo_crpl = func_avg_fluo(mat_fluo[idx_CR, :, :][:, :, np.logical_and(
                mat_time > -.05, mat_time < time_US_on)], (0, -1))
            ampl_CR['fluo_plus'] = fluo_crpl
            ampl_CR['ampl_eyelid_CR'] = func_avg_fluo(mat_ampl_at_US[idx_CR])

        else:
            ampl_CR['fluo_plus'] = np.nan * idx_neurons
            ampl_CR['ampl_eyelid_CR'] = np.nan * idx_neurons

        if len(idx_NOCR) > min_trials:
            fluo_crmn = func_avg_fluo(mat_fluo[idx_NOCR, :, :][:, :, np.logical_and(
                mat_time > -.05, mat_time < time_US_on)], (0, -1))
            ampl_CR['fluo_minus'] = fluo_crmn

        else:
            ampl_CR['fluo_minus'] = np.nan * idx_neurons

        ampl_CR['session'] = session
        ampl_CR['mouse'] = mouse
        ampl_CR['chunk'] = ss

        ampl_CR['idx_component'] = idx_neurons
        ampl_CR['perc_CR'] = len(idx_CR) * 1. / (len(idx_NOCR) + len(idx_CR))

        if len(sess_) > 1:

            if len(idx_CR) * 1. / (len(idx_NOCR) + len(idx_CR)) > thresh_middle and learning_phase == 0:
                learning_phase = 1
                print('middle')
            elif len(idx_CR) * 1. / (len(idx_NOCR) + len(idx_CR)) > thresh_advanced and learning_phase == 1:
                learning_phase = 2
                print('advanced')
            elif len(idx_CR) * 1. / (len(idx_NOCR) + len(idx_CR)) > thresh_late and learning_phase == 3:
                learning_phase = 3
                print('late')

        else:
            if (len(idx_CR) * 1. / (len(idx_NOCR) + len(idx_CR))) > thresh_late:
                learning_phase = 3
                print('late')
            elif len(idx_CR) * 1. / (len(idx_NOCR) + len(idx_CR)) > thresh_advanced:
                learning_phase = 2
                print('advanced')
            elif len(idx_CR) * 1. / (len(idx_NOCR) + len(idx_CR)) > thresh_middle:
                learning_phase = 1
                print('middle')

        ampl_CR['learning_phase'] = learning_phase
        ampl_CR['ampl_eyelid_CSCSUS'] = func_avg_fluo(
            mat_ampl_at_US[np.union1d(idx_NOCR, idx_CR)])
        ampl_CR['session_id'] = session_id

        print((len(idxCSCSUS) + len(idx_US)))
        ampl_CR['num_trials'] = len(idxCSCSUS) + len(idx_US)
        ampl_CR['ampl_FOV'] = 130 * 50
        ampl_CR['fluo_range_responsive'] = 3

        cr_ampl = pd.concat([cr_ampl, ampl_CR])

    print(learning_phase)
    bh_correl_tmp['learning_phase'] = learning_phase

    bh_correl = pd.concat([bh_correl, bh_correl_tmp])

print((bh_correl.isnull().sum()))
print((cr_ampl.isnull().sum()))
#%%
# print mat_idxCR.size
#amplitudes_responses=np.nanmedian(mat_fluo[:,:,np.logical_and(mat_time > -.05,mat_time < time_US_on)],(-1))
#
#
#
#
# mat_cr_ampl=pd.DataFrame()
# mat_cr_ampl_tmp=pd.DataFrame()
# for counter,resp in enumerate(amplitudes_responses):
#    mat_cr_ampl_tmp['fluo']=resp
#    mat_cr_ampl_tmp['trial_counter']=counter
#    mat_cr_ampl_tmp['trialName']=cr_ampl_dic['trialName'][counter]
#    mat_cr_ampl_tmp['trials']=cr_ampl_dic['trials'][counter]
#    mat_cr_ampl_tmp['trialsTypeOrig']=cr_ampl_dic['trialsTypeOrig'][counter]
#    mat_cr_ampl_tmp['session']=cr_ampl_dic['session'][counter]
#    mat_cr_ampl_tmp['mouse']=cr_ampl_dic['animal'][counter]
#    mat_cr_ampl_tmp['day']=cr_ampl_dic['day'][counter]
#    mat_cr_ampl_tmp['session_id']=cr_ampl_dic['realDay'][counter]
#    mat_cr_ampl_tmp['amplAtUs']=mat_ampl_at_US[counter]
#    if any(sstr in cr_ampl_dic['trialName'][counter] for sstr in ['CSUSwCR','CSwCR']):
#        mat_cr_ampl_tmp['type_CR']=1
#    elif any(sstr in cr_ampl_dic['trialName'][counter] for sstr in ['US']):
#        mat_cr_ampl_tmp['type_CR']=np.nan
#    else:
#        mat_cr_ampl_tmp['type_CR']=0
#    mat_cr_ampl = pd.concat([mat_cr_ampl,mat_cr_ampl_tmp])
#%%
# sess_grp=mat_cr_ampl.groupby('session')
# for nm,idxs in sess_grp.indices.iteritems():
#    print nm
#    mat_cr_ampl_tmp=mat_cr_ampl[idxs]
#    nm_idxCR=np.where(mat_cr_ampl_tmp['type_CR'])[0]
#    print (len(nm_idxCR))
#    if len(nm_idxCR)>min_trials:
#
#        ampl_CR['fluo_plus']=fluo_crpl
#        ampl_CR['ampl_eyelid_CR']=np.mean(amplitudes_at_US[nm_idxCR])
#    else:
#        ampl_CR['fluo_plus']=np.nan
#        ampl_CR['ampl_eyelid_CR']=np.nan
#
#    if len(nm_idxNOCR)>min_trials:
#        ampl_CR['fluo_minus']=fluo_crmn
#    else:
#        ampl_CR['fluo_minus']=np.nan
#
#    ampl_CR['session']=session;
#    ampl_CR['mouse']=mouse;
#    ampl_CR['chunk']=chunk
#    ampl_CR['idx_component']=idx_components_final;
#    ampl_CR['perc_CR']=len(nm_idxCR)*1./len(nm_idxCSCSUS)
#    if  len(nm_idxCR)*1./len(nm_idxCSCSUS)> thresh_middle and learning_phase==0:
#        learning_phase=1
#        print 'middle'
#    elif len(nm_idxCR)*1./len(nm_idxCSCSUS)> thresh_late and learning_phase==1:
#        learning_phase=2
#        print 'late'
#    ampl_CR['learning_phase']= learning_phase
#    ampl_CR['ampl_eyelid_CSCSUS']=np.mean(amplitudes_at_US[nm_idxCSCSUS])
#    ampl_CR['session_id']=session_id
#    cr_ampl=pd.concat([cr_ampl,ampl_CR])
#mat_eyelid=mat_eyelid-np.nanmean(mat_eyelid[mat_time < -.44,:],0)[np.newaxis,:]
#UR_size=np.median(np.nanmax(mat_eyelid[np.logical_and(mat_time > .03,mat_time < .25) ,:],0))
# mat_eyelid=mat_eyelid/UR_size
#%%
# bins_trials=pd.cut(mat_cr_ampl['trial_counter'],[0,200,600,822],include_lowest=True)
# grouped_session=mat_cr_ampl.groupby(['mouse',bins_trials,'type_CR'])
# mean_plus=grouped_session.mean().loc['AG051514-01'].loc[(slice(None),[1.0]),:]
# mean_minus=grouped_session.mean().loc['AG051514-01'].loc[(slice(None),[0.0]),:]
# std_plus=grouped_session.sem().loc['AG051514-01'].loc[(slice(None),[1.0]),:]
# std_minus=grouped_session.sem().loc['AG051514-01'].loc[(slice(None),[0.0]),:]
#
#
# mean_plus['amplAtUs'].plot(kind='line',marker='o',markersize=15)
# mean_minus['amplAtUs'].plot(kind='line',marker='o',markersize=15)
#
# mean_plus['fluo'].plot(kind='line',yerr=std_plus,marker='o',markersize=15)
# mean_minus['fluo'].plot(kind='line',yerr=std_minus,marker='o',markersize=15)
#%%
cr_ampl_old = cr_ampl.copy()
#%%
cr_ampl = cr_ampl_old.copy()
#%% ****** INIT ANALYSIS
pl.rcParams['pdf.fonttype'] = 42
font = {'family': 'Myriad Pro',
        'weight': 'regular',
        'size': 20}
pl.rc('font', **font)

#%%
pl.close('all')
grouped_session = cr_ampl.groupby(['mouse', 'session'])

if False:
    grouped_session.mean().loc['b35'][['ampl_eyelid_CR', 'perc_CR']].plot(kind='line', subplots=True, layout=(
        2, 1), marker='o', markersize=15, xticks=list(range(len(grouped_session.mean().loc['b35']))))
grouped_session.mean().loc['b37'][['ampl_eyelid_CR', 'perc_CR']].plot(kind='line', subplots=True, layout=(
    2, 1), marker='o', markersize=15, xticks=list(range(len(grouped_session.mean().loc['b37']))))
grouped_session.mean().loc['gc-AGGC6f-031213-03'][['ampl_eyelid_CR', 'perc_CR']].plot(kind='line', subplots=True, layout=(
    2, 1), marker='o', markersize=15, xticks=list(range(len(grouped_session.mean().loc['gc-AGGC6f-031213-03']))))
grouped_session.mean().loc['gc-AG052014-02'][['ampl_eyelid_CR', 'perc_CR']].plot(kind='line', subplots=True, layout=(
    2, 1), marker='o', markersize=15, xticks=list(range(len(grouped_session.mean().loc['gc-AG052014-02']))))
grouped_session.mean().loc['AG052014-01'][['ampl_eyelid_CR', 'perc_CR']].plot(kind='line', subplots=True, layout=(
    2, 1), marker='o', markersize=15, xticks=list(range(len(grouped_session.mean().loc['AG052014-01']))))
grouped_session.mean().loc['AG051514-01'][['ampl_eyelid_CR', 'perc_CR']].plot(kind='line', subplots=True, layout=(
    2, 1), marker='o', markersize=15, xticks=list(range(len(grouped_session.mean().loc['AG051514-01']))))
pl.rc('font', **font)

# pl.ylim([0,.5])
#%%


def get_stats(group):
    return {'min': group.min(), 'max': group.max(), 'count': group.count(), 'mean': group.mean()}


#%%
min_trials_pd = 8
max_range_fluo_pd = 4
min_neuron_number_pd = 8
cr_ampl_sel = cr_ampl[(cr_ampl.fluo_range_responsive <= max_range_fluo_pd) & (
    cr_ampl.num_trials >= min_trials_pd)].copy()
bh_correl_sel = bh_correl[(bh_correl.fluo_range <= max_range_fluo_pd) & (
    bh_correl.num_trials >= min_trials_pd)].copy()
#%% NUMBER OF NEURONS PER FOV
bh_correl_tmp = bh_correl_sel.copy()
bh_correl_tmp.groupby('mouse').count()['ampl_eyelid_CSCSUS'].mean()
large_fovs_number_of_neurons = bh_correl_tmp.groupby(['mouse', 'session', 'chunk']).count(
)['ampl_eyelid_CSCSUS'].groupby(level=0).agg([np.mean, scipy.stats.sem])
small_fovs_number_of_neurons = bh_correl_tmp.groupby(
    ['mouse']).count()['ampl_eyelid_CSCSUS'].groupby(level=0).mean()

small_fovs_number_of_neurons = bh_correl_sel.groupby(
    ['mouse']).count()['ampl_eyelid_CSCSUS'].groupby(level=0).mean()
small_fovs_number_of_neurons = small_fovs_number_of_neurons[[
    'b3' not in ii for ii in small_fovs_number_of_neurons.index]].values
#%% COUNT REPONDING NEURONS, JUST USE LAST SESSION or ALL SESSION  FFOLLOWED TRUE
bh_correl_tmp = bh_correl_sel.copy()

b37_last_sess = bh_correl_tmp[(bh_correl_tmp.mouse == 'b37') & (
    bh_correl_tmp.session == '20160705103903') & (bh_correl_tmp.chunk == '121')]
b35_last_sess = bh_correl_tmp[(bh_correl_tmp.mouse == 'b35') & (
    bh_correl_tmp.session == '20160714143248') & (bh_correl_tmp.chunk == '061')]
AG051514_01_last_sess = bh_correl_tmp[(bh_correl_tmp.mouse == 'AG051514-01')]
AG052014_01_last_sess = bh_correl_tmp[(bh_correl_tmp.mouse == 'AG052014-01')]
gc_AGGC6f_031213_03_last_sess = bh_correl_tmp[(
    bh_correl_tmp.mouse == 'gc-AGGC6f-031213-03')]
#gc_AG052014_02_last_sess =bh_correl_tmp[(bh_correl_tmp.mouse=='gc-AG052014-02')]

cr_neurons = []
all_neurons = []
# for ms in [b37_last_sess, b35_last_sess, AG051514_01_last_sess, AG052014_01_last_sess, gc_AGGC6f_031213_03_last_sess,gc_AG052014_02_last_sess ]:
for ms in [b37_last_sess, b35_last_sess, AG051514_01_last_sess, AG052014_01_last_sess, gc_AGGC6f_031213_03_last_sess]:
    ms_1 = ms.copy()
    print((ms_1.mouse.unique()))
    ms_1['r_CR_eye_fluo_rnd'] = ms.fillna(method='pad').groupby(
        'mouse')['r_CR_eye_fluo_rnd'].transform(lambda x: x.quantile(.95))
    ms_1['r_CR_eye_fluo'] = ms_1['r_CR_eye_fluo'] - ms_1['r_CR_eye_fluo_rnd']
    cr_neurons.append(len(ms_1[ms_1['r_CR_eye_fluo'] > 0]) * 1.)
    all_neurons.append(len(ms_1))
    print((len(ms_1[ms_1['r_CR_eye_fluo'] > 0]) * 1. / len(ms_1)))

all_neurons = np.array(all_neurons)
cr_neurons = np.array(cr_neurons)

print((np.sum(cr_neurons) * 1. / np.sum(all_neurons)))
print((np.mean(cr_neurons * 1 / all_neurons)))
print((np.std(cr_neurons * 1 / all_neurons)))
print((scipy.stats.sem(cr_neurons * 1 / all_neurons)))
#%% NUMBER OF NEURONS PER FOV
large_fov_size = 230 * 230
small_fov_size = 130 * 50
bh_correl_tmp = bh_correl_sel.copy()
bh_correl_tmp.groupby('mouse').count()['ampl_eyelid_CSCSUS'].mean()
large_fovs_number_of_neurons = bh_correl_tmp.groupby(['mouse', 'session', 'chunk']).count(
)['ampl_eyelid_CSCSUS'].groupby(level=0).agg([np.mean, scipy.stats.sem])

neurons_per_100_um2 = large_fovs_number_of_neurons / large_fov_size * 100 * 100

small_fovs_number_of_neurons = bh_correl_tmp.groupby(
    ['mouse']).count()['ampl_eyelid_CSCSUS'].groupby(level=0).mean()


small_fovs_number_of_neurons = bh_correl_sel.groupby(
    ['mouse']).count()['ampl_eyelid_CSCSUS'].groupby(level=0).mean()
small_fovs_number_of_neurons = small_fovs_number_of_neurons[[
    'b3' not in ii for ii in small_fovs_number_of_neurons.index]].values

neurons_per_100_um2 = np.hstack([neurons_per_100_um2.values[:, 0].squeeze(
), small_fovs_number_of_neurons * 1. / small_fov_size * 100 * 100])
print((np.mean(neurons_per_100_um2)))
print((np.std(neurons_per_100_um2)))

#[cr_ampl['mouse']=='b37']
#%% OLD BY BY CELL
# for jjj in range(10):
#    samples_per_FOV=30
#    cr_ampl_m=cr_ampl_sel
#    cr_ampl_m=cr_ampl_m.groupby(['mouse','session'])
#    cr_ampl_m = cr_ampl_m.apply(lambda x:x.sample(samples_per_FOV,replace=True))
#
#    print len(cr_ampl_m)
#    cr_ampl_m=cr_ampl_m[cr_ampl_m.mouse!='gc-AG052014-02']
#    #cr_ampl_m=cr_ampl_m[cr_ampl_m.mouse!='AG052014-01']
#
#    grouped_session=cr_ampl_m.groupby(['learning_phase'])
#    means=grouped_session.mean()[['fluo_plus','fluo_minus']]
#    sems=grouped_session.sem()[['fluo_plus','fluo_minus']]
#    means.plot(kind='line',yerr=sems,marker ='o',xticks=range(6),markersize=15)
#
#    #grouped_session.mean()[['fluo_plus','fluo_minus']].plot(kind='line',yerr=sems,marker='o',markersize=15,xticks=range(len(grouped_session.mean())))
#    #pl.rc('font', **font)
#    pl.xticks(np.arange(4),['Naive','Learning','Advanced','Trained'])
#    pl.xlabel('Learning Phase')
#    pl.ylabel('DF/F')
#    pl.legend(['CR+','CR-'],loc=0)
#    pl.xlim([-.1 ,3.1])
#
#
#    pluss=grouped_session.apply(lambda x: x[['fluo_plus']].values).values
#    pl_m=[]
#    for pl_ in pluss:
#        pl_=np.array(pl_)
#        pl_m.append(pl_[~np.isnan(pl_)])
#
#    pl_m = filter(len,pl_m)
#    scipy.stats.f_oneway(*pl_m)
#    idx2=[]
#    idx1=[]
#    for ii,pl_ in enumerate(pl_m):
#        idx2.append([ii]*len(pl_))
#        idx1.append(pl_)
#
#    idx2=np.hstack(idx2)
#    idx1= np.hstack(idx1)
#    print scipy.stats.pearsonr(idx2, idx1)
##    pl.scatter(idx2, idx1)
#
#    minuss=grouped_session.apply(lambda x: x[['fluo_minus']].values).values
#    pl_m=[]
#    for pl_ in minuss:
#        pl_=np.array(pl_)
#        pl_m.append(pl_[~np.isnan(pl_)])
#
#    pl_m = filter(len,pl_m)
#
#    scipy.stats.f_oneway(*pl_m)
#    pl.close()
#
#    idx2=[]
#    idx1=[]
#    for ii,pl_ in enumerate(pl_m):
#        idx2.append([ii]*len(pl_))
#        idx1.append(pl_)
#
#    idx2=np.hstack(idx2)
#    idx1= np.hstack(idx1)
#    print scipy.stats.pearsonr(idx2, idx1)
#    pl.scatter(idx2, idx1)
#minuss=np.concatenate(grouped_session.apply(lambda x: x[['fluo_plus']].values).values)
import statsmodels.api as sm
#%% values javier
cr_ampl_m = cr_ampl.copy()
for mse in cr_ampl_m.mouse.unique():
    print(mse)
    subset = cr_ampl_m[(cr_ampl_m.mouse == mse)]
    begin = np.vstack(subset[(subset.session_id < 2)]
                      [['fluo_plus', 'fluo_minus']].values[:, 1]).squeeze()
    end = np.vstack(subset[(subset.session_id >= subset.session_id.max(
    ) - 2)][['fluo_plus', 'fluo_minus']].values[:, 0])
    end = end[~np.isnan(end)]
    begin = begin[~np.isnan(begin)]
    print((scipy.stats.ttest_ind(begin, end)))
    end1, end2 = np.vstack(subset[(subset.session_id >= subset.session_id.max(
    ) - 2)][['fluo_plus', 'fluo_minus']].values)[:, [0, 1]]
    end = end[~np.isnan(end)]

#%%
all_rs_plus = []
all_anova_plus = []
all_rs_minus = []
all_anova_minus = []
all_ttest_plus_vs_minus = []
all_ttest_first_vs_last_plus = []
all_ttest_first_vs_last_minus = []

n_all_ttest_plus_vs_minus = []
n_all_ttest_first_vs_last_plus = []
n_all_ttest_first_vs_last_minus = []


samples_per_FOV = 30
#group_var = 'ampl_eyelid_CSCSUS'
#bins=[-0.15,.15, .4,  1]
###
##
group_var = 'perc_CR'
bins = [0, .2, .4, .6, .8, 1]


#group_var = 'learning_phase'
# bins=[0,.1,1,2,3]


all_traces_sem = []
all_traces_mean = []
pl.figure()
for jjj in range(40):
    cr_ampl_m = cr_ampl_sel  # [cr_ampl['mouse']=='b37']
    ax = pl.gca()
    cr_ampl_m = cr_ampl_sel
    cr_ampl_m = cr_ampl_m.groupby(['mouse', 'session'])
    cr_ampl_m = cr_ampl_m.apply(
        lambda x: x.sample(samples_per_FOV, replace=True))

    # grouped_session=cr_ampl_m.sort(['perc_CR'],ascending=True)
#    cr_ampl_m=cr_ampl_m[cr_ampl_m.mouse!='gc-AG052014-02']
    # cr_ampl_m=cr_ampl_m[cr_ampl_m.mouse!='AG052014-01']
    #cr_ampl_m=cr_ampl_m[(cr_ampl_m.mouse=='AG052014-01') | (cr_ampl_m.mouse=='gc-AG052014-02')]
    # cr_ampl_m=cr_ampl_m[(cr_ampl_m.mouse=='AG052014-01')]
    cr_ampl_m.dropna()
    grouped_session = cr_ampl_m.groupby(
        pd.cut(cr_ampl_m[group_var], bins, include_lowest=True))
    means = grouped_session.mean()[['fluo_plus', 'fluo_minus']][grouped_session.count()[
        ['fluo_plus', 'fluo_minus']] > min_neuron_number_pd]
    #sems=grouped_session.apply(lambda x: x.dropna().quantile(.75) - x.dropna().quantile(.25))[['fluo_plus','fluo_minus']]
    sems = grouped_session.sem()[['fluo_plus', 'fluo_minus']][grouped_session.count()[
        ['fluo_plus', 'fluo_minus']] > min_neuron_number_pd]

    means.plot(kind='line', yerr=sems, marker='o', xticks=list(
        range(6)), markersize=15, ax=ax, color=['k', 'r'])

    pl.xlim([-.1, 5.1])
    pl.legend(['CR+', 'CR-'], loc=4)
    pl.xlabel(group_var)
    pl.ylabel('DF/F')
    pl.pause(.1)

    all_traces_mean.append(means.values)

    all_traces_sem.append(sems.values)

    pluss = grouped_session.apply(lambda x: x[['fluo_plus']].values).values
    pl_m = []
    for pl_ in pluss:
        pl_ = np.array(pl_)
        pl_m.append(pl_[~np.isnan(pl_)])

    plus_for_ttest = pl_m
    pl_m = list(filter(len, pl_m))
    all_ttest_first_vs_last_plus.append(
        scipy.stats.ttest_ind(pl_m[0], pl_m[-1]))
    n_all_ttest_first_vs_last_plus.append((len(pl_m[0]), len(pl_m[-2])))

    anova_1 = scipy.stats.f_oneway(*pl_m)

    idx2 = []
    idx1 = []
    for ii, pl_ in enumerate(pl_m):
        idx2.append([ii] * len(pl_))
        idx1.append(pl_)

    idx2 = np.hstack(idx2)
    idx1 = np.hstack(idx1)


#
#    X = sm.add_constant(idx2, prepend=False)
#    y=idx1
#    model=sm.OLS(y,X)
#    results=model.fit()
#    print results.summary()

    all_rs_plus.append(scipy.stats.pearsonr(idx2, idx1))
    all_anova_plus.append((anova_1.statistic, anova_1.pvalue))

    minuss = grouped_session.apply(lambda x: x[['fluo_minus']].values).values
    pl_m = []
    for pl_ in minuss:
        pl_ = np.array(pl_)
        pl_m.append(pl_[~np.isnan(pl_)])

    minus_for_ttest = pl_m
    pl_m = list(filter(len, pl_m))
    all_ttest_first_vs_last_minus.append(
        scipy.stats.ttest_ind(pl_m[0], pl_m[-2]))

    n_all_ttest_first_vs_last_minus.append((len(pl_m[0]), len(pl_m[-2])))

    anova_1 = scipy.stats.f_oneway(*pl_m)

    idx2 = []
    idx1 = []
    for ii, pl_ in enumerate(pl_m):
        idx2.append([ii] * len(pl_))
        idx1.append(pl_)

    idx2 = np.hstack(idx2)
    idx1 = np.hstack(idx1)

    all_rs_minus.append(scipy.stats.pearsonr(idx2, idx1))
    all_anova_minus.append((anova_1.statistic, anova_1.pvalue))

    ttes_tmp = []
    n_ttest_tmp = []
    for pls, mns in zip(plus_for_ttest, minus_for_ttest):
        print((1))
        ttes_tmp.append(scipy.stats.ttest_ind(pls, mns))
        n_ttest_tmp.append((len(pls), len(mns)))

    all_ttest_plus_vs_minus.append(ttes_tmp)
    n_all_ttest_plus_vs_minus.append(n_ttest_tmp)
#    X = sm.add_constant(idx2, prepend=False)
#    y=idx1
#    model=sm.OLS(y,X)
#    results=model.fit()
#    print results.summary()


#
pl.figure()
pl.errorbar(bins[1:], np.mean(all_traces_mean, 0)[:, 0],
            np.std(all_traces_mean, 0)[:, 0], linewidth=2)
pl.errorbar(bins[1:], np.mean(all_traces_mean, 0)[:, 1],
            np.std(all_traces_mean, 0)[:, 1], linewidth=2)
pl.legend(['CR+', 'CR-'], loc=4)
pl.xlabel(group_var)
pl.ylabel('DF/F')
#%% size effects
old_div((np.mean(all_traces_mean, 0)[:, 0] - np.mean(all_traces_mean, 0)[:, 1]),
        np.sqrt(.5 * np.var(all_traces_mean, 0)[:, 0] + .5 * np.var(all_traces_mean, 0)[:, 1]))
print((np.mean(all_traces_mean, 0)[:, 0]))
print((np.mean(all_traces_mean, 0)[:, 1]))
print((np.std(all_traces_mean, 0)[:, 0]))
print((np.std(all_traces_mean, 0)[:, 1]))

#%% r coeff
print((np.mean(all_rs_plus, 0)))
print((np.std(all_rs_plus, 0)))


print((np.median(all_rs_plus, 0)))
print((np.median(all_rs_minus, 0)))

print((np.median(all_anova_plus, 0)))
print((np.median(all_anova_minus, 0)))


print(('p<:', np.percentile(np.array(all_rs_plus)[:, 1], [50, 25, 75], 0)))
print((np.mean(np.array(all_rs_plus)[:, 0], 0), np.std(
    np.array(all_rs_plus)[:, 0], 0)))

print(('p<:', np.percentile(np.array(all_rs_minus)[:, 1], [50, 25, 75], 0)))
print((np.mean(np.array(all_rs_minus)[:, 0], 0), np.std(
    np.array(all_rs_minus)[:, 0], 0)))


print((np.nanpercentile(all_ttest_plus_vs_minus, [50, 25, 75], 0).T))
print((np.nanpercentile(n_all_ttest_plus_vs_minus, [50], 0)))

print((np.nanpercentile(all_ttest_first_vs_last_minus, [50, 25, 75], 0).T))

print((np.nanpercentile(n_all_ttest_first_vs_last_minus, [50], 0).T))


print((np.nanpercentile(all_ttest_first_vs_last_plus, [50, 25, 75], 0).T))
print((np.nanpercentile(n_all_ttest_first_vs_last_plus, [50], 0).T))


# pl.subplot(2,1,1)
# pl.hist(np.array(all_rs_plus).T[1],bins=np.arange(0,.6,.01))
# pl.xlabel('p-value')
#pl.ylabel('frequency count')
# pl.subplot(2,1,2)
#
# pl.hist(np.array(all_rs_plus).T[0],30)
#pl.xlabel('r coefficient')
#pl.ylabel('frequency count')
#
#
#
# pl.figure()
# pl.subplot(2,1,1)
# pl.hist(np.array(all_rs_minus).T[1],bins=np.arange(0,.6,.01))
# pl.xlabel('p-value')
#pl.ylabel('frequency count')
# pl.subplot(2,1,2)
#
# pl.hist(np.array(all_rs_minus).T[0],30)
#pl.xlabel('r coefficient')
#pl.ylabel('frequency count')
#%%
#pl.rc('font', **font)
#%% compute stats for locomotion
idx_last_sessions = (((bh_correl_sel.mouse == 'b37') & (bh_correl_sel.session == '20160705103903') & (bh_correl_sel.chunk == '121')) |
                     ((bh_correl_sel.mouse == 'b35') & (bh_correl_sel.session == '20160714143248') & (bh_correl_sel.chunk == '061')) |
                     ((bh_correl_sel.mouse == 'AG051514-01')) |
                     ((bh_correl_sel.mouse == 'AG052014-01')) |\
                     #                              ((bh_correl_sel.mouse=='gc-AGGC6f-031213-03')) |\
                     ((bh_correl_sel.mouse == 'gc-AG052014-02')))


#bh_correl_tmp =  bh_correl_sel[idx_last_sessions].copy()
bh_correl_tmp = bh_correl_sel.copy()
# print bh_correl_sel.groupby(['mouse','session','chunk']).count().loc[['b37','b35']]['ampl_eyelid_CSCSUS'].describe()
# print bh_correl_sel.groupby(['mouse']).count().loc[['gc-AGGC6f-031213-03', 'gc-AG052014-02','AG052014-01', 'AG051514-01']]['ampl_eyelid_CSCSUS'].describe()

# fund per mouse threshold
bh_correl_tmp['thresh_wheel_corr'] = bh_correl_tmp.fillna(method='pad').groupby(
    'mouse')['r_wheel_fluo_rnd'].transform(lambda x: x.quantile(.99))
# bh_correl_tmp.thresh_wheel_corr
bh_correl_tmp = bh_correl_tmp[bh_correl_tmp.r_wheel_fluo > .5]
print((bh_correl_tmp.avg_activity_during_locomotion.describe()))
print((bh_correl_tmp.avg_activity_during_locomotion.quantile(.99)))
#%%
cr_ampl_m = cr_ampl_sel.copy()
##
variable = 'ampl_eyelid_CSCSUS'
bins = [-1, .15, .4, 1]
##
# variable='learning_phase'
# bins=[0,.1,1,2,3]
##
variable = 'perc_CR'
bins = [0, .15,  .4, 1]

# grouped_session=cr_ampl_m.sort(['perc_CR'],ascending=True)
# cr_ampl_m=cr_ampl_m[cr_ampl_m.mouse!='gc-AG052014-02']
# cr_ampl_m=cr_ampl_m[cr_ampl_m.mouse!='AG052014-01']
cr_ampl_m.dropna()
cuts = pd.cut(cr_ampl_m[variable], bins, include_lowest=True)
grouped_session = cr_ampl_m.groupby(['mouse', cuts])

means = grouped_session.mean()[['fluo_plus', 'fluo_minus']]
sems = grouped_session.sem()[['fluo_plus', 'fluo_minus']]
stds = grouped_session.sem()[['fluo_plus', 'fluo_minus']]
counts = grouped_session.count()[['fluo_plus', 'fluo_minus']]
#means.plot(by='perc_CR',kind='line',yerr=sems,marker ='o',markersize=15)
#pl.xlim([-.1, 5.1])
# pl.legend(['CR+','CR-'],loc=2)
#pl.xlabel('Fraction of CRs')
# pl.ylabel('DF/F')
pl.figure()
ax = pl.gca()
count = 0
mat_resp = []
mat_resp_std = []
mat_resp_count = []
mat_resp_minus = []

resp_pd = pd.DataFrame()
for mse in means.index.get_level_values(0).unique():
    count += 1
#    ax=pl.subplot(5,1,count)
    mat_resp_minus.append(means.loc[mse]['fluo_minus'].values)

    mat_resp.append(means.loc[mse]['fluo_plus'].values)
    mat_resp_std.append(stds.loc[mse]['fluo_plus'].values)
    mat_resp_count.append(counts.loc[mse]['fluo_plus'].values)

    means.loc[mse]['fluo_plus'].plot(
        ax=ax, marker='o', markersize=15, label=mse, yerr=sems.loc[mse]['fluo_plus'])


fg = pl.figure('MEDIAN')
ax = pl.gca()

mat_resp_count = np.array(mat_resp_count)
mat_resp_std = np.array(mat_resp_std)
mat_resp = np.array(mat_resp)
mat_resp_minus = np.array(mat_resp_minus)

fluo_cr_plus_pd = pd.DataFrame(mat_resp.T, index=cuts.unique())
fluo_cr_minus_pd = pd.DataFrame(mat_resp_minus.T, index=cuts.unique())

fluo_cr_plus_pd.median(1).plot(marker='o', yerr=fluo_cr_plus_pd.mad(1))
fluo_cr_minus_pd.median(1).plot(marker='o', yerr=fluo_cr_minus_pd.mad(1))

pl.xlim([-.5, len(cuts.unique()) + .1])
pl.xticks(np.arange(len(cuts.unique())), cuts.unique())

fg = pl.figure('MEAN')
ax = pl.gca()

fluo_cr_plus_pd.mean(1).plot(marker='o', yerr=fluo_cr_plus_pd.sem(1))
fluo_cr_minus_pd.mean(1).plot(marker='o', yerr=fluo_cr_minus_pd.sem(1))
pl.xlim([-.5, len(cuts.unique()) + .1])
pl.xticks(np.arange(len(cuts.unique())), cuts.unique())
pl.xlabel('amplitude of CR')

pl.figure()
idx1, idx2 = np.where(~np.isnan(mat_resp))
print((scipy.stats.pearsonr(idx2, mat_resp[idx1, idx2].flatten())))
pl.scatter(idx2, mat_resp[idx1, idx2].flatten())

idx = np.where(~np.isnan(np.nanmean(mat_resp_std, 0)))[0]
m_std = [m[~np.isnan(m)] for m in mat_resp_std.T[idx]]
m_count = [m[~np.isnan(m)] for mok in mat_resp_count.T[idx]]
m_mean = [m[~np.isnan(m)] for m in mat_resp.T[idx]]
# print scipy.stats.f_oneway(*m)


#pluss=grouped_session.apply(lambda x: x[['fluo_plus']].values).values
# pl_m=[]
# for pl_ in pluss:
#    pl_=np.array(pl_)
#    pl_m.append(pl_[~np.isnan(pl_)])

print((scipy.stats.f_oneway(*m_mean)))
#%%
import statsmodels.api as sm
print((grouped_session.count()[['fluo_plus', 'fluo_minus']]))
sign_test = scipy.stats.ttest_ind_from_stats(grouped_session.mean()['fluo_plus'], grouped_session.std()['fluo_plus'], grouped_session.count(
)['fluo_plus'], grouped_session.mean()['fluo_minus'], grouped_session.std()['fluo_minus'], grouped_session.count()['fluo_minus'], equal_var=False)
sign_test[1]
sign_test
#%%
X = sm.add_constant(idx2, prepend=False)
y = mat_resp[idx1, idx2].flatten()
model = sm.OLS(y, X)
results = model.fit()
results.summary()
# grouped_session.mean()[['fluo_plus','fluo_minus']]
#
# id1=2
# id2=3
# pd.concat([grouped_session.mean()[['fluo_plus','fluo_minus']],grouped_session.std()[['fluo_plus','fluo_minus']],grouped_session.count()[['fluo_plus','fluo_minus']]],axis=1,keys=['mean_','std_','count_'])
#
#
#
# mouse_stats=pd.concat([grouped_session.mean()[['fluo_plus','fluo_minus']],grouped_session.std()[['fluo_plus','fluo_minus']],grouped_session.count()[['fluo_plus','fluo_minus']]],axis=1,keys=['mean_','std_','count_']).loc['AG051514-01']
#
#scipy.stats.ttest_ind_from_stats(0.349052,0.208635,120.0,0.148381,0.142751,420.0, equal_var=False)[1]


#%%
cr_ampl_m = cr_ampl_sel.copy()
# variable='ampl_eyelid_CSCSUS'
# bins=[-1,.15,.4,1]
##
variable = 'learning_phase'
bins = [0, .1, 1, 2, 3]
#
# variable='perc_CR'
#bins=[0, .15,  .4, .8, 1]

# grouped_session=cr_ampl_m.sort(['perc_CR'],ascending=True)
cr_ampl_m = cr_ampl_m[cr_ampl_m.mouse != 'gc-AG052014-02']
# cr_ampl_m=cr_ampl_m[cr_ampl_m.mouse!='AG052014-01']
cr_ampl_m.dropna()
cuts = pd.cut(cr_ampl_m[variable], bins, include_lowest=True)
grouped_session = cr_ampl_m.groupby(['mouse', cuts])

means = grouped_session.mean()[['fluo_plus', 'fluo_minus']]
sems = grouped_session.sem()[['fluo_plus', 'fluo_minus']]
#%%  create excel file

fluo_whichs = ['fluo_plus', 'fluo_minus']

for fluo_which in fluo_whichs:
    writer = pd.ExcelWriter(
        '/mnt/xfs1/home/agiovann/dropbox/' + fluo_which + '_mean.xlsx')
    ddff = pd.DataFrame(cr_ampl_m.groupby(['mouse', 'session', 'chunk', cuts]).mean()[
                        [fluo_which]].dropna()).to_excel(writer)
    writer.close()

    writer = pd.ExcelWriter(
        '/mnt/xfs1/home/agiovann/dropbox/' + fluo_which + '_std.xlsx')
    ddff = pd.DataFrame(cr_ampl_m.groupby(['mouse', 'session', 'chunk', cuts]).std()[
                        [fluo_which]].dropna()).to_excel(writer)
    writer.close()

    writer = pd.ExcelWriter(
        '/mnt/xfs1/home/agiovann/dropbox/' + fluo_which + '_count.xlsx')
    ddff = pd.DataFrame(cr_ampl_m.groupby(
        ['mouse', 'session', 'chunk', cuts]).count()[fluo_which].dropna())
    ddff[ddff[fluo_which] > 0].to_excel(writer)
    writer.close()

#%%
# cr_ampl_m=cr_ampl_sel
#
# remove mouse not learning
# cr_ampl_m=cr_ampl_m[cr_ampl_m.mouse!='gc-AG052014-02']
# cr_ampl_m=cr_ampl_m[cr_ampl_m.mouse!='AG052014-01']
#
# cuts=pd.cut(cr_ampl_m['learning_phase'],bins=[-1,0,1,2,3],include_lowest=False)
# grouped_session=cr_ampl_m.groupby(['mouse',cuts])
#
# means=grouped_session.mean()[['fluo_plus','fluo_minus']]
# sems=grouped_session.sem()[['fluo_plus','fluo_minus']]
##means.plot(by='perc_CR',kind='line',yerr=sems,marker ='o',markersize=15)
# ax=pl.gca()
# count=0
# mat_resp=[]
# mat_resp_minus=[]
# resp_pd=pd.DataFrame()
# for mse in means.index.get_level_values(0).unique():
#    count+=1
# ax=pl.subplot(5,1,count)
#    mat_resp_minus.append( means.loc[mse]['fluo_minus'].values)
#    mat_resp.append( means.loc[mse]['fluo_plus'].values)
# means.loc[mse]['fluo_plus'].plot(ax=ax,marker='o',markersize=15,label=mse,yerr=sems.loc[mse]['fluo_plus'])
# fg=pl.figure('Median')
# ax=pl.gca()
# mat_resp=np.array(mat_resp)
# mat_resp_minus=np.array(mat_resp_minus)
# fluo_cr_plus_pd=pd.DataFrame(mat_resp.T,index=cuts.unique())
# fluo_cr_minus_pd=pd.DataFrame(mat_resp_minus.T,index=cuts.unique())
#
# fluo_cr_plus_pd.median(1).plot(marker='o',yerr=fluo_cr_plus_pd.mad(1))
# fluo_cr_minus_pd.median(1).plot(marker='o',yerr=fluo_cr_minus_pd.mad(1))
# pl.xlim([-.1,4.1])
# pl.xticks(np.arange(4),cuts.unique())
#
#
# fg=pl.figure('Mean')
# ax=pl.gca()
# fluo_cr_plus_pd.mean(1).plot(marker='o',yerr=fluo_cr_plus_pd.sem(1))
# fluo_cr_minus_pd.mean(1).plot(marker='o',yerr=fluo_cr_minus_pd.sem(1))
# pl.xlim([-.1,4.1])
# pl.xticks(np.arange(4),cuts.unique())


#%%
# cr_ampl_m=cr_ampl_sel
#bins=[0, .2,  .4, .9, 1]
# bins=[0,.15,.4,1]
# bins=[0,.2,.8,1]
#
# grouped_session=cr_ampl_m.sort(['perc_CR'],ascending=True)
# cr_ampl_m=cr_ampl_m[cr_ampl_m.mouse!='gc-AG052014-02']
# cr_ampl_m=cr_ampl_m[cr_ampl_m.mouse!='AG052014-01']
# cuts=pd.cut(cr_ampl_m['ampl_eyelid_CSCSUS'],bins,include_lowest=True)
# grouped_session=cr_ampl_m.groupby(['mouse',cuts])
# means=grouped_session.mean()[['fluo_plus','fluo_minus']]
# sems=grouped_session.sem()[['fluo_plus','fluo_minus']]
##means.plot(by='perc_CR',kind='line',yerr=sems,marker ='o',markersize=15)
##pl.xlim([-.1, 5.1])
# pl.legend(['CR+','CR-'],loc=2)
#pl.xlabel('Fraction of CRs')
# pl.ylabel('DF/F')
# pl.figure()
# ax=pl.gca()
# count=0
# mat_resp=[]
# mat_resp_minus=[]
# resp_pd=pd.DataFrame()
# for mse in means.index.get_level_values(0).unique():
#    count+=1
# ax=pl.subplot(5,1,count)
#    mat_resp_minus.append( means.loc[mse]['fluo_minus'].values)
#    mat_resp.append( means.loc[mse]['fluo_plus'].values)
#    means.loc[mse]['fluo_plus'].plot(ax=ax,marker='o',markersize=15,label=mse,yerr=sems.loc[mse]['fluo_plus'])
# fg=pl.figure('MEDIAN')
# ax=pl.gca()
# mat_resp=np.array(mat_resp)
# mat_resp_minus=np.array(mat_resp_minus)
# fluo_cr_plus_pd=pd.DataFrame(mat_resp.T,index=pd.cut(cr_ampl_m['perc_CR'],bins,include_lowest=True).unique())
# fluo_cr_minus_pd=pd.DataFrame(mat_resp_minus.T,index=pd.cut(cr_ampl_m['perc_CR'],bins,include_lowest=True).unique())
#
# fluo_cr_plus_pd.median(1).plot(marker='o',yerr=fluo_cr_plus_pd.mad(1))
# fluo_cr_minus_pd.median(1).plot(marker='o',yerr=fluo_cr_minus_pd.mad(1))
# pl.xlim([-.1,4.1])
# pl.xticks(np.arange(5),pd.cut(cr_ampl_m['perc_CR'],bins,include_lowest=True).unique())
#
# fg=pl.figure('MEAN')
# ax=pl.gca()
# fluo_cr_plus_pd.mean(1).plot(marker='o',yerr=fluo_cr_plus_pd.sem(1))
# fluo_cr_minus_pd.mean(1).plot(marker='o',yerr=fluo_cr_minus_pd.sem(1))
# pl.xlim([-.1,4.1])
# pl.xticks(np.arange(5),pd.cut(cr_ampl_m['perc_CR'],bins,include_lowest=True).unique())
#pl.xlabel('amplitude of CR')


#%%
# pl.errorbar(np.arange(5),np.nanmedian(mat_resp,0),mad(mat_resp,0),'o')

# for mse in means.index.get_level_values(0).unique():
#    count+=1
# ax=pl.subplot(5,1,count)
#    means.loc[mse]['fluo_minus'].plot(ax=ax,marker='o',markersize=15,alpha=.3,label=mse,yerr=sems.loc[mse]['fluo_minus'])
#
# pl.xlim([-.3,4.3])
# pl.legend(loc=2)
pl.rc('font', **font)
#%%

#%%
# bins=np.arange(0,1,.05)
# active_during_wheel=pl.hist(np.array(bh_correl.active_during_wheel.dropna().values,dtype='float'),bins)[0]
# active_during_CS=pl.hist(np.array(bh_correl.active_during_CS.dropna().values,dtype='float'),bins)[0]
# active_during_UR=pl.hist(np.array(bh_correl.active_during_UR.dropna().values,dtype='float'),bins)[0]
# active_during_nose=pl.hist(np.array(bh_correl.active_during_nose.dropna().values,dtype='float'),bins)[0]
# active_during_CR=pl.hist(np.array(bh_correl.active_during_CR.dropna().values),bins)[0]
# active_during_UR_NOCR=pl.hist(np.array(bh_correl.active_during_UR_NOCR.dropna().values,dtype='float'),bins)[0]
# active_during_wheel=pl.hist(np.array(bh_correl.r_wheel_fluo.dropna().values,dtype='float'),bins)[0]
# active_during_CS=pl.hist(np.array(bh_correl.r_eye_fluo.dropna().values,dtype='float'),bins)[0]
# active_during_UR=pl.hist(np.array(bh_correl.r_UR_eye_fluo.dropna().values,dtype='float'),bins)[0]
# active_during_nose=pl.hist(np.array(bh_correl.r_nose_fluo.dropna().values,dtype='float'),bins)[0]
# active_during_CR=pl.hist(np.array(bh_correl.r_CR_eye_fluo.dropna().values),bins)[0]
#
# pl.close()
#all_hists=np.vstack([active_during_wheel,active_during_CS, active_during_UR, active_during_nose, active_during_CR])
# all_hists=scipy.signal.savgol_filter(all_hists,5,1,axis=1).T
# all_hists=all_hists/np.nansum(all_hists,0)[None,:]
# all_hists=np.cumsum(all_hists,axis=0)
#pl.plot(bins[:-1],all_hists )
#pl.legend(['wheel','CS' ,'UR' ,'NOSE' , 'CR','UR_NOCR'])
# pl.xlim([0,1])
# pl.ylim([0.6,1])
#
#pl.rc('font', **font)

#%%

bh_correl_tmp = bh_correl_sel.copy()
for r_ in bh_correl_tmp.keys()[bh_correl_tmp.keys().str.contains('r_.*rnd')]:
    borders_low = bh_correl_tmp.fillna(method='pad').groupby('mouse')[
        r_].quantile(.05)
    borders_high = bh_correl_tmp.fillna(method='pad').groupby('mouse')[
        r_].quantile(.95)

    pl.figure(r_[2:-9], figsize=(14, 14))
    bh_correl_tmp[r_[:-4]].hist(by=bh_correl_tmp['mouse'],
                                bins=30, histtype='step', normed='True', ax=pl.gca())

    for count, b_l, b_h in zip(list(range(len(borders_low))), borders_low, borders_high):
        pl.subplot(3, 2, count + 1)
        mx = pl.gca().get_ylim()[-1]
        pl.fill_between([b_l, b_h], [0, 0], [mx, mx],
                        facecolor='green', alpha=.5)
        pl.xlabel('correlation')
        pl.ylabel('frequency')
#        pl.xlim([0,None])
        pl.rc('font', **font)
#        pl.savefig('hist_'+r_[:-4]+'.pdf')

    pl.legend(['measured', 'shuffled'])
    pl.rc('font', **font)
#%% show the reponses of neurons to each of the conditions
results_corr = pd.DataFrame()
bh_correl_tmp = bh_correl_sel.copy()
bh_correl_tmp = bh_correl_tmp[bh_correl_tmp['learning_phase'] >= 0].copy()
for r_ in bh_correl_tmp.keys()[bh_correl_tmp.keys().str.contains('r_.*rnd')]:
    #    print bh_correl_tmp.fillna(method='pad').groupby('mouse')[r_].quantile(.95).values
    bh_correl_tmp[r_] = bh_correl_tmp.fillna(method='pad').groupby('mouse')[
        r_].transform(lambda x: x.quantile(.99))
    bh_correl_tmp[r_[:-4]] = bh_correl_tmp[r_[:-4]] - bh_correl_tmp[r_]
#    print bh_correl_tmp.fillna(method='pad').groupby('mouse')[r_].quantile(.95).values.T
    results_corr = pd.concat([results_corr, bh_correl_tmp.fillna(method='pad').groupby(
        'mouse')[r_[:-4]].agg({r_[2:-9]:lambda x: np.nanmean(x >= 0)})], axis=1)
#%% CHOOSE LAST SESSIONS
results_corr = pd.DataFrame()
#bh_correl_tmp=bh_correl.fillna(method='pad')[(bh_correl.session_id==8.0)  | (np.isnan(bh_correl.session_id))  | ((bh_correl.mouse=='b35') & (bh_correl.session=='20160714143248') & (bh_correl.chunk=='061')) & (bh_correl.mouse!='b35')]
idx_last_sessions = (((bh_correl_sel.mouse == 'b37') & (bh_correl_sel.session == '20160705103903') & (bh_correl_sel.chunk == '121')) |
                     ((bh_correl_sel.mouse == 'b35') & (bh_correl_sel.session == '20160714143248') & (bh_correl_sel.chunk == '061')) |
                     ((bh_correl_sel.mouse == 'AG051514-01')) |
                     ((bh_correl_sel.mouse == 'AG052014-01')) |\
                     #                              ((bh_correl_sel.mouse=='gc-AGGC6f-031213-03')) |\
                     ((bh_correl_sel.mouse == 'gc-AG052014-02')))


bh_correl_tmp = bh_correl_sel[idx_last_sessions].copy()
for r_ in bh_correl_tmp.keys()[bh_correl_tmp.keys().str.contains('r_.*rnd')]:
    #    print bh_correl_tmp.fillna(method='pad').groupby('mouse')[r_].quantile(.95).values
    bh_correl_tmp[r_] = bh_correl_tmp.fillna(method='pad').groupby('mouse')[
        r_].transform(lambda x: x.quantile(.99))
    bh_correl_tmp[r_[:-4]] = bh_correl_tmp[r_[:-4]] - bh_correl_tmp[r_]

#    print bh_correl_tmp.fillna(method='pad').groupby('mouse')[r_].quantile(.95).values.T
    #results_corr=pd.concat([results_corr,bh_correl_tmp.fillna(method='pad').groupby('mouse')[r_[:-4]].agg({r_[2:-9]:lambda x: np.nanmean(x>=0)})],axis=1)
#%%
#from collections import Counter
#last_sessions = bh_correl_tmp.copy()
##last_sessions_tmp = last_sessions.sort_values(['r_CR_eye_fluo'])[['r_CR_eye_fluo','r_nose_fluo','r_wheel_fluo','r_UR_eye_fluo']]
##last_sessions_tmp = last_sessions.sort_values(['r_CR_eye_fluo'])[['mouse','r_CR_eye_fluo','r_nose_fluo','r_wheel_fluo','r_UR_eye_fluo']]
#last_sessions_tmp = last_sessions[['mouse','r_CR_eye_fluo','r_nose_fluo','r_wheel_fluo','r_UR_eye_fluo']]
##last_sessions_tmp = last_sessions_tmp.groupby('mouse')
# for mouse_ in last_sessions_tmp.mouse.unique():
#    print mouse_
#    aggr_mouse = last_sessions_tmp[last_sessions_tmp.mouse==mouse_][['r_CR_eye_fluo','r_nose_fluo','r_wheel_fluo','r_UR_eye_fluo']].idxmax(axis=1)
#    counts = Counter(aggr_mouse.values)
#    print dict(counts)

#    df = pd.DataFrame.from_dict(counts, orient='index')
#    df.plot(kind='bar')
#%%
probabilities = pd.DataFrame(index=['CR', 'NOSE', 'WHEEL', 'UR', 'Unrelated'])
for mouse in bh_correl_tmp.mouse.unique():
    last_sessions = bh_correl_tmp[bh_correl_tmp.mouse == mouse]
    last_sessions = last_sessions.sort_values(['r_CR_eye_fluo'])[
        ['r_CR_eye_fluo', 'r_nose_fluo', 'r_wheel_fluo', 'r_UR_eye_fluo']].values

    thresh_corr = 0
    thresh_corr_neg = 1
    idx = np.min(last_sessions, 1) > -thresh_corr_neg
    last_sessions = last_sessions[idx]
    bests = np.argmax(last_sessions, 1)
    bests[np.max(last_sessions, 1) <= thresh_corr] = 4
    best_cr = last_sessions[(last_sessions[:, 0] >
                             thresh_corr) & (bests == 0)].T
    best_nose = last_sessions[(
        last_sessions[:, 1] > thresh_corr) & (bests == 1)].T
    best_wheel = last_sessions[(
        last_sessions[:, 2] > thresh_corr) & (bests == 2)].T
    best_ur = last_sessions[(last_sessions[:, 3] >
                             thresh_corr) & (bests == 3)].T
    unrelated = last_sessions[bests == 4].T
    all_neuron = len(last_sessions)

#    pl.bar([0,1,2,3,4],np.array([len(best_cr.T),len(best_nose.T),len(best_wheel.T),len(best_ur.T),len(unrelated.T)])*1./all_neuron,width=[-.1])
#    pl.xlim([-.5,4.5])
#    pl.xticks([0,1,2,3,4],['cr','nose','wheel','ur','unrelated'])
#    pl.title('All significant cells')
    probabilities[mouse] = (np.array([len(best_cr.T), len(best_nose.T), len(
        best_wheel.T), len(best_ur.T), len(unrelated.T)]) * 1. / all_neuron).T

pl.subplot(3, 1, 1)
probabilities.mean(axis=1).plot(kind='bar', yerr=probabilities.sem(axis=1))
pl.title('All mice (N=5)')
pl.subplot(3, 1, 2)
# probabilities[['b37','gc-AGGC6f-031213-03','AG052014-01','AG051514-01']].mean(axis=1).plot(kind='bar',yerr=probabilities.sem(axis=1))
probabilities[['b37', 'AG052014-01', 'AG051514-01', 'gc-AG052014-02']].mean(axis=1).plot(
    kind='bar', yerr=probabilities[['b37', 'AG052014-01', 'AG051514-01', 'gc-AG052014-02']].sem(axis=1))

pl.title('Only responding > 30 % (N=4)')
pl.subplot(3, 1, 3)
# probabilities[['b37','gc-AGGC6f-031213-03','AG052014-01','AG051514-01']].mean(axis=1).plot(kind='bar',yerr=probabilities.sem(axis=1))
probabilities[['b37', 'AG052014-01', 'AG051514-01']].mean(axis=1).plot(
    kind='bar', yerr=probabilities[['b37', 'AG052014-01', 'AG051514-01']].sem(axis=1))

pl.ylabel('Probability in each FOV')
pl.title('Only responding > 60 %(N=3)')
#%%
pl.subplot(5, 1, 1)

all_neuron = len(last_sessions)

pl.bar([0, 1, 2, 3, 4], np.array([len(best_cr.T), len(best_nose.T), len(
    best_wheel.T), len(best_ur.T), len(unrelated.T)]) * 1. / all_neuron, width=[-.1])
pl.xlim([-.5, 4.5])
pl.xticks([0, 1, 2, 3, 4], ['cr', 'nose', 'wheel', 'ur', 'unrelated'])
pl.title('All significant cells')
#

#%%
#last_sessions = bh_correl_tmp
#last_sessions = last_sessions.sort_values(['r_CR_eye_fluo'])[['r_CR_eye_fluo','r_nose_fluo','r_wheel_fluo','r_UR_eye_fluo']].values
# pl.figure()
#
# thresh_corr=0
# thresh_corr_neg=1
# idx=np.min(last_sessions,1)>-thresh_corr_neg
# last_sessions=last_sessions[idx]
# bests=np.argmax(last_sessions,1)
# bests[np.max(last_sessions,1)<=thresh_corr]=4
#best_cr=last_sessions[(last_sessions[:,0]>thresh_corr) & (bests==0)].T
#best_nose=last_sessions[(last_sessions[:,1]>thresh_corr) & (bests==1)].T
#best_wheel=last_sessions[(last_sessions[:,2]>thresh_corr) & (bests==2)].T
#best_ur=last_sessions[(last_sessions[:,3]>thresh_corr) & (bests==3)].T
# unrelated=last_sessions[bests==4].T


pl.subplot(5, 1, 1)

# all_neuron=len(last_sessions)

# pl.bar([0,1,2,3,4],np.array([len(best_cr.T),len(best_nose.T),len(best_wheel.T),len(best_ur.T),len(unrelated.T)])*1./all_neuron,width=[-.1])

# probabilities.mean(axis=1).plot(kind='bar',yerr=probabilities.sem(axis=1))
# probabilities[['b35','b37','AG052014-01','AG051514-01','gc-AG052014-02']].mean(axis=1).plot(kind='bar',yerr=probabilities[['b35','b37','AG052014-01','AG051514-01','gc-AG052014-02']].sem(axis=1))
probabilities[['b37', 'AG052014-01', 'AG051514-01', 'gc-AG052014-02']].mean(axis=1).plot(
    kind='bar', yerr=probabilities[['b37', 'AG052014-01', 'AG051514-01', 'gc-AG052014-02']].sem(axis=1))


pl.xlim([-.5, 4.5])
pl.xticks([0, 1, 2, 3, 4], ['cr', 'nose', 'wheel', 'ur', 'unrelated'])
pl.title('All significant cells')


#last_sessions= bh_correl_tmp_tmp.fillna(method='pad')[(bh_correl_tmp.session_id==8.0)  | (np.isnan(bh_correl_tmp.session_id))  | ((bh_correl_tmp.mouse=='b35') & (bh_correl_tmp.session=='20160714143248') & (bh_correl_tmp.chunk=='061'))]
#last_sessions=bh_correl.fillna(method='pad')[(bh_correl.session_id==8.0)  | (np.isnan(bh_correl.session_id))  | ((bh_correl.mouse=='b35') & (bh_correl.session=='20160714143248') & (bh_correl.chunk=='061'))]
last_sessions = bh_correl_sel[idx_last_sessions].copy().fillna(method='pad')
last_sessions.groupby('mouse')['perc_CR'].count()
last_sessions = last_sessions.sort_values(['r_CR_eye_fluo'])[
    ['r_CR_eye_fluo', 'r_nose_fluo', 'r_wheel_fluo', 'r_UR_eye_fluo']].values
thresh_corr = 0.2
thresh_corr_neg = .3
idx = np.min(last_sessions, 1) > -thresh_corr_neg
last_sessions = last_sessions[idx]
bests = np.argmax(last_sessions, 1)
bests[np.max(last_sessions, 1) <= thresh_corr] = 4
best_cr = last_sessions[(last_sessions[:, 0] > thresh_corr) & (bests == 0)].T
best_nose = last_sessions[(last_sessions[:, 1] > thresh_corr) & (bests == 1)].T
best_wheel = last_sessions[(
    last_sessions[:, 2] > thresh_corr) & (bests == 2)].T
best_ur = last_sessions[(last_sessions[:, 3] > thresh_corr) & (bests == 3)].T
unrelated = last_sessions[bests == 4].T


pl.subplot(5, 1, 2)
pl.plot(best_cr, 'o-', color='lightgray', linewidth=1)
pl.title('CR best')
pl.xlim([-.5, 4.5])
pl.ylim([-thresh_corr_neg, .8])
pl.ylabel('Pearson\'s r')

pl.subplot(5, 1, 3)
pl.plot(best_nose, 'o-', color='lightgray', linewidth=1)
pl.title('Nose best')
pl.xlim([-.5, 4.5])
pl.ylim([-thresh_corr_neg, .8])

pl.ylabel('Pearson\'s r')

pl.subplot(5, 1, 4)
pl.plot(best_wheel, 'o-', color='lightgray', linewidth=1)
pl.title('Wheel best')
pl.xlim([-.5, 4.5])
pl.ylim([-thresh_corr_neg, .8])

pl.ylabel('Pearson\'s r')

pl.subplot(5, 1, 5)
pl.plot(best_ur, 'o-', color='lightgray', linewidth=1)
pl.title('UR best')
pl.xlim([-.5, 4.5])
pl.ylim([-thresh_corr_neg, .8])

pl.ylabel('Pearson\'s r')

# pl.subplot(6,1,6)
# pl.plot(unrelated,'o-',color='gray')
# pl.title('unrelated')
# pl.xlim([-.5,4.5])
#pl.ylim([-thresh_corr_neg, .8])
#
#pl.ylabel('Pearson\'s r')

pl.rc('font', **font)

#pl.figure('Cells selectivity all',figsize=(6,12))
# results_corr.mean().plot(kind='bar',yerr=results_corr.sem(),ax=pl.gca())
# pl.xlabel('behavior')
#pl.ylabel('percentage of neurons > 95th percentile')
#
#pl.figure('Cells selectivity_each',figsize=(6,12))
# results_corr.T.plot(kind='bar',ax=pl.gca())
# pl.xlabel('behavior')
#pl.ylabel('percentage of neurons > 95th percentile')
#%% respond to two conditions
import itertools
resp_couples = bh_correl_tmp.copy()
couples = ['r_CR_eye_fluo', 'r_nose_fluo', 'r_UR_eye_fluo', 'r_wheel_fluo']


nm_c = itertools.combinations(['CR', 'nose', 'UR', 'wheel'], 1)
for cp in itertools.combinations(couples, 1):
    resp_couples['R_' + nm_c.next()[0]] = (bh_correl_tmp[cp[0]] >= 0)

nm_c = itertools.combinations(['CR', 'nose', 'UR', 'wheel'], 2)
for cp in itertools.combinations(couples, 2):
    cp_ = next(nm_c)
    resp_couples['R_' + cp_[0] + '_' + cp_[1]
                 ] = (bh_correl_tmp[cp[0]] >= 0) & (bh_correl_tmp[cp[1]] >= 0)

nm_c = itertools.combinations(['CR', 'nose', 'UR', 'wheel'], 3)
for cp in itertools.combinations(couples, 3):
    cp_ = next(nm_c)
    resp_couples['R_' + cp_[0] + '_' + cp_[1] + '_' + cp_[2]] = (
        bh_correl_tmp[cp[0]] >= 0) & (bh_correl_tmp[cp[1]] >= 0) & (bh_correl_tmp[cp[2]] >= 0)
#%%
pl.figure('Cells responding three conditions all', figsize=(12, 12))

resp_couples_agg = resp_couples.groupby('mouse').mean().filter(regex="^R_")
resp_couples_agg.filter(regex="^R_").plot(kind='bar', ax=pl.gca())
labels = pl.gca().get_xticklabels()
pl.setp(labels, rotation=0)
pl.rc('font', **font)
pl.figure('Cells responding three conditions each', figsize=(12, 12))
# pl.savefig('cell_selectivity_all.pdf')

resp_couples_agg.mean().plot(kind='bar', yerr=resp_couples_agg.sem(), ax=pl.gca())
labels = pl.gca().get_xticklabels()
pl.setp(labels, rotation=30)
pl.rc('font', **font)
# pl.savefig('cell_selectivity_each.pdf')
#%%
counter = 0
pl.figure(figsize=(12, 12))
legends = []
for r_ in bh_correl.keys()[bh_correl.keys().str.contains('r_.*rnd')]:
    counter += 1

    ax = pl.subplot(3, 2, counter)
    bh_correl[r_[:-4]].hist(bins=30, histtype='step', normed='True', ax=ax)
    bh_correl[r_].hist(bins=30, histtype='step', normed='True', ax=ax)
    bords_ = np.nanpercentile(bh_correl[r_].values.astype(np.float32), [5, 95])
    pl.fill_between(bords_, [0, 0], [12, 12], facecolor='green', alpha=.5)
    pl.title(r_[:-4])

#%%
# bh_correl_tmp[['r_nose_fluo','r_nose_fluo_rnd']].hist(by=bh_correl_tmp['mouse'],bins=30,histtype='step',normed='True')
# bh_correl_tmp[['r_wheel_fluo','r_wheel_fluo_rnd']].hist(by=bh_correl_tmp['mouse'],bins=30,histtype='step',normed='True')
# bh_correl_tmp[['r_UR_eye_fluo','r_UR_eye_fluo_rnd']].hist(by=bh_correl_tmp['mouse'],bins=30,histtype='step',normed='True')
#
# bh_correl_tmp=bh_correl[bh_correl['learning_phase']>1]
# bh_correl_tmp[['r_CR_eye_fluo','r_CR_eye_fluo_rnd']].hist(by=bh_correl_tmp['mouse'],bins=30,histtype='step',normed='True')
#%%
bh_correl_tmp = bh_correl[bh_correl['learning_phase'] >= 0]
bh_correl_tmp['active_during_nose'].hist(
    by=bh_correl_tmp['mouse'], bins=30, histtype='step', normed='True')
bh_correl_tmp['active_during_wheel'].hist(
    by=bh_correl_tmp['mouse'], bins=30, histtype='step', normed='True')
bh_correl_tmp['active_during_CS'].hist(
    by=bh_correl_tmp['mouse'], bins=30, histtype='step', normed='True')
bh_correl_tmp['active_during_UR'].hist(
    by=bh_correl_tmp['mouse'], bins=30, histtype='step', normed='True')

bh_correl_tmp = bh_correl[bh_correl['learning_phase'] > 1]
bh_correl_tmp[['active_during_CR']].hist(
    by=bh_correl_tmp['mouse'], bins=30, histtype='step', normed='True')
#    pl.figure()
#    group.plot(x='neuron_id', y='r_nose_fluo', title=str(i))
#%%
cumulative = False
show_neg = True

pl.subplot(2, 2, 1)
bh_correl.groupby(['mouse', 'neuron_id']).r_nose_fluo.mean().plot(
    kind='hist', x='mouse', bins=50, normed=True, histtype='step', cumulative=cumulative)
bh_correl.r_nose_fluo_rnd.plot(
    kind='hist', bins=50, normed=True, histtype='step', cumulative=cumulative)

bords_nose = np.nanpercentile(
    bh_correl.r_nose_fluo_rnd.values.astype(np.float32), [5, 95])
pl.fill_between(bords_nose, [0, 0], [1, 1], facecolor='green', alpha=.5)


pl.xlabel('Correlation')
pl.ylabel('Percentage of neurons')
pl.title('Nose  vs fluo')
if not show_neg:
    pl.xlim([0, 1])
# pl.ylim([0,2.5])
pl.subplot(2, 2, 2)
bh_correl.r_wheel_fluo.plot(
    kind='hist', bins=50, normed=True, histtype='step', cumulative=cumulative)
bh_correl.r_wheel_fluo_rnd.plot(
    kind='hist', bins=50, normed=True, histtype='step', cumulative=cumulative)

bords_wheel = np.nanpercentile(
    bh_correl.r_wheel_fluo_rnd.values.astype(np.float32), [5, 95])
pl.fill_between(bords_wheel, [0, 0], [1, 1], facecolor='green', alpha=.5)

pl.title('Wheel  vs fluo')
pl.xlabel('Correlation')
pl.ylabel('Percentage of neurons')
if not show_neg:
    pl.xlim([0, 1])

# pl.ylim([0,2.5])


pl.subplot(2, 2, 3)
bh_correl.r_UR_eye_fluo.plot(
    kind='hist', bins=50, normed=True, histtype='step', cumulative=cumulative)
bh_correl.r_UR_eye_fluo_rnd.plot(
    kind='hist', bins=50, normed=True, histtype='step', cumulative=cumulative)
bords_UR = np.nanpercentile(
    bh_correl.r_UR_eye_fluo_rnd.values.astype(np.float32), [5, 95])
pl.fill_between(bords_UR, [0, 0], [1, 1], facecolor='green', alpha=.5)


pl.title('UR  vs fluo')
pl.xlabel('Correlation')
pl.ylabel('Percentage of neurons')
if not show_neg:
    pl.xlim([0, 1])
# pl.ylim([0,2.5])

pl.subplot(2, 2, 4)
bh_correl.r_CR_eye_fluo.plot(
    kind='hist', bins=100, normed=True, histtype='step', cumulative=cumulative)
bh_correl.r_CR_eye_fluo_rnd.plot(
    kind='hist', bins=100, normed=True, histtype='step', cumulative=cumulative)
bords_CR = np.nanpercentile(
    bh_correl.r_CR_eye_fluo_rnd.values.astype(np.float32), [5, 95])
pl.fill_between(bords_CR, [0, 0], [10, 10], facecolor='green', alpha=.5)

# bh_correl_tmp.delta_norm.plot(kind='hist',bins=200,normed=True)
# bh_correl.delta_norm.plot(kind='hist',bins=30,normed=True)
pl.title('CR vs fluo')
pl.legend(['Measured (p<0.05)', 'Random shuffle (5th - 95th percentile)'], loc=2)
pl.xlabel('Correlation')
pl.ylabel('Percentage of neurons')
if not show_neg:
    pl.xlim([0, 1])
# pl.ylim([0,5])
#%%
bh_correl_tmp = bh_correl.copy()
bh_correl_tmp['respond_to_wheel'] = (
    bh_correl_tmp['r_wheel_fluo'] > bords_UR[-1])
print((bh_correl_tmp.groupby('mouse')['respond_to_wheel'].mean()))


# print np.mean(bh_correl_tmp['r_CR_eye_fluo']>bords_CR[-1])
# print np.mean(bh_correl_tmp['r_wheel_fluo']>bords_wheel[-1])
# print np.mean(bh_correl_tmp['r_nose_fluo']>bords_nose[-1])
# np.mean(bh_correl_tmp['r_eye_fluo']>bords_eye[-1])


#%%
from itertools import cycle, islice
cr_ampl_m = cr_ampl_sel.copy()

grouped_session = cr_ampl_m.groupby(['mouse', pd.cut(cr_ampl_m['perc_CR'], bins, include_lowest=True), pd.cut(
    cr_ampl_m['fluo_plus'], [0, .2, .8, 1.5], include_lowest=True)])
pl.close('all')

for counter, ms in enumerate(cr_ampl_m.mouse.unique()):
    cr_now = cr_ampl_m[cr_ampl_m.mouse == ms]
    if 'b3' in ms:
        cr_now = cr_now[cr_now.session_id >=
                        np.maximum(0, cr_now.session_id.max() - 3)]
    else:
        if ms == 'AG052014-01':
            print('*')
            cr_now = cr_now[cr_now.session_id ==
                            np.maximum(0, cr_now.session_id.max() - 3)]
        elif ms == 'gc-AG052014-02':
            print('**')
            cr_now = cr_now[cr_now.session_id >=
                            np.maximum(0, cr_now.session_id.max() - 5)]
        elif ms == 'AG051514-01':
            print('**')
            cr_now = cr_now[cr_now.session_id >=
                            np.maximum(0, cr_now.session_id.max() - 2)]
        else:
            cr_now = cr_now[cr_now.session_id ==
                            np.maximum(0, cr_now.session_id.max())]

    cr_now['delta'] = (cr_now['fluo_plus'] - cr_now['fluo_minus'])
    cr_now['delta_norm'] = old_div(
        (cr_now['fluo_plus'] - cr_now['fluo_minus']), np.abs(cr_now['fluo_plus'] + cr_now['fluo_minus']))
    cr_now['fluo_minus'] = -cr_now['fluo_minus']
    cr_now = cr_now[cr_now.fluo_minus <= 0]
    cr_now = cr_now[cr_now.delta_norm > 0]
    ax = pl.subplot(1, 6, counter + 1)
    cr_now = cr_now.sort('delta')
    my_colors = list(islice(cycle(['k', 'r']), None, len(cr_now)))
    cr_now[['delta', 'fluo_minus']].dropna(
        0)[-61:-1].plot(ax=ax, kind='bar', stacked=True, color=my_colors, width=.9, edgecolor='none')
    pl.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off')  # labels along the bottom edge are off
    ax.legend().remove()
    pl.ylabel('DF/F')
    pl.ylim([-1.4, 1.4])
#    cr_now[['fluo_plus','fluo_minus']].dropna(0).sort('fluo_plus')[-61:-1].plot(kind='bar',stacked=True,color=my_colors,width=.9)
    pl.title(ms)
    pl.xlabel('Granule cells')


pl.legend(['CR+-CR-', 'CR-'])
#%%
bins = [0, .1, .5, 1]
grouped_session = cr_ampl.groupby(
    [pd.cut(cr_ampl_m['ampl_eyelid_CSCSUS'], bins, include_lowest=True)])
means = grouped_session.mean()[['fluo_plus', 'fluo_minus']]
sems = grouped_session.sem()[['fluo_plus', 'fluo_minus']]
means.plot(kind='line', yerr=sems, marker='o',
           xticks=list(range(3)), markersize=15)
#pl.xlim([-.1, 2.1])
pl.legend(['CR+', 'CR-'], loc=3)
pl.xlabel('Fraction of CRs')
pl.ylabel('DF/F')

pl.rc('font', **font)
