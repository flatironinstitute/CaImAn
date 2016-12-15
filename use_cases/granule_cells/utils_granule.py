# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 17:56:14 2016

@author: agiovann
"""
from __future__ import division
from __future__ import print_function
from past.builtins import basestring
from builtins import zip
from builtins import map
from builtins import str
from builtins import range
from past.utils import old_div
import os
import cv2
import h5py
import numpy as np
import pylab as pl
from glob import glob
# import ca_source_extraction as cse
import caiman as cb
from scipy import signal
import scipy
import sys
from ipyparallel import Client
from time import time
from scipy.sparse import csc,csr,coo_matrix
from scipy.spatial.distance import cdist
from scipy import ndimage
from scipy.optimize import linear_sum_assignment   
from sklearn.utils.linear_assignment_ import linear_assignment    
import re
import pickle
#%% Process triggers
def extract_triggers(file_list,read_dictionaries=False): 

    """Extract triggers from Bens' tiff file and create readable dictionaries

    Parameterskdkd
    -----------
    file_list: list of tif files or npz files containing the iage description

    Returns
    -------
    triggers: list 
        [idx_CS, idx_US, trial_type, number_of_frames]. Trial types: 0 CS alone, 1 US alone, 2 CS US

   trigger_names: list
        file name associated (without extension)

    Example: 

    fls=glob.glob('2016*.tif')     
    fls.sort()     
    triggers,trigger_names=extract_triggers(fls[:5],read_dictionaries=False)     
    np.savez('all_triggers.npz',triggers=triggers,trigger_names=trigger_names)     

    """
    triggers=[]

    trigger_names=[]

    for fl in file_list:

        print(fl)   

        fn=fl[:-4]+'_ImgDescr.npz'

        if read_dictionaries:

            with np.load(fn) as idr:

                image_descriptions=idr['image_descriptions']

        else:

            image_descriptions=cb.utils.get_image_description_SI(fl)

            print('*****************')

            np.savez(fn,image_descriptions=image_descriptions)


        trig_vect=np.zeros(4)*np.nan    

        for idx,image_description in enumerate(image_descriptions): 

            i2cd=image_description['I2CData']

            if isinstance(i2cd,basestring):

                if i2cd.find('US_ON')>=0:

                    trig_vect[1]=image_description['frameNumberAcquisition']-1

                if i2cd.find('CS_ON')>=0:

                    trig_vect[0]=image_description['frameNumberAcquisition']-1  


        if np.nansum(trig_vect>0)==2:

            trig_vect[2]=2

        elif trig_vect[0]>0:

            trig_vect[2]=0

        elif trig_vect[1]>0:

            trig_vect[2]=1  
        else:
            raise Exception('No triggers present in trial')        

        trig_vect[3]=idx+1

        triggers.append(trig_vect)

        trigger_names.append(fl[:-4])

        print((triggers[-1]))

    return triggers,trigger_names





#%%
def downsample_triggers(triggers,fraction_downsample=1):
    """ downample triggers so as to make them in line with the movies
    Parameters
    ----------

    triggers: list=Ftraces[idx]
        output of  extract_triggers function

    fraction_downsample: float
        fraction the data is shrinked in the time axis
    """

    triggers[:,[0,1,3]]=np.round(triggers[:,[0,1,3]]*fraction_downsample)
#    triggers[-1,[0,1,3]]=np.floor(triggers[-1,[0,1,3]]*fraction_downsample)
#    triggers[-1]=np.cumsum(triggers[-1])

#    real_triggers=triggers[:-1]+np.concatenate([np.atleast_1d(0), triggers[-1,:-1]])[np.newaxis,:]
#
#    trg=real_triggers[1][triggers[-2]==2]+np.arange(-5,8)[:,np.newaxis]  
#
#    trg=np.int64(trg)

    return triggers

#%%
def get_behavior_traces(fname,t0,t1,freq,ISI,draw_rois=False,plot_traces=False,mov_filt_1d=True,window_hp=201,window_lp=3,interpolate=True,EXPECTED_ISI=.25):
    """
    From hdf5 movies extract eyelid closure and wheel movement


    Parameters
    ----------
    fname: str    
        file name of the hdf5 file

    t0,t1: float. 
        Times of beginning and end of trials (in general 0 and 8 for our dataset) to build the absolute time vector

    freq: float
        frequency used to build the final time vector    

    ISI: float
        inter stimulu interval

    draw_rois: bool
        whether to manually draw the eyelid contour

    plot_traces: bool
        whether to plot the traces during extraction        

    mov_filt_1d: bool 
        whether to filter the movie after extracting the average or ROIs. The alternative is a 3D filter that can be very computationally expensive

    window_lp, window_hp: ints
        number of frames to be used to median filter the data. It is needed because of the light IR artifact coming out of the eye

    Returns
    -------
    res: dict
        dictionary with fields 
            'eyelid': eyelid trace
            'wheel': wheel trace
            'time': absolute tim vector
            'trials': corresponding indexes of the trials
            'trial_info': for each trial it returns start trial, end trial, time CS, time US, trial type  (CS:0 US:1 CS+US:2)
            'idx_CS_US': idx trial CS US
            'idx_US': idx trial US
            'idx_CS': idx trial CS 
    """
    CS_ALONE=0
    US_ALONE=1
    CS_US=2
    meta_inf = fname[:-7]+'data.h5'

    time_abs=np.linspace(t0,t1,freq*(t1-t0))

    T=len(time_abs)
    t_us=0
    t_cs=0
    n_samples_ISI=np.int(ISI*freq)
    t_uss=[]
    ISIs=[]
    eye_traces=[]
    wheel_traces=[]
    trial_info=[]
    tims=[]
    with h5py.File(fname) as f:

        with h5py.File(meta_inf) as dt:

            rois=np.asarray(dt['roi'],np.float32)

            trials = list(f.keys())

            trials.sort(key=lambda x: np.int(x.replace('trial_','')))

            trials_idx=[np.int(x.replace('trial_',''))-1 for x in trials]

            trials_idx_=[]



            for tr,idx_tr in zip(trials[:],trials_idx[:]):
                if plot_traces:
                    pl.cla()

                print(tr)



                trial=f[tr]  

                mov=np.asarray(trial['mov'])        

                if draw_rois:

                    pl.imshow(np.mean(mov,0))
                    pl.xlabel('Draw eye')
                    pts=pl.ginput(-1)

                    pts = np.asarray(pts, dtype=np.int32)

                    data = np.zeros(np.shape(mov)[1:], dtype=np.int32)
            #        if CV_VERSION == 2:
                    #lt = cv2.CV_AA
            #        elif CV_VERSION == 3:
                    lt = cv2.LINE_AA

                    cv2.fillConvexPoly(data, pts, (1,1,1), lineType=lt)

                    rois[0]=data

                    pl.close()

                    pl.imshow(np.mean(mov,0))
                    pl.xlabel('Draw wheel')            
                    pts=pl.ginput(-1)

                    pts = np.asarray(pts, dtype=np.int32)

                    data = np.zeros(np.shape(mov)[1:], dtype=np.int32)
            #        if CV_VERSION == 2:
                    #lt = cv2.CV_AA
            #        elif CV_VERSION == 3:
                    lt = cv2.LINE_AA

                    cv2.fillConvexPoly(data, pts, (1,1,1), lineType=lt)

                    rois[1]=data

                    pl.close()
    #            eye_trace=np.mean(mov*rois[0],axis=(1,2))
    #            mov_trace=np.mean((np.diff(np.asarray(mov,dtype=np.float32),axis=0)**2)*rois[1],axis=(1,2))
                mov=np.transpose(mov,[0,2,1])

                mov=mov[:,:,::-1]

                if  mov.shape[0]>0:

                    ts=np.array(trial['ts'])

                    if np.size(ts)>0:

                        assert np.std(np.diff(ts))<0.005, 'Time stamps of behaviour are unreliable'


                        if interpolate:

                            new_ts=np.linspace(0,ts[-1,0]-ts[0,0],np.shape(mov)[0])

                            if dt['trials'][idx_tr,-1] == US_ALONE:

                                t_us=np.maximum(t_us,dt['trials'][idx_tr,3]-dt['trials'][idx_tr,0])  

                                mmm=mov[:n_samples_ISI].copy()

                                mov=mov[:-n_samples_ISI]

                                mov=np.concatenate([mmm,mov])

                            elif dt['trials'][idx_tr,-1] == CS_US: 

                                t_cs=np.maximum(t_cs,dt['trials'][idx_tr,2]-dt['trials'][idx_tr,0])                                

                                t_us=np.maximum(t_us,dt['trials'][idx_tr,3]-dt['trials'][idx_tr,0])   

                                t_uss.append(t_us)                                                     

                                ISI=t_us-t_cs

                                ISIs.append(ISI)

                                n_samples_ISI=np.int(ISI*freq)

                            else:

                                t_cs=np.maximum(t_cs,dt['trials'][idx_tr,2]-dt['trials'][idx_tr,0])   

                            new_ts=new_ts

                            tims.append(new_ts)

                        else:

                            start,end,t_CS,t_US= dt['trials'][idx_tr,:-1]-dt['trials'][idx_tr,0]

                            f_rate=np.median(np.diff(ts[:,0]))
                            ISI=t_US-t_CS
                            idx_US=np.int(old_div(t_US,f_rate))
                            idx_CS=np.int(old_div(t_CS,f_rate))
                            fr_before_US=np.int(old_div((t_US - start -.1),f_rate))
                            fr_after_US=np.int(old_div((end -.1  - t_US),f_rate))
                            idx_abs=np.arange(-fr_before_US,fr_after_US)
                            time_abs=idx_abs*f_rate


                            assert np.abs(ISI-EXPECTED_ISI)<.01, str(np.abs(ISI-EXPECTED_ISI)) + ':the distance form CS and US is different from what expected'

#                            trig_US=
#                            new_ts=




                    mov_e=cb.movie(mov*rois[0][::-1].T,fr=old_div(1,np.mean(np.diff(new_ts))))
                    mov_w=cb.movie(mov*rois[1][::-1].T,fr=old_div(1,np.mean(np.diff(new_ts))))

                    x_max_w,y_max_w=np.max(np.nonzero(np.max(mov_w,0)),1)
                    x_min_w,y_min_w=np.min(np.nonzero(np.max(mov_w,0)),1)

                    x_max_e,y_max_e=np.max(np.nonzero(np.max(mov_e,0)),1)
                    x_min_e,y_min_e=np.min(np.nonzero(np.max(mov_e,0)),1)


                    mov_e=mov_e[:,x_min_e:x_max_e,y_min_e:y_max_e] 
                    mov_w=mov_w[:,x_min_w:x_max_w,y_min_w:y_max_w] 


#                    mpart=mov[:20].copy()
#                    md=cse.utilities.mode_robust(mpart.flatten())
#                    N=np.sum(mpart<=md)
#                    mpart[mpart>md]=md
#                    mpart[mpart==0]=md
#                    mpart=mpart-md
#                    std=np.sqrt(np.sum(mpart**2)/N)
#                    thr=md+10*std
#                    
#                    thr=np.minimum(255,thr)
#                    return mov     
                    if mov_filt_1d:

                        mov_e=np.mean(mov_e, axis=(1,2))
                        window_hp_=window_hp
                        window_lp_=window_lp
                        if plot_traces:
                            pl.plot(old_div((mov_e-np.mean(mov_e)),(np.max(mov_e)-np.min(mov_e))))

                    else: 

                        window_hp_=(window_hp,1,1)
                        window_lp_=(window_lp,1,1)



                    bl=signal.medfilt(mov_e,window_hp_)
                    mov_e=signal.medfilt(mov_e-bl,window_lp_)


                    if mov_filt_1d:

                        eye_=np.atleast_2d(mov_e)

                    else:

                        eye_=np.atleast_2d(np.mean(mov_e, axis=(1,2)))



                    wheel_=np.concatenate([np.atleast_1d(0),np.nanmean(np.diff(mov_w,axis=0)**2,axis=(1,2))])                   

                    if np.abs(new_ts[-1]  - time_abs[-1])>1:
                        raise Exception('Time duration is significantly larger or smaller than reference time')


                    wheel_=np.squeeze(wheel_)
                    eye_=np.squeeze(eye_)

                    f1=scipy.interpolate.interp1d(new_ts , eye_,bounds_error=False,kind='linear')                    
                    eye_=np.array(f1(time_abs))

                    f1=scipy.interpolate.interp1d(new_ts , wheel_,bounds_error=False,kind='linear')                    
                    wheel_=np.array(f1(time_abs))

                    if plot_traces:
                        pl.plot( old_div((eye_), (np.nanmax(eye_)-np.nanmin(eye_))),'r')
                        pl.plot( old_div((wheel_ -np.nanmin(wheel_)), np.nanmax(wheel_)),'k')
                        pl.pause(.01)

                    trials_idx_.append(idx_tr)

                    eye_traces.append(eye_)
                    wheel_traces.append(wheel_)                        

                    trial_info.append(dt['trials'][idx_tr,:])   


            res=dict()


            res['eyelid'] =  eye_traces   
            res['wheel'] = wheel_traces
            res['time'] = time_abs - np.median(t_uss) 
            res['trials'] = trials_idx_
            res['trial_info'] = trial_info
            res['idx_CS_US'] = np.where(list(map(int,np.array(trial_info)[:,-1]==CS_US)))[0]           
            res['idx_US'] = np.where(list(map(int,np.array(trial_info)[:,-1]==US_ALONE)))[0]
            res['idx_CS'] = np.where(list(map(int,np.array(trial_info)[:,-1]==CS_ALONE)))[0]


            return res

#%%
def process_eyelid_traces(traces,time_vect,idx_CS_US,idx_US,idx_CS,thresh_CR=.1,time_CR_on=-.1,time_US_on=.05):
    """ 
    preprocess traces output of get_behavior_traces 

    Parameters:
    ----------

    traces: ndarray (N trials X t time points)
        eyelid traces output of get_behavior_traces. 

    thresh_CR: float
        fraction of eyelid closure considered a CR

    time_CR_on: float
        time of alleged beginning of CRs

    time_US_on: float
        time when US is considered to induce have a UR


    Returns:
    -------
    eye_traces: ndarray 
        normalized eyelid traces

    trigs: dict
        dictionary containing various subdivision of the triggers according to behavioral responses

        'idxCSUSCR': index of trials with  CS+US with CR
        'idxCSUSNOCR': index of trials with  CS+US without CR
        'idxCSCR':   
        'idxCSNOCR':
        'idxNOCR': index of trials with no CRs
        'idxCR': index of trials with CRs
        'idxUS':

    """
    #normalize by max amplitudes at US    

    eye_traces=old_div(traces,np.nanmax(np.nanmedian(traces[np.hstack([idx_CS_US,idx_US])][:,np.logical_and(time_vect>time_US_on,time_vect<time_US_on +.4 )],0)))

    amplitudes_at_US=np.mean(eye_traces[:,np.logical_and( time_vect > time_CR_on , time_vect <= time_US_on )],1)

    trigs=dict()

    trigs['idxCSUSCR']=idx_CS_US[np.where(amplitudes_at_US[idx_CS_US]>thresh_CR)[-1]]
    trigs['idxCSUSNOCR']=idx_CS_US[np.where(amplitudes_at_US[idx_CS_US]<thresh_CR)[-1]]
    trigs['idxCSCR']=idx_CS[np.where(amplitudes_at_US[idx_CS]>thresh_CR)[-1]]
    trigs['idxCSNOCR']=idx_CS[np.where(amplitudes_at_US[idx_CS]<thresh_CR)[-1]]
    trigs['idxNOCR']=np.union1d(trigs['idxCSUSNOCR'],trigs['idxCSNOCR'])
    trigs['idxCR']=np.union1d(trigs['idxCSUSCR'],trigs['idxCSCR'])
    trigs['idxUS']=idx_US

    return eye_traces,amplitudes_at_US, trigs


#%%
def process_wheel_traces(traces,time_vect,thresh_MOV_iqr=3,time_CS_on=-.25,time_US_on=0):

    tmp = traces[:,time_vect<time_CS_on]
    wheel_traces=old_div(traces,(np.percentile(tmp,75)-np.percentile(tmp,25)))

    movement_at_CS=np.max(wheel_traces[:,np.logical_and( time_vect > time_CS_on, time_vect <= time_US_on )],1)

    trigs=dict()

    trigs['idxMOV']=np.where(movement_at_CS>thresh_MOV_iqr)[-1]
    trigs['idxNO_MOV']=np.where(movement_at_CS<thresh_MOV_iqr)[-1]

    return wheel_traces, movement_at_CS, trigs

#%%    
def process_wheel_traces_talmo(wheel_mms_TM_,timestamps_TM_,tm,thresh_MOV=.2,time_CS_on=-.25,time_US_on=0):     

    wheel_traces=[]
    for tr_,tm_ in zip(wheel_mms_TM_,timestamps_TM_):
        if len(tm_)<len(tm):
            #print ['Adjusting the samples:',len(tm)-len(tm_)]
            wheel_traces.append(np.pad(tr_,(0,len(tm)-len(tm_)),mode='edge'))
        elif len(tm_)>len(tm):
            wheel_traces.append(tr_[len(tm_)-len(tm):])
            #print ['Removing the samples:',len(tm)-len(tm_)]
        else:
            wheel_traces.append(tr_)

#    wheel_traces=np.abs(np.array(wheel_traces))/10 # to cm
#    tmp = traces[:,time_vect<time_CS_on]
    wheel_traces=np.abs(np.array(wheel_traces))
#    wheel_traces=traces/(np.percentile(tmp,75)-np.percentile(tmp,25))

    movement_at_CS=np.max(wheel_traces[:,np.logical_and( tm > time_CS_on, tm <= time_US_on )],1)

    trigs=dict()

    trigs['idxMOV']=np.where(movement_at_CS>thresh_MOV)[-1]
    trigs['idxNO_MOV']=np.where(movement_at_CS<thresh_MOV)[-1]

    return wheel_traces, movement_at_CS, trigs

#%%
def load_results(f_results):
    """
    Load results from CNMF on various FOVs and merge them after some preprocessing

    """
    # load data
    i=0
    A_s=[]
    C_s=[]
    YrA_s=[]
    Cn_s=[]
    shape = None
    b_s=[]
    f_s=[]
    for f_res in f_results:
        print(f_res)
        i+=1
        with  np.load(f_res) as ld:
            A_s.append(csc.csc_matrix(ld['A2']))
            C_s.append(ld['C2'])
            YrA_s.append(ld['YrA'])
            Cn_s.append(ld['Cn'])
            b_s.append(ld['b2'])
            f_s.append(ld['f2'])            
            if shape is not None:
                shape_new=(ld['d1'],ld['d2'])
                if shape_new != shape:
                    raise Exception('Shapes of FOVs not matching')
                else:
                    shape = shape_new
            else:            
                shape=(ld['d1'],ld['d2'])

    return A_s,C_s,YrA_s, Cn_s, b_s, f_s, shape  

#%% threshold and remove spurious components    
def threshold_components(A_s,shape,min_size=5,max_size=np.inf,max_perc=.5,remove_unconnected_components=True):        
    """
    Threshold components output of a CNMF algorithm (A matrices)

    Parameters:
    ----------

    A_s: list 
        list of A matrice output from CNMF

    min_size: int
        min size of the component in pixels

    max_size: int
        max size of the component in pixels

    max_perc: float        
        fraction of the maximum of each component used to threshold 

    remove_unconnected_components: boolean
        whether to remove components that are fragmented in space
    Returns:
    -------        

    B_s: list of the thresholded components

    lab_imgs: image representing the components in ndimage format

    cm_s: center of masses of each components
    """

    B_s=[]
    lab_imgs=[]

    cm_s=[]
    for A_ in A_s:
        print('*')
        max_comps=A_.max(0).todense().T
        tmp=[]
        cm=[]
        lim=np.zeros(shape)
        for idx,a in enumerate(A_.T):        
            #create mask by thresholding to 50% of the max
            mask=np.reshape(a.todense()>(max_comps[idx]*max_perc),shape)        
            label_im, nb_labels = ndimage.label(mask)
            sizes = ndimage.sum(mask, label_im, list(range(nb_labels + 1)))

            if remove_unconnected_components: 
                l_largest=(label_im==np.argmax(sizes))
                cm.append(scipy.ndimage.measurements.center_of_mass(l_largest,l_largest))
                lim[l_largest] = (idx+1)
        #       #remove connected components that are too small
                mask_size=np.logical_or(sizes<min_size,sizes>max_size)
                if np.sum(mask_size[1:])>1:
                    print(('removing ' + str( np.sum(mask_size[1:])-1) + ' components'))
                remove_pixel=mask_size[label_im]
                label_im[remove_pixel] = 0




            label_im=(label_im>0)*1    

            tmp.append(label_im.flatten())


        cm_s.append(cm)    
        lab_imgs.append(lim)        
        B_s.append(csc.csc_matrix(np.array(tmp)).T)

    return B_s, lab_imgs, cm_s           

#%% compute mask distances
def distance_masks(M_s,cm_s,max_dist):
    """
    Compute distance matrix based on an intersection over union metric. Matrix are compared in order, with matrix i compared with matrix i+1

    Parameters
    ----------
    M_s: list of ndarrays
        The thresholded A matrices (masks) to compare, output of threshold_components

    cm_s: list of list of 2-ples
        the centroids of the components in each M_s

    max_dist: float
        maximum distance among centroids allowed between components. This corresponds to a distance at which two components are surely disjoined



    Returns:
    --------
    D_s: list of matrix distances
    """
    D_s=[]

    for M1,M2,cm1,cm2 in zip(M_s[:-1],M_s[1:],cm_s[:-1],cm_s[1:]):
        print('New Pair **')
        M1= M1.copy()[:,:]
        M2= M2.copy()[:,:]
        d_1=np.shape(M1)[-1]
        d_2=np.shape(M2)[-1]
        D = np.ones((d_1,d_2));

        cm1=np.array(cm1)
        cm2=np.array(cm2)
        for i in range(d_1):
            if i%100==0:
                print(i)
            k=M1[:,np.repeat(i,d_2)]+M2
    #        h=M1[:,np.repeat(i,d_2)].copy()
    #        h.multiply(M2)
            for j  in range(d_2): 

                dist = np.linalg.norm(cm1[i]-cm2[j])
                if dist<max_dist:
                    union = k[:,j].sum()
    #                intersection = h[:,j].nnz
                    intersection= np.array(M1[:,i].T.dot(M2[:,j]).todense()).squeeze()
        ##            intersect= np.sum(np.logical_xor(M1[:,i],M2[:,j]))
        ##            union=np.sum(np.logical_or(M1[:,i],M2[:,j]))
                    if union  > 0:
                        D[i,j] = 1-1.*intersection/(union-intersection)
                    else:
#                        print 'empty component: setting distance to max'
                        D[i,j] = 1.

                    if np.isnan(D[i,j]):
                        raise Exception('Nan value produced. Error in inputs')
                else:
                    D[i,j] = 1

        D_s.append(D)            
    return D_s   

#%% find matches
def find_matches(D_s, print_assignment=False):

    matches=[]
    costs=[]
    t_start=time()
    for ii,D in enumerate(D_s):
        DD=D.copy()    
        if np.sum(np.where(np.isnan(DD)))>0:
            raise Exception('Distance Matrix contains NaN, not allowed!')


    #    indexes = m.compute(DD)
#        indexes = linear_assignment(DD)
        indexes = linear_sum_assignment(DD)
        indexes2=[(ind1,ind2) for ind1,ind2 in zip(indexes[0],indexes[1])]
        matches.append(indexes)
        DD=D.copy()   
        total = []
        for row, column in indexes2:
            value = DD[row,column]
            if print_assignment:
                print(('(%d, %d) -> %f' % (row, column, value)))
            total.append(value)      
        print(('FOV: %d, shape: %d,%d total cost: %f' % (ii, DD.shape[0],DD.shape[1], np.sum(total))))
        print((time()-t_start))
        costs.append(total)      

    return matches,costs

#%%
def link_neurons(matches,costs,max_cost=0.6,min_FOV_present=None):
    """
    Link neurons from different FOVs given matches and costs obtained from the hungarian algorithm

    Parameters
    ----------
    matches: lists of list of tuple
        output of the find_matches function

    costs: list of lists of scalars
        cost associated to each match in matches

    max_cost: float
        maximum allowed value of the 1- intersection over union metric    

    min_FOV_present: int
        number of FOVs that must consequently contain the neuron starting from 0. If none 
        the neuro must be present in each FOV
    Returns:
    --------
    neurons: list of arrays representing the indices of neurons in each FOV

    """
    if min_FOV_present is None:
        min_FOV_present=len(matches)

    neurons=[]
    num_neurons=0
#    Yr_tot=[]
    num_chunks=len(matches)+1
    for idx in range(len(matches[0][0])):
        neuron=[]
        neuron.append(idx)
#        Yr=YrA_s[0][idx]+C_s[0][idx]
        for match,cost,chk in zip(matches,costs,list(range(1,num_chunks))):
            rows,cols=match        
            m_neur=np.where(rows==neuron[-1])[0].squeeze()
            if m_neur.size > 0:                           
                if cost[m_neur]<=max_cost:
                    neuron.append(cols[m_neur])
#                    Yr=np.hstack([Yr,YrA_s[chk][idx]+C_s[chk][idx]])
                else:                
                    break
            else:
                break
        if len(neuron)>min_FOV_present:           
            num_neurons+=1        
            neurons.append(neuron)
#            Yr_tot.append(Yr)


    neurons=np.array(neurons).T
    print(('num_neurons:' + str(num_neurons)))
#    Yr_tot=np.array(Yr_tot)
    return neurons

#%%
def generate_linked_traces(mov_names,chunk_sizes,A,b,f):
    """
    Generate traces (DFF,BL and DF) for a group of movies that share the same A,b and f,
    by applying the same transformation over a set of movies. This removes
    the contamination of neuropil and then masks the components.



    Parameters:
    -----------
    mov_names: list of path to movies associated with the same A,b,and f

    chunk_sizes:list containing the number of frames in each movie    

    A,b and f: from CNMF




    Returns:
    --------


    """
    num_chunks=np.sum(chunk_sizes)
#    A = A_s[idx][:,neurons[idx]] 
    nA = (A.power(2)).sum(0)
#    bckg=cb.movie(cb.to_3D(b.dot(f).T,(-1,shape[0],shape[1])),fr=1)
    f=np.array(f).squeeze()
#    bckg=bckg.resize(1,1,1.*num_chunks/b_size)
    b_size=f.shape[0]
#    if num_chunks != b_size:        
#        raise Exception('The number of frames are not matching')
#        

    counter=0



    f_in=np.atleast_2d(scipy.signal.resample(f,num_chunks))



#    C,f,S,bl,c1,neurons_sn,g,YrA = cse.temporal.update_temporal_components(Yr,A,b,C_in,f_in,p=0)



    traces=[]
    traces_BL=[]
    traces_DFF=[]

    for jj,mv in enumerate(mov_names):

        mov_chunk_name=os.path.splitext(os.path.split(mv)[-1])[0]+'.hdf5'        
        mov_chunk_name=os.path.join(os.path.dirname(mv),mov_chunk_name)
        print(mov_chunk_name)

        m=cb.load(mov_chunk_name).to_2D().T      
        bckg_1=b.dot(f_in[:,counter:counter+chunk_sizes[jj]])

        m=m-bckg_1          
#        (m).play(backend='opencv',gain=10.,fr=33)
#        m=np.reshape(m,(-1,np.prod(shape)),order='F').T
#        bckg_1=np.reshape(bckg_1,(-1,np.prod(shape)),order='F').T

        counter+=chunk_sizes[jj]

        Y_r_sig=A.T.dot(m)
        Y_r_sig= scipy.sparse.linalg.spsolve(scipy.sparse.spdiags(np.sqrt(nA),0,nA.size,nA.size),Y_r_sig)
        traces.append(Y_r_sig)

        Y_r_bl=A.T.dot(bckg_1)
        Y_r_bl= scipy.sparse.linalg.spsolve(scipy.sparse.spdiags(np.sqrt(nA),0,nA.size,nA.size),Y_r_bl)                
        traces_BL.append(Y_r_bl)        
        Y_r_bl=cse.utilities.mode_robust(Y_r_bl,1)        
        traces_DFF.append(old_div(Y_r_sig,Y_r_bl[:,np.newaxis]))

    return traces,traces_DFF,traces_BL
#%%
def extract_traces_mat(traces,triggers_idx,f_rate,time_before=2.7,time_after=5.3):
    """
    Equivalent of take for the input format we are using. 

    Parameters:
    -----------
    traces: list of ndarrays
        each element is one trial, the dimensions are n_neurons x time
    triggers_idx: list of ints
        one for each element of traces, is the index of the trigger to align the traces to  
    f_rate: double
        frame rate associated to the traces
    time_before,time_after: double
        time before and after the trigger establishing the boundary of the extracted subtraces        

    Returns:
    --------
    traces_mat: matrix containing traces with dimensions trials X cell X time

    time_mat: associated time vector
    """
    samples_before = np.int(time_before*f_rate)
    samples_after = np.int(time_after*f_rate)

    if traces[0].ndim > 1:   
        traces_mat = np.zeros([len(traces),len(traces[0]),samples_after+samples_before])       
    else:
        traces_mat = np.zeros([len(traces),1,samples_after+samples_before]) 

    for idx,tr in enumerate(traces):  
#            print samples_before,samples_after
#            print np.int(triggers_idx[idx]-samples_before),np.int(triggers_idx[idx]+samples_after)
        traces_mat[idx]=traces[idx][:,np.int(triggers_idx[idx]-samples_before):np.int(triggers_idx[idx]+samples_after)]



    time_mat=old_div(np.arange(-samples_before,samples_after),f_rate)



    return traces_mat,time_mat


#%%
def load_data_from_stored_results(base_folder, load_masks=False, thresh_CR = 0.1,threshold_responsiveness=0.1,
                                  is_blob=True,time_CR_on=-.1,time_US_on=.05,thresh_MOV_iqr=1000,time_CS_on_MOV=-.25,time_US_on_MOV=0):
    """
    From the partial data stored retrieves variables of interest 

    """
    import calblitz as cb
    import numpy as np
    import scipy 
    import pylab as pl
    import pickle
    from glob import glob



#    base_folder='/mnt/ceph/users/agiovann/ImagingData/eyeblink/b35/20160714143248/'
    if is_blob:
        with np.load(base_folder+'distance_masks.npz') as ld:
            D_s=ld['D_s']

        with np.load(base_folder+'neurons_matching.npz') as ld:
            neurons=ld['neurons']
            locals().update(ld)

    with np.load(base_folder+'all_triggers.npz') as at:
        triggers_img=at['triggers']
        trigger_names_img=at['trigger_names'] 
    if load_masks:
        f_results= glob(base_folder+'*results_analysis.npz')
        f_results.sort()
        for rs in f_results:
            print(rs)     
        print('*****')        
        A_s,C_s,YrA_s, Cn_s, b_s, f_s, shape =  load_results(f_results) 

        if is_blob:
            remove_unconnected_components=True
        else:
            remove_unconnected_components=False        
            neurons=[]
            for xx in A_s:
                neurons.append(np.arange(A_s[0].shape[-1]))

    #    B_s, lab_imgs, cm_s  = threshold_components(A_s,shape, min_size=5,max_size=50,max_perc=.5,remove_unconnected_components=remove_unconnected_components)

        tmpl_name=glob(base_folder+'*template_total.npz')[0]

        with np.load(tmpl_name) as ld:
            mov_names_each=ld['movie_names']

        A_each=[]
        b_each=[]    
        f_each=[]
        for idx, mov_names in enumerate(mov_names_each):
            idx=0
            A_each.append(A_s[idx][:,neurons[idx]])
        #    C=C_s[idx][neurons[idx]]
        #    YrA=YrA_s[idx][neurons[idx]]
            b_each.append(b_s[idx])
            f_each.append(f_s[idx])
    else:
        A_each=[]
        b_each=[]    
        f_each=[]


    with np.load(base_folder+'behavioral_traces.npz') as ld: 
        res_bt = dict(**ld)
        tm=res_bt['time']
        f_rate_bh=old_div(1,np.median(np.diff(tm)))
        ISI=res_bt['trial_info'][0][3]-res_bt['trial_info'][0][2]
        eye_traces=np.array(res_bt['eyelid'])
        idx_CS_US=res_bt['idx_CS_US']
        idx_US=res_bt['idx_US']
        idx_CS=res_bt['idx_CS']

        idx_ALL=np.sort(np.hstack([idx_CS_US,idx_US,idx_CS]))
        eye_traces,amplitudes_at_US, trig_CRs=process_eyelid_traces(eye_traces,tm,idx_CS_US,idx_US,idx_CS,thresh_CR=thresh_CR,time_CR_on=time_CR_on,time_US_on=time_US_on)

        idxCSUSCR = trig_CRs['idxCSUSCR']
        idxCSUSNOCR = trig_CRs['idxCSUSNOCR']
        idxCSCR = trig_CRs['idxCSCR']
        idxCSNOCR = trig_CRs['idxCSNOCR']
        idxNOCR = trig_CRs['idxNOCR']
        idxCR = trig_CRs['idxCR']
        idxUS = trig_CRs['idxUS']
        idxCSCSUS=np.concatenate([idx_CS,idx_CS_US]) 

    with open(base_folder+'traces.pk','r') as f:    
        trdict= pickle.load(f)
        traces_DFF=trdict['traces_DFF']

    triggers_img=np.array(triggers_img)   
    idx_expected_US = np.zeros_like(triggers_img[:,1]) 
    idx_expected_US = triggers_img[:,1]
    idx_expected_US[idx_CS]=np.nanmedian(triggers_img[:,1]) 
    triggers_img =  np.concatenate([triggers_img,   idx_expected_US[:,np.newaxis].astype(np.int)],-1)   

    img_descr=cb.utils.get_image_description_SI(glob(base_folder+'2016*.tif')[0])[0]
    f_rate=img_descr['scanimage.SI.hRoiManager.scanFrameRate']
    print(f_rate)              
    #%%
    time_before=3
    time_after=3
    wheel,time_w=res_bt['wheel'],res_bt['time']
    eye=eye_traces
    time_e=tm
    wheel_mat=np.array([wh[np.logical_and(time_w>-time_before,time_w<time_after)] for wh in wheel])
    eye_mat=np.array([e[np.logical_and(time_e>-time_before,time_e<time_after)] for e in eye])

    time_w_mat=time_w[np.logical_and(time_w>-time_before,time_w<time_after)]
    time_e_mat=time_e[np.logical_and(time_e>-time_before,time_e<time_after)]
    traces_mat,time_mat=extract_traces_mat(traces_DFF,triggers_img[:,1],f_rate,time_before=time_before,time_after=time_after)
#    traces_mat,time_mat=scipy.signal.resample(traces_mat, len(time_w_mat),t=time_mat ,axis=-1)
    #%
    wheel_traces, movement_at_CS, trigs_mov = process_wheel_traces(np.array(res_bt['wheel']),tm,thresh_MOV_iqr=thresh_MOV_iqr,time_CS_on=time_CS_on_MOV,time_US_on=time_US_on_MOV)    
    print('fraction with movement:')    
    print((len(trigs_mov['idxMOV'])*1./len(trigs_mov['idxNO_MOV'])))

    #%%
    triggers_out=dict()
    triggers_out['mn_idx_CS_US'] =np.intersect1d(idx_CS_US,trigs_mov['idxNO_MOV'])
    triggers_out['nm_idx_US']= np.intersect1d(idx_US,trigs_mov['idxNO_MOV'])
    triggers_out['nm_idx_CS']= np.intersect1d(idx_CS,trigs_mov['idxNO_MOV'])
    triggers_out['nm_idxCSUSCR'] = np.intersect1d(idxCSUSCR,trigs_mov['idxNO_MOV'])
    triggers_out['nm_idxCSUSNOCR'] = np.intersect1d(idxCSUSNOCR,trigs_mov['idxNO_MOV'])
    triggers_out['nm_idxCSCR'] = np.intersect1d(idxCSCR,trigs_mov['idxNO_MOV'])
    triggers_out['nm_idxCSNOCR'] = np.intersect1d(idxCSNOCR,trigs_mov['idxNO_MOV'])
    triggers_out['nm_idxNOCR'] = np.intersect1d(idxNOCR,trigs_mov['idxNO_MOV'])
    triggers_out['nm_idxCR'] = np.intersect1d(idxCR,trigs_mov['idxNO_MOV'])
    triggers_out['nm_idxUS'] = np.intersect1d(idxUS,trigs_mov['idxNO_MOV'])
    triggers_out['nm_idxCSCSUS']=np.intersect1d(idxCSCSUS,trigs_mov['idxNO_MOV'])  
    #%%

    newf_rate=old_div(1,np.median(np.diff(time_mat)))
    ftraces=traces_mat.copy()  
    samples_before=np.int(time_before*newf_rate)  
    ISI_frames=np.int(ISI*newf_rate)
    ftraces=ftraces-np.median(ftraces[:,:,np.logical_and(time_mat>-1,time_mat<-ISI)],axis=(2))[:,:,np.newaxis]   
    amplitudes_responses=np.mean(ftraces[:,:,np.logical_and(time_mat>-.03,time_mat<.04)],-1)
    cell_responsiveness=np.median(amplitudes_responses[triggers_out['nm_idxCSCSUS']],axis=0)
    idx_responsive = np.where(cell_responsiveness>threshold_responsiveness)[0]
    fraction_responsive=len(np.where(cell_responsiveness>threshold_responsiveness)[0])*1./np.shape(ftraces)[1]

    print('fraction responsive:')    
    print(fraction_responsive)

    ftraces=ftraces[:,cell_responsiveness>threshold_responsiveness,:]
    amplitudes_responses=np.mean(ftraces[:,:,samples_before+ISI_frames-1:samples_before+ISI_frames+1],-1)

    traces=dict()       
    traces['fluo_traces']=ftraces
    traces['eye_traces']=eye_mat
    traces['wheel_traces']=wheel_mat
    traces['time_fluo']=time_mat
    traces['time_eye']=time_e_mat
    traces['time_wheel']=time_w_mat

    amplitudes=dict()
    amplitudes['amplitudes_fluo']=amplitudes_responses
    amplitudes['amplitudes_eyelid']=amplitudes_at_US

    masks=dict()
    masks['A_each']=[A[:,idx_responsive] for A in A_each]
    masks['b_each']=b_each
    masks['f_each']=f_each

    return traces, masks, triggers_out, amplitudes, ISI 

#%%
def fast_process_day(base_folder,min_radius=3,max_radius=4):
    import pickle
    import pylab as pl
    try:
        tmpl_name=glob(base_folder+'*template_total.npz')[0]

        print(tmpl_name)
        with np.load(tmpl_name) as ld:
            mov_names_each=ld['movie_names']

        f_results= glob(base_folder+'*results_analysis.npz')
        f_results.sort()
        A_s,C_s, YrA_s, Cn_s, b_s, f_s, shape =  load_results(f_results)     
    #    B_s, lab_imgs, cm_s  = threshold_components(A_s,shape, min_size=10,max_size=50,max_perc=.5)
        traces=[]
        traces_BL=[]
        traces_DFF=[]

        for idx, mov_names in enumerate(mov_names_each):
            A=A_s[idx]
        #    C=C_s[idx][neurons[idx]]
        #    YrA=YrA_s[idx][neurons[idx]]
            b=b_s[idx]
            f=f_s[idx]
            chunk_sizes=[]
            for mv in mov_names:
                base_name=os.path.splitext(os.path.split(mv)[-1])[0]
                with np.load(base_folder+base_name+'.npz') as ld:
                    TT=len(ld['shifts'])            
                chunk_sizes.append(TT)


            masks_ws,pos_examples,neg_examples=cse.utilities.extract_binary_masks_blob(A, min_radius, \
            shape, num_std_threshold=1, minCircularity= 0.5, minInertiaRatio = 0.2,minConvexity = .8)        
            #sizes=np.sum(masks_ws,(1,2))
            #pos_examples=np.intersect1d(pos_examples,np.where(sizes<max_radius**2*np.pi)[0])     
            print((len(pos_examples)))
    #        pl.close()
    #        pl.imshow(np.mean(masks_ws[pos_examples],0))
            pl.pause(.1)
            #A=A.tocsc()[:,pos_examples]
            traces,traces_DFF,traces_BL = generate_linked_traces(mov_names,chunk_sizes,A,b,f)        

            np.savez(f_results[idx][:-4]+'_masks.npz',masks_ws=masks_ws,pos_examples=pos_examples, neg_examples=neg_examples, A=A.todense(),b=b,f=f)

            with open(f_results[idx][:-4]+'_traces.pk','w') as f: 
                pickle.dump(dict(traces=traces,traces_BL=traces_BL,traces_DFF=traces_DFF),f)   

    except:
        print('Failed')
        return False

    return True
#%%
def process_fast_process_day(base_folders,save_name='temp_save.npz'):
    """
    Use this after having used fast_process_day

    Parameters:
    ----------

    base_folders: list of path to base folders

    Returns:
    --------

    triggers_chunk_fluo: triggers associated to fluorescence (one per chunk)
    eyelid_chunk: eyelid (one per chunk)
    wheel_chunk: wheel (one per chunk)
    triggers_chunk_bh: triggers associated to behavior(one per chunk)
    tm_behav: time of behavior (one per chunk)
    names_chunks: names of the file associated to each chunk(one per chunk)
    fluo_chunk: fluorescence traces (one per chunk)
    pos_examples_chunks: indexes of examples that were classified as good by the blob detector (one per chunk)
    A_chunks: masks associated   (one per chunk)
    """
    triggers_chunk_fluo = []  
    eyelid_chunk = []
    wheel_chunk = []
    triggers_chunk_bh = []
    tm_behav=[]
    names_chunks=[]
    fluo_chunk=[]
    pos_examples_chunks=[]     
    A_chunks=[]  
    for base_folder in base_folders:
        try:         
            print (base_folder)
            with np.load(os.path.join(base_folder,'all_triggers.npz')) as ld:
                triggers=ld['triggers']
                trigger_names=ld['trigger_names']

            with np.load(glob(os.path.join(base_folder,'*-template_total.npz'))[0]) as ld:
                movie_names=ld['movie_names']
                template_each=ld['template_each']


            idx_chunks=[] 
            for name_chunk in movie_names:
                idx_chunks.append([np.int(re.search('_00[0-9][0-9][0-9]_0',nm).group(0)[2:6])-1 for nm in name_chunk])



            with np.load(base_folder+'behavioral_traces.npz') as ld: 
                res_bt = dict(**ld)
                tm=res_bt['time']
                f_rate_bh=old_div(1,np.median(np.diff(tm)))
                ISI=np.median([rs[3]-rs[2] for rs in res_bt['trial_info'][res_bt['idx_CS_US']]])
                trig_int=np.hstack([((res_bt['trial_info'][:,2:4]-res_bt['trial_info'][:,0][:,None])*f_rate_bh),res_bt['trial_info'][:,-1][:,np.newaxis]]).astype(np.int)
                trig_int[trig_int<0]=-1
                trig_int=np.hstack([trig_int,len(tm)+trig_int[:,:1]*0])
                trig_US=np.argmin(np.abs(tm))
                trig_CS=np.argmin(np.abs(tm+ISI))
                trig_int[res_bt['idx_CS_US'],0]=trig_CS
                trig_int[res_bt['idx_CS_US'],1]=trig_US
                trig_int[res_bt['idx_US'],1]=trig_US
                trig_int[res_bt['idx_CS'],0]=trig_CS
                eye_traces=np.array(res_bt['eyelid']) 
                wheel_traces=np.array(res_bt['wheel'])




            fls=glob(os.path.join(base_folder,'*.results_analysis_traces.pk'))
            fls.sort()
            fls_m=glob(os.path.join(base_folder,'*.results_analysis_masks.npz'))
            fls_m.sort()     


            for indxs,name_chunk,fl,fl_m in zip(idx_chunks,movie_names,fls,fls_m):
                if np.all([nmc[:-4] for nmc in name_chunk] == trigger_names[indxs]):
                    triggers_chunk_fluo.append(triggers[indxs,:])
                    eyelid_chunk.append(eye_traces[indxs,:])
                    wheel_chunk.append(wheel_traces[indxs,:])
                    triggers_chunk_bh.append(trig_int[indxs,:])
                    tm_behav.append(tm)
                    names_chunks.append(fl)
                    with open(fl,'r') as f: 
                        tr_dict=pickle.load(f)   
                        print(fl)
                        fluo_chunk.append(tr_dict['traces_DFF'])
                    with np.load(fl_m) as ld:
                        A_chunks.append(scipy.sparse.coo_matrix(ld['A']))
                        pos_examples_chunks.append(ld['pos_examples'])                
                else:
                    raise Exception('Names of triggers not matching!')
        except :
            print(("ERROR in:"+base_folder))  
#            raise

    import pdb
    pdb.set_trace()
    if save_name is not None:
        np.savez(save_name,triggers_chunk_fluo=triggers_chunk_fluo, triggers_chunk_bh=triggers_chunk_bh, eyelid_chunk=eyelid_chunk, wheel_chunk=wheel_chunk, tm_behav=tm_behav, fluo_chunk=fluo_chunk,names_chunks=names_chunks,pos_examples_chunks=pos_examples_chunks,A_chunks=A_chunks)            

    return triggers_chunk_fluo, eyelid_chunk,wheel_chunk ,triggers_chunk_bh ,tm_behav,names_chunks,fluo_chunk,pos_examples_chunks,A_chunks
