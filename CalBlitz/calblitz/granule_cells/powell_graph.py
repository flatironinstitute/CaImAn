# -*- coding: utf-8 -*-
"""
Created on Sun Aug  7 13:44:32 2016

@author: agiovann
"""

#%%    
%load_ext autoreload
%autoreload 2    
from glob import glob
import numpy as np
import pylab as pl
import os
mpl.rcParams['pdf.fonttype'] = 42

#%%
fls=[         '/mnt/ceph/users/agiovann/ImagingData/eyeblink/b37/20160627105123/',
              '/mnt/ceph/users/agiovann/ImagingData/eyeblink/b37/20160627154015/',
              '/mnt/ceph/users/agiovann/ImagingData/eyeblink/b37/20160628162522/',
              '/mnt/ceph/users/agiovann/ImagingData/eyeblink/b37/20160629123648/',
              '/mnt/ceph/users/agiovann/ImagingData/eyeblink/b37/20160630120544/',
              '/mnt/ceph/users/agiovann/ImagingData/eyeblink/b37/20160701113525/',
              '/mnt/ceph/users/agiovann/ImagingData/eyeblink/b37/20160702152950/',
              '/mnt/ceph/users/agiovann/ImagingData/eyeblink/b37/20160703173620/',
              '/mnt/ceph/users/agiovann/ImagingData/eyeblink/b37/20160704130454/',
              ]
fls=[
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
from skimage.morphology import disk
from skimage.filter import rank
from skimage import exposure    
counter=0          
for counter,fl in enumerate(fls):
    print os.path.abspath(fl)
    
    with np.load(glob(os.path.join(os.path.abspath(fl),'*-template_total.npz'))[0]) as ld:
        templs=ld['template_each']
        for mn1 in templs:
            pl.subplot(3,4,counter+1)        
#            mn=np.median(templs,0)
            mn=mn1
#            selem = disk(50)
#            mn=(mn1 - np.min(mn1))/(np.max(mn1)-np.min(mn1))
#            mn = rank.equalize(mn, selem=selem)
#            mn = exposure.equalize_hist(mn,nbins=1024)
            os.path.split(fl)[-1]
#            pl.imshow(mn,cmap='gray')
            pl.imshow(mn,cmap='gray',vmax=np.percentile(mn,98))
#
#            pl.xlim([0,512])
#            pl.ylim([300,512])
            pl.axis('off')
            pl.title(fl.split('/')[-2][:8])
            counter+=1

#%% image powell locomotion
%load_ext autoreload
%autoreload 2    
import calblitz as cb
from calblitz.granule_cells import utils_granule as gc
from glob import glob
import numpy as np
import os
import scipy 
import pylab as pl
import ca_source_extraction as cse
import pickle

is_blob=True
base_folder='/mnt/ceph/users/agiovann/ImagingData/eyeblink/b37/20160627105123/'
if is_blob:
    with np.load(base_folder+'distance_masks.npz') as ld:
        D_s=ld['D_s']
    with np.load(base_folder+'neurons_matching.npz') as ld:
        locals().update(ld)

        

with np.load(base_folder+'all_triggers.npz') as at:
    triggers_img=at['triggers']
    trigger_names_img=at['trigger_names'] 
    
with np.load(base_folder+'behavioral_traces.npz') as ld: 
    res_bt = dict(**ld)
    tm=res_bt['time']
    f_rate_bh=1/np.median(np.diff(tm))
    ISI=res_bt['trial_info'][0][3]-res_bt['trial_info'][0][2]
    eye_traces=np.array(res_bt['eyelid'])
    idx_CS_US=res_bt['idx_CS_US']
    idx_US=res_bt['idx_US']
    idx_CS=res_bt['idx_CS']
    
    idx_ALL=np.sort(np.hstack([idx_CS_US,idx_US,idx_CS]))
    eye_traces,amplitudes_at_US, trig_CRs=gc.process_eyelid_traces(eye_traces,tm,idx_CS_US,idx_US,idx_CS,thresh_CR=.15,time_CR_on=-.1,time_US_on=.05)
    
    idxCSUSCR = trig_CRs['idxCSUSCR']
    idxCSUSNOCR = trig_CRs['idxCSUSNOCR']
    idxCSCR = trig_CRs['idxCSCR']
    idxCSNOCR = trig_CRs['idxCSNOCR']
    idxNOCR = trig_CRs['idxNOCR']
    idxCR = trig_CRs['idxCR']
    idxUS = trig_CRs['idxUS']
    idxCSCSUS=np.concatenate([idx_CS,idx_CS_US]) 

with open(base_folder+'traces.pk','r') as f:    
            locals().update(pickle.load(f))  

triggers_img=np.array(triggers_img)    
idx_expected_US=  np.repeat( np.nanmedian(triggers_img[:,1]),len(triggers_img[:,1]))     
triggers_img =  np.concatenate([triggers_img,   idx_expected_US[:,np.newaxis].astype(np.int)],-1)   

img_descr=cb.utils.get_image_description_SI(glob(base_folder+'2016*.tif')[0])[0]
f_rate=img_descr['scanimage.SI.hRoiManager.scanFrameRate']
print f_rate              
#%%
#traces_flat=np.concatenate(traces,1)
#%%
time_before=4
time_after=3.5
wheel,time_w=res_bt['wheel'],res_bt['time']
wheel_mat=np.array([wh[np.logical_and(time_w>-time_before,time_w<time_after)] for wh in wheel])
time_w_mat=time_w[np.logical_and(time_w>-time_before,time_w<time_after)]
traces_mat,time_mat=gc.extract_traces_mat(traces_DFF,triggers_img[:,-1],f_rate,time_before=time_before,time_after=time_after)
traces_mat,time_mat=scipy.signal.resample(traces_mat, len(time_w_mat),t=time_mat ,axis=-1)

#%% 1,17
pl.close()
wheel_flat=np.concatenate(wheel_mat[11:21],0)


for idx in [126]:#[169, 249, 392, 600, 434,  17, 907, 755, 834,  58, 586, 404, 737]:
    traces_flat=np.concatenate(traces_mat[11:21,idx,:],0)
    time_vect=np.median(np.diff(time_mat))*np.arange(len(wheel_flat))
    pl.subplot(2,1,1)
    pl.plot(time_vect,wheel_flat/np.max(wheel_flat))
    traces_flat=scipy.signal.savgol_filter(traces_flat,5,1)
    pl.axis('tight')
    pl.ylabel('Action index')

    pl.subplot(2,1,2)
    pl.cla()
    time_vect=np.median(np.diff(time_mat))*np.arange(len(traces_flat))
    pl.plot(time_vect,traces_flat-cse.utilities.mode_robust(traces_flat))
    pl.axis('tight')
    pl.xlabel('Time (s)')
    pl.ylabel('\Delta F/F')
    pl.pause(2)
#%%
import pandas as pd
    
pl.close()
wheel_flat=np.concatenate(wheel_mat,0)

dick=dict()
dick['wheel']=wheel_flat
dick['id']=np.arange(len(wheel_flat))
rs=[]
for idx in range(traces_mat.shape[1]):
    print idx
    tr=np.concatenate(traces_mat[:,idx,:],0)
    rs.append(np.corrcoef(tr,wheel_flat)[0,1])  
    if rs[-1]>.25 and np.min(tr)<.2 and np.max(tr)<7:
        dick['neuron_'+str(idx)]=tr
#%%
bins=np.arange(0,len(wheel_flat),20)
df = pd.DataFrame(dick)
t_bins  = df.groupby(pd.cut(df.id, bins))
wh_binned=t_bins.median()
thr_down=wh_binned.wheel.quantile(.01)
thr_up=wh_binned.wheel.quantile(.99)
mean_down =wh_binned[wh_binned.wheel<thr_down].median()
mean_up =  wh_binned[wh_binned.wheel>thr_up].median()
mean_down=mean_down.filter(regex=("neuron*")).values
mean_up=mean_up.filter(regex=("neuron*")).values
pl.figure(facecolor="white")
ax = pl.gca()
ax.plot(np.tile([0,1],(mean_up.shape[0],1)).T ,np.vstack([mean_down,mean_up])/.2,'-ro',markersize=20)
ax.get_yaxis().set_tick_params(direction='out')
ax.get_xaxis().set_tick_params(direction='out')
ax.xaxis.set_tick_params(width=1)
ax.yaxis.set_tick_params(width=1)
ax.xaxis.set_tick_params(size=8)
ax.yaxis.set_tick_params(size=8)
pl.xlim([-.5,1.5])
pl.ylabel('GC estimated rate (Hz)')
pl.xticks(np.arange(2),('Quiet','Walking'))

font = {'family' : 'Myriad Pro',
        'weight' : 'regular',
        'size'   : 30}

pl.rc('font', **font)