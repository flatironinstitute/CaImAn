# -*- coding: utf-8 -*-
"""
Created on Sun Aug  7 13:44:32 2016

@author: agiovann
"""
from __future__ import print_function

#%%    
from builtins import zip
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')    
from glob import glob
import numpy as np
import pylab as pl
import os
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
import calblitz as cb
from calblitz.granule_cells import utils_granule as gc
from glob import glob
import numpy as np
import os
import scipy 
import pylab as pl
import ca_source_extraction as cse
import pickle
import calblitz as cb
from calblitz.granule_cells.utils_granule import  load_data_from_stored_results
#%%
base_folder='/mnt/ceph/users/agiovann/ImagingData/eyeblink/b37/20160705103903/'
traces, masks, triggers_out, amplitudes, ISI = load_data_from_stored_results(base_folder, thresh_CR = 0.1, \
threshold_responsiveness=0.1, is_blob=True,time_CR_on=-.1,time_US_on=.05,thresh_MOV_iqr=1000,\
time_CS_on_MOV =-.25,time_US_on_MOV=0)
wheel_mat = traces['wheel_traces']
ftraces = traces['fluo_traces']
time_mat =  traces['time_fluo']
time_e_mat = traces['time_eye']
time_w_mat = traces['time_wheel']
eye_mat = traces['eye_traces']
amplitudes_eyelid=amplitudes['amplitudes_eyelid']
amplitudes_fluo=amplitudes['amplitudes_fluo']
#%%
counter=0
with np.load(glob(os.path.join(base_folder,'*-template_total.npz'))[0]) as ld:
    templs=ld['template_each']
    for mn1,A in zip(templs,masks['A_each']):

        pl.subplot(2,3,counter+1)        
#            mn=np.median(templs,0)
        mn=mn1
        d1,d2=np.shape(mn)
#            selem = disk(50)
#            mn=(mn1 - np.min(mn1))/(np.max(mn1)-np.min(mn1))
#            mn = rank.equalize(mn, selem=selem)
#            mn = exposure.equalize_hist(mn,nbins=1024)
#            os.path.split(fl)[-1]
#            pl.imshow(mn,cmap='gray')
#            pl.imshow(mn,cmap='gray',vmax=np.percentile(mn,99))

#            pl.imshow(mn,cmap='gray',vmax=np.percentile(mn,98))
        pl.imshow(A.mean(1).reshape((d1,d2),order='F'),alpha=1,cmap='hot')
#            pl.xlim([0,512])
#            pl.ylim([300,512])
        pl.axis('off')
        counter+=1
#            pl.title(fl.split('/')[-2][:8])

#%%
pl.plot(time_e_mat,np.mean(eye_mat[triggers_out['nm_idxCSUSCR']],0))       
pl.plot(time_e_mat,np.mean(eye_mat[triggers_out['nm_idxCSUSNOCR']],0))     
pl.plot(time_e_mat,np.mean(eye_mat[triggers_out['nm_idxCSCR']],0))
pl.plot(time_e_mat,np.mean(eye_mat[triggers_out['nm_idxCSNOCR']],0))    
pl.plot(time_e_mat,np.mean(eye_mat[triggers_out['nm_idx_US']],0))
pl.legend(['idxCSUSCR','idxCSUSNOCR','idxCSCR','idxCSNOCR','idxUS'])
pl.xlabel('time to US (s)')
pl.ylabel('eyelid closure')
pl.axvspan(-ISI,ISI, color='g', alpha=0.2, lw=0)
pl.axvspan(0,0.03, color='r', alpha=0.2, lw=0)
pl.xlim([-.5, .5])
pl.ylim([-.1, None])

#%%
pl.close()
pl.subplot(2,2,1)
pl.plot(time_e_mat,np.mean(eye_mat[triggers_out['nm_idxCR']],0),'-*')       
pl.plot(time_e_mat,np.mean(eye_mat[triggers_out['nm_idxNOCR']],0),'-d')     
pl.plot(time_e_mat,np.mean(eye_mat[triggers_out['nm_idxUS']],0),'-o')

pl.xlabel('Time to US (s)')
pl.ylabel('Eyelid closure')
pl.axvspan(-ISI,ISI, color='g', alpha=0.2, lw=0)
pl.axvspan(0,0.03, color='r', alpha=0.2, lw=0)

pl.xlim([-.5,.5])
pl.ylim([-.1, None])
pl.legend(['CR+','CR-','US'],loc='upper left')

pl.subplot(2,2,2)

pl.plot(time_mat,np.median(ftraces[triggers_out['nm_idxCR']],axis=(0,1)),'-*')
pl.plot(time_mat,np.median(ftraces[triggers_out['nm_idxNOCR']],axis=(0,1)),'-d')
pl.plot(time_mat,np.median(ftraces[triggers_out['nm_idxUS']],axis=(0,1)),'-o')

pl.axvspan((-ISI), ISI, color='g', alpha=0.2, lw=0)
pl.axvspan(0, 0.03, color='r', alpha=0.5, lw=0)
pl.xlabel('Time to US (s)')
pl.ylabel('DF/F')
pl.xlim([-.5, .5])
pl.ylim([-.1, None])

#%
import pandas

bins=np.arange(-.15,0.7,.2)
n_bins=6
thresh__correlation=.93
dfs=[];
dfs_random=[];
idxCSCSUS=triggers_out['nm_idxCSCSUS']
x_name='ampl_eye'
y_name='ampl_fl'
for resps in amplitudes_fluo.T:
    idx_order=np.arange(len(idxCSCSUS))    
    dfs.append(pandas.DataFrame(
        {y_name: resps[idxCSCSUS[idx_order]],
         x_name: amplitudes_eyelid[idxCSCSUS]}))

    idx_order=np.random.permutation(idx_order)         
    dfs_random.append(pandas.DataFrame(
        {y_name: resps[idxCSCSUS[idx_order]],
         x_name: amplitudes_eyelid[idxCSCSUS]}))


r_s=[]
r_ss=[]

for df,dfr in zip(dfs,dfs_random): # random scramble

    if bins is None:
        [_,bins]=np.histogram(dfr.ampl_eye,n_bins)         
    groups = dfr.groupby(np.digitize(dfr.ampl_eye, bins))
    grouped_mean = groups.mean()
    grouped_sem = groups.sem()
    (r,p_val)=scipy.stats.pearsonr(grouped_mean.ampl_eye,grouped_mean.ampl_fl)
#    r=np.corrcoef(grouped_mean.ampl_eye,grouped_mean.ampl_fl)[0,1]

    r_ss.append(r)

    if bins is None:
        [_,bins]=np.histogram(df.ampl_eye,n_bins)         

    groups = df.groupby(np.digitize(df.ampl_eye, bins))    
    grouped_mean = groups.mean()
    grouped_sem= groups.sem()    
    (r,p_val)=scipy.stats.pearsonr(grouped_mean.ampl_eye,grouped_mean.ampl_fl)
#    r=np.corrcoef(grouped_mean.ampl_eye,grouped_mean.ampl_fl)[0,1]
    r_s.append(r)    
    if r_s[-1]>thresh__correlation:
        pl.subplot(2,2,3)
        print('found')
        pl.errorbar(grouped_mean.ampl_eye,grouped_mean.ampl_fl,grouped_sem.ampl_fl.as_matrix(),grouped_sem.ampl_eye.as_matrix(),fmt='.')
        pl.scatter(grouped_mean.ampl_eye,grouped_mean.ampl_fl,s=groups.apply(len).values*3)#
        pl.xlabel(x_name)
        pl.ylabel(y_name)

mu_scr=np.mean(r_ss)

std_scr=np.std(r_ss)
[a,b]=np.histogram(r_s,20)

pl.subplot(2,2,4)
pl.plot(b[1:],scipy.signal.savgol_filter(a,3,1))  
pl.axvspan(mu_scr-std_scr, mu_scr+std_scr, color='r', alpha=0.2, lw=0)
pl.xlabel('Correlation coefficients')
pl.ylabel('bin counts')
#%%
pl.savefig(base_folder+'correlations.pdf')
pl.close()
