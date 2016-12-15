# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 15:53:15 2016

@author: agiovann
"""
from __future__ import division
from __future__ import print_function
#%%
from builtins import zip
from builtins import str
from builtins import map
from builtins import range
from past.utils import old_div
import cv2
try:
    cv2.setNumThreads(1)
except:
    print('Open CV is naturally single threaded')


try:
    if __IPYTHON__:
        print((1))
        # this is used for debugging purposes only. allows to reload classes when changed
        get_ipython().magic('load_ext autoreload')
        get_ipython().magic('autoreload 2')
except NameError:
    print('Not IPYTHON')
    pass

import caiman as cm
import numpy as np
import time
import pylab as pl
import psutil
import sys
import os
from ipyparallel import Client
from skimage.external.tifffile import TiffFile
from caiman.motion_correction import tile_and_correct#, motion_correction_piecewise
#%% in parallel
def tile_and_correct_wrapper(params):

    from skimage.external.tifffile import imread
    import numpy as np
    import cv2
    try:
        cv2.setNumThreads(1)
    except:
        1 #'Open CV is naturally single threaded'

    from caiman.motion_correction import tile_and_correct

    img_name,  out_fname,idxs, shape_mov, template, strides, overlaps, max_shifts,\
        add_to_movie,max_deviation_rigid,upsample_factor_grid, newoverlaps, newstrides  = params



    imgs = imread(img_name,key = idxs)
    mc = np.zeros(imgs.shape,dtype = np.float32)
    shift_info = []
    for count, img in enumerate(imgs): 
        if count % 10 == 0:
            print(count)
        mc[count],total_shift,start_step,xy_grid = tile_and_correct(img, template, strides, overlaps,max_shifts, add_to_movie=add_to_movie, newoverlaps = newoverlaps, newstrides = newstrides,\
                upsample_factor_grid= upsample_factor_grid, upsample_factor_fft=10,show_movie=False,max_deviation_rigid=max_deviation_rigid)
        shift_info.append([total_shift,start_step,xy_grid])
    if out_fname is not None:           
        outv = np.memmap(out_fname,mode='r+', dtype=np.float32, shape=shape_mov, order='F')
        outv[:,idxs] = np.reshape(mc.astype(np.float32),(len(imgs),-1),order = 'F').T

    return shift_info, idxs, np.nanmean(mc,0)


#%%
def motion_correction_piecewise(fname, splits, strides, overlaps, add_to_movie=0, template = None, max_shifts = (12,12),max_deviation_rigid = 3,newoverlaps = None, newstrides = None,\
                                upsample_factor_grid = 4, order = 'F',dview = None,save_movie= True, base_name = 'none'):
    '''

    '''

    with TiffFile(fname) as tf:
        d1,d2 = tf[0].shape
        T = len(tf)    

    if type(splits) is int:
        idxs = np.array_split(list(range(T)),splits)
    else:
        idxs = splits
        save_movie = False


    if template is None:
        raise Exception('Not implemented')



    shape_mov =  (d1*d2,T)

    dims = d1,d2

    if save_movie:
        if base_name is None:
            base_name = fname[:-4]

        fname_tot = base_name + '_d1_' + str(dims[0]) + '_d2_' + str(dims[1]) + '_d3_' + str(1 if len(dims) == 2 else dims[2]) + '_order_' + str(order) + '_frames_' + str(T) + '_.mmap'
        fname_tot = os.path.join(os.path.split(fname)[0],fname_tot) 

        np.memmap(fname_tot, mode='w+', dtype=np.float32, shape=shape_mov, order=order)
    else:
        fname_tot = None

    pars = []

    for idx in idxs:
        pars.append([fname,fname_tot,idx,shape_mov, template, strides, overlaps, max_shifts, np.array(add_to_movie,dtype = np.float32),max_deviation_rigid,upsample_factor_grid, newoverlaps, newstrides ])

    t1 = time.time()
    if dview is not None:
        res =dview.map_sync(tile_and_correct_wrapper,pars)
    else:
        res = list(map(tile_and_correct_wrapper,pars))

    print((time.time()-t1))    

    return fname_tot, res
#%%
#backend='SLURM'
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
            cm.stop_server(is_slurm=True)
        except:
            print('Nothing to stop')
        slurm_script = '/mnt/xfs1/home/agiovann/SOFTWARE/Constrained_NMF/SLURM/slurmStart.sh'
        cm.start_server(slurm_script=slurm_script)
        pdir, profile = os.environ['IPPPDIR'], os.environ['IPPPROFILE']
        c = Client(ipython_dir=pdir, profile=profile)
    else:
        cm.stop_server()
        cm.start_server()
        c = Client()

    print(('Using ' + str(len(c)) + ' processes'))
    dview = c[:len(c)]
#%% set parameters and create template by rigid motion correction
fname = 'k56_20160608_RSM_125um_41mW_zoom2p2_00001_00034.tif'
#fname = 'M_FLUO_t.tif'
#fname = 'M_FLUO_4.tif'
max_shifts = (12,12)
m = cm.load(fname,subindices=slice(None,None,None))
template = cm.motion_correction.bin_median( m[100:500].copy().motion_correct(max_shifts[0],max_shifts[1],template=None)[0])
pl.imshow(template)
#%%
splits = 28 # for parallelization split the movies in  num_splits chuncks across time
new_templ = template
add_to_movie=-np.min(template)
save_movie = False
num_iter = 3 
for iter_ in range(num_iter):
    print(iter_)
    old_templ = new_templ.copy()
    if iter_ == num_iter-1:
        save_movie = True
        print('saving!')
#        templ_to_save = old_templ

    fname_tot, res = motion_correction_piecewise(fname,splits, None, None,\
                            add_to_movie=add_to_movie, template = old_templ, max_shifts = max_shifts,max_deviation_rigid = 0,\
                            newoverlaps = None, newstrides = None,\
                            upsample_factor_grid = 4, order = 'F',dview = dview, save_movie=save_movie ,base_name  = 'rig_sue_')



    new_templ = np.nanmedian(np.dstack([r[-1] for r in res ]),-1)
    print((old_div(np.linalg.norm(new_templ-old_templ),np.linalg.norm(old_templ))))


pl.imshow(new_templ,cmap = 'gray',vmax = np.percentile(new_templ,95))     
#%%
import scipy
np.save(str(np.shape(m)[-1])+'_templ_rigid.npy',new_templ)
scipy.io.savemat('/mnt/xfs1/home/agiovann/dropbox/Python_progress/' + str(np.shape(m)[-1])+'_templ_rigid.mat',{'template':new_templ}) 
#%%
template = new_templ
#%%
mr = cm.load(fname_tot)
#%% online does not seem to work!
#overlaps = (16,16)
#if template.shape == (512,512):
#    strides = (128,128)# 512 512 
#    #strides = (48,48)# 128 64
#elif template.shape == (64,128):
#    strides = (48,48)# 512 512
#else:
#    raise Exception('Unknown size, set manually')    
#upsample_factor_grid = 4
#
#T = m.shape[0]
#idxs_outer = np.array_split(range(T),T/1000)
#for iddx in idxs_outer:
#    num_fr = len(iddx)
#    splits = np.array_split(iddx,num_fr/n_processes)
#    print (splits[0][0]),(splits[-1][-1])
#    fname_tot, res = motion_correction_piecewise(fname,splits, strides, overlaps,\
#                            add_to_movie=add_to_movie, template = template, max_shifts = (12,12),max_deviation_rigid = 3,\
#                            upsample_factor_grid = upsample_factor_grid,dview = dview)
#%%
## for 512 512 this seems good
overlaps = (64,64)
if template.shape == (512,512):
    strides = (128,128)# 512 512 
    strides = (16,16)# 512 512 
    newoverlaps = None
    newstrides = None
    #strides = (48,48)# 128 64
elif template.shape == (64,128):
    strides = (48,48)# 
    newoverlaps = None
    newstrides = None
else:
    raise Exception('Unknown size, set manually')    

splits = 28 # for parallelization split the movies in  num_splits chuncks across time

upsample_factor_grid = 1
max_deviation_rigid = 3
new_templ = template
add_to_movie = -np.min(new_templ)
num_iter = 2
save_movie = False

for iter_ in range(num_iter):
    print(iter_)
    old_templ = new_templ.copy()

    if iter_ == num_iter-1:
        save_movie = True
        print('saving!')



    fname_tot, res = motion_correction_piecewise(fname,splits, strides, overlaps,\
                            add_to_movie=add_to_movie, template = old_templ, max_shifts = (12,12),max_deviation_rigid = max_deviation_rigid,\
                            newoverlaps = newoverlaps, newstrides = newstrides,\
                            upsample_factor_grid = upsample_factor_grid, order = 'F',dview = dview,save_movie = save_movie, base_name = 'els_sue_')


    new_templ = np.nanmedian(np.dstack([r[-1] for r in res ]),-1)
    print((old_div(np.linalg.norm(new_templ-old_templ),np.linalg.norm(old_templ))))
    pl.imshow(new_templ,cmap = 'gray',vmax = np.percentile(new_templ,99))
    pl.pause(.1)

mc = cm.load(fname_tot)        
 #%%
np.save(str(np.shape(m)[-1])+'_templ_pw_rigid.npy',templ_to_save)
scipy.io.savemat('/mnt/xfs1/home/agiovann/dropbox/Python_progress/' + str(np.shape(m)[-1])+'_templ_pw_rigid.mat',{'template':templ_to_save}) 
#%%
#%
#total_shifts = []
#start_steps = []
#xy_grids = []
#mc = np.zeros(m.shape)
#for count,img in enumerate(np.array(m)):
#    if count % 10  == 0:
#        print count
#    mc[count],total_shift,start_step,xy_grid = tile_and_correct(img, template, strides, overlaps,(12,12), newoverlaps = None, \
#                newstrides = newstrides, upsample_factor_grid=upsample_factor_grid,\
#                upsample_factor_fft=10,show_movie=False,max_deviation_rigid=2,add_to_movie=add_to_movie)
#    
#    total_shifts.append(total_shift)
#    start_steps.append(start_step)
#    xy_grids.append(xy_grid)



#mc = cm.load('M_FLUO_4_d1_64_d2_128_d3_1_order_F_frames_4620_.mmap')
#mc = cm.load('M_FLUO_t_d1_64_d2_128_d3_1_order_F_frames_6764_.mmap')

#%%
mc.resize(1,1,.2).play(gain=30,fr = 30, offset = 300,magnification=1.)
#%%
m.resize(1,1,.2).play(gain=10,fr = 30, offset = 0,magnification=1.)
#%%
cm.concatenate([mr.resize(1,1,.5),mc.resize(1,1,.5)],axis=1).play(gain=10,fr = 100, offset = 300,magnification=1.)

#%%
import h5py
with  h5py.File('sueann_pw_rigid_movie.mat') as f:
    mef = np.array(f['M2'])

mef = cm.movie(mef.transpose([0,2,1]))    

#%%
cm.concatenate([mef.resize(1,1,.15),mc.resize(1,1,.15)],axis=1).play(gain=30,fr = 40, offset = 300,magnification=1.)
#%%
(mef-mc).resize(1,1,.1).play(gain=50,fr = 20, offset = 0,magnification=1.)
#%%
(mc-mef).resize(1,1,.1).play(gain=50,fr = 20, offset = 0,magnification=1.)
#%%
T,d1,d2 = np.shape(m)
shape_mov = (d1*d2,m.shape[0])

Y = np.memmap('M_FLUO_4_d1_64_d2_128_d3_1_order_F_frames_4620_.mmap',mode = 'r',dtype=np.float32, shape=shape_mov, order='F')
mc = cm.movie(np.reshape(Y,(d2,d1,T),order = 'F').transpose([2,1,0]))
mc.resize(1,1,.25).play(gain=10.,fr=50)
#%%
pl.plot(np.reshape(np.array(total_shifts),(len(total_shifts),-1))) 
#%%
#m_raw = cm.motion_correction.bin_median(m,exclude_nans=True)
#m_rig = cm.motion_correction.bin_median(mr,exclude_nans=True)
#m_el = cm.motion_correction.bin_median(mc,exclude_nans=True)

m_raw = np.nanmean(m,0)
m_rig = np.nanmean(mr,0)
m_el = np.nanmean(mc,0)
m_ef = np.nanmean(mef,0)
#%%
import scipy
r_raw = []
r_rig = []
r_el = []
r_ef = []
max_shft = 12
for fr_id in range(m.shape[0]):
    fr = m[fr_id].copy()[max_shft:-max_shft,max_shft :-max_shft]
    templ_ = m_raw.copy()[max_shft:-max_shft,max_shft :-max_shft]
    r_raw.append(scipy.stats.pearsonr(fr.flatten(),templ_.flatten())[0]) 

    fr = mr[fr_id].copy()[max_shft:-max_shft,max_shft :-max_shft]
    templ_ = m_rig.copy()[max_shft:-max_shft,max_shft :-max_shft]    
    r_rig.append(scipy.stats.pearsonr(fr.flatten(),templ_.flatten())[0]) 

    fr = mc[fr_id].copy()[max_shft:-max_shft,max_shft :-max_shft]
    templ_ = m_el.copy()[max_shft:-max_shft,max_shft :-max_shft]    
    r_el.append(scipy.stats.pearsonr(fr.flatten(),templ_.flatten())[0])        
    if 1:
        fr = mef[fr_id].copy()[max_shft:-max_shft,max_shft :-max_shft]
        templ_ = m_ef.copy()[max_shft:-max_shft,max_shft :-max_shft]    
        r_ef.append(scipy.stats.pearsonr(fr.flatten(),templ_.flatten())[0])   

r_raw =np.array(r_raw)
r_rig =np.array(r_rig)     
r_el =np.array(r_el)     
r_ef =np.array(r_ef)        
#%%
#r_ef = scipy.io.loadmat('sueann.mat')['cM2'].squeeze()
#r_efr = scipy.io.loadmat('sueann.mat')['cY'].squeeze()
#pl.close()
#%%
pl.plot(r_raw)
pl.plot(r_rig)
pl.plot(r_el)
pl.plot(r_ef)
#%%
pl.scatter(r_el,r_ef)
pl.plot([0,1],[0,1],'r--')

#%%
pl.plot(old_div((r_ef-r_el),np.abs(r_el)))
#%%
import pylab as pl
vmax=150
max_shft = 3
pl.subplot(2,3,1)
pl.scatter(r_raw,r_rig)
pl.plot([0,1],[0,1],'r--')
pl.xlabel('raw')
pl.ylabel('rigid')
pl.xlim([0,1])
pl.ylim([0,1])
pl.subplot(2,3,2)
pl.scatter(r_rig,r_el)
pl.plot([0,1],[0,1],'r--')
pl.ylabel('pw-rigid')
pl.xlabel('rigid')
pl.xlim([0,1])
pl.ylim([0,1])
#%



pl.subplot(2,3,4);
pl.title('rigid mean')
pl.imshow(np.nanmean(mr,0)[max_shft:-max_shft,max_shft:-max_shft],cmap='gray', vmax = vmax,interpolation = 'none');

pl.axis('off')
pl.subplot(2,3,5);
pl.imshow(np.nanmean(mc,0)[max_shft:-max_shft,max_shft:-max_shft],cmap='gray',vmax = vmax,interpolation = 'none')
pl.title('pw-rigid mean')
pl.axis('off')


if 0:
    pl.subplot(2,3,3);
    pl.scatter(r_el,r_ef)
    pl.plot([0,1],[0,1],'r--')
    pl.ylabel('pw-rigid')
    pl.xlabel('pw-rigid eft')
    pl.xlim([0,1])
    pl.ylim([0,1])

    pl.subplot(2,3,6);
    pl.imshow(np.nanmean(mef,0)[max_shft:-max_shft,max_shft:-max_shft],cmap='gray',vmax = vmax,interpolation = 'none')
    pl.title('pw-rigid eft mean')
    pl.axis('off')

#%%
pl.plot(r_ef)
#%%
mc = cm.movie(mc)
mc[np.isnan(mc)] = 0
#%% play movie
(mc+add_to_movie).resize(1,1,.25).play(gain=10.,fr=50)
#%% compute correlation images
ccimage = m.local_correlations(eight_neighbours=True,swap_dim=False)
ccimage_rig = mr.local_correlations(eight_neighbours=True,swap_dim=False)
ccimage_els = mc.local_correlations(eight_neighbours=True,swap_dim=False)
ccimage_ef = mef.local_correlations(eight_neighbours=True,swap_dim=False)
#%% check correlation images
pl.subplot(2,2,1)
pl.imshow(ccimage,vmin = 0, vmax = 0.6, interpolation = 'none')
pl.subplot(2,2,2)
pl.imshow(ccimage_rig,vmin = 0, vmax = 0.6, interpolation = 'none')
pl.subplot(2,2,3)
pl.imshow(ccimage_els,vmin = 0, vmax = 0.6, interpolation = 'none')
pl.subplot(2,2,4)
pl.imshow(ccimage_ef,vmin = 0, vmax = 0.6, interpolation = 'none')
#%%
all_mags = [] 
all_mags_eig = [] 
for chunk in res:
    for frame in chunk[0]:
        shifts,pos,init = frame
        x_sh = np.zeros(np.add(init[-1],1))
        y_sh = np.zeros(np.add(init[-1],1))

        for nt,sh in zip(init,shifts):
            x_sh[nt] = sh[0]
            y_sh[nt] = sh[1]

        jac_xx = x_sh[1:,:] - x_sh[:-1,:]
        jac_yx = y_sh[1:,:] - y_sh[:-1,:]
        jac_xy = x_sh[:,1:] - x_sh[:,:-1]
        jac_yy = y_sh[:,1:] - y_sh[:,:-1]

        mag_norm = np.sqrt(jac_xx[:,:-1]**2 + jac_yx[:,:-1]**2 + jac_xy[:-1,:]**2 +  jac_yy[:-1,:]**2)
        all_mags.append(mag_norm)


#        pl.cla()            
#        pl.imshow(mag_der,vmin=0,vmax =1,interpolation = 'none')
#        pl.pause(.1)
#%%
mam = cm.movie(np.dstack(all_mags)).transpose([2,0,1])
mam.play(magnification=10,gain = 5.)
#%%
pl.imshow(np.max(mam,0),interpolation = 'none')
#%%
m = cm.load('rig_sue__d1_512_d2_512_d3_1_order_F_frames_3000_.mmap')
m1 = cm.load('els_sue__d1_512_d2_512_d3_1_order_F_frames_3000_.mmap')
m0 = cm.load('k56_20160608_RSM_125um_41mW_zoom2p2_00001_00034.tif')
tmpl = cm.motion_correction.bin_median(m)
tmpl1 = cm.motion_correction.bin_median(m1)
tmpl0 = cm.motion_correction.bin_median(m0)

#%%
vmin, vmax = -1, 1
count = 0
pyr_scale = .5 
levels = 3
winsize = 100 
iterations = 15
poly_n = 5
poly_sigma = old_div(1.2,5)
flags = 0 #cv2.OPTFLOW_FARNEBACK_GAUSSIAN
norms = []
flows = []
for fr,fr1,fr0 in zip(m.resize(1,1,.2),m1.resize(1,1,.2),m0.resize(1,1,.2)):
    count +=1
    print(count)

    flow1 = cv2.calcOpticalFlowFarneback(tmpl1[12:-12,12:-12],fr1[12:-12,12:-12],None,pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags);
    flow = cv2.calcOpticalFlowFarneback(tmpl[12:-12,12:-12],fr[12:-12,12:-12],None,pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags);
    flow0 = cv2.calcOpticalFlowFarneback(tmpl0[12:-12,12:-12],fr0[12:-12,12:-12],None,pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags);

#    pl.subplot(2,3,1)    
#    pl.cla()    
#    pl.imshow(flow1[:,:,1],vmin=vmin,vmax=vmax)       
#    pl.subplot(2,3,2)    
#    pl.cla()
#    pl.imshow(flow[:,:,1],vmin=vmin,vmax=vmax)     
#    pl.subplot(2,3,3)       
#    pl.cla()
#    pl.imshow(flow0[:,:,1],vmin=vmin,vmax=vmax)         
#    
#    pl.subplot(2,3,4)    
#    pl.cla()    
#    pl.imshow(flow1[:,:,0],vmin=vmin,vmax=vmax)       
#    pl.subplot(2,3,5)    
#    pl.cla()
#    pl.imshow(flow[:,:,0],vmin=vmin,vmax=vmax)     
#    pl.subplot(2,3,6)       
#    pl.cla()
#    pl.imshow(flow0[:,:,0],vmin=vmin,vmax=vmax)         
#    pl.pause(.1)
    n1,n,n0 = np.linalg.norm(flow1), np.linalg.norm(flow), np.linalg.norm(flow0)
    flows.append([flow1,flow,flow0])
    norms.append([n1,n,n0 ])
#%%
flm = cm.movie(np.dstack( [np.concatenate(fl[0][:,:,:],axis=0) for fl  in flows])).transpose([2,0,1])

