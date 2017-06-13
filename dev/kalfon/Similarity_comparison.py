##@package demos  
#\brief      for the user/programmer to understand and try the code
#\details    all of other usefull functions (demos available on jupyter notebook) -*- coding: utf-8 -*- 
#\version   1.0
#\pre       EXample.First initialize the system.
#\bug       
#\warning   
#\copyright GNU General Public License v2.0 
#\date Created on Mon Nov 21 15:53:15 2016
#\author agiovann
#toclean

from __future__ import division
from __future__ import print_function
from builtins import zip
from builtins import str
from builtins import range
import cv2
import glob
import platform as plt
import datetime

try:
    cv2.setNumThreads(1)
except:
    print('Open CV is naturally single threaded')

try:
    if __IPYTHON__:
        print((1))
        # this is used for debugging purposes only. allows to reload classes
        # when changed
        get_ipython().magic('load_ext autoreload')
        get_ipython().magic('autoreload 2')
except NameError:
    print('Not IPYTHON')
    pass

import caiman as cm
import numpy as np
import os
import time
import pylab as pl
import scipy


from caiman.source_extraction.cnmf import cnmf as cnmf
from caiman.components_evaluation import estimate_components_quality
from caiman.motion_correction import MotionCorrect
from caiman.utils.visualization import plot_contours, view_patches_bar
from caiman.utils.utils import download_demo

#@params params_movie set parameters and create template by RIGID MOTION CORRECTION
params_movie = {'fname': ['example_movies/demoSue2x.tif'],
                'niter_rig': 1,
                'max_shifts': (6, 6),  # maximum allow rigid shift
                'splits_rig': 28,  # for parallelization split the movies in  num_splits chuncks across time
                # if none all the splits are processed and the movie is saved
                'num_splits_to_process_rig': None,
                # intervals at which patches are laid out for motion correction
                'strides': (48, 48),
                # overlap between pathes (size of patch strides+overlaps)
                'overlaps': (24, 24),
                'splits_els': 28,  # for parallelization split the movies in  num_splits chuncks across time
                # if none all the splits are processed and the movie is saved
                'num_splits_to_process_els': [14, None],
                'upsample_factor_grid': 4,  # upsample factor to avoid smearing when merging patches
                # maximum deviation allowed for patch with respect to rigid
                # shift
                'max_deviation_rigid': 3,
                'p': 1,  # order of the autoregressive system
                'merge_thresh': 0.8,  # merging threshold, max correlation allowed
                'rf': 15,  # half-size of the patches in pixels. rf=25, patches are 50x50
                'stride_cnmf': 6,  # amounpl.it of overlap between the patches in pixels
                'K': 4,  # number of components per patch
                # if dendritic. In this case you need to set init_method to
                # sparse_nmf
                'is_dendrites': False,
                'init_method': 'greedy_roi',
                'gSig': [4, 4],  # expected half size of neurons
                'alpha_snmf': None,  # this controls sparsity
                'final_frate': 30
                }

#%%
#Precision you want for the validation of your comparison
comparison ={'rig_shifts': {},
             'pwrig_shifts': {},
             'cnmf_on_patch': {},
             'cnmf_full_frame': {},
             'is_ground_truth': True,
             'ground_truth':None,
             'sensibility':None
        }
#the sensibility USER TO CHOOSE
comparison['sensibility']={  
        'rig_shifts': 0.001,
        'pwrig_shifts':0.1,
        'cnmf_on_patch':0.01,
        'cnmf_full_frame':0.001
        }
comparison['rig_shifts']={
                          'ourdata': None,
                          'timers': None
                         }
comparison['pwrig_shifts']={
                          'ourdata': None,
                          'timers': None
                         }
comparison['cnmf_on_patch']={
                          'ourdata': None,
                          'timers': None
                         }
comparison['cnmf_full_frame']={
                          'ourdata': None,
                          'timers': None
                         }
#%% load movie (in memory!)

#TODO: do find&replace on those parameters and delete this paragrph

#@params fname name of the movie 
fname = params_movie['fname']

niter_rig = params_movie['niter_rig']

#@params max_shifts maximum allow rigid shift
max_shifts = params_movie['max_shifts']  

#@params splits_rig for parallelization split the movies in  num_splits chuncks across time
splits_rig = params_movie['splits_rig']  

#@params num_splits_to_process_ri if none all the splits are processed and the movie is saved
num_splits_to_process_rig = params_movie['num_splits_to_process_rig']

#@params strides intervals at which patches are laid out for motion correction
strides = params_movie['strides']

#@ prams overlaps overlap between pathes (size of patch strides+overlaps)
overlaps = params_movie['overlaps']

#@params splits_els for parallelization split the movies in  num_splits chuncks across time
splits_els = params_movie['splits_els'] 

#@params num_splits_to_process_els  if none all the splits are processed and the movie is saved
num_splits_to_process_els = params_movie['num_splits_to_process_els']

#@params upsample_factor_grid upsample factor to avoid smearing when merging patches
upsample_factor_grid = params_movie['upsample_factor_grid'] 

#@params max_deviation_rigid maximum deviation allowed for patch with respect to rigid shift
max_deviation_rigid = params_movie['max_deviation_rigid']

#%% download movie if not there
if fname == 'example_movies/demoSue2x.tif':
    #TODO: todocument
    download_demo()
    #TODO: todocument
m_orig = cm.load_movie_chain(fname[:1])


#TODO: load screenshot 1
#%% RUN ANALYSIS
c, dview, n_processes = cm.cluster.setup_cluster(
    backend='local', n_processes=20, single_thread=False)


# movie must be mostly positive for this to work
#TODO : TOdocument
#setting timer to see how the changement in functions make the code react on a same computer. 
t1 = time.time()
t=[]
min_mov = cm.load(fname[0], subindices=range(400)).min()
mc_list = []
new_templ = None
for each_file in fname:
    #TODO: needinfo how the classes works 
    mc = MotionCorrect(each_file, min_mov,
                   dview=dview, max_shifts=max_shifts, niter_rig=niter_rig, splits_rig=splits_rig, 
                   num_splits_to_process_rig=num_splits_to_process_rig,
                   shifts_opencv = True, nonneg_movie = True)
    mc.motion_correct_rigid(template = new_templ, save_movie=True)
    new_templ = mc.total_template_rig
    m_rig = cm.load(mc.fname_tot_rig)


    mc_list.append(mc)

comparison['rig_shifts']['timer'] = time.time() - t1
#TODO: show screenshot 2,3
#%% store rigid shifts
comparison['rig_shifts']['ourdata']=mc.shifts_rig
#%% plot rigid shifts
pl.close()
pl.plot(mc.shifts_rig)
pl.legend(['x shifts','y shifts'])
pl.xlabel('frames')
pl.ylabel('pixels')
#TODO: show screenshot 4

#%%
#a computing intensive but parralellized part
t1 = time.time()
mc.motion_correct_pwrigid(save_movie=True,
                          template=mc.total_template_rig, show_template = True)
#TODO: change var name els= pwr
m_els = cm.load(mc.fname_tot_els)
#TODO: show screenshot 5
#TODO: bug sometimes saying there is no y_shifts_els 
bord_px_els = np.ceil(np.maximum(np.max(np.abs(mc.x_shifts_els)),
                                 np.max(np.abs(mc.y_shifts_els)))).astype(np.int)
comparison['pwrig_shifts']['timer'] = time.time() - t1
comparison['pwrig_shifts']['ourdata'] = [mc.x_shifts_els, mc.y_shifts_els]
        
#%% visualize elastic shifts
pl.close()
pl.subplot(2, 1, 1) 
pl.plot(mc.x_shifts_els)
pl.ylabel('x shifts (pixels)')
pl.subplot(2, 1, 2)
pl.plot(mc.y_shifts_els)
pl.ylabel('y_shifts (pixels)')
pl.xlabel('frames')
#TODO: show screenshot 6
#%% compute metrics for the results, just to check that motion correction worked properly
t1 = time.time()
final_size = np.subtract(mc.total_template_els.shape, 2 * bord_px_els)
winsize = 100
swap_dim = False
resize_fact_flow = .2
#computationnaly intensive 
#TODO: todocument
tmpl, correlations, flows_orig, norms, smoothness = cm.motion_correction.compute_metrics_motion_correction(
    mc.fname_tot_els, final_size[0], final_size[1], swap_dim, winsize=winsize, play_flow=False, resize_fact_flow=resize_fact_flow)
tmpl, correlations, flows_orig, norms, smoothness = cm.motion_correction.compute_metrics_motion_correction(
    mc.fname_tot_rig, final_size[0], final_size[1], swap_dim, winsize=winsize, play_flow=False, resize_fact_flow=resize_fact_flow)
tmpl, correlations, flows_orig, norms, smoothness = cm.motion_correction.compute_metrics_motion_correction(
    fname, final_size[0], final_size[1], swap_dim, winsize=winsize, play_flow=False, resize_fact_flow=resize_fact_flow)
t[3] = time.time() - t1
#%% plot the results of metrics
fls = [mc.fname_tot_els[:-4] + '_metrics.npz', mc.fname_tot_rig[:-4] +
       '_metrics.npz', mc.fname[:-4] + '_metrics.npz']
#%%
for cnt, fl, metr in zip(range(len(fls)),fls,['pw_rigid','rigid','raw']):
    with np.load(fl) as ld:
        print(ld.keys())
#        pl.figure()
        print(fl)
        print(str(np.mean(ld['norms'])) + '+/-' + str(np.std(ld['norms'])) +
              ' ; ' + str(ld['smoothness']) + ' ; ' + str(ld['smoothness_corr']))
        #here was standing an iftrue ..
        pl.subplot(len(fls), 4, 1 + 4 * cnt)
        pl.ylabel(metr)
        try:
            mean_img = np.mean(
                cm.load(fl[:-12] + 'mmap'), 0)[12:-12, 12:-12]
        except:
            try:
                mean_img = np.mean(
                    cm.load(fl[:-12] + '.tif'), 0)[12:-12, 12:-12]
            except:
                mean_img = np.mean(
                    cm.load(fl[:-12] + 'hdf5'), 0)[12:-12, 12:-12]
                
        lq, hq = np.nanpercentile(mean_img, [.5, 99.5])
        pl.imshow(mean_img, vmin=lq, vmax=hq)
        pl.title('Mean')
    #        pl.plot(ld['correlations'])

        pl.subplot(len(fls), 4, 4 * cnt + 2)
        pl.imshow(ld['img_corr'], vmin=0, vmax=.35)
        pl.title('Corr image')
#        pl.colorbar()
        pl.subplot(len(fls), 4, 4 * cnt + 3)
#
        pl.plot(ld['norms'])
        pl.xlabel('frame')
        pl.ylabel('norm opt flow')
        pl.subplot(len(fls), 4, 4 * cnt + 4)
        flows = ld['flows']
        pl.imshow(np.mean(
            np.sqrt(flows[:, :, :, 0]**2 + flows[:, :, :, 1]**2), 0), vmin=0, vmax=0.3)
        pl.colorbar()
        pl.title('Mean optical flow')
#TODO: show screenshot 9
#%% restart cluster to clean up memory
#TODO: todocument
c, dview, n_processes = cm.cluster.setup_cluster(
    backend='local', n_processes=None, single_thread=False)    
#%% save each chunk in F format
t1 = time.time()
if not params_movie.has_key('max_shifts'):
    fnames = [params_movie['fname']]
    border_to_0 = 0
elif not params_movie.has_key('overlaps'):
    fnames = [mc.fname_tot_rig]
    border_to_0 = bord_px_rig
    m_els = m_rig
else:
    fnames = [mc.fname_tot_els]
    border_to_0 = bord_px_els
    
# if you need to crop the borders use slicing    
# idx_x=slice(border_nan,-border_nan,None)
# idx_y=slice(border_nan,-border_nan,None)
# idx_xy=(idx_x,idx_y)
idx_xy = None
#TODO: needinfo
add_to_movie = -np.nanmin(m_els) + 1  # movie must be positive
# if you need to remove frames from the beginning of each file
remove_init = 0
# downsample movie in time: use .2 or .1 if file is large and you want a quick answer             
downsample_factor = 1 
base_name = fname[0].split('/')[-1][:-4]
#TODO: todocument
name_new = cm.save_memmap_each(fnames, dview=dview, base_name=base_name, resize_fact=(
    1, 1, downsample_factor), remove_init=remove_init, idx_xy=idx_xy, add_to_movie=add_to_movie, border_to_0=border_to_0)
name_new.sort()
print(name_new)

#%% if multiple files were saved in C format, now put them together in a single large file. 
if len(name_new) > 1:
    fname_new = cm.save_memmap_join(

        name_new, base_name='Yr', n_chunks=30, dview=dview)
else:
    print('One file only, not saving!')
    fname_new = name_new[0]



#%% LOAD MEMMAP FILE
# fname_new='Yr_d1_501_d2_398_d3_1_order_F_frames_369_.mmap'
Yr, dims, T = cm.load_memmap(fname_new)
d1, d2 = dims
images = np.reshape(Yr.T, [T] + list(dims), order='F')
#TODO: needinfo
Y = np.reshape(Yr, dims + (T,), order='F')
m_images = cm.movie(images)
t[4] = time.time() - t1
#TODO: show screenshot 10
#%%  checks on movies 
#computationnally intensive
if np.min(images) < 0:
    #TODO: should do this in an automatic fashion with a while loop at the 367 line
    raise Exception('Movie too negative, add_to_movie should be larger')
if np.sum(np.isnan(images)) > 0:
    #TODO: same here
    raise Exception('Movie contains nan! You did not remove enough borders')
#%% correlation image
#TODO: needinfo it is not the same and not used
#for fff in fname_new:
#    Cn = cm.movie(images[:1000]).local_correlations(eight_neighbours=True,swap_dim=True)
#    #Cn[np.isnan(Cn)] = 0
#    pl.imshow(Cn, cmap='gray', vmax=.35)
#%% correlation image
Cn = cm.local_correlations(Y)
Cn[np.isnan(Cn)] = 0
pl.imshow(Cn, cmap='gray', vmax=.35)
#TODO: show screenshot 11
#%% some parameter settings
# order of the autoregressive fit to calcium imaging in general one (slow gcamps) or two (fast gcamps fast scanning)
p = params_movie['p']  
# merging threshold, max correlation allowed
merge_thresh= params_movie['merge_thresh'] 
# half-size of the patches in pixels. rf=25, patches are 50x50
rf = params_movie['rf']  
# amounpl.it of overlap between the patches in pixels
stride_cnmf = params_movie['stride_cnmf'] 
 # number of components per patch
K =  params_movie['K'] 
# if dendritic. In this case you need to set init_method to sparse_nmf
is_dendrites = params_movie['is_dendrites']
# iinit method can be greedy_roi for round shapes or sparse_nmf for denritic data
init_method = params_movie['init_method']
# expected half size of neurons
gSig = params_movie['gSig']  
# this controls sparsity
alpha_snmf = params_movie['alpha_snmf']  
#frame rate of movie (even considering eventual downsampling)
final_frate = params_movie['final_frate']


if params_movie['is_dendrites'] == True:
    if params_movie['init_method'] is not 'sparse_nmf':
        raise Exception('dendritic requires sparse_nmf')
    if params_movie['alpha_snmf'] is None:
        raise Exception('need to set a value for alpha_snmf')
#%% Extract spatial and temporal components on patches
t1 = time.time()
#TODO: todocument
#TODO: warnings 3
cnm = cnmf.CNMF(n_processes, k=K, gSig=gSig, merge_thresh=0.8, p=0, dview=dview, Ain=None, rf=rf, stride=stride_cnmf, memory_fact=1,
                method_init=init_method, alpha_snmf=alpha_snmf, only_init_patch=True, gnb=1, method_deconvolution='oasis')
cnm = cnm.fit(images)

A_tot = cnm.A
C_tot = cnm.C
YrA_tot = cnm.YrA
b_tot = cnm.b
f_tot = cnm.f
sn_tot = cnm.sn
comparison['cnmf_on_patch']['timer'] = time.time() - t1
comparison['cnmf_on_patch']['ourdata'] = [cnm.A, cnm.C]

print(('Number of components:' + str(A_tot.shape[-1])))
#%% DISCARD LOW QUALITY COMPONENT
t1 = time.time()
final_frate = params_movie['final_frate']
r_values_min = .7  # threshold on space consistency
fitness_min = -40  # threshold on time variability
# threshold on time variability (if nonsparse activity)
fitness_delta_min = -40
Npeaks = 10
traces = C_tot + YrA_tot
#TODO: todocument
idx_components, idx_components_bad = estimate_components_quality(
    traces, Y, A_tot, C_tot, b_tot, f_tot, final_frate=final_frate, Npeaks=Npeaks, r_values_min=r_values_min, fitness_min=fitness_min, fitness_delta_min=fitness_delta_min)
t[6] = time.time() - t1
print(('Keeping ' + str(len(idx_components)) +
       ' and discarding  ' + str(len(idx_components_bad))))
#%%
A_tot = A_tot.tocsc()[:, idx_components]
C_tot = C_tot[idx_components]
#%% rerun updating the components to refine
t1 = time.time()
cnm = cnmf.CNMF(n_processes, k=A_tot.shape, gSig=gSig, merge_thresh=merge_thresh, p=p, dview=dview, Ain=A_tot, Cin=C_tot,
                f_in=f_tot, rf=None, stride=None, method_deconvolution='oasis')
cnm = cnm.fit(images)
comparison['cnmf_full_frame']['timer'] = time.time() - t1
comparison['cnmf_full_frame']['ourdata'] = [cnm.A, cnm.C]
#%%
A, C, b, f, YrA, sn = cnm.A, cnm.C, cnm.b, cnm.f, cnm.YrA, cnm.sn
#%% again recheck quality of components, stricter criteria
final_frate = params_movie['final_frate']
r_values_min = .85
fitness_min = - 50
fitness_delta_min = - 50
Npeaks = 10
traces = C + YrA
idx_components,idx_components_bad, fitness_raw, fitness_delta, r_values = estimate_components_quality(
    traces, Y, A, C, b, f, final_frate=final_frate, Npeaks=Npeaks, r_values_min=r_values_min, fitness_min=fitness_min, fitness_delta_min=fitness_delta_min, return_all = True)
print(' ***** ')
print((len(traces)))
print((len(idx_components)))
t[7] = time.time() - t1

#%% STOP CLUSTER and clean up log files
#TODO: todocument
cm.stop_server()

log_files = glob.glob('Yr*_LOG_*')
for log_file in log_files:
    os.remove(log_file)
#%% reconstruct denoised movie

#%%
#writing the information about the efficiency of the code
dt = datetime.date.today()
dt=str(dt)
plat=plt.platform()
plat=str(plat)
pro=plt.processor()
pro=str(pro)
information ={
        'platform': plat,
        'processor':pro,
        'values':None
        }
#%%
try:
    with np.load('CaImAn/dev/kalfon/efficiency.npz') as data:  
        
        
        #if we want the ground truth:
        if comparison['is_ground_truth']:
            data['ground_truth']=None
            for k in comparison.keys:
                if type(comparison[k]) is dict:
                    #ma que bella, we put everything into the ground truth       
                    data['ground_truth']['information']['values'].append({
                                k: { 'data': comparison[k]['ourdata'],
                                     'timer': comparison[k]['timer']
                                     }
                                })
#we add the parameters of the ground truth in the same fashion as in everyother entry
            data['ground_truth'].append({
                    'params':params_movie
                    })
            data['groun_truth']['information'].append({
                    'platform':plat,
                    'processor':pro
                    })
            print('we now have ground truth')
        else:
            #if not we save the value of the difference into 
#for rigid
           A = data['ground_truth']['rig_shifts'][()] #we do this [()] because of REASONS
           B = comparison['rig_shifts'] 
           A = np.linalg.norm(A-B)/np.linalg.norm(A)
           B = A < comparison['sensibility']['rig_shifts'][()] #we do this [()] because of REASONS
           information['values'].append({
                   'rig_shifts' :{'isdifferent':B,
                                  'difference': A,
                                  'timing': data['ground_truth']['rig_shifts']['timer']
                                  - comparison['rig_shifts']['timer']
                                }
                        })
#for pwrigid
           A= data['ground_truth']['pwrig_shifts'][0][()]
           B = comparison['pwrig_shifts'][0]
           C = np.linalg.norm(A-B)/np.linalg.norm(A)
           #there is xs and ys
           A= data['ground_truth']['pwrig_shifts'][1][()]
           B = comparison['pwrig_shifts'][1]
           A = np.linalg.norm(A-B)/np.linalg.norm(A)
           A=A+C
           B = A < comparison['sensibility']['pwrig_shifts']
           information['values'].append({
                   'pwrig_shifts' :{'isdifferent':B,
                                    'difference': A,
                                    'timing': data['ground_truth']['pwrig_shifts']['timer']
                                    - comparison['pwrig_shifts']['timer']
                                }
                        })
#for cnmf on patches 
           A= data['ground_truth']['cnmf_on_patch'][0][()]
           B = comparison['cnmf_on_patch'][0]
           C = np.linalg.norm(A-B)/np.linalg.norm(A)
           #there is temporal and spatial
           A= data['ground_truth']['cnmf_on_patch'][1][()]
           B = comparison['cnmf_on_patch'][1]
           A = np.linalg.norm(A-B)/np.linalg.norm(A)
           A=A+C
           B = A < comparison['sensibility']['cnmf_on_patch']
           information['values'].append({
                   'cnmf_on_patch' :{'isdifferent':B,
                                    'difference': A,
                                    'timing': data['ground_truth']['cnmf_on_patch']['timer']
                                    - comparison['cnmf_on_patch']['timer']
                                }
                        })
#for cnmf full frame
           A= data['ground_truth']['cnmf_full_frame'][0][()]
           B = comparison['cnmf_full_frame'][0]
           C = np.linalg.norm(A-B)/np.linalg.norm(A)
           #there is temporal and spatial
           A= data['ground_truth']['cnmf_full_frame'][1][()]
           B = comparison['cnmf_full_frame'][1]
           A = np.linalg.norm(A-B)/np.linalg.norm(A)
           A=A+C
           B = A < comparison['sensibility']['cnmf_full_frame']
           information['values'].append({
                   'cnmf_full_frame' :{'isdifferent':B,
                                    'difference': A,
                                    'timing': data['ground_truth']['cnmf_full_frame']['timer']
                                    - comparison['cnmf_full_frame']['timer']
                                }
                        })
           #we put everything back into data
           data['entries'].append({
                   'information': information,
                   'parameters': params_movie,
                   'sensibility': comparison['sensibility']
                   })
        #if we cannot manage to open it or it doesnt exist:
    np.savez('CaImAn/dev/kalfon/efficiency.npz', **data)
    
    
except (IOError, OSError) as e:
    
    
    data={'ground_truth':None,
          'entries':None
            }
    for k in comparison.keys:
                if type(comparison[k]) is dict:
                    #ma que bella, we put everything into the ground truth       
                    data['ground_truth']['information']['values'].append({
                                k: { 'data': comparison[k]['ourdata'],
                                     'timer': comparison[k]['timer']
                                     }
                                })
#we add the parameters of the ground truth in the same fashion as in everyother entry
    data['ground_truth'].append({
                    'params':params_movie
                    })
    data['groun_truth']['information'].append({
                    'platform':plat,
                    'processor':pro
                    })
    print('we finally have ground truth')
              
    

