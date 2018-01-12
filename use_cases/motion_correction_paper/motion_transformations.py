#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on Tue May 23 21:41:17 2017

@author: epnevmatikakis
"""

import numpy as np

try:
    if __IPYTHON__:
        print('Debugging!')
        # this is used for debugging purposes only. allows to reload classes when changed
        get_ipython().magic('load_ext autoreload')
        get_ipython().magic('autoreload 2')
except NameError:
    print('Not launched under iPython')

import caiman as cm

import cv2
import caiman
import pylab as pl
import pandas
#%%


def create_rotation_field(max_displacement=(10, 10), center=(0, 0), nx=512, ny=512):

    x = np.linspace(-max_displacement[0] - center[0],
                    max_displacement[0] - center[0], nx)
    y = np.linspace(-max_displacement[1] - center[1],
                    max_displacement[1] - center[1], ny)
    X, Y = np.meshgrid(x, y)
    rho = np.sqrt(X**2, Y**2)
    theta = np.arctan2(Y, X)

    vx = -np.sin(theta) * rho
    vy = np.cos(theta) * rho

    return vx.astype(np.float32), vy.astype(np.float32)


def create_stretching_field(stretching_factor=(1., 1.), max_displacement=(5, 5), center=(0, 0), nx=512, ny=512):
    '''max shift would be equal to (max_displacement-center)*(stretching_factor-1)'''

    x = np.linspace(-max_displacement[0] - center[0],
                    max_displacement[0] - center[0], nx)
    y = np.linspace(-max_displacement[1] - center[1],
                    max_displacement[1] - center[1], ny)
    X, Y = np.meshgrid(x, y)

    X_stretch = X * stretching_factor[0]
    Y_stretch = Y * stretching_factor[1]

    vx = X_stretch - X
    vy = Y_stretch - Y

    return vx.astype(np.float32), vy.astype(np.float32)


def create_shearing_field(shearing_factor=0., max_displacement=(5, 5), along_x=True, center=(0, 0), nx=512, ny=512):
    '''set along_x to False for shearing along y axis'''

    x = np.linspace(-max_displacement[0] - center[0],
                    max_displacement[0] - center[0], nx)
    y = np.linspace(-max_displacement[1] - center[1],
                    max_displacement[1] - center[1], ny)
    X, Y = np.meshgrid(x, y)

    if along_x:
        vx = shearing_factor * Y
        vy = np.zeros((nx, ny))
    else:
        vx = np.zeros((nx, ny))
        vy = shearing_factor * X

    return vx.astype(np.float32), vy.astype(np.float32)

#%


def apply_field(inputImage, vx, vy):
    mapX = np.zeros_like(inputImage, dtype=np.float32)
    mapY = np.zeros_like(inputImage, dtype=np.float32)

    map_orig_x = np.zeros(
        (inputImage.shape[0], inputImage.shape[1]), dtype=np.float32)
    map_orig_y = np.zeros(
        (inputImage.shape[0], inputImage.shape[1]), dtype=np.float32)

    for j in range(inputImage.shape[0]):
        # print(j)
        for i in range(inputImage.shape[1]):
            map_orig_x[j, i] = i
            map_orig_y[j, i] = j

    mapX = map_orig_x + vx
    mapY = map_orig_y + vy
    new_img = cv2.remap(inputImage, mapX, mapY,
                        cv2.INTER_CUBIC, None, cv2.BORDER_CONSTANT)
    return new_img
#%%


def dispOpticalFlow(Image, Flow, Divisor=1):
    "Display image with a visualisation of a flow over the top. A divisor controls the density of the quiver plot."
    PictureShape = np.shape(Image)
    # determine number of quiver points there will be
    Imax = int(PictureShape[0] / Divisor)
    Jmax = int(PictureShape[1] / Divisor)
    # create a blank mask, on which lines will be drawn.
    mask = np.zeros_like(Image)
    for i in range(1, Imax):
        for j in range(1, Jmax):
            X1 = (i) * Divisor
            Y1 = (j) * Divisor
            X2 = int(X1 + Flow[X1, Y1, 1])
            Y2 = int(Y1 + Flow[X1, Y1, 0])
            X2 = np.clip(X2, 0, PictureShape[0])
            Y2 = np.clip(Y2, 0, PictureShape[1])
            # add all the lines to the mask
#         mask = cv2.line(mask, (Y1,X1),(Y2,X2), [0, 0, 100], 2)
            mask = cv2.arrowedLine(mask, (Y1, X1), (Y2, X2), [100, 0, 0], 1)

    # superpose lines onto image
    img = cv2.add(Image / np.max(Image) * 2, mask)
    # print image
    return img


#%%
if False:
    #%%
    import caiman as cm
    m = cm.load(
        '/mnt/ceph/neuro/Sue_2000_els_opencv__d1_512_d2_512_d3_1_order_F_frames_2000_.hdf5').resize(1, 1, .1)
    templ = np.nanmean(m, 0)
    new_templ = templ * 1. / np.max(np.nanmean(templ, 0)) / 2
    new_templ[np.isnan(new_templ)] = np.nanmean(new_templ)
#    min_val = np.min(new_templ)
    new_templ = new_templ + 0.1

    #%%
    a, b = 256, 256
    n = 512
    r = 280

    y, x = np.ogrid[-a:n - a, -b:n - b]
    mask = x * x + y * y <= r * r

    array = np.zeros((n, n), dtype=np.float32)
    array[mask] = 1
    pl.imshow(array)
    #%% exlcude the shadow regions
    vx_s = []
    vy_s = []
    max_shift = 6
    imgs_with_flow = []
    for count in [max_shift - 1]:
        print(count)
        vx_, vy_ = 0, 0
        vx1, vy1 = create_rotation_field(max_displacement=(
            count % max_shift, count % max_shift), center=(0, 0), nx=512, ny=512)
        vx_ += vx1
        vy_ += vy1
#        vx2,vy2 = create_shearing_field(shearing_factor=2, max_displacement = (count%max_shift,count%max_shift), along_x = True, center = (2,2), nx = 512, ny = 512)
#        vx_ += vx2
#        vy_ += vy2
#        vx_ /= 2
#        vy_ /= 2
#
        pl.subplot(1, 2, 1)
        pl.imshow(vx_, interpolation='none')
        pl.colorbar()
        pl.subplot(1, 2, 2)
        pl.imshow(vy_, interpolation='none')
        pl.colorbar()
#        break
    #    vx,vy = create_stretching_field(stretching_factor=(1.5,1.5),max_displacement=(-count%max_shift,count%max_shift),center=(0,0),nx=512,ny=512)
    #    vx_ += vx
    #    vy_ += vy
        vx_s.append(vx_)
        vy_s.append(vy_)
        imgs_with_flow.append(dispOpticalFlow(np.dstack(
            [new_templ, new_templ, new_templ]) / 100, np.dstack([vx_ * array * 2, vy_ * array * 2]), Divisor=20))
#%% oracle
for patch_size in np.array([24, 48, 96, 128]):
    best_possible_x = cv2.resize(cv2.resize(vx_, tuple(
        np.divide(new_templ.shape, (patch_size, patch_size)))), new_templ.shape)
    best_possible_y = cv2.resize(cv2.resize(vy_, tuple(
        np.divide(new_templ.shape, (patch_size, patch_size)))), new_templ.shape)
    gt = np.dstack([vx_ * array, vy_ * array])
    best_possible = np.dstack(
        [best_possible_x * array, best_possible_y * array])
    print(np.linalg.norm(gt - best_possible) / np.linalg.norm(gt))

    #%%
    all_movs_dist = []
    all_movs_tosave = []
    for patch_size in [32, 48, 96, 128]:
        for noise_sigma in np.arange(0.01, 4, .4, dtype=np.float32):
            for iter_ in range(10):
                vx, vy, img_flow = vx_s[0], vy_s[0], imgs_with_flow[0]
                rand_add = np.random.randn(
                    *new_templ.shape).astype(np.float32) * noise_sigma * (new_templ)
                newim = apply_field(new_templ, vx, vy)
                newim += rand_add
                mov_dist = dict()
                mov_dist['newim'] = newim
                mov_dist['patch_size'] = patch_size
                mov_dist['noise_sigma'] = noise_sigma
                mov_dist['iter_'] = iter_
                mov_dist['vx'] = vx
                mov_dist['vy'] = vy
                all_movs_tosave.append(mov_dist.copy())
                mov_dist['mask'] = array
                mov_dist['template'] = new_templ
                mov_dist['gt_shifts_masked'] = np.dstack(
                    [vx * array, vy * array])
                mov_dist['img_flow'] = img_flow
                print(mov_dist['patch_size'],
                      mov_dist['noise_sigma'], mov_dist['iter_'])
                all_movs_dist.append(mov_dist)
    #%%
    import scipy
    scipy.io.savemat('/mnt/ceph/neuro/DataForPublications/Piecewise-Rigid-Analysis-paper/Simulations/test_simulation_shifts_nrmcorre.mat',
                     {'all_movs_tosave': all_movs_tosave, 'template': new_templ, 'mask': mask})
    np.savez('/mnt/ceph/neuro/DataForPublications/Piecewise-Rigid-Analysis-paper/Simulations/test_simulation_shifts_nrmcorre.npz',
             {'all_movs_dist': all_movs_dist, 'template': new_templ, 'mask': mask})

    #%% res optical flow
    cv2.destroyAllWindows()
    for mov_dist in all_movs_dist:
        print(mov_dist['patch_size'],
              mov_dist['noise_sigma'], mov_dist['iter_'])

        shifts_oflow = caiman.motion_correction.compute_flow_single_frame(
            mov_dist['newim'] * 255, mov_dist['template'] * 255)
        shifts_oflow_masked = np.dstack(
            [shifts_oflow[:, :, 0] * array, shifts_oflow[:, :, 1] * array])
        mov_dist['shifts_oflow'] = shifts_oflow
        mov_dist['shifts_oflow_masked'] = shifts_oflow_masked
        mov_dist['norm_diff_shifts_oflow'] = np.linalg.norm(
            shifts_oflow_masked + mov_dist['gt_shifts_masked']) / np.linalg.norm(mov_dist['gt_shifts_masked'])
        print(mov_dist['norm_diff_shifts_oflow'])
        cv2.imshow('frame', np.concatenate([img_flow, dispOpticalFlow(np.dstack(
            [mov_dist['newim'], mov_dist['newim'], mov_dist['newim']]), 5 * (shifts_oflow_masked + mov_dist['gt_shifts_masked']), Divisor=20)], axis=0))
        cv2.waitKey(1)

        #%%
    cv2.destroyAllWindows()
    for mov_dist in all_movs_dist:
        print(mov_dist['patch_size'],
              mov_dist['noise_sigma'], mov_dist['iter_'])
        newim = mov_dist['newim']
        new_templ = mov_dist['template']
        patch_size = mov_dist['patch_size']
        correct_img, total_shifts, start_step, xy_grid = caiman.motion_correction.tile_and_correct(newim, new_templ, (patch_size, patch_size), (48, 48), (max_shift, max_shift), newoverlaps=None, newstrides=None, upsample_factor_grid=4,
                                                                                                   upsample_factor_fft=10, show_movie=False, max_deviation_rigid=max_shift, add_to_movie=0, shifts_opencv=True)
        x_gr, y_gr = list(1 + np.array(xy_grid[-1]))
        shifts_ncorre = np.zeros(list(1 + np.array(xy_grid[-1])) + [2])
        for (i, j), (x_sh, y_sh) in zip(xy_grid, total_shifts):
            shifts_ncorre[i, j, 0] = y_sh
            shifts_ncorre[i, j, 1] = x_sh
        shifts_ncorre_masked = np.dstack([cv2.resize(
            shifts_ncorre[:, :, 0], new_templ.shape) * array, cv2.resize(shifts_ncorre[:, :, 1], new_templ.shape) * array])
#        shifts_ncorre_masked= np.dstack([cv2.resize(sh_efty[0],new_templ.shape)*array  ,cv2.resize(sh_efty[1],new_templ.shape)*array ]).T

        cv2.imshow('frame', np.concatenate([img_flow, dispOpticalFlow(np.dstack(
            [mov_dist['newim'], mov_dist['newim'], mov_dist['newim']]), 10 * (shifts_ncorre_masked - mov_dist['gt_shifts_masked']), Divisor=20)], axis=0))
        cv2.waitKey(1)

        correct_img[np.isnan(correct_img)] = np.nanmean(correct_img)
        shifts_oflow_res = caiman.motion_correction.compute_flow_single_frame(
            correct_img * 255, mov_dist['template'] * 255)
        shifts_oflow_res_masked = np.dstack(
            [shifts_oflow_res[:, :, 0] * array, shifts_oflow_res[:, :, 1] * array])

        mov_dist['shifts_ncorre'] = shifts_ncorre
        mov_dist['shifts_ncorre_masked'] = shifts_ncorre_masked
        mov_dist['norm_diff_shifts_ncorre'] = np.linalg.norm(
            shifts_ncorre_masked - mov_dist['gt_shifts_masked']) / np.linalg.norm(mov_dist['gt_shifts_masked'])

        mov_dist['shifts_oflow_res'] = shifts_oflow_res
        mov_dist['shifts_oflow_res_masked'] = shifts_oflow_res_masked

        mov_dist['gt_resid_shifts_masked'] = shifts_ncorre_masked - \
            mov_dist['gt_shifts_masked']
        mov_dist['norm_diff_shifts_oflow_res'] = np.linalg.norm(
            shifts_oflow_res_masked - mov_dist['gt_resid_shifts_masked']) / np.linalg.norm(mov_dist['gt_shifts_masked'])

        print(mov_dist['norm_diff_shifts_ncorre'])
        print(mov_dist['norm_diff_shifts_oflow_res'])


#%% FROM EP
#import h5py
# with h5py.File('EP_73.mat','r') as fl:
#    print(fl.keys())
#    all_movs_tosave = fl['all_movs_tosave']
##    all_movs_dist = fl['all_movs_tosave']
##    new_templ = fl['template']
##    mask = fl['mask']

#%%
# cv2.destroyAllWindows()
# for mov_dist in all_movs_dist:
#    print(mov_dist['patch_size'],mov_dist['noise_sigma'],mov_dist['iter_'])
#    newim = mov_dist['newim']
#    new_templ = mov_dist['template']
#    patch_size = mov_dist['patch_size']
#    mov_dist['shifts_ncorre'] = shifts_ncorre
#    mov_dist['shifts_ncorre_masked'] = shifts_ncorre_masked
    #%%
#    cv2.destroyAllWindows()
#    for mov_dist in all_movs_dist:
#        print(mov_dist['patch_size'],mov_dist['noise_sigma'],mov_dist['iter_'])
#        newim = mov_dist['newim']
#        new_templ = mov_dist['template']
#        patch_size = mov_dist['patch_size']
#
#        patch_size = mov_dist['patch_size']
##    sequences = [sima.Sequence.create('ndarray',np.repeat(np.concatenate([new_templ[None,None,:,:,None],newim[None,None,:,:,None]],axis = 0),10,axis = 0))]
#
#        sequences = [sima.Sequence.create('ndarray',np.concatenate([newim[None,None,:,:,None],new_templ[None,None,:,:,None],newim[None,None,:,:,None],new_templ[None,None,:,:,None],newim[None,None,:,:,None],
#                                                                new_templ[None,None,:,:,None],newim[None,None,:,:,None],new_templ[None,None,:,:,None],newim[None,None,:,:,None],new_templ[None,None,:,:,None],
#                                                                newim[None,None,:,:,None],
#                                                                new_templ[None,None,:,:,None],newim[None,None,:,:,None],new_templ[None,None,:,:,None],newim[None,None,:,:,None],
#                                                                new_templ[None,None,:,:,None],newim[None,None,:,:,None],new_templ[None,None,:,:,None],newim[None,None,:,:,None],
#                                                                new_templ[None,None,:,:,None],newim[None,None,:,:,None]],axis = 0))]
#
#
#
#        correct_img, total_shifts ,start_step, xy_grid  = caiman.motion_correction.tile_and_correct(newim ,new_templ, (patch_size,patch_size), (48,48),(max_shift,max_shift), newoverlaps = None, newstrides = None, upsample_factor_grid=4,\
#                                upsample_factor_fft=10,show_movie=False,max_deviation_rigid=max_shift,add_to_movie=0, shifts_opencv = True)
#        x_gr,y_gr = list(1+np.array(xy_grid[-1]))
#        shifts_ncorre = np.zeros(list(1+np.array(xy_grid[-1]))+[2])
#        for (i,j),(x_sh,y_sh) in zip(xy_grid,total_shifts):
#            shifts_ncorre[i,j,0] = y_sh
#            shifts_ncorre[i,j,1] = x_sh
#        shifts_ncorre_masked = np.dstack([cv2.resize(shifts_ncorre[:,:,0],new_templ.shape)*array  ,cv2.resize(shifts_ncorre[:,:,1],new_templ.shape)*array ])
#
#        mov_dist['shifts_ncorre'] = shifts_ncorre
#        mov_dist['shifts_ncorre_masked'] = shifts_ncorre_masked
#        mov_dist['norm_diff_shifts_ncorre'] = np.linalg.norm(shifts_ncorre_masked-mov_dist['gt_shifts_masked'])/np.linalg.norm(mov_dist['gt_shifts_masked'])
#        print(mov_dist['norm_diff_shifts_ncorre'] )
#%%
        np.savez('/mnt/ceph/neuro/DataForPublications/Piecewise-Rigid-Analysis-paper/Simulations/res_rotation_total.npz',
                 all_movs_dist=all_movs_dist, max_shift=max_shift)
#%%
with np.load('/mnt/ceph/neuro/DataForPublications/Piecewise-Rigid-Analysis-paper/Simulations/res_rotation_total.npz') as ld:
    all_movs_dist = ld['all_movs_dist']
    new_templ = all_movs_dist[0]['template']
    max_shift = ld['max_shift']
#%%
cv2.destroyAllWindows()
for mov_dist in all_movs_dist:
    print(mov_dist['patch_size'], mov_dist['noise_sigma'], mov_dist['iter_'])
    mov_dist['norm_shifts_oflow_res_masked'] = np.linalg.norm(
        mov_dist['shifts_oflow_res_masked'])
#%%
pl.rcParams['pdf.fonttype'] = 42

pl.subplot(2, 3, 1)
pl.imshow(imgs_with_flow[0], cmap='gray')
pl.axis('off')
pl.subplot(2, 3, 4)
pl.imshow(all_movs_dist[0]['newim'] - new_templ, cmap='viridis', vmax=1)
pl.axis('off')
aa = pl.subplot(2, 3, 2)
pl.imshow(all_movs_dist[10]['newim'], cmap='gray', vmin=0, vmax=1)
pl.title(all_movs_dist[10]['noise_sigma'])
pl.axis('off')
aa = pl.subplot(2, 3, 5)

pl.imshow(all_movs_dist[20]['newim'], cmap='gray', vmin=0, vmax=1)
pl.title(all_movs_dist[20]['noise_sigma'])
pl.axis('off')
#%
a = pl.subplot(2, 3, 3)

df = pandas.DataFrame()
for key in ['noise_sigma', 'patch_size', 'iter_', 'norm_diff_shifts_ncorre']:
    df[key] = [mmm[key] for mmm in all_movs_dist]
df.groupby(['patch_size', 'noise_sigma']).mean().unstack(level=0).plot(
    y=['norm_diff_shifts_ncorre'], yerr=df.groupby(['patch_size', 'noise_sigma']).sem().unstack(level=0), ax=a)
pl.xlabel('noise level')
pl.ylabel('Norm ratio delta/gt')
df = pandas.DataFrame()
for key in ['noise_sigma', 'patch_size', 'iter_', 'norm_diff_shifts_oflow']:
    df[key] = [mmm[key] for mmm in all_movs_dist]
df.groupby(['noise_sigma']).mean().unstack(level=0)['norm_diff_shifts_oflow'].plot(
    y=['norm_diff_shifts_oflow'], yerr=df.groupby(['noise_sigma']).sem().unstack(level=0)['norm_diff_shifts_oflow'], ax=a)
pl.xlim([-.1, 3.7])
# pl.ylim([.12,.5])
pl.legend(['24/48', '48/48', '96/48', '128/48'] + ['optical flow'])
aa = pl.subplot(2, 3, 6)
df = pandas.DataFrame()
for key in ['noise_sigma', 'patch_size', 'iter_', 'norm_diff_shifts_oflow_res']:
    df[key] = [mmm[key] for mmm in all_movs_dist]
df.groupby(['patch_size', 'noise_sigma']).mean().unstack(level=0).plot(y=['norm_diff_shifts_oflow_res'],
                                                                       yerr=df.groupby(['noise_sigma']).sem().unstack(level=0)['norm_diff_shifts_oflow_res'], ax=aa)
pl.xlabel('noise level')
pl.ylabel('Norm ratio residual/gt')
pl.legend(['24/48', '48/48', '96/48', '128/48'])
pl.xlim([-.1, 3.7])
#%%

# %%
# cv2.destroyAllWindows()
#new_templ_ = new_templ*255
#norm_diff_shifts_all_oc = []
# for i in range(10):
#    for noise_sigma in np.arange(0.01,4,.4):
#
#        all_shifts_mot_corr = []
#        rand_add = np.random.randn(*new_templ_.shape)*noise_sigma*new_templ_
#        gt_shifts = []
#        norm_diff_shifts = []
#        for vx,vy,frame, img_flow in zip(vx_s,vy_s,m, imgs_with_flow):
#
#            newim = apply_field(new_templ_,vx,vy)
#            newim += rand_add
#            shifts_mot_corr = caiman.motion_correction.compute_flow_single_frame(newim,new_templ_)
#            gt_shift_each = np.dstack([vx*array  ,vy*array ])
#        #    cv2.imshow('frame',newim/200)
#        #    cv2.imshow('frame',np.concatenate([correct_img[max_shift:-max_shift,max_shift:-max_shift]/3,new_templ_noise[max_shift:-max_shift,max_shift:-max_shift]/3],1))
#        #    cv2.imshow('frame',np.concatenate([vx,vy],1)/10)
# cv2.imshow('frame',shifts_mot_corr[:,:,1].astype(np.float32))
#            shifts_mot_corr = np.dstack([shifts_mot_corr[:,:,0]*array  ,shifts_mot_corr[:,:,1]*array ])
#            cv2.imshow('frame',np.concatenate([img_flow,dispOpticalFlow(np.dstack([newim,newim,newim])/255,shifts_mot_corr+gt_shift_each,Divisor=20)],axis=0))
#            cv2.waitKey(1)
#            all_shifts_mot_corr.append(shifts_mot_corr)
#            gt_shifts.append(gt_shift_each)
#            print('*')
#            norm_diff_shifts.append(np.linalg.norm(shifts_mot_corr+gt_shift_each)/np.linalg.norm(gt_shift_each))
#            print(norm_diff_shifts[-1])
#        norm_diff_shifts_all_oc.append(norm_diff_shifts)

#
#%%
import sima
granularity = 'row'
gran_n = 1
cv2.destroyAllWindows()
for mov_dist in all_movs_dist:
    print(mov_dist['patch_size'], mov_dist['noise_sigma'], mov_dist['iter_'])
    newim = mov_dist['newim']
    newim = new_templ.copy()
#    newim[:,:] = (np.concatenate([newim[-5:,:],newim[:-5,:]],axis = 0))
#    newim[:,:] = (np.concatenate([newim[:,-3:],newim[:,:-3]],axis = 1))
#    newim[::2,:] = (np.concatenate([newim[::2,5:],newim[::2,:5]],axis = 1))
#    newim[:,::2] = (np.concatenate([newim[5:,::2],newim[:4,::2]],axis = 0))
#    newim[:,50:100] = (np.concatenate([newim[3:,50:100],newim[:3,50:100]],axis = 0))

    newim[::2, :] = (np.concatenate(
        [newim[::2, -3:], newim[::2, :-3]], axis=1))
#    newim[:,::2] = (np.concatenate([newim[-3:,::2],newim[:-3,::2]],axis = 0))

#    newim[:,:200] = (np.concatenate([newim[-3:,:200],newim[:-3,:200]],axis = 0))

    new_templ = mov_dist['template']
    patch_size = mov_dist['patch_size']
#    sequences = [sima.Sequence.create('ndarray',np.repeat(np.concatenate([new_templ[None,None,:,:,None],newim[None,None,:,:,None]],axis = 0),10,axis = 0))]

    sequences = [sima.Sequence.create('ndarray', np.concatenate([newim[None, None, :, :, None], new_templ[None, None, :, :, None], newim[None, None, :, :, None], new_templ[None, None, :, :, None], newim[None, None, :, :, None],
                                                                 new_templ[None, None, :, :, None], newim[None, None, :, :, None], new_templ[None,
                                                                                                                                             None, :, :, None], newim[None, None, :, :, None], new_templ[None, None, :, :, None],
                                                                 newim[None, None,
                                                                       :, :, None],
                                                                 new_templ[None, None, :, :, None], newim[None, None, :, :,
                                                                                                          None], new_templ[None, None, :, :, None], newim[None, None, :, :, None],
                                                                 new_templ[None, None, :, :, None], newim[None, None, :, :,
                                                                                                          None], new_templ[None, None, :, :, None], newim[None, None, :, :, None],
                                                                 new_templ[None, None, :, :, None], newim[None, None, :, :, None]], axis=0))]
    dataset = sima.ImagingDataset(sequences, None)
    mc_approach = sima.motion.HiddenMarkov2D(granularity=(
        granularity, gran_n), max_displacement=[max_shift] * 2, verbose=True, n_processes=14)
    shifts_sima = mc_approach.estimate(dataset)
    vx = cv2.resize(shifts_sima[0][0][0][:, 0].astype(np.float32), newim.shape)
    vy = cv2.resize(shifts_sima[0][0][0][:, 1].astype(np.float32), newim.shape)
    pl.imshow(apply_field(newim, vx, vy))
#    mov_sima = mc_approach.correct(dataset,None,correction_channels = [1])
    print(np.sum(shifts_sima))
    break
#%%
pl.subplot(1, 2, 1)
pl.imshow(shifts_sima[0][:, 0, :, 0].squeeze(), aspect='auto')
pl.colorbar()
pl.subplot(1, 2, 2)
pl.imshow(shifts_sima[0][:, 0, :, 1].squeeze(), aspect='auto')
pl.colorbar()
# pl.subplot(2,2,3)
#pl.imshow(shifts_sima[0][-2:,0,:,0].squeeze(),aspect = 'auto')
# pl.colorbar()
# pl.subplot(2,2,4)
#pl.imshow(shifts_sima[0][-2:,0,:,1].squeeze(),aspect = 'auto')
# pl.colorbar()
#%%
cv2.destroyAllWindows()
for mov_dist in all_movs_dist:
    print(mov_dist['patch_size'], mov_dist['noise_sigma'], mov_dist['iter_'])
    newim = mov_dist['newim']
    new_templ = mov_dist['new_templ']
    patch_size = mov_dist['patch_size']

# %%
# cv2.destroyAllWindows()
#res_per_patch = dict()
#
# for patch_size in [32,64,96]:
#    norm_diff_shifts_all = []
#    for noise_sigma in np.arange(0.01,4,.4):
#        for i in range(2):
#
#            all_shifts_mot_corr = []
#            rand_add = np.random.randn(*new_templ.shape)*noise_sigma*new_templ
#            gt_shifts = []
#            norm_diff_shifts = []
#            for vx,vy,frame, img_flow in zip(vx_s,vy_s,m, imgs_with_flow):
#
#                newim = apply_field(new_templ,vx,vy)
#                newim += rand_add
#                correct_img, total_shifts ,start_step, xy_grid  = caiman.motion_correction.tile_and_correct(newim ,new_templ, (patch_size,patch_size), (48,48),(max_shift,max_shift), newoverlaps = None, newstrides = None, upsample_factor_grid=4,\
#                            upsample_factor_fft=10,show_movie=False,max_deviation_rigid=max_shift,add_to_movie=0, shifts_opencv = True)
#                x_gr,y_gr = list(1+np.array(xy_grid[-1]))
#
#                shifts_mot_corr = np.zeros(list(1+np.array(xy_grid[-1]))+[2])
#                for (i,j),(x_sh,y_sh) in zip(xy_grid,total_shifts):
#                    shifts_mot_corr[i,j,0] = y_sh
#                    shifts_mot_corr[i,j,1] = x_sh
#                gt_shift_each = np.dstack([vx*array  ,vy*array ])
#            #    cv2.imshow('frame',newim/200)
#            #    cv2.imshow('frame',np.concatenate([correct_img[max_shift:-max_shift,max_shift:-max_shift]/3,new_templ_noise[max_shift:-max_shift,max_shift:-max_shift]/3],1))
#            #    cv2.imshow('frame',np.concatenate([vx,vy],1)/10)
#            #    cv2.imshow('frame',img_flow)
#
#                shifts_mot_corr = np.dstack([cv2.resize(shifts_mot_corr[:,:,0],new_templ.shape)*array  ,cv2.resize(shifts_mot_corr[:,:,1],new_templ.shape)*array ])
#                cv2.imshow('frame',np.concatenate([img_flow,dispOpticalFlow(np.dstack([newim,newim,newim]),shifts_mot_corr-gt_shift_each,Divisor=20)],axis=0))
#                cv2.waitKey(1)
#                all_shifts_mot_corr.append(shifts_mot_corr)
#                gt_shifts.append(gt_shift_each)
#                print('*')
#                norm_diff_shifts.append(np.linalg.norm(shifts_mot_corr-gt_shift_each)/np.linalg.norm(gt_shift_each))
#                print(norm_diff_shifts[-1])
#            norm_diff_shifts_all.append(norm_diff_shifts)
#    res_per_patch[patch_size] =  norm_diff_shifts_all
#    #    print(np.linalg.norm(shifts_mot_corr[:,:,1]-gt_shift_each[:,:,1])/np.linalg.norm(gt_shift_each[:,:,1]))

    #    delta_img = correct_img[max_shift:-max_shift,max_shift:-max_shift]-new_templ_noise[max_shift:-max_shift,max_shift:-max_shift]
    #    print(np.linalg.norm(delta_img)/np.linalg.norm(new_templ_noise[max_shift:-max_shift,max_shift:-max_shift]))
