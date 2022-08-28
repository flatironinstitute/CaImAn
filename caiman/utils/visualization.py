#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" List of plotting functions to visualize what's happening in the code """

#\package Caiman/utils
#\version   1.0
#\copyright GNU General Public License v2.0
#\date Created on Tue Jun 30 21:01:17 2016
#\author: andrea giovannucci

from builtins import str
from builtins import range
from past.utils import old_div

import base64
import cv2
from IPython.display import HTML
from math import sqrt, ceil
import matplotlib as mpl
import matplotlib.cm as cm
from matplotlib.widgets import Slider
import numpy as np
import pylab as pl
from scipy.ndimage.measurements import center_of_mass
from scipy.ndimage.filters import median_filter
from scipy.sparse import issparse, spdiags, coo_matrix, csc_matrix
from skimage.measure import find_contours
import sys
from tempfile import NamedTemporaryFile
from typing import Dict
from warnings import warn
import functools as fct

from ..base.rois import com
from ..summary_images import local_correlations

try:
    cv2.setNumThreads(0)
except:
    pass


#%%
def view_patches(Yr, A, C, b, f, d1, d2, YrA=None, secs=1):
    """view spatial and temporal components (secs=0 interactive)

     Args:
         Yr:        np.ndarray
                movie in format pixels (d) x frames (T)

         A:     sparse matrix
                    matrix of spatial components (d x K)

         C:     np.ndarray
                    matrix of temporal components (K x T)

         b:     np.ndarray
                    spatial background (vector of length d)

         f:     np.ndarray
                    temporal background (vector of length T)

         d1,d2: np.ndarray
                    frame dimensions

         YrA:   np.ndarray
                     ROI filtered residual as it is given from update_temporal_components
                     If not given, then it is computed (K x T)

         secs:  float
                    number of seconds in between component scrolling. secs=0 means interactive (click to scroll)

         imgs:  np.ndarray
                    background image for contour plotting. Default is the image of all spatial components (d1 x d2)

    """

    pl.ion()
    nr, T = C.shape
    nb = f.shape[0]
    A2 = A.copy()
    A2.data **= 2
    nA2 = np.sqrt(np.array(A2.sum(axis=0))).squeeze()
    if YrA is None:
        Y_r = np.array(A.T * np.matrix(Yr) - (A.T * np.matrix(b[:, np.newaxis])) * np.matrix(
            f[np.newaxis]) - (A.T.dot(A)) * np.matrix(C) + C)
    else:
        Y_r = YrA + C

    A = A.todense()
    bkgrnd = np.reshape(b, (d1, d2) + (nb,), order='F')
    fig = pl.figure()
    thismanager = pl.get_current_fig_manager()
    thismanager.toolbar.pan()
    print('In order to scroll components you need to click on the plot')
    sys.stdout.flush()
    for i in range(nr + 1):
        if i < nr:
            ax1 = fig.add_subplot(2, 1, 1)
            pl.imshow(np.reshape(old_div(np.array(A[:, i]), nA2[i]),
                                 (d1, d2), order='F'), interpolation='None')
            ax1.set_title('Spatial component ' + str(i + 1))
            ax2 = fig.add_subplot(2, 1, 2)
            pl.plot(np.arange(T), np.squeeze(
                np.array(Y_r[i, :])), 'c', linewidth=3)
            pl.plot(np.arange(T), np.squeeze(
                np.array(C[i, :])), 'r', linewidth=2)
            ax2.set_title('Temporal component ' + str(i + 1))
            ax2.legend(labels=['Filtered raw data', 'Inferred trace'])

            if secs > 0:
                pl.pause(secs)
            else:
                pl.waitforbuttonpress()

            fig.delaxes(ax2)
        else:
            ax1 = fig.add_subplot(2, 1, 1)
            pl.imshow(bkgrnd[:, :, i - nr], interpolation='None')
            ax1.set_title('Spatial background ' + str(i - nr + 1))
            ax2 = fig.add_subplot(2, 1, 2)
            pl.plot(np.arange(T), np.squeeze(np.array(f[i - nr, :])))
            ax2.set_title('Temporal background ' + str(i - nr + 1))


def get_contours(A, dims, thr=0.9, thr_method='nrg', swap_dim=False):
    """Gets contour of spatial components and returns their coordinates

     Args:
         A:   np.ndarray or sparse matrix
                   Matrix of Spatial components (d x K)

             dims: tuple of ints
                   Spatial dimensions of movie (x, y[, z])

             thr: scalar between 0 and 1
                   Energy threshold for computing contours (default 0.9)

             thr_method: [optional] string
                  Method of thresholding:
                      'max' sets to zero pixels that have value less than a fraction of the max value
                      'nrg' keeps the pixels that contribute up to a specified fraction of the energy

     Returns:
         Coor: list of coordinates with center of mass and
                contour plot coordinates (per layer) for each component
    """

    if 'csc_matrix' not in str(type(A)):
        A = csc_matrix(A)
    d, nr = np.shape(A)
    # if we are on a 3D video
    if len(dims) == 3:
        d1, d2, d3 = dims
        x, y = np.mgrid[0:d2:1, 0:d3:1]
    else:
        d1, d2 = dims
        x, y = np.mgrid[0:d1:1, 0:d2:1]

    coordinates = []

    # get the center of mass of neurons( patches )
    cm = com(A, *dims)

    # for each patches
    for i in range(nr):
        pars:Dict = dict()
        # we compute the cumulative sum of the energy of the Ath component that has been ordered from least to highest
        patch_data = A.data[A.indptr[i]:A.indptr[i + 1]]
        indx = np.argsort(patch_data)[::-1]
        if thr_method == 'nrg':
            cumEn = np.cumsum(patch_data[indx]**2)
            if len(cumEn) == 0:
                pars = dict(
                    coordinates=np.array([]),
                    CoM=np.array([np.NaN, np.NaN]),
                    neuron_id=i + 1,
                )
                coordinates.append(pars)
                continue
            else:
                # we work with normalized values
                cumEn /= cumEn[-1]
                Bvec = np.ones(d)
                # we put it in a similar matrix
                Bvec[A.indices[A.indptr[i]:A.indptr[i + 1]][indx]] = cumEn
        else:
            if thr_method != 'max':
                warn("Unknown threshold method. Choosing max")
            Bvec = np.zeros(d)
            Bvec[A.indices[A.indptr[i]:A.indptr[i + 1]]] = patch_data / patch_data.max()

        if swap_dim:
            Bmat = np.reshape(Bvec, dims, order='C')
        else:
            Bmat = np.reshape(Bvec, dims, order='F')
        pars['coordinates'] = []
        # for each dimensions we draw the contour
        for B in (Bmat if len(dims) == 3 else [Bmat]):
            vertices = find_contours(B.T, thr)
            # this fix is necessary for having disjoint figures and borders plotted correctly
            v = np.atleast_2d([np.nan, np.nan])
            for _, vtx in enumerate(vertices):
                num_close_coords = np.sum(np.isclose(vtx[0, :], vtx[-1, :]))
                if num_close_coords < 2:
                    if num_close_coords == 0:
                        # case angle
                        newpt = np.round(old_div(vtx[-1, :], [d2, d1])) * [d2, d1]
                        vtx = np.concatenate((vtx, newpt[np.newaxis, :]), axis=0)
                    else:
                        # case one is border
                        vtx = np.concatenate((vtx, vtx[0, np.newaxis]), axis=0)
                v = np.concatenate(
                    (v, vtx, np.atleast_2d([np.nan, np.nan])), axis=0)

            pars['coordinates'] = v if len(
                dims) == 2 else (pars['coordinates'] + [v])
        pars['CoM'] = np.squeeze(cm[i, :])
        pars['neuron_id'] = i + 1
        coordinates.append(pars)
    return coordinates


def matrixMontage(spcomps, *args, **kwargs):
    numcomps, _, _ = spcomps.shape
    rowcols = int(np.ceil(np.sqrt(numcomps)))
    for k, comp in enumerate(spcomps):
        pl.subplot(rowcols, rowcols, k + 1)
        pl.imshow(comp, *args, **kwargs)
        pl.axis('off')


VIDEO_TAG = """<video controls>
 <source src="data:video/x-m4v;base64,{0}" type="video/mp4">
 Your browser does not support the video tag.
</video>"""


def anim_to_html(anim, fps=20):
    # todo: todocument
    if not hasattr(anim, '_encoded_video'):
        with NamedTemporaryFile(suffix='.mp4') as f:
            anim.save(f.name, fps=fps, extra_args=['-vcodec', 'libx264'])
            video = open(f.name, "rb").read()
        anim._encoded_video = base64.b64encode(video)

    return VIDEO_TAG.format(anim._encoded_video.decode('ascii'))

#%%


def display_animation(anim, fps=20):
    pl.close(anim._fig)
    return HTML(anim_to_html(anim, fps=fps))
#%%


def view_patches_bar(Yr, A, C, b, f, d1, d2, YrA=None, img=None,
                     r_values=None, SNR=None, cnn_preds=None):
    """view spatial and temporal components interactively

    Args:
        Yr:    np.ndarray
            movie in format pixels (d) x frames (T)

        A:     sparse matrix
                matrix of spatial components (d x K)

        C:     np.ndarray
                matrix of temporal components (K x T)

        b:     np.ndarray
                spatial background (vector of length d)

        f:     np.ndarray
                temporal background (vector of length T)

        d1,d2: np.ndarray
                frame dimensions

        YrA:   np.ndarray
                    ROI filtered residual as it is given from update_temporal_components
                    If not given, then it is computed (K x T)

        img:   np.ndarray
                background image for contour plotting. Default is the image of all spatial components (d1 x d2)

        r_values: np.ndarray
            space correlation values

        SNR: np.ndarray
            peak-SNR over the length of a transient for each component

        cnn_preds: np.ndarray
            prediction values from the CNN classifier
    """

    pl.ion()
    if 'csc_matrix' not in str(type(A)):
        A = csc_matrix(A)

    nr, T = C.shape
    nb = 0 if f is None else f.shape[0]
    nA2 = np.sqrt(np.array(A.power(2).sum(axis=0))).squeeze()

    if YrA is None:
        Y_r = spdiags(old_div(1, nA2), 0, nr, nr) * (A.T.dot(Yr) -
                                                     (A.T.dot(b)).dot(f) - (A.T.dot(A)).dot(C)) + C
    else:
        Y_r = YrA + C

    if img is None:
        img = np.reshape(np.array(A.mean(axis=1)), (d1, d2), order='F')

    fig = pl.figure(figsize=(10, 10))

    axcomp = pl.axes([0.05, 0.05, 0.9, 0.03])

    ax1 = pl.axes([0.05, 0.55, 0.4, 0.4])
    ax3 = pl.axes([0.55, 0.55, 0.4, 0.4])
    ax2 = pl.axes([0.05, 0.1, 0.9, 0.4])

    s_comp = Slider(axcomp, 'Component', 0, nr + nb - 1, valinit=0)
    vmax = np.percentile(img, 95)

    def update(val):
        i = np.int(np.round(s_comp.val))
        print(('Component:' + str(i)))

        if i < nr:

            ax1.cla()
            imgtmp = np.reshape(A[:, i].toarray(), (d1, d2), order='F')
            ax1.imshow(imgtmp, interpolation='None', cmap=pl.cm.gray, vmax=np.max(imgtmp)*0.5)
            ax1.set_title('Spatial component ' + str(i + 1))
            ax1.axis('off')

            ax2.cla()
            ax2.plot(np.arange(T), Y_r[i], 'c', linewidth=3)
            ax2.plot(np.arange(T), C[i], 'r', linewidth=2)
            ax2.set_title('Temporal component ' + str(i + 1))
            ax2.legend(labels=['Filtered raw data', 'Inferred trace'], loc=1)
            if r_values is not None:
                metrics = 'Evaluation Metrics\nSpatial corr:% 7.3f\nSNR:% 18.3f\nCNN:' % (
                    r_values[i], SNR[i])
                metrics += ' '*15+'N/A' if np.sum(cnn_preds) in (0, None) else '% 18.3f' % cnn_preds[i]
                ax2.text(0.02, 0.97, metrics, ha='left', va='top', transform=ax2.transAxes,
                        bbox=dict(edgecolor='k', facecolor='w', alpha=.5))

            ax3.cla()
            ax3.imshow(img, interpolation='None', cmap=pl.cm.gray, vmax=vmax)
            imgtmp2 = imgtmp.copy()
            imgtmp2[imgtmp2 == 0] = np.nan
            ax3.imshow(imgtmp2, interpolation='None',
                       alpha=0.5, cmap=pl.cm.hot)
            ax3.axis('off')
        else:
            ax1.cla()
            bkgrnd = np.reshape(b[:, i - nr], (d1, d2), order='F')
            ax1.imshow(bkgrnd, interpolation='None')
            ax1.set_title('Spatial background ' + str(i + 1 - nr))
            ax1.axis('off')

            ax2.cla()
            ax2.plot(np.arange(T), np.squeeze(np.array(f[i - nr, :])))
            ax2.set_title('Temporal background ' + str(i + 1 - nr))

    def arrow_key_image_control(event):

        if event.key == 'left':
            new_val = np.round(s_comp.val - 1)
            if new_val < 0:
                new_val = 0
            s_comp.set_val(new_val)

        elif event.key == 'right':
            new_val = np.round(s_comp.val + 1)
            if new_val > nr + nb:
                new_val = nr + nb
            s_comp.set_val(new_val)
        else:
            pass

    s_comp.on_changed(update)
    s_comp.set_val(0)
    fig.canvas.mpl_connect('key_release_event', arrow_key_image_control)
    pl.show()
#%%


def plot_contours(A, Cn, thr=None, thr_method='max', maxthr=0.2, nrgthr=0.9, display_numbers=True, max_number=None,
                  cmap=None, swap_dim=False, colors='w', vmin=None, vmax=None, coordinates=None,
                  contour_args={}, number_args={}, **kwargs):
    """Plots contour of spatial components against a background image and returns their coordinates

     Args:
         A:   np.ndarray or sparse matrix
                   Matrix of Spatial components (d x K)
    
         Cn:  np.ndarray (2D)
                   Background image (e.g. mean, correlation)
    
         thr_method: [optional] string
                  Method of thresholding:
                      'max' sets to zero pixels that have value less than a fraction of the max value
                      'nrg' keeps the pixels that contribute up to a specified fraction of the energy
    
         maxthr: [optional] scalar
                    Threshold of max value
    
         nrgthr: [optional] scalar
                    Threshold of energy
    
         thr: scalar between 0 and 1
                   Energy threshold for computing contours (default 0.9)
                   Kept for backwards compatibility. If not None then thr_method = 'nrg', and nrgthr = thr
    
         display_number:     Boolean
                   Display number of ROIs if checked (default True)
    
         max_number:    int
                   Display the number for only the first max_number components (default None, display all numbers)
    
         cmap:     string
                   User specifies the colormap (default None, default colormap)

     Returns:
          coordinates: list of coordinates with center of mass, contour plot coordinates and bounding box for each component
    """

    if swap_dim:
        Cn = Cn.T
        print('Swapping dim')

    if thr is None:
        try:
            thr = {'nrg': nrgthr, 'max': maxthr}[thr_method]
        except KeyError:
            thr = maxthr
    else:
        thr_method = 'nrg'


    for key in ['c', 'colors', 'line_color']:
        if key in kwargs.keys():
            color = kwargs[key]
            kwargs.pop(key)

    ax = pl.gca()
    if vmax is None and vmin is None:
        pl.imshow(Cn, interpolation=None, cmap=cmap,
                  vmin=np.percentile(Cn[~np.isnan(Cn)], 1),
                  vmax=np.percentile(Cn[~np.isnan(Cn)], 99))
    else:
        pl.imshow(Cn, interpolation=None, cmap=cmap, vmin=vmin, vmax=vmax)

    if coordinates is None:
        coordinates = get_contours(A, np.shape(Cn), thr, thr_method, swap_dim)
    for c in coordinates:
        v = c['coordinates']
        c['bbox'] = [np.floor(np.nanmin(v[:, 1])), np.ceil(np.nanmax(v[:, 1])),
                     np.floor(np.nanmin(v[:, 0])), np.ceil(np.nanmax(v[:, 0]))]
        pl.plot(*v.T, c=colors, **contour_args)

    if display_numbers:
        d1, d2 = np.shape(Cn)
        d, nr = np.shape(A)
        cm = com(A, d1, d2)
        if max_number is None:
            max_number = A.shape[1]
        for i in range(np.minimum(nr, max_number)):
            if swap_dim:
                ax.text(cm[i, 0], cm[i, 1], str(i + 1), color=colors, **number_args)
            else:
                ax.text(cm[i, 1], cm[i, 0], str(i + 1), color=colors, **number_args)
    return coordinates


def plot_shapes(Ab, dims, num_comps=15, size=(15, 15), comps_per_row=None,
                cmap='viridis', smoother=lambda s: median_filter(s, 3)):

    def GetBox(centers, R, dims):
        D = len(R)
        box = np.zeros((D, 2), dtype=int)
        for dd in range(D):
            box[dd, 0] = max((centers[dd] - R[dd], 0))
            box[dd, 1] = min((centers[dd] + R[dd] + 1, dims[dd]))
        return box

    nx = int(sqrt(num_comps) * 1.3) if comps_per_row is None else comps_per_row
    ny = int(ceil(num_comps / float(nx)))
    pl.figure(figsize=(nx, ny))
    for i, a in enumerate(Ab.T[:num_comps]):
        ax = pl.subplot(ny, nx, i + 1)
        s = a.toarray().reshape(dims, order='F')
        box = GetBox(np.array(center_of_mass(s), dtype=np.int16), size, dims)
        pl.imshow(smoother(s[list(map(lambda a: slice(*a), box))]),
                  cmap=cmap, interpolation='nearest')
        ax.axis('off')
    pl.subplots_adjust(0, 0, 1, 1, .06, .06)


def inspect_correlation_pnr(correlation_image_pnr, pnr_image):
    """
    inspect correlation and pnr images to infer the min_corr, min_pnr

    Args:
        correlation_image_pnr: ndarray
            correlation image created with caiman.summary_images.correlation_pnr
    
        pnr_image: ndarray
            peak-to-noise image created with caiman.summary_images.correlation_pnr
    """

    fig = pl.figure(figsize=(10, 4))
    pl.axes([0.05, 0.2, 0.4, 0.7])
    im_cn = pl.imshow(correlation_image_pnr, cmap='jet')
    pl.title('correlation image')
    pl.colorbar()
    pl.axes([0.5, 0.2, 0.4, 0.7])
    im_pnr = pl.imshow(pnr_image, cmap='jet')
    pl.title('PNR')
    pl.colorbar()

    s_cn_max = Slider(pl.axes([0.05, 0.01, 0.35, 0.03]), 'vmax',
                      correlation_image_pnr.min(), correlation_image_pnr.max(), valinit=correlation_image_pnr.max())
    s_cn_min = Slider(pl.axes([0.05, 0.07, 0.35, 0.03]), 'vmin',
                      correlation_image_pnr.min(), correlation_image_pnr.max(), valinit=correlation_image_pnr.min())
    s_pnr_max = Slider(pl.axes([0.5, 0.01, 0.35, 0.03]), 'vmax',
                       pnr_image.min(), pnr_image.max(), valinit=pnr_image.max())
    s_pnr_min = Slider(pl.axes([0.5, 0.07, 0.35, 0.03]), 'vmin',
                       pnr_image.min(), pnr_image.max(), valinit=pnr_image.min())

    def update(val):
        im_cn.set_clim([s_cn_min.val, s_cn_max.val])
        im_pnr.set_clim([s_pnr_min.val, s_pnr_max.val])
        fig.canvas.draw_idle()

    s_cn_max.on_changed(update)
    s_cn_min.on_changed(update)
    s_pnr_max.on_changed(update)
    s_pnr_min.on_changed(update)
