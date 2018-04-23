#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" List of plotting functions to visualize what's happening in the code """

#\package Caiman/utils
#\version   1.0
#\copyright GNU General Public License v2.0
#\date Created on Tue Jun 30 21:01:17 2016
#\author: andrea giovannucci
from __future__ import division
from __future__ import print_function
from builtins import str
from builtins import range
from past.utils import old_div
import base64
import cv2
import numpy as np
import pylab as pl
from tempfile import NamedTemporaryFile
from IPython.display import HTML
import sys
from warnings import warn
from scipy.sparse import issparse, spdiags, coo_matrix, csc_matrix
from matplotlib.widgets import Slider
from ..base.rois import com
from scipy.ndimage.measurements import center_of_mass
from scipy.ndimage.filters import median_filter
import matplotlib.cm as cm
import matplotlib as mpl
from math import sqrt, ceil
try:
    import bokeh
    import bokeh.plotting as bpl
    from bokeh.models import CustomJS, ColumnDataSource, Range1d
except:
    print("Bokeh could not be loaded. Either it is not installed or you are not running within a notebook")

from ..summary_images import local_correlations
from skimage.measure import find_contours


#%%
def view_patches(Yr, A, C, b, f, d1, d2, YrA=None, secs=1):
    """view spatial and temporal components (secs=0 interactive)

     Parameters:
     -----------
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

    See Also:
    ------------
    ..image:: doc/img/

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


def nb_view_patches(Yr, A, C, b, f, d1, d2, YrA=None, image_neurons=None, thr=0.99, denoised_color=None, cmap='jet'):
    """
    Interactive plotting utility for ipython notebook

    Parameters:
    -----------
    Yr: np.ndarray
        movie

    A,C,b,f: np.ndarrays
        outputs of matrix factorization algorithm

    d1,d2: floats
        dimensions of movie (x and y)

    YrA:   np.ndarray
        ROI filtered residual as it is given from update_temporal_components
        If not given, then it is computed (K x T)

    image_neurons: np.ndarray
        image to be overlaid to neurons (for instance the average)

    thr: double
        threshold regulating the extent of the displayed patches

    denoised_color: string or None
        color name (e.g. 'red') or hex color code (e.g. '#F0027F')

    cmap: string
        name of colormap (e.g. 'viridis') used to plot image_neurons
    """
    colormap = cm.get_cmap(cmap)
    grayp = [mpl.colors.rgb2hex(m) for m in colormap(np.arange(colormap.N))]
    nr, T = C.shape
    nA2 = np.ravel(np.power(A, 2).sum(0)) if type(
        A) == np.ndarray else np.ravel(A.power(2).sum(0))
    b = np.squeeze(b)
    f = np.squeeze(f)
    if YrA is None:
        Y_r = np.array(spdiags(old_div(1, nA2), 0, nr, nr) *
                       (A.T * np.matrix(Yr) -
                        (A.T * np.matrix(b[:, np.newaxis])) * np.matrix(f[np.newaxis]) -
                        A.T.dot(A) * np.matrix(C)) + C)
    else:
        Y_r = C + YrA

    x = np.arange(T)
    if image_neurons is None:
        image_neurons = A.mean(1).reshape((d1, d2), order='F')

    coors = get_contours(A, (d1, d2), thr)
    cc1 = [cor['coordinates'][:, 0] for cor in coors]
    cc2 = [cor['coordinates'][:, 1] for cor in coors]
    c1 = cc1[0]
    c2 = cc2[0]

    # split sources up, such that Bokeh does not warn
    # "ColumnDataSource's columns must be of the same length"
    source = ColumnDataSource(data=dict(x=x, y=Y_r[0] / 100, y2=C[0] / 100))
    source_ = ColumnDataSource(data=dict(z=Y_r / 100, z2=C / 100))
    source2 = ColumnDataSource(data=dict(c1=c1, c2=c2))
    source2_ = ColumnDataSource(data=dict(cc1=cc1, cc2=cc2))

    callback = CustomJS(args=dict(source=source, source_=source_, source2=source2, source2_=source2_), code="""
            var data = source.data
            var data_ = source_.data
            var f = cb_obj.value - 1
            x = data['x']
            y = data['y']
            y2 = data['y2']

            for (i = 0; i < x.length; i++) {
                y[i] = data_['z'][i+f*x.length]
                y2[i] = data_['z2'][i+f*x.length]
            }

            var data2_ = source2_.data;
            var data2 = source2.data;
            c1 = data2['c1'];
            c2 = data2['c2'];
            cc1 = data2_['cc1'];
            cc2 = data2_['cc2'];

            for (i = 0; i < c1.length; i++) {
                   c1[i] = cc1[f][i]
                   c2[i] = cc2[f][i]
            }
            source2.change.emit();
            source.change.emit();
        """)

    plot = bpl.figure(plot_width=600, plot_height=300)
    plot.line('x', 'y', source=source, line_width=1, line_alpha=0.6)
    if denoised_color is not None:
        plot.line('x', 'y2', source=source, line_width=1,
                  line_alpha=0.6, color=denoised_color)

    slider = bokeh.models.Slider(start=1, end=Y_r.shape[0], value=1, step=1,
                                 title="Neuron Number", callback=callback)
    xr = Range1d(start=0, end=image_neurons.shape[1])
    yr = Range1d(start=image_neurons.shape[0], end=0)
    plot1 = bpl.figure(x_range=xr, y_range=yr, plot_width=300, plot_height=300)

    plot1.image(image=[image_neurons[::-1, :]], x=0,
                y=image_neurons.shape[0], dw=d2, dh=d1, palette=grayp)
    plot1.patch('c1', 'c2', alpha=0.6, color='purple',
                line_width=2, source=source2)

    if Y_r.shape[0] > 1:
        bpl.show(bokeh.layouts.layout([[slider], [bokeh.layouts.row(plot1, plot)]]))
    else:
        bpl.show(bokeh.layouts.row(plot1, plot))

    return Y_r


def get_contours(A, dims, thr=0.9, thr_method='nrg', swap_dim=False):
    """Gets contour of spatial components and returns their coordinates

     Parameters:
     -----------
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
     --------
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
        pars = dict()
        # we compute the cumulative sum of the energy of the Ath component that has been ordered from least to highest
        patch_data = A.data[A.indptr[i]:A.indptr[i + 1]]
        indx = np.argsort(patch_data)[::-1]
        if thr_method == 'nrg':
            cumEn = np.cumsum(patch_data[indx]**2)
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


def nb_view_patches3d(Y_r, A, C, dims, image_type='mean', Yr=None,
                      max_projection=False, axis=0, thr=0.9, denoised_color=None, cmap='jet'):
    """
    Interactive plotting utility for ipython notbook

    Parameters:
    -----------
    Y_r: np.ndarray
        residual of each trace

    A,C,b,f: np.ndarrays
        outputs of matrix factorization algorithm

    dims: tuple of ints
        dimensions of movie (x, y and z)

    image_type: 'mean', 'max' or 'corr'
        image to be overlaid to neurons
        (average of shapes, maximum of shapes or nearest neigbor correlation of raw data)

    Yr: np.ndarray
        movie, only required if image_type=='corr' to calculate correlation image

    max_projection: boolean
        plot max projection along specified axis if True, plot layers if False

    axis: int (0, 1 or 2)
        axis along which max projection is performed or layers are shown

    thr: scalar between 0 and 1
        Energy threshold for computing contours

    denoised_color: string or None
        color name (e.g. 'red') or hex color code (e.g. '#F0027F')

    cmap: string
        name of colormap (e.g. 'viridis') used to plot image_neurons

    Raise:
    ------
    ValueError("image_type must be 'mean', 'max' or 'corr'")

    """

    bokeh.io.curdoc().clear()  # prune old orphaned models, otherwise filesize blows up
    d = A.shape[0]
    order = list(range(4))
    order.insert(0, order.pop(axis))
    Y_r = Y_r + C
    index_permut = np.reshape(np.arange(d), dims, order='F').transpose(
        order[:-1]).reshape(d, order='F')
    A = csc_matrix(A)[index_permut, :]
    dims = tuple(np.array(dims)[order[:3]])
    d1, d2, d3 = dims
    colormap = cm.get_cmap(cmap)
    grayp = [mpl.colors.rgb2hex(m) for m in colormap(np.arange(colormap.N))]
    nr, T = C.shape

    x = np.arange(T)

    source = ColumnDataSource(data=dict(x=x, y=Y_r[0] / 100, y2=C[0] / 100))
    source_ = ColumnDataSource(data=dict(z=Y_r / 100, z2=C / 100))
    sourceN = ColumnDataSource(data=dict(N=[nr], nan=np.array([np.nan])))

    if max_projection:
        if image_type == 'corr':
            tmp = [(local_correlations(
                Yr.reshape(dims + (-1,), order='F'))[:, ::-1]).max(i)
                for i in range(3)]

        elif image_type == 'mean':
            tmp = [(np.array(A.mean(axis=1)).reshape(dims, order='F')[:, ::-1]).max(i)
                   for i in range(3)]

        elif image_type == 'max':
            tmp = [(A.max(axis=1).toarray().reshape(dims, order='F')[:, ::-1]).max(i)
                   for i in range(3)]

        else:
            raise ValueError("image_type must be 'mean', 'max' or 'corr'")

        image_neurons = np.nan * \
            np.ones((int(1.05 * (d1 + d2)), int(1.05 * (d1 + d3))))
        image_neurons[:d2, -d3:] = tmp[0][::-1]
        image_neurons[:d2, :d1] = tmp[2].T[::-1]
        image_neurons[-d1:, -d3:] = tmp[1]
        offset1 = image_neurons.shape[1] - d3
        offset2 = image_neurons.shape[0] - d1

        proj_ = [coo_matrix([A[:, nnrr].toarray().reshape(dims, order='F').max(
            i).reshape(-1, order='F') for nnrr in range(A.shape[1])]) for i in range(3)]
        proj_ = [pproj_.T for pproj_ in proj_]

        coors = [get_contours(proj_[i], tmp[i].shape, thr=thr)
                 for i in range(3)]

        pl.close()
        K = np.max([[len(cor['coordinates']) for cor in cc] for cc in coors])
        cc1 = np.nan * np.zeros(np.shape(coors) + (K,))
        cc2 = np.nan * np.zeros(np.shape(coors) + (K,))
        for i, cor in enumerate(coors[0]):
            cc1[0, i, :len(cor['coordinates'])
                ] = cor['coordinates'][:, 0] + offset1
            cc2[0, i, :len(cor['coordinates'])] = cor['coordinates'][:, 1]
        for i, cor in enumerate(coors[2]):
            cc1[1, i, :len(cor['coordinates'])] = cor['coordinates'][:, 1]
            cc2[1, i, :len(cor['coordinates'])] = cor['coordinates'][:, 0]
        for i, cor in enumerate(coors[1]):
            cc1[2, i, :len(cor['coordinates'])
                ] = cor['coordinates'][:, 0] + offset1
            cc2[2, i, :len(cor['coordinates'])
                ] = cor['coordinates'][:, 1] + offset2

        c1x = cc1[0][0]
        c2x = cc2[0][0]
        c1y = cc1[1][0]
        c2y = cc2[1][0]
        c1z = cc1[2][0]
        c2z = cc2[2][0]
        source2_ = ColumnDataSource(data=dict(cc1=cc1, cc2=cc2))
        source2 = ColumnDataSource(data=dict(c1x=c1x, c1y=c1y, c1z=c1z,
                                             c2x=c2x, c2y=c2y, c2z=c2z))
        callback = CustomJS(args=dict(source=source, source_=source_, sourceN=sourceN,
                                      source2=source2, source2_=source2_), code="""
                var data = source.data;
                var data_ = source_.data;
                var f = cb_obj.value - 1
                x = data['x']
                y = data['y']
                y2 = data['y2']
                for (i = 0; i < x.length; i++) {
                    y[i] = data_['z'][i+f*x.length]
                    y2[i] = data_['z2'][i+f*x.length]
                }

                var data2_ = source2_.data;
                var data2 = source2.data;
                c1x = data2['c1x'];
                c2x = data2['c2x'];
                c1y = data2['c1y'];
                c2y = data2['c2y'];
                c1z = data2['c1z'];
                c2z = data2['c2z'];
                cc1 = data2_['cc1'];
                cc2 = data2_['cc2'];
                var N = sourceN.data['N'][0];
                for (i = 0; i < c1x.length; i++) {
                       c1x[i] = cc1[f*c1x.length + i]
                       c2x[i] = cc2[f*c1x.length + i]
                }
                for (i = 0; i < c1x.length; i++) {
                       c1y[i] = cc1[N*c1y.length + f*c1y.length + i]
                       c2y[i] = cc2[N*c1y.length + f*c1y.length + i]
                }
                for (i = 0; i < c1x.length; i++) {
                       c1z[i] = cc1[2*N*c1z.length + f*c1z.length + i]
                       c2z[i] = cc2[2*N*c1z.length + f*c1z.length + i]
                }
                source2.change.emit();
                source.change.emit();
            """)
    else:

        if image_type == 'corr':
            image_neurons = local_correlations(
                Yr.reshape(dims + (-1,), order='F'))[:-1, ::-1]

        elif image_type == 'mean':
            image_neurons = np.array(A.mean(axis=1)).reshape(
                dims, order='F')[:, ::-1]

        elif image_type == 'max':
            image_neurons = A.max(axis=1).toarray().reshape(
                dims, order='F')[:, ::-1]

        else:
            raise ValueError('image_type must be mean, max or corr')

        cmap = bokeh.models.mappers.LinearColorMapper([mpl.colors.rgb2hex(m)
                                                       for m in colormap(np.arange(colormap.N))])
        cmap.high = image_neurons.max()
        coors = get_contours(A, dims, thr=thr)
        pl.close()
        cc1 = [[(l[:, 0]) for l in n['coordinates']] for n in coors]
        cc2 = [[(l[:, 1]) for l in n['coordinates']] for n in coors]
        length = np.ravel([list(map(len, cc)) for cc in cc1])
        idx = np.cumsum(np.concatenate([[0], length[:-1]]))
        cc1 = np.concatenate(list(map(np.concatenate, cc1)))
        cc2 = np.concatenate(list(map(np.concatenate, cc2)))
        # pick initial layer in which first neuron lies
        linit = int(round(coors[0]['CoM'][0]))
        K = length.max()
        c1 = np.nan * np.zeros(K)
        c2 = np.nan * np.zeros(K)
        c1[:length[linit]] = cc1[idx[linit]:idx[linit] + length[linit]]
        c2[:length[linit]] = cc2[idx[linit]:idx[linit] + length[linit]]
        source2 = ColumnDataSource(data=dict(c1=c1, c2=c2))
        source2_ = ColumnDataSource(data=dict(cc1=cc1, cc2=cc2))
        source2_idx = ColumnDataSource(data=dict(idx=idx, length=length))
        source3 = ColumnDataSource(
            data=dict(image=[image_neurons[linit]], im=[image_neurons],
                      x=[0], y=[d2], dw=[d3], dh=[d2]))
        callback = CustomJS(args=dict(source=source, source_=source_, sourceN=sourceN,
                                      source2=source2, source2_=source2_, source2_idx=source2_idx),
                            code="""
                var data = source.data;
                var data_ = source_.data;
                var f = slider_neuron.value-1;
                var l = slider_layer.value-1;
                x = data['x']
                y = data['y']
                y2 = data['y2']
                for (i = 0; i < x.length; i++) {
                    y[i] = data_['z'][i+f*x.length]
                    y2[i] = data_['z2'][i+f*x.length]
                }

                var data2 = source2.data;
                var data2_ = source2_.data;
                var data2_idx = source2_idx.data;
                var idx = data2_idx['idx'];
                c1 = data2['c1'];
                c2 = data2['c2'];
                var nz = idx.length / sourceN.data['N'][0];
                var nan = sourceN.data['nan'][0];
                for (i = 0; i < c1.length; i++) {
                       c1[i] = nan;
                       c2[i] = nan;
                }
                for (i = 0; i < data2_idx['length'][l+f*nz]; i++) {
                       c1[i] = data2_['cc1'][idx[l+f*nz] + i];
                       c2[i] = data2_['cc2'][idx[l+f*nz] + i];
                }
                source2.change.emit();
                source.change.emit();
            """)

        callback_layer = CustomJS(args=dict(source=source3, sourceN=sourceN, source2=source2,
                                            source2_=source2_, source2_idx=source2_idx), code="""
                var f = slider_neuron.value-1;
                var l = slider_layer.value-1;
                var dh = source.data['dh'][0];
                var dw = source.data['dw'][0];
                var image = source.data['image'][0];
                var images = source.data['im'][0];
                for (var i = 0; i < x.length; i++) {
                    for (var j = 0; j < dw; j++){
                        image[i*dh+j] = images[l*dh*dw + i*dh + j];
                    }
                }

                var data2 = source2.data;
                var data2_ = source2_.data;
                var data2_idx = source2_idx.data;
                var idx = data2_idx['idx']
                c1 = data2['c1'];
                c2 = data2['c2'];
                var nz = idx.length / sourceN.data['N'][0];
                var nan = sourceN.data['nan'][0];
                for (i = 0; i < c1.length; i++) {
                       c1[i] = nan;
                       c2[i] = nan;
                }
                for (i = 0; i < data2_idx['length'][l+f*nz]; i++) {
                       c1[i] = data2_['cc1'][idx[l+f*nz] + i];
                       c2[i] = data2_['cc2'][idx[l+f*nz] + i];
                }
                source.change.emit()
                source2.change.emit();
            """)

    plot = bpl.figure(plot_width=600, plot_height=300)
    plot.line('x', 'y', source=source, line_width=1, line_alpha=0.6)
    if denoised_color is not None:
        plot.line('x', 'y2', source=source, line_width=1,
                  line_alpha=0.6, color=denoised_color)
    slider = bokeh.models.Slider(start=1, end=Y_r.shape[0], value=1, step=1,
                                 title="Neuron Number", callback=callback)
    xr = Range1d(start=0, end=image_neurons.shape[1] if max_projection else d3)
    yr = Range1d(start=image_neurons.shape[0] if max_projection else d2, end=0)
    plot1 = bpl.figure(x_range=xr, y_range=yr, plot_width=300, plot_height=300)

    if max_projection:
        plot1.image(image=[image_neurons[::-1, :]], x=0, y=image_neurons.shape[0],
                    dw=image_neurons.shape[1], dh=image_neurons.shape[0], palette=grayp)
        plot1.patch('c1x', 'c2x', alpha=0.6, color='purple',
                    line_width=2, source=source2)
        plot1.patch('c1y', 'c2y', alpha=0.6, color='purple',
                    line_width=2, source=source2)
        plot1.patch('c1z', 'c2z', alpha=0.6, color='purple',
                    line_width=2, source=source2)
        layout = bokeh.layouts.layout([[slider], [bokeh.layouts.row(plot1, plot)]],
                                      sizing_mode="scale_width")
    else:
        slider_layer = bokeh.models.Slider(start=1, end=d1, value=linit + 1, step=1,
                                           title="Layer", callback=callback_layer)
        callback.args['slider_neuron'] = slider
        callback.args['slider_layer'] = slider_layer
        callback_layer.args['slider_neuron'] = slider
        callback_layer.args['slider_layer'] = slider_layer
        plot1.image(image='image', x='x', y='y', dw='dw', dh='dh',
                    color_mapper=cmap, source=source3)
        plot1.patch('c1', 'c2', alpha=0.6, color='purple',
                    line_width=2, source=source2)
        layout = bokeh.layouts.layout([[slider], [slider_layer], [bokeh.layouts.row(plot1, plot)]],
                                      sizing_mode="scale_width")
    if Y_r.shape[0] == 1:
        layout = bokeh.layouts.row(plot1, plot)
    bpl.show(layout)

    return Y_r


def nb_imshow(image, cmap='jet'):
    """
    Interactive equivalent of imshow for ipython notebook
    """
    colormap = cm.get_cmap(cmap)  # choose any matplotlib colormap here
    grayp = [mpl.colors.rgb2hex(m) for m in colormap(np.arange(colormap.N))]
    xr = Range1d(start=0, end=image.shape[1])
    yr = Range1d(start=image.shape[0], end=0)
    p = bpl.figure(x_range=xr, y_range=yr)

    p.image(image=[image[::-1, :]], x=0, y=image.shape[0],
            dw=image.shape[1], dh=image.shape[0], palette=grayp)

    return p


def nb_plot_contour(image, A, d1, d2, thr=None, thr_method='max', maxthr=0.2, nrgthr=0.9,
                    face_color=None, line_color='black', alpha=0.4, line_width=2, **kwargs):
    """Interactive Equivalent of plot_contours for ipython notebook

    Parameters:
    -----------
    A:   np.ndarray or sparse matrix
            Matrix of Spatial components (d x K)

    Image:  np.ndarray (2D)
            Background image (e.g. mean, correlation)

    d1,d2: floats
            dimensions os image

    thr: scalar between 0 and 1
            Energy threshold for computing contours
            Kept for backwards compatibility. If not None then thr_method = 'nrg', and nrgthr = thr

    thr_method: [optional] string
            Method of thresholding:
                'max' sets to zero pixels that have value less than a fraction of the max value
                'nrg' keeps the pixels that contribute up to a specified fraction of the energy

    maxthr: [optional] scalar
            Threshold of max value

    nrgthr: [optional] scalar
            Threshold of energy

    display_number:     Boolean
            Display number of ROIs if checked (default True)

    max_number:    int
            Display the number for only the first max_number components (default None, display all numbers)

    cmap:     string
            User specifies the colormap (default None, default colormap)

    """
    p = nb_imshow(image, cmap='jet')
    center = com(A, d1, d2)
    p.circle(center[:, 1], center[:, 0], size=10, color="black",
             fill_color=None, line_width=2, alpha=1)
    coors = plot_contours(coo_matrix(A), image, thr=thr,
                          thr_method=thr_method, maxthr=maxthr, nrgthr=nrgthr)
    pl.close()
    cc1 = [np.clip(cor['coordinates'][:, 0], 0, d2) for cor in coors]
    cc2 = [np.clip(cor['coordinates'][:, 1], 0, d1) for cor in coors]

    p.patches(cc1, cc2, alpha=.4, color=face_color,
              line_color=line_color, line_width=2, **kwargs)
    return p

#%%


def playMatrix(mov, gain=1.0, frate=.033):
    for frame in mov:
        if gain != 1:
            cv2.imshow('frame', frame * gain)
        else:
            cv2.imshow('frame', frame)

        if cv2.waitKey(int(frate * 1000)) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
    cv2.destroyAllWindows()
#%% montage


def matrixMontage(spcomps, *args, **kwargs):
    numcomps, _, _ = spcomps.shape
    rowcols = int(np.ceil(np.sqrt(numcomps)))
    for k, comp in enumerate(spcomps):
        pl.subplot(rowcols, rowcols, k + 1)
        pl.imshow(comp, *args, **kwargs)
        pl.axis('off')


#%%
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


def view_patches_bar(Yr, A, C, b, f, d1, d2, YrA=None, img=None):
    """view spatial and temporal components interactively

     Parameters:
     -----------
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

    """

    pl.ion()
    if 'csc_matrix' not in str(type(A)):
        A = csc_matrix(A)
    if 'array' not in str(type(b)):
        b = b.toarray()

    nr, T = C.shape
    nb = f.shape[0]
    nA2 = np.sqrt(np.array(A.power(2).sum(axis=0))).squeeze()

    if YrA is None:
        Y_r = spdiags(old_div(1, nA2), 0, nr, nr) * (A.T.dot(Yr) -
                                                     (A.T.dot(b)).dot(f) - (A.dot(A)).dot(C)) + C
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
    vmax = np.percentile(img, 98)

    def update(val):
        i = np.int(np.round(s_comp.val))
        print(('Component:' + str(i)))

        if i < nr:

            ax1.cla()
            imgtmp = np.reshape(A[:, i].toarray(), (d1, d2), order='F')
            ax1.imshow(imgtmp, interpolation='None', cmap=pl.cm.gray)
            ax1.set_title('Spatial component ' + str(i + 1))
            ax1.axis('off')

            ax2.cla()
            ax2.plot(np.arange(T), Y_r[i], 'c', linewidth=3)
            ax2.plot(np.arange(T), C[i], 'r', linewidth=2)
            ax2.set_title('Temporal component ' + str(i + 1))
            ax2.legend(labels=['Filtered raw data', 'Inferred trace'])

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
                  cmap=None, swap_dim=False, colors='w', vmin=None, vmax=None, **kwargs):
    """Plots contour of spatial components against a background image and returns their coordinates

     Parameters:
     -----------
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
     --------
     Coor: list of coordinates with center of mass, contour plot coordinates and bounding box for each component
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
        warn("The way to call utilities.plot_contours has changed. " +
             "Look at the definition for more details.")

    ax = pl.gca()
    if vmax is None and vmin is None:
        pl.imshow(Cn, interpolation=None, cmap=cmap,
                  vmin=np.percentile(Cn[~np.isnan(Cn)], 1),
                  vmax=np.percentile(Cn[~np.isnan(Cn)], 99))
    else:
        pl.imshow(Cn, interpolation=None, cmap=cmap, vmin=vmin, vmax=vmax)

    coordinates = get_contours(A, np.shape(Cn), thr, thr_method, swap_dim)
    for c in coordinates:
        v = c['coordinates']
        c['bbox'] = [np.floor(np.nanmin(v[:, 1])), np.ceil(np.nanmax(v[:, 1])),
                     np.floor(np.nanmin(v[:, 0])), np.ceil(np.nanmax(v[:, 0]))]
        pl.plot(*v.T, c=colors)

    if display_numbers:
        d1, d2 = np.shape(Cn)
        d, nr = np.shape(A)
        cm = com(A, d1, d2)
        if max_number is None:
            max_number = A.shape[1]
        for i in range(np.minimum(nr, max_number)):
            if swap_dim:
                ax.text(cm[i, 0], cm[i, 1], str(i + 1), color=colors)
            else:
                ax.text(cm[i, 1], cm[i, 0], str(i + 1), color=colors)

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

    Parameters:
    -----------
    correlation_image_pnr: ndarray
        correlation image created with caiman.summary_images.correlation_pnr

    pnr_image: ndarray
        peak-to-noise image created with caiman.summary_images.correlation_pnr


    Returns:
    -------


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
