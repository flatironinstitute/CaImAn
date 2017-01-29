# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 14:50:22 2016

@author: agiovann
"""
from __future__ import division
from __future__ import print_function
from builtins import str
from builtins import range
from past.utils import old_div
import cv2
import numpy as np
import pylab as pl
from tempfile import NamedTemporaryFile
from IPython.display import HTML
import sys
from warnings import warn
from scipy.sparse import issparse, spdiags, coo_matrix
from matplotlib.widgets import Slider
from caiman.base.rois import com
from scipy.ndimage.measurements import center_of_mass
import matplotlib.cm as cm
import matplotlib as mpl
try:
    import bokeh
    import bokeh.plotting as bpl
    from bokeh.io import vform, hplot
    from bokeh.models import CustomJS, ColumnDataSource, Range1d
except:
    print("Bokeh could not be loaded. Either it is not installed or you are not running within a notebook")

from caiman.summary_images import local_correlations


#%%
def view_patches(Yr, A, C, b, f, d1, d2, YrA=None, secs=1):
    """view spatial and temporal components (secs=0 interactive)

     Parameters
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

    """
    pl.ion()
    nr, T = C.shape
    nb = f.shape[0]
    A2 = A.copy()
    A2.data **= 2
    nA2 = np.sqrt(np.array(A2.sum(axis=0))).squeeze()
    #A = A*spdiags(1/nA2,0,nr,nr)
    #C = spdiags(nA2,0,nr,nr)*C
    #b = np.squeeze(b)
    #f = np.squeeze(f)
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
            pl.plot(np.arange(T), np.squeeze(np.array(Y_r[i, :])), 'c', linewidth=3)
            pl.plot(np.arange(T), np.squeeze(np.array(C[i, :])), 'r', linewidth=2)
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


def nb_view_patches(Yr, A, C, b, f, d1, d2, image_neurons=None, thr=0.99, denoised_color=None):
    '''
    Interactive plotting utility for ipython notbook

    Parameters
    -----------
    Yr: np.ndarray
        movie
    A,C,b,f: np.ndarrays
        outputs of matrix factorization algorithm

    d1,d2: floats
        dimensions of movie (x and y)

    image_neurons: np.ndarray
        image to be overlaid to neurons (for instance the average)

    thr: double
        threshold regulating the extent of the displayed patches

    denoised_color: string or None
        color name (e.g. 'red') or hex color code (e.g. '#F0027F')

    '''
    colormap = cm.get_cmap("jet")  # choose any matplotlib colormap here
    grayp = [mpl.colors.rgb2hex(m) for m in colormap(np.arange(colormap.N))]
    nr, T = C.shape
    nA2 = np.sum(np.array(A)**2, axis=0)
    b = np.squeeze(b)
    f = np.squeeze(f)
    #Y_r = np.array(spdiags(1/nA2,0,nr,nr)*(A.T*np.matrix(Yr-b[:,np.newaxis]*f[np.newaxis] - A.dot(C))) + C)
    Y_r = np.array(spdiags(old_div(1, nA2), 0, nr, nr) * (A.T * np.matrix(Yr) - (A.T *
                                                                                 np.matrix(b[:, np.newaxis])) * np.matrix(f[np.newaxis]) - (A.T.dot(A)) * np.matrix(C)) + C)

    bpl.output_notebook()
    x = np.arange(T)
    z = old_div(np.squeeze(np.array(Y_r[:, :].T)), 100)
    k = np.reshape(np.array(A), (d1, d2, A.shape[1]), order='F')
    if image_neurons is None:
        image_neurons = np.nanmean(k, axis=2)

    fig = pl.figure()
    coors = plot_contours(coo_matrix(A), image_neurons, thr=thr)
    pl.close()
#    cc=coors[0]['coordinates'];
    cc1 = [cor['coordinates'][:, 0] for cor in coors]
    cc2 = [cor['coordinates'][:, 1] for cor in coors]
    c1 = cc1[0]
    c2 = cc2[0]
    npoints = list(range(len(c1)))

    source = ColumnDataSource(
        data=dict(x=x, y=z[:, 0], y2=old_div(C[0], 100), z=z, z2=old_div(C.T, 100)))
    source2 = ColumnDataSource(data=dict(x=npoints, c1=c1, c2=c2, cc1=cc1, cc2=cc2))

    plot = bpl.figure(plot_width=600, plot_height=300)
    plot.line('x', 'y', source=source, line_width=1, line_alpha=0.6)
    if denoised_color is not None:
        plot.line('x', 'y2', source=source, line_width=1, line_alpha=0.6, color=denoised_color)

    callback = CustomJS(args=dict(source=source, source2=source2), code="""
            var data = source.get('data');
            var f = cb_obj.get('value')-1
            x = data['x']
            y = data['y']
            y2 = data['y2']

            for (i = 0; i < x.length; i++) {
                y[i] = data['z'][i][f]
                y2[i] = data['z2'][i][f]
            }

            var data2 = source2.get('data');
            c1 = data2['c1'];
            c2 = data2['c2'];
            cc1 = data2['cc1'];
            cc2 = data2['cc2'];

            for (i = 0; i < c1.length; i++) {
                   c1[i] = cc1[f][i]
                   c2[i] = cc2[f][i]
            }
            source2.trigger('change');
            source.trigger('change');

        """)

    slider = bokeh.models.Slider(start=1, end=Y_r.shape[
                                 0], value=1, step=1, title="Neuron Number", callback=callback)
    xr = Range1d(start=0, end=image_neurons.shape[1])
    yr = Range1d(start=image_neurons.shape[0], end=0)
    plot1 = bpl.figure(x_range=xr, y_range=yr, plot_width=300, plot_height=300)

    plot1.image(image=[image_neurons[::-1, :]], x=0,
                y=image_neurons.shape[0], dw=d2, dh=d1, palette=grayp)
    plot1.patch('c1', 'c2', alpha=0.6, color='purple', line_width=2, source=source2)

    layout = vform(slider, hplot(plot1, plot))

    bpl.show(layout)

    return Y_r


def get_contours3d(A, dims, thr=0.9):
    """Gets contour of spatial components and returns their coordinates

     Parameters
     -----------
     A:   np.ndarray or sparse matrix
               Matrix of Spatial components (d x K)
     dims: tuple of ints
               Spatial dimensions of movie (x, y, z)
     thr: scalar between 0 and 1
               Energy threshold for computing contours (default 0.995)

     Returns
     --------
     Coor: list of coordinates with center of mass and
            contour plot coordinates (per layer) for each component
    """
    d, nr = np.shape(A)
    d1, d2, d3 = dims
    x, y = np.mgrid[0:d2:1, 0:d3:1]

    coordinates = []
    cm = np.asarray([center_of_mass(a.reshape(dims, order='F')) for a in A.T])
    for i in range(nr):
        pars = dict()
        indx = np.argsort(A[:, i], axis=None)[::-1]
        cumEn = np.cumsum(A[:, i].flatten()[indx]**2)
        cumEn /= cumEn[-1]
        Bvec = np.zeros(d)
        Bvec[indx] = cumEn
        Bmat = np.reshape(Bvec, dims, order='F')
        pars['coordinates'] = []
        for B in Bmat:
            cs = pl.contour(y, x, B, [thr])
            # this fix is necessary for having disjoint figures and borders plotted correctly
            p = cs.collections[0].get_paths()
            v = np.atleast_2d([np.nan, np.nan])
            for pths in p:
                vtx = pths.vertices
                num_close_coords = np.sum(np.isclose(vtx[0, :], vtx[-1, :]))
                if num_close_coords < 2:
                    if num_close_coords == 0:
                        # case angle
                        newpt = np.round(old_div(vtx[-1, :], [d2, d1])) * [d2, d1]
                        vtx = np.concatenate((vtx, newpt[np.newaxis, :]), axis=0)

                    else:
                        # case one is border
                        vtx = np.concatenate((vtx, vtx[0, np.newaxis]), axis=0)

                v = np.concatenate((v, vtx, np.atleast_2d([np.nan, np.nan])), axis=0)

            pars['coordinates'] += [v]
        pars['CoM'] = np.squeeze(cm[i, :])
        pars['neuron_id'] = i + 1
        coordinates.append(pars)
    return coordinates


def nb_view_patches3d(Yr, A, C, b, f, dims, image_type='mean',
                      max_projection=False, axis=0, thr=0.9, denoised_color=None):
    '''
    Interactive plotting utility for ipython notbook

    Parameters
    -----------
    Yr: np.ndarray
        movie

    A,C,b,f: np.ndarrays
        outputs of matrix factorization algorithm

    dims: tuple of ints
        dimensions of movie (x, y and z)

    image_type: 'mean', 'max' or 'corr'
        image to be overlaid to neurons (average, maximum or nearest neigbor correlation)

    max_projection: boolean
        plot max projection along specified axis if True, plot layers if False

    axis: int (0, 1 or 2)
        axis along which max projection is performed or layers are shown

    thr: scalar between 0 and 1
        Energy threshold for computing contours

    denoised_color: string or None
        color name (e.g. 'red') or hex color code (e.g. '#F0027F')

    '''
    d, T = Yr.shape
    order = list(range(4))
    order.insert(0, order.pop(axis))
    Yr = Yr.reshape(dims + (-1,), order='F').transpose(order).reshape((d, T), order='F')
    A = A.reshape(dims + (-1,), order='F').transpose(order).reshape((d, -1), order='F')
    dims = tuple(np.array(dims)[order[:3]])
    d1, d2, d3 = dims
    colormap = cm.get_cmap("jet")  # choose any matplotlib colormap here
    grayp = [mpl.colors.rgb2hex(m) for m in colormap(np.arange(colormap.N))]
    nr, T = C.shape
    nA2 = np.sum(np.array(A)**2, axis=0)
    b = np.squeeze(b)
    f = np.squeeze(f)
    Y_r = np.array(spdiags(1. / nA2, 0, nr, nr) *
                   (A.T * np.matrix(Yr) - (A.T * np.matrix(b[:, np.newaxis])) *
                    np.matrix(f[np.newaxis]) - (A.T.dot(A)) * np.matrix(C)) + C)

    bpl.output_notebook()
    x = np.arange(T)
    z = np.squeeze(np.array(Y_r[:, :].T)) / 100
    k = np.reshape(np.array(A), dims + (A.shape[1],), order='F')
    source = ColumnDataSource(data=dict(x=x, y=z[:, 0], y2=C[0] / 100,
                                        z=z.tolist(), z2=(C.T / 100).tolist()))

    if max_projection:
        if image_type == 'corr':
            image_neurons = [(local_correlations(
                Yr.reshape(dims + (-1,), order='F'))[:, ::-1]).max(i)
                for i in range(3)]
        elif image_type in ['mean', 'max']:
            tmp = [({'mean': np.nanmean, 'max': np.nanmax}[image_type]
                    (k, axis=3)[:, ::-1]).max(i) for i in range(3)]
        else:
            raise ValueError("image_type must be 'mean', 'max' or 'corr'")

        image_neurons = np.nan * np.ones((int(1.05 * (d1 + d2)), int(1.05 * (d1 + d3))))
        image_neurons[:d2, -d3:] = tmp[0][::-1]
        image_neurons[:d2, :d1] = tmp[2].T[::-1]
        image_neurons[-d1:, -d3:] = tmp[1]
        offset1 = image_neurons.shape[1] - d3
        offset2 = image_neurons.shape[0] - d1
        coors = [plot_contours(coo_matrix(A.reshape(dims + (-1,), order='F').max(i)
                                          .reshape((np.prod(dims) // dims[i], -1), order='F')),
                               tmp[i], thr_method='nrg', nrgthr=thr)
                 for i in range(3)]
        pl.close()
        cc1 = [[list(cor['coordinates'][:, 0] + offset1) for cor in coors[0]],
               [list(cor['coordinates'][:, 1]) for cor in coors[2]],
               [list(cor['coordinates'][:, 0] + offset1) for cor in coors[1]]]
        cc2 = [[list(cor['coordinates'][:, 1]) for cor in coors[0]],
               [list(cor['coordinates'][:, 0]) for cor in coors[2]],
               [list(cor['coordinates'][:, 1] + offset2) for cor in coors[1]]]
        K = np.max([[len(c) for c in cc] for cc in cc1])
        c1x = [np.nan] * K
        c2x = [np.nan] * K
        c1y = [np.nan] * K
        c2y = [np.nan] * K
        c1z = [np.nan] * K
        c2z = [np.nan] * K
        c1x[:len(cc1[0][0])] = cc1[0][0]
        c2x[:len(cc2[0][0])] = cc2[0][0]
        c1y[:len(cc1[1][0])] = cc1[1][0]
        c2y[:len(cc2[1][0])] = cc2[1][0]
        c1z[:len(cc1[2][0])] = cc1[2][0]
        c2z[:len(cc2[2][0])] = cc2[2][0]
        source2_ = ColumnDataSource(data=dict(cc1=cc1, cc2=cc2))
        source2 = ColumnDataSource(data=dict(c1x=c1x, c1y=c1y, c1z=c1z,
                                             c2x=c2x, c2y=c2y, c2z=c2z))
        callback = CustomJS(args=dict(source=source, source2=source2,
                                      source2_=source2_), code="""
                var data = source.get('data');
                var f = cb_obj.get('value')-1
                y = data['y']
                y2 = data['y2']
                for (i = 0; i < y.length; i++) {
                    y[i] = data['z'][i][f];
                    y2[i] = data['z2'][i][f];
                }

                var data2_ = source2_.get('data');
                var data2 = source2.get('data');
                c1x = data2['c1x'];
                c2x = data2['c2x'];
                c1y = data2['c1y'];
                c2y = data2['c2y'];
                c1z = data2['c1z'];
                c2z = data2['c2z'];
                cc1 = data2_['cc1'];
                cc2 = data2_['cc2'];
                for (i = 0; i < cc1[0][f].length; i++) {
                       c1x[i] = cc1[0][f][i]
                       c2x[i] = cc2[0][f][i]
                }
                for (i = 0; i < cc1[1][f].length; i++) {
                       c1y[i] = cc1[1][f][i]
                       c2y[i] = cc2[1][f][i]
                }
                for (i = 0; i < cc1[2][f].length; i++) {
                       c1z[i] = cc1[2][f][i]
                       c2z[i] = cc2[2][f][i]
                }
                source2.trigger('change');
                source.trigger('change');
            """)

    else:
        if image_type == 'corr':
            image_neurons = local_correlations(Yr.reshape(dims + (-1,), order='F'))[:, ::-1]
        elif image_type in ['mean', 'max']:
            image_neurons = {'mean': np.nanmean, 'max': np.nanmax}[image_type](k, axis=3)[:, ::-1]
        else:
            raise ValueError('image_type must be mean, max or corr')

        cmap = bokeh.models.mappers.LinearColorMapper([mpl.colors.rgb2hex(m)
                                                       for m in colormap(np.arange(colormap.N))])
        cmap.high = image_neurons.max()
        coors = get_contours3d(A, dims, thr=thr)
        pl.close()
        cc1 = [[list(l[:, 0]) for l in n['coordinates']] for n in coors]
        cc2 = [[list(l[:, 1]) for l in n['coordinates']] for n in coors]
        linit = int(round(coors[0]['CoM'][0]))  # pick initial layer in which first neuron lies
        c1 = cc1[0][linit]
        c2 = cc2[0][linit]
        source2_ = ColumnDataSource(data=dict(cc1=cc1, cc2=cc2))
        source2 = ColumnDataSource(data=dict(c1=c1, c2=c2))
        x = list(range(d2))
        y = list(range(d3))
        source3 = ColumnDataSource(
            data=dict(im1=[image_neurons[linit].tolist()], im=[image_neurons.tolist()],
                      xx=[x], yy=[y], x=[0], y=[d2], dw=[d3], dh=[d2]))

        callback = CustomJS(args=dict(source=source, source2=source2,
                                      source2_=source2_), code="""
                var data = source.get('data');
                var f = slider_neuron.get('value')-1;
                var l = slider_layer.get('value')-1;
                y = data['y']
                y2 = data['y2']
                for (i = 0; i < y.length; i++) {
                    y[i] = data['z'][i][f];
                    y2[i] = data['z2'][i][f];
                }

                var data2 = source2.get('data');
                var data2_ = source2_.get('data');
                c1 = data2['c1'];
                c2 = data2['c2'];
                for (i = 0; i < c1.length; i++) {
                       c1[i] = data2_['cc1'][f][l][i];
                       c2[i] = data2_['cc2'][f][l][i];
                }
                source2.trigger('change');
                source.trigger('change');
            """)

        callback_layer = CustomJS(args=dict(source=source3, source2=source2, source2_=source2_), code="""
                var f = slider_neuron.get('value')-1;
                var l = slider_layer.get('value')-1;
                var im1 = source.get('data')['im1'][0];
                for (var i = 0; i < source.get('data')['xx'][0].length; i++) {
                    for (var j = 0; j < source.get('data')['yy'][0].length; j++){
                        im1[i][j] = source.get('data')['im'][0][l][i][j];
                    }
                }
                var data2 = source2.get('data');
                var data2_ = source2_.get('data');
                c1 = data2['c1'];
                c2 = data2['c2'];
                for (i = 0; i < c1.length; i++) {
                       c1[i] = data2_['cc1'][f][l][i];
                       c2[i] = data2_['cc2'][f][l][i];
                }
                source.trigger('change');
                source2.trigger('change');
            """)

    plot = bpl.figure(plot_width=600, plot_height=300)
    plot.line('x', 'y', source=source, line_width=1, line_alpha=0.6)
    if denoised_color is not None:
        plot.line('x', 'y2', source=source, line_width=1, line_alpha=0.6, color=denoised_color)
    slider = bokeh.models.Slider(start=1, end=Y_r.shape[0], value=1, step=1,
                                 title="Neuron Number", callback=callback)
    xr = Range1d(start=0, end=image_neurons.shape[1] if max_projection else d3)
    yr = Range1d(start=image_neurons.shape[0] if max_projection else d2, end=0)
    plot1 = bpl.figure(x_range=xr, y_range=yr, plot_width=300, plot_height=300)

    if max_projection:
        plot1.image(image=[image_neurons[::-1, :]], x=0, y=image_neurons.shape[0],
                    dw=image_neurons.shape[1], dh=image_neurons.shape[0], palette=grayp)
        plot1.patch('c1x', 'c2x', alpha=0.6, color='purple', line_width=2, source=source2)
        plot1.patch('c1y', 'c2y', alpha=0.6, color='purple', line_width=2, source=source2)
        plot1.patch('c1z', 'c2z', alpha=0.6, color='purple', line_width=2, source=source2)
        layout = bokeh.layouts.layout([[slider], [bokeh.layouts.row(plot1, plot)]],
                                      sizing_mode="scale_width")
    else:
        slider_layer = bokeh.models.Slider(start=1, end=d1, value=linit + 1, step=1,
                                           title="Layer", callback=callback_layer)
        callback.args['slider_neuron'] = slider
        callback.args['slider_layer'] = slider_layer
        callback_layer.args['slider_neuron'] = slider
        callback_layer.args['slider_layer'] = slider_layer
        plot1.image(image='im1', x='x', y='y', dw='dw', dh='dh',
                    color_mapper=cmap, source=source3)
        plot1.patch('c1', 'c2', alpha=0.6, color='purple', line_width=2, source=source2)
        layout = bokeh.layouts.layout([[slider], [slider_layer], [bokeh.layouts.row(plot1, plot)]],
                                      sizing_mode="scale_width")

    bpl.show(layout)

    return Y_r


def nb_imshow(image, cmap='jet'):
    '''
    Interactive equivalent of imshow for ipython notebook
    '''
    colormap = cm.get_cmap(cmap)  # choose any matplotlib colormap here
    grayp = [mpl.colors.rgb2hex(m) for m in colormap(np.arange(colormap.N))]
    xr = Range1d(start=0, end=image.shape[1])
    yr = Range1d(start=image.shape[0], end=0)
    p = bpl.figure(x_range=xr, y_range=yr)
#    p = bpl.figure(x_range=[0,image.shape[1]], y_range=[0,image.shape[0]])
#    p.image(image=[image], x=0, y=0, dw=image.shape[1], dh=image.shape[0], palette=grayp)
    p.image(image=[image[::-1, :]], x=0, y=image.shape[0],
            dw=image.shape[1], dh=image.shape[0], palette=grayp)

    return p


def nb_plot_contour(image, A, d1, d2, thr=None, thr_method='max', maxthr=0.2, nrgthr=0.9,
                    face_color=None, line_color='black', alpha=0.4, line_width=2, **kwargs):
    '''Interactive Equivalent of plot_contours for ipython notebook

    Parameters
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

    '''
    p = nb_imshow(image, cmap='jet')
    center = com(A, d1, d2)
    p.circle(center[:, 1], center[:, 0], size=10, color="black",
             fill_color=None, line_width=2, alpha=1)
    coors = plot_contours(coo_matrix(A), image, thr=thr,
                          thr_method=thr_method, maxthr=maxthr, nrgthr=nrgthr)
    pl.close()
    #    cc=coors[0]['coordinates'];
    cc1 = [np.clip(cor['coordinates'][:, 0], 0, d2) for cor in coors]
    cc2 = [np.clip(cor['coordinates'][:, 1], 0, d1) for cor in coors]

    p.patches(cc1, cc2, alpha=.4, color=face_color, line_color=line_color, line_width=2, **kwargs)
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
    numcomps, width, height = spcomps.shape
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
    if not hasattr(anim, '_encoded_video'):
        with NamedTemporaryFile(suffix='.mp4') as f:
            anim.save(f.name, fps=fps, extra_args=['-vcodec', 'libx264'])
            video = open(f.name, "rb").read()
        anim._encoded_video = video.encode("base64")

    return VIDEO_TAG.format(anim._encoded_video)

#%%


def display_animation(anim, fps=20):
    pl.close(anim._fig)
    return HTML(anim_to_html(anim, fps=fps))
#%%
#%%


def view_patches_bar(Yr, A, C, b, f, d1, d2, YrA=None, secs=1, img=None):
    """view spatial and temporal components interactively

     Parameters
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
    nr, T = C.shape
    nb = f.shape[0]
    A2 = A.copy()
    A2.data **= 2
    nA2 = np.sqrt(np.array(A2.sum(axis=0))).squeeze()
    #A = A*spdiags(1/nA2,0,nr,nr)
    #C = spdiags(nA2,0,nr,nr)*C
    #b = np.squeeze(b)
    #f = np.squeeze(f)
    if YrA is None:
        Y_r = np.array(A.T * np.matrix(Yr) - (A.T * np.matrix(b[:, np.newaxis])) * np.matrix(
            f[np.newaxis]) - (A.T.dot(A)) * np.matrix(C) + C)
    else:
        Y_r = YrA + C

    A = A * spdiags(old_div(1, nA2), 0, nr, nr)
    A = A.todense()
    imgs = np.reshape(np.array(A), (d1, d2, nr), order='F')
    if img is None:
        img = np.mean(imgs[:, :, :-1], axis=-1)

    bkgrnd = np.reshape(b, (d1, d2) + (nb,), order='F')
    fig = pl.figure(figsize=(10, 10))

    axcomp = pl.axes([0.05, 0.05, 0.9, 0.03])

    ax1 = pl.axes([0.05, 0.55, 0.4, 0.4])
#    ax1.axis('off')
    ax3 = pl.axes([0.55, 0.55, 0.4, 0.4])
#    ax1.axis('off')
    ax2 = pl.axes([0.05, 0.1, 0.9, 0.4])
#    axcolor = 'lightgoldenrodyellow'
#    axcomp = pl.axes([0.25, 0.1, 0.65, 0.03], axisbg=axcolor)

    s_comp = Slider(axcomp, 'Component', 0, nr + nb - 1, valinit=0)
    vmax = np.percentile(img, 98)

    def update(val):
        i = np.int(np.round(s_comp.val))
        print(('Component:' + str(i)))

        if i < nr:

            ax1.cla()
            imgtmp = imgs[:, :, i]
            ax1.imshow(imgtmp, interpolation='None', cmap=pl.cm.gray)
            ax1.set_title('Spatial component ' + str(i + 1))
            ax1.axis('off')

            ax2.cla()
            ax2.plot(np.arange(T), np.squeeze(np.array(Y_r[i, :])), 'c', linewidth=3)
            ax2.plot(np.arange(T), np.squeeze(np.array(C[i, :])), 'r', linewidth=2)
            ax2.set_title('Temporal component ' + str(i + 1))
            ax2.legend(labels=['Filtered raw data', 'Inferred trace'])

            ax3.cla()
            ax3.imshow(img, interpolation='None', cmap=pl.cm.gray, vmax=vmax)
            imgtmp2 = imgtmp.copy()
            imgtmp2[imgtmp2 == 0] = np.nan
            ax3.imshow(imgtmp2, interpolation='None', alpha=0.5, cmap=pl.cm.hot)
        else:
            #import pdb; pdb.set_trace()
            ax1.cla()
            ax1.imshow(bkgrnd[:, :, i - nr], interpolation='None')
            ax1.set_title('Spatial background ' + str(i + 1 - nr))

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
    id2 = fig.canvas.mpl_connect('key_release_event', arrow_key_image_control)
    pl.show()
#%%


def plot_contours(A, Cn, thr=None, thr_method='max', maxthr=0.2, nrgthr=0.9, display_numbers=True, max_number=None, cmap=None, swap_dim=False, colors='w', vmin=None, vmax=None, **kwargs):
    """Plots contour of spatial components against a background image and returns their coordinates

     Parameters
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

     Returns
     --------
     Coor: list of coordinates with center of mass, contour plot coordinates and bounding box for each component
    """
    if issparse(A):
        A = np.array(A.todense())
    else:
        A = np.array(A)

    if swap_dim:
        Cn = Cn.T
        print('Swapping dim')

    d1, d2 = np.shape(Cn)
    d, nr = np.shape(A)
    if max_number is None:
        max_number = nr

    if thr is not None:
        thr_method = 'nrg'
        nrgthr = thr
        warn("The way to call utilities.plot_contours has changed. Look at the definition for more details.")

    x, y = np.mgrid[0:d1:1, 0:d2:1]

    ax = pl.gca()
    # Cn[np.isnan(Cn)]=0
    if vmax is None and vmin is None:
        pl.imshow(Cn, interpolation=None, cmap=cmap,
                  vmin=np.percentile(Cn[~np.isnan(Cn)], 1), vmax=np.percentile(Cn[~np.isnan(Cn)], 99))
    else:
        pl.imshow(Cn, interpolation=None, cmap=cmap,
                  vmin=vmin, vmax=vmax)

    coordinates = []
    cm = com(A, d1, d2)
    for i in range(np.minimum(nr, max_number)):
        pars = dict(kwargs)
        if thr_method == 'nrg':
            indx = np.argsort(A[:, i], axis=None)[::-1]
            cumEn = np.cumsum(A[:, i].flatten()[indx]**2)
            cumEn /= cumEn[-1]
            Bvec = np.zeros(d)
            Bvec[indx] = cumEn
            thr = nrgthr

        else:  # thr_method = 'max'
            if thr_method != 'max':
                warn("Unknown threshold method. Choosing max")
            Bvec = A[:, i].flatten()
            Bvec /= np.max(Bvec)
            thr = maxthr

        if swap_dim:
            Bmat = np.reshape(Bvec, np.shape(Cn), order='C')
        else:
            Bmat = np.reshape(Bvec, np.shape(Cn), order='F')
        cs = pl.contour(y, x, Bmat, [thr], colors=colors)
        # this fix is necessary for having disjoint figures and borders plotted correctly
        p = cs.collections[0].get_paths()
        v = np.atleast_2d([np.nan, np.nan])
        for pths in p:
            vtx = pths.vertices
            num_close_coords = np.sum(np.isclose(vtx[0, :], vtx[-1, :]))
            if num_close_coords < 2:
                if num_close_coords == 0:
                    # case angle
                    newpt = np.round(old_div(vtx[-1, :], [d2, d1])) * [d2, d1]
                    #import ipdb; ipdb.set_trace()
                    vtx = np.concatenate((vtx, newpt[np.newaxis, :]), axis=0)

                else:
                    # case one is border
                    vtx = np.concatenate((vtx, vtx[0, np.newaxis]), axis=0)
                    #import ipdb; ipdb.set_trace()

            v = np.concatenate((v, vtx, np.atleast_2d([np.nan, np.nan])), axis=0)
#        p = cs.collections[0].get_paths()[0]
#        v = p.vertices

        pars['CoM'] = np.squeeze(cm[i, :])
        pars['coordinates'] = v
        pars['bbox'] = [np.floor(np.min(v[:, 1])), np.ceil(np.max(v[:, 1])),
                        np.floor(np.min(v[:, 0])), np.ceil(np.max(v[:, 0]))]
        pars['neuron_id'] = i + 1
        coordinates.append(pars)

    if display_numbers:
        for i in range(np.minimum(nr, max_number)):
            if swap_dim:
                ax.text(cm[i, 0], cm[i, 1], str(i + 1), color=colors)
            else:
                ax.text(cm[i, 1], cm[i, 0], str(i + 1), color=colors)

    return coordinates
