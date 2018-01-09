#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on Wed Sep 21 11:30:44 2016

@author: agiovann
"""

from __future__ import division
from __future__ import print_function
#%%
#%%
from builtins import zip
from builtins import range
from past.utils import old_div

try:
    get_ipython().magic('load_ext autoreload')
    get_ipython().magic('autoreload 2')
    print((1))
except:
    print('Not launched under iPython')

import matplotlib as mpl
mpl.use('TKAgg')
from matplotlib import pyplot as plt
# plt.ion()

import sys
import numpy as np
import ca_source_extraction as cse

# sys.path.append('../SPGL1_python_port')
#%
from time import time
from scipy.sparse import coo_matrix
import tifffile
import subprocess
import time as tm
from time import time
import pylab as pl
import psutil
import glob
import os
import scipy
from ipyparallel import Client
import calblitz as cb
import openpyxl
#%%
excel_file = '/mnt/xfs1/home/agiovann/dropbox/2p_GCaMP6fhsyn/andrea/good_cells.xlsx'

wb = openpyxl.load_workbook(excel_file)
sheet = wb.get_sheet_by_name(wb.get_sheet_names()[0])
vals = []
for sh in sheet.rows[1:]:
    vls = [vl.value for vl in sh[:11]]
    vals.append(vls)
    print(vls)
#%%
ephys_folder = '/mnt/xfs1/home/agiovann/dropbox/2p_GCaMP6fhsyn/slice2016/ephys/'
img_folder = '/mnt/xfs1/home/agiovann/dropbox/2p_GCaMP6fhsyn/slice2016/Substacks/'

vl = vals[1]
params = []
file_names_hdf5 = []
diameters = []
for vl in vals:
    if vl[0] is not None:
        tmp1 = ['{0}'.format(kk.zfill(2)) for kk in vl[0].split('-')]
        tmp1[-1] = tmp1[-1][-2:]
        fn = ('_').join(tmp1)

        fls = glob.glob(os.path.join(img_folder, fn, vl[-4] + '*.tif'))
        fls.sort(key=lambda x: int(x[-6:-4].replace('s', '').replace('c', '')))
        print(vl)
        if len(fls) == 0:

            print('NOT FOUND: ')
            print(vl)

        else:
            diameters.append(np.int(vl[-1]))
            params.append(vl)

            m = cb.load_movie_chain(fls, fr=vl[-2])
            print(('Saving ' + fn + '_' + vl[-4] + '_mov.hdf5'))
            m.save(fn + '_' + vl[-4] + '_mov.hdf5')
            file_names_hdf5.append(fn + '_' + vl[-4] + '_mov.hdf5')
            np.savez(fn + '_' + vl[-4] + '_mov.npz', pars=vl)

    #        img=m.local_correlations()
    #        pl.imshow(img>.3)
    #        pl.pause(1)
    #        pl.cla()
    #            print fl
    #        print '*'
#%%
prs = list(zip(file_names_hdf5, diameters))
#%%
c[:].map_sync(place_holder, prs)
#%%


def place_holder(prs):
    fl, diam = prs
    import calblitz as cb
    import ca_source_extraction as cse
    import numpy as np
    m = cb.load(fl)
    Cn = m.local_correlations()
    cnmf = cse.CNMF(1, k=6, gSig=[old_div(diam, 2), old_div(
        diam, 2)], merge_thresh=0.8, p=2, dview=None, Ain=None)
    cnmf = cnmf.fit(m)
    A, C, b, f, YrA = cnmf.A, cnmf.C, cnmf.b, cnmf.f, cnmf.YrA
    np.savez(fl[:-5] + '_result.npz', A=A, C=C, b=b, f=f, YrA=YrA, Cn=Cn)
    return fl[:-5] + '_result.npz'


#%%
import glob
binary_mask = True
fls = glob.glob('*.hdf5')
pars = []
for fl in fls:
    with np.load(fl[:-5] + '_result.npz') as ld:
        A = ld['A'][()]
        C = ld['C']
        b = ld['b']
        f = ld['f']
        YrA = ld['YrA']
        Cn = ld['Cn']

#        traces=C+YrA
        m = cb.load(fl)
        m_fl = m.to_2D().T
        if binary_mask:
            masks = A.toarray()
            masks = old_div(masks, np.max(masks, 0))
            traces = (masks > 0.5).T.dot(m_fl)
        else:
            nA = (A.power(2)).sum(0)
            f = np.array(f).squeeze()
            b_size = f.shape[0]
            f_in = np.atleast_2d(f)

            bckg_1 = b.dot(f_in)
            m_fl = m_fl - bckg_1
            Y_r_sig = A.T.dot(m_fl)
            Y_r_sig = scipy.sparse.linalg.spsolve(
                scipy.sparse.spdiags(np.sqrt(nA), 0, nA.size, nA.size), Y_r_sig)

            Y_r_bl = A.T.dot(bckg_1)
            Y_r_bl = scipy.sparse.linalg.spsolve(
                scipy.sparse.spdiags(np.sqrt(nA), 0, nA.size, nA.size), Y_r_bl)
            Y_r_bl = cse.utilities.mode_robust(Y_r_bl, 1)
            traces = Y_r_sig + Y_r_bl[:, np.newaxis]

        T, d1, d2 = m.shape
        Y = m.transpose([1, 2, 0])

        with np.load(fl[:-4] + 'npz') as ld:
            pars = ld['pars']

        if T % len(m.file_name[0]):

            raise exception('Issue with the number of components!')

        num_trials = len(m.file_name[0])
        traces_f = []
        traces_dff = []
        time = old_div(list(range(old_div(T, len(m.file_name[0])))), m.fr)
        for tr in traces:
            tr_tmp = np.reshape(tr, (num_trials, -1)).T
            traces_f.append(tr_tmp)
            f = np.median(tr_tmp[time < 1.2, :], 0)
            f = np.maximum(f, 1)
            traces_dff.append((old_div((tr_tmp - f), f)))
#            traces_dff.append(np.reshape(tr,(num_trials,-1)))

        pars_ = ['' if p is None else p for p in pars]
        pars_ = np.array(pars_, dtype=object)
        masks = np.reshape(np.array(A.tocsc().todense()), [
                           d1, d2, A.shape[-1]], order='F').transpose([2, 0, 1])
        scipy.io.savemat(fl[:-5] + '_result.mat', {'traces_f': traces_f,
                                                   'traces_dff': traces_dff, 'masks': masks, 'time_img': time, 'pars': pars_})
        print((fl[:-5] + '_result.mat'))
        if 1:
            tB = np.minimum(-2, np.floor(-5. / 30 * m.fr))
            tA = np.maximum(5, np.ceil(25. / 30 * m.fr))
            Npeaks = 10
            #        traces_a=traces-scipy.ndimage.percentile_filter(traces,8,size=[1,np.shape(traces)[-1]/5])
            #        traces_b=np.diff(traces,axis=1)
            fitness_raw, fitness_delta, erfc_raw, erfc_delta, r_values, significant_samples\
                = cse.utilities.evaluate_components(Y, traces, A, C, b,
                                                    f, remove_baseline=True, N=5, robust_std=False, Athresh=0.1, Npeaks=Npeaks, tB=tB, tA=tA, thresh_C=0.3)

            idx_components_r = np.where(r_values >= .4)[0]
            idx_components_raw = np.where(fitness_raw < -20)[0]
            idx_components_delta = np.where(fitness_delta < -10)[0]

            idx_components = np.union1d(idx_components_r, idx_components_raw)
            idx_components = np.union1d(idx_components, idx_components_delta)
            idx_components_bad = np.setdiff1d(
                list(range(len(traces))), idx_components)
            idx_components = idx_components[np.argsort(idx_components_r)]

            pos = idx_components[0:1]
            masks = np.reshape(np.array(A.tocsc()[:, idx_components].todense()), [
                               d1, d2, -1], order='F').transpose([2, 0, 1])
            pl.close()
            pl.subplot(3, 1, 1)
            crd = cse.utilities.plot_contours(
                A.tocsc()[:, idx_components], Cn, cmap='gray')

            if len(pos) > 0:
                print(fl)
                pl.subplot(3, 1, 2)
                pl.imshow(masks[pos[0]])
                pl.subplot(3, 1, 3)
                pl.plot(traces_dff[0])

            print(r_values)
            pl.pause(.1)
#%%

#%%
Cn = m.local_correlations()
pl.imshow(Cn, cmap='gray')
#%%
crd = cse.utilities.plot_contours(A, Cn)
#%%
traces = C + YrA
idx_components, fitness, erfc, r_values, num_significant_samples = cse.utilities.evaluate_components(
    np.transpose(m, [1, 2, 0]), traces, A, N=5, robust_std=True)
#%%
crd = cse.utilities.plot_contours(A.tocsc()[:, idx_components], Cn)
