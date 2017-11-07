try:
    get_ipython().magic(u'load_ext autoreload')
    get_ipython().magic(u'autoreload 2')
except:
    print('NOT IPYTHON')

import numpy as np
from scipy.io import loadmat
from operator import itemgetter
import matplotlib.pyplot as plt
import caiman as cm
from caiman.source_extraction import cnmf
from caiman.source_extraction.cnmf.utilities import compute_residuals
import itertools
from past.utils import old_div
import pylab as pl
import scipy
#%%
def get_mapping(inferredC, trueC, A):
   """
   finds the mapping that maps each true neuron to the best inferred one
   such that overall Ca correlation is maximized, trueC[n] ~ inferredC[mapIdx[n]].
   For neurons that have not been found, mapIdx will contain NaNs.
   """
   N, T = trueC.shape
   cc = np.corrcoef(A.T.reshape(N, -1)) > .2
   blocks = [set(np.where(c)[0]) for c in cc]
   for k in range(len(blocks)):
       for _ in range(10):
           for j in range(len(blocks) - 1, k, -1):
               if len(blocks[k].intersection(blocks[j])):
                   blocks[k] = blocks[k].union(blocks[j])
                   blocks.pop(j)
   mapIdx = np.nan * np.zeros(N)
   corT = np.asarray([[np.corrcoef(s, tC)[0, 1]
                       for s in inferredC] for tC in trueC])
   # first assign neurons that have mutually highest correlation
   noTarget = list(range(len(inferredC)))  # indices that haven't been a target of the mapping yet
   for _ in range(10):
       if np.any(np.isnan(mapIdx)) and len(noTarget):
           nanIdx = np.where(np.isnan(mapIdx))[0]
           q = corT[np.isnan(mapIdx)][:, noTarget]
           to_del = []
           for k in range(len(q)):
               if np.argmax(q[:, np.argmax(q[k])]) == k:  # mutually highest correlation
                   mapIdx[nanIdx[k]] = noTarget[np.argmax(q[k])]
                   to_del.append(noTarget[np.argmax(q[k])])
           for d in to_del:
               noTarget.remove(d)
   # check permutations of nearby neurons
   while np.any(np.isnan(mapIdx)) and len(noTarget):
       nanIdx = np.where(np.isnan(mapIdx))[0]
       block = filter(lambda b: nanIdx[0] in b, blocks)[0]
       idx = list(block.intersection(nanIdx))  # ground truth indices
       candidates = list([np.argmax(corT[i, noTarget]) for i in idx])  # inferred indices
       if len(candidates) == len(set(candidates)):
           # the easier part: neurons within the group of nearby ones are
           # highly correlated with different inferred neurons
           for i in idx:
               k = np.argmax(corT[i, noTarget])
               mapIdx[i] = noTarget[k]
               del noTarget[k]
       else:  
           # the tricky part: neurons within the group of nearby ones are
           # highly correlated with the same inferred neurons
           candidates = list(set(np.concatenate([np.argsort(corT[i, noTarget])[-2:] for i in idx])))
           bestcorr = -np.inf
           for perm in itertools.permutations(candidates):
               perm = list(perm)
               c = np.diag(corT[idx][:, perm[:len(idx)]]).sum()
               if c > bestcorr:
                   bestcorr = c
                   bestperm = perm
           mapIdx[list(idx)] = bestperm[:len(idx)]
           for d in bestperm[:len(idx)]:
               noTarget.remove(d)
   return mapIdx
#%%
fname = 'test_sim.mat'
test_sim = loadmat(fname)
A, C, b, A_cnmfe, f, C_cnmfe, Craw_cnmfe, b0, sn, Yr, S_cnmfe = itemgetter(
    'A', 'C', 'b', 'A_cnmfe', 'f', 'C_cnmfe',
    'Craw_cnmfe', 'b0', 'sn', 'Y', 'S_cnmfe')(test_sim)
dims = (253, 316)

Y = Yr.T.reshape((-1,) + dims, order='F')

plt.imshow(Y[0])
#cm.movie(Y).play(fr=30, magnification=2)
plt.figure(figsize=(20, 4))
plt.plot(C.T)


gSig = 3   # gaussian width of a 2D gaussian kernel, which approximates a neuron
gSiz = 10  # average diameter of a neuron
min_corr = .9
min_pnr = 15
# If True, the background can be roughly removed. This is useful when the background is strong.
center_psf = True

fname_new = cm.save_memmap([Y], base_name='Yr',idx_xy=(slice(96,2*96),slice(96,2*96)))
dims = (96,96)
#fname_new = cm.save_memmap([Y], base_name='Yr')

Yr, dims, T = cm.load_memmap(fname_new)
Y = Yr.T.reshape((T,) + dims, order='F')
#%%
cn_filter, pnr = cm.summary_images.correlation_pnr(
    Y, gSig=gSig, center_psf=center_psf, swap_dim=False)
#%%
c, dview, n_processes = cm.cluster.setup_cluster(
    backend='local', n_processes=12, single_thread=False)
#%%
patches = True
if patches:
    cnm = cnmf.CNMF(n_processes=n_processes, method_init='corr_pnr', k=25, gSig=(3, 3), gSiz=(10, 10),
                    merge_thresh=.8, p=1, dview=dview, tsub=1, ssub=1, Ain=None, rf=(32, 32), stride=(32, 32),
                    only_init_patch=True, gnb=6, nb_patch=6, method_deconvolution='oasis',
                    low_rank_background=True, update_background_components=False, min_corr=min_corr,
                    min_pnr=min_pnr, normalize_init=False, deconvolve_options_init=None,
                    ring_size_factor=1.5, center_psf=True)
   

    
    cnm.fit(Y)
    #%%
    import scipy
    A_, C_, b_, f_, YrA_, sn_ = scipy.sparse.coo_matrix(cnm.A), cnm.C, cnm.b, cnm.f, cnm.YrA, cnm.sn 
         
    YrA_ = compute_residuals(np.array(Yr), A_,b_,C_,f_, dview = dview)        
#%%    ## %% DISCARD LOW QUALITY COMPONENT
    final_frate = 10
    r_values_min = .9  # threshold on space consistency
    fitness_min = -1  # threshold on time variability
    # threshold on time variability (if nonsparse activity)
    fitness_delta_min = - 1
    Npeaks = 5
    traces = C_ + YrA_
    # TODO: todocument
    idx_components, idx_components_bad = cm.components_evaluation.estimate_components_quality(
        traces, Yr, A_, C_, b_, f_, final_frate=final_frate, Npeaks=Npeaks,
        r_values_min=r_values_min, fitness_min=fitness_min, fitness_delta_min=fitness_delta_min, dview=dview)
    
    print(('Keeping ' + str(len(idx_components)) + ' and discarding  ' + str(len(idx_components_bad))))
    #%%
    crd = cm.utils.visualization.plot_contours(A_.tocsc()[:, idx_components], cn_filter, thr=.95)
    #%%
    import scipy
    cm.utils.visualization.view_patches_bar(Yr, scipy.sparse.coo_matrix(A_.tocsc()[:, idx_components]), C_[idx_components, :],
    b_, f_, dims[0], dims[1], YrA=YrA_[idx_components, :], img=cn_filter)
    #%%
    A_ = A_.tocsc()[:,idx_components]
    C_ = C_[idx_components]
    YrA_ = YrA_[idx_components]
    #%%
    YrA_GT = compute_residuals(np.array(Yr), A,b,C,f, dview = None)    

    cm.utils.visualization.view_patches_bar(Yr, scipy.sparse.coo_matrix(A), C,
                                            b, f, dims[0], dims[1], YrA=Craw_cnmfe, img=cn_filter)
#%%    
else:    
    #%%
    # cnm = cnmf.CNMF(n_processes=n_processes, method_init='corr_pnr', k=25, gSig=(3, 3), gSiz=(10, 10), merge_thresh=.8,
    #                 p=1, dview=dview, tsub=1, ssub=1, Ain=None, rf=(25, 25), stride=(25, 25),
    #                 only_init_patch=True, gnb=10, nb_patch=6, method_deconvolution='oasis',
    #                 low_rank_background=False, update_background_components=False, min_corr=min_corr,
    #                 min_pnr=min_pnr, normalize_init=False, deconvolve_options_init=None,
    #                 ring_size_factor=1.5, center_psf=True)
    
    cnm = cnmf.CNMF(n_processes=n_processes, method_init='corr_pnr', k=200, gSig=(3, 3), gSiz=(10, 10),
                    merge_thresh=.8, p=1, dview=dview, tsub=1, ssub=1, Ain=None,
                    only_init_patch=True, gnb=10, nb_patch=6, method_deconvolution='oasis',
                    low_rank_background=False, update_background_components=False, min_corr=min_corr,
                    min_pnr=min_pnr, normalize_init=False, deconvolve_options_init=None,
                    ring_size_factor=1.5, center_psf=True)
    
    
    #%
    
    Yr, dims, T = cm.load_memmap(fname_new)
    cnm.fit(Yr.T.reshape((T,) + dims, order='F'))


A_, C_, b_, f_, YrA_, sn_ = cnm.A, cnm.C, cnm.b, cnm.f, cnm.YrA, cnm.sn  
#%%


#%%
import scipy
cm.utils.visualization.view_patches_bar(Yr, scipy.sparse.coo_matrix(A_), C_,b_, f_, dims[0], dims[1], YrA=YrA_, img=cn_filter)
#%%
crd = cm.utils.visualization.plot_contours(A_, cn_filter, thr=.95, vmax=0.95)
plt.imshow(A_.sum(-1).reshape(dims, order='F'))
#%%

# mapping of neuron indices to ground truth indices
N, T = C.shape
#mapIdx = np.nan * np.zeros(N, dtype='uint8')
#corT = np.asarray([[np.corrcoef(s, tC)[0, 1]
#                    for s in C_] for tC in C])
#noTarget = list(range(N))
#while len(noTarget)>0:
#    print('iter')
#    if np.any(np.isnan(mapIdx)):
#        nanIdx = np.where(np.isnan(mapIdx))[0]
#        q = corT[np.isnan(mapIdx)][:, noTarget]
#        to_del = []
#        for k in range(len(q)):
#            if np.argmax(q[:, np.argmax(q[k])]) == k:  # mutually highest correlation
#                mapIdx[nanIdx[k]] = noTarget[np.argmax(q[k])]
#                to_del.append(noTarget[np.argmax(q[k])])
#        for d in to_del:
#            noTarget.remove(d)
#            
#
#mapIdx = mapIdx.astype(int)
mapIdx= get_mapping(C_, C, A)
mapIdx= mapIdx.astype(np.int)
if False:
    corC = np.array([np.corrcoef(C_[mapIdx[n]], C[n])[0, 1] for n in range(N)])
    corA = np.array([np.corrcoef(A_[:, mapIdx[n]].toarray().squeeze(), A[:, n])[0, 1] for n in range(N)])
    corC_cnmfe = np.array([np.corrcoef(C_cnmfe[n], C[n])[0, 1] for n in range(N)])
    corA_cnmfe = np.array([np.corrcoef(A_cnmfe.toarray()[:, n], A[:, n])[0, 1] for n in range(N)])

else:
    corC = np.array([np.corrcoef(C_[mapIdx[n]]+YrA_[mapIdx[n]], C[n]+YrA_GT[n])[0, 1] for n in range(N)])
    corA = np.array([np.corrcoef(A_[:, mapIdx[n]].toarray().squeeze(), A[:, n])[0, 1] for n in range(N)])
    corC_cnmfe = np.array([np.corrcoef(Craw_cnmfe[n], C[n]+YrA_GT[n])[0, 1] for n in range(N)])
    corA_cnmfe = np.array([np.corrcoef(A_cnmfe.toarray()[:, n], A[:, n])[0, 1] for n in range(N)])

print(np.median(corC), np.median(corA))
print(np.median(corC_cnmfe), np.median(corA_cnmfe))
