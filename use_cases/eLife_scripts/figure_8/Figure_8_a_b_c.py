# -*- coding: utf-8 -*-
"""
This script reproduces the results for Figure 8a-c (timing information)
by loading saved values.
More info can be found in the companion paper.
"""

try:
    if __IPYTHON__:
        print(1)
        # this is used for debugging purposes only. allows to reload classes
        # when changed
        get_ipython().magic('load_ext autoreload')
        get_ipython().magic('autoreload 2')
except NameError:
    print('Not launched under iPython')

import pylab as pl
from pandas import DataFrame
import numpy as np

#%%
pl.rcParams['pdf.fonttype'] = 42
font = {'family': 'Arial',
        'weight': 'regular',
        'size': 20}
pl.rc('font', **font)
#%%FIGURE 8 c  time performances (results have been manually annotated on an excel spreadsheet and reported here below)
# To regenerate the values run batch_evaluate_timing.py
pl.rcParams['pdf.fonttype'] = 42
font = {'family': 'Arial',
        'weight': 'regular',
        'size': 20}


all_timings = DataFrame([
    {'name': 'N.02.00', 'n_processes': 24, 'time_patch': 83.30931520462036, 'time_refit': 150.8432295322418,
     'time_eval': 8.844662427902222},
    {'name': 'J123', 'n_processes': 24, 'time_patch': 274.6004159450531, 'time_refit': 180.42933320999146,
     'time_eval': 41.29946994781494},
    {'name': 'J115', 'n_processes': 24, 'time_patch': 858.1903641223907, 'time_refit': 881.201664686203,
     'time_eval': 104.10698699951172},
    {'name': 'K53', 'n_processes': 24, 'time_patch': 1797.6236493587494, 'time_refit': 2088.205850839615,
     'time_eval': 214.41207218170166},

    {'name': 'N.02.00', 'n_processes': 12, 'time_patch': 93.71227407455444, 'time_refit': 171.99205470085144,
     'time_eval': 9.873874425888062},
    {'name': 'J123', 'n_processes': 12, 'time_patch': 297.5868630409241, 'time_refit': 209.39823293685913,
     'time_eval': 37.03397536277771},
    {'name': 'J115', 'n_processes': 12, 'time_patch': 910.1763546466827, 'time_refit': 867.1253700256348,
     'time_eval': 91.3763816356659},
    {'name': 'K53', 'n_processes': 12, 'time_patch': 2131.146340608597, 'time_refit': 1840.5280983448029,
     'time_eval': 226.46999287605286},

    {'name': 'N.02.00', 'n_processes': 6, 'time_patch': 119.35262084007263, 'time_refit': 237.7939488887787,
     'time_eval': 13.037508964538574},
    {'name': 'J123', 'n_processes': 6, 'time_patch': 474.36822843551636, 'time_refit': 291.2786304950714,
     'time_eval': 48.68373775482178},
    {'name': 'J115', 'n_processes': 6, 'time_patch': 1234.7518901824951, 'time_refit': 988.6932106018066,
     'time_eval': 168.36487412452698},
    {'name': 'K53', 'n_processes': 6, 'time_patch': 2756.9168894290924, 'time_refit': 2216.247531414032,
     'time_eval': 399.46128821372986},

    {'name': 'N.02.00', 'n_processes': 3, 'time_patch': 177.89352083206177, 'time_refit': 364.8443193435669,
     'time_eval': 19.78028917312622},
    {'name': 'J123', 'n_processes': 3, 'time_patch': 855.328578710556, 'time_refit': 540.6532158851624,
     'time_eval': 81.68031716346741},
    {'name': 'J115', 'n_processes': 3, 'time_patch': 2253.3668415546417, 'time_refit': 1791.0478348731995,
     'time_eval': 333.98589515686035},
    {'name': 'K53', 'n_processes': 3, 'time_patch': 4108.806590795517, 'time_refit': 3596.7499675750732,
     'time_eval': 733.7702996730804},

    {'name': 'N.02.00', 'n_processes': 2, 'time_patch': 228.71087837219238, 'time_refit': 495.9476628303528,
     'time_eval': 27.659457445144653},
    {'name': 'J123', 'n_processes': 2, 'time_patch': 1191.312772512436, 'time_refit': 712.5731451511383,
     'time_eval': 109.54649567604065},
    {'name': 'J115', 'n_processes': 2, 'time_patch': 2983.893436193466, 'time_refit': 2210.7389879226685,
     'time_eval': 471.05265831947327},
    {'name': 'K53', 'n_processes': 2, 'time_patch': 5008.163422584534, 'time_refit': np.nan, 'time_eval': np.nan},

    {'name': 'N.02.00', 'n_processes': 1, 'time_patch': 389.0271465778351, 'time_refit': 863.1781265735626,
     'time_eval': 40.629820346832275},
    {'name': 'J123', 'n_processes': 1, 'time_patch': 2390.3756499290466, 'time_refit': 1399.526368379593,
     'time_eval': 221.93205952644348},
    {'name': 'J115', 'n_processes': 1, 'time_patch': 5922.906694412231, 'time_refit': 4353.50389456749,
     'time_eval': 919.8520407676697},
    {'name': 'K53', 'n_processes': 1, 'time_patch': np.nan, 'time_refit': np.nan, 'time_eval': np.nan},
])
pl.figure()
pl.subplot(2, 4, 1)
all_timings['total'] = all_timings['time_patch'] + all_timings['time_refit'] + all_timings['time_eval']

all_timings = all_timings[all_timings['name'] != 'K53']
all_timings.index = all_timings['n_processes']
all_timings.groupby(by=['name'])['time_patch'].plot(logx=True, logy=True, label='name')
# pl.gca().invert_xaxis()
pl.title('initialization')
pl.subplot(2, 4, 2)
all_timings.groupby(by=['name'])['time_refit'].plot(logx=True, logy=True, label='name')
# pl.gca().invert_xaxis()
pl.ylabel('time(s)')
pl.title('refinement')
pl.subplot(2, 4, 3)
all_timings.groupby(by=['name'])['time_eval'].plot(logx=True, logy=True, label='name')
# pl.gca().invert_xaxis()
pl.title('quality evaluation')
pl.subplot(2, 4, 4)
all_timings.groupby(by=['name'])['total'].plot(x='n_processes', logx=True, logy=True, label='name')
# pl.gca().invert_xaxis()
pl.title('total')
pl.tight_layout()

all_timings_norm = all_timings.copy()
fact = all_timings_norm[all_timings_norm['n_processes'] == 1]

for indx, row in fact.iterrows():
    print(row['name'])
    for nm in ['time_patch', 'time_refit', 'time_eval', 'total']:
        all_timings_norm[nm][all_timings_norm['name'] == row['name']] /= row[nm]
        all_timings_norm[nm][all_timings_norm['name'] == row['name']] = 1 / all_timings_norm[nm][
            all_timings_norm['name'] == row['name']]
all_timings = all_timings_norm
logx = False
logy = False
pl.subplot(2, 4, 5)
all_timings.index = all_timings['n_processes']
all_timings.groupby(by=['name'])['time_patch'].plot(logx=logx, logy=logy)
# pl.gca().invert_xaxis()
pl.title('initialization')
pl.subplot(2, 4, 6)
all_timings.groupby(by=['name'])['time_refit'].plot(logx=logx, logy=logy, )
# pl.gca().invert_xaxis()
pl.ylabel('time(s)')
pl.title('refinement')
pl.subplot(2, 4, 7)
all_timings.groupby(by=['name'])['time_eval'].plot(logx=logx, logy=logy)
# pl.gca().invert_xaxis()
pl.title('quality evaluation')
pl.subplot(2, 4, 8)
all_timings.groupby(by=['name'])['total'].plot(x='n_processes', logx=logx, logy=logy, legend='name')
# pl.gca().invert_xaxis()
pl.title('total')
pl.tight_layout()

# %%FIGURE 8 a and b time performances (results have been manually annotated on an excel spreadsheet and reported here below)

pl.figure("Figure 8a and 8b", figsize=(20, 4))

pl.rcParams['pdf.fonttype'] = 42
font = {'family': 'Arial',
        'weight': 'regular',
        'size': 20}

t_mmap = dict()
t_patch = dict()
t_refine = dict()
t_filter_comps = dict()

size = np.log10(np.array([8.4, 121.7, 78.7, 35.8]) * 1000)
components = np.array([1099, 1541, 1013, 398])
components = np.array([1099, 1541, 1013, 398])

t_mmap['cluster'] = np.array([109, 561, 378, 135])
t_patch['cluster'] = np.array([92, 1063, 469, 142])
t_refine['cluster'] = np.array([256, 1065, 675, 265])
t_filter_comps['cluster'] = np.array([11, 143, 77, 30])

# t_mmap['desktop'] = np.array([25, 41, 11, 41, 135, 23, 690, 510, 176, 163])
# t_patch['desktop'] = np.array([21, 43, 16, 48, 85, 45, 2150, 949, 316, 475])
# t_refine['desktop'] = np.array([105, 205, 43, 279, 216, 254, 1749, 837, 237, 493])
# t_filter_comps['desktop'] = np.array([3, 5, 2, 5, 9, 7, 246, 81, 36, 38])
t_mmap['desktop'] = np.array([135, 690, 510, 176])
t_patch['desktop'] = np.array([83.309, 1797, 858, 274])
t_refine['desktop'] = np.array([150.8, 2088.20, 881.20, 180.4])
t_filter_comps['desktop'] = np.array([8.844, 214.41, 104.1, 41.299])

t_mmap['laptop'] = np.array([144, 731, 287, 125])
t_patch['laptop'] = np.array([177.893, 4108.8, 2253.366, 855.3])
t_refine['laptop'] = np.array([364.8, 3596.7, 1791.04, 540.6])
t_filter_comps['laptop'] = np.array([19.78, 733.77, 33.985, 81.68])

# these can be read from the final portion of of the script output
t_mmap['online'] = np.array([0, 0, 0, 0])
t_patch['online'] = np.array([0, 0, 0, 0])
t_refine['online'] = np.array([0, 0, 0, 0])
t_filter_comps['online'] = np.array([909.07959843, 9996.35542846, 7175.75227594, 1371.42159343])

pl.subplot(1, 4, 1)
for key in ['cluster', 'desktop', 'laptop', 'online']:
    np.log10(t_mmap[key] + t_patch[key] + t_refine[key] + t_filter_comps[key])
    pl.scatter((size), np.log10((t_mmap[key] + t_patch[key] + t_refine[key] + t_filter_comps[key])),
                s=np.array(components) / 10)
    pl.xlabel('size (GB)')
    pl.ylabel('time (minutes)')

pl.plot((np.sort(size)), np.log10((np.sort(10 ** size)) / 31.45), '--.k')

axx = pl.gca()
axx.locator_params(nbins=7)
axx.set_yticklabels([str(int((10 ** ss) / 60))[:5] for ss in axx.get_yticks()])
axx.set_xticklabels([str(int((10 ** ss) / 1000))[:5] for ss in axx.get_xticks()])
pl.legend(
    ['acquisition-time', 'cluster (112 CPUs)', 'workstation (24 CPUs)', 'workstation (3 CPUs)', 'online (6 CPUs)'])
pl.title('Total execution time')
# pl.xlim([3.8, 5.2])
# pl.ylim([2.35, 4.2])

counter = 2
for key in ['cluster', 'desktop']:
    pl.subplot(1, 3, counter)
    counter += 1
    if counter == 3:
        pl.title('Time per phase (cluster)')

    elif counter == 4:
        pl.title('Time per phase (workstation)')
    else:
        pl.title('Time per phase (online)')

    pl.bar((size), (t_mmap[key]), width=0.12, bottom=0)
    pl.bar((size), (t_patch[key]), width=0.12, bottom=(t_mmap[key]))
    pl.bar((size), (t_refine[key]), width=0.12, bottom=(t_mmap[key] + t_patch[key]))
    pl.bar((size), (t_filter_comps[key]), width=0.12, bottom=(t_mmap[key] + t_patch[key] + t_refine[key]))
    if counter == 5:
        pl.legend(['Initialization', 'track activity', 'update shapes'])
    else:
        pl.legend(['mem mapping', 'patch init', 'refine sol', 'quality  filter', 'acquisition time'])

    pl.plot((np.sort(size)), (10 ** np.sort(size)) / 31.45, '--k')
    pl.xlim([3.6, 5.2])
    axx = pl.gca()
    axx.locator_params(nbins=7)
    axx.set_yticklabels([str(int((ss) / 60))[:5] for ss in axx.get_yticks()])
    axx.set_xticklabels([str(int((10 ** ss) / 1000))[:5] for ss in axx.get_xticks()])
    pl.xlabel('size (GB)')
    pl.ylabel('time (minutes)')