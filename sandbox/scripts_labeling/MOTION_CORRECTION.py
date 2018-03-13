# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 09:46:09 2016

@author: agiovann
"""
from __future__ import division
from __future__ import print_function
#%%
# TAKE BEGINNING OF ParallelProcessing.py
#%%
from builtins import zip
from past.utils import old_div
tmpls = []
fls = []
frates = []
resize_facts = []
for reg, img, proj, masks, template, f_rate, do_mot in zip(regions, images, projections, masks_all, templates, f_rates, do_motion_correct):
    fl = glob.glob(img + '/*.tif')
    fl.sort()
    if len(fl) > 0 and do_mot:
        for ff in fl:
            #            if os.path.exists(ff[:-3]+'npz'):
            #                print "existing:" + ff[:-3]+'npz'
            #            else:
            fls = fls + [ff]
            tmpls = tmpls + [template]
            frates = frates + [f_rate]
            resize_facts = resize_facts + \
                [(1, 1, old_div(final_f_rate, f_rate))]
#%%
file_res = cb.motion_correct_parallel(fls, fr=6, template=tmpls, margins_out=0, max_shift_w=45,
                                      max_shift_h=45, dview=c[::2], apply_smooth=True, save_hdf5=False, remove_blanks=False)
#%%
fls = []
templates = []
master_templates = []
for reg, img, proj, f_rate in zip(regions, images, projections, f_rates):
    fls = glob.glob(img + '/*.tif')
    fls.sort()
    all_movs = []
    all_shifts = []
    for f in fls:
        if os.path.exists(f[:-3] + 'npz'):
            with np.load(f[:-3] + 'npz') as fl:
                print(f)
                img_templ = fl['template'][np.newaxis, :, :]
                erode = old_div(np.shape(img_templ)[-1], 10)
                img_templ = img_templ[:, erode:-erode, erode:-erode]
                all_movs.append(img_templ)
                all_shifts.append(fl['shifts'])

    if len(all_movs) > 1:
        all_movs = cb.movie(np.concatenate(all_movs, axis=0), fr=1)
        img_ = np.nanmedian(all_movs, 0)
        all_movs = cb.motion_correct_parallel(file_names=[
                                              all_movs], fr=6, max_shift_w=45, max_shift_h=45, template=None, apply_smooth=True)
        all_movs = all_movs[0]
        templates.append(all_movs)
    else:
        templates.append(all_movs)

    master_templates.append(np.nanmedian(all_movs, 0))
    np.savez(os.path.join(img, 'master_template_one.npz'), all_movs=all_movs,
             fls=fls, master_template=master_templates[-1], all_shifts=all_shifts)

#%%
counter = 1
for ts, reg, mt in zip(templates, images, master_templates)[:]:
    print(reg)

    if 0:
        for t in ts:
            lq, hq = np.percentile(np.array(t), [10, 90])
            pl.cla()
            pl.imshow(np.squeeze(t), cmap='gray', vmin=lq, vmax=hq)
            pl.pause(.01)
    else:
        pl.subplot(5, 6, counter)
        lq, hq = np.percentile(np.array(mt), [5, 95])
        pl.imshow(np.squeeze(mt), cmap='gray', vmin=lq, vmax=hq)
        counter += 1
#%%
tmpls = []
fls = []
counter = 1
for reg, img, proj, f_rate in zip(regions, images, projections, f_rates):
    pl.subplot(5, 6, counter)
    counter += 1

    with np.load(os.path.join(img, 'master_template_one.npz')) as ld:
        template = ld['master_template']
        all_movs = ld['all_movs']
        fl = ld['fls']
        shifts = ld['all_shifts']

#    pl.imshow(template,cmap='gray')
    pl.plot(np.concatenate(shifts))
    pl.title(img)
    fls = fls + list(fl)
#    tmpls=tmpls+[template]*len(fl)
    tmpls = tmpls + list(all_movs)


#%%
file_res = cb.motion_correct_parallel(fls, fr=6, template=tmpls, margins_out=0, max_shift_w=45,
                                      max_shift_h=45, dview=c[::2], apply_smooth=True, save_hdf5=False, remove_blanks=False)
#%%

templates = []
master_templates = []
for reg, img, proj, f_rate in zip(regions, images, projections, f_rates):
    fls = glob.glob(img + '/*.tif')
    fls.sort()

    all_movs = []
    all_shifts = []
    for f in fls:

        with np.load(f[:-3] + 'npz') as fl:
            print(f)
            img_templ = fl['template'][np.newaxis, :, :]
            erode = old_div(np.shape(img_templ)[-1], 10)
            img_templ = img_templ[:, erode:-erode, erode:-erode]
            all_movs.append(img_templ)
            all_shifts.append(fl['shifts'])

    if len(all_movs) > 1:
        all_movs = cb.movie(np.concatenate(all_movs, axis=0), fr=1)
#        all_movs=cb.motion_correct_parallel(file_names=[all_movs],fr=6, max_shift_w=5, max_shift_h=5,template=None,apply_smooth=True)
#        all_movs=all_movs[0]
        templates.append(all_movs)
    else:
        templates.append(all_movs)

    master_templates.append(np.nanmedian(all_movs, 0))
#    np.savez(os.path.join(img,'master_template_two.npz'),all_movs=all_movs,fls=fls,master_template=master_templates[-1],all_shifts=all_shifts)
#%%
counter = 1
for ts, reg, mt in zip(templates, images, master_templates):
    print(reg)

    if 0:
        for t in ts:
            lq, hq = np.percentile(np.array(t), [1, 99])
            pl.cla()
            pl.imshow(np.squeeze(t), cmap='gray', vmin=lq, vmax=hq)
            pl.pause(.01)
    else:
        lq, hq = np.percentile(np.array(mt), [5, 95])
        pl.subplot(5, 6, counter)
        pl.imshow(np.squeeze(mt), cmap='gray', vmin=lq, vmax=hq)
        counter += 1


#%% check the x and y shifts

counter = 0
for reg, img, proj, template in zip(regions, images, projections, templates):
    pl.subplot(5, 6, counter + 1)
    print(counter)
    shifts_files = glob.glob(img + '/*.tif')
    all_shifts = []
    for sh_fl in shifts_files:

        with np.load(sh_fl) as ld:
            all_shifts.append(ld['shifts'])

    pl.plot(np.concatenate(all_shifts))
    pl.pause(.1)
    counter += 1

#%% visualize averages and masks
counter = 0
for reg, img, proj, masks, template in zip(regions, images, projections, masks_all, templates):
    pl.subplot(5, 6, counter + 1)
    print(counter)
    counter += 1

    template[np.isnan(template)] = 0
    lq, hq = np.percentile(template, [10, 99])
    pl.imshow(template, cmap='gray', vmin=lq, vmax=hq)
    pl.imshow(np.sum(masks, 0), cmap='hot', alpha=.3)
    pl.axis('off')
    pl.title(img.split('/')[-2])
    pl.pause(.1)

#%% check averages
counter = 0
for reg, img, proj, masks, template in zip(regions, images, projections, masks_all, templates):
    pl.subplot(5, 6, counter + 1)
    print(counter)
    movie_files = glob.glob(img + '/*.mmap')
    m = cb.load(movie_files[0], fr=6)
    template = np.mean(m, 0)
    lq, hq = np.percentile(template, [10, 99])
    pl.imshow(template, cmap='gray', vmin=lq, vmax=hq)
    pl.pause(.1)
    counter += 1
    pl.title(img.split('/')[-2])

#%% compute shifts so that everybody is well aligned
tmpls = []
fls = []
frates = []
resize_facts = []
for reg, img, proj, masks, template, f_rate in zip(regions, images, projections, masks_all, templates_path, f_rates):
    fl = glob.glob(img + '/*.tif')
    fl.sort()
    fls = fls + fl
    tmpls = tmpls + [template] * len(fl)
    frates = frates + [f_rate] * len(fl)
    resize_facts = resize_facts + \
        [(1, 1, old_div(final_f_rate, f_rate))] * len(fl)
#%%
if 0:
    new_fls = []
    new_tmpls = []
    xy_shifts = []
    for fl, tmpl in zip(fls, tmpls):
        if not os.path.exists(fl[:-3] + 'npz'):
            new_fls.append(fl)
            new_tmpls.append(tmpl)
        else:
            1

    fls = new_fls
    tmpls = new_tmpls


#    fls=glob.glob(img+'/*.tif')
#    fls.sort()
#    print fls
#%%
xy_shifts = []
for fl, tmpl in zip(fls, tmpls):
    if os.path.exists(fl[:-3] + 'npz'):
        print((fl[:-3] + 'npz'))
        with np.load(fl[:-3] + 'npz') as ld:
            xy_shifts.append(ld['shifts'])
    else:
        raise Exception('*********************** ERROR, FILE NOT EXISTING!!!')
#        with np.load(fl[:-3]+'npz') as ld:
#%%
name_new = cse.utilities.save_memmap_each(
    fls, dview=c[::3], base_name=None, resize_fact=resize_facts, remove_init=0, xy_shifts=xy_shifts)
#%%
frate_different = []
new_fls = []
new_frs = []
new_shfts = []
for fl, tmpl, fr, rs_f, shfts in zip(fls, tmpls, frates, resize_facts, xy_shifts):
    if len(glob.glob(fl[:-4] + '_*.mmap')) == 0 or fr != 30:
        new_fls.append(fl)
        new_frs.append(rs_f)
        new_shfts.append(shfts)
        if len(glob.glob(fl[:-4] + '_*.mmap')) > 0:
            frate_different.append(glob.glob(fl[:-4] + '_*.mmap')[0])
#%%
name_new = cse.utilities.save_memmap_each(
    new_fls, dview=c[::4], base_name=None, resize_fact=new_frs, remove_init=0, xy_shifts=new_shfts)
#%%
pars = []
import re

for bf in base_folders:
    fls = glob.glob(os.path.join(bf, 'images/*.mmap'))
    try:
        fls.sort(key=lambda fn: np.int(
            re.findall('_[0-9]{1,5}_d1_', fn)[0][1:-4]))
    except:
        fls.sort()
        print(fls)

    base_name_ = 'TOTAL_'
    n_chunks_ = 6
    dview_ = None
    pars.append([fls, base_name_, n_chunks_, dview_])
#%%
name_new = []


def memmap_place_holder(par):
    import ca_source_extraction as cse
    fls, base_name_, n_chunks_, dview_ = par
    return cse.utilities.save_memmap_join(fls, base_name=base_name_, n_chunks=n_chunks_, dview=dview_)


#%%
dview = c[::3]
names_map = dview.map_sync(memmap_place_holder, pars)
#%%
#fname_new=cse.utilities.save_memmap_join(fls,base_name='TOTAL_', n_chunks=6, dview=c[::3])
#%%
fnames_mmap = []
for reg, img, proj, masks, template in zip(regions, images, projections, masks_all, templates):
    if len(glob.glob(os.path.join(img, 'TOTAL_*.mmap'))) == 1:
        fnames_mmap.append(glob.glob(os.path.join(img, 'TOTAL_*.mmap'))[0])
    else:
        raise Exception('Number of files not as expected!')
#%%
counter = 0
for nm, tmpl, masks in zip(fnames_mmap, templates, masks_all):
    print(nm)
    counter += 1
    pl.subplot(3, 3, counter)
    Yr, dims, T = cse.utilities.load_memmap(nm)
    d1, d2 = dims
    Y = np.reshape(Yr, dims + (T,), order='F')
    img = np.mean(Y, -1)
#    np.allclose(img,tmpl)
    pl.imshow(img)
    pl.pause(1)
