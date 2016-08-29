# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 17:56:14 2016

@author: agiovann
"""

#%%
try:
    %load_ext autoreload
    %autoreload 2
    print 1
except:

    print 'NOT IPYTHON'
import matplotlib as mpl
mpl.use('TKAgg')
from matplotlib import pyplot as plt
#plt.ion()

import sys
import numpy as np
import ca_source_extraction as cse

#sys.path.append('../SPGL1_python_port')
#%
from time import time
from scipy.sparse import coo_matrix
import tifffile
import subprocess
import time as tm
from time import time
import pylab as pl
import psutil
import calblitz as cb
import shutil
import glob
from ipyparallel import Client
import os
import glob
import h5py
import re
 
#%%
#backend='SLURM'
backend='SLURM'
if backend == 'SLURM':
    n_processes = np.int(os.environ.get('SLURM_NPROCS'))
else:
    n_processes = np.maximum(np.int(psutil.cpu_count()),1) # roughly number of cores on your machine minus 1
print 'using ' + str(n_processes) + ' processes'


#%% start cluster for efficient computation
single_thread=False

if single_thread:
    dview=None
else:    
    try:
        c.close()
    except:
        print 'C was not existing, creating one'
    print "Stopping  cluster to avoid unnencessary use of memory...."
    sys.stdout.flush()  
    if backend == 'SLURM':
        try:
            cse.utilities.stop_server(is_slurm=True)
        except:
            print 'Nothing to stop'
        slurm_script='/mnt/xfs1/home/agiovann/SOFTWARE/Constrained_NMF/SLURM/slurmStart.sh'
        cse.utilities.start_server(slurm_script=slurm_script)
        pdir, profile = os.environ['IPPPDIR'], os.environ['IPPPROFILE']
        c = Client(ipython_dir=pdir, profile=profile)        
    else:
        cse.utilities.stop_server()
        cse.utilities.start_server()        
        c=Client()

    print 'Using '+ str(len(c)) + ' processes'
    dview=c[:]

#%% get all the right folders
params=[
'FN.151102_001',30,
'J115_2015-12-09_L01',30,
'J123_2015-11-20_L01_0',30,
'Jan25_2015_07_13',30,
'Jan40_exp2_001',30,
'Jan42_exp4_001',30,
'Jan-AMG1_exp2_new_001',30,
'Jan-AMG_exp3_001',30,
'k26_v1_176um_target_pursuit_002_013',30,
'k31_20151223_AM_150um_65mW_zoom2p2',30,
'k31_20160104_MMA_150um_65mW_zoom2p2',30,
'k31_20160106_MMA_400um_118mW_zoom2p2_00001_1-19',30,
'k31_20160107_MMP_150um_65mW_zoom2p2_00001_1-15',30,
'k36_20151229_MMA_200um_65mW_zoom2p2_00001_1-17',30,
'k36_20160115_RSA_400um_118mW_zoom2p2_00001_20-38',30,
'k36_20160127_RL_150um_65mW_zoom2p2_00002_22-41',30,
'k37_20160109_AM_150um_65mW_zoom2p2_00001_1-16',30,
'neurofinder.00.00',7,
'neurofinder.00.01',7,
'neurofinder.00.02',7,
'neurofinder.00.03',7,
'neurofinder.00.04',7,
'neurofinder.01.00',7.5,
'neurofinder.01.01',7.5,
'neurofinder.02.00',8,
'neurofinder.04.00',6.75,
'packer.001',15,
'Yi.data.001',30,
'Yi.data.002',30,
'yuste.Single_150u',10
]
f_rates=np.array([el for el in params[1::2]])
base_folders=[os.path.join('/mnt/ceph/neuro/labeling',el) for el in params[::2]]
#%%
final_f_rate=6.0
#%%
#for fld in base_folders[-13:-4]:
#    print fld
#    with open(os.path.join(fld,'info.json')) as f:
#        a=json.load(f)
#        print a['rate-hz']
        
#%%
counter=0
images=[os.path.join(el,'images') for el in base_folders]
regions=[os.path.join(el,'regions') for el in base_folders]
projections=[os.path.join(el,'projections') for el in base_folders]

#%% get masks and templates
counter=0         
masks_all=[];
templates=[]
templates_path=[]
for reg,img,proj in zip(regions,images,projections):
    
        
    print counter
    counter+=1   
    m=cb.load(proj+'/median_projection.tif',fr=1)
    templates_path.append(proj+'/median_projection.tif')
    templates.append(m)
    masks=cse.utilities.nf_read_roi_zip(reg+'/ben_regions.zip',m.shape)
    masks_all.append(masks)   
    

#%% compute shifts so that everybody is well aligned
counter=0     
for reg,img,proj,masks,template in zip(regions,images,projections,masks_all,templates):
    pl.subplot(5,6,counter+1)
    print counter
    counter+=1   
    
    template[np.isnan(template)]=0
    lq,hq=np.percentile(template,[10,99])
    pl.imshow(template,cmap='gray',vmin=lq,vmax=hq)
    pl.imshow(np.sum(masks,0),cmap='hot',alpha=.3)
    pl.axis('off')
    pl.title(img.split('/')[-2])
    pl.pause(.1)
#%% compute shifts so that everybody is well aligned
tmpls=[]
fls=[]
frates=[]
resize_facts=[]
for reg,img,proj,masks,template,f_rate in zip(regions,images,projections,masks_all,templates_path,f_rates):
    fl=glob.glob(img+'/*.tif')    
    fl.sort()
    fls=fls+fl
    tmpls=tmpls+[template]*len(fl)
    frates=frates+[f_rate]*len(fl)
    resize_facts=resize_facts+[(1,1,final_f_rate/f_rate)]*len(fl)
#%%
if 0:
    new_fls=[]
    new_tmpls=[]
    xy_shifts=[]
    for fl,tmpl in zip(fls,tmpls):
        if not os.path.exists(fl[:-3]+'npz'):
            new_fls.append(fl)
            new_tmpls.append(tmpl)
        else:
    
    fls=new_fls
    tmpls=new_tmpls  

    
#%%    
file_res=cb.motion_correct_parallel(fls,fr=30,template=tmpls,margins_out=0,max_shift_w=45, max_shift_h=45,dview=c[::2],apply_smooth=False,save_hdf5=False)    
file_res_all.append(file_res)
#    fls=glob.glob(img+'/*.tif')
#    fls.sort()
#    print fls
#%%
xy_shifts=[]
for fl,tmpl in zip(fls,tmpls):
    if os.path.exists(fl[:-3]+'npz'):
        print fl[:-3]+'npz'
        with np.load(fl[:-3]+'npz') as ld:
            xy_shifts.append(ld['shifts'])
    else:
        raise Exception('*********************** ERROR, FILE NOT EXISTING!!!')
#        with np.load(fl[:-3]+'npz') as ld:    
#%%
name_new=cse.utilities.save_memmap_each(fls, dview=c[::3],base_name=None, resize_fact=resize_facts, remove_init=0,xy_shifts=xy_shifts)
#%% 
frate_different=[]
new_fls=[]   
new_frs=[] 
new_shfts=[]
for fl,tmpl,fr,rs_f,shfts in zip(fls,tmpls,frates,resize_facts,xy_shifts):
    if len(glob.glob(fl[:-4]+'_*.mmap'))==0  or fr != 30:
        new_fls.append(fl)
        new_frs.append(rs_f)
        new_shfts.append(shfts)
        if len(glob.glob(fl[:-4]+'_*.mmap'))>0:
            frate_different.append(glob.glob(fl[:-4]+'_*.mmap')[0])
#%%
name_new=cse.utilities.save_memmap_each(new_fls, dview=c[::4],base_name=None, resize_fact=new_frs, remove_init=0,xy_shifts=new_shfts)
#%%
pars=[]
import re

for bf in base_folders:
    fls=glob.glob(os.path.join(bf,'images/*.mmap'))   
    try:
        fls.sort(key=lambda fn: np.int(re.findall('_[0-9]{1,5}_d1_',fn)[0][1:-4]))
    except:
        fls.sort() 
        print fls
        
    base_name_='TOTAL_'
    n_chunks_=6
    dview_=None
    pars.append([fls,base_name_,n_chunks_,dview_])
#%%
name_new=[]
def memmap_place_holder(par):
    import ca_source_extraction as cse
    fls,base_name_,n_chunks_,dview_=par
    return cse.utilities.save_memmap_join(fls,base_name=base_name_, n_chunks=n_chunks_, dview=dview_)
#%%
dview=c[::3]
names_map=dview.map_sync(memmap_place_holder,pars)    
#%%    
fname_new=cse.utilities.save_memmap_join(fls,base_name='TOTAL_', n_chunks=6, dview=c[::3])
#%%
fnames_mmap=[]
for reg,img,proj,masks,template in zip(regions,images,projections,masks_all,templates):
    if len(glob.glob(os.path.join(img,'TOTAL_*.mmap')))==1:
        fnames_mmap.append(glob.glob(os.path.join(img,'TOTAL_*.mmap'))[0])
    else:
        raise Exception('Number of files not as expected!')
#%%
for nm,tmpl,masks in zip(fnames_mmap,templates,masks_all):
    print nm
    Yr,dims,T=cse.utilities.load_memmap(nm)
    d1,d2=dims
    Y=np.reshape(Yr,dims+(T,),order='F')
    img=np.mean(Y,-1)
    np.allclose(img,tmpl)

#%% process files sequntially in case of failure
if 0:
    fnames1=[]        
    for f in fnames:
        if os.path.isfile(f[:-3]+'hdf5'):
            1
        else:
            print 1
            fnames1.append(f)
    #%% motion correct
    t1 = time()
    file_res=cb.motion_correct_parallel(fnames1,fr=30,template=None,margins_out=0,max_shift_w=45, max_shift_h=45,dview=None,apply_smooth=True)
    t2=time()-t1
    print t2        
#%% LOGIN TO MASTER NODE
# TYPE salloc -n n_nodes --exclusive
# source activate environment_name

#%%#%%
backend='local'
if backend == 'slurm':
    slurm_script='/mnt/xfs1/home/agiovann/SOFTWARE/Constrained_NMF/SLURM/slurmStart.sh'
    cse.utilities.start_server(slurm_script=slurm_script)
    #n_processes = 27#np.maximum(psutil.cpu_count() - 2,1) # roughly number of cores on your machine minus 1
    pdir, profile = os.environ['IPPPDIR'], os.environ['IPPPROFILE']
    client_ = Client(ipython_dir=pdir, profile=profile)
else:
    cse.utilities.stop_server()
    cse.utilities.start_server()
    client_ = Client()
print 'Using '+ str(len(client_)) + ' processes'

#%% motion correct
t1 = time()
file_res=cb.motion_correct_parallel(fnames,fr=30,template=None,margins_out=0,max_shift_w=45, max_shift_h=45,dview=client_[::2],apply_smooth=True)
t2=time()-t1
print t2  

#%%   
all_movs=[]
for f in  fnames:
    print f
    with np.load(f[:-3]+'npz') as fl:
#        pl.subplot(1,2,1)
#        pl.imshow(fl['template'],cmap=pl.cm.gray)
#        pl.subplot(1,2,2)
#        pl.plot(fl['shifts'])       
        all_movs.append(fl['template'][np.newaxis,:,:])
#        pl.pause(.1)
#        pl.cla()
#%%        
all_movs=cb.movie(np.concatenate(all_movs,axis=0),fr=10)
all_movs,shifts,corss,_=all_movs.motion_correct(template=all_movs[1],max_shift_w=45, max_shift_h=45)
#%%
template=np.median(all_movs[:],axis=0)
np.save(base_folder+'template_total',template)
pl.imshow(template,cmap=pl.cm.gray,vmax=120)
#%%
all_movs.play(backend='opencv',gain=5,fr=30)
#%%
t1 = time()
file_res=cb.motion_correct_parallel(fnames,30,template=template,margins_out=0,max_shift_w=45, max_shift_h=45,dview=client_[::2],remove_blanks=False)
t2=time()-t1
print t2
#%%
fnames=[]
for file in glob.glob(base_folder+'k31_20160107_MMP_150um_65mW_zoom2p2_000*[0-9].hdf5'):
        fnames.append(file)
fnames.sort()
print fnames  
#%%
file_res=cb.utils.pre_preprocess_movie_labeling(client_[::2], fnames, median_filter_size=(2,1,1), 
                                  resize_factors=[.2,.1666666666],diameter_bilateral_blur=4)

#%%
client_.close()
cse.utilities.stop_server(is_slurm=True)

#%%

#%%
fold=os.path.split(os.path.split(fnames[0])[-2])[-1]
os.mkdir(fold)
#%%
files=glob.glob(fnames[0][:-20]+'*BL_compress_.tif')
files.sort()
print files
#%%
m=cb.load_movie_chain(files,fr=3)
m.play(backend='opencv',gain=10,fr=40)
#%%
m.save(files[0][:-20]+'_All_BL.tif')
#%%
files=glob.glob(fnames[0][:-20]+'*[0-9]._compress_.tif')
files.sort()
print files
#%%
m=cb.load_movie_chain(files,fr=3)
m.play(backend='opencv',gain=3,fr=40)
#%%
m.save(files[0][:-20]+'_All.tif')
