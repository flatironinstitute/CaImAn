# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 12:07:09 2016

@author: agiovann
"""
import subprocess
import time
import ipyparallel
from ipyparallel import Client
import shutil
import glob
import shlex
import psutil
import sys
import os
import numpy as np
from caiman.mmapping import load_memmap
#%%
def extract_patch_coordinates(d1,d2,rf=(7,7),stride = (2,2)):
    """
    Function that partition the FOV in patches and return the indexed in 2D and 1D (flatten, order='F') formats
    Parameters
    ----------    
    d1,d2: int
        dimensions of the original matrix that will be  divided in patches
    rf: int
        radius of receptive field, corresponds to half the size of the square patch        
    stride: int
        degree of overlap of the patches
    """
    coords_flat=[]
    coords_2d=[]
    rf1,rf2 = rf
    stride1,stride2 = stride
    
    for xx in range(rf1,d1-rf1,2*rf1-stride1)+[d1-rf1]:   
        for yy in range(rf2,d2-rf2,2*rf2-stride2)+[d2-rf2]:
            
            coords_x=np.array(range(xx - rf1, xx + rf1 + 1))     
            coords_y=np.array(range(yy - rf2, yy + rf2 + 1))  
            print([xx - rf1, xx + rf1 + 1,yy - rf2, yy + rf2 + 1])
            coords_y = coords_y[(coords_y >= 0) & (coords_y < d2)]
            coords_x = coords_x[(coords_x >= 0) & (coords_x < d1)]
            idxs = np.meshgrid( coords_x,coords_y)
            coords_2d.append(idxs)
            coords_ =np.ravel_multi_index(idxs,(d1,d2),order='F')
            coords_flat.append(coords_.flatten())
      
    return coords_flat,coords_2d
#%%
def extract_rois_patch(file_name,d1,d2,rf=5,stride = 5):
    idx_flat,idx_2d=extract_patch_coordinates(d1, d2, rf=rf,stride = stride)
    perctl=95
    n_components=2
    tol=1e-6
    max_iter=5000
    args_in=[]    
    for id_f,id_2d in zip(idx_flat,idx_2d):        
        args_in.append((file_name, id_f,id_2d[0].shape, perctl,n_components,tol,max_iter))
    st=time.time()
    print len(idx_flat)
    try:
        if 1:
            c = Client()   
            dview=c[:]
            file_res = dview.map_sync(nmf_patches, args_in)                         
        else:
            file_res = map(nmf_patches, args_in)                         
    finally:
        dview.results.clear()   
        c.purge_results('all')
        c.purge_everything()
        c.close()
    
    print time.time()-st
    
    A1=lil_matrix((d1*d2,len(file_res)))
    C1=[]
    A2=lil_matrix((d1*d2,len(file_res)))
    C2=[]
    for count,f in enumerate(file_res):
        idx_,flt,ca,d=f
        A1[idx_,count]=flt[:,0][:,np.newaxis]        
        A2[idx_,count]=flt[:,1][:,np.newaxis]        
        C1.append(ca[0,:])
        C2.append(ca[1,:])
#        pl.imshow(np.reshape(flt[:,0],d,order='F'),vmax=10)
#        pl.pause(.1)
        
        
    return A1,A2,C1,C2
#%%
def apply_to_patch(mmap_file, shape, dview, rf , stride , function, *args, **kwargs):
    '''
    apply function to patches in parallel or not
    
    Parameters    
    ----------        
    file_name: string
        full path to an npy file (2D, pixels x time) containing the movie        
        
    shape: tuple of three elements
        dimensions of the original movie across y, x, and time 
    
   
    rf: int 
        half-size of the square patch in pixel
    
    stride: int
        amount of overlap between patches
           
        
    dview: ipyparallel view on client
        if None
    
   
    Returns
    -------
    results
    '''    
    
    (T,d1,d2)=shape
    d=d1*d2
    
    if not np.isscalar(rf):
        rf1,rf2=rf
    else:
        rf1=rf
        rf2=rf
        
    if not np.isscalar(stride):    
        stride1,stride2=stride
    else:
        stride1=stride
        stride2=stride
        
  
    
    idx_flat,idx_2d=extract_patch_coordinates(d1, d2, rf=(rf1,rf2), stride = (stride1,stride2))

    args_in=[]    
    
    for id_f,id_2d in zip(idx_flat[:],idx_2d[:]):        

        args_in.append((mmap_file.filename, id_f,id_2d[0].shape, function, args, kwargs))
        
        

    print len(idx_flat)

    
    if dview is not None:
        
        try:
            
            file_res = dview.map_sync(function_place_holder, args_in)  
            
            dview.results.clear()   

        except:
            print('Something went wrong')  
            raise
        finally:
            print('You may think that it went well but reality is harsh')
                    

    else:
        
        file_res = map(function_place_holder, args_in)      

    return file_res   
#%%
def function_place_holder(args_in):

    file_name, idx_,shapes,function, args, kwargs = args_in
    Yr, _, _ = load_memmap(file_name)   
    Yr = Yr[idx_,:]
    Yr.filename=file_name
    d,T=Yr.shape      
    Y=np.reshape(Yr,(shapes[1],shapes[0],T),order='F').transpose([2,0,1])           
    [T,d1,d2]=Y.shape
    
    res_fun = function(Y,*args,**kwargs)
    if type(res_fun) is not tuple:

        if res_fun.shape == (d1,d2):
            print '** reshaping form 2D to 1D'
            res_fun = np.reshape(res_fun,d1*d2,order = 'F')

    return res_fun, idx_
     
#%%
def start_server(slurm_script=None, ipcluster="ipcluster"):
    '''
    programmatically start the ipyparallel server

    Parameters
    ----------
    ncpus: int
        number of processors
    ipcluster : str
        ipcluster binary file name; requires 4 path separators on Windows. ipcluster="C:\\\\Anaconda2\\\\Scripts\\\\ipcluster.exe"
         Default: "ipcluster"    
    '''
    sys.stdout.write("Starting cluster...")
    sys.stdout.flush()
    ncpus=psutil.cpu_count()
    
    if slurm_script is None:
        if ipcluster == "ipcluster":
            p1 = subprocess.Popen("ipcluster start -n {0}".format(ncpus), shell=True, close_fds=(os.name != 'nt'))
        else:
            p1 = subprocess.Popen(shlex.split("{0} start -n {1}".format(ipcluster, ncpus)), shell=True, close_fds=(os.name != 'nt'))
#
        while True:
            try:
                c = ipyparallel.Client()
                if len(c) < ncpus:
                    sys.stdout.write(".")
                    sys.stdout.flush()
                    raise ipyparallel.error.TimeoutError
                c.close()                

                break
            except (IOError, ipyparallel.error.TimeoutError):
                sys.stdout.write(".")
                sys.stdout.flush()
                time.sleep(1)
                
    else:
        shell_source(slurm_script)
        pdir, profile = os.environ['IPPPDIR'], os.environ['IPPPROFILE']
        c = Client(ipython_dir=pdir, profile=profile)
        ee = c[:]
        ne = len(ee)
        print 'Running on %d engines.' % (ne)
        c.close()
        sys.stdout.write(" done\n")
        

#%%


def shell_source(script):
    """Sometime you want to emulate the action of "source" in bash,
    settings some environment variables. Here is a way to do it."""


    pipe = subprocess.Popen(". %s; env" % script,  stdout=subprocess.PIPE, shell=True)
    output = pipe.communicate()[0]
    env = dict()
    for line in output.splitlines():
        lsp = line.split("=", 1)
        if len(lsp) > 1:
            env[lsp[0]] = lsp[1]
#    env = dict((line.split("=", 1) for line in output.splitlines()))
    os.environ.update(env)
    pipe.stdout.close()
#%%


def stop_server(is_slurm=False, ipcluster='ipcluster',pdir=None,profile=None):
    '''
    programmatically stops the ipyparallel server
    Parameters
     ----------
     ipcluster : str
         ipcluster binary file name; requires 4 path separators on Windows
         Default: "ipcluster"
    '''
    sys.stdout.write("Stopping cluster...\n")
    sys.stdout.flush()

    if is_slurm:
        
        if pdir is None and profile is None:
            pdir, profile = os.environ['IPPPDIR'], os.environ['IPPPROFILE']
        c = Client(ipython_dir=pdir, profile=profile)
        ee = c[:]
        ne = len(ee)
        print 'Shutting down %d engines.' % (ne)
        c.shutdown(hub=True)
        shutil.rmtree('profile_' + str(profile))
        try:
            shutil.rmtree('./log/')
        except:
            print 'creating log folder'

        files = glob.glob('*.log')
        os.mkdir('./log')

        for fl in files:
            shutil.move(fl, './log/')

    else:
        if ipcluster == "ipcluster":
            proc = subprocess.Popen("ipcluster stop", shell=True, stderr=subprocess.PIPE, close_fds=(os.name != 'nt'))
        else:
            proc = subprocess.Popen(shlex.split(ipcluster + " stop"),
                                    shell=True, stderr=subprocess.PIPE, close_fds=(os.name != 'nt'))

        line_out = proc.stderr.readline()
        if 'CRITICAL' in line_out:
            sys.stdout.write("No cluster to stop...")
            sys.stdout.flush()
        elif 'Stopping' in line_out:
            st = time.time()
            sys.stdout.write('Waiting for cluster to stop...')
            while (time.time() - st) < 4:
                sys.stdout.write('.')
                sys.stdout.flush()
                time.sleep(1)
        else:
            print line_out
            print '**** Unrecognized Syntax in ipcluster output, waiting for server to stop anyways ****'
        
        proc.stderr.close()
        
    sys.stdout.write(" done\n")