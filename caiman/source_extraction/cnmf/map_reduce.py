# -*- coding: utf-8 -*-
"""
Function for implementing parallel scalable segmentation of two photon imaging data

Created on Wed Feb 17 14:58:26 2016

@author: agiovann
"""
from __future__ import division
from __future__ import print_function
from builtins import zip
from builtins import str
from builtins import map
from builtins import range
from past.utils import old_div
from ipyparallel import Client
import numpy as np
from scipy.sparse import lil_matrix,coo_matrix
import time
import scipy
import os
from caiman.mmapping import load_memmap
from caiman.cluster import extract_patch_coordinates,extract_rois_patch


#%%    
def cnmf_patches(args_in):
    import numpy as np
    import caiman as cm
    import time
    import logging


#    file_name, idx_,shapes,p,gSig,K,fudge_fact=args_in
    file_name, idx_,shapes,options=args_in

    name_log=os.path.basename(file_name[:-5])+ '_LOG_ ' + str(idx_[0])+'_'+str(idx_[-1])
    logger = logging.getLogger(name_log)
    hdlr = logging.FileHandler('./'+name_log)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr) 
    logger.setLevel(logging.INFO)



    p=options['temporal_params']['p']

    logger.info('START')

    logger.info('Read file')
    Yr,_,_=load_memmap(file_name)    

    

    Yr=Yr[idx_,:]
    
    
    if (np.sum(np.abs(np.diff(Yr))))>0.1:

        Yr.filename=file_name
        d,T=Yr.shape      
        
        Y=np.reshape(Yr,(shapes[1],shapes[0],T),order='F')  
        Y.filename=file_name
        
        [d1,d2,T]=Y.shape

        options['spatial_params']['dims']=(d1,d2)
        logger.info('Preprocess Data')
        Yr,sn,g,psx=cm.source_extraction.cnmf.pre_processing.preprocess_data(Yr,**options['preprocess_params'])
        

        logger.info('Initialize Components') 
        
        Ain, Cin, b_in, f_in, center=cm.source_extraction.cnmf.initialization.initialize_components(Y, **options['init_params']) 
        
        nA = np.squeeze(np.array(np.sum(np.square(Ain),axis=0)))

        nr=nA.size
        Cin=coo_matrix(Cin)
        

        YA = (Ain.T.dot(Yr).T)*scipy.sparse.spdiags(old_div(1.,nA),0,nr,nr)
        AA = ((Ain.T.dot(Ain))*scipy.sparse.spdiags(old_div(1.,nA),0,nr,nr))
        YrA = YA - Cin.T.dot(AA)
        Cin=Cin.todense()           

        if options['patch_params']['only_init']:

            return idx_,shapes, coo_matrix(Ain), b_in, Cin, f_in, None, None , None, None, g, sn, options, YrA.T

        else:

            logger.info('Spatial Update')                                                      
            A,b,Cin = cm.source_extraction.cnmf.spatial.update_spatial_components(Yr, Cin, f_in, Ain, sn=sn, **options['spatial_params'])  
            options['temporal_params']['p'] = 0 # set this to zero for fast updating without deconvolution
            

            logger.info('Temporal Update')  
            C,f,S,bl,c1,neurons_sn,g,YrA = cm.source_extraction.cnmf.temporal.update_temporal_components(Yr,A,b,Cin,f_in,bl=None,c1=None,sn=None,g=None,**options['temporal_params'])

            logger.info('Merge Components') 
            A_m,C_m,nr_m,merged_ROIs,S_m,bl_m,c1_m,sn_m,g_m=cm.source_extraction.cnmf.merging.merge_components(Yr,A,b,C,f,S,sn,options['temporal_params'], options['spatial_params'], bl=bl, c1=c1, sn=neurons_sn, g=g, thr=options['merging']['thr'], fast_merge = True)

            logger.info('Update Spatial II')
            A2,b2,C2 = cm.source_extraction.cnmf.spatial.update_spatial_components(Yr, C_m, f, A_m, sn=sn, **options['spatial_params'])

            logger.info('Update Temporal II')                                                       
            options['temporal_params']['p'] = p # set it back to original value to perform full deconvolution
            C2,f2,S2,bl2,c12,neurons_sn2,g21,YrA = cm.source_extraction.cnmf.temporal.update_temporal_components(Yr,A2,b2,C2,f,bl=None,c1=None,sn=None,g=None,**options['temporal_params'])


            Y=[]
            Yr=[]

            logger.info('Done!')
            return idx_,shapes,A2,b2,C2,f2,S2,bl2,c12,neurons_sn2,g21,sn,options,YrA

    else:
        return None                




#%%
def run_CNMF_patches(file_name, shape, options, rf=16, stride = 4, gnb = 1, dview=None, memory_fact=1):
    """Function that runs CNMF in patches, either in parallel or sequentiually, and return the result for each. It requires that ipyparallel is running

    Parameters
    ----------        
    file_name: string
        full path to an npy file (2D, pixels x time) containing the movie        

    shape: tuple of thre elements
        dimensions of the original movie across y, x, and time 

    options:
        dictionary containing all the parameters for the various algorithms

    rf: int 
        half-size of the square patch in pixel

    stride: int
        amount of overlap between patches

    gnb: int
        number of global background components

    backend: string
        'ipyparallel' or 'single_thread' or SLURM

    n_processes: int
        nuber of cores to be used (should be less than the number of cores started with ipyparallel)

    memory_fact: double
        unitless number accounting how much memory should be used. It represents the fration of patch processed in a single thread. You will need to try different values to see which one would work


    Returns
    -------
    A_tot: matrix containing all the componenents from all the patches

    C_tot: matrix containing the calcium traces corresponding to A_tot

    sn_tot: per pixel noise estimate

    optional_outputs: set of outputs related to the result of CNMF ALGORITHM ON EACH patch   
    """
    (d1,d2,T)=shape
    d=d1*d2
    K=options['init_params']['K']

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

    options['preprocess_params']['n_pixels_per_process']=np.int(old_div((rf1*rf2),memory_fact))
    options['spatial_params']['n_pixels_per_process']=np.int(old_div((rf1*rf2),memory_fact))
    options['temporal_params']['n_pixels_per_process']=np.int(old_div((rf1*rf2),memory_fact))
    nb = options['spatial_params']['nb']

    idx_flat,idx_2d=extract_patch_coordinates(d1, d2, rf=(rf1,rf2), stride = (stride1,stride2))
#    import pdb 
#    pdb.set_trace()
    args_in=[]    
    for id_f,id_2d in zip(idx_flat[:],idx_2d[:,:,0].flatten()):        

        args_in.append((file_name, id_f,id_2d.shape, options))

    print((len(idx_flat)))

    st=time.time()        

    if dview is not None:

        try:

            file_res = dview.map_sync(cnmf_patches, args_in)        
            dview.results.clear()   

        except:
            print('Something went wrong')  
            raise
        finally:
            print('You may think that it went well but reality is harsh')


    else:

        file_res = list(map(cnmf_patches, args_in))                         


    print((time.time()-st))

     
    # count components
    count=0
    count_bgr = 0
    patch_id=0
    num_patches=len(file_res)
    for fff in file_res:
        if fff is not None:
            idx_,shapes,A,b,C,f,S,bl,c1,neurons_sn,g,sn,_,YrA=fff
            
            for ii in range(np.shape(b)[-1]):

                count_bgr += 1

            for ii in range(np.shape(A)[-1]):            
                new_comp=old_div(A.tocsc()[:,ii],np.sqrt(np.sum(np.array(A.tocsc()[:,ii].todense())**2)))
                if new_comp.sum()>0:
                    count+=1

            patch_id+=1  

            
    A_tot=scipy.sparse.csc_matrix((d,count))
    B_tot=scipy.sparse.csc_matrix((d,nb*num_patches))
    C_tot=np.zeros((count,T))
    YrA_tot=np.zeros((count,T))
    F_tot=np.zeros((nb*num_patches,T))
    mask=np.zeros(d)
    sn_tot=np.zeros((d1*d2))
    b_tot=[]
    f_tot=[]
    bl_tot=[]
    c1_tot=[]
    neurons_sn_tot=[]
    g_tot=[]    
    idx_tot=[];
    shapes_tot=[]    
    id_patch_tot=[]

    count=0
    count_bgr = 0
    patch_id=0

    print('Transforming patches into full matrix')

    for fff in file_res:
        if fff is not None:
            idx_,shapes,A,b,C,f,S,bl,c1,neurons_sn,g,sn,_,YrA=fff
            sn_tot[idx_]=sn
            b_tot.append(b)
            f_tot.append(f)
            bl_tot.append(bl)
            c1_tot.append(c1)
            neurons_sn_tot.append(neurons_sn)
            g_tot.append(g)
            idx_tot.append(idx_)
            shapes_tot.append(shapes)
            mask[idx_] += 1

            for ii in range(np.shape(b)[-1]):
#                import pdb
#                pdb.set_trace()
#                print ii

                B_tot[idx_,patch_id]=b[:,ii,np.newaxis]
                F_tot[patch_id,:]=f[ii,:]
                count_bgr += 1

            for ii in range(np.shape(A)[-1]):            
                new_comp=old_div(A.tocsc()[:,ii],np.sqrt(np.sum(np.array(A.tocsc()[:,ii].todense())**2)))
                if new_comp.sum()>0:
                    A_tot[idx_,count]=new_comp
                    C_tot[count,:]=C[ii,:]                      
                    YrA_tot[count,:]=YrA[ii,:]
                    id_patch_tot.append(patch_id)
                    count+=1

            patch_id+=1  
        else:
            print('Skipped Empty Patch')

    A_tot=A_tot[:,:count]
    C_tot=C_tot[:count,:]  
    YrA_tot=YrA_tot[:count,:]  

    optional_outputs=dict()
    optional_outputs['b_tot']=b_tot
    optional_outputs['f_tot']=f_tot
    optional_outputs['bl_tot']=bl_tot
    optional_outputs['c1_tot']=c1_tot
    optional_outputs['neurons_sn_tot']=neurons_sn_tot
    optional_outputs['g_tot']=g_tot
    optional_outputs['idx_tot']=idx_tot
    optional_outputs['shapes_tot']=shapes_tot
    optional_outputs['id_patch_tot']= id_patch_tot
    optional_outputs['B'] = B_tot
    optional_outputs['F'] = F_tot
    optional_outputs['mask'] = mask

    print("Generating backgound")
    Im = scipy.sparse.csr_matrix((old_div(1.,mask),(np.arange(d),np.arange(d))))
    Bm = Im.dot(B_tot)
    A_tot = Im.dot(A_tot)

    f = np.r_[np.atleast_2d(np.mean(F_tot,axis=0)),np.random.rand(gnb-1,T)]

    for iter in range(100):
        b = np.fmax(Bm.dot(F_tot.dot(f.T)).dot(np.linalg.inv(f.dot(f.T))),0)
        #f = np.fmax(np.dot((Bm.T.dot(b)).T,F_tot).dot(np.linalg.inv(b.T.dot(b))),0)
#        import pdb
#        pdb.set_trace()
        f = np.fmax(np.linalg.inv(b.T.dot(b)).dot((Bm.T.dot(b)).T.dot(F_tot)),0)

    return A_tot,C_tot,YrA_tot,b,f,sn_tot, optional_outputs


