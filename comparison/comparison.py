""" compare how the elements behave
 
We create a folder ground truth that possess the same thing than the other in a form of a dictionnary containing nparrays and other info.
the other files contains every test and the name is the date of the test
 
See Also
------------
 
Link 
 
"""
#\package Caiman/self.comparison
#\image html X.jpg
#\version   1.0
#\copyright GNU General Public License v2.0
#\date Created on Tue Jun 30 21:01:17 2015
#\author: jeremie KALFON


import platform as plt
import datetime
import numpy as np
import os
import matplotlib.pyplot as pl
import json
import codecs
import glob 

import appJar as gui
import caiman as cm
import scipy
"""
   Comparison(object): class you instanciate to compare the different functions you are calling in your program.
 
   Here it has been made for 3 different functions. for it to compare well you need to set your 
   ground truth with the same computer with which you are comparing the files
 
    
 
    Attributes
    ----------
    self : object
        see the linked image
    sensitivity: inside
        the user to change it manualy
 
    Methods
    -------
    __init__()
        Initialize the function be instanciating a comparison object  
        
    save(istruth)
        save the comparison object on a file
   
    See Also
    --------
    
    .. image:: /Users/jeremie/CaImAn/dev/kalfon/img/datacomparison.png
    """
class Comparison(object):

    def __init__(self):
        
        self.comparison ={'rig_shifts': {},
             'pwrig_shifts': {},
             'cnmf_on_patch': {},
             'cnmf_full_frame': {},   
             'diff_neurons':None
        }
    
        self.comparison['rig_shifts']={
                           'ourdata': None,
                          'timer': None,
                          'sensitivity': 0.001    #the sensitivity USER TO CHOOSE
                         }
        #apparently pwrig shift are not used any more and the comparison are useless
        #self.comparison['pwrig_shifts']={
         #                 'ourdata': None,
          #                'timer': None,
           #               'sensitivity': 0.001
             #           }
        self.comparison['cnmf_on_patch']={
                          'ourdata': None,
                          'timer': None,
                          'sensitivity': 0.01
                         }
        self.comparison['cnmf_full_frame']={
                          'ourdata': None,
                          'timer': None,
                          'sensitivity': 0.01
                         }
    
        
    def save(self, istruth=False, params=None):
        """save the comparison object on a file
 
 
            depending on if we say this file will be ground truth or not, it wil be saved in either the tests or the groung truth folder
            if saved in test, a comparison to groundtruth will be add to the object 
            dededededede
            this comparison will be on 
                data : a normized difference of the normalized value of the arrays
                time : difference
            
 
            Parameters
            -----------
 
            self:  dictionnary
               the object of this class tha tcontains every value
            istruth: Boolean
                if we want it ot be the ground truth
             
            	See Also
            	---------
            
             
            	.. image:: CaImAn/dev/kalfon/img/datacomparison.png
            
             
                """
#\bug       
#\warning 

        
        #we get the informetion of the computer
        dt = datetime.datetime.today()
        dt=str(dt)
        plat=plt.platform()
        plat=str(plat)
        pro=plt.processor()
        pro=str(pro)
        
        
        
            
            
        print('we now only have lists')
        
        
        #we store a big file which is containing everything ( INFORMATION)
        self.information ={
                'platform': plat,
                'processor':pro,
                'values':self.comparison,
                'params': params
                }
        
        file_path="comparison/groundtruth/groundtruth.json"
        
        #if we want to set this data as truth
        if istruth:
                #we just save it
            if os._exists("comparison/groundtruth/groundtruth.json"):
               os.remove("comparison/groundtruth/groundtruth.json")
            else:
               print("nothing to remove")
               
            
            json.dump(self.information, codecs.open(file_path, 'w', encoding='utf-8'),
                      separators=(',', ':'), sort_keys=True, indent=4) ### this saves the array in .json format  
                      
            #np.savez('comparison/groundtruth/groundtruth.npz', **self.information)
            print('we now have ground truth')
        else:
            #if not we create a comparison first
            try: 
                data = codecs.open(file_path, 'r', encoding='utf-8').read()
                data = json.loads(data)
            
            #if we cannot manage to open it or it doesnt exist:
            except (IOError, OSError) :
                #we save but we explain why there were a problem
                print('we were not able to read the file to compare it')
                
                
                
                
                file_path="comparison/tests/NC"+dt+".json"
                json.dump(self.information, codecs.open(file_path, 'w', encoding='utf-8'),
                      separators=(',', ':'), sort_keys=True, indent=4) ### this saves the array in .json format
            
                return
                 
        
        #they need to be run on the same computer
            if data['processor']==self.information['processor']:
                #they need to have the same name
               if data['params']['fname']==self.information['params']['fname']:
               
            #if not we save the value of the difference into 
            
        #for neurons
        #           self.information['values']['diff_neurons'] = data['values']['neurons'] - self.comparison['neurons'] 
            
            
        #for rigid
                   init = data['values']['rig_shifts']['ourdata']
                    #we do this [()] because of REASONS
                   curr = self.comparison['rig_shifts']['ourdata']
                   diff = np.linalg.norm(np.asarray(init)-np.asarray(curr))/np.linalg.norm(init)
                   isdiff = diff < self.comparison['rig_shifts']['sensitivity']
                   self.information['values']['rig_shifts'].update({'isdifferent':int(isdiff),
                                          'diff_data': diff,
                                          'diff_timing': data['values']['rig_shifts']['timer']
                                          - self.comparison['rig_shifts']['timer']
                                        
                                })
        #for pwrigid
                   init= data['values']['pwrig_shifts']['ourdata'][0]
                   curr = self.comparison['pwrig_shifts']['ourdata'][0]
                   curr=np.asarray(curr)
                   init = np.asarray(init)
                   
                   # they can be different in size of second element  for both ( same)
                   numinit = init.shape
                   numinit = numinit[1]
                   numcurr = curr.shape
                   numcurr = numcurr[1]
                   if numinit > numcurr : 
                          i=0
                          currnew = np.zeros(init.shape)
                          while i < curr.shape[0]: 
                                  currnew[i] = np.concatenate((curr[i], np.zeros(numinit-numcurr)))
                                  
                                  i += 1
                          curr=currnew
                   else :
                       if numinit < numcurr :
                          i=0
                          initnew = np.zeros(curr.shape)
                          while i < init.shape[0]:
                                  initnew[i] = np.concatenate((init[i], np.zeros(numcurr-numinit)))
                                  i += 1
                          init=initnew
                    #a simple comparison algo
                
                   diff = np.linalg.norm(init-curr)/np.linalg.norm(init)
                  
                    
                   #there is xs and ys
                   init= data['values']['pwrig_shifts']['ourdata'][1]
                   curr = self.comparison['pwrig_shifts']['ourdata'][1]
                   
                   curr=np.asarray(curr)
                   init = np.asarray(init)
                   numinit = init.shape
                   numinit = numinit[1]
                   numcurr = curr.shape
                   numcurr = numcurr[1]
                   if numinit > numcurr : 
                          i=0
                          currnew = np.zeros(init.shape)
                          while i < curr.shape[0]: 
                                  currnew[i] = np.concatenate((curr[i], np.zeros(numinit-numcurr)))
                                  
                                  i += 1
                          curr=currnew
                   else :
                       if numinit < numcurr :
                          i=0
                          initnew = np.zeros(curr.shape)
                          while i < init.shape[0]:
                                  initnew[i] = np.concatenate((init[i], np.zeros(numcurr-numinit)))
                                  i += 1
                          init=initnew
                             
                   diffA = np.linalg.norm(init-curr)/np.linalg.norm(init)
                   #we add both errors
                   diff=diff+diffA
                   isdiff = diff < self.comparison['pwrig_shifts']['sensitivity']
                   self.information['values']['pwrig_shifts'].update({'isdifferent':int(isdiff),
                                            'diff_data': diff,
                                            'diff_timing': data['values']['pwrig_shifts']['timer']
                                            - self.comparison['pwrig_shifts']['timer']
                                        
                                })
        #for cnmf on patches 
                   init= data['values']['cnmf_on_patch']['ourdata'][0]
                   curr = self.comparison['cnmf_on_patch']['ourdata'][0]
                   curr=np.asarray(curr)
                   init = np.asarray(init)
                   numinit = init.shape
                   numinit = numinit[1]
                   numcurr = curr.shape
                   numcurr = numcurr[1]
                   if numinit > numcurr : 
                          i=0
                          currnew = np.zeros(init.shape)
                          while i < curr.shape[0]: 
                                  currnew[i] = np.concatenate((curr[i], np.zeros(numinit-numcurr)))
                                  
                                  i += 1
                          curr=currnew
                   else :
                       if numinit < numcurr :
                          i=0
                          initnew = np.zeros(curr.shape)
                          while i < init.shape[0]:
                                  initnew[i] = np.concatenate((init[i], np.zeros(numcurr-numinit)))
                                  i += 1
                          init=initnew
                   diffA = np.linalg.norm(init-curr)/np.linalg.norm(init)
                   
                   
                   #there is temporal and spatial
                   init= data['values']['cnmf_on_patch']['ourdata'][1]
                   curr = self.comparison['cnmf_on_patch']['ourdata'][1]
                   curr=np.asarray(curr)
                   init = np.asarray(init)
                   #here the problem can happen on the first elements
                   numinit = init.shape
                   numinit = numinit[0]
                   numcurr = curr.shape
                   numcurr = numcurr[0]
                   if numinit > numcurr : 
                       a = numinit-numcurr
                       b = curr.shape[1]
                       curr = np.concatenate((curr, np.zeros((a, b))))
                   if numinit < numcurr :
                       init = np.append(init, np.zeros((numcurr-numinit, init.shape[1])))
                   diff = np.linalg.norm(init-curr)/np.linalg.norm(init)
                   diff=diff+diffA
                   isdiff = diff < self.comparison['cnmf_on_patch']['sensitivity']
                   print(isdiff)
                   self.information['values']['cnmf_on_patch'].update({'isdifferent':int(isdiff),
                                            'diff_data': diff,
                                            'diff_timing': data['values']['cnmf_on_patch']['timer']
                                            - self.comparison['cnmf_on_patch']['timer']
                                        
                                })
        #for cnmf full frame
                   init= data['values']['cnmf_full_frame']['ourdata'][0]
                   curr = self.comparison['cnmf_full_frame']['ourdata'][0]
                   curr=np.asarray(curr)
                   init = np.asarray(init)
                   numinit = init.shape
                   numinit = numinit[1]
                   numcurr = curr.shape
                   numcurr = numcurr[1]
                   if numinit > numcurr : 
                          i=0
                          currnew = np.zeros(init.shape)
                          while i < curr.shape[0]: 
                                  currnew[i] = np.concatenate((curr[i], np.zeros(numinit-numcurr)))
                                  
                                  i += 1
                          curr=currnew
                   else :
                       if numinit < numcurr :
                          i=0
                          initnew = np.zeros(curr.shape)
                          while i < init.shape[0]:
                                  initnew[i] = np.concatenate((init[i], np.zeros(numcurr-numinit)))
                                  i += 1
                          init=initnew
                   diff = np.linalg.norm(init-curr)/np.linalg.norm(init)
                   
                   
                   #there is temporal and spatial
                   init= data['values']['cnmf_full_frame']['ourdata'][1]
                   curr = self.comparison['cnmf_full_frame']['ourdata'][1]
                   curr=np.asarray(curr)
                   init = np.asarray(init)
                   numinit = init.shape
                   numinit = numinit[0]
                   numcurr = curr.shape
                   numcurr = numcurr[0]
                   if numinit > numcurr : 
                       a = numinit-numcurr
                       b = curr.shape[1]
                       curr = np.concatenate((curr, np.zeros((a, b))))
                   if numinit < numcurr :
                       init = np.concatenate((init, np.zeros((numcurr-numinit, init.shape[1]))))
                   diffA = np.linalg.norm(init-curr)/np.linalg.norm(init)
                   #we add both errors
                   diff=diffA+diff
                   isdiff = diffA < self.comparison['cnmf_full_frame']['sensitivity']
                   
                   self.information['values']['cnmf_full_frame'].update({'isdifferent':int(isdiff),
                                            'diff_data': diff,
                                            'diff_timing': data['values']['cnmf_full_frame']['timer']
                                            - self.comparison['cnmf_full_frame']['timer']
                                        
                                })
                  
                #we save with the system date
                   file_path="comparison/tests/"+dt+".json"
                   json.dump(self.information, codecs.open(file_path, 'w', encoding='utf-8'),
                      separators=(',', ':'), sort_keys=True, indent=4) ### this saves the array in .json format
               else:
                   print('you need to use the same movie for ground truth')
            else:
                print("you need to set ground trut with your own computer")
            
            
            
            
            
            
            
    def save_with_compare(self, istruth=False, params=None, dview=None):
        
        
        
        dt = datetime.datetime.today()
        dt=str(dt)
        plat=plt.platform()
        plat=str(plat)
        pro=plt.processor()
        pro=str(pro)
        
        
        #actions on the sparse matrix
        
        
        
        
        #we store a big file which is containing everything ( INFORMATION)
        self.information ={
                'platform': plat,
                'processor':pro,
                'values':self.comparison,
                'params': params
                }
        
        file_path="comparison/groundtruth/groundtruth.json"
        
        #if we want to set this data as truth
        if istruth:
                #we just save it
            if os._exists("comparison/groundtruth/groundtruth.json"):
               os.remove("comparison/groundtruth/groundtruth.json")
            else:
               print("nothing to remove")
               
            
            json.dump(self.information, codecs.open(file_path, 'w', encoding='utf-8'),
                      separators=(',', ':'), sort_keys=True, indent=4) ### this saves the array in .json format  
                      
            #np.savez('comparison/groundtruth/groundtruth.npz', **self.information)
            print('we now have ground truth')
        else:
            #if not we create a comparison first
            try: 
                data = codecs.open("comparison/groundtruth/groundtruth.json", 'r', encoding='utf-8').read()
                data = json.loads(data)
            
            #if we cannot manage to open it or it doesnt exist:
            except (IOError, OSError) :
                #we save but we explain why there were a problem
                print('we were not able to read the file to compare it')
                sparsetolist(self)
                file_path="comparison/tests/NC"+dt+".json"
                json.dump(self.information, codecs.open(file_path, 'w', encoding='utf-8'),
                      separators=(',', ':'), sort_keys=True, indent=4) ### this saves the array in .json format
            
                return
                 
        
        #they need to be run on the same computer
            if data['processor']==self.information['processor']:
                #they need to have the same name
               if data['params']['fname']==self.information['params']['fname']:
               
            #if not we save the value of the difference into 
            
        #for neurons
        #           self.information['values']['diff_neurons'] = data['values']['neurons'] - self.comparison['neurons'] 
            
            
        #for rigid
                   init = data['values']['rig_shifts']['ourdata']
                    #we do this [()] because of REASONS
                   curr = self.comparison['rig_shifts']['ourdata']
                   diff = np.linalg.norm(np.asarray(init)-np.asarray(curr))/np.linalg.norm(init)
                   isdiff = diff < self.comparison['rig_shifts']['sensitivity']
                   self.information['values']['rig_shifts'].update({'isdifferent':int(isdiff),
                                          'diff_data': diff,
                                          'diff_timing': data['values']['rig_shifts']['timer']
                                          - self.comparison['rig_shifts']['timer']
                                        
                                })
            
        #for cnmf on patch
            #get the values
                  
                    #trial 
                   dims_gt = 60,80
                   dims_test = 60,80
                   #to change with the last computed size returned by the program
                   m = data['params']['gSig'][1]
                   min_size_neuro = (m*m/2) +m
                   print("min_size_neuro\n")
                   print(min_size_neuro)
                           #endt rial 
                   n_frames_per_bin = 10
                   A_gt= data['values']['cnmf_on_patch']['ourdata'][0] #A gt
                   A_test = self.comparison['cnmf_on_patch']['ourdata'][0] #A test
                   A_test=A_test.toarray() #coo sparse matrix
                   A_gt = np.asarray(A_gt)   # list matrix
                   C_gt= data['values']['cnmf_on_patch']['ourdata'][1] #C gt
                   C_test = self.comparison['cnmf_on_patch']['ourdata'][1] #C test
                   C_test=np.asarray(C_test)
                   C_gt = np.asarray(C_gt)
           #proceed to a trhreshold
                   A_test_thr = cm.source_extraction.cnmf.spatial.threshold_components(A_test, dims_test, medw=None, thr_method='max', maxthr=0.2, nrgthr=0.99, extract_cc=True,
                         se=None, ss=None, dview=dview) 
                   A_gt_thr = cm.source_extraction.cnmf.spatial.threshold_components(A_gt, dims_gt, medw=None, thr_method='max', maxthr=0.2, nrgthr=0.99, extract_cc=True,
                         se=None, ss=None, dview=dview) 
                   #compute C using this A thr
                   
                   
                   A_test_thr  = A_test_thr  > 0  
                   
                   size_neurons_test  = A_test_thr.sum(0)
                   print("size_neurons_test\n ")
                   print(size_neurons_test)
                   A_test_thr = A_test_thr[:,size_neurons_test>min_size_neuro]
                   print("A_test_thr.shape\n")
                   print(A_test_thr.shape)
                   C_test_frame = C_test.shape[1]
                   C_test_thr = C_test[size_neurons_test>min_size_neuro,:C_test_frame]
                   #same for gt
                   print("C_test_thr\n")
                   print(C_test_thr)
                   A_gt_thr  = A_gt_thr  > 0  
                   size_neurons_gt = A_gt_thr.sum(0)
                   A_gt_thr = A_gt_thr[:,size_neurons_gt>min_size_neuro]
                   C_gt_frame = C_gt.shape[1]
                   C_gt_thr = C_gt[size_neurons_gt>min_size_neuro,:C_gt_frame]
                   
                   #we would also like the difference in the number of neurons
                   self.comparison['diff_neurons'] = A_test_thr.shape[1] - A_gt_thr.shape[1] 
                   print(self.comparison['diff_neurons'])
                  
                   
                   C_test_thr = np.array([CC.reshape([-1,n_frames_per_bin]).max(1) for CC in C_test_thr])
                   C_gt_thr = np.array([CC.reshape([-1,n_frames_per_bin]).max(1) for CC in C_gt_thr])

                   idx_tp_gt,idx_tp_comp, idx_fn_gt, idx_fp_comp, performance_off_on =  cm.base.rois.nf_match_neurons_in_binary_masks(
                           A_gt_thr[:,:].reshape([dims_gt[0],dims_gt[1],-1],order = 'F').transpose([2,0,1])*1.
                           ,A_test_thr[:,:].reshape([dims_test[0],dims_test[1],-1],order = 'F').transpose([2,0,1])*1.)
                   
                   #the pearson's correlation coefficient of the two Calcium activities thresholded
                   #comparing Calcium activities of all the components that are defined by the matching algo as the same.
                   corrs = np.array([scipy.stats.pearsonr(C_gt_thr[gt,:],C_test_thr[comp,:])[0] for gt,comp in zip(idx_tp_gt,idx_tp_comp)])
                   
                   
                   isdiff = self.comparison['diff_neurons'] == 0
                   isdiff = isdiff and np.linalg.norm(corrs) < self.comparison['cnmf_on_patch']['sensitivity'] 
                       
                   self.information['values']['cnmf_on_patch'].update({'isdifferent':int(isdiff),
                                          'diff_data': {
                                                  
                                                  'performance':performance_off_on,
                                                  'corelations': corrs.tolist()
                                                    #performance = dict() 
                                                    #performance['recall'] = old_div(TP,(TP+FN))
                                                    #performance['precision'] = old_div(TP,(TP+FP)) 
                                                    #performance['accuracy'] = old_div((TP+TN),(TP+FP+FN+TN))
                                                    #performance['f1_score'] = 2*TP/(2*TP+FP+FN)
                                                    
                                                                                                  
                                                  
                                                  
                                                  
                                                  },
                                          'diff_timing': data['values']['cnmf_on_patch']['timer']
                                          - self.comparison['cnmf_on_patch']['timer']
                                        
                                })
            
                    
            #CNMF FULL FRAME
                           
                   n_frames_per_bin = 10
                   A_gt= data['values']['cnmf_full_frame']['ourdata'][0] #A gt
                   A_test = self.comparison['cnmf_full_frame']['ourdata'][0] #A test
                   A_test=A_test.toarray() #coo sparse matrix
                   A_gt = np.asarray(A_gt)   # list matrix
                   C_gt= data['values']['cnmf_full_frame']['ourdata'][1] #C gt
                   C_test = self.comparison['cnmf_full_frame']['ourdata'][1] #C test
                   C_test=np.asarray(C_test)
                   C_gt = np.asarray(C_gt)
           #proceed to a trhreshold
                   A_test_thr = cm.source_extraction.cnmf.spatial.threshold_components(A_test, dims_test, medw=None, thr_method='max', maxthr=0.2, nrgthr=0.99, extract_cc=True,
                         se=None, ss=None, dview=dview) 
                   A_gt_thr = cm.source_extraction.cnmf.spatial.threshold_components(A_gt, dims_gt, medw=None, thr_method='max', maxthr=0.2, nrgthr=0.99, extract_cc=True,
                         se=None, ss=None, dview=dview) 
                   #compute C using this A thr
                   A_test_thr  = A_test_thr  > 0  

                   size_neurons_test  = A_test_thr.sum(0)
                   A_test_thr = A_test_thr[:,size_neurons_test>min_size_neuro]
                   C_test_frame = C_test.shape[1]
                   C_test_thr = C_test[size_neurons_test>min_size_neuro,:C_test_frame]
                   #same for gt
                   A_gt_thr  = A_gt_thr  > 0 
                   print(A_test_thr.shape)
                   print("\n")
                   size_neurons_gt = A_gt_thr.sum(0)
                    
                   A_gt_thr = A_gt_thr[:,size_neurons_gt>min_size_neuro]
                   C_gt_frame = C_gt.shape[1]
                   C_gt_thr = C_gt[size_neurons_gt>min_size_neuro,:C_gt_frame]
                   
                   #we would also like the difference in the number of neurons
                   self.comparison['diff_neurons'] = A_test_thr.shape[1] - A_gt_thr.shape[1] 
                   
                   print(self.comparison['diff_neurons'])
                   
                   C_test_thr = np.array([CC.reshape([-1,n_frames_per_bin]).max(1) for CC in C_test_thr])
                   C_gt_thr = np.array([CC.reshape([-1,n_frames_per_bin]).max(1) for CC in C_gt_thr])


                   idx_tp_gt,idx_tp_comp, idx_fn_gt, idx_fp_comp, performance_off_on =  cm.base.rois.nf_match_neurons_in_binary_masks(
                           A_gt_thr[:,:].reshape([dims_gt[0],dims_gt[1],-1],order = 'F').transpose([2,0,1])*1.
                           ,A_test_thr[:,:].reshape([dims_test[0],dims_test[1],-1],order = 'F').transpose([2,0,1])*1.)
                   
                   #the pearson's correlation coefficient of the two Calcium activities thresholded
                   #comparing Calcium activities of all the components that are defined by the matching algo as the same.
                   corrs = np.array([scipy.stats.pearsonr(C_gt_thr[gt,:],C_test_thr[comp,:])[0] for gt,comp in zip(idx_tp_gt,idx_tp_comp)])
                   
                   
                   isdiff = self.comparison['diff_neurons'] == 0
                   isdiff = isdiff and np.linalg.norm(corrs) < self.comparison['cnmf_full_frame']['sensitivity'] 
                       
                   self.information['values']['cnmf_full_frame'].update({'isdifferent':int(isdiff),
                                          'diff_data': {
                                                  
                                                  'performance':performance_off_on,
                                                  'corelations':corrs.tolist()
                                                    #performance = dict() 
                                                    #performance['recall'] = old_div(TP,(TP+FN))
                                                    #performance['precision'] = old_div(TP,(TP+FP)) 
                                                    #performance['accuracy'] = old_div((TP+TN),(TP+FP+FN+TN))
                                                    #performance['f1_score'] = 2*TP/(2*TP+FP+FN)
                                                    
                                                                                                  
                                                  
                                                  
                                                  
                                                  },
                                          'diff_timing': data['values']['cnmf_on_patch']['timer']
                                          - self.comparison['cnmf_on_patch']['timer']
                                        
                                })
                  
        i=2
        for name in os.listdir('/Users/jeremie/CaImAn/comparison/tests'):
            i+=1
        sparsetolist(self)
        i=str(i)
        file_path="comparison/tests/"+i+".json"
        json.dump(self.information, codecs.open(file_path, 'w', encoding='utf-8'),
                      separators=(',', ':'), sort_keys=True, indent=4) ### this saves the array in .json format
        
        
        
        
                
    def plot(self):
        
        """save the comparison object on a file
 
 
            depending on if we say this file will be ground truth or not, it wil be saved in either the tests or the groung truth folder
            if saved in test, a comparison to groundtruth will be add to the object 
            this comparison will be on 
                data : a normized difference of the normalized value of the arrays
                time : difference
 
            Parameters
            -----------
 
            self:  dictionnary
               the object of this class tha tcontains every value
            istruth: Boolean
                if we want it ot be the ground truth
             
            	See Also
            	---------
            
             
            	.. image:: /Users/jeremie/CaImAn/dev/kalfon/img/datacomparison.png
            
             
                """
        dr='comparison/tests/'
        #what we wille plot
        self.rig_time= []
        self.rig_comp= []
        self.diffneur=[]
        self.on_patch_time=[]
        self.full_frame_time=[]
        self.on_patch_diffneur=[]
        self.full_frame_diffneur=[]
        self.on_patch_score=[]
        self.full_frame_score=[]
        self.on_patch_rec=[]
        self.full_frame_rec=[]
        self.on_patch_acc=[]
        self.full_frame_acc=[]
        self.on_patch_prec=[]
        self.full_frame_prec=[]
        self.numb=[]

        
        try:
                
                drect='/Users/jeremie/CaImAn/comparison/groundtruth/groundtruth.json'
                data = codecs.open(drect, 'r', encoding='utf-8').read()
                data = json.loads(data)
                
                                
        except:
           print('no groundtruth \n')
           return 
        i=0
        count1=0
        count2=0
        count3=0
        count4=0
        for name in glob.glob(dr+"*.json"):
                       
                            
                            print(name)
                            
                            try: 
                            
                                    n = codecs.open(name, 'r', encoding='utf-8').read()
                                    tfile = json.loads(n)
                            
                           
                                
                                
                            
                            
                            #if we are on similar proc
                                    if tfile['processor']==data['processor']:
                                        if tfile['params']['fname']==data['params']['fname']:
                                            val=tfile['values']
                                           # if val['rig_shifts']['isdifferent']==1:
                                            i+=1
                                            self.numb.append(int(name[17:-5]))
                                            self.totest = val['rig_shifts']['diff_timing']
                                            self.rig_time.append(val['rig_shifts']['diff_timing'])
                                            self.rig_comp.append(val['rig_shifts']['diff_data'])
                                            self.on_patch_time.append(val['cnmf_on_patch']['diff_timing'])
                                            self.full_frame_time.append(val['cnmf_full_frame']['diff_timing'])
                                            self.diffneur.append(val['diff_neurons'])
                                            
                                            perpatch = val['cnmf_on_patch']['diff_data']['performance']
                                            perframe=val['cnmf_full_frame']['diff_data']['performance']
                                            
                                            self.on_patch_score.append(perpatch['f1_score'])
                                            self.full_frame_score.append(perframe['f1_score'])
                                            self.on_patch_rec.append(perpatch['recall'])
                                            self.full_frame_rec.append(perframe['recall'])
                                            self.on_patch_acc.append(perpatch['accuracy'])
                                            self.full_frame_acc.append(perframe['accuracy'])
                                            self.on_patch_prec.append(perpatch['precision'])
                                            self.full_frame_prec.append(perframe['precision'])
                                           # else:
                                              #  count2+=1
                                                    
                                                
                                        else:
                                            count3+=1
                                    else:
                                        count4+=1
                         #if we cannot manage to open it or it doesnt exist:
                            except (IOError, OSError) :
                                #we save but we explain why there were a problem
                                print("pb")

                                count1+=1
        """
app = Gui()


       
app.showSplash('CaImAn : Comparison', fill='red', stripe='black', fg='white', font=44)
app.addButton("rigid motion correction comparison", press)
app.addButton("nb of neuron found comparison", press)
app.addButton("cnmf on patches comparison", press)
app.addButton("cnmf full frame comparison", press)
app.addButton("computing power comparison", press)
app.go()

        """
        #PLOTTING PART
        print('\n'+str(count3)+' file were not made using the same movie than ground truth')
        print('\n'+str(count4)+' file were not made using the same processor than ground truth')
        print('\n'+str(count2)+'file were not different than ground truth')     
        print('\n'+str(count1)+'file could not be read') 
        print('\n'+str(i)+' file were compared') 
        
        
        
        #self.x=np.arange(i)
        
        #for the motion correction correlation
        pl.figure(1)
        pl.subplot(2,1,1)
        pl.title('rig_motoin correction')
        pl.plot(self.numb,self.rig_comp,'bo',
                         label='motion correlation')
        
        pl.legend(loc='lower right')               
        
        
        pl.subplot(2,1,2)
        pl.plot(self.numb,self.rig_time,
                         label='computing comp (clock time difference)')
        pl.legend(loc='lower right')
       
        #timing and neuron
        pl.figure(2)
        pl.subplot(2,1,1)
        pl.title('difftiming')
        
        
        line1, = pl.plot(self.numb,self.rig_time,'bo',
                 label='rig shifts')
        line2, = pl.plot(self.numb,self.on_patch_time,'<',
                 label='cnmf on patch')
        line3, = pl.plot(self.numb,self.full_frame_time,'>',
                 label='cnmf full frame')
        
        pl.legend(loc='lower right')               
        
        
        pl.subplot(2,1,2)
       
        pl.title('diff neurons')
        pl.plot(self.numb,self.diffneur,':')
    
        pl.legend(loc='lower right')
        
        #precision and recall
        pl.figure(3)
        pl.subplot(2,1,1)
        pl.title('cnmf full frame')
        
        
        line1, = pl.plot(self.numb,self.full_frame_prec,'bo',
                 label='precision')
        line2, = pl.plot(self.numb,self.full_frame_rec,'<',
                 label='recall')
        line3, = pl.plot(self.numb,self.full_frame_score,'>',
                 label='f1')
        line4, = pl.plot(self.numb,self.full_frame_acc,'^',
                 label='accuracy')
        
        
        pl.legend(loc='lower right')               
        
        
        pl.subplot(2,1,2)
       
        pl.title('cnmf on patch')
        
        
        line1, = pl.plot(self.numb,self.on_patch_prec,'bo',
                 label='precision')
        line2, = pl.plot(self.numb,self.on_patch_rec,'<',
                 label='recall')
        line3, = pl.plot(self.numb,self.on_patch_score,'>',
                 label='f1')
        line4, = pl.plot(self.numb,self.on_patch_acc,'^',
                 label='accuracy')
        
        
        pl.legend(loc='lower right')
        
        pl.show()
        
        """
        fig, ax = pl.subplots(2,1,1)
        aX.title('rig_motoin correction')
        line1, = ax.plot(i,rig_comps , '--', linewidth=2,
                         label='Dashes set retroactively')
        
        
        line2, = ax.plot(x, -1 * np.sin(x),
                         label='Dashes set proactively')
                       
        
        ax.legend(loc='lower right')
        pl.subplot(2,1,2)
        
        pl.text(1,1,t,wrap=True)
        fig, ay = pl.subplots(2,1,3)
        pl.show()
        """
    
    
                
                
                
        
            
    def see(self,filename=None):
        
        if filename == None:
            dr='comparison/groundtruth/groundtruth.json'
        else:
            dr='comparison/tests/'
        
            dr=dr+filename+'.json'
            print(dr)
        try:
            
                data = codecs.open(dr, 'r', encoding='utf-8').read()
                data = json.loads(data)
                
                                
        except:
           print(' the name is not valid')
           return
           
        print('here is the info :\n')
        see_it(data)
        
        
           
  
def see_it(data=None):
            for key in data:
                
                val= data[key] 
                if isinstance(val, dict):
                    print('\n')
                    print(key)
                    print('\n')
                    see_it(val)
                else:
                    if not isinstance(val, list):
                        
                        print(key)
                        print(val)
                    
def press(self,btn):
        pl.close()
        if btn == "rigid motion correction comparison":
                    fig, ax = plt.subplots(2,1,1)
                    ax.title('rig_motoin correction')
                    line1, = ax.plot(self.x,self.rig_comps , '--', linewidth=2,
                                     label='motion correlation')
                    
                                   
                    
                    ax.legend(loc='lower right')
                    pl.subplot(2,1,2)
                    pl.plot(self.x,self.rig_times , '--', linewidth=2,
                                     label='computing comp (clock time difference)')
                    pl.show()
                    
            
def sparsetolist(self):
#actions on the sparse matrix
        A = self.information['values']['cnmf_full_frame']['ourdata'][0]
        if not isinstance(A, list):
            print(type(A))
            if not isinstance(A, np.ndarray):
                A=A.toarray()
            self.information['values']['cnmf_full_frame']['ourdata'][0] = A.tolist()
        
        A = self.information['values']['cnmf_full_frame']['ourdata'][1]
        if not isinstance(A, list):
            print(type(A))
            if not isinstance(A, np.ndarray):
                A=A.toarray()
            self.information['values']['cnmf_full_frame']['ourdata'][1] = A.tolist()
    
        A = self.information['values']['cnmf_on_patch']['ourdata'][0]
        if not isinstance(A, list):
            print(type(A))
            if not isinstance(A, np.ndarray):
                A=A.toarray()
            self.information['values']['cnmf_on_patch']['ourdata'][0] = A.tolist()
        """
        it=0
        for A in self.information['values']['pwrig_shifts']['ourdata']:
            ite=0
            for B in A:
                if not isinstance(B, list):
                    self.information['values']['pwrig_shifts']['ourdata'][it][ite] = B.tolist()
                ite+=1
            it+=1
            """
        
        A = self.information['values']['cnmf_on_patch']['ourdata'][1]
        if not isinstance(A, list):
            print(type(A))
            if not isinstance(A, np.ndarray):
                A=A.toarray()
            self.information['values']['cnmf_on_patch']['ourdata'][1] = A.tolist()      
            
        
        
            
            
                                    
                
            