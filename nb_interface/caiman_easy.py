from __future__ import print_function
import glob
import os
import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from IPython.display import HTML, display
from matplotlib.colors import cnames
from matplotlib import animation, rc
from scipy.sparse import coo_matrix
from past.utils import old_div
import caiman as cm

from caiman.source_extraction import cnmf
from caiman.motion_correction import MotionCorrect
from caiman.motion_correction import tile_and_correct, motion_correction_piecewise
from caiman import save_memmap_join
from caiman.mmapping import load_memmap

from caiman.utils.visualization import inspect_correlation_pnr
from caiman.components_evaluation import estimate_components_quality_auto
from caiman.motion_correction import motion_correct_oneP_rigid, motion_correct_oneP_nonrigid
from caiman.source_extraction.cnmf.utilities import detrend_df_f, detrend_df_f_auto
import scipy
import pylab as pl
import matplotlib as mpl
import matplotlib.cm as mcm
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
from caiman_easy_widgets import *
#import select_file as sf_
from typing import Dict, Tuple, List
import datetime
from sys import getsizeof
import threading

#from ..summary_images import local_correlations


'''
Interface Code Developed by Brandon Brown in the Khakh Lab at UCLA
"CaImAn" algorithms developed by Simons Foundation
Nov 2017
'''

#Import the context that will save information through the analysis pipeline
from caiman_context import *



#setup cluster
#@out.capture()()
def start_procs(n_processes=None):
	c, dview, n_processes = cm.cluster.setup_cluster(backend='local', n_processes=n_processes, single_thread=False)
	return c, dview, n_processes


#converts all files to tif for motion correction, unless they're already tif
'''def load_files(path,dsx=0.5,dsy=0.5,dst=1): #NOTE: This downsamples spatially
	files = glob.glob(fldr + '*.tif') + glob.glob(fldr + '*.tiff') + glob.glob(fldr + '*.avi')
	list_of_files = []
	for each_file in files:
		splitFile = os.path.splitext(each_file)
		tiff_file = each_file
		toTiff = cm.load(each_file)
		if splitFile[1] != '.tif' and splitFile[1] != '.tiff':
			tiff_file = splitFile[0] + "_dsx_" + dsx + "_dsy_" + dsy + "_dst_" + dst + '.tif'
			toTiff.save(tiff_file)
		tiff_file = splitFile[0] + "_dsx_" + dsx + "_dsy_" + dsy + "_dst_" + dst + '.tif'
		list_of_files.append(tiff_file)
		toTiff.resize(dsx,dsy,dst).save(tiff_file)
	return list_of_files'''

#@out.capture()
def load_files(fldr, print_values=False):
	if os.path.isfile(fldr):
		if print_values: print("Loaded: " + fldr)
		return [fldr]
	else:
		fldr = os.path.join(fldr, '')
		files = glob.glob(fldr + '*.tif') + glob.glob(fldr + '*.tiff') + glob.glob(fldr + '*.avi')
		files = sorted(files)
		if print_values:
			print("Loaded files:")
			for each_file in files:
				print(each_file)
		return files

#Normalize signal trace to be in range [0,1]
def normalize_signal(signal):
	#signal should be flat array, e.g. shape=(2000,)
	signal_copy = signal.copy()
	signal_copy = signal_copy + np.abs(signal_copy.min())  #make all values positive
	signal_copy = signal_copy / signal_copy.max() #normalize so max value is 1
	return signal_copy

#run motion correction
def run_mc(fnames, mc_params, dsfactors, rigid=True, batch=True, scope_type=2):
	min_mov = 0
	mc_list = []
	new_templ = None
	counter = 0
	tmp_files = [] #list of temporary files that this function may create, so we can delete later
	def resave_and_resize(each_file, dsfactors):
		toTiff = cm.load(each_file)
		tiff_file = each_file
		tiff_file = os.path.splitext(each_file)[0] + '_tmp.tif'
		if any(x < 1 for x in dsfactors):
			toTiff.resize(*dsfactors).save(tiff_file)
		else:
			toTiff.save(tiff_file)
		return tiff_file
	for each_file in fnames:
		#first convert AVI to TIFF and DOWNSAMPLE SPATIALLY
		tiff_file = each_file
		is_converted = False
		file_ext = os.path.splitext(each_file)[1]
		if (file_ext == '.avi' or any(x < 1 for x in dsfactors)):
			is_converted = True
			if file_ext != '.tif':
				print("Converting %s to TIF file format..." % (os.path.basename(tiff_file,)))
			if any(x < 1 for x in dsfactors):
				print("Downsampling by %s x, %s y, %s t (frames), resaving as temporary file" % dsfactors)
			tiff_file = resave_and_resize(each_file, dsfactors)
			tmp_files.append(tiff_file)
		#get min_mov
		if counter == 0:
			min_mov = np.array([cm.motion_correction.low_pass_filter_space(m_,mc_params['gSig_filt']) for m_ in cm.load(tiff_file, subindices=range(999))]).min()
			#min_mov = cm.load(tiff_file, subindices=range(400)).min()
			print("Min Mov: ", min_mov)
			print("Motion correcting: " + tiff_file)

	# TODO: needinfo how the classes works
		new_templ = None
		mc_mov = None
		bord_px_rig = None
		bord_px_els = None
		#setup new class object
		mc = MotionCorrect(tiff_file, min_mov, **mc_params)

		if rigid:
			print("Starting RIGID motion correction...")
			mc.motion_correct_rigid(save_movie=True, template = new_templ)
			'''mc = motion_correct_oneP_rigid([tiff_file], gSig_filt = mc_params['gSig_filt'], \
				splits_rig = mc_params['splits_rig'], dview = mc_params['dview'], max_shifts = mc_params['max_shifts'])'''
			new_templ = mc.total_template_rig
			mc_mov = cm.load(mc.fname_tot_rig)
			bord_px_rig = np.ceil(np.max(mc.shifts_rig)).astype(np.int)
		else:
			print("Starting NON-rigid motion correction...")
			#mc.motion_correct_pwrigid(save_movie=True, template=new_templ, show_template=False)
			if scope_type == 1:
				mc = motion_correct_oneP_nonrigid(tiff_file, **mc_params)
			else:
				mc.motion_correct_pwrigid(save_movei=True, template=new_templ)
			new_templ = mc.total_template_els
			mc_mov = cm.load(mc.fname_tot_els)
			bord_px_els = np.ceil(np.maximum(np.max(np.abs(mc.x_shifts_els)),
								 np.max(np.abs(mc.y_shifts_els)))).astype(np.int)

		# TODO : needinfo
		#pl.imshow(new_templ, cmap='gray')
		#pl.pause(.1)
		mc_list.append(mc)
		#remove generated TIFF, if applicable
		if is_converted:
			os.remove(tiff_file)
		counter += 1
	clean_up() #remove log files
	if batch:
		print("Batch mode, combining files")
		mode = 'rig' if rigid else 'nonrig'
		combined_file = combine_mc_mmaps(mc_list,mc_params['dview'],mode=mode)
		#delete individual files
		for old_f in mc_list:
			if rigid:
				try:
					print("Deleting: {}".format(old_f.fname_tot_rig))
					os.remove(old_f.fname_tot_rig[0])
				except PermissionError:
					print("PermissionError: Cannot delete temporary file. (Non-Fatal)")
			else:
				try:
					print("Deleting: {}".format(old_f.fname_tot_els))
					os.remove(old_f.fname_tot_els[0])
				except PermissionError:
					print("PermissionError: Cannot delete temporary file. (Non-Fatal)")
		return [combined_file]
	else:
		return mc_list

def flatten(l):
	return [item for sublist in l for item in sublist]

def combine_mc_mmaps(mc_list, dview, mode='rig'):
	if mode=='rig':
		mc_names = [i.fname_tot_rig for i in mc_list]
	else:
		mc_names = [i.fname_tot_els for i in mc_list]
	mc_names = flatten(mc_names)
	mc_mov_name = save_memmap_join(mc_names, base_name='mc_rig', dview=dview)
	print(mc_mov_name)
	return mc_mov_name

def resize_mov(Yr, fx=0.521, fy=0.3325):
	t,h,w = Yr.shape
	newshape=(int(w*fy),int(h*fx))
	mov=[]
	print(newshape)
	for frame in Yr:
		mov.append(cv2.resize(frame,newshape,fx=fx,fy=fy,interpolation=cv2.INTER_AREA))
	return np.asarray(mov)

#clean up tif files if originals were not already tif
def clean_up_files():
	pass

def cnmf_run(fname, cnmf_params): #fname is a full path, mmap file
	#SETTINGS
	#gSig = 4   # gaussian width of a 2D gaussian kernel, which approximates a neuron
	#gSiz = 12  # average diameter of a neuron
	#min_corr = 0.8 #0.8 default   ([0.65, 3] => 91 kept neurons; good specificity, poor sensitivity in periphery)
	#min_pnr = 1.1 #10 default
	# If True, the background can be roughly removed. This is useful when the background is strong.
	center_psf = True
	Yr, dims, T = cm.load_memmap(fname)
	print(dims, " ; ", T)
	Yr = Yr.T.reshape((T,) + dims, order='F')
	#configs
	cnm = cnmf.CNMF(**cnmf_params)
	cnm.fit(Yr)
	#get results
	A, C, b, f, YrA, sn, conv = cnm.A, cnm.C, cnm.b, cnm.f, cnm.YrA, cnm.sn, cnm.S
	print("(Sparse) Mem Size A: {0}, Mem Size C: {1}".format(getsizeof(A), getsizeof(C)))
	print("(Dense) Mem Size A: {0}, Mem Size C: {1}".format(np.asarray(A).nbytes, np.asarray(A).nbytes))
	#Let's normalize the Ca2+ signal traces
	for i in range(C.shape[0]): #for each trace
		C[i] = normalize_signal(C[i])
	idx_components = np.arange(A.shape[-1])
	clean_up() #remove log files
	return A, C, b, f, YrA, sn, idx_components, conv


def plot_contours(YrDT, cnmf_results, cn_filter):
	Yr, dims, T = YrDT
	Yr = np.rollaxis(np.reshape(Yr, dims + (T,), order='F'), 2)
	A, C, b, f, YrA, sn, idx_components, conv = cnmf_results
	pl.figure()
	crd = cm.utils.visualization.plot_contours(A.tocsc()[:, idx_components], cn_filter, thr=.9)
	'''
	#%%
	plt.imshow(A.sum(-1).reshape(dims, order='F'), vmax=200)
	#%%
	'''
	cm.utils.visualization.view_patches_bar(
		YrA, coo_matrix(A.tocsc()[:, idx_components]), C[idx_components],
		b, f, dims[0], dims[1], YrA=YrA[idx_components], img=cn_filter)

def filter_rois(YrDT, cnmf_results, dview, gSig, gSiz, fr=0.3, min_SNR=3, r_values_min=0.85,decay_time=0.4, cnn_thr=0.8):
	Yr, dims, T = YrDT
	A, C, b, f, YrA, sn, idx_components_orig, conv = cnmf_results
	#final_frate = 20 # approx final rate  (after eventual downsampling )
	Npeaks = 10
	traces = None
	#min_SNR = 3 # adaptive way to set threshold on the transient size
	#r_values_min = 0.85  # threshold on space consistency (if you lower more components will be accepted, potentially with worst quality)
	#decay_time = 0.4  #decay time of transients/indocator
	#gSig = 4   # gaussian width of a 2D gaussian kernel, which approximates a neuron
	#gSiz = 12  # average diameter of a neuron
	try:
		traces = C + YrA
	except ValueError:
		traces = C + YrA.T
	#        traces_a=traces-scipy.ndimage.percentile_filter(traces,8,size=[1,np.shape(traces)[-1]/5])
	#        traces_b=np.diff(traces,axis=1)
	Y = np.reshape(Yr, dims + (T,), order='F')
	print("Debugging (caiman_easy.py line 323 filter_rois): A.shape {0}, C.shape {1}, Y.shape {2}, Yr.shape {3}, idx_components_orig {4}".format(
		A.shape,C.shape,Y.shape,Yr.shape,idx_components_orig
		))
	'''fitness_raw, fitness_delta, erfc_raw, erfc_delta, r_values, significant_samples = cm.components_evaluation.evaluate_components(
						Y, traces, A, C, b, f, final_frate, remove_baseline=True, N=5, robust_std=False, Athresh=0.1, Npeaks=Npeaks,  thresh_C=0.3)
				'''	# %% DISCARD LOW QUALITY COMPONENTS
	idx_components, idx_components_bad, comp_SNR, r_values, pred_CNN = estimate_components_quality_auto(
							Y, A, C, b, f, YrA, fr,
							decay_time, gSig, dims, dview = dview,
							min_SNR=min_SNR, r_values_min = r_values_min, thresh_cnn_min=cnn_thr, min_std_reject = 0.5, use_cnn = False)
	'''	idx_components_r = np.where(r_values >= .5)[0]
	idx_components_raw = np.where(fitness_raw < -40)[0]
	idx_components_delta = np.where(fitness_delta < -20)[0]

	idx_components = np.union1d(idx_components_r, idx_components_raw)
	idx_components = np.union1d(idx_components, idx_components_delta)
	idx_components_bad = np.setdiff1d(list(range(len(traces))), idx_components)'''
	print(('Keeping ' + str(len(idx_components)) + \
		   ' and discarding  ' + str(len(idx_components_bad))))
	return idx_components, idx_components_bad


def corr_img(Yr, gSig, center_psf, plot=True):
	# show correlation image of the raw data; show correlation image and PNR image of the filtered data
	cn_raw = cm.summary_images.max_correlation_image(Yr, swap_dim=False, bin_size=3000) #default 3000
	#%% TAKES MEMORY!!!
	cn_filter, pnr = cm.summary_images.correlation_pnr(
		Yr, gSig=gSig, center_psf=center_psf, swap_dim=False)
	plot_ = None
	if plot:
		plot_ = plt.figure(figsize=(10, 5))
		#%%
		for i, (data, title) in enumerate(((Yr.mean(0), 'Mean image (raw)'),
										   (Yr.max(0), 'Max projection (raw)'),
										   (cn_raw[1:-1, 1:-1], 'Correlation (raw)'),
										   (cn_filter, 'Correlation (filtered)'),
										   (pnr, 'PNR (filtered)'),
										   (cn_filter * pnr, 'Correlation*PNR (filtered)'))):
			plt.subplot(2, 3, 1 + i)
			plt.imshow(data, cmap='jet', aspect='equal')
			plt.axis('off')
			plt.colorbar()
			plt.title(title)
	return cn_raw, cn_filter, pnr, plot_

def save_denoised_avi(data, dims, idx_components_keep, working_dir=""):
	A, C, b, f, YrA, sn, idx_components, conv = data
	idx_components = idx_components_keep
	x = None
	if type(A) != np.ndarray:
		x = cm.movie(A.tocsc()[:, idx_components].dot(C[idx_components, :])).reshape(dims + (-1,), order='F').transpose([2, 0, 1])
	else:
		x = cm.movie(A[:, idx_components].dot(C[idx_components, :])).reshape(dims + (-1,), order='F').transpose([2, 0, 1])
	currentDT = datetime.datetime.now()
	ts_ = currentDT.strftime("%Y%m%d_%H_%M_%S")
	avi_save_path = working_dir + "denoised_" + ts_ + ".avi"
	x.save(avi_save_path)
	print("Saved denoised AVI movie to: {}".format(avi_save_path))
	return avi_save_path

def load_data():
	#Need to import contour plot somehow
	#np.load()...
	pass

#Use Matplotlib's animation ability to play embedded HTML5 video of Numpy array
#Need FFMPEG installed: brew install ffmpeg
def play_movie(movie, interval=50, blit=True, cmap='gist_gray', vmin=None, vmax=None):
	frames = movie.shape[0]
	fig = plt.figure()
	im = plt.imshow(movie[0,:,:], cmap=cmap, vmin=vmin, vmax=vmax, animated=True)
	def updatefig(i):
		im.set_array(movie[i,:,:])
		return im,
	anim = animation.FuncAnimation(fig, updatefig, frames=frames, interval=50, blit=blit)
	return HTML(anim.to_html5_video())

def clean_up(stop_server=False):
	if stop_server: cm.stop_server()
	log_files = glob.glob('*_LOG_*')
	for log_file in log_files:
		os.remove(log_file)
