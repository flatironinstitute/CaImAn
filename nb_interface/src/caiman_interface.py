
import bqplot
from bqplot import (
	LogScale, LinearScale, OrdinalColorScale, ColorAxis,
	Axis, Scatter, Lines, CATEGORY10, Label, Figure, Tooltip, Toolbar
)
from functools import partial #partial function application
import traitlets
from sklearn.preprocessing import scale
import pandas as pd
import io
import glob
import os
import numpy as np
from IPython.display import HTML
from caiman_easy import *
from cnmf_results_widgets import *
from cnmf_results_plots import *
from mc_widgets import *
from cnmf_results_logic import *
#from event_logic import *

'''
Interface Code Developed by Brandon Brown in the Khakh Lab at UCLA
"CaImAn" package developed by Simons Foundation
Nov 2017
'''

#create context
context = Context(start_procs(backend='local',n_processes=None))

def parse_output(output): #input tupel of dicts
	out_ = ''
	for item in output:
		if 'traceback' in item:
			out_ += str(item['evalue'])
		elif 'text' in item:
			out_ += str(item['text'])
		else:
			out_ += '...'
	return out_

def out_on_change(_):
	'''if 'traceback' in out.outputs[-1]:
		log_widget.value = out.outputs[-1]['traceback']
	elif 'text' in out.outputs[-1]:
		log_widget.value = out.outputs[-1]['text']
	else:
		log_widget.value = '...'
	out.clear_output()'''
	log_widget.value = parse_output(out.outputs)#parse_output(out.outputs)

#out = widgets.Output(layout={'border': '1px solid black'})  #moved to caiman_easy.py
#out.observe(out_on_change)

def update_status(status, output=None):
	if output is None:
		status_bar_widget.value = status
	else:
		new_status = status + '<br />Output: ' + output
		status_bar_widget.value = new_status
	#out_on_change(None)
#motion correction interface

#check if file exists and is valid extension
def check_file(fpath, expected_ext=None): #expected_ext should be list
	path_ = pathlib.Path(fpath)
	if not path_.is_file():
		#print("Not a file.")
		return False
	if expected_ext is not None and path_.suffix not in expected_ext:
		print("Invalid file extension.")
		return False
	return True

#see if directory is valid and if it contains files of the right type [tif, tiff, avi]
def check_dir(fldr):
	fldr = os.path.join(fldr, '')
	path_ = pathlib.Path(fldr)
	if not path_.is_dir():
		print("Error: Directory not found.")
		return False
	files = glob.glob(fldr + '*.tif') + glob.glob(fldr + '*.tiff') + glob.glob(fldr + '*.avi')
	if len(files) == 0:
		print("Error: Directory contains no valid files.")
		return False
	return True

def check_file_or_dir(fpath):
	if not check_file(fpath, ['.tif','.tiff','.avi']) and not check_dir(fpath):
		print("Error: Not a valid file or directory.")
		return False
	else:
		return True

#@out.capture()
def load_context_event(_):
	cluster = [context.c, context.dview, context.n_processes]
	#check if valid file
	ctxf_ = str(context_path_txt.value)
	ctxf = pathlib.Path(ctxf_)
	if not ctxf.is_file() or ctxf.suffix != '.pkl':
		update_status("Invalid context.")
		return None
	context.load(ctxf_, cluster)
	context.dview = None
	update_status("Loaded context.")


def save_context_event(_):
	wkdir_ = workingdir_selector.value
	if not pathlib.Path(wkdir_).is_dir():
		update_status("Invalid working directory. Context not saved.")
		return None
	currentDT = datetime.datetime.now()
	ts_ = currentDT.strftime("%Y%m%d_%H_%M_%S")
	c_loc = os.path.join(wkdir_,'') + "context_" + ts_
	context.save(c_loc)
	update_status("Context saved to working directory.")

def context_browser_click(change):
	context_path_txt.value = change['new'][0]

def wkdir_browser_click(change):
	workingdir_selector.value = change['new'][0]

context_load_btn.on_click(load_context_event)
context_save_btn.on_click(save_context_event)
context_browser_btn.observe(context_browser_click)
wkdir_browser_btn.observe(wkdir_browser_click)


def rigid_btn_click(change):
	disabled = False
	if change['new'] == 'Rigid':
		disabled = True
	mc_strides_widget.disabled = disabled
	overlaps_widget.disabled = disabled
	upsample_factor_grid_widget.disabled = disabled
	max_deviation_rigid_widget.disabled = disabled

is_rigid_widget.observe(rigid_btn_click, names='value')

'''wkdir_box = widgets.HBox()
wkdir_box.children = [widgets.Label("Set Working Directory:"), workingdir_selector, workingdir_btn]
context_box = widgets.HBox()
context_box.children = [widgets.Label("Load context from:"),context_path_txt, context_load_btn, context_save_btn]
wkdir_context_box = widgets.VBox()
wkdir_context_box.children = [wkdir_box,context_box]'''


#Get file paths for *.tif and *.avi files, load into the context object
@out.capture()
#loads tif/avis for motion correction
def load_f(x):
	fpath = file_selector.value
	if check_file_or_dir(fpath):
		context.working_mc_files = load_raw_files(fpath, print_values=True)
		update_status("Loaded files for motion correction.")
	else:
		update_status("Error: Directory not found or no valid files found.")

#Load Files button click handler
load_files_btn.on_click(load_f)

def plot_shifts(mc_results, is_rigid=True):
	try:
		if is_rigid:
			shifts = np.array(mc_results[0].shifts_rig)
			x_shifts_x = np.arange(shifts.shape[0])
			y_shifts_x = x_shifts_x
			x_shifts_y = shifts[:,0]
			y_shifts_y = shifts[:,1]
			#update plot marks
			xshifts_line.x = x_shifts_x
			xshifts_line.y = x_shifts_y
			yshifts_line.x = y_shifts_x
			yshifts_line.y = y_shifts_y
			#add marks to bqplot Figure
			shifts_plot.marks=[xshifts_line, yshifts_line]
			#update legend
			ax_x.label = 'Time / Frames'
			ax_y.label = 'Shift'
		else: #non-rigid
			#'x_shifts_els',
 			#'y_shifts_els'
			#shifts = np.array(mc_results[0].shifts_nonrig)
			#x_shifts_x = np.arange(shifts.shape[0])
			x_shifts_mean = np.array([np.mean(x) for x in context.mc_nonrig[0].x_shifts_els])
			x_shifts_var = np.array([np.var(x) for x in context.mc_nonrig[0].x_shifts_els])
			y_shifts_mean = np.array([np.mean(y) for y in context.mc_nonrig[0].y_shifts_els])
			y_shifts_var = np.array([np.var(y) for y in context.mc_nonrig[0].y_shifts_els])
			xshifts_scat.x = x_shifts_mean
			xshifts_scat.y = x_shifts_var
			yshifts_scat.x = y_shifts_mean
			yshifts_scat.y = y_shifts_var
			#add marks to bqplot Figure
			shifts_plot.marks=[xshifts_scat, yshifts_scat]
			#update legend
			ax_x.label = 'Patch Shifts Mean'
			ax_y.label = 'Patch Shifts Variance'
		# Un-HIDE (display) plot
		mc_shifts_box.layout.display = ''
	except Exception as e:
		print("Error occurred when building shifts plot. \n {}".format(e))

@out.capture()
def run_mc_ui(_):
	#make sure user has loaded files
	if len(context.working_mc_files) == 0 or context.working_mc_files is None:
		update_status("Error: Files Not Loaded")
		print("You must click the Load Files button to load files before running motion correction.")
		return None
	#update status bar
	update_status("Running motion correction...")
	out.outputs = ()
	#out.clear_output()
	#get settings:
	scope_type = int(microscopy_type_widget.value)
	is_batch = True if is_batch_widget.value == 'Group' else False
	is_rigid = True if is_rigid_widget.value == 'Rigid' else False
	ms_ = int(max_shifts_widget.value)
	niter_rig_ = int(niter_rig_widget.value)
	splits_ = int(splits_widget.value)
	strides_ = int(mc_strides_widget.value)
	overlaps_ = int(overlaps_widget.value)
	upsample_factor_ = int(upsample_factor_grid_widget.value)
	max_dev_ = int(max_deviation_rigid_widget.value)
	mc_params = {
		'dview': context.dview, #refers to ipyparallel object for parallelism
		'max_shifts':(ms_, ms_),  # maximum allow rigid shift; default (6,6)
		'niter_rig':niter_rig_,
		'splits_rig':splits_,
		'num_splits_to_process_rig':None,
		'strides':(strides_,strides_), #default 48,48
		'overlaps':(overlaps_,overlaps_), #default 12,12
		'splits_els':splits_,
		'num_splits_to_process_els':[14, None],
		'upsample_factor_grid':upsample_factor_,
		'max_deviation_rigid':max_dev_,
		'shifts_opencv':True,
		'nonneg_movie':True,
		'gSig_filt' : [int(gSigFilter_widget.value)] * 2, #default 9,9  best 6,6,
	}
	#call run_mc
	#< run_mc(fnames, mc_params, rigid=True, batch=True) > returns list of mmap file names
	dsfactors = (float(dsx_widget.value),float(dsy_widget.value),float(dst_widget.value)) #or (1,1,1)   (ds x, ds y, ds t)
	context.mc_dsfactors = dsfactors
	mc_results, mmap_files = run_mc(context.working_mc_files, mc_params, dsfactors, rigid=is_rigid, batch=is_batch)
	#combined_file is None if not batch mode
	if is_rigid:
		context.mc_rig = mc_results
	else:
		context.mc_nonrig = mc_results
		context.border_pix = np.ceil(np.maximum(np.max(np.abs(mc_results[0].x_shifts_els)), \
			np.max(np.abs(mc_results[0].y_shifts_els)))).astype(np.int)
	context.mc_mmaps = mmap_files

	if is_batch:
		update_status("Motion Correction DONE!", str(mmap_files[0]))
	else:
		#mc_mov_name = str([x.fname_tot_rig[0] for x in mc_results]) if len(context.mc_rig) > 0 else str([x.fname_tot_els[0] for x in mc_results])
		mc_mov_name = ''
		for n in mmap_files:
			mc_mov_name += n + '<br />'
		update_status("Motion Correction DONE!", mc_mov_name)
	cnmf_file_selector.value = str(mmap_files[0])
	if len(mc_results) > 0: #TODO: If multiple MC results (i.e. run in independent mode, non-batch) need to show all
		plot_shifts(mc_results, is_rigid)
	#print("Output file(s): ")
	#[print(x) for x in mc_results]

run_mc_btn.on_click(run_mc_ui)

def mc_file_browser_click(change):
	file_selector.value = change['new'][0]

mc_file_browser_btn.observe(mc_file_browser_click)
'''major_col = widgets.VBox()
major_col.children = [file_box,is_batch_widget,settings, run_mc_btn]'''

#for after motion correction
#@out.capture()
def show_movies(_):
	update_status("Launching movie")
	gain = int(mc_mov_gain_widget.value)
	mag = int(mc_mov_mag_widget.value)
	orig_mov = cm.load(context.working_mc_files)
	orig_mov = orig_mov.resize(*context.mc_dsfactors)
	fr_ = float(fr_widget.value)
	orig_mov.fr = fr_
	mc_mov_ = context.mc_mmaps[0]
	#mc_mov_ = context.mc_rig[0].fname_tot_rig[0] if len(context.mc_rig) > 0 else context.mc_nonrig[0].fname_tot_els[0]
	mc_mov = cm.load(mc_mov_)
	mc_mov.fr = fr_
	offset_mov = -np.min(orig_mov[:100])  # make the dataset mostly non-negative
	cm.concatenate([orig_mov+offset_mov, mc_mov],
               axis=2).play(fr=60, offset=0, gain=gain, magnification=mag)  # press q to exit
			   #.play(fr=60, gain=15, magnification=2, offset=0)
	#orig_mov.play()
	#mc_mov.play()
	update_status("Idle")

#Load's memmap file into context
def cnmf_load_file_btn_click(_):
	fname = cnmf_file_selector.value
	if check_file(fname, ['.mmap']):
		context.working_cnmf_file = load_mmap_files(fname, print_values=True)[0]
		update_status("Loaded File for CNMF: {}".format(fname))
	else:
		update_status("Error: File not found or invalid file type.")

cnmf_load_file_btn.on_click(cnmf_load_file_btn_click)

#event when file browser button is clicked
def cnmf_file_browser_click(change):
	cnmf_file_selector.value = change['new'][0]

cnmf_file_browser_btn.observe(cnmf_file_browser_click)

def show_cor_plot(_):
	# load memory mappable file
	#Yr, dims, T = cm.load_memmap(fname_new)
	#Y = Yr.T.reshape((T,) + dims, order='F')
	if context.working_cnmf_file in [None, [], '']:
		update_status("Error: Must press Load Files button first.")
		print("Error: Must press Load Files button first.")
		return None
	update_status("Launching correlation plot...")
	gSig = int(gSig_widget.value)
	mc_mov = cm.load(context.working_cnmf_file)
	# compute some summary images (correlation and peak to noise)
	cn_filter, pnr = cm.summary_images.correlation_pnr(mc_mov, gSig=gSig, swap_dim=False) # change swap dim if output looks weird, it is a problem with tiffile
	# inspect the summary images and set the parameters
	inspect_correlation_pnr(cn_filter,pnr)
	update_status("Idle")

#after motion correction
play_mov_btn.on_click(show_movies)
correlation_plot_btn.on_click(show_cor_plot)

def patches_on_value_change(change):
	if change['new'] == 'Patches':
		k_widget.value = 4
		stride_widget.disabled = False
		rf_widget.disabled = False
	else:
		k_widget.value = 100
		stride_widget.disabled = True
		rf_widget.disabled = True

is_patches_widget.observe(patches_on_value_change, names='value')

@out.capture()
def run_cnmf_ui(_):
	update_status("Running CNMF...")
	out.outputs = ()
	if context.working_cnmf_file in [None, [], '']:
		update_status("Error: Must press Load Files button before running CNMF.")
		print("Error: Must press Load Files button before running CNMF.")
		return None
	#main settings
	fr_ = float(fr_widget.value)
	decay_time_ = float(decay_time_widget.value)
	#get settings:
	ds_spatial = int(1.0 / float(ds_spatial_widget.value))
	ds_temporal = int(1.0 / float(ds_temporal_widget.value))
	min_corr = float(min_corr_widget.value)
	min_pnr = float(min_pnr_widget.value)
	is_patches = True if is_patches_widget.value == 'Patches' else False
	K = int(k_widget.value)
	gSig = (int(gSig_widget.value),) * 2
	gSiz = (int(gSiz_widget.value),) * 2
	stride_ = int(stride_widget.value)
	rf_ = int(rf_widget.value)
	#advanced settings
	p_ = int(p_widget.value)
	gnb_ = int(gnb_widget.value)
	merge_thr_ = float(merge_thresh_widget.value)
	min_snr_ = float(min_snr_widget.value)
	is_dendrites_ = bool(is_dendrites_widget.value)
	rval_thr_ = float(rval_thr_widget.value)
	cnn_thr_ = float(cnn_thr_widget.value)

	cnmf_params = {
		'n_processes':context.n_processes,
		'method_init':'corr_pnr',
		'k':K,
		'gSig':gSig,
		'gSiz':gSiz,
		'merge_thresh':merge_thr_,
		'rval_thr':rval_thr_,
		'p':p_,
		'dview':context.dview,
		'tsub':1 if is_patches else ds_temporal, # x if not patches else 1 #THIS IS INTEGER NOT FLOAT
		'ssub':1 if is_patches else ds_spatial,
		'p_ssub': ds_spatial if is_patches else None,  #THIS IS INTEGER NOT FLOAT
		'p_tsub': ds_temporal if is_patches else None,
		'Ain':None,
		'rf': rf_ if is_patches else None, #enables patches;
		'stride': stride_ if is_patches else None,
		'only_init_patch': False,
		'gnb':gnb_,
		'nb_patch':gnb_, #number of background components per patch
		'method_deconvolution':'oasis',
		'low_rank_background': True,
		'update_background_components': False,
		'min_corr':min_corr,
		'min_pnr':min_pnr,
		'normalize_init': False,
		#'deconvolve_options_init': None,
		'ring_size_factor':1.5,
		'center_psf': True,
		'deconv_flag': bool(deconv_flag_widget.value),
		'simultaneously': False,
		'del_duplicates':True,
		'border_pix':context.border_pix,
	}
	#save params to context
	context.cnmf_params = cnmf_params
	#RUN CNMF-E
	#get original movie as mmap
	filename=os.path.split(context.working_cnmf_file)[-1]
	# =
	Yr, dims, T = load_memmap(os.path.join(os.path.split(context.working_cnmf_file)[0],filename))
	#bord_px_els = np.ceil(np.maximum(np.max(np.abs(mc.x_shifts_els)),
                                 #np.max(np.abs(mc.y_shifts_els)))).astype(np.int)
	#get correlation image
	context.YrDT = Yr, dims, T
	print("Starting CNMF-E...")
	print("Using patches") if is_patches else print("Single FOV")
	print("Deconvolution: ON") if bool(deconv_flag_widget.value) else print("Deconvolution: OFF")
	A, C, b, f, YrA, sn, idx_components, conv = cnmf_run(context.working_cnmf_file, cnmf_params)
	print("Debugging (caiman_interface.py line 397 filter_rois): A.shape {0}, C.shape {1}, Yr.shape {2}, idx_components_orig {3}".format(A.shape,C.shape,Yr.shape,idx_components))
	print("{}".format(type(A)))
	'''    if not is_patches: #for some reason, need to convert to ndarray if doing Single FOV;
	A = np.asarray(A) #make sure A is ndarray not matrix
	C = np.asarray(C) #make sure C is ndarray not matrix'''
	print("{}".format(type(A)))
	context.cnmf_results = A, C, b, f, YrA, sn, idx_components, conv
	#print("CNMF-E FINISHED!")
	#update_status("CNMF Finished")
	#results: A, C, b, f, YrA, sn, idx_components, S
	refine_results = bool(refine_components_widget.value) #automatically refine results
	save_movie_bool = bool(save_movie_widget.value)
	if refine_results:
		update_status("Automatically refining results...")
		context.idx_components_keep, context.idx_components_toss = \
			filter_rois(context.YrDT, context.cnmf_results, context.dview, gSig, gSiz, fr=fr_, \
				min_SNR=min_snr_, r_values_min=rval_thr_,decay_time=decay_time_, cnn_thr=cnn_thr_)
	else:
		update_status("Skipping automatic results refinement...")
		context.idx_components_keep = context.cnmf_results[6]
	#def corr_img(Yr, gSig, center_psr :bool):
	#save denoised movie:
	fn = ''
	if save_movie_bool:
		update_status("Saving denoised movie as .avi")
		fn = save_denoised_avi(context.cnmf_results, dims, context.idx_components_keep, context.working_dir)
		update_status("CNMF Finished", fn)
	else:
		update_status("CNMF Finished")
run_cnmf_btn.on_click(run_cnmf_ui)
major_cnmf_col = widgets.VBox()
major_cnmf_col.children = [cnmf_file_box, cnmf_settings, run_cnmf_btn]

# ---------------------

# view cnmf results interface

# ---------------------



def is_edit_changed(changed):
	if changed['new'] == 'View':
		fig4.layout.display = ''
		fig3.layout.display = ''
		fig2.layout.display = ''
		edit_panel_widget.layout.display = 'None'
		fig.layout.width = '67%'
	else: #Edit mode
		fig4.layout.display = 'None'
		fig3.layout.display = 'None'
		fig2.layout.display = 'None'
		edit_panel_widget.layout.display = ''

def toggle_deconv(change):
	if deconv_chk.value == 'Deconvolution':
		deconv_signal_mark.visible = True
		signal_mark.visible = False
	elif deconv_chk.value == 'Both':
		deconv_signal_mark.visible = True
		signal_mark.visible = True
	else:
		deconv_signal_mark.visible = False
		signal_mark.visible = True

def download_data_func(_):
	try:
		A, C, b, f, YrA, sn, idx_components, conv = load_context_data(context)
	except Exception as e:
		update_status("Error: Unable to load data.")
		return None
	adj_c = C[context.idx_components_keep,:]
	adj_s = conv
	currentDT2 = datetime.datetime.now()
	ts2_ = currentDT2.strftime("%Y%m%d_%H_%M_%S")
	metadata_ = ts2_
	deld_rois_ = []#list(delete_list_widget.value)
	wkdir_ = os.path.join(workingdir_selector.value, '') #adds ending slash if not present
	print("Excluding ROIs: %s" % (deld_rois_))
	def save_traces():
		nonlocal adj_c
		if dff_chk.value == True:
			#adj_c = detrend_df_f(A, b, C, f, YrA = YrA.T)
			#adj_c = detrend_df_f(A, b, C, f)
			adj_c = detrend_df_f_auto(A, b, C, f)
			#metadata_ += '_dFF'
			print("Using dF/F values")
		traces_path = wkdir_ + 'traces_' + metadata_ + '_dFF' + '.csv'

		df = pd.DataFrame(data=adj_c)
		df.index += 1
		deld_rois = list(map(lambda x: x-1, deld_rois_)) #remove ROIs that the user wants to exclude
		df.drop(df.index[deld_rois], inplace=True)

		df.to_csv(traces_path, header=False)
		print("Ca2+ Signal Traces saved to: %s" % (traces_path))

	def save_deconv():
		if conv is None:
			print("No deconvolution data found.")
			return None
		deconv_path = wkdir_ + 'deconv_' + metadata_ + '.csv'

		df = pd.DataFrame(data=adj_s)
		df.index += 1
		deld_rois = list(map(lambda x: x-1, deld_rois_)) #remove ROIs that the user wants to exclude
		df.drop(df.index[deld_rois], inplace=True)

		df.to_csv(deconv_path, header=False)
		print("Deconvolution data saved to: %s" % (deconv_path))

	if deconv_chk.value == 'Deconvolution':
		save_deconv()
		update_status("Deconvolution data saved to working directory.")
	elif deconv_chk.value == 'Both':
		save_traces()
		save_deconv()
		update_status("Traces and Deconvolution data saved to working directory.")
	else:
		save_traces()
		update_status("Traces saved to working directory.")


#@out.capture()()
def gen_image_data(image_np_array, name=""):
	with io.BytesIO() as img_file:
		plt.imsave(img_file, image_np_array, format='PNG')
		data = bytes(img_file.getbuffer())
		#print("{0} : {1}".format(name, len(data)))
		return data

def get_contour_coords(index, contours, dims):
	x_ = [x['coordinates'][:,0] for x in contours][index]
	y_ = dims[0]-[y['coordinates'][:,1] for y in contours][index]
	return x_,y_

def get_roi_image(A, index, dims):
	img = A[:,index].reshape(dims[1],dims[0]).T
	#print("roi_img dtype: {}".format(img.dtype))
	return gen_image_data(img, "Selected ROI")

#update plotted signal trace
def get_signal(C, index, conv=None):
	if conv is None:
		return C[index], np.repeat(0,len(C[index]))
	else:
		return C[index], conv[index]

def slider_change(change):
	global contours
	A, C, b, f, YrA, sn, idx_components, conv = load_context_data(context)
	if type(A) != np.ndarray: #probably sparse array, need to convert to dense array in order to reshape
		A = A.toarray()
	dims = context.YrDT[1]
	idx_components_keep = context.idx_components_keep
	A_ = A[:, idx_components_keep]
	C_ = C[idx_components_keep, :]
	contours_ = [contours[i] for i in idx_components_keep]

	contour_mark.x,contour_mark.y = get_contour_coords(change-1, contours_, dims)
	roi_image_mark.image = widgets.Image(value=get_roi_image(A_,(change-1),dims))
	deconv = True if conv is not None else False
	new_signal = get_signal(C_, change-1, conv)
	signal_mark.y = new_signal[0]
	new_signal_max = new_signal[0].max()
	if new_signal[1] is not None:
		deconv_signal_mark.y = new_signal[1]
	scale_y4.max = new_signal_max + 0.10*new_signal_max
	return [change-1]


def update_plots(A, C, dims, conv, contours):
	#A = A[:,idx_components_keep]
	#C = C[idx_components_keep,:]
	if type(A) != np.ndarray: #probably sparse array, need to convert to dense array in order to reshape
		A = A.toarray()
	#get ROI contours
	#contours = cm.utils.visualization.get_contours(A, (dims[0],dims[1]))
	centers = np.array([x['CoM'] for x in contours])
	centers = centers.T
	# isolate all ROIs (background substracted)
	a_image_np = np.mean(A.reshape(dims[1], dims[0], A.shape[1]), axis=2).T
	a_image_np = scale( a_image_np, axis=1, with_mean=False, with_std=True, copy=True )
	a_image.value = gen_image_data(a_image_np, "All ROIs")#a_data
	a_image.width = dims[1]
	a_image.height = dims[0]
	# show individual ROI
	roi_image.value = get_roi_image(A,1,dims)
	roi_image.width = dims[1]
	roi_image.height = dims[0]
	# Update ROI dots
	rois.x = centers[1]
	rois.y = (dims[0] - centers[0])
	# update slider
	roi_slider.max = A.shape[1]
	# update contours

	contour_x,contour_y = get_contour_coords(0, contours, dims)

	contour_mark.x = contour_x
	contour_mark.y = contour_y

	####
	'''	def slider_change_new(change):
		#error: Contours #: 13 ; len(A): 13; (change-1): 13
		#print("Contours #: {0} ; len(A): {1}; (change-1): {2}".format(len(contours), A.shape[1], (change-1)))
		contour_mark.x,contour_mark.y = get_contour_coords(change-1, contours, dims)
		roi_image_mark.image = widgets.Image(value=get_roi_image(A,(change-1),dims))
		#deconv = True if conv is not None else False
		new_signal = get_signal(C, change-1, conv)
		signal_mark.y = new_signal[0]
		new_signal_max = new_signal[0].max()
		if new_signal[1] is not None:
			deconv_signal_mark.y = new_signal[1]
		scale_y4.max = new_signal_max + 0.10*new_signal_max
		return [change-1]'''

	# View/Edit Section

	full_a_mark.image = widgets.Image(value=gen_image_data(a_image_np, "Full A Mark (2)"))

	'''	l2 = traitlets.directional_link((rois, 'selected'),(roi_slider, 'value'), roi_change)
	l1 = traitlets.directional_link((roi_slider, 'value'), (rois, 'selected'), slider_change)'''

@out.capture()
def update_btn_click(_):
	global contours
	if context.cnmf_results in [None, [], ''] or len(context.YrDT) == 0:
		update_status("Error: No data loaded.")
		return None
	update_status("Updating plots...")

	min_snr_ = float(min_snr_edit_widget.value)
	cnnlowest_ = float(cnnlowest_edit_widget_.value)#
	cnnmin_ = float(cnnmin_edit_widget_.value)#
	rvallowest_ = float(rvallowest_edit_widget_.value)#
	rvalmin_ = float(rvalmin_edit_widget_.value)#
	fitness_delta_ = float(fitness_delta_edit_widget_.value)
	min_std_reject_ = float(minstdreject_edit_widget_.value) #

	fr_ = float(fr_widget.value)
	decay_time_ = float(decay_time_widget.value)
	gSig = int(gSig_edit_widget_.value)
	gSiz = 12 #isn't actually used, need to fix
	'''filter_rois(YrDT, cnmf_results, dview, gSig, gSiz, fr=0.3, decay_time=0.4, min_SNR=3, \
		r_values_min=0.85, cnn_thr=0.8, min_std_reject=0.5, thresh_fitness_delta=-20., thresh_cnn_lowest=0.1, \
		r_values_lowest=-1)'''
	if context.YrDT[0].filename is None:
		context.YrDT = getYrDT()
	context.idx_components_keep, context.idx_components_toss = \
		filter_rois(context.YrDT, context.cnmf_results, context.dview, gSig, gSiz, fr=fr_, \
			min_SNR=min_snr_, r_values_min=rvalmin_,decay_time=decay_time_, cnn_thr=cnnmin_, \
			min_std_reject=min_std_reject_, thresh_fitness_delta=fitness_delta_,thresh_cnn_lowest=cnnlowest_, \
			r_values_lowest=rvallowest_)
	A, C, b, f, YrA, sn, idx_components, conv = load_context_data(context)
	idx_components_keep = context.idx_components_keep
	A_ = A[:, idx_components_keep]
	C_ = C[idx_components_keep, :]
	contours_ = [contours[i] for i in idx_components_keep]
	#sometimes plot width gets messed up, so set width
	#fig.layout.width = '67%'
	update_plots(A_, C_, context.YrDT[1], conv, contours_)
	update_status("Idle.")

def roi_change(change):
	if change is not None:
		return change[0] + 1
	else:
		return 1

def verify_context_cnmf(context):
	if not pathlib.Path(context.working_cnmf_file).is_file():
		return False
	else:
		return True

@out.capture()
def show_cnmf_results_interface(context):
	global contours
	if not verify_context_cnmf(context):
		update_status("Unable to load interactive viewer. Context is corrupted or files are missing.")
		return False
	update_status("Launching interactive results viewer...this may take a few moments.")
	gSig = context.cnmf_params['gSig'][0]
	#Yr, dims, T = context.YrDT
	Yr, dims, T = getYrDT()
	Yr_reshaped = np.rollaxis(np.reshape(Yr, dims + (T,), order='F'),2)
	#interactive ROI refinement
	try:
		A, C, b, f, YrA, sn, idx_components, conv = load_context_data(context)
	except:
		update_status("Error: Unable to load data.")
		return None
	idx_components_keep = context.idx_components_keep
	A_ = A[:, idx_components_keep]
	C_ = C[idx_components_keep, :]
	#A spatial matrix, C temporal matrix, S deconvolution results (if applicable)
	#print("Mem Size A: {0}, Mem Size C: {1}".format(getsizeof(A), getsizeof(C)))
	#setup scales

	scale_x2.max = dims[1]
	scale_y2.max = dims[0]
	#correlation plots
	correlation_img = corr_img(Yr_reshaped, gSig, center_psf=True, plot=False)
	#print("correlation_img dtype: {}".format(correlation_img[1].dtype))

	#generate contours
	contours = cm.utils.visualization.get_contours(A, (dims[0],dims[1]))
	contours_ = [contours[i] for i in idx_components_keep]
	centers = np.array([x['CoM'] for x in contours_])
	centers = centers.T

	#correlation image
	cor_image.value = gen_image_data(correlation_img[1], "Correlation Image")
	cor_image.width = dims[1]
	cor_image.height = dims[0]

	if type(A) != np.ndarray: #probably sparse array, need to convert to dense array in order to reshape
		A_ = A_.toarray()
	#a_image = np.mean(A.reshape(dims[1], dims[0], A.shape[1]), axis=2)
	a_image_np = np.mean(A_.reshape(dims[1], dims[0], A_.shape[1]), axis=2).T
	a_image_np = scale( a_image_np, axis=1, with_mean=False, with_std=True, copy=True ) #normalize pixel values (enhances contrast)

	a_image.value = gen_image_data(a_image_np, "All ROIs")#a_data
	a_image.width = dims[1]
	a_image.height = dims[0]
	#a_img_file.close()

	#full_a_mark.image = a_image

	'''	#for updating individual ROI spatial footprint
	def get_roi_image(A, index, dims):
		img = A[:,index].reshape(dims[1],dims[0]).T
		#print("roi_img dtype: {}".format(img.dtype))
		return gen_image_data(img, "Selected ROI")'''

	roi_image.value = get_roi_image(A_,1,dims)
	roi_image.width = dims[1]
	roi_image.height = dims[0]

	#roi_image_mark.image = roi_image
	rois.x = centers[1]
	rois.y = (dims[0] - centers[0])

	'''	def get_contour_coords(index):
		x = [x['coordinates'][:,0] for x in contours][index]
		y = dims[0]-[y['coordinates'][:,1] for y in contours][index]
		return x,y'''

	'''	def get_signal(index, deconv=False):
		if not deconv:
			return C[index], np.repeat(0,len(C[index]))
		else:
			return C[index], conv[index]'''

	#roi_slider = IntSlider(min=1, max=A.shape[1], step=1, description='ROI#', value=1)
	roi_slider.max = A_.shape[1]

	contour_x,contour_y = get_contour_coords(0, contours, dims)

	contour_mark.x = contour_x
	contour_mark.y = contour_y

	scale_x4.max = C_.shape[1]
	if conv is not None:
		scale_x5.max = conv.shape[1]
	else:
		scale_x5.max = C_.shape[1]

	deconv = True if conv is not None else False
	init_signal = get_signal(C_, roi_slider.value, conv) #returns tuple (C, S) if deconv is True, else returns (C, np.arange(0,len(C)))
	init_signal_max = init_signal[0].max()
	#Deconvolved signal (if applicable)
	init_deconv_signal_max = 0
	if type(conv) == np.ndarray: #or deconv=True
		init_deconv_signal_max = init_signal[1].max()

	scale_y4.max=(1.10 * init_signal_max)
	scale_y5.max=(1.10 * init_deconv_signal_max)

	signal_mark.x = np.arange(C_.shape[1])
	signal_mark.y = init_signal[0]

	deconv_signal_mark.x = np.arange(C_.shape[1])
	deconv_signal_mark.y=init_signal[1]

	if init_signal[1] is not None:
		deconv_signal_mark.y = init_signal[1]

	deconv_chk.observe(toggle_deconv)

	#def detrend_df_f(A, b, C, f, YrA = None, quantileMin=8, frames_window=200, block_size=400):
	#def detrend_df_f_auto(A, b, C, f, YrA=None, frames_window=1000, use_fast = False):

	'''	def slider_change(change):
		contour_mark.x,contour_mark.y = get_contour_coords(change-1, contours, dims)
		roi_image_mark.image = widgets.Image(value=get_roi_image(A_,(change-1),dims))
		deconv = True if conv is not None else False
		new_signal = get_signal(C_, change-1, conv)
		signal_mark.y = new_signal[0]
		new_signal_max = new_signal[0].max()
		if new_signal[1] is not None:
			deconv_signal_mark.y = new_signal[1]
		scale_y4.max = new_signal_max + 0.10*new_signal_max
		return [change-1]'''

	# View/Edit Section

	full_a_mark.image = widgets.Image(value=gen_image_data(a_image_np, "Full A Mark (2)"))
	cor_image_mark.image = widgets.Image(value=gen_image_data(correlation_img[1], "Cor Image mark (2)"))

	l2 = traitlets.directional_link((rois, 'selected'),(roi_slider, 'value'), roi_change)
	l1 = traitlets.directional_link((roi_slider, 'value'), (rois, 'selected'), slider_change)

##########
download_btn.on_click(download_data_func)
#delete_roi_btn.on_click(delete_roi_func)
is_edit_widget.observe(is_edit_changed, names='value')
update_edit_btn.on_click(update_btn_click)

	#update_status("Idle")
	#return view_cnmf_widget

interface_edit = VBox([VBox([HBox([roi_slider, tb0]), HBox([is_edit_widget, deconv_chk, dff_chk, download_btn])]),
	  HBox([fig, fig4, edit_panel_widget]), HBox([fig2, fig3])])


#@out.capture()
def set_wkdir(_):
	wkdir_ = workingdir_selector.value
	wkdir_ = os.path.join(wkdir_, '')
	if not pathlib.Path(wkdir_).is_dir():
		update_status("Invalid directory.")
		return None
	workingdir_selector.value = wkdir_ #add trailing slash if not present
	context.working_dir = wkdir_
	update_status("Working Directory set to: {}".format(context.working_dir))
	context_path_txt.value = context.working_dir
	cnmf_file_selector.value = context.working_dir
	file_selector.value = context.working_dir
workingdir_btn.on_click(set_wkdir)

view_results_col = widgets.VBox()
view_results_tmp = widgets.VBox()
view_results_col.children = [view_cnmf_results_widget, view_results_tmp]

###### Validation Column #######
validate_col = widgets.VBox()

#@out.capture()
def getYrDT():
	filename=os.path.split(context.working_cnmf_file)[-1]
	if len(context.YrDT) == 0 or context.YrDT[0].filename is None:
		Yr, dims, T = load_memmap(os.path.join(os.path.split(context.working_cnmf_file)[0],filename))
	else:
		Yr, dims, T = context.YrDT

	return Yr, dims, T

#@out.capture()
def view_cnmf_mov_click(_):
	update_status("Launching movie")
	A, C, b, f, YrA, sn, idx_components, conv = context.cnmf_results
	Yr, dims, T = getYrDT()
	mag_val = validate_col_mag_slider.value
	cm.movie(np.reshape(A.tocsc()[:, context.idx_components_keep].dot(
    C[context.idx_components_keep]), dims + (-1,), order='F').transpose(2, 0, 1)).play(magnification=mag_val, gain=10.)
	update_status("Idle")

def view_bgmov_click(_):
	update_status("Launching movie")
	Yr, dims, T = getYrDT()
	Y = Yr.T.reshape((T,) + dims, order='F')
	A, C, b, f, YrA, sn, idx_components, conv = context.cnmf_results
	mag_val = validate_col_mag_slider.value
	cm.movie(np.reshape(b.dot(f), dims + (-1,),
                    order='F').transpose(2, 0, 1)).play(magnification=mag_val, gain=1.)#magnification=3, gain=1.
	update_status("Idle")

def view_residual_click(_):
	update_status("Launching movie")
	A, C, b, f, YrA, sn, idx_components, conv = context.cnmf_results
	Yr, dims, T = getYrDT()
	Y = Yr.T.reshape((T,) + dims, order='F')
	mag_val = validate_col_mag_slider.value
	cm.movie(np.array(Y) - np.reshape(A.tocsc()[:, :].dot(C[:]) + b.dot(
    f), dims + (-1,), order='F').transpose(2, 0, 1)).play(magnification=mag_val, gain=10., fr=10) #magnification=3, gain=10., fr=10
	update_status("Idle")

validate_col_cnmf_mov_btn.on_click(view_cnmf_mov_click)
validate_col_bgmov_btn.on_click(view_bgmov_click)
validate_col_residual_btn.on_click(view_residual_click)
validate_col.children = [validate_col_mag_box, validate_col_cnmf_mov_btn, \
validate_col_bgmov_btn, validate_col_residual_btn]

def view_results_(_):
	#Yr_reshaped.reshape(np.prod(dims), T)
	if context.cnmf_results in [None, []]:
		update_status("Error: No data loaded.")
		return None
	update_status("Launching interactive results viewer...this may take a few moments.")
	#interface_edit = show_cnmf_results_interface(context)
	if show_cnmf_results_interface(context) == False:
		return None
	else:
		update_status("Idle.")
	#produce interface...
	#display(interface_edit)
	#view_results_col.children = [view_cnmf_results_widget,interface_edit]


view_results_col.children = [view_cnmf_results_widget,interface_edit]
view_cnmf_results_widget.on_click(view_results_)



'''    mc_params = { #for processing individual movie at a time using MotionCorrect class object
	'dview': dview, #refers to ipyparallel object for parallelism
	'max_shifts':(2, 2),  # maximum allow rigid shift; default (2,2)
	'niter_rig':1,
	'splits_rig':20,
	'num_splits_to_process_rig':None,
	'strides':(24,24), #default 48,48
	'overlaps':(12,12), #default 12,12
	'splits_els':28,
	'num_splits_to_process_els':[14, None],
	'upsample_factor_grid':4,
	'max_deviation_rigid':2,
	'shifts_opencv':True,
	'nonneg_movie':True,
	'gSig_filt' : [int(x) for x in gSigFilter.value.split(',')] #default 9,9  best 6,6,
	'dsfactors': None #or (1,1,1)   (ds x, ds y, ds t)
}'''
#import event_logic_v2 as event_ui
#event_ui.setup_context(context)

app_ui = VBox()
ui_tab = Tab()
#children = [wkdir_context_box,major_col,mc_results_box,major_cnmf_col,view_results_col, validate_col, event_ui.set_event_widgets()]
children = [wkdir_context_box,major_col,mc_results_box,major_cnmf_col,view_results_col, validate_col]
#tab_titles = ['Main','Motion Correction','MC Results','CNMF', 'CNMF Results','CNMF Validation', 'Event Detection']
tab_titles = ['Main','Motion Correction','MC Results','CNMF', 'CNMF Results','CNMF Validation']
ui_tab.children = children
for i in range(len(children)):
    ui_tab.set_title(i, str(tab_titles[i]))
app_ui.children = [status_bar_widget, ui_tab, out]
