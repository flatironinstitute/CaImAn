from ipywidgets import interact, interactive, fixed, interact_manual, HBox, VBox, IntSlider, Play, jslink
import ipywidgets as widgets
import bqplot
from bqplot import (
	LogScale, LinearScale, OrdinalColorScale, ColorAxis,
	Axis, Scatter, Lines, CATEGORY10, Label, Figure, Tooltip, Toolbar
)
import traitlets
from sklearn.preprocessing import scale
import pandas as pd
import io
import glob
import os
import numpy as np

from caiman_easy import *
#from event_logic import *

'''
Interface Code Developed by Brandon Brown in the Khakh Lab at UCLA
"CaImAn" algorithms developed by Simons Foundation
Nov 2017
'''

#create context
context = Context(start_procs(10))

#motion correction interface

# UX
workingdir_selector = widgets.Text(
	value=os.getcwd(),
	placeholder=os.getcwd(),
	#description='Working Directory:',
	layout=widgets.Layout(width='35%'),
	disabled=False
)
workingdir_btn = widgets.Button(
	description='Set WkDir',
	disabled=False,
	button_style='success', # 'success', 'info', 'warning', 'danger' or ''
	tooltip='Set working directory'
)
context_path_txt = widgets.Text(
	value=os.getcwd(),
	placeholder=os.getcwd(),
	#description='Load Context from:',
	layout=widgets.Layout(width='35%'),
	disabled=False
)
context_load_btn = widgets.Button(
	description='Load Context',
	disabled=False,
	button_style='success', # 'success', 'info', 'warning', 'danger' or ''
	tooltip='Load Context'
)
context_save_btn = widgets.Button(
	description='Save Context',
	disabled=False,
	button_style='success', # 'success', 'info', 'warning', 'danger' or ''
	tooltip='Save Context'
)
def load_context_event(_):
	context.load(context_path_txt.value)

def save_context_event(_):
	currentDT = datetime.datetime.now()
	ts_ = currentDT.strftime("%Y%m%d_%H_%M_%S")
	c_loc = workingdir_selector.value + "context_" + ts_
	context.save(c_loc)

context_load_btn.on_click(load_context_event)
context_save_btn.on_click(save_context_event)



wkdir_box = widgets.HBox()
wkdir_box.children = [widgets.Label("Set Working Directory:"), workingdir_selector, workingdir_btn]
context_box = widgets.HBox()
context_box.children = [widgets.Label("Load context from:"),context_path_txt, context_load_btn, context_save_btn]
wkdir_context_box = widgets.VBox()
wkdir_context_box.children = [wkdir_box,context_box]

file_selector = widgets.Text(
	value=os.getcwd(),
	placeholder=os.getcwd(),
	description='File/Folder Path:',
	layout=widgets.Layout(width='50%'),
	disabled=False
)
load_files_btn = widgets.Button(
	description='Load Files',
	disabled=False,
	button_style='success', # 'success', 'info', 'warning', 'danger' or ''
	tooltip='Load Files',
	icon='check'
)

#Get file paths for *.tif and *.avi files, load into the context object
def load_f(x):
	context.working_mc_files = load_files(file_selector.value, print_values=True)

#Load Files button click handler
load_files_btn.on_click(load_f)

file_box = widgets.HBox()
file_box.children = [file_selector, load_files_btn]
is_batch_widget = widgets.ToggleButtons(
	options=['Group', 'Independent'],
	description='Grouped?:',
	disabled=False,
	button_style='', # 'success', 'info', 'warning', 'danger' or ''
	tooltips=['Run all movies together as if one movie', 'Run each movie independently'],
#     icons=['check'] * 3
)
dslabel = widgets.Label(value="Downsample Percentage (height, width, frames/time)")
ds_layout = widgets.Layout(width="20%")
dsx_widget = widgets.BoundedFloatText(
	value=1,
	min=0.1,
	max=1.0,
	step=0.1,
	description='height:',
	disabled=False,
	layout=ds_layout
)
dsy_widget = widgets.BoundedFloatText(
	value=1,
	min=0.1,
	max=1.0,
	step=0.1,
	description='width:',
	disabled=False,
	layout=ds_layout
)
dst_widget = widgets.BoundedFloatText(
	value=1,
	min=0.1,
	max=1.0,
	step=0.1,
	description='frames:',
	disabled=False,
	layout=ds_layout
)
ds_factors_box = widgets.HBox()
ds_factors_box.children = [dsx_widget, dsy_widget, dst_widget]
gSigFilter_widget = widgets.IntSlider(
	value=7,
	min=0,
	max=50,
	step=1,
	description='High Pass Filter:',
	disabled=False,
	continuous_update=False,
	orientation='horizontal',
	readout=True,
	readout_format='d',
	tooltip='Gaussian Filter Size (1p data only)'
)
is_rigid_widget = widgets.ToggleButtons(
	options=['Rigid', 'Non-Rigid'],
	description='MC Mode:',
	disabled=False,
	button_style='warning', # 'success', 'info', 'warning', 'danger' or ''
	tooltips=['Rigid correction (faster)', 'Non-rigid correction (slow, more accurate)'],
#     icons=['check'] * 3
)
basic_settings = widgets.VBox()
basic_settings.children = [dslabel, ds_factors_box, gSigFilter_widget, is_rigid_widget]

advanced_settings = widgets.VBox()
advanced_settings.children = [widgets.Label(value='Under Construction')]
settings = widgets.Accordion(children=[basic_settings, advanced_settings])
settings.set_title(0, 'Basic MC Settings') #MC = Motion Correction
settings.set_title(1, 'Advanced MC Settings')

run_mc_btn = widgets.Button(
	description='Run Motion Correction',
	disabled=False,
	button_style='info', # 'success', 'info', 'warning', 'danger' or ''
	tooltip='Run motion correction',
	layout=widgets.Layout(width="30%")
)

def run_mc_ui(_):
	#get settings:
	is_batch = True if is_batch_widget.value == 'Group' else False
	is_rigid = True if is_rigid_widget.value == 'Rigid' else False
	mc_params = {
		'dview': context.dview, #refers to ipyparallel object for parallelism
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
		'gSig_filt' : [int(gSigFilter_widget.value)] * 2, #default 9,9  best 6,6,
	}
	#call run_mc
	#< run_mc(fnames, mc_params, rigid=True, batch=True) > returns list of mmap file names
	dsfactors = (float(dsx_widget.value),float(dsy_widget.value),float(dst_widget.value)) #or (1,1,1)   (ds x, ds y, ds t)
	mc_results = run_mc(context.working_mc_files, mc_params, dsfactors, rigid=is_rigid, batch=is_batch)
	if is_rigid:
		context.mc_rig = mc_results
	else:
		context.mc_nonrig = mc_results
	print("Motion Correction DONE!")
	print("Output file(s): ")
	[print(x) for x in mc_results]

run_mc_btn.on_click(run_mc_ui)
major_col = widgets.VBox()
major_col.children = [file_box,is_batch_widget,settings, run_mc_btn]

# play motion corrected movies

def show_movies(_):
	orig_mov_widget = widgets.HTML(
		value=play_movie(cm.load(context.working_mc_files[0]),cmap='gist_gray').data,
		description='Original Movie',
	)
	mov_mc_ = cm.load(context.mc_rig[0]) if len(context.mc_rig) > 0 else cm.load(context.mc_nonrig[0])
	mc_mov_widget = widgets.HTML(
		value= play_movie(mov_mc_,cmap='gist_gray').data,
		description='Motion Corrected',
	)
	mov_row = widgets.HBox()
	mov_row.children = [orig_mov_widget, mc_mov_widget]
	display(mov_row)

play_mov_btn = widgets.Button(
	description='Play Movies',
	disabled=False,
	button_style='success', # 'success', 'info', 'warning', 'danger' or ''
	tooltip='Play Movies'
)
play_mov_btn.on_click(show_movies)
play_mov_btn_box = widgets.HBox()
play_mov_btn_box.children = [play_mov_btn]


#cnmf-e run analysis interface

'''n_processes=n_processes, method_init='corr_pnr', k=20, gSig=(5, 5), gSiz=(5, 5), merge_thresh=.8,
				p=1, dview=dview, tsub=1, ssub=1,p_ssub=2, Ain=None, rf=(25, 25), stride=(15, 15),
				only_init_patch=True, gnb=5, nb_patch=3, method_deconvolution='oasis',
				low_rank_background=False, update_background_components=False, min_corr=min_corr,
				min_pnr=min_pnr, normalize_init=False, deconvolve_options_init=None,
				ring_size_factor=1.5, center_psf=True'''

cnmf_file_selector = widgets.Text(
	value=os.getcwd(),
	placeholder=os.getcwd(),
	description='File (.mmap):',
	layout=widgets.Layout(width='50%'),
	disabled=False
)

#Get file paths for *.tif and *.avi files, load into the context object
'''def cnmf_load_f(x):
	context.working_cnmf_file = load_files(cnmf_file_selector.value, print_values=True)[0]'''

cnmf_file_box = widgets.HBox()
cnmf_file_box.children = [cnmf_file_selector]
is_patches_widget = widgets.ToggleButtons(
	value='Single FOV',
	options=['Patches', 'Single FOV'],
	description='Patches?:',
	disabled=False,
	button_style='', # 'success', 'info', 'warning', 'danger' or ''
	tooltips=['Run each frame in parallel by breaking into overlapping FOVs', 'The whole frame is analyed as a single FOV'],
#     icons=['check'] * 3
)
dslabel = widgets.Label(value="Downsample Percentage (spatial, temporal)")
ds_layout = widgets.Layout(width="20%")
ds_spatial_widget = widgets.BoundedFloatText(
	value=1.0,
	min=0.0,
	max=1.0,
	step=0.1,
	description='spatial:',
	disabled=False,
	layout=ds_layout
)
ds_temporal_widget = widgets.BoundedFloatText(
	value=1.0,
	min=0.0,
	max=1.0,
	step=0.1,
	description='temporal:',
	disabled=False,
	layout=ds_layout
)
basic_row0 = widgets.HBox()
basic_row0.children = [is_patches_widget]

basic_row1 = widgets.HBox()
basic_row1.children = [dslabel,ds_spatial_widget,ds_temporal_widget]

k_widget = widgets.BoundedIntText(
	value=100,
	min=1,
	max=1000,
	step=5,
	description='K:',
	tooltip='Expected # Cells (Per Patch)',
	disabled=False,
	layout=ds_layout
)
gSig_widget = widgets.BoundedIntText(
	value=4,
	min=1,
	max=50,
	step=1,
	description='gSig:',
	tooltip='Gaussian Kernel Size',
	disabled=False,
	layout=ds_layout
)
gSiz_widget = widgets.BoundedIntText(
	value=12,
	min=1,
	max=50,
	step=1,
	description='gSiz:',
	tooltip='Average Cell Diamter',
	disabled=False,
	layout=ds_layout
)
basic_row2 = widgets.HBox()
basic_row2.children = [k_widget, gSig_widget, gSiz_widget]

min_corr_widget = widgets.FloatSlider(
	value=0.85,
	min=0.0,
	max=1.0,
	step=0.05,
	description='Min. Corr.:',
	disabled=False,
	continuous_update=False,
	orientation='horizontal',
	readout=True,
	readout_format='.2',
	tooltip='Minimum Correlation'
)
min_pnr_widget = widgets.IntSlider(
	value=15,
	min=1,
	max=50,
	step=1,
	description='Min. PNR.:',
	disabled=False,
	continuous_update=False,
	orientation='horizontal',
	readout=True,
	readout_format='d',
	tooltip='Minimum Peak-to-Noise Ratio'
)

deconv_flag_widget = widgets.Checkbox(
	value=False,
	description='Run Deconvolution',
	disabled=False,
	tooltip='The oasis deconvolution algorithm will run along with CNMF',
	layout=widgets.Layout(width="30%")
)

save_movie_widget = widgets.Checkbox(
	value=False,
	description='Save Denoised Movie (.avi)',
	disabled=False,
	tooltip='Saves a background-substracted and denoised movie',
	layout=widgets.Layout(width="30%")
)

#min_corr, min_pnr
basic_row3 = widgets.HBox()
basic_row3.children = [min_corr_widget, min_pnr_widget]

basic_row4 = widgets.HBox()
basic_row4.children = [deconv_flag_widget, save_movie_widget]

cnmf_basic_settings = widgets.VBox()
cnmf_basic_settings.children = [basic_row0,basic_row1,basic_row2, basic_row3, basic_row4]

cnmf_advanced_settings = widgets.VBox()
cnmf_advanced_settings.children = [widgets.Label(value='Under Construction')]
cnmf_settings = widgets.Accordion(children=[cnmf_basic_settings, cnmf_advanced_settings])
cnmf_settings.set_title(0, 'Basic CNMF-E Settings') #MC = Motion Correction
cnmf_settings.set_title(1, 'Advanced CNMF-E Settings')

run_cnmf_btn = widgets.Button(
	description='Run CNMF-E',
	disabled=False,
	button_style='info', # 'success', 'info', 'warning', 'danger' or ''
	tooltip='Run CNMF-E',
	layout=widgets.Layout(width="30%")
)

def run_cnmf_ui(_):
	#get file
	context.working_cnmf_file = load_files(cnmf_file_selector.value, print_values=True)[0]
	# get memmory mapped file
	#context.Yr = cm.load(context.working_cnmf_file)
	#get settings:
	ds_spatial = int(1.0 / float(ds_spatial_widget.value))
	ds_temporal = int(1.0 / float(ds_temporal_widget.value))
	min_corr = float(min_corr_widget.value)
	min_pnr = float(min_pnr_widget.value)
	is_patches = True if is_patches_widget.value == 'Patches' else False
	K = int(k_widget.value)
	gSig = (int(gSig_widget.value),) * 2
	gSiz = (int(gSiz_widget.value),) * 2

	cnmf_params = {
		'n_processes':context.n_processes,
		'method_init':'corr_pnr',
		'k':K,
		'gSig':gSig,
		'gSiz':gSiz,
		'merge_thresh':0.8,
		'p':1,
		'dview':context.dview,
		'tsub':1 if is_patches else ds_temporal, # x if not patches else 1 #THIS IS INTEGER NOT FLOAT
		'ssub':1 if is_patches else ds_spatial,
		'p_ssub': ds_spatial if is_patches else None,  #THIS IS INTEGER NOT FLOAT
		'p_tsub': ds_temporal if is_patches else None,
		'Ain':None,
		'rf':(40, 40) if is_patches else None, #enables patches; was 25x25
		'stride':(20, 20), #was 15,15
		'only_init_patch': True,
		'gnb':16,
		'nb_patch':16,
		'method_deconvolution':'oasis',
		'low_rank_background': True,
		'update_background_components': False,
		'min_corr':min_corr,
		'min_pnr':min_pnr,
		'normalize_init': False,
		'deconvolve_options_init': None,
		'ring_size_factor':1.5,
		'center_psf': True,
		'deconv_flag': bool(deconv_flag_widget.value),
		'simultaneously': False,
		'del_duplicates':True
	}
	#save params to context
	context.cnmf_params = cnmf_params
	#RUN CNMF-E
	#get original movie as mmap
	filename=os.path.split(context.working_cnmf_file)[-1]
	# =
	Yr, dims, T = load_memmap(os.path.join(os.path.split(context.working_cnmf_file)[0],filename))
	#get correlation image
	context.YrDT = Yr, dims, T
	#Yr_reshaped = np.reshape(Yr, dims + (T,), order='F')
	#correlation_img = corr_img(Yr_reshaped, gSig[0], center_psf=True)
	#Yr_reshaped = np.rollaxis(np.reshape(Yr, dims + (T,), order='F'),2)
	#correlation_plot = corr_img(Yr_reshaped, cnmf_params['gSig'][0], center_psf=True)
	#def cnmf_run(fname, cnmf_params):
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
	print("CNMF-E FINISHED!")
	#results: A, C, b, f, YrA, sn, idx_components, S
	refine_results = True
	save_movie_bool = bool(save_movie_widget.value)
	if refine_results:
		print("Automatically refining results...")
		context.idx_components_keep, context.idx_components_toss = \
			filter_rois(context.YrDT, context.cnmf_results, context.dview, gSig, gSiz)
	#def corr_img(Yr, gSig, center_psr :bool):
	#save denoised movie:
	if save_movie_bool:
		save_denoised_avi(context.cnmf_results, dims, context.idx_components_keep, context.working_dir)
run_cnmf_btn.on_click(run_cnmf_ui)
major_cnmf_col = widgets.VBox()
major_cnmf_col.children = [cnmf_file_box, cnmf_settings, run_cnmf_btn]

# ---------------------

# view cnmf results interface

# ---------------------
def show_cnmf_results_interface():
	gSig = context.cnmf_params['gSig'][0]
	Yr, dims, T = context.YrDT
	Yr_reshaped = np.rollaxis(np.reshape(Yr, dims + (T,), order='F'),2)
	#interactive ROI refinement
	if len(context.cnmf_results) == 8: #for backward compatibility
		A, C, b, f, YrA, sn, idx_components, conv = context.cnmf_results
	else:
		A, C, b, f, YrA, sn, idx_components = context.cnmf_results
		conv = None
	#A spatial matrix, C temporal matrix, S deconvolution results (if applicable)
	#print("Mem Size A: {0}, Mem Size C: {1}".format(getsizeof(A), getsizeof(C)))
	#setup scales
	scale_x = bqplot.LinearScale(min=0.0, max=1) #for images
	scale_y = bqplot.LinearScale(min=0.0, max=1) #for images
	scale_x2 = bqplot.LinearScale(min=0.0, max=dims[1]) #eg 376
	scale_y2 = bqplot.LinearScale(min=0.0, max=dims[0]) #eg 240
	#correlation plots
	correlation_img = corr_img(Yr_reshaped, gSig, center_psf=True, plot=False)

	#generate contours

	contours = cm.utils.visualization.get_contours(A, (dims[0],dims[1]))
	centers = np.array([x['CoM'] for x in contours])
	centers = centers.T

	#generate interface

	#correlation image
	cor_img_file = io.BytesIO()
	plt.imsave(cor_img_file, correlation_img[1], format='PNG')
	cor_data = bytes(cor_img_file.getbuffer())
	cor_image = widgets.Image(
		value=cor_data,
		format='png',
		width=dims[1],
		height=dims[0],
		)#,width=376,height=240
	cor_img_file.close()
	#full spatial matrix image
	a_img_file = io.BytesIO()
	if type(A) != np.ndarray: #probably sparse array, need to convert to dense array
		A = A.toarray()
	#a_image = np.mean(A.reshape(dims[1], dims[0], A.shape[1]), axis=2)
	a_image = np.mean(A.reshape(dims[1], dims[0], A.shape[1]), axis=2).T
	a_image = scale( a_image, axis=1, with_mean=False, with_std=True, copy=True ) #normalize pixel values (enhances contrast)

	plt.imsave(a_img_file, a_image, format='PNG')
	a_data = bytes(a_img_file.getbuffer())
	a_image = widgets.Image(
		value=a_data,
		format='png',
		width=dims[1],
		height=dims[0],
		)#,width=376,height=240
	a_img_file.close()

	#for updating individual ROI spatial footprint
	def get_roi_image(A,index,dims):
		img = A[:,index].reshape(dims[1],dims[0]).T
		#img_file = io.BytesIO()
		with io.BytesIO() as img_file:
			plt.imsave(img_file, img, format='PNG')
			data = bytes(img_file.getbuffer())
			#img_file.close()
			return data
	#correlation image, background
	cor_image_mark = bqplot.Image(image = cor_image, scales={'x': scale_x, 'y': scale_y})#, scales={'x': scale_x, 'y': scale_y}
	#reconstructed neuronal spatial maps
	full_a_mark = bqplot.Image(image = a_image, scales={'x': scale_x, 'y': scale_y})#, scales={'x': scale_x2, 'y': scale_y2}
	#individiaul roi spatial map
	roi_image = widgets.Image(
		value = get_roi_image(A,1,dims),
		format='png',
		width=dims[1],
		height=dims[0],
		)
	roi_image_mark = bqplot.Image(image = roi_image, scales={'x': scale_x, 'y': scale_y})#, scales={'x': scale_x2, 'y': scale_y2}

	rois = Scatter(x=(centers[1]), y=(dims[0]-centers[0]),scales={'x': scale_x2, 'y': scale_y2}, default_size=30,
				  unhovered_style={'opacity': 0.3}, colors=['red'], default_opacity=0.1, selected=[0])
	rois.interactions = {'click': 'select'}
	rois.selected_style = {'opacity': 1.0, 'fill': 'Black', 'stroke': 'Black', 'size':30}

	def get_contour_coords(index):
		x = [x['coordinates'][:,0] for x in contours][index]
		y = dims[0]-[y['coordinates'][:,1] for y in contours][index]
		return x,y

	def get_signal(index, deconv=False):
		if not deconv:
			return C[index], None
		else:
			return C[index], conv[index]

	roi_slider = IntSlider(min=1, max=A.shape[1], step=1, description='ROI#', value=1)
	#roi_slider.observe(slider_change)
	def roi_change(change):
		if change is not None:
			return change[0] + 1
		else:
			return 1

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


	contour_x,contour_y = get_contour_coords(0)
	contour_mark = bqplot.Lines(x = contour_x, y = contour_y, colors=['yellow'], scales={'x': scale_x2, 'y': scale_y2})

	#rois.on_element_click(roi_click)

	fig = bqplot.Figure(padding_x=0, padding_y=0, title='Detected ROIs')
	fig.marks = [cor_image_mark, rois]
	fig.axes = [bqplot.Axis(scale=scale_x2), bqplot.Axis(scale=scale_y2, orientation='vertical')]

	fig2 = bqplot.Figure(padding_x=0, padding_y=0, title='Background Subtracted')
	fig2.marks = [full_a_mark, contour_mark]
	fig2.axes = [bqplot.Axis(scale=scale_x2), bqplot.Axis(scale=scale_y2, orientation='vertical')]

	fig3 = bqplot.Figure(padding_x=0, padding_y=0, title='Selected ROI (Isolated)')
	fig3.marks = [roi_image_mark]
	fig3.axes = [bqplot.Axis(scale=scale_x2), bqplot.Axis(scale=scale_y2, orientation='vertical')]

	# Fluorescence trace
	scale_x4 = bqplot.LinearScale(min=0.0, max=C.shape[1])
	scale_x5 = bqplot.LinearScale(min=0.0, max=conv.shape[1])
	deconv = True if conv is not None else False
	init_signal = get_signal(roi_slider.value, deconv) #returns tuple (C, S) if deconv is True, else returns (C, None)
	init_signal_max = init_signal[0].max()
	#Deconvolved signal (if applicable)
	init_deconv_signal_max = 0
	if type(conv) == np.ndarray: #or deconv=True
		init_deconv_signal_max = init_signal[1].max()
	scale_y4 = bqplot.LinearScale(min=0.0, max=(1.10 * init_signal_max)) # add 10% to give some upper margin
	scale_y5 = bqplot.LinearScale(min=0.0, max=(1.10 * init_deconv_signal_max)) # add 10% to give some upper margin
	signal_mark = bqplot.Lines(x = np.arange(C.shape[1]), y = init_signal[0], colors=['black'],
							   scales={'x': scale_x4, 'y': scale_y4}, display_legend=True)
	deconv_signal_mark = bqplot.Lines(x = np.arange(C.shape[1]), y=init_signal[1], colors=['red'],
							  scales={'x': scale_x5, 'y': scale_y5}, display_legend=True, visible=False)
	if init_signal[1] is not None:
		deconv_signal_mark.y = init_signal[1]

	fig4 = bqplot.Figure(padding_x=0, padding_y=0, title='Denoised/Demixed Fluorescence Trace',
						 background_style={'background-color':'white'})
	fig4.marks = [signal_mark, deconv_signal_mark]
	fig4.axes = [bqplot.Axis(scale=scale_x4, label='Time (Frame #)',grid_lines='none'), bqplot.Axis(scale=scale_y4, orientation='vertical',label='Amplitude',grid_lines='none')]
	tb0 = Toolbar(figure=fig4)

	#delete/refine ROIs control, and save data
	delete_roi_btn = widgets.Button(
		description='Delete ROI',
		disabled=False,
		button_style='warning', # 'success', 'info', 'warning', 'danger' or ''
		tooltip='Exclude ROI'
	)
	delete_list_widget = widgets.SelectMultiple(
		options=[],
		value=[],
		rows=3,
		description='Exclud. ROIs',
		disabled=False
	)
	download_btn = widgets.Button(
		description='Download Data',
		disabled=False,
		button_style='info', # 'success', 'info', 'warning', 'danger' or ''
		tooltip='Download fluorescence traces as CSV file'
	)
	dff_chk = widgets.Checkbox(
		value=False,
		description='Use dF/F',
		disabled=False,
		tooltip='Compute the delta F/F values',
		layout=widgets.Layout(width="18%")
	)
	# deconv_chk = widgets.Checkbox(
	# 	value=False,
	# 	description='Deconvolution',
	# 	disabled=False,
	# 	tooltip='Show the results of deconvolution (if applicable)',
	# 	layout=widgets.Layout(width="22%")
	# )
	deconv_chk = widgets.ToggleButtons(
	    options=['Signal', 'Deconvolution', 'Both'],
	    description='Plot options:',
	    disabled=False,
	    button_style='', # 'success', 'info', 'warning', 'danger' or ''
	    tooltips=['Denoised/demixed Ca2+ signal', 'Deconvolved Ca2+ signal', 'Plot both'],
		layout=widgets.Layout(width="18%"),
	)
	deconv_chk.observe(toggle_deconv)

	def delete_roi_func(_):
		delete_list_widget.options += (roi_slider.value,)
		delete_list_widget.value = delete_list_widget.options


#def detrend_df_f(A, b, C, f, YrA = None, quantileMin=8, frames_window=200, block_size=400):
	def download_data_func(_):
		#traces = np.delete(C, delete_list_widget.value, axis=0)
		#traces = ma.masked_array(C, mask=delete_list_widget.value)

		adj_c = C
		adj_s = conv
		currentDT2 = datetime.datetime.now()
		ts2_ = currentDT2.strftime("%Y%m%d_%H_%M_%S")
		metadata_ = ts2_
		deld_rois_ = list(delete_list_widget.value)
		print("Excluding ROIs: %s" % (deld_rois_))
		def save_traces():
			if dff_chk.value == True:
				#adj_c = detrend_df_f(A, b, C, f, YrA = YrA.T)
				adj_c = detrend_df_f(A, b, C, f)
				metadata_ += '_dFF'
				print("Using dF/F values")
			traces_path = workingdir_selector.value + 'traces_' + metadata_ + '.csv'

			df = pd.DataFrame(data=adj_c)
			df.index += 1
			deld_rois = list(map(lambda x: x-1, deld_rois_)) #remove ROIs that the user wants to exclude
			df.drop(df.index[deld_rois], inplace=True)

			df.to_csv(traces_path, header=False)
			print("Ca2+ Signal Traces saved to: %s" % (traces_path))
		def save_deconv():
			deconv_path = workingdir_selector.value + 'deconv_' + metadata_ + '.csv'

			df = pd.DataFrame(data=adj_s)
			df.index += 1
			deld_rois = list(map(lambda x: x-1, deld_rois_)) #remove ROIs that the user wants to exclude
			df.drop(df.index[deld_rois], inplace=True)

			df.to_csv(deconv_path, header=False)
			print("Deconvolution data saved to: %s" % (deconv_path))

		if deconv_chk.value == 'Deconvolution':
			save_deconv()
		elif deconv_chk.value == 'Both':
			save_traces()
			save_deconv()
		else:
			save_traces()

	def slider_change(change):
		contour_mark.x,contour_mark.y = get_contour_coords(change-1)
		roi_image_mark.image = widgets.Image(value=get_roi_image(A,(change-1),dims))
		deconv = True if conv is not None else False
		new_signal = get_signal(change-1, deconv)
		signal_mark.y = new_signal[0]
		new_signal_max = new_signal[0].max()
		if new_signal[1] is not None:
			deconv_signal_mark.y = new_signal[1]
		scale_y4.max = new_signal_max + 0.10*new_signal_max
		return [change-1]

	download_btn.on_click(download_data_func)
	delete_roi_btn.on_click(delete_roi_func)
	l2 = traitlets.directional_link((rois, 'selected'),(roi_slider, 'value'), roi_change)
	l1 = traitlets.directional_link((roi_slider, 'value'), (rois, 'selected'), slider_change)

	view_cnmf_widget = VBox([VBox([HBox([roi_slider, tb0]), HBox([delete_roi_btn, delete_list_widget, deconv_chk, dff_chk, download_btn])]),
		  HBox([fig, fig4]), HBox([fig2, fig3])])

	return view_cnmf_widget


view_cnmf_results_widget = widgets.Button(
	description='View/Refine CNMF Results',
	disabled=False,
	button_style='info', # 'success', 'info', 'warning', 'danger' or ''
	tooltip='View CNMF Results',
	layout=widgets.Layout(width="30%")
)


def set_wkdir(_):
	context.working_dir = workingdir_selector.value
	print("Working Directory set to: {}".format(context.working_dir))
	context_path_txt.value = context.working_dir
	cnmf_file_selector.value = context.working_dir
	file_selector.value = context.working_dir
workingdir_btn.on_click(set_wkdir)


def view_results_(_):
	#Yr_reshaped.reshape(np.prod(dims), T)
	interface_edit = show_cnmf_results_interface()
	#produce interface...
	display(interface_edit)


view_cnmf_results_widget.on_click(view_results_)
view_results_col = widgets.VBox()
view_results_col.children = [view_cnmf_results_widget]


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
from event_logic import *
setup_context(context)
