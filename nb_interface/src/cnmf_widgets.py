import ipywidgets as widgets
import os




#####
layout_small = widgets.Layout(width="10%")
ds_layout = widgets.Layout(width="20%")

correlation_plot_btn = widgets.Button(
	description='Inspect Correlation Plot',
	disabled=False,
	button_style='success', # 'success', 'info', 'warning', 'danger' or ''
	tooltip='Inspect Correlation Plot',
	layout=widgets.Layout(width="30%")
)

cnmf_file_selector = widgets.Text(
	value=os.getcwd(),
	placeholder=os.getcwd(),
	description='File (.mmap):',
	layout=widgets.Layout(width='50%'),
	disabled=False
)

cnmf_load_file_btn = widgets.Button(
	description='Load File',
	disabled=False,
	button_style='success', # 'success', 'info', 'warning', 'danger' or ''
	tooltip='Load File',
	layout=widgets.Layout(width="30%")
)

cnmf_load_file_box = widgets.HBox([cnmf_file_selector, cnmf_load_file_btn])


######

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

####


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

stride_widget = widgets.BoundedIntText(
	value=20,
	min=1,
	max=50,
	step=1,
	description='Stride:',
	tooltip='Stride',
	disabled=True,
	layout=ds_layout
)

rf_widget = widgets.BoundedIntText(
	value=40,
	min=1,
	max=75,
	step=1,
	description='Patch Size:',
	tooltip='Patch Size',
	disabled=True,
	layout=ds_layout
)

######

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

refine_components_widget = widgets.Checkbox(
	value=True,
	description='Auto Refine Components',
	disabled=False,
	tooltip='Automatically refines the detected components to the most plausible ones',
	layout=widgets.Layout(width="30%")
)

############ Advanced Settings
'''
# parameters for source extraction and deconvolution
p = 1                       # order of the autoregressive system
gnb = 2                     # number of global background components
merge_thresh = 0.8          # merging threshold, max correlation allowed
# half-size of the patches in pixels. e.g., if rf=25, patches are 50x50
rf = 15
stride_cnmf = 6             # amount of overlap between the patches in pixels
K = 4                       # number of components per patch
gSig = [4, 4]               # expected half size of neurons
# initialization method (if analyzing dendritic data using 'sparse_nmf')
init_method = 'greedy_roi'
is_dendrites = False        # flag for analyzing dendritic data
# sparsity penalty for dendritic data analysis through sparse NMF
alpha_snmf = None

# parameters for component evaluation
min_SNR = 2.5               # signal to noise ratio for accepting a component
rval_thr = 0.8              # space correlation threshold for accepting a component
cnn_thr = 0.8               # threshold for CNN based classifier '''

p_widget = widgets.BoundedIntText(
	value=1,
	min=1,
	max=2,
	step=1,
	description='P (AR order):',
	tooltip='Order of the autoregressive system',
	disabled=False,
	layout=ds_layout
)
gnb_widget = widgets.BoundedIntText(
	value=2,
	min=1,
	max=5,
	step=1,
	#description='GNB:',
	#tooltip='Number of global background components',
	disabled=False,
	layout=layout_small
)
gnb_box = widgets.HBox([widgets.Label(value="Num global background components:"),gnb_widget])

merge_thresh_widget = widgets.BoundedFloatText(
	value=0.8,
	min=0.0,
	max=50,
	step=0.1,
	#description='',
	#tooltip='Number of global background components',
	disabled=False,
	layout=layout_small
)
merge_thresh_box = widgets.HBox([widgets.Label(value="Max Correlation (Merge Thresh):"),merge_thresh_widget])

is_dendrites_widget = widgets.Checkbox(
    value=False,
    #description='',
    disabled=False,
	layout=widgets.Layout(width="30%")
)
is_dendrites_box = widgets.HBox([widgets.Label(value="Analyze dendrites?"),is_dendrites_widget])

min_snr_widget = widgets.BoundedFloatText(
	value=2.5,
	min=1,
	max=1000,
	step=0.5,
	description='Min SNR:',
	#tooltip='Number of global background components',
	disabled=False,
	layout=ds_layout
)

rval_thr_widget = widgets.BoundedFloatText(
	value=0.8,
	min=0.0,
	max=50.0,
	step=0.1,
	#description='Spatial Correlation Threshold (rval_thr):',
	disabled=False,
	layout=layout_small
)
rval_thr_box = widgets.HBox([widgets.Label(value="Spatial Correlation Threshold (rval_thr):"),rval_thr_widget])

cnn_thr_widget = widgets.BoundedFloatText(
	value=0.8,
	min=0.0,
	max=50.0,
	step=0.1,
	#description='',
	#tooltip='Number of global background components',
	disabled=False,
	layout=layout_small
)
cnn_thr_box = widgets.HBox([widgets.Label(value="CNN threshold:"),cnn_thr_widget])



#############

run_cnmf_btn = widgets.Button(
	description='Run CNMF',
	disabled=False,
	button_style='info', # 'success', 'info', 'warning', 'danger' or ''
	tooltip='Run CNMF',
	layout=widgets.Layout(width="30%")
)

######


############ ############

cnmf_file_box = widgets.VBox()
cnmf_file_box.children = [cnmf_load_file_box, correlation_plot_btn]

basic_row0 = widgets.HBox()
basic_row0.children = [is_patches_widget]

basic_row1 = widgets.HBox()
basic_row1.children = [dslabel,ds_spatial_widget,ds_temporal_widget]
##
basic_row2 = widgets.HBox()
basic_row2.children = [k_widget, gSig_widget, gSiz_widget]

basic_row3 = widgets.HBox()
basic_row3.children = [stride_widget, rf_widget]

####

#min_corr, min_pnr
basic_row4 = widgets.HBox()
basic_row4.children = [min_corr_widget, min_pnr_widget]

basic_row5 = widgets.HBox()
basic_row5.children = [deconv_flag_widget, save_movie_widget, refine_components_widget]

cnmf_basic_settings = widgets.VBox()
cnmf_basic_settings.children = [basic_row0,basic_row1,basic_row2, basic_row3, basic_row4, basic_row5]

cnmf_advanced_settings = widgets.VBox()
cnmf_advanced_settings.children = [ \
	p_widget, \
	gnb_box, \
	merge_thresh_box, \
	is_dendrites_box, \
	min_snr_widget, \
	rval_thr_box, \
	cnn_thr_box, \
]
cnmf_settings = widgets.Accordion(children=[cnmf_basic_settings, cnmf_advanced_settings])
cnmf_settings.set_title(0, 'Basic CNMF Settings') #MC = Motion Correction
cnmf_settings.set_title(1, 'Advanced CNMF Settings')
