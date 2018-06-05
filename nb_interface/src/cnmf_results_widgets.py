from ipywidgets import interact, interactive, fixed, interact_manual, HBox, VBox, IntSlider, Play, jslink, Tab
import ipywidgets as widgets
from file_browser import FileBrowserBtn
import os
# UX

#main tab widgets
from main_widgets import *

#Motion Correction widgets:

from mc_widgets import *
#########  CNMF widgets

#tab_contents = ['P0', 'P1', 'P2', 'P3', 'P4']
#children = [widgets.Text(description=name) for name in tab_contents]
#tab = widgets.Tab()
#tab.children = children

from cnmf_widgets import *
####


#delete/refine ROIs control, and save data
'''delete_roi_btn = widgets.Button(
	description='Delete ROI',
	disabled=False,
	button_style='warning', # 'success', 'info', 'warning', 'danger' or ''
	tooltip='Exclude ROI'
)'''
'''delete_list_widget = widgets.SelectMultiple(
	options=[],
	value=[],
	rows=3,
	description='Exclud. ROIs',
	disabled=False,
	layout=widgets.Layout(width="18%")
)'''

roi_slider = IntSlider(min=1, step=1, description='ROI#', value=1)

is_edit_widget = widgets.ToggleButtons(
	options=['View', 'Edit'],
	description='Mode:',
	disabled=False,
	button_style='', # 'success', 'info', 'warning', 'danger' or ''
	tooltips=['View or refine the results'],
#     icons=['check'] * 3
)

# --------EDIT PANEL -------
from cnmf_results_edit_panel import *
#- ---- --- ----------------

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
	layout=widgets.Layout(width="25%")
)


#####


deconv_chk = widgets.ToggleButtons(
    options=['Signal', 'Deconvolution', 'Both'],
    description='Plot options:',
    disabled=False,
    button_style='', # 'success', 'info', 'warning', 'danger' or ''
    tooltips=['Denoised/demixed Ca2+ signal', 'Deconvolved Ca2+ signal', 'Plot both'],
	layout=widgets.Layout(width="18%"),
)



#####
#children = []
#for i in range(len(children)):
#    tab.set_title(i, str(i))
#tab

view_cnmf_results_widget = widgets.Button(
	description='View/Refine CNMF Results',
	disabled=False,
	button_style='info', # 'success', 'info', 'warning', 'danger' or ''
	tooltip='View CNMF Results',
	layout=widgets.Layout(width="30%")
)

##### Validation Tab: ######
############################

validate_col_mag_slider = widgets.IntSlider(
	value=2,
	min=1,
	max=10,
	step=1,
	#description='Movie Magnification:',
	disabled=False,
	continuous_update=False,
	orientation='horizontal',
	readout=True,
	readout_format='d',
	tooltip='How much to magnify the movie',
	layout=widgets.Layout(width="30%")
)
validate_col_mag_box = widgets.HBox([widgets.Label("Movie Magnification:"), validate_col_mag_slider])

validate_col_cnmf_mov_btn = widgets.Button(
	description='View CNMF Analyzed Movie',
	disabled=False,
	button_style='info', # 'success', 'info', 'warning', 'danger' or ''
	tooltip='View Background Movie',
	layout=widgets.Layout(width="30%")
)

validate_col_bgmov_btn = widgets.Button(
	description='View Background Movie',
	disabled=False,
	button_style='info', # 'success', 'info', 'warning', 'danger' or ''
	tooltip='View Background Movie',
	layout=widgets.Layout(width="30%")
)

validate_col_residual_btn = widgets.Button(
	description='View Residual Movie',
	disabled=False,
	button_style='info', # 'success', 'info', 'warning', 'danger' or ''
	tooltip='View Residual Movie',
	layout=widgets.Layout(width="30%")
)


#####
