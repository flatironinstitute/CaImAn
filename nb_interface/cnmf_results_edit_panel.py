from ipywidgets import interact, interactive, fixed, interact_manual, HBox, VBox, IntSlider, Play, jslink, Tab
import ipywidgets as widgets
import os

#---------------- EDIT PANEL
# need: min_SNR, r_values_min, r_values_lowest, thresh_cnn_min,
# thresh_cnn_lowest,thresh_fitness_delta, min_std_reject
# SAME: min_SNR=2, r_values_min=0.9, r_values_lowest=-1, thresh_cnn_min=0.95, thresh_cnn_lowest=0.1,
# thresh_fitness_delta=-20., min_std_reject=0.5,
min_snr_edit_widget = widgets.BoundedFloatText(
	value=2.5,
	min=1,
	max=1000,
	step=0.5,
	description='Min SNR:',
	#tooltip='Number of global background components',
	disabled=False,
	layout=widgets.Layout(width="70%", margin='50px 0px 0px 0px')
)

rvalmin_edit_widget_ = widgets.BoundedFloatText(
	value=0.8,
	min=0.0,
	max=50.0,
	step=0.1,
	#description='R Values Min:',
	disabled=False,
	layout=widgets.Layout(width="50%")
)
rvalmin_edit_widget = widgets.HBox([widgets.Label(value="R Values Min:"),rvalmin_edit_widget_])

rvallowest_edit_widget_ = widgets.BoundedFloatText(
	value=0.8,
	min=0.0,
	max=50.0,
	step=0.1,
	#description='R Values Lowest:',
	disabled=False,
	layout=widgets.Layout(width="50%")
)
rvallowest_edit_widget = widgets.HBox([widgets.Label(value="R Values Lowest:"),rvallowest_edit_widget_])

cnnmin_edit_widget_ = widgets.BoundedFloatText(
	value=0.8,
	min=0.0,
	max=50.0,
	step=0.1,
	#description='Thresh CNN Min:',
	disabled=False,
	layout=widgets.Layout(width="50%")
)
cnnmin_edit_widget = widgets.HBox([widgets.Label(value="Thresh CNN Min:"),cnnmin_edit_widget_])

cnnlowest_edit_widget_ = widgets.BoundedFloatText(
	value=0.8,
	min=0.0,
	max=50.0,
	step=0.1,
	#description='Thresh CNN Lowest:',
	disabled=False,
	layout=widgets.Layout(width="50%")
)
cnnlowest_edit_widget = widgets.HBox([widgets.Label(value="Thresh CNN Lowest:"),cnnlowest_edit_widget_])

fitness_delta_edit_widget_ = widgets.BoundedFloatText(
	value=0.8,
	min=0.0,
	max=50.0,
	step=0.1,
	#description='Thresh Fitness Delta:',
	disabled=False,
	layout=widgets.Layout(width="50%")
)
fitness_delta_edit_widget = widgets.HBox([widgets.Label(value="Thresh Fitness Delta:"),fitness_delta_edit_widget_])

minstdreject_edit_widget_ = widgets.BoundedFloatText(
	value=0.8,
	min=0.0,
	max=50.0,
	step=0.1,
	#description='Min Std Reject:',
	disabled=False,
	layout=widgets.Layout(width="50%")
)
minstdreject_edit_widget = widgets.HBox([widgets.Label(value="Min Std Reject:"),minstdreject_edit_widget_])

update_edit_btn = widgets.Button(
	description='Update',
	disabled=False,
	button_style='info', # 'success', 'info', 'warning', 'danger' or ''
	tooltip='Refine ROIs',
	layout=widgets.Layout(width="50%")
)

edit_panel_widget = widgets.VBox([min_snr_edit_widget, rvalmin_edit_widget, rvallowest_edit_widget, \
			cnnmin_edit_widget, cnnlowest_edit_widget, fitness_delta_edit_widget, minstdreject_edit_widget, \
			update_edit_btn], layout=widgets.Layout(display='None'))

#--------------- END EDIT PANEL
