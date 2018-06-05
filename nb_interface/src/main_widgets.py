import ipywidgets as widgets
import os
from file_browser import FileBrowserBtn, DirBrowserBtn

#ds_layout_main = widgets.Layout(width="20%")

workingdir_selector = widgets.Text(
	value=os.getcwd(),
	placeholder=os.getcwd(),
	#description='Working Directory:',
	layout=widgets.Layout(width='35%'),
	disabled=False
)

wkdir_browser_btn = DirBrowserBtn(desc='Browse')
wkdir_browser_btn.layout.width='9%'

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

context_browser_btn = FileBrowserBtn(desc='Browse')
context_browser_btn.layout.width='9%'

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

#### Basic movie parameters

main_params_label = widgets.HTML(
    value="<b>Basic Movie Parameters:</b>",
    #placeholder='Some HTML',
    #description='Some HTML',
)

fr_widget = widgets.BoundedIntText(
	value=30,
	min=1,
	max=500,
	step=1,
	#description='Frame Rate (frames/sec):',
	disabled=False,
	layout=widgets.Layout(width='10%')
)
decay_time_widget = widgets.BoundedFloatText(
	value=0.4,
	min=0.001,
	max=10.0,
	step=0.05,
	#description='Decay Time (Ca2+ Transient Duration):',
	disabled=False,
	layout=widgets.Layout(width='10%')
)

microscopy_type_widget = widgets.Dropdown(
    options={'One-Photon': 1, 'Two-Photon': 2,},
    value=2,
    #description='',
)
microscopy_type_box = widgets.HBox([widgets.Label("Microscopy Type:"),microscopy_type_widget])

# Layout

wkdir_box = widgets.HBox()
wkdir_box.children = [widgets.Label("Set Working Directory:"), workingdir_selector, wkdir_browser_btn, workingdir_btn]
context_box = widgets.HBox()
context_box.children = [widgets.Label("Load context from:"),context_path_txt, context_browser_btn, context_load_btn, context_save_btn]
wkdir_context_box = widgets.VBox()
main_params = widgets.VBox()
main_params.children = [main_params_label,widgets.HBox([widgets.Label(value="Frame Rate (frames/sec):"),fr_widget]), \
	widgets.HBox([widgets.Label(value="Decay Time (Ca2+ Transient Duration):"),decay_time_widget]), \
	microscopy_type_box \
	]

wkdir_context_box.children = [wkdir_box,context_box,main_params]
