import ipywidgets as widgets
import os
import bqplot
from file_browser import FileBrowserBtn
##### Motion Correction Widgets

layout_small = widgets.Layout(width="10%")

file_selector = widgets.Text(
	value= "../example_movies/demoMovie.tif",#os.getcwd(),
	placeholder=os.getcwd(),
	description='File/Folder Path:',
	layout=widgets.Layout(width='50%'),
	disabled=False
)

mc_file_browser_btn = FileBrowserBtn(desc='Browse')
mc_file_browser_btn.layout.width='9%'

load_files_btn = widgets.Button(
	description='Load Files',
	disabled=False,
	button_style='success', # 'success', 'info', 'warning', 'danger' or ''
	tooltip='Load Files',
	icon='check'
)

#####

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

#####

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

max_shifts_widget = widgets.BoundedFloatText(
	value=6,
	min=1,
	max=100,
	step=1,
	description='Max Shifts:',
	disabled=False,
	layout=ds_layout
)

is_rigid_widget = widgets.ToggleButtons(
	options=['Rigid', 'Non-Rigid'],
	description='MC Mode:',
	disabled=False,
	button_style='warning', # 'success', 'info', 'warning', 'danger' or ''
	tooltips=['Rigid correction (faster)', 'Non-rigid correction (slow, more accurate)'],
#     icons=['check'] * 3
)


######
run_mc_btn = widgets.Button(
	description='Run Motion Correction',
	disabled=False,
	button_style='info', # 'success', 'info', 'warning', 'danger' or ''
	tooltip='Run motion correction',
	layout=widgets.Layout(width="30%")
)

###### Advanced section

'''niter_rig = 1               # number of iterations for rigid motion correction
max_shifts = (6, 6)         # maximum allow rigid shift
# for parallelization split the movies in  num_splits chuncks across time
splits_rig = 56 #20
# start a new patch for pw-rigid motion correction every x pixels
strides = (48, 48)
# overlap between pathes (size of patch strides+overlaps)
overlaps = (24, 24)
# for parallelization split the movies in  num_splits chuncks across time
splits_els = 56
upsample_factor_grid = 4    # upsample factor to avoid smearing when merging patches
# maximum deviation allowed for patch with respect to rigid shifts
max_deviation_rigid = 3'''
niter_rig_widget = widgets.BoundedIntText(
    value=1,
	min = 1,
	max=10,
    #description=':',
    disabled=False,
	layout=layout_small
)
niter_rig_box = widgets.HBox([widgets.Label(value="Num iterations (rigid):"),niter_rig_widget])

splits_widget = widgets.BoundedIntText(
    value=20,
	min = 1,
	max = 1000,
    #description='',
    disabled=False,
	layout=layout_small
)

splits_box = widgets.HBox([widgets.Label(value="Num splits (parallelization):"),splits_widget])

mc_strides_widget = widgets.BoundedIntText(
    value=20,
	min=1,
	max=100,
    description='Strides:',
    disabled=True,
	layout=widgets.Layout(width="20%")
)

#strides_box = widgets.HBox([widgets.Label(value="Strides:"),strides_widget])

overlaps_widget = widgets.BoundedIntText(
    value=24,
	min=1,
	max=1000,
    description='Overlaps:',
    disabled=True,
	layout=widgets.Layout(width="20%")
)

#Will use single num splits widget for both rigid and non-rigid
'''splits_els_widget = widgets.Text(
    value='Hello World',
    placeholder='Type something',
    description='String:',
    disabled=False
)'''

upsample_factor_grid_widget = widgets.BoundedIntText(
    value=4,
    disabled=True,
	layout=layout_small
)
upsample_factor_box = widgets.HBox([widgets.Label(value="Upsample Factor:"),upsample_factor_grid_widget])

max_deviation_rigid_widget = widgets.BoundedIntText(
    value=3,
    disabled=True, #disabled because rigid mode is default
	layout=layout_small
)
max_deviation_box = widgets.HBox([widgets.Label(value="Max Deviation (rigid):"),max_deviation_rigid_widget])

######        ##########
file_box = widgets.HBox()
file_box.children = [file_selector, mc_file_browser_btn, load_files_btn]

ds_factors_box = widgets.HBox()
ds_factors_box.children = [dsx_widget, dsy_widget, dst_widget]

mc_line3 = widgets.HBox()
mc_line3.children = [gSigFilter_widget, max_shifts_widget]

basic_settings = widgets.VBox()
basic_settings.children = [dslabel, ds_factors_box, mc_line3, is_rigid_widget]

advanced_settings = widgets.VBox()
advanced_settings.children = [ \
	niter_rig_box,
	splits_box,
	mc_strides_widget,
	overlaps_widget,
	upsample_factor_box,
	max_deviation_box
]
settings = widgets.Accordion(children=[basic_settings, advanced_settings])
settings.set_title(0, 'Basic MC Settings') #MC = Motion Correction
settings.set_title(1, 'Advanced MC Settings')

major_col = widgets.VBox()
major_col.children = [file_box,is_batch_widget,settings, run_mc_btn]

#### MC results section

mc_mov_mag_widget = widgets.IntSlider(
    value=2,
    min=1,
    max=10,
    step=1,
    #description='Magnification:',
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='d',
    #layout=widgets.Layout(width="40%")
)
mc_mov_mag_box = widgets.HBox([widgets.Label(value="Magnification:"),mc_mov_mag_widget])

mc_mov_gain_widget = widgets.IntSlider(
    value=4,
    min=0,
    max=100,
    step=1,
    description='Gain:',
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='d'
)

play_mov_btn = widgets.Button(
	description='Play Movies',
	disabled=False,
	button_style='success', # 'success', 'info', 'warning', 'danger' or ''
	tooltip='Play Movies'
)


#Extracted shifts plot
'''shifts_plot = bqplot.Figure(marks=[bar, line], axes=[ax_x, ax_y], title='API Example',
       legend_location='bottom-right')'''

sc_x = bqplot.LinearScale()
sc_y = bqplot.LinearScale(min= (-1 * float(max_shifts_widget.value) / 3), max=(float(max_shifts_widget.value) / 3))

xshifts_line = bqplot.Lines(scales={'x': sc_x, 'y': sc_y},
             stroke_width=3, colors=['red'], display_legend=True, labels=['X Shifts'])

yshifts_line = bqplot.Lines(scales={'x': sc_x, 'y': sc_y},
             stroke_width=3, colors=['orange'], display_legend=True, labels=['Y Shifts'])

xshifts_scat = bqplot.marks.Scatter(scales={'x':bqplot.LinearScale(), 'y':bqplot.LinearScale()}, colors=['red'], display_legend=True, labels=['X Shifts'])
yshifts_scat = bqplot.marks.Scatter(scales={'x':bqplot.LinearScale(), 'y':bqplot.LinearScale()}, colors=['orange'], display_legend=True, labels=['Y Shifts'])

ax_x = bqplot.Axis(scale=sc_x, grid_lines='solid', label='Time/Frames') #Time / Frames
ax_y = bqplot.Axis(scale=sc_y, orientation='vertical',
            grid_lines='solid', label='Shift') #shift

#marks=[xshifts_line, yshifts_line]
shifts_plot = bqplot.Figure(axes=[ax_x, ax_y], title='MC Extracted Shifts',
       legend_location='bottom-right')

tb_shifts = bqplot.Toolbar(figure=shifts_plot)
mc_shifts_box = widgets.VBox([tb_shifts, shifts_plot])
mc_shifts_box.layout.display = 'None'

mc_results_box = widgets.VBox()

mc_results_box.children = [mc_mov_mag_box, mc_mov_gain_widget, play_mov_btn, mc_shifts_box]
