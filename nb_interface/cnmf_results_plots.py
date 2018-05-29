from ipywidgets import interact, interactive, fixed, interact_manual, HBox, VBox, IntSlider, Play, jslink, Tab
import ipywidgets as widgets
import bqplot
from bqplot import (
	LogScale, LinearScale, OrdinalColorScale, ColorAxis,
	Axis, Scatter, Lines, CATEGORY10, Label, Figure, Tooltip, Toolbar
)
import os


# This file defines the bqplot components for the cnmf_results viewer

scale_x = bqplot.LinearScale(min=0.0, max=1) #for images
scale_y = bqplot.LinearScale(min=0.0, max=1) #for images
scale_x2 = bqplot.LinearScale(min=0.0) #eg 376
scale_y2 = bqplot.LinearScale(min=0.0) #eg 240
########

#bqplot.Image needs to have its image attribute initialized at creation and it needs scales

cor_image = widgets.Image(
	format='png',
	)#,width=376,height=240


a_image = widgets.Image(
	format='png',
	)#,width=376,height=240

#individiaul roi spatial map
roi_image = widgets.Image(
#	value = get_roi_image(A,1,dims),
	format='png',
#	width=dims[1],
#	height=dims[0],
	)
#######
#correlation image, background
cor_image_mark = bqplot.Image(image = cor_image, scales={'x': scale_x, 'y': scale_y})#, scales={'x': scale_x, 'y': scale_y}
#reconstructed neuronal spatial maps
full_a_mark = bqplot.Image(image = a_image, scales={'x': scale_x, 'y': scale_y})#, scales={'x': scale_x2, 'y': scale_y2}

roi_image_mark = bqplot.Image(image = roi_image, scales={'x': scale_x, 'y': scale_y})#, scales={'x': scale_x2, 'y': scale_y2}

rois = Scatter(scales={'x': scale_x2, 'y': scale_y2}, default_size=30,
			  unhovered_style={'opacity': 0.3}, colors=['red'], default_opacity=0.1, selected=[0])
rois.interactions = {'click': 'select'}
rois.selected_style = {'opacity': 1.0, 'fill': 'Black', 'stroke': 'Black', 'size':30}

#roi_slider = IntSlider(min=1, step=1, description='ROI#', value=1)


######
contour_mark = bqplot.Lines(colors=['yellow'], scales={'x': scale_x2, 'y': scale_y2})

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
scale_x4 = bqplot.LinearScale(min=0.0)
scale_x5 = bqplot.LinearScale(min=0.0)

scale_y4 = bqplot.LinearScale(min=0.0) # add 10% to give some upper margin
scale_y5 = bqplot.LinearScale(min=0.0) # add 10% to give some upper margin
signal_mark = bqplot.Lines(colors=['black'],
						   scales={'x': scale_x4, 'y': scale_y4}, display_legend=True)
deconv_signal_mark = bqplot.Lines(colors=['red'],
						  scales={'x': scale_x5, 'y': scale_y5}, display_legend=True, visible=False)

fig4 = bqplot.Figure(padding_x=0, padding_y=0, title='Denoised/Demixed Fluorescence Trace',
					 background_style={'background-color':'white'})
fig4.marks = [signal_mark, deconv_signal_mark]
fig4.axes = [bqplot.Axis(scale=scale_x4, label='Time (Frame #)',grid_lines='none'), \
	bqplot.Axis(scale=scale_y4, orientation='vertical',label='Amplitude',grid_lines='none')]
tb0 = Toolbar(figure=fig4)
