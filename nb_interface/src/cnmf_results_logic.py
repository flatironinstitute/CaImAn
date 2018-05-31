import bqplot
from bqplot import (
	LogScale, LinearScale, OrdinalColorScale, ColorAxis,
	Axis, Scatter, Lines, CATEGORY10, Label, Figure, Tooltip, Toolbar
)
from ipywidgets import interact, interactive, fixed, interact_manual, HBox, VBox, IntSlider, Play, jslink, Tab
import ipywidgets as widgets
import os

def load_context_data(context):
	if len(context.cnmf_results) == 8: #for backward compatibility
		A, C, b, f, YrA, sn, idx_components, conv = context.cnmf_results
	else:
		A, C, b, f, YrA, sn, idx_components = context.cnmf_results
		conv = None
	return A, C, b, f, YrA, sn, idx_components, conv
