from ipywidgets import interact, interactive, fixed, interact_manual, HBox, VBox, IntSlider, Play, jslink, Output
import ipywidgets as widgets


roi_slider_widget = widgets.BoundedIntText(
    value=2,
    min=1,
    max=200,
    step=1,
    description='ROI#:',
    disabled=False,
    width=300
)
min_thresh_widget = widgets.BoundedFloatText(
    value=0.1,
    min=0.000,
    max=10.000,
    step=0.001,
    description='Min Thresh:',
    disabled=False,
    readout=True,
    readout_format='.3f',
)
min_dist_widget = widgets.BoundedFloatText(
    value=50.0,
    min=0,
    max=100.0,
    step=0.001,
    description='Min Dist:',
    disabled=False,
    readout=True,
    readout_format='.3f',
)
fr_widget = widgets.BoundedFloatText(
    value=0.03333,
    min=0,
    max=10.0,
    step=0.01,
    description='Frame Dur:',
    disabled=False
)





start_event_btn = widgets.Button(
    description='Event Detection',
    disabled=False,
    button_style='success', # 'success', 'info', 'warning', 'danger' or ''
    tooltip='Event Detection & Analysis',
    icon='check',
    width=200,
)

dl_events_data_btn = widgets.Button(
    description='Download Data',
    disabled=False,
    button_style='info', # 'success', 'info', 'warning', 'danger' or ''
    tooltip='Download Events Data as CSV File',
    icon='check',
    width=200,
)

show_all_events_btn = widgets.Button(
    description='Show All Events',
    disabled=False,
    button_style='', # 'success', 'info', 'warning', 'danger' or ''
    tooltip='Show all events in data table',
    icon='check',
    width=200,
)
