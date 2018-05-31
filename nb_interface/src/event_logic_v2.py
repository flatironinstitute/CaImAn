import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from scipy.interpolate import UnivariateSpline
from typing import Dict, Tuple, List
from IPython.display import display, HTML
#from ipywidgets import interact, interactive, fixed, interact_manual, HBox, VBox, IntSlider, Play, jslink, Output
#import ipywidgets as widgets
from IPython.display import clear_output
import bqplot
from bqplot import (
    LogScale, LinearScale, OrdinalColorScale, ColorAxis, OrdinalScale,
    Axis, Scatter, Lines, CATEGORY10, Label, Figure, Tooltip, Toolbar, Bars
)
import traitlets
import qgrid
#%matplotlib inline
from event_widgets import *
from caiman_interface import context
import peakutils

def setup_context(context):
    if len(context.cnmf_results) > 0:
        signal_data = context.cnmf_results[1]
    else:
        signal_data = None
    return signal_data

def detect_events(signal, base_shifted, thresh=0.1, min_dist=50):
    # list of events, each event is itself a list of the form
    # start (index), peak value, peak index, end (index); need to convert indices to time later
    events = [] #[start index, peak index, end index]
    #main loop
    in_event = False
    cur_event = [None, None, None] #start index, peak index, end index
    #params
    lag = 5 #used to compute average of window around baseline pt
    indexes = peakutils.indexes(signal, thres=thresh, min_dist=50) #finds peak indices
    for peak in indexes:
        #first find left bound, then right bound
        start = peak
        for i in range(peak,0,-1):
            if signal[i] <= np.mean(base_shifted[(i-lag):(i+lag)]):
                start = i
                break;
            if (peak - i) > 166:
                start = i
                break;
        end = peak
        for j in range(peak,len(signal),1):
            if signal[j] <= np.mean(base_shifted[(j-lag):(j+lag)]):
                end = j
                break;
            if (j - peak) > 166:
                end = j
                break;
        event = [start, peak, end]
        events.append(event)

    return events


#[0] event start, [1] event peak, [2] event end, amplitude, duration, rise-time, decay-time, area
def analyze_event(signal, event, fr):
    #new encoding: [0] start index, [1] peak index, [2] end index
    start_time = event[0] * fr + fr
    end_time = event[2] * fr + fr
    peak_time = event[1] * fr + fr
    amplitude = signal[event[1]]
    duration = end_time - start_time
    rise_time = peak_time - start_time
    decay_time = end_time - peak_time
    area = np.trapz(signal[event[0]:event[2]],dx=0.33)
    #calculuate half-width
    spike = signal[event[0]:event[2]]
    x = np.arange(len(spike))
    try:
        spline = UnivariateSpline(x,spike-np.max(spike)/2, s=0)
        r1, r2 = spline.roots() # find the roots
    except Exception:
        r1,r2 = 0,0
    half_width = (r2-r1)*fr + fr
    return start_time, end_time, peak_time, amplitude, duration, half_width, rise_time, decay_time, area

#Iterate through events, call analyze_roi(...) to compute stats for each event
def compute_stats(signal, events, fr):
    results = np.zeros((len(events),9))
    columns=['Start', 'End', 'Peak Time','Amplitude', 'Duration', 'Half-Width', 'rise-time', 'decay-time', 'Area']
    df = pd.DataFrame(data=results, columns=columns)
    #event: [start index, peak value, peak index, end index]
    c = 0
    for event in events:
        df.iloc[c,:] = analyze_event(signal, event, fr)
        c += 1
    return df

#Convert frame index to time in milliseconds
def frames_to_time(a, fr=0.0333):
    return a * fr + fr

#Compute event statistics
def analyze_roi(signal, min_thresh, min_dist, fr):
    ####
    #def detect_events(signal, base_shifted, thresh=0.1, min_dist=50):
    base = peakutils.baseline(signal, 12) #Fit polynomial baseline to each signal
    shift_amt = 0.05 * np.max(signal)
    base_shifted = base + shift_amt

    events = detect_events(signal, base_shifted, min_thresh, min_dist) #list of lists of start, peak, end indices
    event_results = compute_stats(signal, events, fr)
    return events, event_results, base #baseline

#Analyze all ROI signals
def analyze_all(signals, min_thresh, min_dist, fr):
    #list of tuples, each tuple is an ROI
    #tuple of: x_range :ndarray, base :ndarray, events_x, events_y, thresh_line_y : ndarray, event_results : List
    results = []
    #print(len(signals))
    for i in range(len(signals)): #for each ROI
        #print(i)
        results.append(analyze_roi(signals[i], min_thresh, min_dist, fr))

    return results


#convert list of tuples from analyze_all to a dataframe
def convToDF(results):
    #rows = rois * #events
    #df_init = np.zeros((len(results),9))
    columns=['ROI', 'Start', 'End', 'Peak Time','Amplitude', 'Duration', 'Half-Width', 'rise-time', 'decay-time', 'Area']
    df = pd.DataFrame(columns=columns)
    #event: [start index, peak value, peak index, end index]
    c = 0
    for result in results:
        _, event_results, _ = result
        #print("Result {}".format(event_results.shape))
        #df.iloc[c,:] = event_results
        event_results['ROI'] = c + 1
        df = pd.concat([df, event_results], ignore_index=True)
        c += 1
    #print(len(df))
    df = df[columns]
    return df

#start_event_btn.on_click(show_plot)
settings_box = HBox([min_thresh_widget, min_dist_widget, fr_widget])
results_area = VBox()
ret_box = VBox([settings_box,start_event_btn,results_area])

def set_event_widgets():
    return ret_box

def get_event_pts(signal, events):
    start_pts = [x[0] for x in events]
    start_pts_x = list(map(frames_to_time, start_pts))
    end_pts = [x[2] for x in events]
    end_pts_x = list(map(frames_to_time, end_pts))
    start_pts_y = signal[start_pts]
    peaks_indices = [x[1] for x in events]
    peaks_x = list(map(frames_to_time, peaks_indices))
    peaks_y = signal[peaks_indices]
    end_pts_y = signal[end_pts]

    return start_pts_x, start_pts_y, peaks_x, peaks_y, end_pts_x, end_pts_y

def show_plot():
    #clear_output()
    #ret_box = show_start_menu()
    signal_data = setup_context(context) #array of arrays (matrix), each row is a ROI signal
    roi_slider_widget.max = signal_data.shape[0]
    min_thresh = float(min_thresh_widget.value)
    min_dist = float(min_dist_widget.value)
    fr = float(fr_widget.value) #duration of each frame in seconds
    bump_perc = 0.05 #added to peak y values for aesthetics when plotting (percentage of max value)
    cur_roi = (roi_slider_widget.value - 1)
    #print(df_results.head(n=25))
    signal = signal_data[cur_roi] #intial signal to display
    bump = np.max(signal) * bump_perc
    x_range = np.arange(0,frames_to_time(len(signal)), step=fr)
        #plt.plot(x_range,y_thresh)

    thresh = min_thresh * np.max(signal)
    #thresh_line_y = np.repeat(min_thresh,len(signal))
    thresh_line_y = np.repeat(thresh,len(signal))

    ##### Analyze all Signal Data for Events
    results = analyze_all(signal_data, min_thresh, min_dist, fr)
    df_results = convToDF(results) #convert numpy array `results` to a DataFrame, so we can nicely display using qgrid

    events, event_results, base = results[cur_roi] #get events for current ROI signal
    #get each event's start, peak and end points
    start_pts_x, start_pts_y, peaks_x, peaks_y, end_pts_x, end_pts_y = get_event_pts(signal, events)

    x_sc = LinearScale()
    y_sc = LinearScale()
    scat_tt = Tooltip(fields=['index'], formats=['i'], labels=['Event#'])
    sig_line = Lines(scales={'x': x_sc, 'y': y_sc},
                 stroke_width=3, colors=['blue'], display_legend=True, labels=['Ca Signal'])

    #Threshold line (universal for all signals)
    thresh_line = Lines(x=x_range, y=thresh_line_y, scales={'x': x_sc, 'y': y_sc},
                 stroke_width=1.5, colors=['orange'], display_legend=True, labels=['Threshold'],
                 opacities=[0.9], line_style='dashed')
     #Baseline (depends on each ROI)
    base_line = Lines(x=x_range, y=base, scales={'x': x_sc, 'y': y_sc},
                 stroke_width=1.5, colors=['gray'], display_legend=True, labels=['Baseline'])

    #Start, peak, and end points for each event
    start_pts_scat = Scatter(x=start_pts_x, y=start_pts_y, scales={'x': x_sc, 'y': y_sc}, tooltip=scat_tt,
                 stroke_width=3, colors=['green'], display_legend=True, labels=['Event Start'], marker='diamond')
    peaks_scat = Scatter(x=peaks_x, y=(peaks_y+bump), scales={'x': x_sc, 'y': y_sc}, tooltip=scat_tt,
                 stroke_width=3, colors=['red'], display_legend=True, labels=['Event Peak'], marker='triangle-down')
    end_pts_scat = Scatter(x=end_pts_x, y=end_pts_y, scales={'x': x_sc, 'y': y_sc}, tooltip=scat_tt,
                 stroke_width=3, colors=['purple'], display_legend=True, labels=['Event End'], marker='cross')

    ax_x = Axis(scale=x_sc, grid_lines='solid', label='Time (seconds)')
    ax_y = Axis(scale=y_sc, orientation='vertical', tick_format='0.2f',
                grid_lines='solid', label='Amplitude')

    fig = Figure(marks=[sig_line, base_line, thresh_line, start_pts_scat, peaks_scat, end_pts_scat],
                 axes=[ax_x, ax_y], title='Event Detection & Analysis', legend_location='top-left')
    tb0 = Toolbar(figure=fig)

    out = Output()
    event_results_widget = qgrid.QgridWidget(df=df_results, show_toolbar=True)
    def update_plot(change):
        cur_roi2 = (roi_slider_widget.value - 1)
        new_signal = signal_data[cur_roi2]
        bump = np.max(new_signal) * bump_perc
        #x_range, y_base, events_x, events_y, thresh_line_y, event_results = analyze_roi(new_signal)
        events, event_results, base2 = results[cur_roi2]
        start_pts_x, start_pts_y, peaks_x, peaks_y, end_pts_x, end_pts_y = get_event_pts(new_signal, events)
        thresh = min_thresh * np.max(new_signal)
        thresh_line_y = np.repeat(thresh,len(new_signal))

        sig_line.y = new_signal
        sig_line.x = x_range
        thresh_line.x = x_range
        thresh_line.y = thresh_line_y
        base_line.x = x_range
        base_line.y = base2
        start_pts_scat.x = start_pts_x
        start_pts_scat.y = start_pts_y
        peaks_scat.x = peaks_x
        peaks_scat.y = (peaks_y+bump)
        end_pts_scat.x = end_pts_x
        end_pts_scat.y = end_pts_y
        #event_results_widget.df = event_results
        indx_range = df_results.loc[df_results['ROI'] == (cur_roi2+1)].index.values.tolist()
        if len(indx_range) == 0:
            indx_range=[1,1]
        min_ind = indx_range[0]
        max_ind = indx_range[-1]
        event_results_widget._handle_qgrid_msg_helper({
        'field': "index",
        'filter_info': {
            'field': "index",
            'max': max_ind,
            'min': min_ind,
            'type': "slider"
        },
        'type': "filter_changed"
        })
    def dl_events_click(_):
        event_results_widget.get_changed_df().to_csv(path_or_buf = context.working_dir + 'events_data.csv')
        print("Data saved to current working directory as: events_data.csv")
    dl_events_data_btn.on_click(dl_events_click)

    def reset_filters(_):
        event_results_widget._handle_qgrid_msg_helper({
        'field': "index",
        'filter_info': {
            'field': "index",
            'max': 9999,
            'min': 0,
            'type': "slider"
        },
        'type': "filter_changed"
        })
    show_all_events_btn.on_click(reset_filters)

    update_plot({'new':1})
    roi_slider_widget.observe(update_plot, names='value')
    fig_widget = VBox([HBox([VBox([roi_slider_widget,tb0,fig])])])
    btns = HBox([show_all_events_btn,dl_events_data_btn])
    figslist = VBox([fig_widget,event_results_widget,btns])
    return figslist
    #display(figslist)
    #display(df_results)
#show_start_menu()
def update_figs(_):
    figslist = show_plot()
    ret_box.children = [settings_box,start_event_btn,figslist]

start_event_btn.on_click(update_figs)
