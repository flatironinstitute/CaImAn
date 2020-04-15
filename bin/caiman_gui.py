#!/usr/bin/env python

import cv2
from datetime import datetime
from dateutil.tz import tzlocal
import numpy as np
import pyqtgraph as pg
import scipy
import os
from pyqtgraph import FileDialog
from pyqtgraph.Qt import QtGui
from pyqtgraph.parametertree import Parameter, ParameterTree
from scipy.sparse import csc_matrix

import caiman as cm
from caiman.source_extraction.cnmf.cnmf import load_CNMF
from caiman.source_extraction.cnmf.params import CNMFParams

# Always start by initializing Qt (only once per application)
app = QtGui.QApplication([])

try:
    cv2.setNumThreads(1)
except:
    print('Open CV is naturally single threaded')

try:
    if __IPYTHON__:
        print(1)
        # this is used for debugging purposes only. allows to reload classes
        # when changed
        get_ipython().magic('load_ext autoreload')
        get_ipython().magic('autoreload 2')
except NameError:
    print('Not launched under iPython')


def make_color_img(img, gain=255, min_max=None, out_type=np.uint8):
    if min_max is None:
        min_ = img.min()
        max_ = img.max()
    else:
        min_, max_ = min_max

    img = (img-min_)/(max_-min_)*gain
    img = img.astype(out_type)
    img = np.dstack([img]*3)
    return img


F = FileDialog()

# load object saved by CNMF
fpath = F.getOpenFileName(caption='Load CNMF Object',
                          filter='HDF5 (*.h5 *.hdf5);;NWB (*.nwb)')[0]
cnm_obj = load_CNMF(fpath)

# movie
if not os.path.exists(cnm_obj.mmap_file):
    M = FileDialog()
    cnm_obj.mmap_file = M.getOpenFileName(caption='Load memory mapped file',
                                          filter='*.mmap')[0]

if fpath[-3:] == 'nwb':
    mov = cm.load(cnm_obj.mmap_file,
                  var_name_hdf5='acquisition/TwoPhotonSeries')
else:
    mov = cm.load(cnm_obj.mmap_file)

estimates = cnm_obj.estimates
params_obj = cnm_obj.params
dims = estimates.dims                           #original dimension
estimates.rotation = False                       # flag for rotation
min_mov = np.min(mov)
max_mov = np.max(mov)


if not hasattr(estimates, 'Cn'):
    estimates.Cn = cm.local_correlations(mov, swap_dim=False)
#Cn = estimates.Cn

# We rotate our components 90 degrees right because of incompatiability of pyqtgraph and pyplot
def rotate90(img, right=None, vector=None, sparse=False):
    # rotate the img 90 degrees
    # we first transpose the img then flip axis
    # If right is ture, then rotate 90 degrees right, otherwise, rotate left
    # If vector is True, then we first reshape the spatial 1D vector to 2D then rotate
    # If vector is False, then we directly rotate the matrix
    global dims
    a = int(right)
    if vector == False:
        img = img.transpose([1,0])
        img = np.flip(img, axis=a)
    elif vector == True:
        if sparse == False:
            img = img.reshape((dims[1-a], dims[a], img.shape[1]), order='F')
            img = img.transpose([1,0,2])
            img = np.flip(img, axis=a)
            img = img.reshape((dims[0]*dims[1], img.shape[2]), order='F')
        else:
            img = img.toarray()
            img = img.reshape((dims[1-a], dims[a], img.shape[1]), order='F')
            img = img.transpose([1,0,2])
            img = np.flip(img, axis=a)
            img = img.reshape((dims[0]*dims[1], img.shape[2]), order='F')
            img = csc_matrix(img)
    return img

estimates.Cn = rotate90(estimates.Cn, right=True, vector=False)  
estimates.A = rotate90(estimates.A, right=True, vector=True, sparse=True)
estimates.b = rotate90(estimates.b, right=True, vector=True)
estimates.dims = (estimates.dims[1], estimates.dims[0])
estimates.rotation = True               # rotation flag becomes true after rotation

min_mov_denoise = np.min(estimates.A)*estimates.C.min()
max_mov_denoise = np.max(estimates.A)*estimates.C.max()
background_num = -1
neuron_selected = False
nr_index = 0

min_background = np.min(estimates.b, axis=0)*np.min(estimates.f, axis=1)
max_background = np.max(estimates.b, axis=0)*np.max(estimates.f, axis=1)

accepted_empty = True                       # check if accepted list exist and empty
if hasattr(estimates, 'accepted_list'):
    accepted_empty = (len(estimates.accepted_list)==0)

if (not hasattr(estimates, 'accepted_list')) or accepted_empty:
    # if estimates.discarded_components.A.shape[-1] > 0:
    #     estimates.restore_discarded_components()
    estimates.accepted_list = np.array([], dtype=np.int)
    estimates.rejected_list = np.array([], dtype=np.int)

estimates.img_components = estimates.A.toarray().reshape((estimates.dims[0], estimates.dims[1], -1), order='F').transpose([2,0,1])
estimates.cms = np.array([scipy.ndimage.measurements.center_of_mass(comp) for comp in estimates.img_components])
estimates.idx_components = np.arange(estimates.nr)
estimates.idx_components_bad = np.array([])
estimates.background_image = make_color_img(estimates.Cn)
# Generate image data
estimates.img_components /= estimates.img_components.max(axis=(1, 2))[:, None, None]
estimates.img_components *= 255
estimates.img_components = estimates.img_components.astype(np.uint8)


def draw_contours_overall(md):
    if md == "reset":
        draw_contours()
    elif md == "neurons":
        if neuron_selected is True:
            #  if a specific neuron has been selected, only one contour
            #  should be changed while thrshcomp_line is changing
            if nr_index == 0:
                #  if user does not start to move through the frames
                draw_contours_update(estimates.background_image, img)
                draw_contours_update(comp2_scaled, img2)
            else:
                draw_contours_update(raw_mov_scaled, img)
                draw_contours_update(frame_denoise_scaled, img2)
        else:
            #  if no specific neuron has been selected, redraw all the contours
            draw_contours()
    else:
        #  md is "background":
        return


def draw_contours():
    global thrshcomp_line, estimates, img
    bkgr_contours = estimates.background_image.copy()

    if len(estimates.idx_components) > 0:
        contours = [cv2.findContours(cv2.threshold(img, np.int(thrshcomp_line.value()), 255, 0)[1], cv2.RETR_TREE,
                                     cv2.CHAIN_APPROX_SIMPLE)[0] for img in estimates.img_components[estimates.idx_components]]
        SNRs = np.array(estimates.r_values)
        iidd = np.array(estimates.idx_components)

        idx1 = np.where(SNRs[iidd] < 0.1)[0]
        idx2 = np.where((SNRs[iidd] >= 0.1) &
                        (SNRs[iidd] < 0.25))[0]
        idx3 = np.where((SNRs[iidd] >= 0.25) &
                        (SNRs[iidd] < 0.5))[0]
        idx4 = np.where((SNRs[iidd] >= 0.5) &
                        (SNRs[iidd] < 0.75))[0]
        idx5 = np.where((SNRs[iidd] >= 0.75) &
                        (SNRs[iidd] < 0.9))[0]
        idx6 = np.where(SNRs[iidd] >= 0.9)[0]

        cv2.drawContours(bkgr_contours, sum([contours[jj] for jj in idx1], []), -1, (255, 0, 0), 1)
        cv2.drawContours(bkgr_contours, sum([contours[jj] for jj in idx2], []), -1, (0, 255, 0), 1)
        cv2.drawContours(bkgr_contours, sum([contours[jj] for jj in idx3], []), -1, (0, 0, 255), 1)
        cv2.drawContours(bkgr_contours, sum([contours[jj] for jj in idx4], []), -1, (255, 255, 0), 1)
        cv2.drawContours(bkgr_contours, sum([contours[jj] for jj in idx5], []), -1, (255, 0, 255), 1)
        cv2.drawContours(bkgr_contours, sum([contours[jj] for jj in idx6], []), -1, (0, 255, 255), 1)

    img.setImage(bkgr_contours, autoLevels=False)

pg.setConfigOptions(imageAxisOrder='col-major')

def draw_contours_update(cf, im):
    global thrshcomp_line, estimates
    curFrame = cf.copy()

    if len(estimates.idx_components) > 0:
        contours = [cv2.findContours(cv2.threshold(img, np.int(thrshcomp_line.value()), 255, 0)[1], cv2.RETR_TREE,
                                     cv2.CHAIN_APPROX_SIMPLE)[0] for img in estimates.img_components[estimates.idx_components]]
        SNRs = np.array(estimates.r_values)
        iidd = np.array(estimates.idx_components)

        idx1 = np.where(SNRs[iidd] < 0.1)[0]
        idx2 = np.where((SNRs[iidd] >= 0.1) &
                        (SNRs[iidd] < 0.25))[0]
        idx3 = np.where((SNRs[iidd] >= 0.25) &
                        (SNRs[iidd] < 0.5))[0]
        idx4 = np.where((SNRs[iidd] >= 0.5) &
                        (SNRs[iidd] < 0.75))[0]
        idx5 = np.where((SNRs[iidd] >= 0.75) &
                        (SNRs[iidd] < 0.9))[0]
        idx6 = np.where(SNRs[iidd] >= 0.9)[0]

        if min_dist_comp in idx1:
            cv2.drawContours(curFrame, contours[min_dist_comp], -1, (255, 0, 0), 1)
        if min_dist_comp in idx2:
            cv2.drawContours(curFrame, contours[min_dist_comp], -1, (0, 255, 0), 1)
        if min_dist_comp in idx3:
            cv2.drawContours(curFrame, contours[min_dist_comp], -1, (0, 0, 255), 1)
        if min_dist_comp in idx4:
            cv2.drawContours(curFrame, contours[min_dist_comp], -1, (255, 255, 0), 1)
        if min_dist_comp in idx5:
            cv2.drawContours(curFrame, contours[min_dist_comp], -1, (255, 0, 255), 1)
        if min_dist_comp in idx6:
            cv2.drawContours(curFrame, contours[min_dist_comp], -1, (0, 255, 255), 1)

    im.setImage(curFrame, autoLevels=False)


# %%

#  Define a top-level widget to hold everything
w = QtGui.QWidget()

#  Create some widgets to be placed inside
btn = QtGui.QPushButton('press me')
text = QtGui.QLineEdit('enter text')
win = pg.GraphicsLayoutWidget()
win.setMaximumWidth(300)
win.setMinimumWidth(200)
hist = pg.HistogramLUTItem()  # Contrast/color control
win.addItem(hist)
p1 = pg.PlotWidget()
p2 = pg.PlotWidget()
p3 = pg.PlotWidget()
t = ParameterTree()
t_action = ParameterTree()
action_layout = QtGui.QGridLayout()

#  Create a grid layout to manage the widgets size and position
layout = QtGui.QGridLayout()
w.setLayout(layout)

# A plot area (ViewBox + axes) for displaying the image
# p1 = win.addPlot(title="Image here")
# Item for displaying image data
img = pg.ImageItem()
p1.addItem(img)

img2 = pg.ImageItem()
p3.addItem(img2)

hist.setImageItem(img)

# Draggable line for setting isocurve level
thrshcomp_line = pg.InfiniteLine(angle=0, movable=True, pen='g')
hist.vb.addItem(thrshcomp_line)
hist.vb.setMouseEnabled(y=False) # makes user interaction a little easier
thrshcomp_line.setValue(100)
thrshcomp_line.setZValue(1000) # bring iso line above contrast controls


## Add widgets to the layout in their proper positions
layout.addWidget(win, 1, 0)   # text edit goes in middle-left
layout.addWidget(p3, 0, 2)   # text edit goes in middle-left

layout.addWidget(t, 0, 0)   # button goes in upper-left
layout.addWidget(t_action, 1, 2)  # list widget goes in bottom-left
layout.addWidget(p1, 0, 1)  # plot goes on right side, spanning 2 rows
layout.addWidget(p2, 1, 1)  # plot goes on right side, spanning 2 rows


#enable only horizontal zoom for the traces component
p2.setMouseEnabled(x=True, y=False)


draw_contours()

hist.setLevels(estimates.background_image.min(), estimates.background_image.max())


# Another plot area for displaying ROI data
#win.nextRow()
#p2 = win.addPlot(colspan=2)
p2.setMaximumHeight(250)
#win.resize(800, 800)
#win.show()


# set position and scale of image
img.scale(1, 1)
# img.translate(-50, 0)

# zoom to fit imageo
p1.autoRange()


mode = "reset"
p2.setTitle("mode: %s" % (mode))

thrshcomp_line.sigDragged.connect(lambda: draw_contours_overall(mode))


def imageHoverEvent(event):
    #  Show the position, pixel, and value under the mouse cursor.
    global x, y, i, j, val
    pos = event.pos()
    i, j = pos.y(), pos.x()
    i = int(np.clip(i, 0, estimates.background_image.shape[0] - 1))
    j = int(np.clip(j, 0, estimates.background_image.shape[1] - 1))
    val = estimates.background_image[i, j, 0]
    ppos = img.mapToParent(pos)
    x, y = ppos.x(), ppos.y()


# Monkey-patch the image to use our custom hover function.
# This is generally discouraged (you should subclass ImageItem instead),
# but it works for a very simple use like this.
img.hoverEvent = imageHoverEvent


def mouseClickEvent(event):
    global mode
    global x, y, i, j, val

    pos = img.mapFromScene(event.pos())
    x = int(pos.x())
    y = int(pos.y())

    if x < 0 or x > estimates.dims[0] or y < 0 or y > estimates.dims[1]:
        #  if the user click outside of the movie, jump out of the function
        return

    i, j = pos.y(), pos.x()
    i = int(np.clip(i, 0, estimates.background_image.shape[0] - 1))
    j = int(np.clip(j, 0, estimates.background_image.shape[1] - 1))
    val = estimates.background_image[i, j, 0]

    if mode == "neurons":
        show_neurons_clicked()

p1.mousePressEvent = mouseClickEvent


#  A general rule in Qt is that if you override one mouse event handler, you must override all of them.
def release(event):
    pass

p1.mouseReleaseEvent = release

def move(event):
    pass

p1.mouseMoveEvent = move 




## PARAMS
params = [{'name': 'min_cnn_thr', 'type': 'float', 'value': 0.99, 'limits': (0, 1),'step':0.01},
            {'name': 'cnn_lowest', 'type': 'float', 'value': 0.1, 'limits': (0, 1),'step':0.01},
            {'name': 'rval_thr', 'type': 'float', 'value': 0.85, 'limits': (-1, 1),'step':0.01},
            {'name': 'rval_lowest', 'type': 'float', 'value': -1, 'limits': (-1, 1),'step':0.01},
            {'name': 'min_SNR', 'type': 'float', 'value': 2, 'limits': (0, 20),'step':0.1},
            {'name': 'SNR_lowest', 'type': 'float', 'value': 0, 'limits': (0, 20),'step':0.1},
            {'name': 'RESET', 'type': 'action'},
            {'name': 'SHOW BACKGROUND', 'type': 'action'},
            {'name': 'SHOW NEURONS', 'type': 'action'}
            ]
    
## Create tree of Parameter objects
pars = Parameter.create(name='params', type='group', children=params) 


params_action = [{'name': 'Filter components', 'type': 'bool', 'value': True, 'tip': "Filter components"},          
                 {'name': 'View components', 'type': 'list', 'values': ['All','Accepted',
                                                       'Rejected', 'Unassigned'], 'value': 'All'},
                 {'name': 'ADD GROUP', 'type': 'action'},
                 {'name': 'REMOVE GROUP', 'type': 'action'},
                 {'name': 'ADD SINGLE', 'type': 'action'},
                 {'name': 'REMOVE SINGLE', 'type': 'action'},
                 {'name': 'SAVE OBJECT', 'type': 'action'}
                 ]


pars_action = Parameter.create(name='params_action', type='group', children=params_action) 

t_action.setParameters(pars_action, showTop=False)
t_action.setWindowTitle('Parameter Action')


def reset_button():
    global mode
    mode = "reset"
    p2.setTitle("mode: %s" % (mode))
    #  clear the upper right image
    zeros = np.zeros(estimates.dims, order='F')
    img2.setImage(make_color_img(zeros), autoLevels=False)
    draw_contours()


pars.param('RESET').sigActivated.connect(reset_button)


def show_background_button():
    global bg_vline, min_background, max_background, background_num
    global mode, background_first_frame_scaled
    #  clear thhe upper right image
    zeros = np.zeros(estimates.dims, order='F')
    img2.setImage(make_color_img(zeros), autoLevels=False)

    background_num = (background_num + 1) % estimates.f.shape[0]
    mode = "background"
    p2.setTitle("mode: %s %d" % (mode,background_num))

    # display the first frame of the background
    background_first_frame = estimates.b[:, background_num].reshape(estimates.dims, order='F')
    min_background_first_frame = np.min(background_first_frame)
    max_background_first_frame = np.max(background_first_frame)
    background_first_frame_scaled = make_color_img(background_first_frame,
                                                   min_max=(min_background_first_frame, max_background_first_frame))
    img.setImage(background_first_frame_scaled,autoLevels=False)

    # draw the trace and the infinite line
    trace_background = estimates.f[background_num]
    p2.plot(trace_background, clear=True)
    bg_vline = pg.InfiniteLine(angle=90, movable=True)
    p2.addItem(bg_vline, ignoreBounds=True)
    bg_vline.setValue(0)
    bg_vline.sigPositionChanged.connect(show_background_update)


def show_background_update():
    global bg_index, min_background, max_background, background_scaled
    bg_index = int(bg_vline.value())
    if bg_index > -1 and bg_index < estimates.f.shape[-1]:
        # upper left component scrolls through the frames of the background
        background = estimates.b[:,background_num].dot(estimates.f[background_num,bg_index]).reshape(estimates.dims, order='F')
        background_scaled=make_color_img(background, min_max=(min_background[background_num], max_background[background_num]))
        img.setImage(background_scaled, autoLevels=False)


pars.param('SHOW BACKGROUND').sigActivated.connect(show_background_button)


def show_neurons_button():
    global mode, neuron_selected
    mode = "neurons"
    neuron_selected = False
    p2.setTitle("mode: %s" % (mode))
    #  clear the upper right image
    zeros = np.zeros(estimates.dims, order='F')
    img2.setImage(make_color_img(zeros), autoLevels=False)


def show_neurons_clicked():
    global nr_vline, nr_index
    global x,y,i,j,val,min_dist_comp,contour_single, neuron_selected, comp2_scaled
    neuron_selected = True
    distances = np.sum(((x,y)-estimates.cms[estimates.idx_components])**2, axis=1)**0.5
    min_dist_comp = np.argmin(distances)
    contour_all =[cv2.threshold(img, np.int(thrshcomp_line.value()), 255, 0)[1] for img in estimates.img_components[estimates.idx_components]]
    contour_single = contour_all[min_dist_comp] 

    # draw the traces (lower left component)
    estimates.components_to_plot = estimates.idx_components[min_dist_comp]
    p2.plot(estimates.C[estimates.components_to_plot] + estimates.YrA[estimates.components_to_plot], clear=True)   

    # plot img (upper left component)
    img.setImage(estimates.background_image, autoLevels=False)
    draw_contours_update(estimates.background_image, img)
    # plot img2 (upper right component)
    comp2 = np.multiply(estimates.Cn, contour_single > 0)
    comp2_scaled = make_color_img(comp2, min_max=(np.min(comp2), np.max(comp2)))
    img2.setImage(comp2_scaled,autoLevels=False)
    draw_contours_update(comp2_scaled, img2)
    # set title for the upper two components
    p3.setTitle("pos: (%0.1f, %0.1f)  component: %d  value: %g dist:%f" % (x, y, estimates.components_to_plot,
                                                                            val, distances[min_dist_comp]))
    p1.setTitle("pos: (%0.1f, %0.1f)  component: %d  value: %g dist:%f" % (x, y, estimates.components_to_plot,
                                                                           val, distances[min_dist_comp]))
    # draw the infinite line
    nr_vline = pg.InfiniteLine(angle=90, movable=True)
    p2.addItem(nr_vline, ignoreBounds=True)
    nr_vline.setValue(0)
    nr_vline.sigPositionChanged.connect(show_neurons_update)
    nr_index = 0


def show_neurons_update():
    global nr_index, frame_denoise_scaled, estimates, raw_mov_scaled
    global min_mov, max_mov, min_mov_denoise, max_mov_denoise
    if neuron_selected is False:
        return
    nr_index = int(nr_vline.value())
    if nr_index > 0 and nr_index < mov[:,0,0].shape[0]:
        # upper left compoenent scrolls through the raw movie        
        raw_mov = rotate90(mov[nr_index,:,:], right=True, vector=False)
        raw_mov_scaled = make_color_img(raw_mov, min_max=(min_mov,max_mov))
        img.setImage(raw_mov_scaled, autoLevels=False)
        draw_contours_update(raw_mov_scaled, img)
        # upper right component scrolls through the denoised movie
        frame_denoise = estimates.A[:,estimates.idx_components].dot(estimates.C[estimates.idx_components,nr_index]).reshape(estimates.dims, order='F')
        frame_denoise_scaled = make_color_img(frame_denoise, min_max=(min_mov_denoise,max_mov_denoise))
        img2.setImage(frame_denoise_scaled,autoLevels=False)
        draw_contours_update(frame_denoise_scaled, img2)
        

pars.param('SHOW NEURONS').sigActivated.connect(show_neurons_button)


def add_group():
    estimates.accepted_list = np.union1d(estimates.accepted_list,estimates.idx_components)
    estimates.rejected_list = np.setdiff1d(estimates.rejected_list,estimates.idx_components)
    change(None, None)

pars_action.param('ADD GROUP').sigActivated.connect(add_group)

def remove_group():
    estimates.rejected_list = np.union1d(estimates.rejected_list,estimates.idx_components)
    estimates.accepted_list = np.setdiff1d(estimates.accepted_list,estimates.idx_components)
    change(None, None)

pars_action.param('REMOVE GROUP').sigActivated.connect(remove_group)


def add_single():
    estimates.accepted_list = np.union1d(estimates.accepted_list,estimates.components_to_plot)
    estimates.rejected_list = np.setdiff1d(estimates.rejected_list,estimates.components_to_plot)
    change(None, None)

pars_action.param('ADD SINGLE').sigActivated.connect(add_single)


def remove_single():
    estimates.rejected_list = np.union1d(estimates.rejected_list,estimates.components_to_plot)
    estimates.accepted_list = np.setdiff1d(estimates.accepted_list,estimates.components_to_plot)
    change(None, None)

pars_action.param('REMOVE SINGLE').sigActivated.connect(remove_single)


def save_object():
    global estimates, cnm_obj, rotation_flag
    print('Saving')
    ffll = F.getSaveFileName(filter='HDF5 (*.hdf5);;NWB (*.nwb)')
    print(ffll[0])
    # rotate back
    if estimates.rotation == True:
        estimates.Cn = rotate90(estimates.Cn, right=False, vector=False)  
        estimates.A = rotate90(estimates.A, right=False, vector=True, sparse=True)
        estimates.b = rotate90(estimates.b, right=False, vector=True)
        estimates.dims = (estimates.dims[1], estimates.dims[0])
        estimates.rotation = False
    cnm_obj.estimates = estimates
    if os.path.splitext(ffll[0])[1] == '.hdf5':
        cnm_obj.save(ffll[0])
    elif os.path.splitext(ffll[0])[1] == '.nwb':
        from pynwb import NWBHDF5IO
        with NWBHDF5IO(fpath, 'r') as io:
            nwb = io.read()
            raw_data_file = nwb.acquisition['TwoPhotonSeries'].external_file[0]
        cnm_obj.estimates.save_NWB(ffll[0], imaging_rate=cnm_obj.params.data['fr'], session_start_time=datetime.now(tzlocal()),
                            raw_data_file=raw_data_file)


pars_action.param('SAVE OBJECT').sigActivated.connect(save_object)


def action_pars_activated(param, changes):  
    change(None, None)
    
pars_action.sigTreeStateChanged.connect(action_pars_activated)


#  If anything changes in the tree, print a message
def change(param, changes):
    global estimates, pars, pars_action
    set_par = pars.getValues()
    if pars_action.param('Filter components').value():
        for keyy in set_par.keys():
            params_obj.quality.update({keyy: set_par[keyy][0]})
    else:
            params_obj.quality.update({'cnn_lowest': .1,
                                       'min_cnn_thr': 0.99,
                                       'rval_thr': 0.85,
                                       'rval_lowest': -1,
                                       'min_SNR': 2,
                                       'SNR_lowest': 0})
    estimates.filter_components(mov, params_obj, dview=None,
                                select_mode=pars_action.param('View components').value())
    if mode == "background":
        return
    else:
        draw_contours()                

pars.sigTreeStateChanged.connect(change)

change(None, None) # set params to default
t.setParameters(pars, showTop=False)
t.setWindowTitle('Parameter Quality')

#  END PARAMS

#  Display the widget as a new window
w.show()

#  Start the Qt event loop
app.exec_()

# Rotate back
if estimates.rotation == True:
    estimates.Cn = rotate90(estimates.Cn, right=False, vector=False)  
    estimates.A = rotate90(estimates.A, right=False, vector=True, sparse=True)
    estimates.b = rotate90(estimates.b, right=False, vector=True)
    estimates.dims = (estimates.dims[1], estimates.dims[0])
    cnm_obj.estimates = estimates
    estimates.rotation = False
