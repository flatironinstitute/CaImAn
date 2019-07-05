import pyqtgraph as pg
from pyqtgraph import FileDialog
from pyqtgraph.Qt import QtCore, QtGui
from pyqtgraph.parametertree import Parameter, ParameterTree, ParameterItem, registerParameterType
import numpy as np
import caiman as cm
from caiman.source_extraction.cnmf.cnmf import load_CNMF
from caiman.source_extraction.cnmf.online_cnmf import load_OnlineCNMF

import cv2
import scipy
import os
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

def make_color_img(img, gain=255, out_type=np.uint8):
    min = img.min()
    max = img.max()
    img = (img-min)/(max-min)*gain
    img = img.astype(out_type)
    img = np.dstack([img]*3)
    return img

F = FileDialog()

# load object saved by CNMF
# cnm_obj = load_CNMF('/Users/agiovann/caiman_data/example_movies/memmap__d1_60_d2_80_d3_1_order_C_frames_2000_save.hdf5')
cnm_obj = load_CNMF(F.getOpenFileName(caption='Load CNMF Object',filter='*.hdf5')[0])


# movie
# mov = cm.load('/Users/agiovann/caiman_data/example_movies/memmap__d1_60_d2_80_d3_1_order_C_frames_2000_.mmap')
mov = cm.load(cnm_obj.mmap_file)

# load summary image
# Cn = cm.load('/Users/agiovann/caiman_data/example_movies/memmap__d1_60_d2_80_d3_1_order_C_frames_2000__Cn.tif')
Cn = cnm_obj.estimates.Cn


estimates = cnm_obj.estimates
if not hasattr(estimates, 'accepted_list'):
    # if estimates.discarded_components.A.shape[-1] > 0:
    #     estimates.restore_discarded_components()
    estimates.accepted_list = np.array([], dtype=np.int)
    estimates.rejected_list = np.array([], dtype=np.int)
    estimates.img_components = estimates.A.toarray().reshape((estimates.dims[0], estimates.dims[1],-1), order='F').transpose([2,0,1])
    estimates.cms = np.array([scipy.ndimage.measurements.center_of_mass(comp) for comp in estimates.img_components])
    estimates.idx_components = np.arange(estimates.nr)
    estimates.idx_components_bad = np.array([])
    estimates.background_image = make_color_img(Cn)
    # Generate image data
    estimates.img_components /= estimates.img_components.max(axis=(1,2))[:,None,None]
    estimates.img_components *= 255
    estimates.img_components = estimates.img_components.astype(np.uint8)
 



#%%
def draw_contours():
    global thrshcomp_line, estimates, cnm_obj, img
    bkgr_contours = estimates.background_image.copy()
    
    if len(estimates.idx_components) > 0:
        contours = [cv2.findContours(cv2.threshold(img, np.int(thrshcomp_line.value()), 255, 0)[1], cv2.RETR_TREE,
                                     cv2.CHAIN_APPROX_SIMPLE)[0] for img in estimates.img_components[estimates.idx_components]]
        
        SNRs = np.array(estimates.r_values)
        iidd = np.array(estimates.idx_components)
        
        idx1 = np.where(SNRs[iidd]<0.1)[0]
        idx2 = np.where((SNRs[iidd]>=0.1) & 
                        (SNRs[iidd]<0.25))[0]
        idx3 = np.where((SNRs[iidd]>=0.25) & 
                        (SNRs[iidd]<0.5))[0]
        idx4 = np.where((SNRs[iidd]>=0.5) & 
                        (SNRs[iidd]<0.75))[0]
        idx5 = np.where((SNRs[iidd]>=0.75) & 
                        (SNRs[iidd]<0.9))[0]
        idx6 = np.where(SNRs[iidd]>=0.9)[0]
        
        
    
        cv2.drawContours(bkgr_contours, sum([contours[jj] for jj in idx1], []), -1, (255, 0, 0), 1)
        cv2.drawContours(bkgr_contours, sum([contours[jj] for jj in idx2], []), -1, (0, 255, 0), 1)
        cv2.drawContours(bkgr_contours, sum([contours[jj] for jj in idx3], []), -1, (0, 0, 255), 1)
        cv2.drawContours(bkgr_contours, sum([contours[jj] for jj in idx4], []), -1, (255, 255, 0), 1)
        cv2.drawContours(bkgr_contours, sum([contours[jj] for jj in idx5], []), -1, (255, 0, 255), 1)
        cv2.drawContours(bkgr_contours, sum([contours[jj] for jj in idx6], []), -1, (0, 255, 255), 1)
    
    img.setImage(bkgr_contours, autoLevels=False)
# pg.setConfigOptions(imageAxisOrder='row-major')

#%%


## Define a top-level widget to hold everything
w = QtGui.QWidget()

## Create some widgets to be placed inside
btn = QtGui.QPushButton('press me')
text = QtGui.QLineEdit('enter text')
win = pg.GraphicsLayoutWidget()
win.setMaximumWidth(300)
win.setMinimumWidth(200)
hist = pg.HistogramLUTItem() # Contrast/color control
win.addItem(hist)
p1 =  pg.PlotWidget()
p2 =  pg.PlotWidget()
p3 =  pg.PlotWidget()
t = ParameterTree()
t_action = ParameterTree()
action_layout = QtGui.QGridLayout()


## Create a grid layout to manage the widgets size and position
layout = QtGui.QGridLayout()
w.setLayout(layout)

# A plot area (ViewBox + axes) for displaying the image
#p1 = win.addPlot(title="Image here")
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

thrshcomp_line.sigDragged.connect(draw_contours)

def imageHoverEvent(event):
    """Show the position, pixel, and value under the mouse cursor.
    """
    global x,y,i,j,val
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
    global x, y, i, j, val, img2
    distances = np.sum(((x,y)-estimates.cms[estimates.idx_components])**2, axis=1)**0.5
    min_dist_comp = np.argmin(distances)
    estimates.components_to_plot = estimates.idx_components[min_dist_comp]
    p2.plot(estimates.C[estimates.components_to_plot] + estimates.YrA[estimates.components_to_plot], clear=True)
    img2.setImage(estimates.A[:,estimates.components_to_plot].toarray().reshape(estimates.dims, order='F'), autoLevels=True)
    p3.setTitle("pos: (%0.1f, %0.1f)  component: %d  value: %g dist:%f" % (x, y, estimates.components_to_plot,
                                                                            val, distances[min_dist_comp]))
    p1.setTitle("pos: (%0.1f, %0.1f)  component: %d  value: %g dist:%f" % (x, y, estimates.components_to_plot,
                                                                           val, distances[min_dist_comp]))
p1.mousePressEvent = mouseClickEvent

## PARAMS
params = [{'name': 'min_cnn_thr', 'type': 'float', 'value': 0.99, 'limits': (0, 1),'step':0.01},
            {'name': 'cnn_lowest', 'type': 'float', 'value': 0.1, 'limits': (0, 1),'step':0.01},
            {'name': 'rval_thr', 'type': 'float', 'value': 0.85, 'limits': (-1, 1),'step':0.01},
            {'name': 'rval_lowest', 'type': 'float', 'value': -1, 'limits': (-1, 1),'step':0.01},
            {'name': 'min_SNR', 'type': 'float', 'value': 2, 'limits': (0, 20),'step':0.1},
            {'name': 'SNR_lowest', 'type': 'float', 'value': 0, 'limits': (0, 20),'step':0.1}]
    
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
    print('Saving')
    
    ffll = F.getSaveFileName(filter='*.hdf5')
    print(ffll[0])
    cnm_obj.estimates = estimates
    cnm_obj.save(ffll[0])

pars_action.param('SAVE OBJECT').sigActivated.connect(save_object)

    
def action_pars_activated(param, changes):  
    change(None, None)
    
pars_action.sigTreeStateChanged.connect(action_pars_activated)


## If anything changes in the tree, print a message
def change(param, changes):
    global estimates, cnm_obj, pars, pars_action
    params_obj = cnm_obj.params
    set_par = pars.getValues()
    if pars_action.param('Filter components').value():
        for keyy in set_par.keys():
            params_obj.quality.update({keyy: set_par[keyy][0]})
    else:
            params_obj.quality.update({'cnn_lowest': 0,'min_cnn_thr':0,'rval_thr':-1,'rval_lowest':-1,'min_SNR':0,'SNR_lowest':0})
    
    estimates.filter_components(mov, params_obj, dview=None, 
                                    select_mode=pars_action.param('View components').value())
    
    draw_contours()                
    
pars.sigTreeStateChanged.connect(change)

change(None, None) # set params to default
t.setParameters(pars, showTop=False)
t.setWindowTitle('Parameter Quality')

## END PARAMS    

        
draw_contours()   

## Display the widget as a new window
w.show()

## Start the Qt event loop
app.exec_()

