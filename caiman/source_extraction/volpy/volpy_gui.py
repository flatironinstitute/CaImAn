#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 10:48:16 2020
VolPy GUI interface for neuron selection
@author: @caichangjia
"""
import cv2
import numpy as np
import os
import pyqtgraph as pg
from pyqtgraph import FileDialog
from pyqtgraph.Qt import QtGui
from pyqtgraph.parametertree import Parameter, ParameterTree
import random
from skimage.draw import polygon

import caiman as cm
from caiman.external.cell_magic_wand import cell_magic_wand_single_point

try:
    cv2.setNumThreads(1)
except:
    print('Open CV is naturally single threaded')

try:
    if __IPYTHON__:
        print(1)
        # This is used for debugging purposes only. Automatically reloads imports if they change
        get_ipython().magic('load_ext autoreload')
        get_ipython().magic('autoreload 2')
except NameError:
    print('Not launched under iPython')
    

#%%
## Always start by initializing Qt (only once per application)
app = QtGui.QApplication([])

## Define a top-level widget to hold everything
w = QtGui.QWidget()

## Create some widgets to be placed inside
hist = pg.HistogramLUTItem()  # Contrast/color control
win = pg.GraphicsLayoutWidget()
win.setMaximumWidth(300)
win.setMinimumWidth(200)
win.addItem(hist)
p1 = pg.PlotWidget()
neuron_action = ParameterTree()
neuron_list = QtGui.QListWidget()

## Create a grid layout to manage the widgets size and position
layout = QtGui.QGridLayout()
w.setLayout(layout)

## Add widgets to the layout in their proper positions
layout.addWidget(win, 0, 1)  
layout.addWidget(p1, 0, 2)  
layout.addWidget(neuron_action, 0, 3)   
layout.addWidget(neuron_list, 0, 4)  
img = pg.ImageItem()
p1.addItem(img)
hist.setImageItem(img)

params_action = [{'name': 'LOAD', 'type':'action'},                 
                 {'name': 'SAVE', 'type':'action'},                 
                 {'name': 'ADD', 'type': 'action'}, 
                 {'name': 'REMOVE', 'type': 'action'}, 
                 {'name': 'SHOW ALL', 'type': 'action'},
                 {'name': 'CLEAR', 'type': 'action'}, 
                 {'name': 'IMAGES', 'type': 'list', 'values': ['MEAN','CORR']},
                 {'name': 'MODE', 'type': 'list', 'values': ['POLYGON','CELL MAGIC WAND']},
                 {'name': 'MAGIC WAND PARAMS', 'type': 'group', 'children': [{'name': 'MIN RADIUS', 'type': 'int', 'value': 4},
                                                                    {'name': 'MAX RADIUS', 'type': 'int', 'value': 10},
                                                                    {'name': 'ROUGHNESS', 'type': 'int', 'value': 1}]}]

pars_action = Parameter.create(name='params_action', type='group', children=params_action) 

neuron_action.setParameters(pars_action, showTop=False)
neuron_action.setWindowTitle('Parameter Action')
mode = pars_action.getValues()['MODE'][0]
    
def mouseClickEvent(event):
    global mode, x, y, i, j, pts, roi, cur_image
    if mode == "POLYGON":
        pos = img.mapFromScene(event.pos())
        x = int(pos.x())
        y = int(pos.y())
        j, i = pos.y(), pos.x()
        i = int(np.clip(i, 0, dims[0] - 1))
        j = int(np.clip(j, 0, dims[1] - 1))
        p1.plot(x=[i], y=[j], symbol='o', pen=None, symbolBrush='y', symbolSize=5)#, symbolBrush=pg.intColor(i,6,maxValue=128))
        pts.append([i, j])
    elif mode == "CELL MAGIC WAND":
        p1.clear()
        p1.addItem(img)
        pts = []
        pos = img.mapFromScene(event.pos())
        try:
            x = int(pos.x())
            y = int(pos.y())
            j, i = pos.y(), pos.x()
            p1.plot(x=[i], y=[j], symbol='o', pen=None, symbolBrush='y', symbolSize=5)#, symbolBrush=pg.intColor(i,6,maxValue=128))
            min_radius = pars_action.getValues()['MAGIC WAND PARAMS'][1]['MIN RADIUS'][0]
            max_radius = pars_action.getValues()['MAGIC WAND PARAMS'][1]['MAX RADIUS'][0]
            roughness = pars_action.getValues()['MAGIC WAND PARAMS'][1]['ROUGHNESS'][0]
            roi, edge = cell_magic_wand_single_point(adjust_contrast(cur_img.copy(), hist.getLevels()[0], hist.getLevels()[1]), (j, i), 
                                               min_radius=min_radius, max_radius=max_radius, 
                                               roughness=roughness, zoom_factor=1)
            ROI8 = np.uint8(roi * 255)
            contours = cv2.findContours(cv2.threshold(ROI8, 100, 255, 0)[1], cv2.RETR_TREE,
                                             cv2.CHAIN_APPROX_NONE)[0][0][:,0,:]
            pts = contours.tolist()
            pp = pts.copy()
            pp.append(pp[0])
            p1.plot(x=np.array(pp)[:,0], y=np.array(pp)[:,1], pen=pen)
            roi = roi * 1.
            #plt.imshow(roi)
            #plt.scatter(contours[:,0], contours[:,1])
        except:
            pass
        
p1.mousePressEvent = mouseClickEvent

#  A general rule in Qt is that if you override one mouse event handler, you must override all of them.
def release(event):
    pass
p1.mouseReleaseEvent = release

def move(event):
    pass
p1.mouseMoveEvent = move 

def add():
    global mode, pts, all_pts, img, p1, neuron_list, neuron_idx, roi
    if mode == "POLYGON":
        roi = np.zeros(dims)
        ff = np.array(polygon(np.array(pts)[:,0], np.array(pts)[:,1]))
        roi[ff[0],[ff[1]]] = 1
    
    if len(pts) > 2 :
        flag = True
        while flag:
            r1 = random.randrange(1, 10**4)
            r2 = random.randrange(1, 10**4)
            neuron_idx = str(r1) + '-' +str(r2)
            if neuron_idx not in all_pts.keys():
                flag = False                
        neuron_list.addItem(neuron_idx)
        all_pts[neuron_idx] = pts
        all_ROIs[neuron_idx] = roi
       
        pp = pts.copy()
        pp.append(pp[0])
        p1.plot(x=np.array(pp)[:,0], y=np.array(pp)[:,1], pen=pen)
        #p1.clear()
        #p1.addItem(img)
        pts = []
        
pars_action.param('ADD').sigActivated.connect(add)

def remove():
    try:
        item = neuron_list.currentItem()
        del all_pts[str(item.text())]
        del all_ROIs[str(item.text())]
        neuron_list.takeItem(neuron_list.row(item))
        show_all()
    except:
        pass

pars_action.param('REMOVE').sigActivated.connect(remove)

def show_all():
    global all_pts, pen
    p1.clear()
    p1.addItem(img)
    for pp in list(all_pts.values()):
        pp.append(pp[0])
        p1.plot(x=np.array(pp)[:,0], y=np.array(pp)[:,1], pen=pen)
    print(pars_action.getValues())

pars_action.param('SHOW ALL').sigActivated.connect(show_all)

def clear():
    p1.clear()
    p1.addItem(img)

pars_action.param('CLEAR').sigActivated.connect(clear)

def show_neuron():
    global all_pts
    item = neuron_list.currentItem()
    pp = all_pts[str(item.text())]
    pp.append(pp[0])
    p1.clear()
    p1.addItem(img)
    p1.plot(x=np.array(pp)[:,0], y=np.array(pp)[:,1], pen=pen)

neuron_list.itemClicked.connect(show_neuron)

def load():
    global summary_images, dims, cur_img
    fpath = F.getOpenFileName(caption='Load Summary Images',
                          filter='HDF5 (*.h5 *.hdf5)')[0]
    summary_images = cm.load(fpath)
    summary_images = summary_images.transpose([0, 2, 1])
    summary_images = np.flip(summary_images, axis=2)
    cur_img = summary_images[0]
    img.setImage(cur_img)
    dims = summary_images[0].shape
    
pars_action.param('LOAD').sigActivated.connect(load)

def save():
    global all_ROIs, save_ROIs, summary_images
    print('Saving')
    ffll = F.getSaveFileName(filter='HDF5 (*.hdf5)')
    print(ffll[0])
    save_ROIs = np.array(list(all_ROIs.values())).copy()
    save_ROIs = np.flip(save_ROIs, axis=1)
    
    if os.path.splitext(ffll[0])[1] == '.hdf5':
        cm.movie(save_ROIs).save(ffll[0])
    summary_images = summary_images.transpose([0, 2, 1])
    summary_images = np.flip(summary_images, axis=1)

pars_action.param('SAVE').sigActivated.connect(save)

def change(param, changes):
    global mode, cur_img
    print("tree changes:")
    #mode = pars_action.getValues()['MODE'][0]
    for param, change, data in changes:
        if pars_action.childPath(param)[0] == 'MODE':
            if data == 'None':
                mode = 'POLYGON'
            else:
                mode = data
        elif pars_action.childPath(param)[0] == 'IMAGES':
            if data == 'CORR':
                cur_img = summary_images[-1]
            else:
                cur_img = summary_images[0]
            img.setImage(cur_img)
        print(param)
        print(change)
        print(data)

pars_action.sigTreeStateChanged.connect(change)

def adjust_contrast(img, min_value, max_value):
    img[img < min_value] = min_value
    img[img > max_value] = max_value    
    return img

F = FileDialog()
all_pts = {}
all_ROIs = {}
pts = []
pen = pg.mkPen(color=(255, 255, 0), width=4)#, style=Qt.DashDotLine)

## Display the widget as a new window
w.show()

## Start the Qt event loop
app.exec_()
    
    
    
