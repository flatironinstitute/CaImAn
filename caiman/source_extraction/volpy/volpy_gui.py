#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
VolPy GUI interface is used to correct outputs of Mask R-CNN or annotate new datasets.
VolPy GUI uses summary images and ROIs as the input. It outputs binary masks for the trace denoising 
and spike extraction step of VolPy.
@author: @caichangjia
"""
import cv2
import h5py
import numpy as np
import os
from pathlib import Path
import pyqtgraph as pg
from pyqtgraph import FileDialog
from pyqtgraph.Qt import QtGui
from pyqtgraph.parametertree import Parameter, ParameterTree
import PyQt5
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QShortcut
import random
from skimage.draw import polygon
import sys

import caiman as cm
from caiman.external.cell_magic_wand import cell_magic_wand_single_point
    
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = os.fspath(
    Path(PyQt5.__file__).resolve().parent / "Qt5" / "plugins")


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
        if pars_action.param('DISPLAY').value() == 'SPATIAL FOOTPRINTS':
            overlay(all_ROIs)
        pts = []
        pos = img.mapFromScene(event.pos())
        p1.clear()
        p1.addItem(img)
        try:
            x = int(pos.x())
            y = int(pos.y())
            j, i = pos.y(), pos.x()
            i = int(np.clip(i, 0, dims[0] - 1))
            j = int(np.clip(j, 0, dims[1] - 1))
            p1.plot(x=[i], y=[j], symbol='o', pen=None, symbolBrush='y', symbolSize=5)#, symbolBrush=pg.intColor(i,6,maxValue=128))
            min_radius = pars_action.param('MAGIC WAND PARAMS').child('MIN RADIUS').value()
            max_radius = pars_action.param('MAGIC WAND PARAMS').child('MAX RADIUS').value()
            roughness = pars_action.param('MAGIC WAND PARAMS').child('ROUGHNESS').value()
            roi, edge = cell_magic_wand_single_point(adjust_contrast(cur_img.copy().T, hist.getLevels()[0], hist.getLevels()[1]), (j, i), 
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
        except:
            pass
    
    elif mode == 'CHOOSE NEURONS':
        pos = img.mapFromScene(event.pos())
        p1.clear()
        p1.addItem(img)
        x = int(pos.x())
        y = int(pos.y())
        j, i = pos.y(), pos.x()
        i = int(np.clip(i, 0, dims[0] - 1))
        j = int(np.clip(j, 0, dims[1] - 1))
        p1.plot(x=[i], y=[j], symbol='o', pen=None, symbolBrush='y', symbolSize=5)#, symbolBrush=pg.intColor(i,6,maxValue=128))
                
        try:
            loc = np.array(list(all_centers.values()))        
            dis = np.square(loc - np.array([i, j])).sum(1)
            min_idx = np.where(dis == dis.min())[0][0]
            #list(all_centers.keys())[min_idx]
            neuron_list.setCurrentRow(min_idx)
            show_neuron()       
        except:
            pass
        
def release(event):
    pass

def move(event):
    pass

def add():
    global mode, pts, all_pts,all_centers, img, p1, neuron_list, neuron_idx, roi
    if mode == "POLYGON":
        roi = np.zeros((dims[1], dims[0]))
        ff = np.array(polygon(np.array(pts)[:,0], np.array(pts)[:,1]))
        roi[ff[1],[ff[0]]] = 1
    
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
        all_centers[neuron_idx] = np.array(pts).mean(0)
        roi = roi.astype(np.float32)
        all_ROIs[neuron_idx] = roi
       
        pp = pts.copy()
        pp.append(pp[0])
        p1.plot(x=np.array(pp)[:,0], y=np.array(pp)[:,1], pen=pen)
        pts = []
        
    show_all()

def remove():
    try:
        item = neuron_list.currentItem()
        del all_pts[str(item.text())]
        del all_centers[str(item.text())]
        del all_ROIs[str(item.text())]
        neuron_list.takeItem(neuron_list.row(item))
        show_all()
    except:
        pass

def show_all():
    global all_pts, pen, all_ROIs, neuron_list, img_overlay
    p1.clear()
    p1.addItem(img)
    if pars_action.param('DISPLAY').value() == 'CONTOUR':
        for pp in list(all_pts.values()):
            pp.append(pp[0])
            p1.plot(x=np.array(pp)[:,0], y=np.array(pp)[:,1], pen=pen)
    else:
        overlay(all_ROIs)

def clear():
    p1.clear()
    p1.addItem(img)

def show_neuron():
    global all_pts
    item = neuron_list.currentItem()
    pp = all_pts[str(item.text())]
    pp.append(pp[0])
    p1.clear()
    p1.addItem(img)
    if pars_action.param('DISPLAY').value() == 'SPATIAL FOOTPRINTS':
        overlay({str(item.text): all_ROIs[str(item.text())]})
    else:
        p1.plot(x=np.array(pp)[:,0], y=np.array(pp)[:,1], pen=pen)

def load():
    global summary_images, dims, cur_img, p1
    fpath = F.getOpenFileName(caption='Load Summary Images',
                          filter='TIFF (*.tif);;HDF5 (*.h5 *.hdf5)')[0]
    summary_images = cm.load(fpath)
    summary_images = summary_images.transpose([0, 2, 1])
    summary_images = np.flip(summary_images, axis=2)
    cur_img = summary_images[0]
    dims = summary_images[0].shape
    #p1.resize(dims[0], dims[1])
    img.setImage(cur_img)
    p1.setAspectLocked()   

def load_rois():
    global summary_images, neuron_list, all_pts, all_centers, all_ROIs, dims
    fpath = F.getOpenFileName(caption='Load ROIS',
                          filter='HDF5 (*.h5 *.hdf5);;TIFF (*.tif)')[0]
    with h5py.File(fpath, "r") as f:
        l_ROIs = np.array(list(f[list(f.keys())[0]]))
    l_ROIs = np.flip(l_ROIs, axis=1)
    if (l_ROIs.shape[2], l_ROIs.shape[1]) != dims:
        print(dims);print(l_ROIs.shape[1:])
        raise ValueError('Dimentions of movie and rois do not accord')
    
    for roi in l_ROIs:
        flag = True
        while flag:
            r1 = random.randrange(1, 10**4)
            r2 = random.randrange(1, 10**4)
            neuron_idx = str(r1) + '-' +str(r2)
            if neuron_idx not in all_pts.keys():
                flag = False                
        neuron_list.addItem(neuron_idx)
        ROI8 = np.uint8(roi * 255)
        contours = cv2.findContours(cv2.threshold(ROI8, 100, 255, 0)[1], cv2.RETR_TREE,
                                         cv2.CHAIN_APPROX_NONE)[0][0][:,0,:]
        pts = contours.tolist()
        all_pts[neuron_idx] = pts
        all_centers[neuron_idx] = np.array(pts).mean(0)
        all_ROIs[neuron_idx] = roi
    show_all()    

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

def change(param, changes):
    global mode, cur_img
    print("tree changes:")
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

def down():
    global all_pts
    try:
        neuron_list.setCurrentRow(neuron_list.currentRow() + 1)
        item = neuron_list.currentItem()
        pp = all_pts[str(item.text())]
        pp.append(pp[0])
        p1.clear()
        p1.addItem(img)
        if pars_action.param('DISPLAY').value() == 'SPATIAL FOOTPRINTS':
            overlay({str(item.text): all_ROIs[str(item.text())]})
        else:
            p1.plot(x=np.array(pp)[:,0], y=np.array(pp)[:,1], pen=pen)
    except:
        pass

def up():
    global all_pts
    try:
        neuron_list.setCurrentRow(neuron_list.currentRow() - 1)
        item = neuron_list.currentItem()
        pp = all_pts[str(item.text())]
        pp.append(pp[0])
        p1.clear()
        p1.addItem(img)
        if pars_action.param('DISPLAY').value() == 'SPATIAL FOOTPRINTS':
            overlay({str(item.text): all_ROIs[str(item.text())]})
        else:
            p1.plot(x=np.array(pp)[:,0], y=np.array(pp)[:,1], pen=pen)
    except:
        pass

def adjust_contrast(img, min_value, max_value):
    img[img < min_value] = min_value
    img[img > max_value] = max_value    
    return img

def overlay(all_ROIs):
    if len(list(all_ROIs.values())) > 0:
        img_rois = np.array(list(all_ROIs.values()))
        img_rois = img_rois.transpose([0, 2, 1])
        img_rois = img_rois.sum(0)
        img_rois[img_rois>1] = 1
        b,g,r = cv2.split(cv2.cvtColor(img_rois,cv2.COLOR_GRAY2BGR))
        g[g>0] = 0
        r[r>0] = 0
        img_rois = cv2.merge([b,g,r])
        img_overlay = pg.ImageItem()
        img_overlay.setImage(img_rois)
        p1.addItem(img_overlay)
        img_overlay.setZValue(10) # make sure this image is on top
        img_overlay.setOpacity(0.2)   

if __name__ == "__main__":    
    ## Always start by initializing Qt (only once per application)
    app = QApplication(sys.argv)
    
    ## Define a top-level widget to hold everything
    w = QtWidgets.QWidget()
    
    ## Create some widgets to be placed inside
    hist = pg.HistogramLUTItem()  # Contrast/color control
    win = pg.GraphicsLayoutWidget()
    win.setMaximumWidth(300)
    win.setMinimumWidth(200)
    win.addItem(hist)
    p1 = pg.PlotWidget()
    neuron_action = ParameterTree()
    neuron_list = QtWidgets.QListWidget()
    
    ## Create a grid layout to manage the widgets size and position
    layout = QtWidgets.QGridLayout()
    w.setLayout(layout)
    
    ## Add widgets to the layout in their proper positions
    layout.addWidget(win, 0, 1)  
    layout.addWidget(p1, 0, 2)  
    layout.addWidget(neuron_action, 0, 3)   
    layout.addWidget(neuron_list, 0, 4)  
    img = pg.ImageItem()
    p1.addItem(img)
    hist.setImageItem(img)
    
    # Add actions    
    params_action = [{'name': 'LOAD DATA', 'type':'action'},
                     {'name': 'LOAD ROIS', 'type':'action'},
                     {'name': 'SAVE', 'type':'action'},                 
                     {'name': 'ADD', 'type': 'action'}, 
                     {'name': 'REMOVE', 'type': 'action'}, 
                     {'name': 'SHOW ALL', 'type': 'action'},
                     {'name': 'CLEAR', 'type': 'action'}, 
                     {'name': 'IMAGES', 'type': 'list', 'values': ['MEAN','CORR']},
                     {'name': 'DISPLAY', 'type': 'list', 'values': ['CONTOUR','SPATIAL FOOTPRINTS']},
                     {'name': 'MODE', 'type': 'list', 'values': ['POLYGON','CELL MAGIC WAND', 'CHOOSE NEURONS']},
                     {'name': 'MAGIC WAND PARAMS', 'type': 'group', 'children': [{'name': 'MIN RADIUS', 'type': 'int', 'value': 4},
                                                                        {'name': 'MAX RADIUS', 'type': 'int', 'value': 10},
                                                                        {'name': 'ROUGHNESS', 'type': 'int', 'value': 1}]}]
    
    pars_action = Parameter.create(name='params_action', type='group', children=params_action) 
    neuron_action.setParameters(pars_action, showTop=False)
    neuron_action.setWindowTitle('Parameter Action')
    mode = pars_action.getValues()['MODE'][0]
    
    # Add event
    p1.mousePressEvent = mouseClickEvent
    p1.mouseReleaseEvent = release
    p1.mouseMoveEvent = move 
    pars_action.param('ADD').sigActivated.connect(add)
    pars_action.param('REMOVE').sigActivated.connect(remove)
    pars_action.param('SHOW ALL').sigActivated.connect(show_all)
    pars_action.param('CLEAR').sigActivated.connect(clear)
    pars_action.param('LOAD DATA').sigActivated.connect(load)
    pars_action.param('LOAD ROIS').sigActivated.connect(load_rois)
    pars_action.param('SAVE').sigActivated.connect(save)
    pars_action.sigTreeStateChanged.connect(change)    
    shortcut_down = QShortcut(QtGui.QKeySequence("down"), w)
    shortcut_down.activated.connect(down)
    shortcut_up = QShortcut(QtGui.QKeySequence("up"), w)
    shortcut_up.activated.connect(up)
    neuron_list.itemClicked.connect(show_neuron)
    
    # Create dictionary for saving 
    F = FileDialog()
    all_pts = {}
    all_centers = {}
    all_ROIs = {}
    pts = []
    pen = pg.mkPen(color=(255, 255, 0), width=4)#, style=Qt.DashDotLine)
    
    ## Display the widget as a new window
    w.show()
    
    ## Start the Qt event loop
    app.exec_()
        
        
        
