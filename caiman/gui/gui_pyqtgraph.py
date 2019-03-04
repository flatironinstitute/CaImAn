import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
from pyqtgraph.parametertree import Parameter, ParameterTree, ParameterItem, registerParameterType
import numpy as np
import caiman as cm
from caiman.source_extraction.cnmf.cnmf import load_CNMF
import cv2
import scipy
import os

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



# load estimates and movie image
# base_folder = '/mnt/ceph/neuro/DataForPublications/DATA_PAPER_ELIFE/WEBSITE/'
base_folder = '/Users/agiovannucci/SOFTWARE/CaImAn/'
mov = cm.load('/Users/agiovann/caiman_data/example_movies/memmap__d1_60_d2_80_d3_1_order_C_frames_2000_.mmap')
Cn = cm.load('/Users/agiovann/caiman_data/example_movies/demoMovie_CN.tif')
cnm_obj = load_CNMF('/Users/agiovann/caiman_data/example_movies/demoMovie.hdf5')
estimates = cnm_obj.estimates
estimates.restore_discarded_components()
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
    contours = [cv2.findContours(cv2.threshold(img, np.int(thrshcomp_line.value()), 255, 0)[1], cv2.RETR_TREE,
                                 cv2.CHAIN_APPROX_SIMPLE)[1] for img in estimates.img_components[estimates.idx_components]]
    bkgr_contours = estimates.background_image.copy()
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
# Interpret image data as row-major instead of col-major
# pg.setConfigOptions(imageAxisOrder='row-major')

pg.mkQApp()
win = pg.GraphicsLayoutWidget()
win.setWindowTitle('pyqtgraph example: Image Analysis')


#pars_pane = win.addItem(pars)
# A plot area (ViewBox + axes) for displaying the image
p1 = win.addPlot(title="Image here")


# Item for displaying image data
img = pg.ImageItem()
p1.addItem(img)

# Custom ROI for selecting an image region

roi = pg.ROI([-8, 14], [6, 5])
roi.addScaleHandle([0.5, 1], [0.5, 0.5])
roi.addScaleHandle([0, 0.5], [0.5, 0.5])
p1.addItem(roi)
roi.setZValue(10)  # make sure ROI is drawn above image


# Contrast/color control
hist = pg.HistogramLUTItem()
hist.setImageItem(img)
win.addItem(hist)

hist_cnn = pg.HistogramLUTItem()
img_cnn = pg.ImageItem()
img_cnn.setImage(estimates.cnn_preds[np.newaxis,:], autoLevels=False)
hist_cnn.setImageItem(img_cnn)
hist_cnn.setObjectName('metric_CNN')
win.addItem(hist_cnn)

hist_rval = pg.HistogramLUTItem()
img_rval = pg.ImageItem()
img_rval.setImage(estimates.cnn_preds[np.newaxis,:], autoLevels=False)
hist_rval.setImageItem(img_rval)
hist_rval.setObjectName('metric_rval')
win.addItem(hist_rval)


hist_SNR = pg.HistogramLUTItem()
img_SNR = pg.ImageItem()
img_SNR.setImage(estimates.SNR_comp[np.newaxis,:], autoLevels=False)
hist_SNR.setImageItem(img_SNR)
hist_SNR.setObjectName('metric_SNR')
win.addItem(hist_SNR)



def changed_quality_metrics_cnn(event):
    global estimates, cnm_obj
    params = cnm_obj.params
    cnn_lowest, min_cnn_thr = event.getLevels()
    params.quality.update({'cnn_lowest': cnn_lowest,'min_cnn_thr': min_cnn_thr})
    estimates.filter_components(mov, params, dview=None)
    draw_contours()

def changed_quality_metrics_SNR(event):
    global estimates, cnm_obj
    params = cnm_obj.params
    SNR_lowest, min_SNR = event.getLevels()
    params.quality.update({'SNR_lowest': SNR_lowest, 'min_SNR': min_SNR})
    estimates.filter_components(mov, params, dview=None)
    draw_contours()
    
def changed_quality_metrics_rval(event):
    global estimates, cnm_obj, win
    params = cnm_obj.params
    rval_lowest, rval_thr = event.getLevels()
    params.quality.update({'rval_lowest': rval_lowest,'rval_thr': rval_thr})
    estimates.filter_components(mov, params, dview=None)
    draw_contours()    
    

#def changed_quality_metrics(event):
#    global estimates, cnm_obj, event_
#    params = cnm_obj.params
#
#    if event.objectName() == 'metric_CNN':
#        print('Wring')
#    elif event.objectName() == 'metric_SNR':
#        
#    elif event.objectName() == 'metric_rval':
#        
#    else:
#        raise Exception('Unknown Metric Object')
#    estimates.filter_components(mov, params, dview=None)
#
#    # selected_components = np.where(estimates.cnn_preds >= low_cnn)[0]
#    # discarded_components = np.where(estimates.cnn_preds <= low_cnn)[0]
#    # estimates.idx_components = np.setdiff1d(selected_components, discarded_components)
#    draw_contours()


hist_cnn.sigLevelsChanged.connect(changed_quality_metrics_cnn)
hist_rval.sigLevelsChanged.connect(changed_quality_metrics_rval)
hist_SNR.sigLevelsChanged.connect(changed_quality_metrics_SNR)


# Draggable line for setting isocurve level
thrshcomp_line = pg.InfiniteLine(angle=0, movable=True, pen='g')
hist.vb.addItem(thrshcomp_line)
hist.vb.setMouseEnabled(y=False) # makes user interaction a little easier
thrshcomp_line.setValue(100)
thrshcomp_line.setZValue(1000) # bring iso line above contrast controls


# Another plot area for displaying ROI data
win.nextRow()
p2 = win.addPlot(colspan=2)
p2.setMaximumHeight(250)
win.resize(800, 800)
win.show()

draw_contours()
hist.setLevels(estimates.background_image.min(), estimates.background_image.max())


# set position and scale of image
img.scale(1, 1)
# img.translate(-50, 0)

# zoom to fit imageo
p1.autoRange()


# # Callbacks for handling user interaction
# def updatePlot(event):
#     global img, roi, background_image, p2
#     if event.isExit():
#         p1.setTitle("")
#         return
#     pos = event.pos()
#     i, j = pos.y(), pos.x()
#     i = int(np.clip(i, 0, background_image.shape[0] - 1))
#     j = int(np.clip(j, 0, background_image.shape[1] - 1))
#     val = background_image[i, j, 0]
#     ppos = img.mapToParent(pos)
#     x, y = ppos.x(), ppos.y()
#     p1.setTitle("pos: (%0.1f, %0.1f)  pixel: (%d, %d)  value: %g" % (x, y, i, j, val))
#
#
#     # selected = roi.getArrayRegion(data, img)
#     # p2.plot(selected.mean(axis=0), clear=True)
#
# roi.sigRegionChanged.connect(updatePlot)
# updatePlot()

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
    global x, y, i, j, val
    distances = np.sum(((x,y)-estimates.cms[estimates.idx_components])**2, axis=1)**0.5
    min_dist_comp = np.argmin(distances)
    estimates.components_to_plot = estimates.idx_components[min_dist_comp]
    p2.plot(estimates.C[estimates.components_to_plot] + estimates.R[estimates.components_to_plot], clear=True)
    p1.setTitle("pos: (%0.1f, %0.1f)  component: %d  value: %g dist:%f" % (x, y, estimates.components_to_plot,
                                                                           val, distances[min_dist_comp]))
p1.vb.mouseClickEvent = mouseClickEvent


## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()

