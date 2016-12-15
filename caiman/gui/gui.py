#%%
from builtins import zip
import numpy as np
import matplotlib.pyplot as pl
from matplotlib import animation
from matplotlib.widgets import Slider, Button, RadioButtons
import matplotlib.gridspec as gridspec
import time

def pretty_roi(mask):
    mask = mask.astype(np.float)
    colors = np.arange(1,len(mask)+1)
    mask *= colors[...,np.newaxis,np.newaxis] 
    mask = np.max(mask, axis=0)
    mask[mask==0] = None 
    return mask

def isin(r, ax):
    x0,x1,y0,y1 = ax.axis()
    xmin,xmax = min(x0,x1),max(x0,x1)
    ymin,ymax = min(y0,y1),max(y0,y1)
    return (ymin<r[0]<ymax) and (xmin<r[1]<xmax)

class GUI(animation.TimedAnimation):
    """
        interface for viewing a movie and its associated roi, traces, other things

        implementation is only through matplotlib, and is non-blocking

        backend affects performance somewhat dramatically. have achieved decent performance with qt5agg and tkagg
    """
    def __init__(self, mov, roi, traces, images={}, cmap=pl.cm.viridis, **kwargs):
        """
            Parameters:
                mov : 3d np array, 0'th axis is time/frames
                roi : 3d np array, one roi per item in 0'th axis, each of which is a True/False mask indicating roi (True=inside roi)
                traces : 2d np array, 0'th axis is time, 1st axis is sources
                images: dictionary of still images

            Attributes:
                roi_kept : boolean array of length of supplied roi, indicating whether or not roi should be kept based on user input

        """

        self.mov = mov
        self.roi_idxs = np.array([np.argwhere(r.flat).squeeze() for r in roi])
        self.roi_centers = np.array([np.mean(np.argwhere(r),axis=0) for r in roi])
        self.roi_orig = roi.copy()
        self.roi = pretty_roi(roi)
        self.roi_kept = np.ones(len(self.roi_idxs)).astype(bool)
        self.traces = traces
        self.images = images

        # figure setup
        self.cmap = cmap
        self.fig = pl.figure()
        NR,NC = 128,32
        gs = gridspec.GridSpec(nrows=NR, ncols=NC)
        gs.update(wspace=0.1, hspace=0.1, left=.04, right=.96, top=.98, bottom=.02)
        # movie axes
        self.ax_contrast0 = self.fig.add_subplot(gs[0:5,0:NC//3])
        self.ax_contrast1 = self.fig.add_subplot(gs[5:10,0:NC//3])
        self.ax_mov = self.fig.add_subplot(gs[10:55,0:NC//3])
        self.ax_mov.axis('off')
        self.ax_img = self.fig.add_subplot(gs[55:100,0:NC//3])
        self.ax_img.axis('off')
        self.axs_imbuts = [self.fig.add_subplot(gs[110:128,idx*2:idx*2+2]) for idx,i in enumerate(self.images)]
        # trace axes
        self.ax_trcs = self.fig.add_subplot(gs[0:64,NC//2:])
        self.ax_trc = self.fig.add_subplot(gs[65:85,NC//2:])
        self.ax_trc.set_xlim([0, len(self.traces)])
        self.ax_nav = self.fig.add_subplot(gs[85:90,NC//2:])
        self.ax_nav.set_xlim([0, len(self.traces)])
        self.ax_nav.axis('off')
        self.ax_rm = self.fig.add_subplot(gs[95:110,NC//2:])

        # interactivity
        self.c0,self.c1= 0,100
        self.sl_contrast0 = Slider(self.ax_contrast0, 'Low', 0., 255.0, valinit=self.c0, valfmt='%d')
        self.sl_contrast1 = Slider(self.ax_contrast1, 'Hi', 0., 255.0, valinit=self.c1, valfmt='%d')
        self.sl_contrast0.on_changed(self.evt_contrast)
        self.sl_contrast1.on_changed(self.evt_contrast)
        self.img_buttons = [Button(ax,k) for k,ax in zip(list(self.images.keys()),self.axs_imbuts)]
        self.but_rm = Button(self.ax_rm, 'Remove All ROIs Currently in FOV')

        # display initial things
        self.movdata = self.ax_mov.imshow(self.mov[0])
        self.movdata.set_animated(True)
        self.roidata = self.ax_mov.imshow(self.roi, cmap=self.cmap, alpha=0.5, vmin=np.nanmin(self.roi), vmax=np.nanmax(self.roi))
        self.trdata, = self.ax_trc.plot(np.zeros(len(self.traces)))
        self.navdata, = self.ax_nav.plot([-2,-2],[-1,np.max(self.traces)], 'r-')
        if len(self.images):
            lab,im = list(self.images.items())[0]
            self.imgdata = self.ax_img.imshow(im)
            self.ax_img.set_ylabel(lab)
        self.plot_current_traces()

        # callbacks
        for ib,lab in zip(self.img_buttons,list(self.images.keys())):
            ib.on_clicked(lambda evt, lab=lab: self.evt_imbut(evt,lab))
        self.but_rm.on_clicked(self.remove_roi)
        self.fig.canvas.mpl_connect('button_press_event', self.evt_click)
        self.ax_mov.callbacks.connect('xlim_changed', self.evt_zoom)
        self.ax_mov.callbacks.connect('ylim_changed', self.evt_zoom)

        # runtime
        self._idx = -1
        self.t0 = time.clock()
        self.always_draw = [self.movdata, self.roidata, self.navdata]
        self.blit_clear_axes = [self.ax_mov, self.ax_nav]

        # parent init
        animation.TimedAnimation.__init__(self, self.fig, interval=40, blit=True, **kwargs)

    @property
    def frame_seq(self):
        #print (time.clock()-self.t0)
        self._idx += 1
        if self._idx == len(self.mov):
            self._idx = 0
        self.navdata.set_xdata([self._idx, self._idx])
        yield self.mov[self._idx]

    @frame_seq.setter
    def frame_seq(self, val):
        pass

    def new_frame_seq(self):
        return self.mov

    def _init_draw(self):
        self._draw_frame(self.mov[0])
        self._drawn_artists = self.always_draw

    def _draw_frame(self, d):
        self.t0 = time.clock()

        self.movdata.set_data(d)

        # blit
        self._drawn_artists = self.always_draw
        for da in self._drawn_artists:
            da.set_animated(True)

    def _blit_clear(self, artists, bg_cache):
        for ax in self.blit_clear_axes:
            if ax in bg_cache:
                self.fig.canvas.restore_region(bg_cache[ax])

    def evt_contrast(self, val):
        self.c0 = self.sl_contrast0.val
        self.c1 = self.sl_contrast1.val

        if self.c0 > self.c1:
            self.c0 = self.c1-1
            self.sl_contrast0.set_val(self.c0)
        if self.c1 < self.c0:
            self.c1 = self.c0+1
            self.sl_contrast1.set_val(self.c1)

        self.movdata.set_clim(vmin=self.c0, vmax=self.c1)
        self.imgdata.set_clim(vmin=self.c0, vmax=self.c1)

    def evt_imbut(self, evt, lab):
        self.imgdata.set_data(self.images[lab])
        self.ax_img.set_title(lab)

    def evt_click(self, evt):
        if not evt.inaxes:
            return

        elif evt.inaxes == self.ax_mov:
            # select roi
            x,y = int(np.round(evt.xdata)), int(np.round(evt.ydata))
            idx = np.ravel_multi_index((y,x), self.roi.shape)
            inside = np.argwhere([idx in ri for ri in self.roi_idxs])
            if len(inside)==0:
                return
            i = inside[0]
            self.set_current_trace(i)

        elif evt.inaxes in [self.ax_nav]:
            x = int(np.round(evt.xdata))
            self._idx = x

    def evt_zoom(self, *args):
        self.plot_current_traces()

    def set_current_trace(self, idx):
        col = self.cmap(np.linspace(0,1,np.sum(self.roi_kept)))[np.squeeze(idx)]
        t = self.traces[:,idx]
        self.trdata.set_ydata(t)
        self.trdata.set_color(col)
        self.ax_trc.set_ylim([t.min(), t.max()])
        self.ax_trc.set_title('ROI {}'.format(idx))
        self.ax_trc.figure.canvas.draw()

    def get_current_roi(self):
        croi = np.array([isin(rc,self.ax_mov) for rc in self.roi_centers])
        croi[self.roi_kept==False] = False
        return croi

    def remove_roi(self, evt):
        self.current_roi = self.get_current_roi()
        self.roi_kept[self.current_roi] = False
        # update
        if np.sum(self.roi_kept):
            proi = pretty_roi(self.roi_orig[self.roi_kept])
            self.roidata.set_data(proi)
            self.roidata.set_clim(vmin=np.nanmin(proi), vmax=np.nanmax(proi))
        else:
            self.roidata.remove()

    def plot_current_traces(self):
        self.current_roi = self.get_current_roi()

        if np.sum(self.current_roi)==0:
            return
        for line in self.ax_trcs.get_lines():
            line.remove()

        cols = self.cmap(np.linspace(0,1,len(self.roi_idxs)))[self.current_roi]
        lastmax = 0
        for t,c in zip(self.traces.T[self.current_roi],cols):
            self.ax_trcs.plot((t-t.min())+lastmax, color=c)
            lastmax = np.max(t)
        self.ax_trcs.set_ylim([0,lastmax])
        #self.fig.canvas.draw_idle()
#%%
if __name__ == '__main__':
    # load data
    mov = load(fname,fr=30)
    mean,minn,maxx = mov.mean(axis=0),mov.min(axis=0),mov.max(axis=0)
    roi = np.load('/Users/ben/Desktop/roi.npy')
    tr = np.random.random([3000,3])#np.load('/Users/ben/Desktop/tr.npy') roi

    # run interface
    intfc = GUI(mov, roi, tr, images=dict(mean=mean,min=minn,max=maxx))
