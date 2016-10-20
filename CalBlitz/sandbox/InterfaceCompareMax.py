import numpy as np
import wx

import matplotlib
matplotlib.interactive(False)
matplotlib.use('WXAgg')
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg
from matplotlib.figure import Figure
from matplotlib.pyplot import gcf, setp
import calblitz as cb

class Knob:
    """
    Knob - simple class with a "setKnob" method.  
    A Knob instance is attached to a Param instance, e.g. param.attach(knob)
    Base class is for documentation purposes.
    """
    def setKnob(self, value):
        pass


class Param:
    """
    The idea of the "Param" class is that some parameter in the GUI may have
    several knobs that both control it and reflect the parameter's state, e.g.
    a slider, text, and dragging can all change the value of the frequency in
    the waveform of this example.  
    The class allows a cleaner way to update/"feedback" to the other knobs when 
    one is being changed.  Also, this class handles min/max constraints for all
    the knobs.
    Idea - knob list - in "set" method, knob object is passed as well
      - the other knobs in the knob list have a "set" method which gets
        called for the others.
    """
    def __init__(self, initialValue=None, minimum=0., maximum=1.,step=.1):
        self.minimum = minimum
        self.maximum = maximum
        self.step = step
        if initialValue != self.constrain(initialValue):
            raise ValueError('illegal initial value')
        self.value = initialValue
        self.knobs = []
        
    def attach(self, knob):
        self.knobs += [knob]
        
    def set(self, value, knob=None):
        self.value = value
        self.value = self.constrain(value)
        for feedbackKnob in self.knobs:
            if feedbackKnob != knob:
                feedbackKnob.setKnob(self.value)
        return self.value

    def constrain(self, value):
        if value <= self.minimum:
            value = self.minimum
        if value >= self.maximum:
            value = self.maximum
        return value


class BoolParam:
    def __init__(self, initialValue=True):        
        if initialValue != self.constrain(initialValue):
            raise ValueError('illegal initial value')
        self.value = initialValue
        self.knobs = []   
        
    def attach(self, knob):
        self.knobs += [knob]    
            
    def set(self, value, knob=None):
#        import pdb
#        pdb.set_trace()
        self.value = value
        self.value = self.constrain(value)
        for feedbackKnob in self.knobs:
            if feedbackKnob != knob:
                feedbackKnob.setKnob(self.value)
        return self.value
    
    def constrain(self, value):
        return value



class ToggleButton(Knob):
    def __init__(self, parent, label, param):
        self.toggle_button = wx.ToggleButton(parent, -1,label)
        self.setKnob(param.value)
        
        self.toggle_button.Bind(wx.EVT_TOGGLEBUTTON, self.toggleButtonHandler)
        
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer.Add(self.toggle_button, 1, wx.EXPAND)
        self.sizer=sizer
      
        self.param = param
        self.param.attach(self)

    def toggleButtonHandler(self, evt):        
        value = self.toggle_button.GetValue()
        self.param.set(value)        
        
    def setKnob(self, value):
        pass



class SliderGroup(Knob):
    def __init__(self, parent, label, param):
        self.sliderLabel = wx.StaticText(parent, label=label)
        self.sliderText = wx.TextCtrl(parent, -1, style=wx.TE_PROCESS_ENTER)
        self.slider = wx.Slider(parent, -1)
        self.slider.SetMax(param.maximum*1000)
        self.slider.SetMin(param.minimum*1000)
        self.slider.SetPageSize(param.step*1000)
        self.setKnob(param.value)
        
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer.Add(self.sliderLabel, 0, wx.EXPAND | wx.ALIGN_CENTER | wx.ALL, border=2)
        sizer.Add(self.sliderText, 0, wx.EXPAND | wx.ALIGN_CENTER | wx.ALL, border=2)
        sizer.Add(self.slider, 1, wx.EXPAND)
        self.sizer = sizer

        self.slider.Bind(wx.EVT_SLIDER, self.sliderHandler)
        self.sliderText.Bind(wx.EVT_TEXT_ENTER, self.sliderTextHandler)

        self.param = param
        self.param.attach(self)

    def sliderHandler(self, evt):
        value = evt.GetInt() / 1000.
        self.param.set(value)
        
    def sliderTextHandler(self, evt):
        value = float(self.sliderText.GetValue())
        self.param.set(value)
        
    def setKnob(self, value):
        self.sliderText.SetValue('%g'%value)
        self.slider.SetValue(value*1000)


class FourierDemoFrame(wx.Frame):
    def __init__(self, *args, **kwargs):
        wx.Frame.__init__(self, *args, **kwargs)

        self.fourierDemoWindow = FourierDemoWindow(self)
        self.frequencySliderGroup = SliderGroup(self, label='Brightness:', \
            param=self.fourierDemoWindow.f0)
        self.framesSliderGroup = SliderGroup(self, label=' Frames #:', \
            param=self.fourierDemoWindow.A)
            
        self.toggleButtonBCGND = ToggleButton(self, label= 'BCGKND',param=self.fourierDemoWindow.toggle_background)            
        
        

#        self.checkBoxMask1=wx.CheckBox(self.cbbanel, -1, 'Background', (10, 10))
#        self.checkBoxMask1.SetValue(True)
#        self.checkBoxMask2=wx.CheckBox(self.cbbanel, -1, 'Mask GT', (10, 10))
#        self.checkBoxMask2.SetValue(True)
    
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.toggleButtonBCGND.sizer , 0, wx.EXPAND | wx.ALIGN_CENTER | wx.ALL, border=5)
        sizer.Add(self.fourierDemoWindow, 1, wx.EXPAND)
        sizer.Add(self.frequencySliderGroup.sizer, 0, \
            wx.EXPAND | wx.ALIGN_CENTER | wx.ALL, border=5)
        sizer.Add(self.framesSliderGroup.sizer, 0, \
            wx.EXPAND | wx.ALIGN_CENTER | wx.ALL, border=5)
        self.SetSizer(sizer)
        
        
        
class FourierDemoWindow(wx.Window, Knob):
    def __init__(self, *args, **kwargs):
        wx.Window.__init__(self, *args, **kwargs)
        self.lines = []
        self.imgs = []
        self.bkgrnd = []
        self.mean = []
        self.figure = Figure()
        self.canvas = FigureCanvasWxAgg(self, -1, self.figure)
        self.canvas.callbacks.connect('button_press_event', self.mouseDown)
        self.canvas.callbacks.connect('motion_notify_event', self.mouseMotion)
        self.canvas.callbacks.connect('button_release_event', self.mouseUp)
        
        self.state = ''
        self.mov=cb.load('movies/demoMovie.tif',fr=20)
        with np.load('results_analysis.npz')  as ld:
            self.comps=np.transpose(np.reshape(ld['A2'],(ld['d1'],ld['d2'],-1),order='F'),[2,0,1])
            
        self.mean = np.mean(self.mov,0)
        self.mouseInfo = (None, None, None, None)
        self.f0 = Param(np.percentile(self.mean,95), minimum=np.min(self.mean), maximum=np.max(self.mean))
        self.A = Param(1, minimum=0, maximum=self.mov.shape[0],step=1)
        self.toggle_background = BoolParam(True)
        self.draw()
        
        # Not sure I like having two params attached to the same Knob,
        # but that is what we have here... it works but feels kludgy -
        # although maybe it's not too bad since the knob changes both params
        # at the same time (both f0 and A are affected during a drag)
        self.f0.attach(self)
        self.A.attach(self)
        self.toggle_background.attach(self)
        
        self.Bind(wx.EVT_SIZE, self.sizeHandler)
       
    def sizeHandler(self, *args, **kwargs):
        self.canvas.SetSize(self.GetSize())
        
    def mouseDown(self, evt):
        if self.lines[0] in self.figure.hitlist(evt):
            self.state = 'frequency'
        elif self.lines[1] in self.figure.hitlist(evt):
            self.state = 'time'
        else:
            self.state = ''
        self.mouseInfo = (evt.xdata, evt.ydata, max(self.f0.value, .1), self.A.value)

    def mouseMotion(self, evt):
        if self.state == '':
            return
        x, y = evt.xdata, evt.ydata
        if x is None:  # outside the axes
            return
        x0, y0, f0Init, AInit = self.mouseInfo
        self.A.set(AInit+(AInit*(y-y0)/y0), self)
        if self.state == 'frequency':
            self.f0.set(f0Init+(f0Init*(x-x0)/x0))
        elif self.state == 'time':
            if (x-x0)/x0 != -1.:
                self.f0.set(1./(1./f0Init+(1./f0Init*(x-x0)/x0)))
                    
    def mouseUp(self, evt):
        self.state = ''

    def draw(self):
        if not hasattr(self, 'subplot1'):
            self.subplot1 = self.figure.add_subplot(121)
            self.subplot2 = self.figure.add_subplot(122)
        x1, y1, x2, y2 = self.compute(self.f0.value, self.A.value)
        color = (1., 0., 0.)
        
        self.bkgrnd = self.subplot1.imshow(self.mean, cmap='gray',vmax=self.f0.value)
#        self.lines += self.subplot2.plot(x2, y2, color=color, linewidth=2)
        self.img = self.subplot2.imshow(self.mov[0], cmap='gray')

        #Set some plot attributes
        self.subplot1.set_title("Overlaid components", fontsize=12)
        self.subplot2.set_title("Movie frames", fontsize=12)
#        self.subplot1.set_ylabel("Frequency Domain Waveform X(f)", fontsize = 8)
#        self.subplot1.set_xlabel("frequency f", fontsize = 8)
#        self.subplot2.set_ylabel("Time Domain Waveform x(t)", fontsize = 8)
#        self.subplot2.set_xlabel("time t", fontsize = 8)
#        self.subplot1.set_xlim([-6, 6])
#        self.subplot1.set_ylim([0, 1])
#        self.subplot2.set_xlim([-2, 2])
#        self.subplot2.set_ylim([-2, 2])
#        self.subplot1.text(0.05, .95, r'$X(f) = \mathcal{F}\{x(t)\}$', \
#            verticalalignment='top', transform = self.subplot1.transAxes)
#        self.subplot2.text(0.05, .95, r'$x(t) = a \cdot \cos(2\pi f_0 t) e^{-\pi t^2}$', \
#            verticalalignment='top', transform = self.subplot2.transAxes)

    def compute(self, f0, A):
        f = np.arange(-6., 6., 0.02)
        t = np.arange(-2., 2., 0.01)
        x = A*np.cos(2*np.pi*f0*t)*np.exp(-np.pi*t**2)
        X = A/2*(np.exp(-np.pi*(f-f0)**2) + np.exp(-np.pi*(f+f0)**2))
        return f, X, t, x

    def repaint(self):
        self.canvas.draw()

    def setKnob(self, value):
        # Note, we ignore value arg here and just go by state of the params
#        x1, y1, x2, y2 = self.compute(self.f0.value, self.A.value)

        #setp(self.lines[0], xdata=np.arange(100), ydata=np.random.random((100,)))

        
        if self.toggle_background.value == True:
            self.bkgrnd.set_data(self.mean)
            self.bkgrnd.set_clim(vmax=self.f0.value)
        else:
            self.bkgrnd.set_data(self.mean*np.nan)
            
        self.img.set_data(self.mov[np.int(self.A.value)])

        self.repaint()


class App(wx.App):
    def OnInit(self):
        self.frame1 = FourierDemoFrame(parent=None, title="Fourier Demo", size=(640, 480))
        self.frame1.Show()
        return True
        
app = App()
app.MainLoop()
app.Destroy()