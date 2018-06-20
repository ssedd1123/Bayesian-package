from __future__ import print_function
import sys
import os.path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from matplotlib.widgets import Slider, Button, RadioButtons
# matplotlib requires wxPython 2.8+
# set the wxPython version in lib\site-packages\wx.pth file
# or if you have wxversion installed un-comment the lines below
#import wxversion
#wxversion.ensureMinimal('2.8')

import random
import cPickle as pickle
import pandas as pd
import sys
import time
import os
import gc
import matplotlib
matplotlib.use('WXAgg')
from matplotlib.backends.backend_wxagg import Toolbar, FigureCanvasWxAgg
from matplotlib.figure import Figure
import tempfile


#from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.backends.backend_wx import NavigationToolbar2Wx
#from matplotlib.figure import Figure

import numpy as np
from copy import deepcopy

import wx
#import wx.xrc as xrc
import wx.grid as gridlib

class EmulatorTest(wx.Frame):
    def __init__(self, parent, emulator, prior):
        wx.Frame.__init__(self, parent, wx.NewId())
        panel = wx.Panel(self)

        self.fig = Figure((5, 4), 75)
        # Adjust the subplots region to leave some space for the sliders and buttons
        self.fig.subplots_adjust(left=0.25, bottom=0.25)
        self.canvas = FigureCanvasWxAgg(self, -1, self.fig)
        self.toolbar = NavigationToolbar2Wx(self.canvas)  # matplotlib toolbar
        self.toolbar.Realize()

        # Now put all into a sizer
        sizer = wx.BoxSizer(wx.VERTICAL)
        # This way of adding to sizer allows resizing
        sizer.Add(self.canvas, 1, wx.LEFT | wx.TOP | wx.GROW)
        sizer.Add(self.toolbar, 0, wx.GROW)
        # Best to allow the toolbar to resize!
        self.SetSizer(sizer)
        self.Fit()
       
        self.graph = self.fig.add_subplot(111)
        
        """
        Calculation with emulator
        """
        self.emulator = emulator
        prior_range = {}
        ini_par = []
        for par_name in list(prior):
            prior_range[par_name] = (prior[par_name][0], prior[par_name][1])
            ini_par.append(0.5*(prior[par_name][0] + prior[par_name][1]))

        ini_par = np.array(ini_par)
        result, var = emulator.Emulate(ini_par)
        self.num_output = result.shape[0]

        result, var = self.signal(ini_par)
        self.xaxis =  np.arange(0, self.num_output)
        self.line, _, (self.bars,) = self.graph.errorbar(self.xaxis, result, yerr=np.sqrt(np.diag(var)), marker='o', linewidth=2, color='red')
        self.graph.autoscale()

        self.graph.set_xlim([-1, self.num_output+1])
        #self.graph.set_ylim([0.5, 1.9])

        """
        Add slider bar
        """
        self.amp_slider = []
        for index, par_name in enumerate(list(prior)):
            amp_slider_ax  = self.fig.add_axes([0.25, 0.1 + 0.05*index, 0.65, 0.03])
            self.amp_slider.append(Slider(amp_slider_ax, par_name, prior_range[par_name][0], prior_range[par_name][1], valinit=ini_par[index]))

        
        for slider in self.amp_slider:
            slider.on_changed(self.sliders_on_changed)

        self.toolbar.update()  # Not sure why this is needed - ADS

    # Define an action for modifying the line when any slider's value changes
    def sliders_on_changed(self, val):
        par = []
        for slider in self.amp_slider:
            par.append(slider.val)
        ydata, var = self.signal(np.array(par))
        err = np.sqrt(np.diag(var))
    
        yerr_top = ydata + err
        yerr_bot = ydata - err
        x_base = self.line.get_xdata()
        new_segments = [np.array([[x, yt], [x, yb]]) for
                        x, yt, yb in zip(x_base, yerr_top, yerr_bot)]
        self.line.set_ydata(ydata)
        self.bars.set_segments(new_segments)
    
        self.fig.canvas.draw_idle()


    def GetToolBar(self):
        # You will need to override GetToolBar if you are using an
        # unmanaged toolbar in your frame
        return self.toolbar

    def onEraseBackground(self, evt):
        # this is supposed to prevent redraw flicker on some X servers...
        pass

    def signal(self, par):
        result, var = self.emulator.Emulate(par)
        return result, var

if __name__ == "__main__":
    app = wx.App(0)
    with open('../result/e120_LOO.pkl', 'rb') as buff:
        data = pickle.load(buff)

    emulator, trace  = data['model'], data['trace']
    dataloader = data['data']['data']
    prior = data['prior']
    frame = EmulatorTest(None, emulator, prior)
    frame.Show()
    app.MainLoop()




