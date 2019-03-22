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
import pickle as pickle
import pandas as pd
import sys
import time
import os
import gc
import matplotlib
matplotlib.use('WXAgg')
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg
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
from GUI.Grid import MyGrid

class EmulatorTest(wx.Frame):
    def __init__(self, parent, emulator, store):
        wx.Frame.__init__(self, parent, wx.NewId())
        splitterTB = wx.SplitterWindow(self, -1)
        splitterLR = wx.SplitterWindow(splitterTB, -1)
        bottom_panel = wx.Panel(splitterTB, -1)

        right_panel = wx.Panel(splitterLR, -1)
        left_panel = wx.Panel(splitterLR, -1)

        self.fig = Figure((5, 4), 75)
        # Adjust the subplots region to leave some space for the sliders and buttons
        self.fig.subplots_adjust(left=0.25, bottom=0.25)
        self.canvas = FigureCanvasWxAgg(right_panel, -1, self.fig)
        self.toolbar = NavigationToolbar2Wx(self.canvas)  # matplotlib toolbar
        self.toolbar.Realize()

        # Now put all into a sizer
        sizer = wx.BoxSizer(wx.VERTICAL)
        # This way of adding to sizer allows resizing
        sizer.Add(self.canvas, 1, wx.LEFT | wx.TOP | wx.GROW)
        sizer.Add(self.toolbar, 0, wx.GROW)
        # Best to allow the toolbar to resize!
        right_panel.SetSizer(sizer)
        #self.graph = self.fig.add_subplot(111)

        self.prior = store['PriorAndConfig']
        self.model_X = store['Model_X']
        self.model_Y = store['Model_Y']
        self.exp_Y = store['Exp_Y']
        self.exp_Yerr = store['Exp_YErr']

        num_par = self.prior.shape[0]
        num_features = self.model_Y.shape[1]
        num_samples = self.model_Y.shape[0]


        """
        Create list on the right hand side for people to choose which sample to look at
        """
        run_num = ['exp'] 
        run_num += [str(i) for i in range(num_samples)]
        lst = wx.ListBox(left_panel, size=(100,300), style=wx.LB_SINGLE, choices=run_num)#
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer.Add(lst, 1, wx.EXPAND)
        left_panel.SetSizer(sizer)
        splitterLR.SplitVertically(left_panel, right_panel, 100)

        """
        fraction of upper right sizer that should be graph vs slider
        """
        graph_fraction = 0.7
        graph_pad = 0.05

        slider_fraction = 1 - graph_fraction
        nsliders = num_par

        """
        calculate slider location. In fraction of slider location
        """
        slider_pad = 0.1
        slider_gap = 0.05
        slider_height = (1 - 2*slider_pad - slider_gap*(nsliders - 1))/nsliders
        slider_box_height = slider_gap + slider_height

        """
        Convert slider fraction into total sizer fraction
        """
        slider_pad = slider_pad*slider_fraction
        slider_gap = slider_gap*slider_fraction
        slider_height = slider_height*slider_fraction
        slider_box_height = slider_box_height*slider_fraction

                
        """
        Calculation with emulator
        """
        self.emulator = emulator
        ini_par = 0.5*(self.prior['Max'] + self.prior['Min']).values
        result, var = emulator.Predict(ini_par.reshape(1, -1))

        self.graph = self.fig.add_axes([0.25, slider_fraction + graph_pad, 0.65, graph_fraction - graph_pad])
        result, var = self.signal(ini_par)
        self.xaxis =  np.arange(self.model_Y.shape[1])
        self.line, _, (self.bars,) = self.graph.errorbar(self.xaxis, 
                                                         result, 
                                                         yerr=np.sqrt(np.diag(var)), 
                                                         marker='o', linewidth=2, color='red')
        self.bg_line, _, (self.bg_bars,)  = self.graph.errorbar(self.xaxis, 
                                                                np.squeeze(self.exp_Y), 
                                                                yerr=np.squeeze(self.exp_Yerr), 
                                                                marker='o', linewidth=2, color='b')

        self.graph.set_xlim([-1, num_features+1])
        self.graph.set_ylim([np.min(self.model_Y.values), np.max(self.model_Y.values)])

        """
        Add slider bar
        """
        self.amp_slider = []
        for idx, (par_name, row) in enumerate(self.prior.iterrows()):
            amp_slider_ax  = self.fig.add_axes([0.25, 
                                                slider_pad + 0.5*slider_gap + slider_box_height*idx, 
                                                0.65, slider_height])
            self.amp_slider.append(Slider(amp_slider_ax, 
                                          par_name, 
                                          row['Min'], row['Max'], valinit=ini_par[idx]))

        
        for slider in self.amp_slider:
            slider.on_changed(self.sliders_on_changed)

        # total number of columns in bottom grid = number of parameters + number of features
        self.grid = MyGrid(bottom_panel, (1, num_par + num_features), False)      
        self.grid.SetRowLabelValue(0, "Value")
        for idx, (par_name, row) in enumerate(self.prior.iterrows()):
            self.grid.SetColLabelValue(idx, par_name)

        # set the result as read only
        for index in range(num_par, num_par + num_features):
            self.grid.SetReadOnly(0, index, True)
            self.grid.SetCellBackgroundColour(0, index, (211,211,211))

        sizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer.Add(self.grid, 1, wx.LEFT | wx.TOP | wx.GROW)
        bottom_panel.SetSizer(sizer)
        
        # add grid to the bottom of the window
        splitterTB.SplitHorizontally(splitterLR, bottom_panel, 270)
        # When expanded, the grid on the bottom remains the same height
        splitterTB.SetSashGravity(1.0)
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(splitterTB, 1, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP | wx.GROW)
        self.SetSizer(sizer)
        self.Fit()
        self.Centre()
        self.Bind(wx.EVT_LISTBOX, self.onListBox, lst)
        self.grid.Bind(gridlib.EVT_GRID_CELL_CHANGED, self.OnCellChange)

        self.toolbar.update()  # Not sure why this is needed - ADS

    def OnCellChange(self, evt):
        col = evt.GetCol()
        self.amp_slider[col].set_val(float(self.grid.GetCellValue(0, col)))
        self.sliders_on_changed(0)

    # Define an action for modifying the line when any slider's value changes
    def sliders_on_changed(self, val):
        par = [slider.val for slider in self.amp_slider]
        ydata, var = self.signal(np.array(par))
        err = np.sqrt(np.diag(var))
    
        self.SetErrorBar(self.line, self.bars, ydata, err)
        # write data into the bottom grid
        self.grid.SetValue([[0, 0], [0, len(par) + ydata.shape[0]]], [par + ydata.tolist()])
        

    def onListBox(self, event):
        chosen = event.GetEventObject().GetStringSelection()
        if chosen != 'exp':
            num = int(chosen)
            data = self.model_Y.values[num,:]
            par = self.model_X.values[num,:]
            err = 0
            # set marker on slider 
            for index, val in enumerate(par):
                self.amp_slider[index].vline.set_xdata([val, val])
        else:
            data = self.exp_Y
            err = np.squeeze(self.exp_Yerr)

        self.SetErrorBar(self.bg_line, self.bg_bars, data, err)

    def SetErrorBar(self, line, errorbar, values, err):
        yerr_top = values + err
        yerr_bot = values - err
        line.set_ydata(values)
        x_base = line.get_xdata()
        new_segments = [np.array([[x, yt], [x, yb]]) for
                        x, yt, yb in zip(x_base, yerr_top, yerr_bot)]
        errorbar.set_segments(new_segments)
        self.fig.canvas.draw_idle()
        

    def GetToolBar(self):
        # You will need to override GetToolBar if you are using an
        # unmanaged toolbar in your frame
        return self.toolbar

    def onEraseBackground(self, evt):
        # this is supposed to prevent redraw flicker on some X servers...
        pass

    def signal(self, par):
        result, var = self.emulator.Predict(par.reshape(1, -1))
        return np.squeeze(result), np.squeeze(var)

if __name__ == "__main__":
    app = wx.App(0)
    store = pd.HDFStore('result/test.h5')

    config = store.get_storer('PriorAndConfig').attrs.my_attribute
    from numpy import array
    from Preprocessor.PipeLine import *
    emulator  = eval(config['repr'])
    emulator.Fit(store['Model_X'].values, store['Model_Y'].values)
    frame = EmulatorTest(None, emulator, store)
    frame.Show()
    app.MainLoop()




