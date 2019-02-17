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
    def __init__(self, parent, emulator, prior, exp_data=None, model_data=None):
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
        self.graph = self.fig.add_subplot(111)

        run_num = ['exp'] 
        if model_data is not None:
            run_num = run_num + [str(i) for i in range(0, emulator.emulator_list[0].input_.shape[0])]
        lst = wx.ListBox(left_panel, size=(100,300), style=wx.LB_SINGLE, choices=run_num)#, choices=run_num, style=wx.LB_SINGLE)
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer.Add(lst, 1, wx.EXPAND)
        left_panel.SetSizer(sizer)

        splitterLR.SplitVertically(left_panel, right_panel, 100)
                
        """
        Calculation with emulator
        """
        self.emulator = emulator
        prior_range = {}
        ini_par = []
        for par_name, row in prior.iterrows():
            prior_range[par_name] = (row[1], row[2])
            ini_par.append(0.5*(row[1] + row[2]))

        ini_par = np.array(ini_par)
        result, var = emulator.Emulate(ini_par)
        self.num_output = result.shape[0]

        result, var = self.signal(ini_par)
        self.xaxis =  np.arange(0, self.num_output)
        self.line, _, (self.bars,) = self.graph.errorbar(self.xaxis, result, yerr=np.sqrt(np.diag(var)), marker='o', linewidth=2, color='red')
        self.exp_data = exp_data
        self.model_data = model_data
        if exp_data is None:
            exp_data = result
        self.bg_line, = self.graph.plot(self.xaxis, exp_data, marker='o', linewidth=2, color='b')
        self.graph.autoscale()

        self.graph.set_xlim([-1, self.num_output+1])
        #self.graph.set_ylim([0.5, 1.9])

        """
        Add slider bar
        """
        self.amp_slider = []
        for index, par_name in enumerate(list(prior.index.values)):
            amp_slider_ax  = self.fig.add_axes([0.25, 0.1 + 0.05*index, 0.65, 0.03])
            self.amp_slider.append(Slider(amp_slider_ax, par_name, prior_range[par_name][0], prior_range[par_name][1], valinit=ini_par[index]))

        
        for slider in self.amp_slider:
            slider.on_changed(self.sliders_on_changed)

        self.grid = MyGrid(bottom_panel, (1, len(prior) + self.num_output), False)      
        index = 0
        self.grid.SetRowLabelValue(0, "Value")
        for par_name, row in prior.iterrows():
            self.grid.SetColLabelValue(index, par_name)
            index = index + 1

        # set the result as read only
        for index in range(0, self.num_output):
            self.grid.SetReadOnly(0, index + len(prior), True)
            self.grid.SetCellBackgroundColour(0, index + len(prior), (211,211,211))

        sizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer.Add(self.grid, 1, wx.LEFT | wx.TOP | wx.GROW)
        bottom_panel.SetSizer(sizer)
        
        # add grid to the bottom of the window
        splitterTB.SplitHorizontally(splitterLR, bottom_panel, 300)
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

        # write data into the bottom grid
        self.grid.SetValue([[0, 0], [0, len(par) + ydata.shape[0]]], [par + ydata.tolist()])
    
        self.fig.canvas.draw_idle()

    def onListBox(self, event):
        chosen = event.GetEventObject().GetStringSelection()
        if chosen != 'exp':
            num = int(chosen)
            data = self.model_data[num,:]
            """
            mean = np.array([emu.target[num] for emu in self.emulator.emulator_list])
            # calculate the input data 
            data = self.emulator.output_pipe.TransformInv(mean.flatten())        
            """
            # calculate the input parameters
            par = self.emulator.input_pipe.TransformInv(self.emulator.emulator_list[0].input_)
            # set marker on slider 
            for index, val in enumerate(par[num]):
                self.amp_slider[index].vline.set_xdata([val, val])
        else:
            data = self.exp_data
        self.bg_line.set_ydata(data)
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

    emulator  = data['emulator']
    prior = data['data'].prior
    frame = EmulatorTest(None, emulator, prior)
    frame.Show()
    app.MainLoop()




