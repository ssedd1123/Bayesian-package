from __future__ import print_function
from matplotlib.figure import Figure
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg
from matplotlib.backends.backend_wx import NavigationToolbar2Wx
import wx.grid as gridlib
import wx
import numpy as np
import matplotlib.cm as cm
import matplotlib.cbook as cbook
from copy import deepcopy
import tempfile

import gc
import os
import pickle as pickle
import random
import sys
import time

import matplotlib
import pandas as pd

# matplotlib requires wxPython 2.8+
# set the wxPython version in lib\site-packages\wx.pth file
# or if you have wxversion installed un-comment the lines below
# import wxversion
# wxversion.ensureMinimal('2.8')


matplotlib.use("WXAgg")

# import wx.xrc as xrc
# from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas

# from matplotlib.figure import Figure


class MatplotlibFrame(wx.Frame):
    def __init__(self, parent, fig):
        wx.Frame.__init__(self, parent, wx.NewId())
        panel = wx.Panel(self)

        self.fig = fig  # Figure((5, 4), 75)
        self.canvas = FigureCanvasWxAgg(self, -1, self.fig)
        self.toolbar = NavigationToolbar2Wx(self.canvas)  # matplotlib toolbar
        self.toolbar.Realize()

        # Now put all into a sizer
        sizer = wx.BoxSizer(wx.VERTICAL)
        # This way of adding to sizer allows resizing
        sizer.Add(self.canvas, 1, wx.LEFT | wx.TOP | wx.GROW)
        # Best to allow the toolbar to resize!
        sizer.Add(self.toolbar, 0, wx.GROW)
        self.SetSizer(sizer)
        self.Fit()

        self.Bind(wx.EVT_CLOSE, self.OnClose)
        self.toolbar.update()  # Not sure why this is needed - ADS

    def GetToolBar(self):
        # You will need to override GetToolBar if you are using an
        # unmanaged toolbar in your frame
        return self.toolbar

    def SetData(self):
        self.canvas.draw()

    def OnClose(self, event):
        self.Destroy()

    def onEraseBackground(self, evt):
        # this is supposed to prevent redraw flicker on some X servers...
        pass


if __name__ == "__main__":
    app = wx.App(0)
    fig = Figure((15, 12), 75)
    x = np.linspace(0, 2, 100)
    y = np.sin(x)
    frame = MatplotlibFrame(None, fig)
    axes2d = fig.subplots(1, 1)
    print(axes2d)
    axes2d.plot(x, y)
    frame.SetData()
    frame.Show()
    print(frame)
    app.MainLoop()
    if frame:
        print(frame.IsShown())
    else:
        print("closed")
