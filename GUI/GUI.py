"""
Copyright (C) 2003-2004 Andrew Straw, Jeremy O'Donoghue and others

License: This work is licensed under the PSF. A copy should be included
with this source code, and is also available at
http://www.python.org/psf/license.html

This is yet another example of using matplotlib with wx.  Hopefully
this is pretty full-featured:

  - both matplotlib toolbar and WX buttons manipulate plot
  - full wxApp framework, including widget interaction
  - XRC (XML wxWidgets resource) file to create GUI (made with XRCed)

This was derived from embedding_in_wx and dynamic_image_wxagg.

Thanks to matplotlib and wx teams for creating such great software!

"""
from __future__ import print_function

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
import matplotlib.cm as cm
import matplotlib.cbook as cbook
from matplotlib.backends.backend_wxagg import Toolbar, FigureCanvasWxAgg
from matplotlib.figure import Figure
import tempfile
from matplotlib.backends.backend_wx import NavigationToolbar2Wx
import numpy as np
from copy import deepcopy
import wx
#import wx.xrc as xrc
import wx.grid as gridlib

from ID import *
from ..Training import Training
from TrainingFrame import TrainingFrame
from Grid import MyGrid


matplotlib.rc('image', origin='lower')


class LeftPanel(wx.Panel):

    def __init__(self, parent, plotpanel):
        wx.Panel.__init__(self, parent)
        self.plotpanel = plotpanel
        self.parent = parent

        self.grid = MyGrid(self)
        sizer = wx.BoxSizer(wx.VERTICAL)

     # toolbar
        save_ico = wx.ArtProvider.GetBitmap(wx.ART_FILE_SAVE, wx.ART_TOOLBAR, (16,16))
        self.toolbar = wx.ToolBar(self, id=100, style=wx.TB_HORIZONTAL | wx.NO_BORDER |
                                        wx.TB_FLAT | wx.TB_TEXT)
        self.toolbar.AddSimpleTool(ID_UNDO, save_ico, 'Undo', '')
        self.toolbar.AddSimpleTool(ID_REDO, wx.Bitmap('/projects/hira/tsangc/GaussianEmulator/development/human-icon-theme/16x16/stock/generic/stock_exit.png'), 'Redo', '')
        self.toolbar.AddSimpleTool(ID_PLOT, wx.Bitmap('/projects/hira/tsangc/GaussianEmulator/development/human-icon-theme/16x16/stock/generic/stock_exit.png'), 'plot', '')
        open_ico = wx.ArtProvider.GetBitmap(wx.ART_FILE_OPEN, wx.ART_TOOLBAR, (16,16))
        self.toolbar.AddSimpleTool(ID_INDOPENFILE, open_ico, 'Open csv', '')
 

        self.toolbar.EnableTool(ID_UNDO, False)
        self.toolbar.EnableTool(ID_REDO, False)
        #self.toolbar.AddSeparator()
        #self.toolbar.AddSimpleTool(ID_EXIT, wx.Bitmap('/projects/hira/tsangc/GaussianEmulator/development/human-icon-theme/16x16/stock/generic/stock_exit.png'),'Quit', '')
        self.toolbar.Realize()
        self.toolbar.Bind(wx.EVT_TOOL, self.OnUndo, id=ID_UNDO)
        self.toolbar.Bind(wx.EVT_TOOL, self.OnRedo, id=ID_REDO)
        self.toolbar.Bind(wx.EVT_TOOL, self.OnPlot, id=ID_PLOT)
        self.toolbar.Bind(wx.EVT_TOOL, self.OnOpen, id=ID_INDOPENFILE)
        
        sizer.Add(self.toolbar, border=5)
        sizer.Add(self.grid, 1., wx.LEFT | wx.TOP | wx.GROW)
        self.SetSizer(sizer)
        self.Fit()

    def OnOpen(self, event):
        """
        Create and show the Open FileDialog
        """
        dlg = wx.FileDialog(
            self, message="Choose a file",
            defaultFile="",
            style=wx.OPEN | wx.MULTIPLE | wx.CHANGE_DIR
            )
        result = dlg.ShowModal() 
        path = dlg.GetPaths()
        dlg.Destroy()

        if result != wx.ID_OK:
            return False

        df = pd.read_csv(path[0])
        data = [df.columns.tolist()] + df.values.tolist()
        self.grid.ClearRange([[0,0], [self.grid.num_col - 1, self.grid.num_row - 1]])
        self.grid.SetValue([[0,0], [len(data) - 1, len(data[0]) - 1]], data)
        

    def OnUndo(self, event):
        if len(self.grid.stockUndo) == 0:
            return

        a = self.grid.stockUndo.pop()
        if len(self.grid.stockUndo) == 0:
            self.toolbar.EnableTool(ID_UNDO, False)

        a.undo()
        self.grid.stockRedo.append(a)
        self.toolbar.EnableTool(ID_REDO, True)

    def OnRedo(self, event):
        if len(self.grid.stockRedo) == 0:
            return

        a = self.grid.stockRedo.pop()
        if len(self.grid.stockRedo) == 0:
            self.toolbar.EnableTool(ID_REDO, False)

        a.redo()
        self.grid.stockUndo.append(a)

        self.toolbar.EnableTool(ID_UNDO, True)

    def OnPlot(self, event):
        range_ = self.grid.selected_coords
        if range_ is None:
            return
         
        data = self.grid.GetRange(range_)
        data = np.array(data)
        data[data == ''] = np.nan

        try:
            data = np.array([np.genfromtxt(line) for line in data])
            #data = data[~np.isnan(data).any(axis=0)]
        except:
            print ('This array cannot be converted into float. Abort')
            return
        
        if data.shape[0] == 1:
            self.plotpanel.SetData(range(0, data.shape[1]), data[0, :])
        elif len(data.shape) == 1:
            self.plotpanel.SetData(range(0, data.shape[0]), data[:])
        elif data.shape[1] == 1:
            self.plotpanel.SetData(range(0, data.shape[0]), data[:, 0])
        elif data.shape[0] == 2:
            self.plotpanel.SetData(data[0, :], data[1, :])
        elif data.shape[1] == 2:
            self.plotpanel.SetData(data[:, 0], data[:, 1])
        
        else:
            print(data, 'data shape not recognized')


########################################################################
class RightPanel(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent, -1)

        self.fig = Figure((5, 4), 75)
        self.canvas = FigureCanvasWxAgg(self, -1, self.fig)
        self.toolbar = NavigationToolbar2Wx(self.canvas)  # matplotlib toolbar
        self.toolbar.Realize()
        # self.toolbar.set_active([0,1])

        # Now put all into a sizer
        sizer = wx.BoxSizer(wx.VERTICAL)
        # This way of adding to sizer allows resizing
        sizer.Add(self.canvas, 1, wx.LEFT | wx.TOP | wx.GROW)
        # Best to allow the toolbar to resize!
        sizer.Add(self.toolbar, 0, wx.GROW)
        self.SetSizer(sizer)
        self.Fit()
       
        self.init_plot_data()

    def init_plot_data(self):
        a = self.fig.add_subplot(111)

        x = np.arange(100.0) * 2 * np.pi / 60.0
        y = np.arange(100.0) * 2 * np.pi / 50.0
        self.lines = a.plot(x, y, 'ro')

        self.toolbar.update()  # Not sure why this is needed - ADS

    def GetToolBar(self):
        # You will need to override GetToolBar if you are using an
        # unmanaged toolbar in your frame
        return self.toolbar

    def SetData(self, xdata, ydata):
        print(xdata, ydata)
        self.lines[0].set_data(xdata, ydata)

        self.canvas.draw()

    def onEraseBackground(self, evt):
        # this is supposed to prevent redraw flicker on some X servers...
        pass

class TabPanel(wx.Panel):
    #----------------------------------------------------------------------
    def __init__(self, parent):
        """"""
        wx.Panel.__init__(self, parent=parent)
 
        colors = ["red", "blue", "gray", "yellow", "green"]
        self.SetBackgroundColour(random.choice(colors))
 
        btn = wx.Button(self, label="Press Me")
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(btn, 0, wx.ALL, 10)
        self.SetSizer(sizer)
 

class CommonToolBar(wx.ToolBar):

    def __init__(self, parent, tab1, tab2, tab3, **args):
        wx.ToolBar.__init__(self, parent, **args)
        self.tab1 = tab1
        self.tab2 = tab2
        self.tab3 = tab3
        self.parent = parent
        

        save_ico = wx.ArtProvider.GetBitmap(wx.ART_FILE_SAVE, wx.ART_TOOLBAR, (16,16))
        self.AddSimpleTool(ID_SAVE, save_ico, 'Save', '')
        open_ico = wx.ArtProvider.GetBitmap(wx.ART_FILE_OPEN, wx.ART_TOOLBAR, (16,16))
        self.AddSimpleTool(ID_OPENFILE, open_ico, 'Open', '')
        new_ico = wx.ArtProvider.GetBitmap(wx.ART_NEW, wx.ART_TOOLBAR, (16,16))
        self.AddSimpleTool(ID_SAVEAS, new_ico, 'Save As', '')
 

        self.Realize()
        self.Bind(wx.EVT_TOOL, self.OnFile, id=ID_OPENFILE)
        self.Bind(wx.EVT_TOOL, self.OnSave, id=ID_SAVE)
        self.Bind(wx.EVT_TOOL, self.OnSaveNew, id=ID_SAVEAS)
 
        self.opened_filename = None
        self.opened_data = None


    def OnSaveNew(self, event):

        model = self.tab2.grid.GetAllValues()
        headers = model.pop(0)
        model = pd.DataFrame(model, columns=headers)

        prior = self.tab1.grid.GetAllValues()
        headers = prior.pop(0)
        prior = pd.DataFrame(prior, columns=headers)

        exp = self.tab3.grid.GetAllValues()
        headers = exp.pop(0)
        exp = pd.DataFrame(exp, columns=headers)

        """
        Create and show the Open FileDialog
        """
        wildcard = "Python source (*.pkl)|*.pkl|" \
           "All files (*.*)|*.*"
        dlg = wx.FileDialog(self, message="Save project as ...", defaultFile="", style=wx.SAVE | wx.OVERWRITE_PROMPT)
        result = dlg.ShowModal()            
        outFile = dlg.GetPaths()
        dlg.Destroy()
    
        if result == wx.ID_CANCEL:    #Either the cancel button was pressed or the window was closed
            return False

        with tempfile.NamedTemporaryFile() as tmpmodel, tempfile.NamedTemporaryFile() as tmpprior, tempfile.NamedTemporaryFile() as tmpexp: 
            prior.to_csv(tmpprior.name, sep=',', index=False)
            model.to_csv(tmpmodel.name, sep=',', index=False)
            exp.to_csv(tmpexp.name, sep=',', index=False)

            tmpprior.flush()
            tmpmodel.flush()
            tmpexp.flush()

            args = {}
            args['Prior'] = tmpprior.name
            args['ModelData'] = tmpmodel.name
            args['ExpData'] = tmpexp.name
            args['Training_name'] = outFile[0]
            args['abs'] = True
            args['covariancefunc'] = 'ARD'
            args['principalcomp'] = 1
            args['initialscale'] = [0.5]
            args['initialnugget'] = 1
            args['scalerate'] = 0.003
            args['nuggetrate'] = 0.003
            args['maxsteps'] = 1000

            frame = TrainingFrame()
            res = frame.ShowModal()
            if res == wx.ID_OK:
                frame.AdditionalData(args)
            frame.Destroy()
            
            Training(args)

            with open(outFile[0], 'rb') as buff:
                data = pickle.load(buff)

            self.opened_data = data
            self.opened_filename = outFile
        

    def OnSave(self, event):
        
        model = self.tab2.grid.GetAllValues()
        headers = model.pop(0)
        model = pd.DataFrame(model, columns=headers)
        model.to_csv('/projects/hira/tsangc/GaussianEmulator/development/testing_model.csv', sep=',', index=False)

        prior = self.tab1.grid.GetAllValues()
        headers = prior.pop(0)
        prior = pd.DataFrame(prior, columns=headers)
        with tempfile.NamedTemporaryFile() as temp:
            prior.to_csv(temp.name, sep=',', index=False)
            temp.flush()
            self.opened_data['data'].ChangePrior(temp.name)

        exp = self.tab3.grid.GetAllValues()
        headers = exp.pop(0)
        exp = pd.DataFrame(exp, columns=headers)
        with tempfile.NamedTemporaryFile() as temp:
            exp.to_csv(temp.name, sep=',', index=False)
            temp.flush()
            self.opened_data['data'].ChangeExp(temp.name)

        """
        Create and show the Open FileDialog
        """
        dlg = wx.FileDialog(
            self, message="Save project as ...",
            defaultFile="",
            style=wx.SAVE | wx.OVERWRITE_PROMPT
            )
        result = dlg.ShowModal()            
        inFile = dlg.GetPaths()
        dlg.Destroy()

        if result == wx.ID_OK:          #Save button was pressed
            with open(inFile[0], 'wb') as buff:
                pickle.dump(self.opened_data, buff)
            return True
        elif result == wx.ID_CANCEL:    #Either the cancel button was pressed or the window was closed
            return False


    def OnFile(self, event):
        """
        Create and show the Open FileDialog
        """
        dlg = wx.FileDialog(
            self, message="Choose a file",
            defaultFile="",
            style=wx.OPEN | wx.MULTIPLE | wx.CHANGE_DIR
            )
        result = dlg.ShowModal() 
        path = dlg.GetPaths()
        dlg.Destroy()

        if result != wx.ID_OK:
            return False

        with open(path[0], 'rb') as buff:
            data = pickle.load(buff)

        self.opened_data = data
        """
        Loading prior
        """
        prior = data['data'].prior
        prior = [prior.columns.tolist()] + prior.values.tolist()
        if type(prior[0]) is not list:
            prior = [prior]
        self.tab1.grid.SetValue([[0,0], [len(prior) - 1, len(prior[0]) - 1]], prior)

        """
        Loading model data
        """
        training_data = data['data']
        header = [training_data.par_name + training_data.var_name + [name + "_Error" for name in training_data.var_name]] 
        content = np.concatenate((training_data.sim_para, training_data.sim_data, training_data.sim_error), axis=1).tolist()
        if type(content[0]) is not list:
            content = [content]
        self.tab2.grid.SetValue([[0,0], [0, len(header[0]) - 1]], header)
        self.tab2.grid.SetValue([[1,0], [len(content), len(content[0]) - 1]], content)

        """
        Loading exp data
        """
        header = [training_data.var_name + [name + "_Error" for name in training_data.var_name]] 
        content = np.concatenate((training_data.exp_result, np.sqrt(np.diag(training_data.exp_cov)))).tolist()
        if type(content[0]) is not list:
            content = [content]
        self.tab3.grid.SetValue([[0,0], [0, len(header[0]) - 1]], header)
        self.tab3.grid.SetValue([[1,0], [len(content), len(content[0]) - 1]], content)


class Common(wx.Frame):
 
    def __init__(self, parent):
        wx.Frame.__init__(self, parent, wx.NewId(), "Common", size=(1000,400))
        
        panel = wx.Panel(self)
         
        notebook = wx.Notebook(panel)

        splitter = wx.SplitterWindow(notebook)
        rightP = RightPanel(splitter)
        leftP = LeftPanel(splitter, rightP)
        splitter.SplitVertically(leftP, rightP, 200)
        splitter.SetMinimumPaneSize(500)
        notebook.AddPage(splitter, "Para prior")

        splitter = wx.SplitterWindow(notebook)
        rightP = RightPanel(splitter)
        leftP2 = LeftPanel(splitter, rightP)
        splitter.SplitVertically(leftP2, rightP, 200)
        splitter.SetMinimumPaneSize(500)
        notebook.AddPage(splitter, "Model result")

        splitter = wx.SplitterWindow(notebook)
        rightP = RightPanel(splitter)
        leftP3 = LeftPanel(splitter, rightP)
        splitter.SplitVertically(leftP3, rightP, 200)
        splitter.SetMinimumPaneSize(500)
        notebook.AddPage(splitter, "Exp data data")

        sizer = wx.BoxSizer(wx.VERTICAL)
                
        
    # toolbar
        self.toolbar = CommonToolBar(panel, tab1=leftP, tab2=leftP2, tab3=leftP3, id=100, style=wx.TB_HORIZONTAL | wx.NO_BORDER |
                                        wx.TB_FLAT | wx.TB_TEXT)

        sizer.Add(self.toolbar, border=5)
        
        sizer.Add(notebook, 1, wx.EXPAND | wx.EXPAND, 5)
        
        panel.SetSizer(sizer)
        self.Layout()
        self.Show()

    



app = wx.App(0)
frame = Common(None)
frame.Show()
app.MainLoop()
