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
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
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

from GUI.ID import *
from Training import Training
from GUI.TrainingFrame import TrainingFrame
from GUI.MatplotlibFrame import MatplotlibFrame
from GUI.Grid import MyGrid
from GUI.PlotFrame import PlotFrame
from GUI.EmulatorTestSliderWX import EmulatorTest
from GUI.EmulatorFrame import EmulatorFrame
from GUI.ProgressDisplay import MyFrame
from Utilities.Utilities import PlotTrace


matplotlib.rc('image', origin='lower')


class GridPanel(wx.Panel):

    def __init__(self, parent, size=(150,100)):
        wx.Panel.__init__(self, parent)
        self.parent = parent

        self.grid = MyGrid(self, size)
        sizer = wx.BoxSizer(wx.VERTICAL)

     # toolbar
        undo_ico = wx.ArtProvider.GetBitmap(wx.ART_UNDO, wx.ART_TOOLBAR, (16,16))
        self.toolbar = wx.ToolBar(self, id=100, style=wx.TB_HORIZONTAL | wx.NO_BORDER |
                                        wx.TB_FLAT | wx.TB_TEXT)
        self.toolbar.AddTool(ID_UNDO, 'Undo', undo_ico, '')
        redo_ico = wx.ArtProvider.GetBitmap(wx.ART_REDO, wx.ART_TOOLBAR, (16,16))
        self.toolbar.AddTool(ID_REDO, 'Redo', redo_ico, '')
        open_ico = wx.ArtProvider.GetBitmap(wx.ART_FILE_OPEN, wx.ART_TOOLBAR, (16,16))
        self.toolbar.AddTool(ID_INDOPENFILE, 'Open csv', open_ico, '')
        print_ico = wx.ArtProvider.GetBitmap(wx.ART_PRINT, wx.ART_TOOLBAR, (16,16))
        self.toolbar.AddTool(ID_PRINT, 'Plot data', print_ico, '')
 

        self.toolbar.EnableTool(ID_UNDO, False)
        self.toolbar.EnableTool(ID_REDO, False)
        self.toolbar.Realize()
        self.toolbar.Bind(wx.EVT_TOOL, self.OnUndo, id=ID_UNDO)
        self.toolbar.Bind(wx.EVT_TOOL, self.OnRedo, id=ID_REDO)
        self.toolbar.Bind(wx.EVT_TOOL, self.OnOpen, id=ID_INDOPENFILE)
        self.toolbar.Bind(wx.EVT_TOOL, self.OnPrint, id=ID_PRINT)
        
        sizer.Add(self.toolbar, border=5)
        sizer.Add(self.grid, 1., wx.LEFT | wx.TOP | wx.GROW)
        self.SetSizer(sizer)
        self.Fit()

    def OnPrint(self, event):
        data = np.array(self.grid.selected_data)
        if data is not None:
            data = np.array(data)
            data[data == ''] = np.nan
            try:
                data = np.array([np.genfromtxt(line) for line in data])
                #data = data[~np.isnan(data).any(axis=0)]
            except:
                print ('This array cannot be converted into float. Abort')
                return
            if len(data.shape) == 1:
                xdata = range(0, data.shape[0])
                ydata = data[:]
            elif data.shape[0] == 1:
                xdata = range(0, data.shape[1])
                ydata = data[0, :]
            elif data.shape[1] == 1:
                xdata = range(0, data.shape[0])
                ydata = data[:, 0]
            elif data.shape[0] == 2:
                xdata = data[0, :]
                ydata = data[1, :]
            elif data.shape[1] == 2:
                xdata = data[:, 0]
                ydata = data[:, 1]
    
            frame = PlotFrame(None, xdata, ydata)
            frame.Show()
        else:
            print('No data is selected')
            

    def OnOpen(self, event):
        """
        Create and show the Open FileDialog
        """
        dlg = wx.FileDialog(
            self, message="Choose a file",
            defaultFile="",
            style=wx.FD_OPEN | wx.FD_MULTIPLE | wx.FD_CHANGE_DIR
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

        del self.grid.stockUndo[:]
        self.toolbar.EnableTool(ID_UNDO, False)
        

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


class CommonMenuBar(wx.MenuBar):

    def __init__(self, parent, tab1, tab2, tab3, **args):
        wx.MenuBar.__init__(self)
        self.tab1 = tab1
        self.tab2 = tab2
        self.tab3 = tab3
        self.parent = parent
        

        fileMenu = wx.Menu()
        SaveMenuItem = fileMenu.Append(ID_SAVE, 'Save', '')
        SaveAsMenuItem = fileMenu.Append(ID_SAVEAS, 'Save As', '')
        OpenMenuItem = fileMenu.Append(ID_OPENFILE, 'Open', '')

        emulatorMenu = wx.Menu()
        EmulatorCheckItem = emulatorMenu.Append(ID_EMULATORCHECK, 'Check emulator', '')
        EmulatorItem = emulatorMenu.Append(ID_EMULATE, 'Start Analysis', '')

        self.Append(fileMenu, '&File')
        self.Append(emulatorMenu, '&Emulator')
        self.Bind(wx.EVT_TOOL, self.OnFile, OpenMenuItem)
        self.Bind(wx.EVT_TOOL, self.OnSave, SaveMenuItem)
        self.Bind(wx.EVT_TOOL, self.OnSaveNew, SaveAsMenuItem)
        self.Bind(wx.EVT_TOOL, self.OnEmulatorCheck, EmulatorCheckItem)
        self.Bind(wx.EVT_TOOL, self.OnEmulate, EmulatorItem)
 
        self.opened_filename = None
        self.opened_data = None
        self.correlation_frame = None

    def _CheckOpenedFile(self):
        if self.opened_filename is None:
            wx.MessageBox('No file is associated with this data. You must save and train the emulator first', 'Error', wx.OK | wx.ICON_ERROR)
            return False

        with open(self.opened_filename, 'rb') as buff:
            data = pickle.load(buff)

        if not data['emulator']:
            wx.MessageBox('Emulator not found in the current file. Have you trained the emulator?', 'Error', wx.OK | wx.ICON_ERROR)
            return False
        return True

    def OnEmulatorCheck(self, event):
        if self._CheckOpenedFile():
            with open(self.opened_filename, 'rb') as buff:
                data = pickle.load(buff)
            frame = EmulatorTest(None, data['emulator'], data['data'].prior, data['data'].exp_result)
            frame.Show()

    def OnEmulate(self, event):
        if self._CheckOpenedFile():
            args = {}
            args['Training_file'] = self.opened_filename
            frame = EmulatorFrame()
            res = frame.ShowModal()
            if res == wx.ID_OK:
                frame.AdditionalData(args)
                progress = MyFrame(None, -1, 'stdout to GUI using multiprocessing', args)# {'Training_file': 'training/test', 'Output_name':'para', 'cores':5, 'steps':10000})
                trace, par_name, prior = progress.OnCalculate()

                # if root_numpy module exist, it will be saved there
                """
                try: 
                    from root_numpy import array2root
                    df_ = trace.copy(deep=False)
                    arr = df_.to_records(index=False)
                    array2root(arr, '%s.root' % self.opened_filename, 'my_ttree', 'recreate')
                except ImportError:
                    print('root_numpy module not found. Will not output to root')
                """
    
                if not self.correlation_frame:
                    fig = Figure((15,12), 75)
                    self.correlation_frame = MatplotlibFrame(None, fig)
                PlotTrace(trace, par_name, prior, self.correlation_frame.fig)
                self.correlation_frame.SetData()
                self.correlation_frame.Show()
        

    def _CheckData(self, prior_headers, prior, model_headers, model, exp_headers, exp):
        """
        This function check if necessary variables are included for a successful emulation
        """
        if not set(prior_headers).issubset(set(model_headers)):
            wx.MessageBox('Some variables in prior do not appear in model. Please check again', 'Error', wx.OK | wx.ICON_ERROR)
            return False
        if not set(exp_headers).issubset(set(model_headers)):
            wx.MessageBox('Some variables in experiment result (including its error) do not appear in model. Please check again', 'Error', wx.OK | wx.ICON_ERROR)
            return False
        if set(exp_headers).union(set(prior_headers)) != set(model_headers):
            wx.MessageBox('Some variables in model do not appear in either experiment or prior. Please check again', 'Error', wx.OK | wx.ICON_ERROR)
            return False
        if prior.shape[0] != 5:
            wx.MessageBox('There could only be 5 rows in prior, one for lower bound and the other for higher, nothing more/less', 'Error', wx.OK | wx.ICON_ERROR)
            return False 
        if exp.shape[0] != 1:
            wx.MessageBox('There could only be 1 rows for exp result, nothing more/less', 'Error', wx.OK | wx.ICON_ERROR)
            return False 
        if model.shape[0] < 3:
            wx.MessageBox('Model data has less than 3 entries. I don\'t think this will work. Please check again', 'Error', wx.OK | wx.ICON_ERROR)
            return False
        print(prior_headers, prior)
        if len(prior_headers) != prior.shape[1]:
            wx.MessageBox('Number of variables and numerical columns in prior do not match.', 'Error', wx.OK | wx.ICON_ERROR)
            return False 
        if len(model_headers) != model.shape[1]:
            wx.MessageBox('Number of variables and numerical columns in model do not match.', 'Error', wx.OK | wx.ICON_ERROR)
            return False 
        if len(exp_headers) != exp.shape[1]:
            wx.MessageBox('Number of variables and numerical columns in exp do not match.', 'Error', wx.OK | wx.ICON_ERROR)
            return False 
        if not any([name.endswith('_Error') for name in exp_headers]):
            wx.MessageBox('There are no columns associated with experimental error. Please fill that in.', 'Error', wx.OK | wx.ICON_ERROR)
            return False
        if prior.isnull().values.any():
            wx.MessageBox('There are empty holes in prior.', 'Error', wx.OK | wx.ICON_ERROR)
            return False
        if model.isnull().values.any():
            wx.MessageBox('There are empty holes in model.', 'Error', wx.OK | wx.ICON_ERROR)
            return False
        if exp.isnull().values.any():
            wx.MessageBox('There are empty holes in exp.', 'Error', wx.OK | wx.ICON_ERROR)
            return False
        return True



    def OnSaveNew(self, event):
        model = self.tab2.grid.GetAllValues()
        prior = self.tab1.grid.GetAllValues()
        exp = self.tab3.grid.GetAllValues()

        if not model:
            wx.MessageBox('Model value cannot be empty', 'Error', wx.OK | wx.ICON_ERROR)
            return False
        if not prior:
            wx.MessageBox('Prior value cannot be empty', 'Error', wx.OK | wx.ICON_ERROR)
            return False
        if not exp:
            wx.MessageBox('Exp value cannot be empty', 'Error', wx.OK | wx.ICON_ERROR)
            return False

        model_headers = model.pop(0)
        model = pd.DataFrame(model, columns=model_headers)
        prior_headers = prior.pop(0)
        prior = pd.DataFrame(prior, columns=prior_headers)
        exp_headers = exp.pop(0)
        exp = pd.DataFrame(exp, columns=exp_headers)

        if not self._CheckData(prior_headers, prior, model_headers, model, exp_headers, exp):
            return False

        """
        Create and show the Open FileDialog
        """
        wildcard = "Python source (*.pkl)|*.pkl|" \
           "All files (*.*)|*.*"
        dlg = wx.FileDialog(self, message="Save project as ...", defaultFile="", style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT)
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
            if res == wx.ID_CANCEL:
                #frame.Destroy()
                return False
            else:
                frame.AdditionalData(args)
                frame.Destroy()
                            
            Training(args)

            with open(outFile[0], 'rb') as buff:
                data = pickle.load(buff)

            self.opened_data = data
            self.opened_filename = outFile[0]
        

    def OnSave(self, event):
        
        if self.opened_filename is None:
            wx.MessageBox('No file is associated with this data. You must save and train the emulator first', 'Error', wx.OK | wx.ICON_ERROR)
            return 

        """
        Every time the model data is changed, it needs to be trained again
        It cannot be saved directly
        Warn the user and ask them if they want to proceed
        """
        if self.tab2.grid.stockUndo:
            dlg = wx.MessageDialog(None, "Model data may have changed. Any change here will be discarded. You need to re-train the emulator. Do you want to continue saving?", "", wx.YES_NO | wx.ICON_QUESTION) 
            result = dlg.ShowModal()

            if result != wx.ID_YES:
                return 

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

        with open(self.opened_filename, 'wb') as buff:
            pickle.dump(self.opened_data, buff)

        return True


    def OnFile(self, event):
        """
        Create and show the Open FileDialog
        """
        dlg = wx.FileDialog(
            self, message="Choose a file",
            defaultFile="",
            style=wx.FD_OPEN | wx.FD_MULTIPLE | wx.FD_CHANGE_DIR
            )
        result = dlg.ShowModal() 
        path = dlg.GetPaths()
        dlg.Destroy()

        if result != wx.ID_OK:
            return False

        with open(path[0], 'rb') as buff:
            data = pickle.load(buff)

        self.opened_filename = path[0]
        self.opened_data = data
        """
        Loading prior
        """
        prior = data['data'].prior.T
        prior = [prior.columns.tolist()] + prior.values.tolist()
        if type(prior[0]) is not list:
            prior = [prior]
        self.tab1.grid.ClearAll()
        self.tab1.grid.SetValue([[0,0], [len(prior) - 1, len(prior[0]) - 1]], prior)
        del self.tab1.grid.stockUndo[:]
        self.tab1.toolbar.EnableTool(ID_UNDO, False)

        """
        Loading model data
        """
        training_data = data['data']
        header = [training_data.par_name + training_data.var_name + [name + "_Error" for name in training_data.var_name]] 
        content = np.concatenate((training_data.sim_para, training_data.sim_data, training_data.sim_error), axis=1).tolist()
        if type(content[0]) is not list:
            content = [content]
        self.tab2.grid.ClearAll()
        self.tab2.grid.SetValue([[0,0], [0, len(header[0]) - 1]], header)
        self.tab2.grid.SetValue([[1,0], [len(content), len(content[0]) - 1]], content)
        del self.tab2.grid.stockUndo[:]
        self.tab2.toolbar.EnableTool(ID_UNDO, False)

        """
        Loading exp data
        """
        header = [training_data.var_name + [name + "_Error" for name in training_data.var_name]] 
        content = np.concatenate((training_data.exp_result, np.sqrt(np.diag(training_data.exp_cov)))).tolist()
        if type(content[0]) is not list:
            content = [content]
        self.tab3.grid.ClearAll()
        self.tab3.grid.SetValue([[0,0], [0, len(header[0]) - 1]], header)
        self.tab3.grid.SetValue([[1,0], [len(content), len(content[0]) - 1]], content)
        del self.tab3.grid.stockUndo[:]
        self.tab3.toolbar.EnableTool(ID_UNDO, False)


class Common(wx.Frame):
 
    def __init__(self, parent):
        wx.Frame.__init__(self, parent, wx.NewId(), "Bayesian analysis", size=(1000,400))
        
        panel = wx.Panel(self)
         
        notebook = wx.Notebook(panel)

        GridP1 = GridPanel(notebook, size=(6, 100))
        GridP1.grid.SetRowLabelValue(0, "Name")
        GridP1.grid.SetRowLabelValue(1, "Type")
        GridP1.grid.SetRowLabelValue(2, "Lower bound")
        GridP1.grid.SetRowLabelValue(3, "Upper bound")
        GridP1.grid.SetRowLabelValue(4, "Centre")
        GridP1.grid.SetRowLabelValue(5, "Standard Deviation")
        GridP1.grid.SetRowLabelSize(wx.grid.GRID_AUTOSIZE)
        notebook.AddPage(GridP1, "Para prior")

        GridP2 = GridPanel(notebook)
        notebook.AddPage(GridP2, "Model result")

        GridP3 = GridPanel(notebook, size=(2, 100))
        GridP3.grid.SetRowLabelValue(0, "Name")
        GridP3.grid.SetRowLabelValue(1, "Value")
        notebook.AddPage(GridP3, "Exp data data")

        sizer = wx.BoxSizer(wx.VERTICAL)
                
        
    # toolbar
        self.menubar = CommonMenuBar(panel, tab1=GridP1, tab2=GridP2, tab3=GridP3, id=100, style=wx.TB_HORIZONTAL | wx.NO_BORDER |
                                        wx.TB_FLAT | wx.TB_TEXT)

        self.SetMenuBar(self.menubar)
        
        sizer.Add(notebook, 1, wx.EXPAND | wx.EXPAND, 5)

        self.Bind(wx.EVT_CLOSE, self.OnClose)
        
        panel.SetSizer(sizer)
        self.Layout()
        self.Show()

    def OnClose(self, event):
        self.Destroy()

    



app = wx.App(0)
frame = Common(None)
frame.Show()
app.MainLoop()
