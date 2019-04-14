import sys
import pandas as pd
import numpy as np
import wx
from pubsub import pub
from mpi4py import MPI

from Utilities.Utilities import PlotTrace
from GUI.GridController.GridController import GridController, PriorController
from GUI.GUIController.GUIMenu import GUIMenuBar
from GUI.EmulatorController.EmulatorViewer import EmulatorController
from TrainEmulator import Training, TrainingCurve
from ChangeFileContent import ChangeFileContent
from GUI.SelectTrainingOption import SelectOption
from matplotlib.figure import Figure
from GUI.PlotFrame import PlotFrame
from Utilities.MasterSlave import MasterSlave
from GUI.Model import CalculationFrame
from GUI.MatplotlibFrame import MatplotlibFrame
from GUI.SelectEmulationOption import SelectEmulationOption
import Utilities.GradientDescent as gd
from PlotPosterior import PlotOutput

class GUIController:

    def __init__(self, parent, workenv, app):
        self.workenv = workenv
        self.view = GUIViewer(parent, app)
        self.prior_model = self.view.prior_controller.model
        self.model_obs_model = self.view.model_obs_controller.model
        self.model_obs_view = self.view.model_obs_controller.view
        self.model_par_view = self.view.model_par_controller.view
        self.model_par_model = self.view.model_par_controller.model
        self.exp_model = self.view.exp_controller.model

        self.filename = None
        self.correlation_frame = None
        
        self.model_par_view.Bind(wx.EVT_SCROLLWIN, self.onScrollWin1)
        self.model_obs_view.Bind(wx.EVT_SCROLLWIN, self.onScrollWin2)
        self.model_obs_view.SetRowLabelSize(5)

        pub.subscribe(self._SyncHeaders, 'Data_Changed')
        pub.subscribe(self.CheckObj, 'MenuBar_Check', func=self.EmulatorCheck)
        pub.subscribe(self.CheckObj, 'MenuBar_Open', func=self.OpenFile)
        pub.subscribe(self.CheckObj, 'MenuBar_SaveNew', func=self.SaveNew)
        pub.subscribe(self.CheckObj, 'MenuBar_Save', func=self.Save)
        pub.subscribe(self.CheckObj, 'MenuBar_Emulate', func=self.Emulate)
        pub.subscribe(self.CheckObj, 'MenuBar_Report', func=self.TrainReport)
        pub.subscribe(self.CheckObj, 'MenuBar_Correlation', func=self.Correlation)
        pub.subscribe(self.CheckObj, 'MenuBar_Posterior', func=self.Posterior)

    def CheckObj(self, func, obj, evt):
        orig_obj = obj
        while True:
            if obj is self.view:
                func(obj, evt)
                break
            obj = obj.GetParent()
            if obj is None:
                break

    def TrainReport(self, obj, evt):
        if self.filename is not None:
            fig = Figure((15,12), 75)
            frame = MatplotlibFrame(None, fig)
            TrainingCurve(fig, config_file=self.filename)
            frame.SetData()
            frame.Show()
            

    def Emulate(self, obj, evt):
        if self.filename is not None:
            EmuOption = SelectEmulationOption(self.view)
            res = EmuOption.ShowModal()
            if res == wx.ID_OK:
                options = EmuOption.GetValue()
                nevent = options['nevent']
                frame = CalculationFrame(None, -1, 'Progress', self.workenv, nevent)
                frame.Show()
                frame.OnCalculate({'config_file': self.filename, 'nsteps': nevent, 'clear_trace': options['clear_trace']})
                self.Correlation(None, None)
        
    def Correlation(self, obj, evt):
        if self.filename is not None:
            if not self.correlation_frame:
                fig = Figure((15,12), 75)
                self.correlation_frame = MatplotlibFrame(None, fig)
            self.correlation_frame.fig.clf()
            PlotTrace(self.filename, self.correlation_frame.fig)
            self.correlation_frame.SetData()
            self.correlation_frame.Show()

    def Posterior(self, obj, evt):
        if self.filename is not None:
            if not self.correlation_frame:
                fig = Figure((15,12), 75)
                self.correlation_frame = MatplotlibFrame(None, fig)
            self.correlation_frame.fig.clf()
            PlotOutput(self.filename, self.correlation_frame.fig)
            self.correlation_frame.SetData()
            self.correlation_frame.Show()


    def Save(self, obj, evt):
        prior = self.prior_model.GetData(drop_index=False)
        model_X = self.model_par_model.GetData().astype('float')
        model_Y = self.model_obs_model.GetData().astype('float')
        exp = self.exp_model.GetData(drop_index=False).astype('float')

        store = pd.HDFStore(self.filename, 'a')
        if model_X.equals(store['Model_X']) and model_Y.equals(store['Model_Y']):
            ChangeFileContent(store, prior, exp)
        else:
            wx.MessageBox('Model values has changed. You must train the emulator again', 'Error', wx.OK | wx.ICON_ERROR)
        store.close()

    def SaveNew(self, obj, evt):
        """
        Create and show the Open FileDialog
        """
        wildcard = "Python source (*.h5)|*.h5|" \
           "All files (*.*)|*.*"
        dlg = wx.FileDialog(obj, message="Save project as ...", defaultFile="", style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT)
        result = dlg.ShowModal()            
        outFile = dlg.GetPaths()
        dlg.Destroy()
    
        if result == wx.ID_CANCEL:    #Either the cancel button was pressed or the window was closed
            return False

        frame = SelectOption()
        res = frame.ShowModal()
        if res == wx.ID_CANCEL:
            return False
        else:
            args = frame.AdditionalData()
            frame.Destroy()

        prior = self.prior_model.GetData(drop_index=False)
        model_X = self.model_par_model.GetData()
        model_Y = self.model_obs_model.GetData()
        exp = self.exp_model.GetData(drop_index=False)

        Training(prior, model_X, model_Y, exp, outFile[0], abs_output=True, **args)
        self.filename = outFile[0]
    
    def OpenFile(self, obj, evt):
        dlg = wx.FileDialog(
            obj, message="Choose a file",
            defaultFile="",
            style=wx.FD_OPEN | wx.FD_MULTIPLE | wx.FD_CHANGE_DIR
            )
        result = dlg.ShowModal() 
        path = dlg.GetPaths()
        dlg.Destroy()

        if result != wx.ID_OK:
            return False
        self.LoadFile(path[0])

    def LoadFile(self, filename):
        self.filename = filename
        store = pd.HDFStore(self.filename, 'r')

        self.prior_model.SetData(store['PriorAndConfig'].T)
        self.model_par_model.SetData(store['Model_X'])
        self.model_obs_model.SetData(store['Model_Y'])
        self.exp_model.SetData(pd.concat([store['Exp_Y'], store['Exp_YErr']], axis=1).T)

        self.prior_model.ResetUndo()
        self.model_par_model.ResetUndo()
        self.model_obs_model.ResetUndo()
        self.exp_model.ResetUndo()
        store.close()

    def EmulatorCheck(self, obj, evt):
        if self.filename is not None:
            controller = EmulatorController(self.filename)
            controller.viewer.Show()
 

    def onScrollWin1(self, evt):
        if evt.Orientation == wx.SB_VERTICAL:
            self.model_obs_view.Scroll(-1, evt.Position)
        evt.Skip()

    def onScrollWin2(self, evt):
        if evt.Orientation == wx.SB_VERTICAL:
            self.model_par_view.Scroll(-1, evt.Position)
        evt.Skip()

    def _SyncHeaders(self, obj, evt):
        rows = evt[0]
        cols = evt[1]
        if 0 in rows:
            self._SyncHeaders2Ways(obj, self.prior_model, self.model_par_model)
            self._SyncHeaders2Ways(obj, self.exp_model, self.model_obs_model)
            self.view.Refresh()

    def _SyncHeaders2Ways(self, obj, model1, model2):
        value = obj.data.iloc[0].replace(r'^\s*$', np.nan, regex=True).dropna(how='all')
        if obj is model1:
            model2.ChangeValues(0, np.arange(value.shape[0]), value, send_changed=False)
        elif obj is model2:
            model1.ChangeValues(0, np.arange(value.shape[0]), value, send_changed=False)


class GUIViewer(wx.Frame):
 
    def __init__(self, parent, app):
        wx.Frame.__init__(self, parent, wx.NewId(), "Common", size=(1000,400))
        self.app=app

        panel = wx.Panel(self)
        notebook = wx.Notebook(panel)
        self.menubar = GUIMenuBar(self)
        self.SetMenuBar(self.menubar)

        prior_panel = wx.Panel(notebook)
        self.prior_controller = PriorController(prior_panel, 100)
        prior_sizer = wx.BoxSizer(wx.VERTICAL)
        prior_sizer.Add(self.prior_controller.toolbar)
        prior_sizer.Add(self.prior_controller.view, 1, wx.EXPAND)
        prior_panel.SetSizer(prior_sizer)
        notebook.AddPage(prior_panel, "Parameters prior")

        grid_panel = wx.Panel(notebook)
        splitterLR = wx.SplitterWindow(grid_panel)
       
        left_panel = wx.Panel(splitterLR)
        right_panel = wx.Panel(splitterLR)
        self.model_obs_controller = GridController(right_panel, 100, 100)
        grid_sizer = wx.BoxSizer(wx.VERTICAL)
        grid_sizer.Add(self.model_obs_controller.toolbar)
        grid_sizer.Add(self.model_obs_controller.view, 1, wx.EXPAND)
        right_panel.SetSizer(grid_sizer)

        self.model_par_controller = GridController(left_panel, 100, 100)
        grid_sizer = wx.BoxSizer(wx.VERTICAL)
        grid_sizer.Add(self.model_par_controller.toolbar)
        grid_sizer.Add(self.model_par_controller.view, 1, wx.EXPAND)
        left_panel.SetSizer(grid_sizer)

        splitterLR.SplitVertically(left_panel, right_panel, 300)
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(splitterLR, 1, wx.EXPAND)
        grid_panel.SetSizer(sizer)
        notebook.AddPage(grid_panel, "Model calculations")

        exp_panel = wx.Panel(notebook)
        self.exp_controller = GridController(exp_panel, 3, 100)
        self.exp_controller.model.data.index = ['Name', 'Values', 'Errors']
        exp_sizer = wx.BoxSizer(wx.VERTICAL)
        exp_sizer.Add(self.exp_controller.toolbar)
        exp_sizer.Add(self.exp_controller.view, 1, wx.EXPAND)
        exp_panel.SetSizer(exp_sizer)
        notebook.AddPage(exp_panel, "Experimental data")
       
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(notebook, 1, wx.EXPAND | wx.EXPAND, 5)
       
        self.Bind(wx.EVT_CLOSE, self.OnClose)
        panel.SetSizer(sizer)
        self.Layout()
        self.Show()

    def OnClose(self, evt):
        self.Destroy()
        self.app.ExitMainLoop()


if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    status = MPI.Status()
    root = 0


    work_environment = MasterSlave(comm)

    gd.UseDefaultOutput()
    app = wx.App(0)
    controller = GUIController(None, app=app, workenv=work_environment)
    controller.view.Show()

    if len(sys.argv) == 2:
        controller.LoadFile(sys.argv[1])

    app.MainLoop()
    work_environment.Close()
