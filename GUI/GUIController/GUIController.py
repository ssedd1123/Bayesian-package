import os
import sys

import numpy as np
import pandas as pd
import wx
import wx.grid as gridlib
import wx.lib.inspection
from matplotlib.figure import Figure
from mpi4py import MPI
from pubsub import pub

import Utilities.GradientDescent as gd
from GUI.GridController.GridController import (GridController, PriorController,
                                               SplitViewController)
from GUI.FileController.FileController import FileController
from GUI.MatplotlibFrame import MatplotlibFrame
from Utilities.MasterSlave import MasterSlave, ThreadsException
from Utilities.Utilities import GetTrainedEmulator


class GUIController:
    def __init__(self, parent, workenv, app):
        self.workenv = workenv
        self.view = GUIViewer(parent, app)
        self.prior_model = self.view.prior_controller.model
        self.prior_view = self.view.prior_controller.view

        self.model_obs_model = self.view.model_input_controller.right_model
        self.model_obs_view = self.view.model_input_controller.right_view
        self.model_par_model = self.view.model_input_controller.left_model
        self.model_par_view = self.view.model_input_controller.left_view

        self.exp_view = self.view.exp_controller.view
        self.exp_model = self.view.exp_controller.model

        self.emulator_input_model = self.view.manual_emulation_controller.left_model
        self.emulator_input_view = self.view.manual_emulation_controller.left_view
        self.emulator_output_model = self.view.manual_emulation_controller.right_model
        self.emulator_output_view = self.view.manual_emulation_controller.right_view

        self.file_view = self.view.file_controller.file_view
        self.display_view = self.view.file_controller.display_view
        self.file_model = self.view.file_controller.model

        self.correlation_frame = None

        pub.subscribe(self._SyncHeaders, "Data_Changed")
        pub.subscribe(self.CheckObj, "MenuBar_Check", func=self.EmulatorCheck)
        pub.subscribe(self.CheckObj, "MenuBar_Open", func=self.OpenFile)
        pub.subscribe(self.CheckObj, "MenuBar_SaveNew", func=self.SaveNew)
        pub.subscribe(self.CheckObj, "MenuBar_Save", func=self.Save)
        pub.subscribe(self.CheckObj, "MenuBar_GenHyperCube", func=self.GenHyperCube)
        pub.subscribe(self.CheckObj, "MenuBar_Emulate", func=self.Emulate)
        pub.subscribe(self.CheckObj, "MenuBar_ChainedEmulate", func=self.ChainedEmulate)
        pub.subscribe(self.CheckObj, "MenuBar_EvalEmu", func=self.EvalEmu)
        pub.subscribe(self.CheckObj, "MenuBar_Report", func=self.TrainReport)
        pub.subscribe(self.CheckObj, "MenuBar_Correlation", func=self.Correlation)
        pub.subscribe(self.CheckObj, "MenuBar_Posterior", func=self.Posterior)

        # sync file according the file_controller
        pub.subscribe(self.LoadFile, 'emulatorFileChanged')
        pub.subscribe(self.LoadFile, 'listFileChanged')
        pub.subscribe(self.LoadFile, 'listFileChanged')

    def CheckObj(self, func, obj, evt):
        orig_obj = obj
        while True:
            if obj is self.view:
                func(obj, evt)
                break
            obj = obj.GetParent()
            if obj is None:
                break

    def GenHyperCube(self, obj, evt):
        prior = self.prior_model.GetData(drop_index=False)
        if prior.empty:
            wx.MessageDialog(
                self.view,
                "You need to fill up your parameter range",
                "Warning",
                wx.OK | wx.ICON_WARNING,
            ).ShowModal()
            return None
        ranges = prior.loc[["Min", "Max"], :].astype(float).values.T
        dlg = wx.TextEntryDialog(self.view, "Number of points needed")
        dlg.ShowModal()
        result = dlg.GetValue()
        dlg.Destroy()
        try:
            result = int(result)
        except:
            wx.MessageDialog(
                self.view, "Integers needed", "Warning", wx.OK | wx.ICON_WARNING
            ).ShowModal()
            return None
        from Utilities.LatinHyperCube import GenerateLatinHyperCube

        content = GenerateLatinHyperCube(result, ranges)
        rows = np.arange(1, 1 + content.shape[0])
        cols = np.arange(0, content.shape[1])
        self.model_par_model.SetValue(rows, cols, content)
        self.model_par_view.ForceRefresh()

    def TrainReport(self, obj, evt):
        if self.file_model.emulator_filename is not None:
            fig = Figure((15, 12), 75)
            frame = MatplotlibFrame(None, fig)
            from GUI.TrainingProgressFrame import TrainingProgressFrame

            gauge = TrainingProgressFrame(
                1,
                None,
                -1,
                "Generating report",
                size=(300, -1),
                text_label="Training report generation in progress",
                col_labels=[""],
            )
            PtsFraction = 0.1
            gaugeUpdatePts = lambda progress: gauge.updateProgress(
                progress * PtsFraction * 100
            )  # first half of calculation contributes ~ 10 percent
            gaugeUpdateSteps = lambda progress: gauge.updateProgress(
                progress * (1 - PtsFraction) * 100 + PtsFraction * 100
            )  # second half of the calculation
            pub.subscribe(gaugeUpdatePts, "NumberOfPtsProgress")
            pub.subscribe(gaugeUpdateSteps, "NumberOfStepsProgress")

            from TrainEmulator import TrainingCurve

            gauge.Show()
            TrainingCurve(fig, config_file=self.file_model.emulator_filename)
            pub.unsubscribe(gaugeUpdatePts, "NumberOfPtsProgress")
            pub.unsubscribe(gaugeUpdateSteps, "NumberOfStepsProgress")
            gauge.Destroy()
            frame.SetData()
            frame.Show()

    def ChainedEmulate(self, obj, evt):
        self.Emulate(obj, evt, False)

    def Emulate(self, obj, evt, single_file=True):
        if self.file_model.trace_filename is None:
            wx.MessageBox('No trace file is selected. Cannot start analysis', 'Error', wx.OK | wx.ICON_ERROR)
            return
        if self.file_model.trace_filename != self.file_model.emulator_filename:
            wx.MessageBox('For single file analysis, please make sure trace file and emulator file is the same', 
                          'Error', wx.OK | wx.ICON_ERROR)
            return
        if single_file:
            filenames = self.file_model.trace_filename
        else:
            filenames = self.file_model.get_list_filenames_with_trace_first()

        if filenames is not None:
            from GUI.SelectEmulationOption import SelectEmulationOption

            EmuOption = SelectEmulationOption(self.view)
            res = EmuOption.ShowModal()
            if res == wx.ID_OK:
                options = EmuOption.GetValue()
                nevent = options["nevent"]
                from GUI.Model import CalculationFrame

                frame = CalculationFrame(None, -1, "Progress", self.workenv, nevent)
                frame.Show()
                
                try:
                    frame.OnCalculate(
                        {
                            "config_file": filenames,
                            "nsteps": nevent,
                            "clear_trace": options["clear_trace"],
                            "burnin": options["burnin"],
                        }
                    )
                    self.Correlation(None, None, False)
                except ThreadsException as ex:
                    wx.MessageBox(str(ex), 'Error', wx.OK | wx.ICON_ERROR)

    def EvalEmu(self, obj, evt):
        data = self.emulator_input_model.GetData()
        if data.shape[0] > 0:  # emulator_input contains more than the header
            np_data = data.astype(float).to_numpy()
            if self.file_model.emulator_filename is not None:
                clf = GetTrainedEmulator(self.file_model.emulator_filename)[0]
                for idx, row in enumerate(np_data):
                    if not np.isnan(row).any():
                        try:
                            prediction, cov = clf.Predict(row)
                        except Exception:
                            continue
                        prediction = np.squeeze(prediction)
                        cov = np.squeeze(cov)
                        self.emulator_output_model.ChangeValues(
                            idx + 1,
                            np.arange(prediction.shape[0]),
                            prediction,
                            send_changed=False,
                        )  # y-index add one to not overwrite header
                        self.emulator_output_model.ChangeValues(
                            idx + 1,
                            np.arange(prediction.shape[0], 2 * prediction.shape[0]),
                            np.sqrt(np.diag(cov)),
                            send_changed=False,
                        )
                        self.view.Refresh()

    def Correlation(self, obj, evt, ask_options=True):
        if self.file_model.trace_filename is not None:
            kwargs = {"overlay_pt": False}
            if ask_options:
                from GUI.SelectPosteriorOption import SelectPosteriorOption

                options = SelectPosteriorOption(self.view)
                res = options.ShowModal()
                if res == wx.ID_OK:
                    kwargs = options.GetValue()
                else:
                    # don't show correlation if the user close the option dialog
                    return
            if not self.correlation_frame:
                fig = Figure((15, 12), 75)
                self.correlation_frame = MatplotlibFrame(None, fig)
            self.correlation_frame.fig.clf()
            from Utilities.Utilities import PlotTrace

            if kwargs["overlay_pt"]:
                kwargs["mark_point"] = self.emulator_input_model.GetData()
            del kwargs["overlay_pt"]

            PlotTrace(self.file_model.trace_filename, self.correlation_frame.fig, **kwargs)
            self.correlation_frame.SetData()
            self.correlation_frame.Show()

    def Posterior(self, obj, evt):
        if self.file_model.emulator_filename is not None:
            if not self.correlation_frame:
                fig = Figure((15, 12), 75)
                self.correlation_frame = MatplotlibFrame(None, fig)
            self.correlation_frame.fig.clf()
            from GUI.TrainingProgressFrame import TrainingProgressFrame
            from PlotPosterior import PlotOutput

            gauge = TrainingProgressFrame(
                1,
                None,
                -1,
                "Generating output posterior",
                size=(300, -1),
                text_label="Output posterior generation in progress",
                col_labels=[""],
            )
            gaugeProgress = lambda progress: gauge.updateProgress(progress * 100)
            pub.subscribe(gaugeProgress, "PosteriorOutputProgress")
            gauge.Show()
            PlotOutput(self.file_model.emulator_filename, self.correlation_frame.fig, trace_filename=self.file_model.trace_filename)
            pub.unsubscribe(gaugeProgress, "PosteriorOutputProgress")
            gauge.Destroy()
            self.correlation_frame.SetData()
            self.correlation_frame.Show()

    def Save(self, obj, evt):
        prior = self.prior_model.GetData(drop_index=False)
        model_X = self.model_par_model.GetData().astype("float")
        model_Y = self.model_obs_model.GetData().astype("float")
        exp = self.exp_model.GetData(drop_index=False).astype("float")

        store = pd.HDFStore(self.file_model.emulator_filename, "a")
        if model_X.equals(store["Model_X"]) and model_Y.equals(store["Model_Y"]):
            from ChangeFileContent import ChangeFileContent

            ChangeFileContent(store, prior, exp)
        else:
            wx.MessageBox(
                "Model values has changed. You must train the emulator again",
                "Error",
                wx.OK | wx.ICON_ERROR,
            )
        store.close()

    def SaveNew(self, obj, evt):
        """
        Create and show the Open FileDialog
        """
        wildcard = "Python source (*.h5)|*.h5|" "All files (*.*)|*.*"
        dlg = wx.FileDialog(
            obj,
            message="Save project as ...",
            defaultFile="",
            style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT,
        )
        result = dlg.ShowModal()
        outFile = dlg.GetPaths()
        dlg.Destroy()

        if (
            result == wx.ID_CANCEL
        ):  # Either the cancel button was pressed or the window was closed
            return False

        from GUI.SelectTrainingOption import SelectOption

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

        from GUI.TrainingProgressFrame import TrainingProgressFrame

        if args["principalcomp"] is not None:
            gauge = TrainingProgressFrame(
                args["principalcomp"], None, -1, "Training progress", size=(300, -1)
            )
        else:
            gauge = TrainingProgressFrame(
                1, None, -1, "Training progress", size=(300, -1)
            )

        gauge.Show()
        gaugeUpdate = lambda step, progress, mag, nuggets, scales: [
            gd.DefaultOutput(step, progress, mag, nuggets, scales),
            gauge.updateProgress(progress),
        ]
        pub.subscribe(gaugeUpdate, "GradientProgress")
        from TrainEmulator import Training

        Training(prior, model_X, model_Y, exp, outFile[0], abs_output=True, **args)
        # pub.unsubscribe(gaugeUpdate, 'GradientProgress')
        pub.unsubscribe(gaugeUpdate, "GradientProgress")
        gauge.Destroy()

        if self.file_model.emulator_filename is not None:
            self.view.file_controller.remove_file(self.file_model.emulator_filename)
        self.view.file_controller.add_file(outFile[0])
        self.LoadFile()

    def OpenFile(self, obj, evt):
        #dlg = wx.FileDialog(
        #    obj,
        #    message="Choose a file",
        #    defaultFile="",
        #    style=wx.FD_OPEN | wx.FD_MULTIPLE,
        #)
        #result = dlg.ShowModal()
        #path = dlg.GetPaths()
        #dlg.Destroy()

        #if result != wx.ID_OK:
        #    return False
        #self.view.file_controller.add_file()
        self.view.file_controller.add_file()
        self.LoadFile()#path[0])

    def LoadFile(self, filename=None):
        # if filename is given, it will update the gui display
        # otherwise it will just sync file_model and model_par
        if filename is not None:
            self.view.file_controller.add_file(filename)

        self.view.prior_controller.ClearAll()
        self.view.model_input_controller.controller_left.ClearAll()
        self.view.model_input_controller.controller_right.ClearAll()
        self.view.exp_controller.ClearAll()
        
        if self.file_model.emulator_filename is not None:
            store = pd.HDFStore(self.file_model.emulator_filename, "r")
            self.prior_model.SetData(store["PriorAndConfig"].T)
            self.model_par_model.SetData(store["Model_X"])
            self.model_obs_model.SetData(store["Model_Y"])
            self.exp_model.SetData(pd.concat([store["Exp_Y"], store["Exp_YErr"]], axis=1).T)
            store.close()

        self.prior_model.ResetUndo()
        self.model_par_model.ResetUndo()
        self.model_obs_model.ResetUndo()
        self.exp_model.ResetUndo()


    def EmulatorCheck(self, obj, evt):
        if self.file_model.emulator_filename is not None:
            from GUI.EmulatorController.EmulatorViewer import \
                EmulatorController

            controller = EmulatorController(self.file_model.emulator_filename)
            controller.viewer.Show()

    def _SyncHeaders(self, obj, evt):
        rows = evt[0]
        cols = evt[1]
        if 0 in rows:
            self._SyncHeaders2Ways(obj, self.prior_model, self.model_par_model)
            self._SyncHeaders2Ways(
                self.prior_model, self.prior_model, self.emulator_input_model
            )
            self._SyncHeaders2Ways(obj, self.exp_model, self.model_obs_model)
            # self._SyncHeaders2Ways(self.exp_model, self.exp_model, self.emulator_output_model)
            value = (
                self.exp_model.data.iloc[0]
                .replace(r"^\s*$", np.nan, regex=True)
                .dropna(how="all")
            )
            value = np.append(
                value,
                ["%s_Err" % val for val in value]
                + [None for i in range(self.exp_model.data.shape[1])],
            )
            self.emulator_output_model.ChangeValues(
                0, np.arange(value.shape[0]), value, send_changed=False
            )  # Error is added to the header of emulator output
            self.view.Refresh()

    def _SyncHeaders2Ways(self, obj, model1, model2):
        # value = obj.data.iloc[0].replace(r'^\s*$', np.nan, regex=True).dropna(how='all')
        value = obj.data.iloc[0].replace(r"^\s*$", np.nan, regex=True)
        if obj is model1:
            model2.ChangeValues(0, np.arange(value.shape[0]), value, send_changed=False)
        elif obj is model2:
            model1.ChangeValues(0, np.arange(value.shape[0]), value, send_changed=False)


class GUIViewer(wx.Frame):
    def __init__(self, parent, app):
        wx.Frame.__init__(
            self, parent, wx.NewId(), "Common", size=wx.ScreenDC().GetPPI().Scale(13, 8)
        )  # 1000,400))
        self.app = app

        panel = wx.Panel(self)
        split_panel = wx.SplitterWindow(panel)
        notebook = wx.Notebook(split_panel)
        from GUI.GUIController.GUIMenu import GUIMenuBar

        self.menubar = GUIMenuBar(self)
        self.SetMenuBar(self.menubar)

        prior_panel = wx.Panel(notebook)
        self.prior_controller = PriorController(prior_panel, 100)
        prior_sizer = wx.BoxSizer(wx.VERTICAL)
        prior_sizer.Add(self.prior_controller.toolbar, 0, wx.EXPAND)
        prior_sizer.Add(self.prior_controller.view, 1, wx.EXPAND)
        prior_panel.SetSizer(prior_sizer)
        notebook.AddPage(prior_panel, "Parameters prior")

        grid_panel = wx.Panel(notebook)
        self.model_input_controller = SplitViewController(grid_panel)
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.model_input_controller.view, 1, wx.EXPAND)
        #grid_panel.SetSizer(sizer)
        notebook.AddPage(grid_panel, "Model calculations")

        #exp_panel = wx.Panel(notebook)
        self.exp_controller = GridController(grid_panel, 3, 100, size=(-1, 120))
        self.exp_controller.model.data.index = ["Name", "Values", "Errors"]
        #exp_sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.exp_controller.toolbar, 0.1, wx.EXPAND)
        sizer.Add(self.exp_controller.view, 0, wx.EXPAND)
        grid_panel.SetSizer(sizer)

        #exp_panel.SetSizer(exp_sizer)
        #notebook.AddPage(exp_panel, "Experimental data")

        manual_emulation_panel = wx.Panel(notebook)
        self.manual_emulation_controller = SplitViewController(
            manual_emulation_panel, 300, 100
        )
        self.manual_emulation_controller.right_view.SetDefaultCellBackgroundColour(
            "Grey"
        )
        # disable edition in all cells in this panel
        # this panel is only meant to output data, not for editing
        self.manual_emulation_controller.right_view.EnableEditing(False)
        attr = gridlib.GridCellAttr()
        # first row is reserved for header
        attr.SetReadOnly(True)
        attr.SetBackgroundColour("Grey")
        self.manual_emulation_controller.left_view.SetRowAttr(0, attr)

        EvalEmuButton = wx.Button(manual_emulation_panel, -1, "Evaluate emulator")
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(EvalEmuButton)
        EvalEmuButton.Bind(
            wx.EVT_BUTTON,
            lambda evt: pub.sendMessage("MenuBar_EvalEmu", obj=self, evt=evt),
        )
        sizer.Add(self.manual_emulation_controller.view, 1, wx.EXPAND)
        manual_emulation_panel.SetSizer(sizer)
        notebook.AddPage(manual_emulation_panel, "Ask emulator")


        sizer = wx.BoxSizer(wx.VERTICAL)

        self.file_controller = FileController(file_viewer_kwargs={'parent': split_panel}, display_kwargs={'parent': panel, 'size': (-1,  wx.ScreenDC().GetPPI()[0]/3)})
        split_panel.SplitVertically(notebook, self.file_controller.file_view, wx.ScreenDC().GetPPI()[0] * 30)
        #hsizer = wx.BoxSizer(wx.HORIZONTAL)
        #hsizer.Add(notebook, 1, wx.EXPAND | wx.EXPAND, 5)
        #hsizer.Add(self.file_controller.file_view, 0.2, wx.EXPAND, 0)
        print(wx.ScreenDC().GetPPI()[0], flush=True)

        sizer.Add(split_panel, 1, wx.EXPAND, 0)
        sizer.Add(self.file_controller.display_view, 0., wx.EXPAND)

        self.Bind(wx.EVT_CLOSE, self.OnClose)
        panel.SetSizerAndFit(sizer)
        self.Layout()
        self.Show()

    def OnClose(self, evt):
        self.Destroy()
        self.app.ExitMainLoop()


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    status = MPI.Status()
    root = 0

    work_environment = MasterSlave(comm)

    # gd.UseDefaultOutput()
    app = wx.App(0)
    controller = GUIController(None, app=app, workenv=work_environment)
    controller.view.Show()

    if len(sys.argv) == 2:
        controller.LoadFile(sys.argv[1])

    app.MainLoop()
    work_environment.Close()


if __name__ == "__main__":
    main()
