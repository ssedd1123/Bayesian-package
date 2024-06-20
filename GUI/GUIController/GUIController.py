import os
import sys
import traceback

import shutil
import numpy as np
import pandas as pd
import wx
import wx.grid as gridlib
import wx.lib.inspection
from matplotlib.figure import Figure
from pubsub import pub
from tables import exceptions
from contextlib import contextmanager
import json

import Utilities.GradientDescent as gd
from GUI.GridController.GridController import (GridController, PriorController,
                                               SplitViewController)
from GUI.GUIController.ClosureTestController import ClosureTestController
from GUI.GUIController.EvalEmuController import EvalEmuController

from GUI.FileController.FileController_new import FileController
from GUI.FlexMessageBox import FlexMessageBox
from GUI.MatplotlibFrame import MatplotlibFrame
from GUI.TrainingProgressFrame import TrainingProgressFrame
from Utilities.MasterSlaveMP import ThreadsException


class GUIController:
    def __init__(self, parent, workenv, app, config_file='GUI/config.json'):
        # load config file
        with open(config_file, 'r') as config:
            self.config_data = json.load(config)

        self.workenv = workenv
        self.view = GUIViewer(parent, app, self.config_data)
        self.prior_model = self.view.prior_controller.model
        self.prior_view = self.view.prior_controller.view

        self.model_obs_model = self.view.model_input_controller.right_model
        self.model_obs_view = self.view.model_input_controller.right_view
        self.model_par_model = self.view.model_input_controller.left_model
        self.model_par_view = self.view.model_input_controller.left_view

        self.exp_view = self.view.exp_controller.view
        self.exp_model = self.view.exp_controller.model

        self.plugin_controllers = {}
        for key, controller in self.view.plugins.items():
            controller.SetHeadController(self)
            self.plugin_controllers[key] = controller

        self.file_view = self.view.file_controller.file_view
        self.display_view = self.view.file_controller.display_view
        self.file_model = self.view.file_controller.model

        self.correlation_frame = None

        pub.subscribe(self._SyncHeaders, "Data_Changed")
        pub.subscribe(self.CheckObj, "MenuBar_Check", func=self.EmulatorCheck)
        pub.subscribe(self.CheckObj, "MenuBar_Open", func=self.OpenFile)
        pub.subscribe(self.CheckObj, "MenuBar_SaveNew", func=self.SaveNew)
        pub.subscribe(self.CheckObj, "MenuBar_Save", func=self.Save)
        pub.subscribe(self.CheckObj, "MenuBar_SaveModelName", func=self.SaveWithModelName)
        pub.subscribe(self.CheckObj, "MenuBar_AddComment", func=self.AddComment)

        pub.subscribe(self.CheckObj, "MenuBar_ReTrain", func=self.ReTrain)
        pub.subscribe(
            self.CheckObj,
            "MenuBar_GenHyperCube",
            func=self.GenHyperCube)
        pub.subscribe(self.CheckObj, "MenuBar_Emulate", func=self.Emulate)
        pub.subscribe(
            self.CheckObj,
            "MenuBar_ChainedEmulate",
            func=self.ChainedEmulate)
        pub.subscribe(
            self.CheckObj,
            "MenuBar_IndividualEmulate",
            func=self.IndividualEmulate)

        pub.subscribe(self.CheckObj, "MenuBar_Report", func=self.TrainReport)
        pub.subscribe(
            self.CheckObj,
            "MenuBar_Correlation",
            func=self.Correlation)
        pub.subscribe(self.CheckObj, "MenuBar_Posterior", func=self.Posterior)
        pub.subscribe(
            self.CheckObj,
            "MenuBar_TraceSummary",
            func=self.ShowSummary)
        pub.subscribe(
            self.CheckObj,
            "MenuBar_TraceDiagnosis",
            func=self.ShowDiagnosis)



        # communication between model_par_model and prior_model
        pub.subscribe(self.FillPriorRange, 'PriorToolBar_Refresh')

        # sync file according the file_controller
        pub.subscribe(self.LoadFile, 'emulatorFileChanged')
        pub.subscribe(self.LoadFile, 'listFileChanged')

        pub.subscribe(self.PosteriorGauge, 'Posterior_Drawing')
        pub.subscribe(self.DestroyPosteriorGauge, 'Posterior_Drawn')

        # block repeated load chain prompt when some of the chained files has
        # their own list of chained files
        self.block_load_chain_prompt = False


    def CheckObj(self, func, obj, evt):
        orig_obj = obj
        while True:
            if obj is self.view:
                func(obj, evt)
                break
            obj = obj.GetParent()
            if obj is None:
                break

    def FillPriorRange(self, obj, evt):
        par_data = self.model_par_model.GetData().astype(np.float)
        mins = np.min(par_data)
        maxs = np.max(par_data)
        self.view.prior_controller.FillPriorRange(mins.values, maxs.values)

    def ShowSummary(self, obj, evt):
        if self.file_model.trace_filename is None:
            wx.MessageBox(
                'No trace file is selected. Cannot show trace summary.',
                'Error',
                wx.OK | wx.ICON_ERROR)
            return
        if self.file_model.trace_filename != self.file_model.emulator_filename:
            wx.MessageBox(
                'To display trace summary, please make sure trace file and emulator file are the same.',
                'Error',
                wx.OK | wx.ICON_ERROR)
            return
        with pd.HDFStore(self.file_model.trace_filename, 'r') as store:
            description = 'There are no trace.'
            if 'trace' in store:
                description = ''
                trace = store['trace']
                attrs = store.get_storer('trace').attrs
                id_to_model = ['']
                if 'model_names' in attrs:
                    id_to_model = attrs['model_names']
                # only select columns in prior
                for id_, name in enumerate(id_to_model):
                    if name is None:         
                        name = ''
                    par_names = store['PriorAndConfig'].index
                    description = description + name + '\n'
                    if id_ != 0:
                        par_names = ['%s_%d' % (name, id_) for name in par_names]
                    temp = trace[par_names].describe(percentiles=[0.5*(1-0.95), 0.5*(1-0.68),.5,0.5*(1+0.68), 0.5*(1+.95)]).to_string()
                    if id_ > 0:
                        temp = temp[temp.find('\n') + 1:]
                    description = description + temp + '\n\n'

                # show list of files if the trace is chainged
                if 'chained_files' in attrs:
                    chained_files = attrs.chained_files
                    description += '\n\nChained files:\n'
                    description += '\n'.join(chained_files)
                if 'ModelChoice' in trace:
                    description += '\n\nModel comparison:'
                    id_to_model = attrs['model_names']
                    tot = trace.shape[0]
                    for idx, name in enumerate(id_to_model):
                        description += '\n    %s = %f' % (name, np.sum(trace['ModelChoice'] == idx)/float(tot))
            attrs = store.get_storer('PriorAndConfig').attrs
            if 'comment' in attrs:
                description += '\n\nComment: ' + attrs['comment']
 
            FlexMessageBox(description, self.view, title='Summary').Show()


    def GenHyperCube(self, obj, evt):
        prior = self.prior_model.GetData(drop_index=False)
        if prior.empty:
            raise RuntimeError('You need to fill up your parameter range')
            # wx.MessageDialog(
            #    self.view,
            #    "You need to fill up your parameter range",
            #    "Warning",
            #    wx.OK | wx.ICON_WARNING,
            # ).ShowModal()
            # return None
        ranges = prior.loc[["Min", "Max"], :].astype(float).values.T
        from GUI.SelectHyperCubeOption import SelectHyperCubeOption
        dlg = SelectHyperCubeOption(self.view)
        if dlg.ShowModal() == wx.ID_OK:
            result = dlg.GetValue()
            dlg.Destroy()
            NPts = result['NPts']

            pad = result['Pad']
            span = ranges[:, 1] - ranges[:, 0]
            ranges[:,0] = ranges[:, 0] - pad*span
            ranges[:,1] = ranges[:, 1] + pad*span

            from Utilities.LatinHyperCube import GenerateLatinHyperCube

            content = GenerateLatinHyperCube(NPts, ranges)
            rows = np.arange(1, 1 + content.shape[0])
            cols = np.arange(0, content.shape[1])
            self.model_par_model.SetValue(rows, cols, content)
            self.model_par_view.ForceRefresh()


    def TrainReport(self, obj, evt):
        if self.file_model.emulator_filename is not None:
            fig = Figure((self.config_data['PopUpWidth'], self.config_data['PopUpHeight']), 75)
            frame = MatplotlibFrame(None, fig)

            try:
                gauge = TrainingProgressFrame(
                    1,
                    None,
                    -1,
                    "Generating report",
                    size=(wx.ScreenDC().GetPPI()[0]*self.config_data['OptionWindowWidth'], -1),
                    text_label="Training report generation in progress",
                    col_labels=[""],
                )
                PtsFraction = self.config_data['TrainReportUpdateInterval']

                def gaugeUpdatePts(progress): return gauge.updateProgress(
                    progress * PtsFraction * 100
                )  # first half of calculation contributes ~ 10 percent

                def gaugeUpdateSteps(progress): return gauge.updateProgress(
                    progress * (1 - PtsFraction) * 100 + PtsFraction * 100
                )  # second half of the calculation
                pub.subscribe(gaugeUpdatePts, "NumberOfPtsProgress")
                pub.subscribe(gaugeUpdateSteps, "NumberOfStepsProgress")

                from TrainEmulator import TrainingCurve

                gauge.Show()
                TrainingCurve(fig, config_file=self.file_model.emulator_filename)
                pub.unsubscribe(gaugeUpdatePts, "NumberOfPtsProgress")
                pub.unsubscribe(gaugeUpdateSteps, "NumberOfStepsProgress")
            except Exception as e:
                raise e
            else:
                frame.SetData()
                frame.Show()
            finally:
                gauge.Destroy()

    def ChainedEmulate(self, obj, evt):
        self.Emulate(obj, evt, False)

    def IndividualEmulate(self, obj, evt):
        from GUI.SelectEmulationOption import SelectEmulationOption

        EmuOption = SelectEmulationOption(self.view)
        res = EmuOption.ShowModal()
        if res == wx.ID_OK:
            from GUI.Model import CalculationFrame

            options = EmuOption.GetValue()
            for file_ in self.file_model.list_filenames:
                filenames = file_
                nevent = options["nevent"]
                frame = CalculationFrame(
                    None, -1, "Progress", self.workenv, nevent, 
                    width=self.config_data['GaugeWidth'], height=self.config_data['GaugeHeight'], 
                    max_speed_per_cpu=self.config_data['MaxSpeedPerCPU'])
                frame.Show()

                try:
                    frame.OnCalculate(
                        {
                            "config_file": filenames,
                            "nsteps": nevent,
                            "clear_trace": options["clear_trace"],
                            "burnin": options["burnin"],
                            "model_comp": options['model_comp']
                        }
                    )
                    self.view.file_controller.update_metadata(file_)
                except ThreadsException as ex:
                    wx.MessageBox(str(ex), 'Error', wx.OK | wx.ICON_ERROR)

    def Emulate(self, obj, evt, single_file=True):
        if self.file_model.trace_filename is None:
            wx.MessageBox(
                'No trace file is selected. Cannot start analysis',
                'Error',
                wx.OK | wx.ICON_ERROR)
            return
        if self.file_model.trace_filename != self.file_model.emulator_filename:
            wx.MessageBox(
                'For single file analysis, please make sure trace file and emulator file is the same',
                'Error',
                wx.OK | wx.ICON_ERROR)
            return
        if single_file:
            filenames = self.file_model.trace_filename
        else:
            filenames = self.file_model.get_list_filenames_with_trace_first()

        if filenames is not None:
            from GUI.SelectEmulationOption import SelectEmulationOption

            EmuOption = SelectEmulationOption(self.view, model_comp=not single_file)
            res = EmuOption.ShowModal()
            if res == wx.ID_OK:
                options = EmuOption.GetValue()
                nevent = options["nevent"]

                if options['model_comp']:
                    try:
                        from MCMCTrace import GroupConfigFiles
                        nameCheck = GroupConfigFiles(filenames)
                        description = ''
                        for name, fileList in nameCheck.items():
                           description += 'Files for model: ' + name + '\n'
                           for idx, filename in enumerate(fileList):
                               description += '   %d: %s\n' % (idx, filename)

                        if wx.ID_OK != FlexMessageBox(description, None, title='Files check').ShowModal():
                            return
                    except RuntimeError as ex:
                        wx.MessageBox('Cannot group files by models because:\n' +
                                      str(ex) + '\nModel comparison is disabled', 'Error', wx.OK | wx.ICON_ERROR)
                        return
                else:
                    if not single_file:
                        description = filenames if isinstance(filenames, list) else [filenames]
                        description = 'Files chosend:\n' + '\n'.join(description)
                        if wx.ID_OK != FlexMessageBox(description, None, title='Files check').ShowModal():
                            return


                from GUI.Model import CalculationFrame

                frame = CalculationFrame(
                    None, -1, "Progress", self.workenv, nevent, 
                    width=self.config_data['GaugeWidth'], height=self.config_data['GaugeHeight'], 
                    max_speed_per_cpu=self.config_data['MaxSpeedPerCPU'])
                frame.Show()

                try:
                    frame.OnCalculate(
                        {
                            "config_file": filenames,
                            "nsteps": nevent,
                            "clear_trace": options["clear_trace"],
                            "burnin": options["burnin"],
                            "model_comp": options['model_comp']
                        }
                    )
                    self.view.file_controller.update_metadata(self.file_model.trace_filename)
                    for _, plugins in self.plugin_controllers.items():
                        plugins.Save()
                    self.Correlation(None, None, False)
                except ThreadsException as ex:
                    wx.MessageBox(str(ex), 'Error', wx.OK | wx.ICON_ERROR)

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
                    # don't show correlation if the user close the option
                    # dialog
                    return
            if not self.correlation_frame:
                fig = Figure((self.config_data['PopUpWidth'], self.config_data['PopUpHeight']), 75)
                self.correlation_frame = MatplotlibFrame(None, fig)
            self.correlation_frame.fig.clf()
            from Utilities.Utilities import PlotTrace

            if kwargs["overlay_pt"]:
                kwargs["mark_point"] = self.plugin_controllers['Ask emulator'].left_model.GetData()
            del kwargs["overlay_pt"]
            kwargs['model_filename'] = self.file_model.emulator_filename

            PlotTrace(
                self.file_model.trace_filename,
                self.correlation_frame.fig,
                **kwargs)
            self.correlation_frame.SetData()
            self.correlation_frame.Show()

    def Posterior(self, obj, evt):
        if self.file_model.emulator_filename is not None:
            if not self.correlation_frame:
                fig = Figure((self.config_data['PopUpWidth'], self.config_data['PopUpHeight']), 75)
                self.correlation_frame = MatplotlibFrame(None, fig)
            self.correlation_frame.fig.clf()
            from PlotPosterior import PlotOutput

            dlg = wx.MessageDialog(self.view, 'Are variables dis-continuous?', 'Question', wx.YES_NO | wx.ICON_QUESTION)
            result = dlg.ShowModal()

            try:
                btn = PlotOutput(
                    self.file_model.emulator_filename,
                    self.correlation_frame.fig,
                    trace_filename=self.file_model.trace_filename,
                    discontVar=result == wx.ID_YES)
            except Exception as e:
                raise e
            else:
                self.correlation_frame.SetData()
                self.correlation_frame.Show()

    def ShowDiagnosis(self, obj, evt):
        if self.file_model.trace_filename is not None:
            if not self.correlation_frame:
                fig = Figure((self.config_data['PopUpWidth'], self.config_data['PopUpHeight']), 75)
                self.correlation_frame = MatplotlibFrame(None, fig)
            self.correlation_frame.fig.clf()
            from PlotTraceSteps import PlotTraceSteps
            try:
                PlotTraceSteps(self.file_model.trace_filename, self.correlation_frame.fig, self.config_data['traceRunningWinSize'])
            except Exception as e:
                raise e
            else:
                self.correlation_frame.SetData()
                self.correlation_frame.Show()
        else:
            raise RuntimeError('No trace loaded.')


    def PosteriorGauge(self):
        self.gauge = TrainingProgressFrame(
            1,
            None,
            -1,
            "Generating output posterior",
            size=(wx.ScreenDC().GetPPI()[0]*self.config_data['OptionWindowWidth'], -1),
            text_label="Output posterior generation in progress",
            col_labels=[""],
        )
        pub.subscribe(self.gaugeProgress, "PosteriorOutputProgress")
        self.gauge.Show()

    def gaugeProgress(self, progress): 
        return self.gauge.updateProgress(progress * 100)

    def DestroyPosteriorGauge(self):
        pub.unsubscribe(self.gaugeProgress, "PosteriorOutputProgress")
        self.gauge.Destroy()
       

    def SaveWithModelName(self, obj, evt, changeName=True):
        model_name = None
        if changeName:
            dlg = wx.TextEntryDialog(obj, 'Enter model name', 'Model name')
            if dlg.ShowModal() == wx.ID_OK:
                model_name = dlg.GetValue()
            dlg.Destroy()

        prior = self.prior_model.GetData(drop_index=False)
        model_X = self.model_par_model.GetData().astype("float")
        model_Y = self.model_obs_model.GetData().astype("float")
        exp = self.exp_model.GetData(drop_index=False).astype("float")

        with pd.HDFStore(self.file_model.emulator_filename, "a") as store:
            if model_X.equals(
                    store["Model_X"]) and model_Y.equals(
                    store["Model_Y"]):
                from ChangeFileContent import ChangeFileContent

                ChangeFileContent(store, prior, exp, model_name)
                config = store.get_storer("PriorAndConfig").attrs.my_attribute
                model_name = None
                if 'name' in config:
                    model_name = config['name']
                self.view.prior_controller.SetModelName(model_name)
            else:
                raise RuntimeError('Model values has changed. You must train the emulator again')

        for _, plugins in self.plugin_controllers.items():
            plugins.Save()
        

    def Save(self, obj, evt):
        self.SaveWithModelName(obj, evt, changeName=False)

    def AddComment(self, obj, evt):
        dlg = wx.TextEntryDialog(self.view, "Comments to add")
        dlg.ShowModal()
        comment = dlg.GetValue()
        dlg.Destroy()
        with pd.HDFStore(self.file_model.trace_filename, "a") as store:
            store.get_storer('PriorAndConfig').attrs['comment'] = comment

    def ReTrain(self, obj, evt):
        if self.file_model.emulator_filename is None:
            wx.MessageBox(
                'No emulator file is loaded. The emulator will be saved as new file.',
                'Warning',
                wx.OK | wx.ICON_WARNING)
            self.SaveNew(obj, evt, None)
        else:
            # ? Why does outFile have to be a list....
            # re-save comment
            comments = None
            with pd.HDFStore(self.file_model.emulator_filename, 'r') as store:
                attrs = store.get_storer('PriorAndConfig').attrs
                if 'comment' in attrs:
                    comments = attrs['comment']

            if self.SaveNew(obj, evt, [self.file_model.emulator_filename]):
                if not comments is None:
                    with pd.HDFStore(self.file_model.emulator_filename, 'a') as store:
                        store.get_storer('PriorAndConfig').attrs['comment'] = comments


    def SaveNew(self, obj, evt, outFile=None):
        prior = self.prior_model.GetData(drop_index=False)
        if prior.empty:
            raise RuntimeError('Prior is not filled.')
        model_X = self.model_par_model.GetData()
        model_Y = self.model_obs_model.GetData()
        exp = self.exp_model.GetData(drop_index=False)

        model_name = None

        if outFile is None:
            """
            Create and show the Open FileDialog if no outFile is specified
            """
            default_dir = ''
            if self.file_model.emulator_filename is not None:
                default_dir = os.path.dirname(self.file_model.emulator_filename)
            wildcard = "Python source (*.h5)|*.h5|" "All files (*.*)|*.*"
            dlg = wx.FileDialog(
                obj,
                message="Save project as ...",
                defaultFile="",
                defaultDir=default_dir,
                style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT,
            )
            result = dlg.ShowModal()
            outFile = dlg.GetPaths()
            dlg.Destroy()

            if (
                result == wx.ID_CANCEL
            ):  # Either the cancel button was pressed or the window was closed
                return False
        else:
            # keep model name if we are saving files in-place
            with pd.HDFStore(outFile[0], "r") as store:
                config = store.get_storer("PriorAndConfig").attrs.my_attribute
                if 'name' in config:
                    model_name = config['name']

        from GUI.SelectTrainingOption import SelectOption

        frame = SelectOption()
        res = frame.ShowModal()
        if res == wx.ID_CANCEL:
            return False
        else:
            args = frame.AdditionalData()
            frame.Destroy()

        if args["principalcomp"] is not None:
            gauge = TrainingProgressFrame(
                args["principalcomp"], None, -1, "Training progress", size=(wx.ScreenDC().GetPPI()[0]*self.config_data['OptionWindowWidth'], -1)
            )
        else:
            gauge = TrainingProgressFrame(
                1, None, -1, "Training progress", size=(wx.ScreenDC().GetPPI()[0]*self.config_data['OptionWindowWidth'], -1)
            )
        def gaugeUpdate(step, progress, mag, nuggets, scales): return [
            gd.DefaultOutput(step, progress, mag, nuggets, scales),
            gauge.updateProgress(progress),
        ]
        pub.subscribe(gaugeUpdate, "GradientProgress")

        try:
            gauge.Show()

            from TrainEmulator import Training

            Training(
                prior,
                model_X,
                model_Y,
                exp,
                outFile[0],
                abs_output=True,
                modelname=model_name,
                **args)
            # pub.unsubscribe(gaugeUpdate, 'GradientProgress')
        except Exception as e:
            raise type(e)(str(e) + '\nError occure in training. Try changing training settings to prevent this.').with_traceback(sys.exc_info()[2])
        finally:
            pub.unsubscribe(gaugeUpdate, "GradientProgress")
            gauge.Destroy()

        for _, plugins in self.plugin_controllers.items():
            plugins.SaveNew(outFile[0])

        #if self.file_model.emulator_filename is not None:
        #     if self.file_model.emulator_filename != outFile[0]:
        #    self.view.file_controller.remove_file_highlight_inplace(
        #        self.file_model.emulator_filename)
        self.view.file_controller.add_file(outFile[0], exist_ok=True)
        self.LoadFile()

        if args['TestData'] > 0:
            with pd.HDFStore(outFile[0], "a") as store:
                store["Training_idx"] = pd.DataFrame(list(range(model_X.shape[0])), dtype=int).sample(n=model_X.shape[0] - args['TestData']).sort_index()
            self.TrainReport(None, None)

        return True

    def OpenFile(self, obj, evt):
        # dlg = wx.FileDialog(
        #    obj,
        #    message="Choose a file",
        #    defaultFile="",
        #    style=wx.FD_OPEN | wx.FD_MULTIPLE,
        # )
        #result = dlg.ShowModal()
        #path = dlg.GetPaths()
        # dlg.Destroy()

        # if result != wx.ID_OK:
        #    return False
        # self.view.file_controller.add_file()
        self.view.file_controller.add_file()
        # self.SyncFileControllerAndViewer()#path[0])

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
            with pd.HDFStore(self.file_model.emulator_filename, "r") as store:
                self.prior_model.SetData(store["PriorAndConfig"].T)
                self.exp_model.SetData(
                    pd.concat([store["Exp_Y"], store["Exp_YErr"]], axis=1).T)
                self.model_par_model.SetData(store["Model_X"])
                self.model_obs_model.SetData(store["Model_Y"])
                config = store.get_storer("PriorAndConfig").attrs.my_attribute
                model_name = None
                if 'name' in config:
                    model_name = config['name']
                self.view.prior_controller.SetModelName(model_name)

        self.prior_model.ResetUndo()
        self.model_par_model.ResetUndo()
        self.model_obs_model.ResetUndo()
        self.exp_model.ResetUndo()

        for _, plugins in self.plugin_controllers.items():
            plugins.LoadFile(filename)
 

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
            self._SyncHeaders2Ways(obj, self.exp_model, self.model_obs_model)
            self.view.Refresh()

    def _SyncHeaders2Ways(self, obj, *models): # model1, model2):
        # value = obj.data.iloc[0].replace(r'^\s*$', np.nan, regex=True).dropna(how='all')
        if obj not in models:
            return
        value = obj.data.iloc[0].replace(r"^\s*$", np.nan, regex=True)
        for model in models:
            if obj is model:
                continue
            model.ChangeValues(
                0,
                np.arange(
                    value.shape[0]),
                value,
                send_changed=False)

class GUIViewer(wx.Frame):
    def __init__(self, parent, app, config_data):
        self.config_data = config_data
        wx.Frame.__init__(
            self,
            parent,
            wx.NewId(),
            "Bayesian analysis GUI",
            size=wx.ScreenDC().GetPPI().Scale(
                config_data['WindowsWidth'],
                config_data['WindowsHeight']))  # 1000,400))
        self.app = app
        sys.excepthook = MyExceptionHook

        panel = wx.Panel(self)
        split_panel = wx.SplitterWindow(panel)
        notebook = wx.Notebook(split_panel)
        from GUI.GUIController.GUIMenu import GUIMenuBar

        self.menubar = GUIMenuBar(self)
        self.SetMenuBar(self.menubar)

        prior_panel = wx.Panel(notebook)
        self.prior_controller = PriorController(prior_panel, config_data['GridNCol'])
        prior_sizer = wx.BoxSizer(wx.VERTICAL)
        prior_sizer.Add(self.prior_controller.toolbar, 0, wx.EXPAND)
        prior_sizer.Add(self.prior_controller.view, 1, wx.EXPAND)
        prior_sizer.Add(self.prior_controller.model_name_view, 0.1, wx.EXPAND|wx.LEFT|wx.RIGHT|wx.TOP, border=10)
        prior_panel.SetSizer(prior_sizer)
        notebook.AddPage(prior_panel, "Parameters prior")

        grid_panel = wx.Panel(notebook)
        self.model_input_controller = SplitViewController(grid_panel, config_data['GridNRow'], config_data['GridNCol'])
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.model_input_controller.view, 1, wx.EXPAND)
        # grid_panel.SetSizer(sizer)
        notebook.AddPage(grid_panel, "Model calculations")

        #exp_panel = wx.Panel(notebook)
        self.exp_controller = GridController(
            grid_panel, 3, config_data['GridNCol'], size=(-1, wx.ScreenDC().GetPPI()[1]*config_data['ExpHeight']))
        self.exp_controller.model.data.index = ["Name", "Values", "Errors"]
        #exp_sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.exp_controller.toolbar, 0.1, wx.EXPAND)
        sizer.Add(self.exp_controller.view, 0, wx.EXPAND)
        grid_panel.SetSizer(sizer)

        # exp_panel.SetSizer(exp_sizer)
        #notebook.AddPage(exp_panel, "Experimental data")
        plugin_lists = {'Ask emulator': EvalEmuController, 'Closure': ClosureTestController}
        self.plugins = {}

        for key, controller_constructor in plugin_lists.items():
            plugin_panel = wx.Panel(notebook)
            self.plugins[key] = controller_constructor(plugin_panel, config_data)
            notebook.AddPage(plugin_panel, key)

        sizer = wx.BoxSizer(wx.VERTICAL)
        self.file_controller = FileController(
            file_viewer_kwargs={
                'parent': split_panel}, display_kwargs={
                'parent': panel})  # , 'size': (-1,  wx.ScreenDC().GetPPI()[0]/3)})
        split_panel.SplitVertically(
            self.file_controller.file_view,
            notebook,
            wx.ScreenDC().GetPPI()[0] * config_data['FileWidth'])
        #hsizer = wx.BoxSizer(wx.HORIZONTAL)
        #hsizer.Add(notebook, 1, wx.EXPAND | wx.EXPAND, 5)
        #hsizer.Add(self.file_controller.file_view, 0.2, wx.EXPAND, 0)
        sizer.Add(split_panel, 1, wx.EXPAND, 0)
        sizer.Add(self.file_controller.display_view, 0., wx.EXPAND)

        self.Bind(wx.EVT_CLOSE, self.OnClose)
        panel.SetSizerAndFit(sizer)
        self.Layout()
        self.Show()

    def OnClose(self, evt):
        self.Destroy()
        self.app.ExitMainLoop()


def MyExceptionHook(etype, value, trace):
    """
    Handler for all unhandled exceptions.
    :param `etype`: the exception type (`SyntaxError`, `ZeroDivisionError`, etc...);
    :type `etype`: `Exception`
    :param string `value`: the exception error message;
    :param string `trace`: the traceback header, if any (otherwise, it prints the
     standard Python header: ``Traceback (most recent call last)``.
    """
    frame = wx.GetApp().GetTopWindow()
    tmp = traceback.format_exception(etype, value, trace)
    exception = "".join(tmp)
    #
    sys.stdout.write(exception)
    sys.stdout.flush()
    dlg = wx.MessageDialog(
        None,
        str(value),
        'Warning',
        wx.OK | wx.ICON_WARNING,
    )
    dlg.ShowModal()
    dlg.Destroy()


def main():
    from mpi4py import MPI
    from Utilities.MasterSlave import MasterSlave

    comm = MPI.COMM_WORLD
    size = comm.Get_size()

    work_environment = MasterSlave(comm)

    # gd.UseDefaultOutput()
    app = wx.App(0)
    controller = GUIController(None, app=app, workenv=work_environment)
    controller.view.Show()

    if len(sys.argv) >= 2:
        controller.LoadFile(sys.argv[1:])

    app.MainLoop()
    work_environment.Close()


if __name__ == "__main__":
    main()
