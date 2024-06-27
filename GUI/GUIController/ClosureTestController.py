import wx
import wx.grid as gridlib
from matplotlib.figure import Figure
from pubsub import pub
import shutil
import numpy as np
import pandas as pd
import os

from GUI.FlexMessageBox import FlexMessageBox
from GUI.MatplotlibFrame import MatplotlibFrame
from GUI.GridController.GridController import SplitViewController


class ClosureTestController(SplitViewController):
    def __init__(self, panel, config_data):
        super().__init__(panel, config_data['GridNRow'], config_data['GridNCol'], no_clear_all=True)
        self.config_data = config_data
        # attr has to be duplicated to avoid reference counting error
        # first row of both left and right panel are not multable
        attr = gridlib.GridCellAttr()
        # first row is reserved for header
        attr.SetReadOnly(True)
        attr.SetBackgroundColour("Grey")

        self.left_view.SetRowAttr(0, attr)

        # first row of both left and right panel are not multable
        attr = gridlib.GridCellAttr()
        # first row is reserved for header
        attr.SetReadOnly(True)
        attr.SetBackgroundColour("Grey")
        self.right_view.SetRowAttr(0, attr)

        button_sizer = wx.BoxSizer(wx.HORIZONTAL)
        GenHyperCubeButton = wx.Button(
            panel, -1, "Gen closure para")
        button_sizer.Add(GenHyperCubeButton)
        GenHyperCubeButton.Bind(
            wx.EVT_BUTTON,
            lambda evt: pub.sendMessage("ClosureTest_GenRand", obj=self, evt=evt)
        )

        EvalClosureButton = wx.Button(
            panel, -1, "Test")
        button_sizer.Add(EvalClosureButton)
        EvalClosureButton.Bind(
            wx.EVT_BUTTON,
            lambda evt: pub.sendMessage("ClosureTest_Eval", obj=self, evt=evt)
        )

        GenClosureBayesButton = wx.Button(
            panel, -1, "Gen closure Bayes")
        button_sizer.Add(GenClosureBayesButton)
        GenClosureBayesButton.Bind(
            wx.EVT_BUTTON,
            lambda evt: pub.sendMessage("ClosureTest_Bayes", obj=self, evt=evt)
        )

        GenBayesPosteriorButton = wx.Button(
            panel, -1, "Gen posterior for closure.")
        button_sizer.Add(GenBayesPosteriorButton)
        GenBayesPosteriorButton.Bind(
            wx.EVT_BUTTON,
            lambda evt: pub.sendMessage("ClosureTest_GenBayesPosterior", obj=self, evt=evt)
        )

        EvalClosureBayesButton = wx.Button(
            panel, -1, "Compare posterior with truth")
        button_sizer.Add(EvalClosureBayesButton)
        EvalClosureBayesButton.Bind(
            wx.EVT_BUTTON,
            lambda evt: pub.sendMessage("ClosureTest_Posterior", obj=self, evt=evt)
        )

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(button_sizer)
        sizer.Add(self.view, 1, wx.EXPAND)
        panel.SetSizer(sizer) 

        self.correlation_frame = None

        pub.subscribe(self.CheckObj, "ClosureTest_Eval", func=self.Eval)
        pub.subscribe(self.CheckObj, "ClosureTest_Posterior", func=self.Posterior)
        pub.subscribe(self.CheckObj, "ClosureTest_Bayes", func=self.GenFilesForBayes)
        pub.subscribe(self.CheckObj, "ClosureTest_GenBayesPosterior", func=self.GenBayesPosterior)
        pub.subscribe(self.CheckObj, "ClosureTest_GenRand", func=self.GenClosureRand)
        pub.subscribe(self._SyncHeaders, "Data_Changed")


    def SetHeadController(self, head):
        self.head_controller = head

    def SaveNew(self, filename):
        closure_x = self.left_model.GetData()
        closure_y = self.right_model.GetData()

        assert closure_x.shape[0] == closure_y.shape[0], 'Number of rows of parameters and observables must be the same'
        if closure_x.shape[0] <= 1:
            return # save nothing if there's nothing in the grid

        with pd.HDFStore(filename, 'a') as store:
            attrs = store.get_storer('PriorAndConfig').attrs
            attrs['ClosureData'] = [closure_x, closure_y]


    def _SyncHeaders(self, obj, evt):
        # Remember, headers cannot be changed in this plugin
        self._SyncHeadersFrom(obj, [self.head_controller.prior_model, self.head_controller.model_par_model], self.left_model)
        self._SyncHeadersFrom(obj, [self.head_controller.exp_model, self.head_controller.model_obs_model], self.right_model)


    def _SyncHeadersFrom(self, obj, source_models, dest): # model1, model2):
        # value = obj.data.iloc[0].replace(r'^\s*$', np.nan, regex=True).dropna(how='all')
        if obj not in source_models:
            return
        value = obj.data.iloc[0].replace(r"^\s*$", np.nan, regex=True)
        dest.ChangeValues(
            0,
            np.arange(
                value.shape[0]),
            value,
            send_changed=False)


    def LoadFile(self, filename):
        self.controller_left.ClearAllButHeader()
        self.controller_right.ClearAllButHeader()


        if self.head_controller.file_model.emulator_filename is not None:
            with pd.HDFStore(self.head_controller.file_model.emulator_filename, 'a') as store:
                attrs = store.get_storer('PriorAndConfig').attrs
                if 'ClosureData' in attrs:
                    self.left_model.SetData(attrs['ClosureData'][0])
                    self.right_model.SetData(attrs['ClosureData'][1])
    
    def Save(self):
        assert self.head_controller.file_model.emulator_filename is not None, 'Emulator filename cannot be None'
        self.SaveNew(self.head_controller.file_model.emulator_filename) 


    def GenBayesPosterior(self, obj, evt):
        # perform bayesian analysis on all files within a directory
        dirs = self.head_controller.file_model.list_dirnames
        if len(dirs) == 0:
            raise RuntimeError('No directory selected')
        # check if all files in every directories are identical
        files = None
        for directory in dirs:
            if files is None:
                files = set(os.listdir(directory))
            else:
                if files != set(os.listdir(directory)):
                    raise RuntimeError('Files in directory ' + directory + ' is different from the rest!.')

        # print list of directories and files
        string = "Directories to emulate:\n\n" + '\n'.join(dirs)
        string = string + "\n\nChoose file where traces are saved."

        dlg = wx.SingleChoiceDialog(self.head_controller.view, string, "Trace selection", list(files))
        if dlg.ShowModal() == wx.ID_OK:
            trace_name = dlg.GetStringSelection()
        dlg.Destroy()

        # do analysis on each directory
        from GUI.SelectEmulationOption import SelectEmulationOption
        from GUI.Model import CalculationFrame

        EmuOption = SelectEmulationOption(self.view, model_comp=False)
        res = EmuOption.ShowModal()
        if res == wx.ID_OK:
            options = EmuOption.GetValue()
            nevent = options["nevent"]

            for directory in dirs:
                filenames = [os.path.join(directory, file) for file in files]
                # move trace name to the first element of the list
                trace_file = os.path.join(directory, trace_name)
                filenames.insert(0, filenames.pop(filenames.index(trace_file)))

                frame = CalculationFrame(
                    None, -1, "Progress on " + directory, self.head_controller.workenv, nevent, 
                    width=self.config_data['GaugeWidth'], height=self.config_data['GaugeHeight'], 
                    max_speed_per_cpu=self.config_data['MaxSpeedPerCPU'])
                frame.Show()

                frame.OnCalculate(
                    {
                        "config_file": filenames,
                        "nsteps": nevent,
                        "clear_trace": options["clear_trace"],
                        "burnin": options["burnin"],
                        "model_comp": options['model_comp']
                    }
                )
                self.head_controller.view.file_controller.update_metadata(trace_file)




    def GenFilesForBayes(self, obj, evt, prefix=None):
        if prefix is None:
            dlg = wx.TextEntryDialog(self.head_controller.view, 'Directory prefix for all closure files.', 'Directory name')
            if dlg.ShowModal() == wx.ID_OK:
                prefix = dlg.GetValue()
                dlg.Destroy()
            else:
                dlg.Destroy()
                return

        # find the path to the trace file
        # closure files will be saved to that directories
        orig = self.head_controller.file_model.emulator_filename
        fileList = self.head_controller.file_model.get_list_filenames_with_trace_first()
        curr_path, _ = os.path.split(fileList[0])
        for file in fileList:
            _, basename = os.path.split(file)
            # switch file for bayes
            self.head_controller.file_model.trace_filename = file
            self.head_controller.file_model.emulator_filename = file
            # get all entries from closure_test
            closure_truth = self.right_model.GetData().to_numpy().astype('float')
            # will use the same fractional error as data for closure test
            exp = self.head_controller.exp_model.GetData(drop_index=False).astype("float")
            err_frac = exp.loc['Errors'].astype('float')/exp.loc['Values'].astype('float')


            for i, truth in enumerate(closure_truth):
                new_dir = os.path.join(curr_path, '%s_%d' % (prefix, i))
                os.makedirs(new_dir, exist_ok=True)
                new_file = os.path.join(new_dir, basename)
                shutil.copy2(file, new_file)

                exp = pd.DataFrame.from_dict({"Values": truth, "Errors": np.abs(err_frac*truth)}, orient='index')
    
                with pd.HDFStore(new_file, "a") as store:
                    from ChangeFileContent import ChangeFileContent
                    ChangeFileContent(store, None, exp, None)
                
                self.head_controller.LoadFile(new_file)

                # collapse on directory tree for visualization
                if file == fileList[-1]:
                    self.head_controller.file_view.collapse(new_dir)
                wx.YieldIfNeeded()
        self.head_controller.file_view.select(orig)
 
    def GenClosureRand(self, obj, evt):
        prior = self.head_controller.prior_model.GetData(drop_index=False)
        if prior.empty:
            raise RuntimeError('You need to fill up your parameter range')

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

            from Utilities.LatinHyperCube import GenerateRandomLattice

            content = GenerateRandomLattice(NPts, ranges)
            rows = np.arange(1, 1 + content.shape[0])
            cols = np.arange(0, content.shape[1])
            self.left_model.SetValue(rows, cols, content)
            self.left_view.ForceRefresh()

    def Posterior(self, obj, evt):
        if self.head_controller.file_model.trace_filename is None:
            wx.MessageBox(
                'No trace file is selected. Trace are loaded from highlighted directories with the same name. Please open one trace file with the right filename.',
                'Error',
                wx.OK | wx.ICON_ERROR)
            return
        trace_name = os.path.basename(self.head_controller.file_model.trace_filename)

        # get true parameters
        para_table = self.left_model.GetData()
        para = para_table.astype(float).to_numpy()
        par_names = list(para_table.columns.values)
        if para.shape[0] == 0:
            raise RuntimeError('No truth data is available on tab "Closure test"')

        # verify number of highlighted directories = number of true parameters
        dirs = self.head_controller.file_model.list_dirnames
        ordered_dir = []
        prefix = None

        # check if all highlighted directories share the same suffix
        truths = []
        predictions = []
        prediction_errs = []
        for i, truth  in enumerate(para):
            suffix = '_%d' % i
            for directory in dirs:
                if directory.endswith(suffix):
                    if prefix is None:
                        prefix = directory[:-len(suffix)]
                    elif prefix != directory[:-len(suffix)]:
                        raise RuntimeError('Not all directories has the same prefix! Please only highlight the relavent directories')
                    dirs.remove(directory)

                    # find trace medium and C.I.
                    with pd.HDFStore(os.path.join(directory, trace_name), 'r') as store:
                        if 'trace' in store:
                            trace = store['trace']
                            # only select columns in prior
                            y_range = trace[par_names].quantile([0.5-0.68/2, 0.5, 0.5+0.68/2])

                            truths.append(truth)
                            predictions.append(y_range.iloc[1].astype('float').to_numpy())
                            prediction_errs.append([y_range.iloc[1].astype('float').to_numpy() - y_range.iloc[0].astype('float').to_numpy(),
                                                    y_range.iloc[2].astype('float').to_numpy() - y_range.iloc[1].astype('float').to_numpy()])
                    break

        from PlotClosureTest import PlotClosureTest
        if not self.correlation_frame:
            fig = Figure((self.config_data['PopUpWidth'], self.config_data['PopUpHeight']), 75)
            self.correlation_frame = MatplotlibFrame(self.head_controller.view, fig)
        self.correlation_frame.fig.clf()

        PlotClosureTest(self.correlation_frame.fig, par_names, truths, predictions, prediction_errs)
        self.correlation_frame.SetData()
        self.correlation_frame.Show()


    def Eval(self, obj, evt):
        data = self.left_model.GetData()
        truths = self.right_model.GetData()
        if data.shape[0] > 0:  # emulator_input contains more than the header
            data = data.astype(float).to_numpy()
            truths = truths.astype(float).to_numpy()
            if self.head_controller.file_model.emulator_filename is not None:
                from PlotClosureTest import PlotClosureTestEmulator
                if not self.correlation_frame:
                    fig = Figure((self.config_data['PopUpWidth'], self.config_data['PopUpHeight']), 75)
                    self.correlation_frame = MatplotlibFrame(self.head_controller.view, fig)
                self.correlation_frame.fig.clf()

                PlotClosureTestEmulator(self.head_controller.file_model.trace_filename, self.correlation_frame.fig, data, truths)
                self.correlation_frame.SetData()
                self.correlation_frame.Show()






    def CheckObj(self, func, obj, evt):
        if obj is self:
            func(obj, evt)

