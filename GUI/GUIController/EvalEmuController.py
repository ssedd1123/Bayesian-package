import wx
import wx.grid as gridlib
from pubsub import pub
import numpy as np

from Utilities.Utilities import GetTrainedEmulator
from GUI.GridController.GridController import SplitViewController

class EvalEmuController(SplitViewController):
    def __init__(self, panel, config_data):
        super().__init__(panel, config_data['GridNRow'], config_data['GridNCol'], no_clear_all=True, min_size=config_data['MinPanelSize'])

        self.right_view.SetDefaultCellBackgroundColour(
            "Grey")
        # disable edition in all cells in this panel
        # this panel is only meant to output data, not for editing
        self.right_view.EnableEditing(False)
        attr = gridlib.GridCellAttr()
        # first row is reserved for header
        attr.SetReadOnly(True)
        attr.SetBackgroundColour("Grey")
        self.left_view.SetRowAttr(0, attr)

        EvalEmuButton = wx.Button(
            panel, -1, "Evaluate emulator")
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(EvalEmuButton)
        EvalEmuButton.Bind(
            wx.EVT_BUTTON,
            lambda evt: pub.sendMessage("EvalEmu_Eval", obj=self, evt=evt),
        )
        sizer.Add(self.view, 1, wx.EXPAND)
        panel.SetSizer(sizer)

        pub.subscribe(self.CheckObj, "EvalEmu_Eval", func=self.Eval)
        pub.subscribe(self._SyncHeaders, "Data_Changed")

    def CheckObj(self, func, obj, evt):
        if obj is self:
            func(obj, evt)

    def SetHeadController(self, head_controller):
        self.head_controller = head_controller

    def _SyncHeaders(self, obj, evt):
        # Remember, headers cannot be changed in this plugin
        if obj in [self.head_controller.prior_model, self.head_controller.model_par_model]:
            value = obj.data.iloc[0].replace(r"^\s*$", np.nan, regex=True)
            self.left_model.ChangeValues(
                0,
                np.arange(
                    value.shape[0]),
                value,
                send_changed=False)
        elif obj in [self.head_controller.exp_model, self.head_controller.model_obs_model]:
            value = (
                obj.data.iloc[0]
                .replace(r"^\s*$", np.nan, regex=True)
                .dropna(how="all")
            )
            value = np.append(
                value,
                ["%s_Err" % val for val in value]
                + [None for i in range(obj.data.shape[1])],
            )
            self.right_model.ChangeValues(
                0, np.arange(value.shape[0]), value, send_changed=False
            )  # Error is added to the header of emulator output

    def Eval(self, obj, evt):
        data = self.left_model.GetData()
        if data.shape[0] > 0:  # emulator_input contains more than the header
            np_data = data.astype(float).to_numpy()
            if self.head_controller.file_model.emulator_filename is not None:
                clf = GetTrainedEmulator(self.head_controller.file_model.emulator_filename)[0]
                for idx, row in enumerate(np_data):
                    if not np.isnan(row).any():
                        prediction, cov = clf.Predict(row)
                        prediction = np.atleast_1d(np.squeeze(prediction))
                        dim = int(prediction.shape[0])
                        cov = np.atleast_1d(np.squeeze(cov))
                        self.right_model.ChangeValues(
                            idx + 1,
                            np.arange(dim),
                            prediction,
                            send_changed=False,
                        )  # y-index add one to not overwrite header
                        self.right_model.ChangeValues(
                            idx + 1,
                            np.arange(
                                dim,
                                2 * dim),
                            np.sqrt(
                                np.diag(cov)),
                            send_changed=False,
                        )
                        self.right_view.Refresh()


    def Save(self):
        pass

    def SaveNew(self, filename):
        pass

    def LoadFile(self, filename):
        pass
