import numpy as np
import wx


########################################################################
class SelectOption(wx.Dialog):

    # ----------------------------------------------------------------------
    def __init__(self):
        """Constructor"""
        wx.Dialog.__init__(
            self,
            None,
            title="Training options",
            style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER,
        )
        panel = wx.Panel(self)
        box = wx.BoxSizer(wx.VERTICAL)

        box1 = wx.BoxSizer(wx.HORIZONTAL)
        cblbl = wx.StaticText(
            panel, -1, label="Covariance function", style=wx.ALIGN_CENTRE
        )
        self.Cov_func = ["ARD", "RBF"]
        self.combo_cov_func = wx.Choice(panel, choices=self.Cov_func)
        self.combo_cov_func.SetSelection(0)
        box1.Add(cblbl, 1, wx.EXPAND | wx.ALIGN_LEFT | wx.ALL, 5)
        box1.Add(self.combo_cov_func, 1, wx.EXPAND | wx.ALIGN_LEFT | wx.ALL, 5)
        box.Add(box1)

        # radio box for 2 different ways of choosing number of PCA components
        box2 = wx.BoxSizer(wx.HORIZONTAL)
        self.rbox = wx.RadioBox(
            panel,
            label="Methods for PCA components",
            choices=["PCA Components", "PCA Fraction"],
            majorDimension=1,
            style=wx.RA_SPECIFY_ROWS,
        )
        self.rbox_choice = "PCA Components"
        box2.Add(self.rbox, 1, wx.EXPAND | wx.ALIGN_CENTRE | wx.ALL, 5)
        box.Add(box2)

        list_textctrl = [
            ("PCA Components", "1"),
            ("Initial scale", "10"),
            ("Initial nugget", "1"),
            ("Scale learning rate", "0.01"),
            ("Nugget learning rate", "0.01"),
            ("Maximum iterations", "10000"),
            ("Gradient threshold", "0.001"),
            ("N.O. test data", "0"),
        ]
        self.output = {}
        self.title = {}

        for (name, default_value) in list_textctrl:
            box_new = wx.BoxSizer(wx.HORIZONTAL)
            text = wx.StaticText(panel, -1, name)
            box_new.Add(text, 1, wx.EXPAND | wx.ALIGN_LEFT | wx.ALL, 5)
            textbox = wx.TextCtrl(panel, -1, default_value)
            box_new.Add(textbox, 1, wx.EXPAND | wx.ALIGN_LEFT | wx.ALL, 5)
            self.title[name] = text
            self.output[name] = textbox
            box.Add(box_new)

        box_submit = wx.BoxSizer(wx.HORIZONTAL)
        self.submit = wx.Button(panel, wx.ID_OK, "Submit")
        box_submit.Add(self.submit, 0, wx.ALIGN_CENTER)
        box.Add(box_submit)

        panel.SetSizer(box)

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(panel)
        self.SetSizer(sizer)
        self.Fit()

        self.Bind(wx.EVT_RADIOBOX, self.OnRadioBox)
        self.Bind(wx.EVT_CLOSE, self.OnClose)

    def OnRadioBox(self, evt):
        self.rbox_choice = self.rbox.GetStringSelection()
        self.title["PCA Components"].SetLabel(self.rbox_choice)

    def OnClose(self, evt):
        self.Destroy()
        return wx.ID_CANCEL

    def AdditionalData(self):
        # args['covariancefunc'] = self.Cov_func[self.combo_cov_func.GetSelection()]
        args = {}
        if self.rbox_choice == "PCA Fraction":
            args["principalcomp"] = None
            args["fraction"] = float(self.output["PCA Components"].GetValue())
        else:
            args["principalcomp"] = int(
                self.output["PCA Components"].GetValue())
            args["fraction"] = None
        args["initialscale"] = np.fromstring(
            self.output["Initial scale"].GetValue(), dtype=np.float, sep=","
        )
        args["initialnugget"] = float(self.output["Initial nugget"].GetValue())
        args["scalerate"] = float(
            self.output["Scale learning rate"].GetValue())
        args["nuggetrate"] = float(
            self.output["Nugget learning rate"].GetValue())
        args["maxsteps"] = int(self.output["Maximum iterations"].GetValue())
        args["gradthreshold"] = float(
            self.output["Gradient threshold"].GetValue())
        args["TestData"] = int(self.output["N.O. test data"].GetValue())
        return args
