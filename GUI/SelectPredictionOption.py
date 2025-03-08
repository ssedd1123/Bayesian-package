import wx


########################################################################
class SelectPredictionOption(wx.Dialog):

    # ----------------------------------------------------------------------
    def __init__(self, *args, **kwargs):
        """Constructor"""
        super().__init__(*args, **kwargs)
        sizer = wx.BoxSizer(wx.VERTICAL)

        self.contBox = wx.RadioBox(
            self,
            label="Are observables contineous?",
            choices=["Yes", "No"],
            majorDimension=1,
            style=wx.RA_SPECIFY_ROWS,
        )
        sizer.Add(self.contBox)

        self.predBox = wx.RadioBox(
            self,
            label="Type of credible interval?",
            choices=["Mean", "Prediction"],
            majorDimension=1,
            style=wx.RA_SPECIFY_ROWS,
        )
        sizer.Add(self.predBox)

        submit = wx.Button(self, wx.ID_OK, "Submit")
        sizer.Add(submit)

        self.SetSizer(sizer)
        self.Fit()

        self.Bind(wx.EVT_CLOSE, self.OnClose)

    def OnClose(self, evt):
        self.Destroy()
        return wx.ID_CANCEL

    def GetValue(self):
        return self.contBox.GetSelection() == 1, self.predBox.GetString(self.predBox.GetSelection())
