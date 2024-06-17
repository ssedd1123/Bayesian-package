import wx

class SelectHyperCubeOption(wx.Dialog):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        content = {
            "NPts": ("Number of parameter sets", 50),
            "Pad": ("Padding fraction", 0),
        }
        self.values = {}
        sizer = wx.GridBagSizer(
            len(content), 5
        )

        for i, (key, (title, default)) in enumerate(content.items()):
            title_text = wx.StaticText(self, -1, title)
            value_ctrl = wx.TextCtrl(self, value=str(default))
            sizer.Add(
                title_text,
                pos=(i, 0),
                flag=wx.EXPAND | wx.ALIGN_LEFT | wx.ALL,
                border=5,
            )
            sizer.Add(
                value_ctrl,
                pos=(i, 1),
                span=(1, 4),
                flag=wx.EXPAND | wx.ALIGN_LEFT | wx.ALL,
            )
            self.values[key] = value_ctrl

        self.button = wx.Button(self, wx.ID_OK, "Enter")
        sizer.Add(
            self.button,
            pos=(
                len(content),
                0),
            span=(
                1,
                5),
            flag=wx.ALIGN_LEFT)

        self.SetSizer(sizer)
        self.Fit()

        self.Bind(wx.EVT_CLOSE, self.OnClose)

    def OnClose(self, evt):
        self.Destroy()
        return wx.ID_CANCEL

    def GetValue(self):
        pad = float(self.values["Pad"].GetValue())
        if pad < 0 or pad > 1:
            raise RuntimeError('Pad value must be between 0 - 1')

        result = {
            "NPts": int(self.values["NPts"].GetValue()),
            "Pad": pad,
        }
        return result

