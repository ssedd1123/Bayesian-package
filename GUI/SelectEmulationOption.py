import wx
import sys

class SelectEmulationOption(wx.Dialog):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        sizer = wx.BoxSizer(wx.VERTICAL)

        content = {'nevent': ('Events per core', 10000), 'burnin': ('Burn in size', 1000)}
        self.values = {}

        for key, (title, default) in content.items():
            hsizer = wx.BoxSizer(wx.HORIZONTAL)
            title_text = wx.StaticText(self, -1, title)
            value_ctrl = wx.TextCtrl(self, value=str(default))
            hsizer.Add(title_text, 1, wx.EXPAND|wx.ALIGN_LEFT|wx.ALL, 5)
            hsizer.Add(value_ctrl, 1, wx.EXPAND|wx.ALIGN_LEFT|wx.ALL, 5)
            sizer.Add(hsizer) 
            self.values[key] = value_ctrl

        self.button = wx.Button(self, wx.ID_OK, 'Submit')
        sizer.Add(self.button)
        self.SetSizer(sizer)
        self.Fit()

        self.Bind(wx.EVT_CLOSE, self.OnClose)

    def OnClose(self, evt):
        self.Destroy()
        return wx.ID_CANCEL

    def GetValue(self):
        result = {}
        for key, content in self.values.items():
            try:
                result[key] = int(content.GetValue())
            except Exception as e:
                print('Cannot cast result to integers')
                print(e, flush=True)
        return result


