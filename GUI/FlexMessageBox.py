import wx
from wx.lib import sized_controls
from wx.lib.expando import ExpandoTextCtrl, EVT_ETC_LAYOUT_NEEDED


class FlexMessageBox(sized_controls.SizedDialog):

    def __init__(self, message, *args, **kwargs):
        super(FlexMessageBox, self).__init__(*args, **kwargs)
        sizer = wx.BoxSizer(wx.VERTICAL)
        panel = wx.Panel(self)

        #self.lbl = wx.StaticText(panel, -1, style=wx.ALIGN_LEFT)
        # find width required
        font = wx.Font(
            12,
            wx.FONTFAMILY_TELETYPE,
            wx.NORMAL,
            wx.NORMAL,
            faceName="Monospace")
        dc = wx.ScreenDC()
        dc.SetFont(font)
        w, h = dc.GetTextExtent(max(message.split('\n'), key=len))

        margin = 15
        self.lbl = ExpandoTextCtrl(panel, -1, style=wx.ALIGN_LEFT|wx.TE_READONLY|wx.NO_BORDER, size=(w + margin, -1))
        self.lbl.SetBackgroundColour(panel.GetBackgroundColour())
        self.lbl.SetFont(font)
        sizer.Add(self.lbl, 1, wx.EXPAND | wx.ALL, 25)
        self.lbl.WriteText(message)

        button_ok = wx.Button(panel, wx.ID_OK, label='OK')
        button_ok.Bind(wx.EVT_BUTTON, self.on_button)
        button_cancel = wx.Button(panel, wx.ID_CANCEL, label='Cancel')
        button_cancel.Bind(wx.EVT_BUTTON, self.on_button)

        hs = wx.BoxSizer(wx.HORIZONTAL)
        hs.Add(button_cancel, 0, wx.ALIGN_RIGHT, 5)
        hs.Add(button_ok, 0, wx.ALIGN_RIGHT, 5)
        sizer.Add(hs)
        self.Bind(wx.EVT_CLOSE, self.on_close)

        panel.SetSizer(sizer)
        panel.Fit()

        frameSizer = wx.BoxSizer()
        frameSizer.Add(panel, 0, wx.EXPAND)
        self.SetSizer(frameSizer)
        self.Fit()

    def on_button(self, event):
        if self.IsModal():
            self.EndModal(event.EventObject.Id)
        else:
            self.Destroy()

    def on_close(self, event):
        # for when the dialog is not being shown with showModal
        if self.IsModal():
            self.EndModal(event.EventObject.Id)
        self.Destroy()



if __name__ == '__main__':
    app = wx.App(False)
    dlg = FlexMessageBox(
        'asdfasdfasdf\nasdfasdfasdf',
        None,
        title='Custom Dialog')
    result = dlg.ShowModal()
    dlg.Destroy()
    app.MainLoop()
