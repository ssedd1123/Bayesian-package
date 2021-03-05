import wx
from wx.lib import sized_controls


class FlexMessageBox(wx.Dialog):  # sized_controls.SizedDialog):

    def __init__(self, message, *args, **kwargs):
        super(FlexMessageBox, self).__init__(*args, **kwargs)
        sizer = wx.BoxSizer(wx.VERTICAL)
        panel = wx.Panel(self)

        lbl = wx.StaticText(panel, -1, style=wx.ALIGN_LEFT)
        font = wx.Font(
            12,
            wx.FONTFAMILY_TELETYPE,
            wx.NORMAL,
            wx.NORMAL,
            faceName="Monospace")
        lbl.SetFont(font)
        lbl.SetLabel(message)
        sizer.Add(lbl, 0, wx.EXPAND | wx.ALL, 25)

        button_ok = wx.Button(panel, wx.ID_OK, label='OK')
        button_ok.Bind(wx.EVT_BUTTON, self.on_button)
        button_cancel = wx.Button(panel, wx.ID_CANCEL, label='Cancel')
        button_cancel.Bind(wx.EVT_BUTTON, self.on_button)

        hs = wx.BoxSizer(wx.HORIZONTAL)
        hs.Add(button_cancel, 0, wx.ALIGN_RIGHT, 5)
        hs.Add(button_ok, 0, wx.ALIGN_RIGHT, 5)
        sizer.Add(hs)
        self.Bind(wx.EVT_CLOSE, self.on_button)

        panel.SetSizerAndFit(sizer)
        self.Fit()

    def on_button(self, event):
        if self.IsModal():
            self.EndModal(event.EventObject.Id)
        else:
            self.Close()


if __name__ == '__main__':
    app = wx.App(False)
    dlg = FlexMessageBox(
        'asdfasdfasdf\nasdfasdfasdf',
        None,
        title='Custom Dialog')
    result = dlg.ShowModal()
    dlg.Destroy()
    app.MainLoop()
