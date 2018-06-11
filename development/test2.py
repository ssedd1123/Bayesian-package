import wx

class MyFrame(wx.Frame):
    def __init__(self, *args, **kwds):
        kwds["style"] = wx.DEFAULT_FRAME_STYLE
        wx.Frame.__init__(self, *args, **kwds)
        self.tctrl_1 = wx.TextCtrl(self, -1, "", style=wx.TE_MULTILINE)
        self.tctrl_2 = wx.TextCtrl(self, -1, "", style=wx.TE_MULTILINE)

        self.menubar = wx.MenuBar()
        self.test = wx.Menu()
        self.copy = wx.MenuItem(self.test, wx.NewId(), "copy", "is_going to copy", wx.ITEM_NORMAL)
        self.test.AppendItem(self.copy)
        self.paste = wx.MenuItem(self.test, wx.NewId(), "paste", "will paste", wx.ITEM_NORMAL)
        self.test.AppendItem(self.paste)
        self.menubar.Append(self.test, "Test")
        self.SetMenuBar(self.menubar)

        self.__set_properties()
        self.__do_layout()

        self.Bind(wx.EVT_MENU, self.on_copy, self.copy)
        self.Bind(wx.EVT_MENU, self.on_paste, self.paste)

    def __set_properties(self):
        self.SetTitle("frame_1")

    def __do_layout(self):
        sizer_1 = wx.BoxSizer(wx.VERTICAL)
        sizer_2 = wx.BoxSizer(wx.HORIZONTAL)
        sizer_2.Add(self.tctrl_1, 1, wx.EXPAND, 0)
        sizer_2.Add(self.tctrl_2, 1, wx.EXPAND, 0)
        sizer_1.Add(sizer_2, 1, wx.EXPAND, 0)
        self.SetSizer(sizer_1)
        sizer_1.Fit(self)
        self.Layout()

    def on_copy(self, event): 
        widget = self.FindFocus()
        self.copied = widget.GetStringSelection()

    def on_paste(self, event): 
        widget = self.FindFocus()
        widget.WriteText(self.copied)


if __name__ == "__main__":
    app = wx.PySimpleApp(0)
    frame = MyFrame(None, -1, "")
    frame.Show()
    app.MainLoop()
