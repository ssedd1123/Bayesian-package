import wx

GAP = 12


class VideoSlider(wx.Slider):
    def __init__(self, gap, *args, **kwargs):
        wx.Slider.__init__(self, *args, **kwargs)
        self.gap = gap
        self.Bind(wx.EVT_LEFT_DOWN, self.OnClick)

    def linapp(self, x1, x2, y1, y2, x):
        return (float(x - x1) / (x2 - x1)) * (y2 - y1) + y1

    def OnClick(self, e):
        click_min = self.gap
        click_max = self.GetSize()[0] - self.gap
        click_position = e.GetX()
        result_min = self.GetMin()
        result_max = self.GetMax()
        if click_position > click_min and click_position < click_max:
            result = self.linapp(
                click_min, click_max, result_min, result_max, click_position
            )
        elif click_position <= click_min:
            result = result_min
        else:
            result = result_max
        self.SetValue(result)
        e.Skip()


class MainWindow(wx.Frame):
    def __init__(self, *args, **kwargs):
        wx.Frame.__init__(self, *args, **kwargs)

        self.panel = wx.Panel(self)
        self.slider = VideoSlider(parent=self.panel, size=(300, -1), gap=GAP)
        self.slider.Bind(wx.EVT_SLIDER, self.OnSlider)

        self.sizer = wx.BoxSizer()
        self.sizer.Add(self.slider)

        self.panel.SetSizerAndFit(self.sizer)
        self.Show()

    def OnSlider(self, e):
        print(self.slider.GetValue())


app = wx.App(False)
win = MainWindow(None)
app.MainLoop()
