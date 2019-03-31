import wx
from pubsub import pub

class CustomSlider(wx.Control):

    def __init__(self, parent, value, minValue, maxValue, height=40, pads=20, 
                 title='tesdfst', title_pad=10, **kwargs):
        self.height = height
        self.pads = pads
        self.title = title
        self.title_pad = title_pad
        super().__init__(parent, -1, size=(-1,height), style=wx.NO_BORDER, **kwargs)
        self.SetBackgroundColour(parent.GetBackgroundColour())
        self.parent = parent
        self.minValue = minValue
        self.maxValue = maxValue
        self.highlight = value
        self.current_value = value

        self.initBuffer()
        self.Bind(wx.EVT_PAINT, self.OnPaint)
        self.Bind(wx.EVT_LEFT_DOWN, self.OnPressed)
        self.Bind(wx.EVT_MOTION, self.OnPressed)
        self.Bind(wx.EVT_SIZE, self.OnSize)
        self.Centre()

    def OnPaint(self, event):
        dc = wx.BufferedPaintDC(self, self.buffer)

    def OnSize(self, evt):
        self.initBuffer()

    def ValueToCoord(self, val, xnum_pixel):
        return (val - self.minValue)/(self.maxValue - self.minValue)*(xnum_pixel - 2*self.pads) + self.pads

    def CoordToValue(self, coord, xnum_pixel):
        return (coord - self.pads)/(xnum_pixel - 2*self.pads)*(self.maxValue - self.minValue) + self.minValue

    def Highlight(self, value):
        self.highlight = value
        dc = wx.BufferedDC(wx.ClientDC(self), self.buffer)
        dc.Clear()
        self.DrawLine(dc)

    def initBuffer(self):
        ''' Initialize the bitmap used for buffering the display. '''
        size = self.GetSize()# self.parent.ClientSize
        if size[0] == 0:
            size[0] = 1
        if size[1] == 0:
            size[1] = 1
        self.buffer = wx.Bitmap(size.width, size.height)
        dc = wx.BufferedDC(None, self.buffer)
        dc.Clear()
        self.DrawLine(dc)
 
    def DrawLine(self, dc):
        width = 10
        height = 10
        radius = 7

        dc.SetDeviceOrigin(0, 0)
        font = wx.Font(pointSize = 15, family = wx.DEFAULT, 
                       style = wx.NORMAL, weight = wx.NORMAL)
        dc.SetFont(font)
        tw, th = dc.GetTextExtent(self.title)
        dc.DrawText(self.title, self.title_pad, (self.height-th)/2)
        dc.SetBackground(wx.Brush(self.GetBackgroundColour()))
        if self.pads - radius - self.title_pad < tw:
            self.pads = tw + radius + self.title_pad

        #dc.SetAxisOrientation(True, True)
        size = dc.GetSize()
        center = self.ValueToCoord(self.highlight, size[0])
        dc.SetBrush(wx.Brush('#777'))
        dc.SetPen(wx.Pen("#777"))
        dc.DrawPolygon(((center-width/2, 0), (center, height), (center+width/2, 0)))

        # draw slider line
        dc.DrawLine(self.pads, self.height/2, dc.GetSize()[0]-self.pads, self.height/2)

        # draw slider thumb
        dc.SetPen(wx.Pen(wx.Colour('orange')))
        dc.SetBrush(wx.Brush(wx.Colour('orange')))
        coord = self.ValueToCoord(self.current_value, dc.GetSize()[0])
        dc.DrawCircle(coord, self.height/2, radius)
        
        value_text = '%.2f' % self.current_value
        tw, th = dc.GetTextExtent(value_text)
        dc.DrawText(value_text, coord+radius, self.height-th)

    def OnPressed(self, evt):
        if evt.Dragging() or evt.LeftIsDown():
            dc = wx.BufferedDC(wx.ClientDC(self), self.buffer)
            current_value = self.CoordToValue(evt.GetPosition()[0], dc.GetSize()[0])
            if self.minValue <= current_value <= self.maxValue:
                self.current_value = current_value
                dc.Clear()
                self.DrawLine(dc)
                self.Refresh()
                pub.sendMessage('Slider_Value', obj=self, evt=self.current_value)

    def SetValue(self, val):
       dc = wx.BufferdDC(wx.ClientDC(self), self.buffer)
       if self.minValue <= val <= self.maxValue:
           self.current_value = val
           dc.Clear()
           self.DrawLine(dc)
           self.Refresh()

class Example(wx.Frame):

    def __init__(self, *args, **kw):
        super(Example, self).__init__(*args, **kw)
        sizer = wx.BoxSizer(wx.VERTICAL)
        panel = wx.Panel(self)

        self.highlighter1 = CustomSlider(panel, 5, 0, 10)
        self.highlighter2 = CustomSlider(panel, 1, 1, 8) 

        sizer.Add(self.highlighter1, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 20)
        sizer.Add(self.highlighter2, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 20)
        panel.SetSizer(sizer)
        self.Fit()
        self.Layout()
        self.Centre()

def main():

    app = wx.App()
    ex = Example(None)
    wx.CallLater(5000, ex.highlighter1.Highlight, 3)
    ex.Show()
    app.MainLoop()


if __name__ == '__main__':
    main()
