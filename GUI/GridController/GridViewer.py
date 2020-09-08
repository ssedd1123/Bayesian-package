import wx
import wx.grid as gridlib
from pubsub import pub

from GUI.GridController.GridData import Direction, HistoryType


class MyGrid(gridlib.Grid):
    def __init__(self, parent, size, initial_df=None, **kwargs):
        if initial_df is not None:
            size = initial_df.shape
        super().__init__(parent, -1)
        self.SetDefaultColSize(wx.ScreenDC().GetPPI()[1] * 1.2, True)
        self.SetRowLabelSize(wx.ScreenDC().GetPPI()[1] * 1.2)
        self.parent = parent
        # self.table = GridData(num_row=size[0], num_col=size[1], df=initial_df)

        self.Bind(gridlib.EVT_GRID_CELL_RIGHT_CLICK, self.RightClick)
        self.Bind(
            gridlib.EVT_GRID_RANGE_SELECT,
            lambda evt: pub.sendMessage("Viewer_RangeSelected", obj=self, evt=evt),
        )
        # self.Bind(gridlib.EVT_GRID_CELL_CHANGED, lambda evt: pub.sendMessage('Data_Changed', obj=self.table))

    def RightClick(self, evt):
        pub.sendMessage("viewer_right", obj=self, evt=evt)


class DataDirectionDialog(wx.Dialog):
    def __init__(self, parent, title):
        super().__init__(parent, title=title, size=(110, 110), style=wx.RESIZE_BORDER)
        panel = wx.Panel(self)
        box = wx.BoxSizer(wx.VERTICAL)

        self.rb_left = wx.RadioButton(
            panel, label="Move left", pos=(10, 10), style=wx.RB_GROUP
        )
        self.rb_right = wx.RadioButton(panel, label="Moeve right", pos=(10, 20))
        self.rb_top = wx.RadioButton(panel, label="Move Up", pos=(10, 30))
        self.rb_bottom = wx.RadioButton(panel, label="Move Down", pos=(10, 40))
        self.confirm_button = wx.Button(
            panel, label="Confirm", pos=(10, 50), size=(100, 200)
        )

        box.Add(self.rb_left)
        box.Add(self.rb_right)
        box.Add(self.rb_top)
        box.Add(self.rb_bottom)
        box.Add(self.confirm_button)
        panel.SetSizer(box)

        self.confirm_button.Bind(wx.EVT_BUTTON, self.OnConfirm)

    def OnConfirm(self, evt):
        if self.rb_left.GetValue():
            direction = Direction.Left
        elif self.rb_right.GetValue():
            direction = Direction.Right
        elif self.rb_top.GetValue():
            direction = Direction.Up
        elif self.rb_bottom.GetValue():
            direction = Direction.Down
        else:
            direction = None
        if direction is not None:
            self.EndModal(direction)
