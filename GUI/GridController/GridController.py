import copy
import itertools
import string
from io import StringIO

import numpy as np
import pandas as pd
import wx
import wx.grid as gridlib
from pubsub import pub

from GUI.GridController.GridData import GridData
from GUI.GridController.GridMenu import GridPopupMenu, GridToolBar, PriorToolBar
from GUI.GridController.GridViewer import DataDirectionDialog, MyGrid, PasteSpecialDialog


class GridController:
    def __init__(
            self,
            parent,
            nrows,
            ncols,
            toolbar_type=GridToolBar,
            **kwargs):
        self.model = GridData(nrows, ncols)
        self.view = MyGrid(parent, **kwargs)
        self.view.CreateGrid(nrows, ncols)
        self.view.SetTable(self.model, True)
        self.toolbar = toolbar_type(
            parent=parent,
            id=100,
            style=wx.TB_HORIZONTAL | wx.NO_BORDER | wx.TB_FLAT | wx.TB_TEXT,
        )

        pub.subscribe(self.CheckObj, "viewer_right", func=self.ShowMenu)
        pub.subscribe(self.CheckObj, "Menu_Clear", func=self.Clear)
        pub.subscribe(self.CheckObj, "Menu_Delete", func=self.Delete)
        pub.subscribe(
            self.SelectCellOnKeyEvt,
            "viewer_delete",
            func=self.Clear)
        pub.subscribe(self.CheckObj, "Menu_Copy", func=self.Copy)
        pub.subscribe(self.SelectCellOnKeyEvt, "viewer_CtrlC", func=self.Copy)
        pub.subscribe(self.CheckObj, "Menu_Paste", func=self.Paste)
        pub.subscribe(self.CheckObj, "Menu_PasteSpecial", func=self.PasteSpecial)
        pub.subscribe(self.SelectCellOnKeyEvt, "viewer_CtrlV", func=self.Paste)
        pub.subscribe(self.CheckObj, "Menu_Undo", func=self.Undo)
        pub.subscribe(self.CheckObj, "Menu_Redo", func=self.Redo)

        pub.subscribe(self.CheckObj, "ToolBar_Undo", func=self.Undo)
        pub.subscribe(self.CheckObj, "ToolBar_Redo", func=self.Redo)
        pub.subscribe(self.CheckObj, "ToolBar_Open", func=self.Open)
        pub.subscribe(
            self.CheckObj,
            "ToolBar_ClearContent",
            func=self.ClearAllButHeader)
        pub.subscribe(self.CheckObj, "ToolBar_ClearAll", func=self.ClearAll)
        pub.subscribe(self.CheckObj, "ToolBar_PasteSpecial", func=self.PasteSpecial)

        pub.subscribe(
            self.CheckObj,
            "Data_CanUndo",
            func=self.EnableUndo,
            evt=None)
        pub.subscribe(
            self.CheckObj,
            "Data_CannotUndo",
            func=self.DisableUndo,
            evt=None)
        pub.subscribe(
            self.CheckObj,
            "Data_CanRedo",
            func=self.EnableRedo,
            evt=None)
        pub.subscribe(
            self.CheckObj,
            "Data_CannotRedo",
            func=self.DisableRedo,
            evt=None)
        # pub.subscribe(self.GetSelectedRange, 'Viewer.RangeSelected')
        # pub.subscribe(self.Clear, 'Menu.Clear')

    def CheckObj(self, func, obj, evt):
        orig_obj = obj
        while True:
            if obj is self.model or obj is self.view or obj is self.toolbar:
                func(obj, evt)
                break
            obj = obj.GetParent()
            if obj is None:
                break

    def SelectCellOnKeyEvt(self, func, obj, evt):
        self.GetSelectedCells(obj)
        self.CheckObj(func, obj, evt)

    def Open(self, obj, evt):
        dlg = wx.FileDialog(
            obj,
            message="Choose a file",
            defaultFile="",
            style=wx.FD_OPEN | wx.FD_MULTIPLE | wx.FD_CHANGE_DIR,
        )
        result = dlg.ShowModal()
        path = dlg.GetPaths()
        dlg.Destroy()

        if result != wx.ID_OK:
            return False

        df = pd.read_csv(path[0], index_col=0)
        if not set(df.index).issubset(set(self.model.data.index)):
            wx.MessageBox(
                "Index of the dataset does not agree. Will load index as if it's content",
                "Warning",
                wx.OK | wx.ICON_WARNING,
            )
            self.model.SetData(df, include_index=True)
        else:
            self.model.SetData(df, include_index=False)

    def EnableUndo(self, obj, evt):
        self.toolbar.EnableTool(wx.ID_UNDO, True)

    def DisableUndo(self, obj, evt):
        self.toolbar.EnableTool(wx.ID_UNDO, False)

    def EnableRedo(self, obj, evt):
        self.toolbar.EnableTool(wx.ID_REDO, True)

    def DisableRedo(self, obj, evt):
        self.toolbar.EnableTool(wx.ID_REDO, False)

    def Redo(self, obj, evt):
        self.model.Redo()
        self.view.ForceRefresh()

    def Undo(self, obj, evt):
        self.model.Undo()
        self.view.ForceRefresh()

    def CanBePasted(self, new_data_nrows=0, new_data_ncols=0):
        for row in range(
                self.view.GetGridCursorRow(),
                self.view.GetGridCursorRow() +
                new_data_nrows):
            for col in range(
                    self.view.GetGridCursorCol(),
                    self.view.GetGridCursorCol() +
                    new_data_ncols):  # self.selected_cols:
                if (
                    self.view.IsReadOnly(row, col)
                    or not self.view.CanEnableCellControl()
                ):
                    # break out of both loops once a read only cell is found
                    return False
        return True

    def PasteSpecial(self, obj, evt):
        dlg = PasteSpecialDialog(None)
        dlg.ShowModal()
        self.Paste(obj, evt, dlg.delimiter)
        

    def Paste(self, obj, evt, delimiter='\t'):
        # self.GetSelectedCells(obj)
        # if self.CanBeModified():
        dataObj = wx.TextDataObject()
        if wx.TheClipboard.Open():
            wx.TheClipboard.GetData(dataObj)
            wx.TheClipboard.Close()
        else:
            wx.MessageBox("Can't open the clipboard", "Error")
        string = dataObj.GetText()
        data = pd.DataFrame([line.split(delimiter)
                             for line in string.rstrip().split("\n")])
        # data = pd.read_csv(StringIO(string), sep='\t', header=None)
        row = self.view.GetGridCursorRow()
        col = self.view.GetGridCursorCol()

        if self.CanBePasted(data.shape[0], data.shape[1]):
            rows = np.arange(row, row + data.shape[0])
            cols = np.arange(col, col + data.shape[1])
            self.model.SetValue(rows, cols, data)
            self.view.ForceRefresh()

    def Copy(self, obj, evt):
        #row = self.view.GetGridCursorRow()
        #col = self.view.GetGridCursorCol()
        # self.GetSelectedCells(obj)

        data = self.model.GetValue(self.selected_rows, self.selected_cols)
        if isinstance(data, pd.DataFrame):
            data = data.to_csv(header=None, index=False, sep="\t").strip("\n")

        dataObj = wx.TextDataObject()
        dataObj.SetText(str(data))

        if wx.TheClipboard.Open():
            wx.TheClipboard.SetData(dataObj)
            wx.TheClipboard.Flush()
            wx.TheClipboard.Close()
        else:
            wx.MessageBox("Unable to open the clipboard", "Error")

    def GetSelectedCells(self, obj):
        cell = obj.GetSelectedCells()
        block = obj.GetSelectionBlockTopLeft()
        rows = obj.GetSelectedRows()
        cols = obj.GetSelectedCols()
        if block:
            topleft = obj.GetSelectionBlockTopLeft()[0]
            bottomright = obj.GetSelectionBlockBottomRight()[0]
            self.selected_rows = np.arange(topleft[0], bottomright[0] + 1)
            self.selected_cols = np.arange(topleft[1], bottomright[1] + 1)
        elif cols:
            self.selected_cols = np.arange(cols[0], cols[-1] + 1)
            self.selected_rows = np.arange(0, obj.GetNumberRows())
        elif rows:
            self.selected_cols = np.arange(0, obj.GetNumberCols())
            self.selected_rows = np.arange(rows[0], rows[-1] + 1)
        else:
            self.selected_rows = [self.view.GetGridCursorRow()]
            self.selected_cols = [self.view.GetGridCursorCol()]

    def ShowMenu(self, obj, evt):
        self.GetSelectedCells(obj)
        menu = GridPopupMenu(obj)

        for row in self.selected_rows:
            for col in self.selected_cols:
                if (
                    self.view.IsReadOnly(row, col)
                    or not self.view.CanEnableCellControl()
                ):
                    menu.paste.Enable(False)
                    menu.delete.Enable(False)
                    menu.clear.Enable(False)
                    # break out of both loops once a read only cell is found
                    break
            else:
                continue
            break

        # check if selected range contains read only cell, if so paste, clear
        # and delete will be disabled
        obj.PopupMenu(menu, evt.GetPosition())

    def Clear(self, obj, evt):
        self.model.SetValue(self.selected_rows, self.selected_cols, None)

    def ClearAll(self, obj=None, evt=None):
        self.model.SetValue(list(range(self.model.data.shape[0])),
                            list(range(self.model.data.shape[1])), None)

    def ClearAllButHeader(self, obj=None, evt=None):
        self.model.SetValue(list(range(1, self.model.data.shape[0])),
                            list(range(self.model.data.shape[1])), None)
        self.view.Refresh()

    def Delete(self, obj, evt):
        dlg = DataDirectionDialog(None, "Data direction")
        direction = dlg.ShowModal()
        self.model.DeleteShift(
            self.selected_rows,
            self.selected_cols,
            direction)
        dlg.Destroy()
        self.view.ForceRefresh()


class PriorController(GridController):
    def __init__(self, parent, ncols, **kwargs):
        nrows = 6  # All rows: Name, Type, Low, Up, Mean, SD
        self.nrows = nrows
        self.ncols = ncols
        super().__init__(parent, nrows, ncols, toolbar_type=PriorToolBar, **kwargs)

        var_types = ["Uniform", "Gaussian"]
        self.model.data.index = ["Name", "Type", "Min", "Max", "Mean", "SD"]
        for i in range(ncols):
            choice_editor = gridlib.GridCellChoiceEditor(var_types)
            self.view.SetCellEditor(1, i, choice_editor)

        for i in range(2, nrows):
            attr = gridlib.GridCellAttr()
            attr.SetReadOnly(True)
            attr.SetBackgroundColour("Grey")
            self.view.SetRowAttr(i, attr)

        pub.subscribe(self.CheckObj, "Data_Changed", func=self.EnableRow)

        sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.model_name_text = wx.StaticText(parent, label='Class Name: ')
        sizer.Add(self.model_name_text, wx.RIGHT, border=7)
        self.model_name_view = sizer

    def SetModelName(self, name):
        if name is None:
            name = ''
        self.model_name_text.SetLabel('Class Name: ' + name)

    def EnableRow(self, obj, evt):
        rows = evt[0]  # .GetRow()
        cols = evt[1]  # .GetCol()
        if 1 in rows:
            for i in range(2, 4):
                for col in range(self.ncols):
                    self.view.SetReadOnly(i, col, False)
                    self.view.SetCellBackgroundColour(i, col, "White")
                    self.view.SetCellTextColour(i, col, "Black")
            for col in range(self.ncols):
                if self.model.GetValue(1, col) != "Gaussian":
                    readOnly = True
                    bkcolor = "Grey"
                    textcolor = "Grey"
                else:
                    readOnly = False
                    bkcolor = "White"
                    textcolor = "Black"
                for i in range(4, self.nrows):
                    self.view.SetReadOnly(i, col, readOnly)
                    self.view.SetCellBackgroundColour(i, col, bkcolor)
                    self.view.SetCellTextColour(i, col, textcolor)

                    # self.model.SetValue(i, col, None)
        self.view.Refresh()

    def FillPriorRange(self, mins, maxs):
        existing_var = self.model.GetData(False)
        # append empty rows if the existing_var has less than 4 rows
        for i in range(existing_var.shape[0], 4):
            existing_var = existing_var = existing_var.append(
                pd.Series(), ignore_index=True)

        # any undefined variable types will be defaulted to uniform
        existing_var.iloc[1].fillna(value='Uniform', inplace=True)
        # fill empty numbers
        id_empty = pd.isna(existing_var.iloc[2])
        existing_var.iloc[2][id_empty] = mins[id_empty]
        id_empty = pd.isna(existing_var.iloc[3])
        existing_var.iloc[3][id_empty] = maxs[id_empty]

        self.model.SetData(existing_var, False)


class ScrollSync(wx.EvtHandler):
    def __init__(self, grid1, grid2):
        super(ScrollSync, self).__init__()
        self.grid1 = grid1
        self.grid2 = grid2
        self.grid1ScrollPos = self.getGrid1Pos()
        self.grid2ScrollPos = self.getGrid2Pos()
        self.Bind(wx.EVT_TIMER, self.onTimer)
        self.timer = wx.Timer(self)
        self.timer.Start(20)

    def onTimer(self, event):
        if not self.grid1 or not self.grid2:
            self.timer.Stop()
            return
        if self.grid1ScrollPos != self.getGrid1Pos():
            self.grid1ScrollPos = self.getGrid1Pos()
            self.grid2.Scroll(self.grid1ScrollPos)
        elif self.grid2ScrollPos != self.getGrid2Pos():
            self.grid2ScrollPos = self.getGrid2Pos()
            self.grid1.Scroll(self.grid2ScrollPos)

    def getGrid1Pos(self):
        vertical = self.grid1.GetScrollPos(wx.SB_VERTICAL)
        return -1, vertical

    def getGrid2Pos(self):
        vertical = self.grid2.GetScrollPos(wx.SB_VERTICAL)
        return -1, vertical


class SplitViewController:
    def __init__(self, parent, nrows=100, nlayers=100):
        self.view = wx.SplitterWindow(parent)
        left_panel = wx.Panel(self.view)
        right_panel = wx.Panel(self.view)
        self.controller_right = GridController(right_panel, nrows, nlayers)
        grid_sizer = wx.BoxSizer(wx.VERTICAL)
        grid_sizer.Add(self.controller_right.toolbar, 0, wx.EXPAND)
        grid_sizer.Add(self.controller_right.view, 1, wx.EXPAND)
        self.controller_right.view.SetRowLabelSize(5)
        right_panel.SetSizer(grid_sizer)

        self.controller_left = GridController(left_panel, nrows, nlayers)
        grid_sizer = wx.BoxSizer(wx.VERTICAL)
        grid_sizer.Add(self.controller_left.toolbar, 0, wx.EXPAND)
        grid_sizer.Add(self.controller_left.view, 1, wx.EXPAND)
        left_panel.SetSizer(grid_sizer)

        self.view.SplitVertically(
            left_panel, right_panel, wx.ScreenDC().GetPPI()[0] * 6
        )

        # assign variable name for easier access
        self.left_view = self.controller_left.view
        self.left_model = self.controller_left.model
        self.right_view = self.controller_right.view
        self.right_model = self.controller_right.model

        # link-up the behavior of the two panels
        # new sync function based on timer instead of event is used to handle mouse wheel events
        # old version won't sync when it is scrolled with mouse wheel or page
        # up/down
        ScrollSync(self.left_view, self.right_view)
        # self._SyncScrollPos()

    # def _SyncScrollPos(self):
    #    self.left_view.Bind(wx.EVT_SCROLLWIN, self._onScrollL)
    #    self.right_view.Bind(wx.EVT_SCROLLWIN, self._onScrollR)

    # def _onScrollL(self, evt):
    #    if evt.Orientation == wx.SB_VERTICAL:
    #        self.right_view.Scroll(-1, evt.Position)
    #    evt.Skip()

    # def _onScrollR(self, evt):
    #    if evt.Orientation == wx.SB_VERTICAL:
    #        self.left_view.Scroll(-1, evt.Position)
    #    evt.Skip()


class TestFrame(wx.Frame):
    def __init__(self, parent, title, size, prior=False):
        super().__init__(parent, title=title)
        panel = wx.Panel(self, -1)
        grid = self.create_grid(panel, size, prior)
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.controller.toolbar)
        sizer.Add(grid, wx.EXPAND)
        panel.SetSizer(sizer)

        # Bind Close Event
        # wx.EVT_CLOSE(self, lambda self, event: self.Destroy())
        self.Center()
        self.Show()

    def create_grid(self, panel, size, prior):
        if prior:
            self.controller = PriorController(panel, size[1])
        else:
            self.controller = GridController(panel, size[0], size[1])
        return self.controller.view


if __name__ == "__main__":
    app = wx.App()
    frame1 = TestFrame(None, "test", size=(10, 10))
    frame2 = TestFrame(None, "prior", size=(10, 10), prior=True)
    app.MainLoop()
