from io import StringIO
import copy
import pandas as pd
import numpy as np
import string
import itertools
from pubsub import pub
import wx
import wx.grid as gridlib


from GUI.GridController.GridViewer import MyGrid, DataDirectionDialog
from GUI.GridController.GridData import GridData
from GUI.GridController.GridMenu import GridPopupMenu, GridToolBar

class GridController:

    def __init__(self, parent, nrows, ncols, **kwargs):
        self.model = GridData(nrows, ncols)
        self.view = MyGrid(parent, self.model.data)
        self.view.CreateGrid(nrows, ncols)
        self.view.SetTable(self.model, True)
        self.toolbar = GridToolBar(parent=parent, id=100, style=wx.TB_HORIZONTAL | wx.NO_BORDER | \
                                                                   wx.TB_FLAT | wx.TB_TEXT)

        pub.subscribe(self.CheckObj, 'viewer_right', func=self.ShowMenu)   
        pub.subscribe(self.CheckObj, 'Menu_Clear', func=self.Clear)      
        pub.subscribe(self.CheckObj, 'Menu_Delete', func=self.Delete)     
        pub.subscribe(self.CheckObj, 'Menu_Paste', func=self.Paste)      
        pub.subscribe(self.CheckObj, 'Menu_Undo', func=self.Undo)       
        pub.subscribe(self.CheckObj, 'Menu_Redo', func=self.Redo)       
                                                     
        pub.subscribe(self.CheckObj, 'ToolBar_Undo', func=self.Undo)       
        pub.subscribe(self.CheckObj, 'ToolBar_Redo', func=self.Redo)       
        pub.subscribe(self.CheckObj, 'Data_CanUndo', func=self.EnableUndo, evt=None) 
        pub.subscribe(self.CheckObj, 'Data_CannotUndo', func=self.DisableUndo, evt=None)
        pub.subscribe(self.CheckObj, 'Data_CanRedo', func=self.EnableRedo, evt=None) 
        pub.subscribe(self.CheckObj, 'Data_CannotRedo', func=self.DisableRedo, evt=None)
        #pub.subscribe(self.GetSelectedRange, 'Viewer.RangeSelected')
        #pub.subscribe(self.Clear, 'Menu.Clear')
    def CheckObj(self, func, obj, evt):
        orig_obj = obj
        while True:
            if obj is self.model or obj is self.view or obj is self.toolbar:
                func(obj, evt)
                break
            obj = obj.GetParent()
            if obj is None:
                break

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

    def Paste(self, obj, evt):
        dataObj = wx.TextDataObject()
        if wx.TheClipboard.Open():
             wx.TheClipboard.GetData(dataObj)
             wx.TheClipboard.Close()
        else:
             wx.MessageBox("Can't open the clipboard", "Error")
        string = dataObj.GetText()
        data = pd.DataFrame([line.split('\t') for line in string.rstrip().split('\n')])
        #data = pd.read_csv(StringIO(string), sep='\t', header=None)
        row = self.view.GetGridCursorRow()
        col = self.view.GetGridCursorCol()

        rows = np.arange(row, row + data.shape[0])
        cols = np.arange(col, col + data.shape[1])
        self.model.SetValue(rows, cols, data)
        self.view.ForceRefresh()        


    def ShowMenu(self, obj, evt):
        cell = obj.GetSelectedCells()
        block = obj.GetSelectionBlockTopLeft()
        rows = obj.GetSelectedRows()
        cols = obj.GetSelectedCols()
        if block:
            topleft = obj.GetSelectionBlockTopLeft()[0]
            bottomright = obj.GetSelectionBlockBottomRight()[0]
            self.selected_rows = np.arange(topleft[0], bottomright[0]+1)
            self.selected_cols = np.arange(topleft[1], bottomright[1]+1)
        elif cols:
            self.selected_cols = np.arange(cols[0], cols[-1]+1)
            self.selected_rows = np.arange(0, obj.GetNumberRows())
        elif rows:
            self.selected_cols = np.arange(0, obj.GetNumberCols())
            self.selectedrows = np.arange(rows[0], rows[-1]+1)
        else:
            self.selected_rows = evt.GetRow()
            self.selected_cols = evt.GetCol()
        obj.PopupMenu(GridPopupMenu(obj),evt.GetPosition()) 

    def Clear(self, obj, evt):
        self.model.SetValue(self.selected_rows, self.selected_cols, None)

    def Delete(self, obj, evt):
        dlg = DataDirectionDialog(None, 'Data direction')
        direction = dlg.ShowModal()
        self.model.DeleteShift(self.selected_rows, self.selected_cols, direction)       
        dlg.Destroy()
        self.view.ForceRefresh()

class PriorController(GridController):

    def __init__(self, parent, ncols, **kwargs):
        nrows = 6 # All rows: Name, Type, Low, Up, Mean, SD
        self.nrows = nrows
        super().__init__(parent, nrows, ncols, **kwargs)
        
        var_types = ['Uniform', 'Gaussian']
        self.model.data.index = ['Name', 'Type', 'Min', 'Max', 'Mean', 'SD']
        for i in range(ncols):
            choice_editor = gridlib.GridCellChoiceEditor(var_types)
            self.view.SetCellEditor(1, i, choice_editor)

        for i in range(2, nrows):
            attr = gridlib.GridCellAttr()
            attr.SetReadOnly(True)
            attr.SetBackgroundColour('Grey')
            self.view.SetRowAttr(i, attr)

        pub.subscribe(self.CheckObj, 'Data_Changed', func=self.EnableRow)

    def EnableRow(self, obj, evt):
        rows = evt[0]#.GetRow()
        cols = evt[1]#.GetCol()
        if 1 in rows:
            for i in range(2, self.nrows):
                for col in cols:
                    self.view.SetReadOnly(i, col, False)
                    self.view.SetCellBackgroundColour(i, col, 'White')
                    self.view.SetCellTextColour(i, col, 'Black')

            for col in cols:
                if self.model.GetValue(1, col) == 'Uniform':
                    for i in range(4, self.nrows):
                        self.view.SetReadOnly(i, col, True)
                        self.view.SetCellBackgroundColour(i, col, 'Grey')
                        self.view.SetCellTextColour(i, col, 'Grey')
                        self.model.SetValue(i, col, None)
        self.view.Refresh()


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

if __name__ == '__main__':
    app = wx.App()
    frame1 = TestFrame(None, 'test', size=(10,10))
    frame2 = TestFrame(None, 'prior', size=(10,10), prior=True)
    app.MainLoop()


