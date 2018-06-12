"""
Copyright (C) 2003-2004 Andrew Straw, Jeremy O'Donoghue and others

License: This work is licensed under the PSF. A copy should be included
with this source code, and is also available at
http://www.python.org/psf/license.html

This is yet another example of using matplotlib with wx.  Hopefully
this is pretty full-featured:

  - both matplotlib toolbar and WX buttons manipulate plot
  - full wxApp framework, including widget interaction
  - XRC (XML wxWidgets resource) file to create GUI (made with XRCed)

This was derived from embedding_in_wx and dynamic_image_wxagg.

Thanks to matplotlib and wx teams for creating such great software!

"""
from __future__ import print_function

# matplotlib requires wxPython 2.8+
# set the wxPython version in lib\site-packages\wx.pth file
# or if you have wxversion installed un-comment the lines below
#import wxversion
#wxversion.ensureMinimal('2.8')

import random
import cPickle as pickle
import pandas as pd
import sys
import time
import os
import gc
import matplotlib
matplotlib.use('WXAgg')
import matplotlib.cm as cm
import matplotlib.cbook as cbook
from matplotlib.backends.backend_wxagg import Toolbar, FigureCanvasWxAgg
from matplotlib.figure import Figure
import tempfile


from Training import Training


#from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.backends.backend_wx import NavigationToolbar2Wx
#from matplotlib.figure import Figure

import numpy as np
from copy import deepcopy

import wx
#import wx.xrc as xrc
import wx.grid as gridlib

ERR_TOL = 1e-5  # floating point slop for peak-detection

stockUndo = []
stockRedo = []

ID_SAVE = 10
ID_UNDO = 11
ID_REDO = 12
ID_EXIT = 13
ID_SAVEAS = 14

ID_DELETEMOVEUP = 16
ID_DELETEMOVEDOWN = 17
ID_DELETEMOVELEFT = 18
ID_DELETEMOVERIGHT = 19

ID_COLSIZE = 80
ID_ROWSIZE = 20

ID_PASTE_SPECIAL = 30
ID_PLOT = 40
ID_OPENFILE = 50
ID_INDOPENFILE = 60


matplotlib.rc('image', origin='lower')


class GridPopupMenu(wx.Menu):

    def __init__(self, parent):
        super(GridPopupMenu, self).__init__()
        self.parent = parent
        self.num_col = parent.GetNumberCols()
        self.num_row = parent.GetNumberRows()

        clear = self.Append(wx.ID_CLEAR, 'clear')
        self.Bind(wx.EVT_MENU, self.OnClear, clear)

        copy = self.Append(wx.ID_COPY, 'copy')
        self.Bind(wx.EVT_MENU, self.OnCopy, copy)

        paste = self.Append(wx.ID_PASTE, 'paste')
        self.Bind(wx.EVT_MENU, self.OnPaste, paste)

        paste_special = self.Append(ID_PASTE_SPECIAL, 'paste special')
        self.Bind(wx.EVT_MENU, self.OnPasteSpecial, paste_special)

        insert_left = self.Append(wx.ID_FIRST, 'Insert left')
        self.Bind(wx.EVT_MENU, self.OnInsertLeft, insert_left)

        insert_right = self.Append(wx.ID_LAST, 'Insert right')
        self.Bind(wx.EVT_MENU, self.OnInsertRight, insert_right)

        insert_bottom = self.Append(wx.ID_BOTTOM, 'Insert bottom')
        self.Bind(wx.EVT_MENU, self.OnInsertBottom, insert_bottom)

        insert_top = self.Append(wx.ID_UP, 'Insert top')
        self.Bind(wx.EVT_MENU, self.OnInsertTop, insert_top)
        
        delete_move_left = self.Append(ID_DELETEMOVELEFT, 'Delete move left')
        self.Bind(wx.EVT_MENU, self.OnDeleteMoveLeft, delete_move_left)

        delete_move_right = self.Append(ID_DELETEMOVERIGHT, 'Delete Move right')
        self.Bind(wx.EVT_MENU, self.OnDeleteMoveRight, delete_move_right)
        
        delete_move_down = self.Append(ID_DELETEMOVEDOWN, 'Delete Move down')
        self.Bind(wx.EVT_MENU, self.OnDeleteMoveDown, delete_move_down)
        
        delete_move_up = self.Append(ID_DELETEMOVEUP, 'Delete Move up')
        self.Bind(wx.EVT_MENU, self.OnDeleteMoveUp, delete_move_up)

    def _RecordUndo(self, undo):
        self.parent.stockUndo.append(undo)
        toolbar = self.parent.parent.toolbar
        if (toolbar.GetToolEnabled(ID_UNDO) == False):
            toolbar.EnableTool(ID_UNDO, True)
        if self.parent.stockRedo:
            del self.parent.stockRedo[:]
            toolbar.EnableTool(ID_REDO, False)

    def OnDeleteMoveRight(self, event):
        coords = deepcopy(self.parent.selected_coords)
        width = coords[1][1] - coords[0][1] + 1
 
        range_ = [[coords[0][0], 0], [coords[1][0], coords[1][1] - width]]
        data_list = self.parent.GetRange(range_)
        undo = self.parent.ClearRange(coords)
        undo.merge(self.parent.ClearRange(range_))

        # move everything up
        range_[0][1] = range_[0][1] + width
        range_[1][1] = range_[1][1] + width
        undo.merge(self.parent.SetValue(range_, data_list))

        self._RecordUndo(undo)

    def OnDeleteMoveLeft(self, event):
        coords = deepcopy(self.parent.selected_coords)
        width = coords[1][1] - coords[0][1] + 1
 
        range_ = [[coords[0][0], coords[0][1] + width], [coords[1][0], self.num_col - 1]]
        print('Left getrange ', range_)
        data_list = self.parent.GetRange(range_)
        undo = self.parent.ClearRange(coords)
        undo.merge(self.parent.ClearRange(range_))

        range_[0][1] = range_[0][1] - width
        range_[1][1] = range_[1][1] - width
        print('Left dest', range_)
        undo.merge(self.parent.SetValue(range_, data_list))

        self._RecordUndo(undo)

    def OnDeleteMoveUp(self, event):
        coords = deepcopy(self.parent.selected_coords)
        height = coords[1][0] - coords[0][0] + 1
 
        range_ = [[coords[0][0] + height, coords[0][1]], [self.num_row - 1, coords[1][1]]]
        data_list = self.parent.GetRange(range_)
        undo = self.parent.ClearRange(coords)
        undo.merge(self.parent.ClearRange(range_))

        # move everything up
        range_[0][0] = range_[0][0] - height
        range_[1][0] = range_[1][0] - height
        undo.merge(self.parent.SetValue(range_, data_list))

        self._RecordUndo(undo)

    def OnDeleteMoveDown(self, event):
        coords = deepcopy(self.parent.selected_coords)
        height = coords[1][0] - coords[0][0] + 1
 
        # move everything on top downwards
        range_ = [[0, coords[0][1]], [coords[0][0] - 1, coords[1][1]]]
        data_list = self.parent.GetRange(range_)
        undo = self.parent.ClearRange(coords)
        undo.merge(self.parent.ClearRange(range_))

        # move everything down
        range_[0][0] = range_[0][0] + height
        range_[1][0] = range_[1][0] + height
        undo.merge(self.parent.SetValue(range_, data_list))

        self._RecordUndo(undo)
        

    def OnInsertBottom(self, event):
        coords = deepcopy(self.parent.selected_coords)
        height = coords[1][0] - coords[0][0] + 1
        range_ = [[0, coords[0][1]], coords[1]]
        data_list = self.parent.GetRange(range_)
        undo = self.parent.ClearRange(range_)

        # set data to new range
        range_[0][0] = range_[0][0] - height
        range_[1][0] = range_[1][0] - height
        undo.merge(self.parent.SetValue(range_, data_list))

        self._RecordUndo(undo)
        

    def OnInsertTop(self, event):
        coords = deepcopy(self.parent.selected_coords)
        height = coords[1][0] - coords[0][0] + 1
        range_ = [coords[0], [self.num_row - 1, coords[1][1]]]
        data_list = self.parent.GetRange(range_)

        # empty the old range
        undo = self.parent.ClearRange(range_)

        # set data to new range
        range_[0][0] = range_[0][0] + height
        range_[1][0] = range_[1][0] + height
        undo.merge(self.parent.SetValue(range_, data_list))

        self._RecordUndo(undo)

    def OnInsertLeft(self, event):
        coords = deepcopy(self.parent.selected_coords)
        width = coords[1][1] - coords[0][1] + 1
        range_ = [coords[0], [coords[1][0], self.num_col - 1]]
        data_list = self.parent.GetRange(range_)

        # empty the old range
        undo = self.parent.ClearRange(range_)

        # set data to new range
        range_[0][1] = range_[0][1] + width
        range_[1][1] = range_[1][1] + width
        undo.merge(self.parent.SetValue(range_, data_list))

        self._RecordUndo(undo)

    def OnInsertRight(self, event):
        coords = deepcopy(self.parent.selected_coords)
        width = coords[1][1] - coords[0][1] + 1
        range_ = [[coords[0][0], 0], coords[1]]
        data_list = self.parent.GetRange(range_)

        # empty the old range
        undo = self.parent.ClearRange(range_)

        # set data to new range
        range_[0][1] = range_[0][1] - width
        range_[1][1] = range_[1][1] - width
        undo.merge(self.parent.SetValue(range_, data_list))

        self._RecordUndo(undo)

    def OnCopy(self, event):
        data = []
        for line in self.parent.selected_data:
            data.append('\t'.join(line))
        data = '\n'.join(data)
        dataObj = wx.TextDataObject()
        dataObj.SetText(data)
        if wx.TheClipboard.Open():
            wx.TheClipboard.SetData(dataObj)
            wx.TheClipboard.Close()
        else:
            wx.MessageBox("Unable to open the clipboard", "Error")

    def OnPaste(self, event):
        dataObj = wx.TextDataObject()
        if wx.TheClipboard.Open():
             wx.TheClipboard.GetData(dataObj)
             wx.TheClipboard.Close()
        else:
             wx.MessageBox("Can't open the clipboard", "Error")
        data = dataObj.GetText().splitlines()
       
        first = [self.parent.GetGridCursorRow(), self.parent.GetGridCursorCol()]
        second = [self.parent.GetGridCursorRow() + len(data), self.parent.GetGridCursorCol()]
        undo = self.parent.SetValue([first, second], data)

        self._RecordUndo(undo)


    def OnClear(self, event):
        selected = self.parent.selected_coords
        undo = self.parent.ClearRange(selected)
        self._RecordUndo(undo)
         
    def OnPasteSpecial(self, event):
        dataObj = wx.TextDataObject()
        if wx.TheClipboard.Open():
             wx.TheClipboard.GetData(dataObj)
             wx.TheClipboard.Close()
        else:
             wx.MessageBox("Can't open the clipboard", "Error")
        data = [[ele for ele in line.split('\t')] for line in dataObj.GetText().split('\n')]
        first = [self.parent.GetGridCursorRow(), self.parent.GetGridCursorCol()]
        second = [self.parent.GetGridCursorRow() + len(data), self.parent.GetGridCursorCol() + len(data[0])]
        undo = self.parent.SetValue([first, second], data)
        self._RecordUndo(undo)

class UndoText:
    def __init__(self, sheet, text1, text2, row, column):
        self.RedoText =  text2
        self.row = row
        self.col = column
        self.UndoText = text1
        self.sheet = sheet

    def merge(self, another):
        self.RedoText = self.RedoText + another.RedoText
        self.row = self.row + another.row
        self.col = self.col + another.col
        self.UndoText = self.UndoText + another.UndoText

    def undo(self):
        self.RedoText = []
        for row, col, UndoText in reversed(zip(self.row, self.col, self.UndoText)):
            self.RedoText.append(self.sheet.GetCellValue(row, col))
            if self.UndoText ==  None:
                self.sheet.SetCellValue('')
            else: 
                self.sheet.SetCellValue(row, col, UndoText)

    def redo(self):
        for row, col, RedoText in zip(self.row, self.col, reversed(self.RedoText)):
            if RedoText == None:
                self.sheet.SetCellValue('')
            else: 
                self.sheet.SetCellValue(row, col, RedoText)
        

class MyGrid(gridlib.Grid):
    #----------------------------------------------------------------------
    def __init__(self, parent):
        """Constructor"""
        self.stockUndo = []
        self.stockRedo = []
        #wx.Frame.__init__(self, parent, -1)
        #wx.Panel.__init__(self, parent)
        gridlib.Grid.__init__(self, parent, size=(200, 300))
        self.num_row = 100
        self.num_col = 100
        self.CreateGrid(self.num_row, self.num_col)
        #self.parent = wx.Panel(self)
        self.parent = parent
        self.InitUI()

    def InitUI(self):
        #self.grid = gridlib.Grid(self, -1)#.parent)
        #self.grid.CreateGrid(12, 8)

        self.selected_data = None
        self.selected_coords = None#[(0,0), (0,0)]
 
        # test all the events
        #self.Bind(gridlib.EVT_GRID_CELL_LEFT_CLICK, self.OnCellLeftClick)
        #self.Bind(gridlib.EVT_GRID_CELL_RIGHT_CLICK, self.OnCellRightClick)
        self.Bind(gridlib.EVT_GRID_CELL_LEFT_DCLICK, self.OnCellLeftDClick)
        self.Bind(gridlib.EVT_GRID_CELL_RIGHT_CLICK, self.ShowMenu)#self.OnCellRightDClick)
 
        #self.Bind(gridlib.EVT_GRID_LABEL_LEFT_CLICK, self.OnLabelLeftClick)
        self.Bind(gridlib.EVT_GRID_LABEL_RIGHT_CLICK, self.OnLabelRightClick)
        self.Bind(gridlib.EVT_GRID_LABEL_LEFT_DCLICK, self.OnLabelLeftDClick)
        self.Bind(gridlib.EVT_GRID_LABEL_RIGHT_DCLICK, self.OnLabelRightDClick)
 
        self.Bind(gridlib.EVT_GRID_ROW_SIZE, self.OnRowSize)
        self.Bind(gridlib.EVT_GRID_COL_SIZE, self.OnColSize)
 
        self.Bind(gridlib.EVT_GRID_RANGE_SELECT, self.OnRangeSelect)
        self.Bind(gridlib.EVT_GRID_CELL_CHANGE, self.OnCellChange)
        self.Bind(gridlib.EVT_GRID_SELECT_CELL, self.OnSelectCell)
 
        self.Bind(gridlib.EVT_GRID_EDITOR_SHOWN, self.OnEditorShown)
        self.Bind(gridlib.EVT_GRID_EDITOR_HIDDEN, self.OnEditorHidden)
        self.Bind(gridlib.EVT_GRID_EDITOR_CREATED, self.OnEditorCreated)

        

    def SetValue(self, coords, data):
        changed_row = []
        changed_col = []
        old_value = []
        new_value = []
        if type(data) is not list: data = [data]
        for line, row in zip(data, range(coords[0][0], coords[1][0] + 1)):
            if type(line) is not list: line = [line]
            for element, col in zip(line, range(coords[0][1], coords[1][1] + 1)):
                if row >= 0 and row < self.num_row and col >= 0 and col < self.num_col:
                    changed_row.append(row)
                    changed_col.append(col)
                    old_value.append(self.GetCellValue(row, col))
                    new_value.append(str(element))
                    self.SetCellValue(row, col, str(element))
        undo = UndoText(self, old_value, new_value, changed_row, changed_col)
        return undo

    def ClearRange(self, coords):
        changed_row = []
        changed_col = []
        old_value = []
        new_value = []
        for row in range(coords[0][0], coords[1][0] + 1):
            for col in range(coords[0][1], coords[1][1] + 1):
                if row >= 0 and row < self.num_row and col >= 0 and col < self.num_col:
                    changed_row.append(row)
                    changed_col.append(col)
                    old_value.append(self.GetCellValue(row, col))
                    new_value.append('')
                    self.SetCellValue(row, col, '')
        undo = UndoText(self, old_value, new_value, changed_row, changed_col)
        return undo

    

    def ShowMenu(self, event):
        pos = wx.GetMousePosition()
        win = self.PopupMenu(GridPopupMenu(self), self.ScreenToClient(pos))
 
          

    def OnCellLeftClick(self, evt):
        print("OnCellLeftClick: (%d,%d) %s\n" % (evt.GetRow(),
                                                 evt.GetCol(),
                                                 evt.GetPosition()))
        self.selected_data = [[ self.GetCellValue(evt.GetRow(), evt.GetCol()) ]]
        evt.Skip()
 
    def OnCellRightClick(self, evt):
        print("OnCellRightClick: (%d,%d) %s\n" % (evt.GetRow(),
                                                  evt.GetCol(),
                                                  evt.GetPosition()))
        evt.Skip()
 
    def OnCellLeftDClick(self, evt):
        print("OnCellLeftDClick: (%d,%d) %s\n" % (evt.GetRow(),
                                                  evt.GetCol(),
                                                  evt.GetPosition()))
        evt.Skip()
 
    def OnLabelLeftClick(self, evt):
        pos = wx.GetMousePosition()
        win = self.PopupMenu(LabelPopupMenu(self), self.parent.ScreenToClient(pos))
        print("OnLabelLeftClick: (%d,%d) %s\n" % (evt.GetRow(),
                                                  evt.GetCol(),
                                                  evt.GetPosition()))
        evt.Skip()
 
    def OnLabelRightClick(self, evt):
        print("OnLabelRightClick: (%d,%d) %s\n" % (evt.GetRow(),
                                                   evt.GetCol(),
                                                   evt.GetPosition()))
        evt.Skip()
 
    def OnLabelLeftDClick(self, evt):
        print("OnLabelLeftDClick: (%d,%d) %s\n" % (evt.GetRow(),
                                                   evt.GetCol(),
                                                   evt.GetPosition()))
        evt.Skip()
 
    def OnLabelRightDClick(self, evt):
        print("OnLabelRightDClick: (%d,%d) %s\n" % (evt.GetRow(),
                                                    evt.GetCol(),
                                                    evt.GetPosition()))
        evt.Skip()
 
    def OnRowSize(self, evt):
        print("OnRowSize: row %d, %s\n" % (evt.GetRowOrCol(),
                                           evt.GetPosition()))
        evt.Skip()
 
    def OnColSize(self, evt):
        print("OnColSize: col %d, %s\n" % (evt.GetRowOrCol(),
                                           evt.GetPosition()))
        evt.Skip()
 
    def OnRangeSelect(self, evt):
        if evt.Selecting():
            first = evt.GetTopLeftCoords()
            second = evt.GetBottomRightCoords()
            self.selected_coords = [list(first), list(second)]
            self.selected_data = []
            for row in range(first[0], second[0] + 1):
                col_data = []
                for column in range(first[1], second[1] + 1):
                    col_data.append(self.GetCellValue(row, column))
                self.selected_data.append(col_data)
        else:
            self.selected_coords = None
            self.selected_data = None
        evt.Skip()


    def GetRange(self, coord_list):
        first = coord_list[0]
        second = coord_list[1]
        data = []
        for row in range(first[0], second[0] + 1):
            col_data = []
            for column in range(first[1], second[1] + 1):
                col_data.append(self.GetCellValue(row, column))
            data.append(col_data)
        return data


    def GetAllValues(self):
        data = []
        for row in range(0, self.num_row):
            row_data = []
            for col in range(0, self.num_col):
                value = self.GetCellValue(row, col)
                if value != '':
                    row_data.append(value)
            data.append(row_data)
        # remove trailing empty list
        while not data[-1]:
            data.pop()
        return data 
                
        

 
    def OnCellChange(self, event):
        toolbar = self.parent.toolbar
        if (toolbar.GetToolEnabled(ID_UNDO) == False):
            toolbar.EnableTool(ID_UNDO, True)
        r = event.GetRow()
        c = event.GetCol()
        text = self.GetCellValue(r, c)
        # self.text - text before change
        # text - text after change
        undo = UndoText(self, [event.GetString()], [text], [r], [c])
        self.stockUndo.append(undo)

        if self.stockRedo:
            # this might be surprising, but it is a standard behaviour
            # in all spreadsheets
            del self.stockRedo[:]
            toolbar.EnableTool(ID_REDO, False)

 
    def OnSelectCell(self, evt):
        if evt.Selecting():
            row = evt.GetRow()
            col = evt.GetCol()
            self.selected_data = [[self.GetCellValue(row, col)]]
            self.selected_coords = [[row, col], [row, col]]
 
 
    def OnEditorShown(self, evt):
        if evt.GetRow() == 6 and evt.GetCol() == 3 and \
           wx.MessageBox("Are you sure you wish to edit this cell?",
                        "Checking", wx.YES_NO) == wx.NO:
            evt.Veto()
            return
 
        print("OnEditorShown: (%d,%d) %s\n" % (evt.GetRow(), evt.GetCol(),
                                               evt.GetPosition()))
        evt.Skip()
 
 
    def OnEditorHidden(self, evt):
        if evt.GetRow() == 6 and evt.GetCol() == 3 and \
           wx.MessageBox("Are you sure you wish to  finish editing this cell?",
                        "Checking", wx.YES_NO) == wx.NO:
            evt.Veto()
            return
 
        print("OnEditorHidden: (%d,%d) %s\n" % (evt.GetRow(),
                                                evt.GetCol(),
                                                evt.GetPosition()))
        evt.Skip()
 
 
    def OnEditorCreated(self, evt):
        print("OnEditorCreated: (%d, %d) %s\n" % (evt.GetRow(),
                                                  evt.GetCol(),
                                                  evt.GetControl()))

class LeftPanel(wx.Panel):

    def __init__(self, parent, plotpanel):
        wx.Panel.__init__(self, parent)
        self.plotpanel = plotpanel
        self.parent = parent

        self.grid = MyGrid(self)
        sizer = wx.BoxSizer(wx.VERTICAL)

     # toolbar
        save_ico = wx.ArtProvider.GetBitmap(wx.ART_FILE_SAVE, wx.ART_TOOLBAR, (16,16))
        self.toolbar = wx.ToolBar(self, id=100, style=wx.TB_HORIZONTAL | wx.NO_BORDER |
                                        wx.TB_FLAT | wx.TB_TEXT)
        self.toolbar.AddSimpleTool(ID_UNDO, save_ico, 'Undo', '')
        self.toolbar.AddSimpleTool(ID_REDO, wx.Bitmap('/projects/hira/tsangc/GaussianEmulator/development/human-icon-theme/16x16/stock/generic/stock_exit.png'), 'Redo', '')
        self.toolbar.AddSimpleTool(ID_PLOT, wx.Bitmap('/projects/hira/tsangc/GaussianEmulator/development/human-icon-theme/16x16/stock/generic/stock_exit.png'), 'plot', '')
        open_ico = wx.ArtProvider.GetBitmap(wx.ART_FILE_OPEN, wx.ART_TOOLBAR, (16,16))
        self.toolbar.AddSimpleTool(ID_INDOPENFILE, open_ico, 'Open csv', '')
 

        self.toolbar.EnableTool(ID_UNDO, False)
        self.toolbar.EnableTool(ID_REDO, False)
        #self.toolbar.AddSeparator()
        #self.toolbar.AddSimpleTool(ID_EXIT, wx.Bitmap('/projects/hira/tsangc/GaussianEmulator/development/human-icon-theme/16x16/stock/generic/stock_exit.png'),'Quit', '')
        self.toolbar.Realize()
        self.toolbar.Bind(wx.EVT_TOOL, self.OnUndo, id=ID_UNDO)
        self.toolbar.Bind(wx.EVT_TOOL, self.OnRedo, id=ID_REDO)
        self.toolbar.Bind(wx.EVT_TOOL, self.OnPlot, id=ID_PLOT)
        self.toolbar.Bind(wx.EVT_TOOL, self.OnOpen, id=ID_INDOPENFILE)
        
        sizer.Add(self.toolbar, border=5)
        sizer.Add(self.grid, 1., wx.LEFT | wx.TOP | wx.GROW)
        self.SetSizer(sizer)
        self.Fit()

    def OnOpen(self, event):
        """
        Create and show the Open FileDialog
        """
        dlg = wx.FileDialog(
            self, message="Choose a file",
            defaultFile="",
            style=wx.OPEN | wx.MULTIPLE | wx.CHANGE_DIR
            )
        result = dlg.ShowModal() 
        path = dlg.GetPaths()
        dlg.Destroy()

        if result != wx.ID_OK:
            return False

        df = pd.read_csv(path[0])
        data = [df.columns.tolist()] + df.values.tolist()
        self.grid.ClearRange([[0,0], [self.grid.num_col - 1, self.grid.num_row - 1]])
        self.grid.SetValue([[0,0], [len(data) - 1, len(data[0]) - 1]], data)
        

    def OnUndo(self, event):
        if len(self.grid.stockUndo) == 0:
            return

        a = self.grid.stockUndo.pop()
        if len(self.grid.stockUndo) == 0:
            self.toolbar.EnableTool(ID_UNDO, False)

        a.undo()
        self.grid.stockRedo.append(a)
        self.toolbar.EnableTool(ID_REDO, True)

    def OnRedo(self, event):
        if len(self.grid.stockRedo) == 0:
            return

        a = self.grid.stockRedo.pop()
        if len(self.grid.stockRedo) == 0:
            self.toolbar.EnableTool(ID_REDO, False)

        a.redo()
        self.grid.stockUndo.append(a)

        self.toolbar.EnableTool(ID_UNDO, True)

    def OnPlot(self, event):
        range_ = self.grid.selected_coords
        if range_ is None:
            return
         
        data = self.grid.GetRange(range_)
        data = np.array(data)
        data[data == ''] = np.nan

        try:
            data = np.array([np.genfromtxt(line) for line in data])
            #data = data[~np.isnan(data).any(axis=0)]
        except:
            print ('This array cannot be converted into float. Abort')
            return
        
        if data.shape[0] == 1:
            self.plotpanel.SetData(range(0, data.shape[1]), data[0, :])
        elif len(data.shape) == 1:
            self.plotpanel.SetData(range(0, data.shape[0]), data[:])
        elif data.shape[1] == 1:
            self.plotpanel.SetData(range(0, data.shape[0]), data[:, 0])
        elif data.shape[0] == 2:
            self.plotpanel.SetData(data[0, :], data[1, :])
        elif data.shape[1] == 2:
            self.plotpanel.SetData(data[:, 0], data[:, 1])
        
        else:
            print(data, 'data shape not recognized')


########################################################################
class RightPanel(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent, -1)

        self.fig = Figure((5, 4), 75)
        self.canvas = FigureCanvasWxAgg(self, -1, self.fig)
        self.toolbar = NavigationToolbar2Wx(self.canvas)  # matplotlib toolbar
        self.toolbar.Realize()
        # self.toolbar.set_active([0,1])

        # Now put all into a sizer
        sizer = wx.BoxSizer(wx.VERTICAL)
        # This way of adding to sizer allows resizing
        sizer.Add(self.canvas, 1, wx.LEFT | wx.TOP | wx.GROW)
        # Best to allow the toolbar to resize!
        sizer.Add(self.toolbar, 0, wx.GROW)
        self.SetSizer(sizer)
        self.Fit()
       
        self.init_plot_data()

    def init_plot_data(self):
        a = self.fig.add_subplot(111)

        x = np.arange(100.0) * 2 * np.pi / 60.0
        y = np.arange(100.0) * 2 * np.pi / 50.0
        self.lines = a.plot(x, y, 'ro')

        self.toolbar.update()  # Not sure why this is needed - ADS

    def GetToolBar(self):
        # You will need to override GetToolBar if you are using an
        # unmanaged toolbar in your frame
        return self.toolbar

    def SetData(self, xdata, ydata):
        print(xdata, ydata)
        self.lines[0].set_data(xdata, ydata)

        self.canvas.draw()

    def onEraseBackground(self, evt):
        # this is supposed to prevent redraw flicker on some X servers...
        pass

class TabPanel(wx.Panel):
    #----------------------------------------------------------------------
    def __init__(self, parent):
        """"""
        wx.Panel.__init__(self, parent=parent)
 
        colors = ["red", "blue", "gray", "yellow", "green"]
        self.SetBackgroundColour(random.choice(colors))
 
        btn = wx.Button(self, label="Press Me")
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(btn, 0, wx.ALL, 10)
        self.SetSizer(sizer)
 

class CommonToolBar(wx.ToolBar):

    def __init__(self, parent, tab1, tab2, tab3, **args):
        wx.ToolBar.__init__(self, parent, **args)
        self.tab1 = tab1
        self.tab2 = tab2
        self.tab3 = tab3
        self.parent = parent
        

        save_ico = wx.ArtProvider.GetBitmap(wx.ART_FILE_SAVE, wx.ART_TOOLBAR, (16,16))
        self.AddSimpleTool(ID_SAVE, save_ico, 'Save', '')
        open_ico = wx.ArtProvider.GetBitmap(wx.ART_FILE_OPEN, wx.ART_TOOLBAR, (16,16))
        self.AddSimpleTool(ID_OPENFILE, open_ico, 'Open', '')
        new_ico = wx.ArtProvider.GetBitmap(wx.ART_NEW, wx.ART_TOOLBAR, (16,16))
        self.AddSimpleTool(ID_SAVEAS, new_ico, 'Save As', '')
 

        self.Realize()
        self.Bind(wx.EVT_TOOL, self.OnFile, id=ID_OPENFILE)
        self.Bind(wx.EVT_TOOL, self.OnSave, id=ID_SAVE)
        self.Bind(wx.EVT_TOOL, self.OnSaveNew, id=ID_SAVEAS)
 
        self.opened_filename = None
        self.opened_data = None


    def OnSaveNew(self, event):

        model = self.tab2.grid.GetAllValues()
        headers = model.pop(0)
        model = pd.DataFrame(model, columns=headers)

        prior = self.tab1.grid.GetAllValues()
        headers = prior.pop(0)
        prior = pd.DataFrame(prior, columns=headers)

        exp = self.tab3.grid.GetAllValues()
        headers = exp.pop(0)
        exp = pd.DataFrame(exp, columns=headers)

        """
        Create and show the Open FileDialog
        """
        wildcard = "Python source (*.pkl)|*.pkl|" \
           "All files (*.*)|*.*"
        dlg = wx.FileDialog(self, message="Save project as ...", defaultFile="", style=wx.SAVE | wx.OVERWRITE_PROMPT)
        result = dlg.ShowModal()            
        outFile = dlg.GetPaths()
        dlg.Destroy()
    
        if result == wx.ID_CANCEL:    #Either the cancel button was pressed or the window was closed
            return False

        with tempfile.NamedTemporaryFile() as tmpmodel, tempfile.NamedTemporaryFile() as tmpprior, tempfile.NamedTemporaryFile() as tmpexp: 
            prior.to_csv(tmpprior.name, sep=',', index=False)
            model.to_csv(tmpmodel.name, sep=',', index=False)
            exp.to_csv(tmpexp.name, sep=',', index=False)

            tmpprior.flush()
            tmpmodel.flush()
            tmpexp.flush()

            args = {}
            args['Prior'] = tmpprior.name
            args['ModelData'] = tmpmodel.name
            args['ExpData'] = tmpexp.name
            args['Training_name'] = outFile[0]
            args['abs'] = True
            args['covariancefunc'] = 'ARD'
            args['principalcomp'] = 3
            args['initialscale'] = [0.5]
            args['initialnugget'] = 1
            args['scalerate'] = 0.003
            args['nuggetrate'] = 0.003
            args['maxsteps'] = 1000
            
            Training(args)

            with open(outFile[0], 'rb') as buff:
                data = pickle.load(buff)

            self.opened_data = data
            self.opened_filename = outFile
        

    def OnSave(self, event):
        
        model = self.tab2.grid.GetAllValues()
        headers = model.pop(0)
        model = pd.DataFrame(model, columns=headers)
        model.to_csv('/projects/hira/tsangc/GaussianEmulator/development/testing_model.csv', sep=',', index=False)

        prior = self.tab1.grid.GetAllValues()
        headers = prior.pop(0)
        prior = pd.DataFrame(prior, columns=headers)
        with tempfile.NamedTemporaryFile() as temp:
            prior.to_csv(temp.name, sep=',', index=False)
            temp.flush()
            self.opened_data['data'].ChangePrior(temp.name)

        exp = self.tab2.grid.GetAllValues()
        headers = exp.pop(0)
        exp = pd.DataFrame(exp, columns=headers)
        with tempfile.NamedTemporaryFile() as temp:
            exp.to_csv(temp.name, sep=',', index=False)
            temp.flush()
            self.opened_data['data'].ChangeExp(temp.name)

        """
        Create and show the Open FileDialog
        """
        dlg = wx.FileDialog(
            self, message="Save project as ...",
            defaultFile="",
            style=wx.SAVE | wx.OVERWRITE_PROMPT
            )
        result = dlg.ShowModal()            
        inFile = dlg.GetPaths()
        dlg.Destroy()

        if result == wx.ID_OK:          #Save button was pressed
            with open(inFile[0], 'wb') as buff:
                pickle.dump(self.opened_data, buff)
            return True
        elif result == wx.ID_CANCEL:    #Either the cancel button was pressed or the window was closed
            return False


    def OnFile(self, event):
        """
        Create and show the Open FileDialog
        """
        dlg = wx.FileDialog(
            self, message="Choose a file",
            defaultFile="",
            style=wx.OPEN | wx.MULTIPLE | wx.CHANGE_DIR
            )
        result = dlg.ShowModal() 
        path = dlg.GetPaths()
        dlg.Destroy()

        if result != wx.ID_OK:
            return False

        with open(path[0], 'rb') as buff:
            data = pickle.load(buff)

        self.opened_data = data
        """
        Loading prior
        """
        prior = data['data'].prior
        prior = [prior.columns.tolist()] + prior.values.tolist()
        if type(prior[0]) is not list:
            prior = [prior]
        self.tab1.grid.SetValue([[0,0], [len(prior) - 1, len(prior[0]) - 1]], prior)

        """
        Loading model data
        """
        training_data = data['data']
        header = [training_data.par_name + training_data.var_name + [name + "_Error" for name in training_data.var_name]] 
        content = np.concatenate((training_data.sim_para, training_data.sim_data, training_data.sim_error), axis=1).tolist()
        if type(content[0]) is not list:
            content = [content]
        self.tab2.grid.SetValue([[0,0], [0, len(header[0]) - 1]], header)
        self.tab2.grid.SetValue([[1,0], [len(content), len(content[0]) - 1]], content)

        """
        Loading exp data
        """
        header = [training_data.var_name + [name + "_Error" for name in training_data.var_name]] 
        content = np.concatenate((training_data.exp_result, np.sqrt(np.diag(training_data.exp_cov)))).tolist()
        if type(content[0]) is not list:
            content = [content]
        self.tab3.grid.SetValue([[0,0], [0, len(header[0]) - 1]], header)
        self.tab3.grid.SetValue([[1,0], [len(content), len(content[0]) - 1]], content)


class Common(wx.Frame):
 
    def __init__(self, parent):
        wx.Frame.__init__(self, parent, wx.NewId(), "Common", size=(1000,400))
        
        panel = wx.Panel(self)
         
        notebook = wx.Notebook(panel)

        splitter = wx.SplitterWindow(notebook)
        rightP = RightPanel(splitter)
        leftP = LeftPanel(splitter, rightP)
        splitter.SplitVertically(leftP, rightP, 200)
        splitter.SetMinimumPaneSize(500)
        notebook.AddPage(splitter, "Para prior")

        splitter = wx.SplitterWindow(notebook)
        rightP = RightPanel(splitter)
        leftP2 = LeftPanel(splitter, rightP)
        splitter.SplitVertically(leftP2, rightP, 200)
        splitter.SetMinimumPaneSize(500)
        notebook.AddPage(splitter, "Model result")

        splitter = wx.SplitterWindow(notebook)
        rightP = RightPanel(splitter)
        leftP3 = LeftPanel(splitter, rightP)
        splitter.SplitVertically(leftP3, rightP, 200)
        splitter.SetMinimumPaneSize(500)
        notebook.AddPage(splitter, "Exp data data")

        sizer = wx.BoxSizer(wx.VERTICAL)
                
        
    # toolbar
        self.toolbar = CommonToolBar(panel, tab1=leftP, tab2=leftP2, tab3=leftP3, id=100, style=wx.TB_HORIZONTAL | wx.NO_BORDER |
                                        wx.TB_FLAT | wx.TB_TEXT)

        sizer.Add(self.toolbar, border=5)
        
        sizer.Add(notebook, 1, wx.EXPAND | wx.EXPAND, 5)
        
        panel.SetSizer(sizer)
        self.Layout()
        self.Show()

    



app = wx.App(0)
frame = Common(None)
frame.Show()
app.MainLoop()
