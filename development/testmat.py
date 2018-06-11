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
import numpy as np
from copy import copy

import wx
#import wx.xrc as xrc
import wx.grid as gridlib

ERR_TOL = 1e-5  # floating point slop for peak-detection

stockUndo = []
stockRedo = []

ID_QUIT = 10
ID_UNDO = 11
ID_REDO = 12
ID_EXIT = 13

ID_COLSIZE = 80
ID_ROWSIZE = 20

ID_PASTE_SPECIAL = 30


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


    def OnInsertBottom(self, event):
        coords = self.parent.selected_coords
        height = coords[1][0] - coords[0][0] + 1
        range_ = [[0, coords[0][1]], coords[1]]
        data_list = self.parent.GetRange(range_)

        # empty the old range
        undo = self.parent.ClearRange(range_)

        # set data to new range
        range_[0][0] = range_[0][0] - height
        range_[1][0] = range_[1][0] - height
        undo.merge(self.parent.SetValue(range_, data_list))

        stockUndo.append(undo)
 
        if stockRedo:
            # this might be surprising, but it is a standard behaviour
            # in all spreadsheets
            del stockRedo[:]
            self.parent.parent.toolbar.EnableTool(ID_REDO, False)
        

    def OnInsertTop(self, event):
        coords = self.parent.selected_coords
        height = coords[1][0] - coords[0][0] + 1
        range_ = [coords[0], [self.num_row - 1, coords[1][1]]]
        data_list = self.parent.GetRange(range_)

        # empty the old range
        undo = self.parent.ClearRange(range_)

        # set data to new range
        range_[0][0] = range_[0][0] + height
        range_[1][0] = range_[1][0] + height
        undo.merge(self.parent.SetValue(range_, data_list))

        stockUndo.append(undo)

        if stockRedo:
            # this might be surprising, but it is a standard behaviour
            # in all spreadsheets
            del stockRedo[:]
            self.parent.parent.toolbar.EnableTool(ID_REDO, False)

    def OnInsertLeft(self, event):
        coords = self.parent.selected_coords
        width = coords[1][1] - coords[0][1] + 1
        range_ = [coords[0], [coords[1][0], self.num_col - 1]]
        data_list = self.parent.GetRange(range_)

        # empty the old range
        undo = self.parent.ClearRange(range_)

        # set data to new range
        range_[0][1] = range_[0][1] + width
        range_[1][1] = range_[1][1] + width
        undo.merge(self.parent.SetValue(range_, data_list))

        stockUndo.append(undo)

        if stockRedo:
            # this might be surprising, but it is a standard behaviour
            # in all spreadsheets
            del stockRedo[:]
            self.parent.parent.toolbar.EnableTool(ID_REDO, False)

    def OnInsertRight(self, event):
        coords = self.parent.selected_coords
        width = coords[1][1] - coords[0][1] + 1
        range_ = [[coords[0][0], 0], coords[1]]
        data_list = self.parent.GetRange(range_)

        # empty the old range
        undo = self.parent.ClearRange(range_)

        # set data to new range
        range_[0][1] = range_[0][1] - width
        range_[1][1] = range_[1][1] - width
        undo.merge(self.parent.SetValue(range_, data_list))

        stockUndo.append(undo)
   
        if stockRedo:
            del stockRedo[:]
            self.parent.parent.toolbar.EnableTool(ID_REDO, False)

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
        stockUndo.append(undo)

        if stockRedo:
            del stockRedo[:]
            self.parent.parent.toolbar.EnableTool(ID_REDO, False)
            


    def OnClear(self, event):
        selected = self.parent.selected_coords
        undo = self.parent.ClearRange(selected)
        stockUndo.append(undo)

        if stockRedo:
            del stockRedo[:]
            self.parent.parent.toolbar.EnableTool(ID_REDO, False)
         
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
        stockUndo.append(undo)

        if stockRedo:
            del stockRedo[:]
            self.parent.parent.toolbar.EnableTool(ID_REDO, False)

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
        for row, col, UndoText in zip(self.row, self.col, self.UndoText):
            self.RedoText.append(self.sheet.GetCellValue(row, col))
            if self.UndoText ==  None:
                self.sheet.SetCellValue('')
            else: 
                self.sheet.SetCellValue(row, col, UndoText)

    def redo(self):
        for row, col, RedoText in reversed(zip(self.row, self.col, self.RedoText)):
            if RedoText == None:
                self.sheet.SetCellValue('')
            else: 
                self.sheet.SetCellValue(row, col, RedoText)
        

class MyGrid(gridlib.Grid):
    #----------------------------------------------------------------------
    def __init__(self, parent):
        """Constructor"""
        #wx.Frame.__init__(self, parent, -1)
        #wx.Panel.__init__(self, parent)
        gridlib.Grid.__init__(self, parent.panel)
        self.num_row = 12
        self.num_col = 8
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
                    print('%d, %d, value %s' % (row, col, element))
                    changed_row.append(row)
                    changed_col.append(col)
                    old_value.append(self.GetCellValue(row, col))
                    new_value.append(element)
                    self.SetCellValue(row, col, element)
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
                    print('%d, %d, clear' % (row, col))
                    changed_row.append(row)
                    changed_col.append(col)
                    old_value.append(self.GetCellValue(row, col))
                    new_value.append('')
                    self.SetCellValue(row, col, '')
        undo = UndoText(self, old_value, new_value, changed_row, changed_col)
        return undo

    

    def ShowMenu(self, event):
        pos = wx.GetMousePosition()
        print(self.parent.panel.ScreenToClient(pos))
        print(self.parent.ScreenToClient(pos))
        print(self.ScreenToClient(pos))
        print(self.parent.ScreenToClient(self.ScreenToClient(pos)))
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
        win = self.PopupMenu(LabelPopupMenu(self), self.parent.panel.ScreenToClient(pos))
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
            print(first, second)
            self.selected_coords = [list(first), list(second)]
            self.selected_data = []
            for row in range(first[0], second[0] + 1):
                col_data = []
                for column in range(first[1], second[1] + 1):
                    col_data.append(self.GetCellValue(row, column))
                self.selected_data.append(col_data)
        else:
            print('dselect')
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
        stockUndo.append(undo)

        if stockRedo:
            # this might be surprising, but it is a standard behaviour
            # in all spreadsheets
            del stockRedo[:]
            toolbar.EnableTool(ID_REDO, False)

 
        print("OnCellChange: (%d,%d) %s\n" % (event.GetRow(), event.GetCol(), event.GetPosition()))
 
        # Show how to stay in a cell that has bad data.  We can't just
        # call SetGridCursor here since we are nested inside one so it
        # won't have any effect.  Instead, set coordinates to move to in
        # idle time.
        value = self.GetCellValue(event.GetRow(), event.GetCol())
 
        if value == 'no good':
            self.moveTo = event.GetRow(), event.GetCol()
 
    def OnSelectCell(self, evt):
        if evt.Selecting():
            msg = 'Selected'
            row = evt.GetRow()
            col = evt.GetCol()
            self.selected_data = [[self.GetCellValue(row, col)]]
            self.selected_coords = [[row, col], [row, col]]
        else:
            msg = 'Deselected'
        print("OnSelectCell: %s (%d,%d) %s\n" % (msg, evt.GetRow(),
                                                 evt.GetCol(), evt.GetPosition()))
 
        # Another way to stay in a cell that has a bad value...
        row = self.GetGridCursorRow()
        col = self.GetGridCursorCol()
 
        if self.IsCellEditControlEnabled():
            self.HideCellEditControl()
            self.DisableCellEditControl()
 
        value = self.GetCellValue(row, col)
 
        if value == 'no good 2':
            return  # cancels the cell selection
 
        evt.Skip()
 
 
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
 

class Common(wx.Frame):
 
    def __init__(self, parent):
        wx.Frame.__init__(self, parent, wx.NewId(), "Common")
        self.panel = wx.Panel(self)
        self.grid = MyGrid(self)

        sizer = wx.BoxSizer(wx.VERTICAL)
        
        # toolbar
        save_ico = wx.ArtProvider.GetBitmap(wx.ART_FILE_SAVE, wx.ART_TOOLBAR, (16,16))
        self.toolbar = wx.ToolBar(self.panel, id=100, style=wx.TB_HORIZONTAL | wx.NO_BORDER |
                                        wx.TB_FLAT | wx.TB_TEXT)
        self.toolbar.AddSimpleTool(ID_UNDO, save_ico,
              'Undo', '')
        self.toolbar.AddSimpleTool(ID_REDO, wx.Bitmap('/projects/hira/tsangc/GaussianEmulator/development/human-icon-theme/16x16/stock/generic/stock_exit.png'),
              'Redo', '')
        self.toolbar.EnableTool(ID_UNDO, False)

        self.toolbar.EnableTool(ID_REDO, False)
        #self.toolbar.AddSeparator()
        #self.toolbar.AddSimpleTool(ID_EXIT, wx.Bitmap('/projects/hira/tsangc/GaussianEmulator/development/human-icon-theme/16x16/stock/generic/stock_exit.png'),'Quit', '')
        self.toolbar.Realize()
        self.toolbar.Bind(wx.EVT_TOOL, self.OnUndo, id=ID_UNDO)
        self.toolbar.Bind(wx.EVT_TOOL, self.OnRedo, id=ID_REDO)
        
        sizer.Add(self.toolbar, border=5)
        
        sizer.Add(self.grid, 1., wx.EXPAND, 5)
        self.panel.SetSizer(sizer)
        self.Fit()

    def OnUndo(self, event):
        if len(stockUndo) == 0:
            return

        a = stockUndo.pop()
        if len(stockUndo) == 0:
            self.toolbar.EnableTool(ID_UNDO, False)

        a.undo()
        stockRedo.append(a)
        self.toolbar.EnableTool(ID_REDO, True)

    def OnRedo(self, event):
        if len(stockRedo) == 0:
            return

        a = stockRedo.pop()
        if len(stockRedo) == 0:
            self.toolbar.EnableTool(ID_REDO, False)

        a.redo()
        stockUndo.append(a)

        self.toolbar.EnableTool(ID_UNDO, True)

print('start')
app = wx.App(0)
frame = Common(None)
frame.Show()
app.MainLoop()
