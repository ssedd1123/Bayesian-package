from __future__ import print_function


import numpy as np
from copy import deepcopy
import wx
import wx.grid as gridlib

from GridMenu import GridPopupMenu
from ID import *


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
        self.Bind(gridlib.EVT_GRID_CELL_RIGHT_CLICK, self.ShowMenu)#self.OnCellRightDClick)
 
        self.Bind(gridlib.EVT_GRID_RANGE_SELECT, self.OnRangeSelect)
        self.Bind(gridlib.EVT_GRID_CELL_CHANGE, self.OnCellChange)
        self.Bind(gridlib.EVT_GRID_SELECT_CELL, self.OnSelectCell)
 

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
 
 
        