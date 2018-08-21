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
    def __init__(self, parent, size=(100,100), has_toolbar=True):
        """Constructor"""
        self.has_toolbar = has_toolbar
        self.stockUndo = []
        self.stockRedo = []
        gridlib.Grid.__init__(self, parent, size=size)
        self.num_row = size[0]
        self.num_col = size[1]
        self.CreateGrid(self.num_row, self.num_col)
        self.parent = parent
        self.InitUI()

    def InitUI(self):
        self.selected_data = None
        self.selected_coords = None
 
        # test all the events
        self.Bind(gridlib.EVT_GRID_CELL_RIGHT_CLICK, self.ShowMenu)
        self.Bind(gridlib.EVT_GRID_RANGE_SELECT, self.OnRangeSelect)
        self.Bind(gridlib.EVT_GRID_CELL_CHANGED, self.OnCellChange)


    def OnRangeSelect(self, event):
        self.selected_coords = [list(event.GetTopLeftCoords()), list(event.GetBottomRightCoords())]
        self.selected_data = self.GetRange(self.selected_coords)   


    def _SetValue(self, coords, data):
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

    def SetValue(self, coords, data):
        undo = self._SetValue(coords, data)
        self.stockUndo.append(undo)

        if self.stockRedo:
            del self.stockRedo[:]
            if self.has_toolbar:
                toolbar.EnableTool(ID_REDO, False)

    def _ClearRange(self, coords):
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

    def ClearRange(self, coords):
        undo = self._ClearRange(coords)
        self.stockUndo.append(undo)

        if self.stockRedo:
            del self.stockRedo[:]
            if self.has_toolbar:
                self.parent.toolbar.EnableTool(ID_REDO, False)

    def ClearAll(self):
        self.ClearRange([[0,0], [self.num_row-1, self.num_col-1]])

    def ShowMenu(self, event):
        cell = self.GetSelectedCells()
        if not cell:
            if self.GetSelectionBlockTopLeft():
                top_left = self.GetSelectionBlockTopLeft()[0]
                bottom_right = self.GetSelectionBlockBottomRight()[0]
            elif self.GetSelectedCols():
                col = self.GetSelectedCols()
                top_left = [0, col[0]]
                bottom_right = [self.num_row-1, col[-1]]
            elif self.GetSelectedRows():
                row = self.GetSelectedRows()
                top_left = [row[0], 0]
                bottom_right = [row[-1], self.num_col-1]
            else:
                top_left = bottom_right = [event.GetRow(), event.GetCol()]
        else:
            top_left = cell
            bottom_right = cell
        self.selected_coords = [list(top_left), list(bottom_right)]
        self.selected_data = self.GetRange(self.selected_coords)    
        win = self.PopupMenu(GridPopupMenu(self), event.GetPosition())# self.ScreenToClient(pos))
        
          

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
            if len(data) == 0:
                break
        return data 
                
        

 
    def OnCellChange(self, event):
        if self.has_toolbar:
            if (self.parent.toolbar.GetToolEnabled(ID_UNDO) == False):
                self.parent.toolbar.EnableTool(ID_UNDO, True)
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
            if self.has_toolbar:
                self.parent.toolbar.EnableTool(ID_REDO, False)

 
