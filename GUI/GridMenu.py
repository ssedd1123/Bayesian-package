from __future__ import print_function

import numpy as np
from copy import deepcopy

import wx
import wx.grid as gridlib
from ID import *

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


