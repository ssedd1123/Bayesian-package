#!/usr/bin/python

# spreadsheet.py

from wx.lib import sheet
import wx


class MySheet(sheet.CSheet):
    def __init__(self, parent):
        sheet.CSheet.__init__(self, parent)
        self.row = self.col = 0
        self.SetNumberRows(55)
        self.SetNumberCols(25)

        for i in range(55):
            self.SetRowSize(i, 20)

    def OnGridSelectCell(self, event):
        self.row, self.col = event.GetRow(), event.GetCol()
        control = self.GetParent().GetParent().position
        value =  self.GetColLabelValue(self.col) + self.GetRowLabelValue(self.row)
        control.SetValue(value)
        event.Skip()


class Newt(wx.Frame):
    def __init__(self, parent, id, title):
        wx.Frame.__init__(self, parent, -1, title, size = ( 550, 500))

        fonts = ['Times New Roman', 'Times', 'Courier', 'Courier New', 'Helvetica', 'Sans', 'verdana', 'utkal', 'aakar', 'Arial']
        box = wx.BoxSizer(wx.VERTICAL)
        menuBar = wx.MenuBar()

        menu1 = wx.Menu()
        menuBar.Append(menu1, '&File')
        menu2 = wx.Menu()
        menuBar.Append(menu2, '&Edit')
        menu3 = wx.Menu()
        menuBar.Append(menu3, '&Edit')
        menu4 = wx.Menu()
        menuBar.Append(menu4, '&Insert')
        menu5 = wx.Menu()
        menuBar.Append(menu5, 'F&ormat')
        menu6 = wx.Menu()
        menuBar.Append(menu6, '&Tools')
        menu7 = wx.Menu()
        menuBar.Append(menu7, '&Data')

        menu7 = wx.Menu()
        menuBar.Append(menu7, '&Help')


        self.SetMenuBar(menuBar)

        toolbar1 = wx.ToolBar(self, -1, style= wx.TB_HORIZONTAL | wx.NO_BORDER | wx.TB_FLAT | wx.TB_TEXT)
        toolbar1.AddSimpleTool(-1, wx.Image('icons/stock_new.png', wx.BITMAP_TYPE_PNG).ConvertToBitmap(), 'New', '')
        toolbar1.AddSimpleTool(-1, wx.Image('icons/stock_open.png', wx.BITMAP_TYPE_PNG).ConvertToBitmap(), 'Open', '')
        toolbar1.AddSimpleTool(-1, wx.Image('icons/stock_save.png', wx.BITMAP_TYPE_PNG).ConvertToBitmap(), 'Save', '')
        toolbar1.AddSeparator()
        toolbar1.AddSimpleTool(-1, wx.Image('icons/stock_cut.png', wx.BITMAP_TYPE_PNG).ConvertToBitmap(), 'Cut', '')
        toolbar1.AddSimpleTool(-1, wx.Image('icons/stock_copy.png', wx.BITMAP_TYPE_PNG).ConvertToBitmap(), 'Copy', '')
        toolbar1.AddSimpleTool(-1, wx.Image('icons/stock_paste.png', wx.BITMAP_TYPE_PNG).ConvertToBitmap(), 'Paste', '')
        toolbar1.AddSimpleTool(-1, wx.Image('icons/stock_delete.png', wx.BITMAP_TYPE_PNG).ConvertToBitmap(), 'Delete', '')
        toolbar1.AddSeparator()
        toolbar1.AddSimpleTool(-1, wx.Image('icons/stock_undo.png', wx.BITMAP_TYPE_PNG).ConvertToBitmap(), 'Undo', '')
        toolbar1.AddSimpleTool(-1, wx.Image('icons/stock_redo.png', wx.BITMAP_TYPE_PNG).ConvertToBitmap(), 'Redo', '')
        toolbar1.AddSeparator()
        toolbar1.AddSimpleTool(-1, wx.Image('icons/incr22.png', wx.BITMAP_TYPE_PNG).ConvertToBitmap(), 'Sort Increasing', '')
        toolbar1.AddSimpleTool(-1, wx.Image('icons/decr22.png', wx.BITMAP_TYPE_PNG).ConvertToBitmap(), 'Sort Decreasing', '')
        toolbar1.AddSeparator()
        toolbar1.AddSimpleTool(-1, wx.Image('icons/graph_guru_24.xpm', wx.BITMAP_TYPE_XPM).ConvertToBitmap(), 'Chart', '')
        toolbar1.AddSeparator()
        toolbar1.AddSimpleTool(-1, wx.Image('icons/stock_exit.png', wx.BITMAP_TYPE_PNG).ConvertToBitmap(), 'Quit', '')

        toolbar1.Realize()

        toolbar2 = wx.ToolBar(self, wx.TB_HORIZONTAL | wx.TB_TEXT)
        self.position = wx.TextCtrl(toolbar2)
        font = wx.ComboBox(toolbar2, -1, value = 'Times', choices=fonts, size=(100, -1), style=wx.CB_DROPDOWN)
        font_height = wx.ComboBox(toolbar2, -1, value = '10',  choices=['10', '11', '12', '14', '16'], size=(50, -1), style=wx.CB_DROPDOWN)
        toolbar2.AddControl(self.position)
        toolbar2.AddControl(wx.StaticText(toolbar2, -1, '  '))
        toolbar2.AddControl(font)
        toolbar2.AddControl(wx.StaticText(toolbar2, -1, '  '))
        toolbar2.AddControl(font_height)
        toolbar2.AddSeparator()
        bold = wx.Image('icons/stock_text_bold.png', wx.BITMAP_TYPE_PNG).ConvertToBitmap()
        toolbar2.AddCheckTool(-1, bold , shortHelp = 'Bold')
        italic = wx.Image('icons/stock_text_italic.png', wx.BITMAP_TYPE_PNG).ConvertToBitmap()
        toolbar2.AddCheckTool(-1, italic,  shortHelp = 'Italic')
        under = wx.Image('icons/stock_text_underline.png', wx.BITMAP_TYPE_PNG).ConvertToBitmap()
        toolbar2.AddCheckTool(-1, under, shortHelp = 'Underline')
        toolbar2.AddSeparator()
        toolbar2.AddSimpleTool(-1, wx.Image('icons/stock_text_align_left.png', wx.BITMAP_TYPE_PNG).ConvertToBitmap(), 'Align Left', '')
        toolbar2.AddSimpleTool(-1, wx.Image('icons/stock_text_align_center.png', wx.BITMAP_TYPE_PNG).ConvertToBitmap(), 'Center', '')
        toolbar2.AddSimpleTool(-1, wx.Image('icons/stock_text_align_right.png', wx.BITMAP_TYPE_PNG).ConvertToBitmap(), 'Align Right', '')

        box.Add(toolbar1, border=5)
        box.Add((5,5) , 0)
        box.Add(toolbar2)
        box.Add((5,10) , 0)

        toolbar2.Realize()
        self.SetSizer(box)
        notebook = wx.Notebook(self, -1, style=wx.BOTTOM)

        sheet1 = MySheet(notebook)
        sheet2 = MySheet(notebook)
        sheet3 = MySheet(notebook)
        sheet1.SetFocus()

        notebook.AddPage(sheet1, 'Sheet1')
        notebook.AddPage(sheet2, 'Sheet2')
        notebook.AddPage(sheet3, 'Sheet3')

        box.Add(notebook, 1, wx.EXPAND)

        self.CreateStatusBar()
        self.Centre()
        self.Show(True)

app = wx.App(0)
newt = Newt(None, -1, 'SpreadSheet')
app.MainLoop()
