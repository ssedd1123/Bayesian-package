from pubsub import pub
import wx

class GridPopupMenu(wx.Menu):

    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        clear = self.Append(wx.ID_CLEAR, 'clear')
        copy = self.Append(wx.ID_COPY, 'copy')
        paste = self.Append(wx.ID_PASTE, 'paste')
        delete = self.Append(wx.ID_DELETE, 'delete')
        undo = self.Append(wx.ID_UNDO, 'undo')
        redo = self.Append(wx.ID_REDO, 'redo')
        
        self.Bind(wx.EVT_MENU, lambda evt: pub.sendMessage('Menu_Clear', obj=self.parent, evt=evt), clear)
        self.Bind(wx.EVT_MENU, lambda evt: pub.sendMessage('Menu_Delete', obj=self.parent, evt=evt), delete)
        self.Bind(wx.EVT_MENU, lambda evt: pub.sendMessage('Menu_Copy', obj=self.parent, evt=evt), copy)
        self.Bind(wx.EVT_MENU, lambda evt: pub.sendMessage('Menu_Paste', obj=self.parent, evt=evt), paste)
        self.Bind(wx.EVT_MENU, lambda evt: pub.sendMessage('Menu_Undo', obj=self.parent, evt=evt), undo)
        self.Bind(wx.EVT_MENU, lambda evt: pub.sendMessage('Menu_Redo', obj=self.parent, evt=evt), redo)


class GridToolBar(wx.ToolBar):

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.parent = parent
        undo_ico = wx.ArtProvider.GetBitmap(wx.ART_UNDO, wx.ART_TOOLBAR, (16,16))
        self.AddTool(wx.ID_UNDO, 'Undo', undo_ico, '')
        redo_ico = wx.ArtProvider.GetBitmap(wx.ART_REDO, wx.ART_TOOLBAR, (16,16))
        self.AddTool(wx.ID_REDO, 'Redo', redo_ico, '')
        open_ico = wx.ArtProvider.GetBitmap(wx.ART_FILE_OPEN, wx.ART_TOOLBAR, (16,16))
        self.AddTool(wx.ID_OPEN, 'Open csv', open_ico, '')
        print_ico = wx.ArtProvider.GetBitmap(wx.ART_PRINT, wx.ART_TOOLBAR, (16,16))
        self.AddTool(wx.ID_PRINT, 'Plot data', print_ico, '')
 
        self.Bind(wx.EVT_TOOL, lambda evt: pub.sendMessage('ToolBar_Undo', obj=self, evt=evt), id=wx.ID_UNDO)
        self.Bind(wx.EVT_TOOL, lambda evt: pub.sendMessage('ToolBar_Redo', obj=self, evt=evt), id=wx.ID_REDO)
        self.Bind(wx.EVT_TOOL, lambda evt: pub.sendMessage('ToolBar_Open', obj=self, evt=evt), id=wx.ID_OPEN)
        self.Bind(wx.EVT_TOOL, lambda evt: pub.sendMessage('ToolBar_Paint', obj=self, evt=evt), id=wx.ID_PRINT)

        self.EnableTool(wx.ID_UNDO, False)
        self.EnableTool(wx.ID_REDO, False)
        #self.Realize()
