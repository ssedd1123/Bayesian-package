import wx
from pubsub import pub


class GridPopupMenu(wx.Menu):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.clear = self.Append(wx.ID_CLEAR, "clear")
        self.copy = self.Append(wx.ID_COPY, "copy")
        self.paste = self.Append(wx.ID_PASTE, "paste")
        self.paste_special = self.Append(wx.ID_ANY, "paste special")
        self.delete = self.Append(wx.ID_DELETE, "delete")
        self.undo = self.Append(wx.ID_UNDO, "undo")
        self.redo = self.Append(wx.ID_REDO, "redo")

        self.Bind(
            wx.EVT_MENU,
            lambda evt: pub.sendMessage(
                "Menu_Clear",
                obj=self.parent,
                evt=evt),
            self.clear,
        )
        self.Bind(
            wx.EVT_MENU,
            lambda evt: pub.sendMessage(
                "Menu_Delete",
                obj=self.parent,
                evt=evt),
            self.delete,
        )
        self.Bind(
            wx.EVT_MENU,
            lambda evt: pub.sendMessage("Menu_Copy", obj=self.parent, evt=evt),
            self.copy,
        )
        self.Bind(
            wx.EVT_MENU,
            lambda evt: pub.sendMessage(
                "Menu_PasteSpecial",
                obj=self.parent,
                evt=evt),
            self.paste_special,
        )
        self.Bind(
            wx.EVT_MENU,
            lambda evt: pub.sendMessage("Menu_Redo", obj=self.parent, evt=evt),
            self.redo,
        )

        self.Bind(
            wx.EVT_MENU,
            lambda evt: pub.sendMessage("Menu_Undo", obj=self.parent, evt=evt),
            self.undo,
        )
        self.Bind(
            wx.EVT_MENU,
            lambda evt: pub.sendMessage("Menu_Redo", obj=self.parent, evt=evt),
            self.redo,
        )


class GridToolBar(wx.ToolBar):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.parent = parent
        undo_ico = wx.ArtProvider.GetBitmap(
            wx.ART_UNDO, wx.ART_TOOLBAR, (16, 16))
        self.AddTool(wx.ID_UNDO, "Undo", undo_ico, "")
        redo_ico = wx.ArtProvider.GetBitmap(
            wx.ART_REDO, wx.ART_TOOLBAR, (16, 16))
        self.AddTool(wx.ID_REDO, "Redo", redo_ico, "")
        open_ico = wx.ArtProvider.GetBitmap(
            wx.ART_FILE_OPEN, wx.ART_TOOLBAR, (16, 16))
        self.AddTool(wx.ID_OPEN, "Open csv", open_ico, "")
        clear_ico = wx.ArtProvider.GetBitmap(
            wx.ART_CROSS_MARK, wx.ART_TOOLBAR, (16, 16))
        self.AddTool(wx.ID_CLEAR, "Clear Content", clear_ico, "")
        delete_ico = wx.ArtProvider.GetBitmap(
            wx.ART_DELETE, wx.ART_TOOLBAR, (16, 16))
        self.AddTool(wx.ID_DELETE, "Clear All", delete_ico, "")
        paste_ico = wx.ArtProvider.GetBitmap(
            wx.ART_PASTE, wx.ART_TOOLBAR, (16, 16))
        self.AddTool(wx.ID_PASTE, "Paste Special", paste_ico, "")

        # print_ico = wx.ArtProvider.GetBitmap(wx.ART_PRINT, wx.ART_TOOLBAR, (16,16))
        # self.AddTool(wx.ID_PRINT, 'Plot data', print_ico, '')

        self.Bind(
            wx.EVT_TOOL,
            lambda evt: pub.sendMessage("ToolBar_Undo", obj=self, evt=evt),
            id=wx.ID_UNDO,
        )
        self.Bind(
            wx.EVT_TOOL,
            lambda evt: pub.sendMessage("ToolBar_Redo", obj=self, evt=evt),
            id=wx.ID_REDO,
        )
        self.Bind(
            wx.EVT_TOOL,
            lambda evt: pub.sendMessage("ToolBar_Open", obj=self, evt=evt),
            id=wx.ID_OPEN,
        )
        self.Bind(
            wx.EVT_TOOL,
            lambda evt: pub.sendMessage(
                "ToolBar_ClearContent",
                obj=self,
                evt=evt),
            id=wx.ID_CLEAR,
        )
        self.Bind(
            wx.EVT_TOOL,
            lambda evt: pub.sendMessage("ToolBar_ClearAll", obj=self, evt=evt),
            id=wx.ID_DELETE,
        )
        self.Bind(
            wx.EVT_TOOL,
            lambda evt: pub.sendMessage("ToolBar_PasteSpecial", obj=self, evt=evt),
            id=wx.ID_PASTE,
        )

        # self.Bind(wx.EVT_TOOL, lambda evt: pub.sendMessage('ToolBar_Paint', obj=self, evt=evt), id=wx.ID_PRINT)

        self.EnableTool(wx.ID_UNDO, False)
        self.EnableTool(wx.ID_REDO, False)
        self.Realize()


class PriorToolBar(GridToolBar):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        update_ico = wx.ArtProvider.GetBitmap(
            wx.ART_TICK_MARK, wx.ART_TOOLBAR, (16, 16))
        self.AddTool(wx.ID_REFRESH, "Auto fill", update_ico, "")
        self.Realize()

        self.Bind(
            wx.EVT_TOOL,
            lambda evt: pub.sendMessage(
                "PriorToolBar_Refresh",
                obj=self,
                evt=evt),
            id=wx.ID_REFRESH,
        )
