import wx
from pubsub import pub

class GUIMenuBar(wx.MenuBar):

    def __init__(self, parent, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.parent = parent

        fileMenu = wx.Menu()
        SaveMenuItem = fileMenu.Append(wx.ID_SAVE, 'Save', '')
        SaveAsMenuItem = fileMenu.Append(wx.ID_SAVEAS, 'Save As', '')
        OpenMenuItem = fileMenu.Append(wx.ID_OPEN, 'Open', '')

        emulatorMenu = wx.Menu()
        EmulatorCheckItem = emulatorMenu.Append(-1, 'Check emulator', '')
        EmulatorItem = emulatorMenu.Append(-1, 'Start Analysis', '')

        self.Append(fileMenu, '&File')
        self.Append(emulatorMenu, '&Emulator')
        self.Bind(wx.EVT_MENU, lambda evt: pub.sendMessage('MenuBar_Open', obj=self, evt=evt), OpenMenuItem)
        self.Bind(wx.EVT_MENU, lambda evt: pub.sendMessage('MenuBar_Save', obj=self, evt=evt), SaveMenuItem)
        self.Bind(wx.EVT_MENU, lambda evt: pub.sendMessage('MenuBar_SaveNew', obj=self, evt=evt), SaveAsMenuItem)
        self.Bind(wx.EVT_MENU, lambda evt: pub.sendMessage('MenuBar_Check', obj=self, evt=evt), EmulatorCheckItem)
        self.Bind(wx.EVT_MENU, lambda evt: pub.sendMessage('MenuBar_Emulate', obj=self, evt=evt), EmulatorItem)
 
    def GetParent(self):
        return self.parent
