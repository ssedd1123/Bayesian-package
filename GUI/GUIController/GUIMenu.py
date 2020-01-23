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
        GenHyperCube = fileMenu.Append(-1, 'Generate hyper-cube', '')

        emulatorMenu = wx.Menu()
        EmulatorCheckItem = emulatorMenu.Append(-1, 'Check emulator', '')
        EmulatorItem = emulatorMenu.Append(-1, 'Start Analysis', '')

        plotMenu = wx.Menu()
        PlotPosteriorItem = plotMenu.Append(-1, 'Plot posterior', '')
        PlotCorrelationItem = plotMenu.Append(-1, 'Plot correlation', '')
        TrainReportItem = plotMenu.Append(-1, 'Training report', '')

        self.Append(fileMenu, '&File')
        self.Append(emulatorMenu, '&Emulator')
        self.Append(plotMenu, '&Plot')
        self.Bind(wx.EVT_MENU, lambda evt: pub.sendMessage('MenuBar_Open', obj=self, evt=evt), OpenMenuItem)
        self.Bind(wx.EVT_MENU, lambda evt: pub.sendMessage('MenuBar_Save', obj=self, evt=evt), SaveMenuItem)
        self.Bind(wx.EVT_MENU, lambda evt: pub.sendMessage('MenuBar_SaveNew', obj=self, evt=evt), SaveAsMenuItem)
        self.Bind(wx.EVT_MENU, lambda evt: pub.sendMessage('MenuBar_GenHyperCube', obj=self, evt=evt), GenHyperCube)
        self.Bind(wx.EVT_MENU, lambda evt: pub.sendMessage('MenuBar_Check', obj=self, evt=evt), EmulatorCheckItem)
        self.Bind(wx.EVT_MENU, lambda evt: pub.sendMessage('MenuBar_Emulate', obj=self, evt=evt), EmulatorItem)
        self.Bind(wx.EVT_MENU, lambda evt: pub.sendMessage('MenuBar_Posterior', obj=self, evt=evt), PlotPosteriorItem)
        self.Bind(wx.EVT_MENU, lambda evt: pub.sendMessage('MenuBar_Correlation', obj=self, evt=evt), PlotCorrelationItem)
        self.Bind(wx.EVT_MENU, lambda evt: pub.sendMessage('MenuBar_Report', obj=self, evt=evt), TrainReportItem)



 
    def GetParent(self):
        return self.parent
