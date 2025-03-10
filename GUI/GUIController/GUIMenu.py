import wx
from pubsub import pub


class GUIMenuBar(wx.MenuBar):
    def __init__(self, parent, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.parent = parent

        fileMenu = wx.Menu()
        SaveMenuItem = fileMenu.Append(wx.ID_SAVE, "Save", "")
        SaveModelNameMenuItem = fileMenu.Append(-1, "Save model name", "")
        AddCommentMenuItem = fileMenu.Append(-1, "Add comment", "")
        SaveAsMenuItem = fileMenu.Append(wx.ID_SAVEAS, "Save As", "")
        ReTrainMenuItem = fileMenu.Append(-1, "Re-train", "")
        OpenMenuItem = fileMenu.Append(wx.ID_OPEN, "Open", "")
        GenHyperCube = fileMenu.Append(-1, "Generate hyper-cube", "")

        emulatorMenu = wx.Menu()
        EmulatorCheckItem = emulatorMenu.Append(-1, "Check emulator", "")
        EmulatorItem = emulatorMenu.Append(-1, "Start Analysis", "")
        EmulatorChainedItem = emulatorMenu.Append(
            -1, "Start Chained Analysis", "")
        EmulatorIndividualItem = emulatorMenu.Append(
            -1, "Start individual Analysis", "")
        EvalEmuItem = emulatorMenu.Append(-1, "Eval emulator", "")

        plotMenu = wx.Menu()
        PlotPredictionItem = plotMenu.Append(-1, "Plot prediction", "")
        PlotPosteriorItem = plotMenu.Append(-1, "Plot posterior", "")
        TrainReportItem = plotMenu.Append(-1, "Training report", "")
        TraceSummaryItem = plotMenu.Append(-1, "Trace summary", "")
        TraceDiagnosisItem = plotMenu.Append(-1, "Trace Diagnosis", "")

        self.Append(fileMenu, "&File")
        self.Append(emulatorMenu, "&Emulator")
        self.Append(plotMenu, "&Plot")
        self.Bind(
            wx.EVT_MENU,
            lambda evt: pub.sendMessage("MenuBar_Open", obj=self, evt=evt),
            OpenMenuItem,
        )
        self.Bind(
            wx.EVT_MENU,
            lambda evt: pub.sendMessage("MenuBar_Save", obj=self, evt=evt),
            SaveMenuItem,
        )
        self.Bind(
            wx.EVT_MENU,
            lambda evt: pub.sendMessage("MenuBar_SaveModelName", obj=self, evt=evt),
            SaveModelNameMenuItem,
        )
        self.Bind(
            wx.EVT_MENU,
            lambda evt: pub.sendMessage("MenuBar_SaveNew", obj=self, evt=evt),
            SaveAsMenuItem,
        )
        self.Bind(
            wx.EVT_MENU,
            lambda evt: pub.sendMessage("MenuBar_AddComment", obj=self, evt=evt),
            AddCommentMenuItem,
        )
        self.Bind(
            wx.EVT_MENU,
            lambda evt: pub.sendMessage("MenuBar_ReTrain", obj=self, evt=evt),
            ReTrainMenuItem,
        )
        self.Bind(
            wx.EVT_MENU,
            lambda evt: pub.sendMessage(
                "MenuBar_GenHyperCube",
                obj=self,
                evt=evt),
            GenHyperCube,
        )
        self.Bind(
            wx.EVT_MENU,
            lambda evt: pub.sendMessage("MenuBar_Check", obj=self, evt=evt),
            EmulatorCheckItem,
        )
        self.Bind(
            wx.EVT_MENU,
            lambda evt: pub.sendMessage("MenuBar_Emulate", obj=self, evt=evt),
            EmulatorItem,
        )
        self.Bind(
            wx.EVT_MENU,
            lambda evt: pub.sendMessage(
                "MenuBar_ChainedEmulate",
                obj=self,
                evt=evt),
            EmulatorChainedItem,
        )
        self.Bind(
            wx.EVT_MENU,
            lambda evt: pub.sendMessage(
                "MenuBar_IndividualEmulate",
                obj=self,
                evt=evt),
            EmulatorIndividualItem,
        )
        self.Bind(
            wx.EVT_MENU,
            lambda evt: pub.sendMessage("MenuBar_EvalEmu", obj=self, evt=evt),
            EvalEmuItem,
        )
        self.Bind(
            wx.EVT_MENU,
            lambda evt: pub.sendMessage(
                "MenuBar_Prediction",
                obj=self,
                evt=evt),
            PlotPredictionItem,
        )
        self.Bind(
            wx.EVT_MENU,
            lambda evt: pub.sendMessage(
                "MenuBar_Posterior",
                obj=self,
                evt=evt),
            PlotPosteriorItem,
        )
        self.Bind(
            wx.EVT_MENU,
            lambda evt: pub.sendMessage("MenuBar_Report", obj=self, evt=evt),
            TrainReportItem,
        )
        self.Bind(
            wx.EVT_MENU,
            lambda evt: pub.sendMessage(
                "MenuBar_TraceSummary",
                obj=self,
                evt=evt),
            TraceSummaryItem,
        )
        self.Bind(
            wx.EVT_MENU,
            lambda evt: pub.sendMessage(
                "MenuBar_TraceDiagnosis",
                obj=self,
                evt=evt),
            TraceDiagnosisItem,
        )



    def GetParent(self):
        return self.parent
