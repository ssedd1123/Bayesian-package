import wx

class TrainingProgressFrame(wx.Frame):

    def __init__(self, npca, *args, col_labels=None, text_label=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.gauges = [wx.Gauge(self) for i in range(npca)]
        if col_labels is not None:
            assert len(col_labels) == npca, 'Number of column labels does not agree with the number of progresses to display'
            for label in col_labels:
                self.gaugesLabel = [wx.StaticText(self, label=label) for i in range(npca)]
        else:
            self.gaugesLabel = [wx.StaticText(self, label='PCA %d' % i) for i in range(npca)]
        overallSizer = wx.BoxSizer(wx.VERTICAL)

        '''
        Description of the frame
        '''
        titleSizer = wx.BoxSizer(wx.HORIZONTAL)
        if text_label is not None:
            titleSizer.Add(wx.StaticText(self, label=text_label), 0, wx.ALIGN_CENTER | wx.EXPAND | wx.ALL, 10)
        else:
            titleSizer.Add(wx.StaticText(self, label='Training progress of emulators on each PCA component'), 0, wx.ALIGN_CENTER | wx.EXPAND | wx.ALL, 10)

        '''
        progress display for each PCA components
        '''
        progressSizer = wx.FlexGridSizer(2, npca,0)
        for i, (gauge, text) in enumerate(zip(self.gauges, self.gaugesLabel)):
            progressSizer.Add(text, 0, wx.ALIGN_CENTER | wx.ALL, 10)
            progressSizer.Add(gauge, 0, wx.ALL | wx.EXPAND, 5)
        progressSizer.AddGrowableCol(1)

        overallSizer.Add(titleSizer)
        overallSizer.Add(progressSizer, 1, wx.EXPAND, 0)

        self.SetSizer(overallSizer)
        width, _ = self.GetSize()
        self.SetMinSize((width, -1))
        self.Fit()
        self.NCompleted = 0

   
    def updateProgress(self, progress):
        if len(self.gauges) > self.NCompleted:
            self.gauges[self.NCompleted].SetValue(int(progress))
        else:
            self.gauges[-1].SetValue(int(progress))
        if progress >= 100:
            self.NCompleted = self.NCompleted + 1

        # update cycle the progress bar
        if len(self.gauges) == 1 and self.NCompleted >= 1:
            self.gaugesLabel[-1].SetLabel('PCA %d' % self.NCompleted)
        wx.Yield()
     
#----------------------------------------------------------------------
# Run the program
if __name__ == "__main__":
    app = wx.App(False)
    dlg = TrainingProgressFrame(3, None, -1, 'Training progress', size=(300,-1))
    dlg.Show()

    import time
    import numpy as np
    for n in range(3):
        for progress in np.linspace(0,100,100):
           dlg.updateProgress(progress)
           time.sleep(0.01)
 

    app.MainLoop()

