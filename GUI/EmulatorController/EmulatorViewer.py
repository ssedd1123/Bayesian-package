import numpy as np
import wx
import sys
import os
import time
import pandas as pd
import pickle
from matplotlib.figure import Figure
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg
from matplotlib.backends.backend_wx import NavigationToolbar2Wx
from GUI.EmulatorController.SliderCustom import CustomSlider
from pubsub import pub
from numpy import array 

import Utilities.Utilities as utl
from Preprocessor.PipeLine import * 

class EmulatorViewer(wx.Frame):

    def __init__(self, title, mins, maxs, values, *args, **kw):
        super().__init__(*args, **kw)
        panel = wx.Panel(self)
        panel.SetBackgroundColour(wx.Colour('White'))

        self.fig = Figure((5, 4), 75)
        # Adjust the subplots region to leave some space for the sliders and buttons
        self.fig.subplots_adjust(left=0.25, bottom=0.25)
        self.ax = self.fig.add_axes([0.1,0.2,0.85,0.75])
        self.canvas = FigureCanvasWxAgg(panel, -1, self.fig)
        self.toolbar = NavigationToolbar2Wx(self.canvas)  # matplotlib toolbar
        self.toolbar.Realize()

        self.sliders = []
        mins = np.atleast_1d(mins).flatten()
        maxs = np.atleast_1d(maxs).flatten()
        values = np.array(values).flatten()
        for idx, (min_, max_, val, tit) in enumerate(zip(mins, maxs, values, title)):
            slider = CustomSlider(panel, val, min_, max_, title=tit, pads=50)
            slider.idx = idx
            self.sliders.append(slider)
        #self.highlighter2 = CustomSlider(panel, 1, 1, 8) 
        self.clb = wx.CheckListBox(panel, -1, (50, -1), wx.DefaultSize, [])
        self.retrain = wx.Button(panel, -1, 'Retrain')

        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        rsizer = wx.BoxSizer(wx.VERTICAL)
        # This way of adding to sizer allows resizing
        rsizer.Add(self.canvas, 1, wx.EXPAND)
        for slider in self.sliders:
            rsizer.Add(slider, 0, wx.EXPAND | wx.RIGHT)# | wx.LEFT | wx.RIGHT, 20)
        rsizer.Add(self.toolbar, 0, wx.GROW | wx.RIGHT)

        lsizer = wx.BoxSizer(wx.VERTICAL)
        lsizer.Add(self.clb, 1, wx.EXPAND)
        lsizer.Add(self.retrain, 0)

        hsizer.Add(lsizer, 0, wx.EXPAND)
        hsizer.Add(rsizer, 1, wx.EXPAND)

        self.Bind(wx.EVT_LISTBOX, self.OnListBox)
        self.Bind(wx.EVT_CHECKLISTBOX, self.OnCheckListBox)
        self.Bind(wx.EVT_BUTTON, self.OnRetrain)
        panel.SetSizer(hsizer)
        self.Layout()
        self.Centre()

    def OnRetrain(self, evt):
        pub.sendMessage('Emulator_Retrain', obj=self, evt=evt)

    def OnListBox(self, evt):
        pub.sendMessage('Emulator_ListSelect', obj=self, evt=evt)

    def OnCheckListBox(self, evt):
        pub.sendMessage('Emulator_CheckSelect', obj=self, evt=evt)

    def RefreshFig(self):
        self.fig.canvas.draw()#_idle()

class EmulatorController:

    def __init__(self, store_file):
        self.store_file = store_file
        args = utl.GetTrainedEmulator(store_file)
        emulator = args[0] 
        prior = args[1]
        self.exp_Y = args[2] 
        self.exp_Yerr = args[3] 
        self.model_X = args[4] 
        self.model_Y = args[5]
        init_training_idx = args[6]

        self.viewer = EmulatorViewer(prior.index, prior['Min'], prior['Max'], 0.5*(prior['Min'] + prior['Max']),size=(1000,700), parent=None)
        self.model = emulator

        self.current_values = 0.5*(prior['Min'] + prior['Max']).values.flatten()
        init_value, init_cov = self.model.Predict(self.current_values)

        self.X = np.arange(init_value.shape[1])
        self.line, _, (self.bars,) = self.viewer.ax.errorbar(self.X, init_value.flatten(), yerr=np.sqrt(np.diag(np.squeeze(init_cov))).flatten(),
                                                             marker='o', linewidth=2, color='red')
        self.bg_line, _, (self.bg_bars,)= self.viewer.ax.errorbar(self.X, self.exp_Y, yerr=self.exp_Yerr, 
                                                                  marker='o', linewidth=2, color='b')
        par_name = [name[0:15] if len(name) > 14 else name for name in list(self.model_Y)]
        self.viewer.ax.set_xticks(self.X)
        self.viewer.ax.set_xticklabels(par_name, rotation=45)
        self.viewer.RefreshFig()


        self.items = ['Check All', 'Exp'] + self.model_Y.index.astype(str).tolist()
        self.viewer.clb.SetItems(self.items)
        self.viewer.clb.SetCheckedItems([0, 1] + [item+2 for item in init_training_idx])

        pub.subscribe(self.CheckObj, 'Slider_Value', func=self.OnSlider)
        pub.subscribe(self.CheckObj, 'Emulator_ListSelect', func=self.OnListSelect)
        pub.subscribe(self.CheckObj, 'Emulator_CheckSelect', func=self.OnCheckboxSelect)
        pub.subscribe(self.CheckObj, 'Emulator_Retrain', func=self.OnRetrain)

    def CheckObj(self, func, obj, evt):
        orig_obj = obj
        while True:
            if obj is self.model or obj is self.viewer:
                func(orig_obj, evt)
                break
            obj = obj.GetParent()
            if obj is None:
                break

    def OnRetrain(self, obj, evt):
        checked = np.array(self.viewer.clb.GetCheckedItems(), dtype=int) - 2
        training_idx = checked[checked >= 0]
        store = pd.HDFStore(self.store_file, 'a')
        store['Training_idx'] = pd.DataFrame(training_idx, dtype=int)
        store.close()

    def OnListSelect(self, obj, evt):
        selected = evt.GetString()
        if selected == 'Exp':
            self.SetErrorBar(self.bg_line, self.bg_bars, self.exp_Y, self.exp_Yerr)
        elif selected.isdigit():
            idx = int(selected)
            self.SetErrorBar(self.bg_line, self.bg_bars, self.model_Y.loc[idx], 0)
            self.viewer.RefreshFig()

            selected_x = self.model_X.loc[idx].values
            for i, x in enumerate(selected_x):
                self.viewer.sliders[i].Highlight(x)
        
        
         
    def OnCheckboxSelect(self, obj, evt):
        selected = evt.GetString()
        if selected == 'Check All' and self.viewer.clb.IsChecked(0):
            self.viewer.clb.SetCheckedItems(np.arange(0, len(self.items)))
        else:
            self.viewer.clb.Check(0, False)
        checked = np.array(self.viewer.clb.GetCheckedItems(), dtype=np.int) - 2
        checked = checked[checked >= 0]
        self.model.Fit(self.model_X.loc[checked].values, self.model_Y.loc[checked].values)
        self.ChangeValue(0, self.current_values[0])


    def ChangeValue(self, idx, value):
        self.current_values[idx] = value
        val, cov = self.model.Predict(self.current_values)
        self.SetErrorBar(self.line, self.bars, val.flatten(), np.sqrt(np.diag(np.squeeze(cov))).flatten()) 
        self.viewer.RefreshFig()

    def OnSlider(self, obj, value):
        self.ChangeValue(obj.idx, value)

    def SetErrorBar(self, line, errorbar, values, err):
        yerr_top = values + err
        yerr_bot = values - err
        line.set_ydata(values)
        x_base = line.get_xdata()
        new_segments = [np.array([[x, yt], [x, yb]]) for
                        x, yt, yb in zip(x_base, yerr_top, yerr_bot)]
        errorbar.set_segments(new_segments)
        self.viewer.RefreshFig()
 

if __name__ == '__main__':

    app = wx.App()
    controller = EmulatorController('result/newhist.h5')
    ex = controller.viewer
    ex.Show()
    app.MainLoop()
