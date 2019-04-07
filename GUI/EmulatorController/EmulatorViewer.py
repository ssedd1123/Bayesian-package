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


        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer = wx.BoxSizer(wx.VERTICAL)
        # This way of adding to sizer allows resizing
        sizer.Add(self.canvas, 1, wx.EXPAND)
        for slider in self.sliders:
            sizer.Add(slider, 0, wx.EXPAND | wx.RIGHT)# | wx.LEFT | wx.RIGHT, 20)
        sizer.Add(self.toolbar, 0, wx.GROW | wx.RIGHT)

        hsizer.Add(self.clb, 0, wx.EXPAND)
        hsizer.Add(sizer, 1, wx.EXPAND)

        self.Bind(wx.EVT_LISTBOX, self.OnListBox)
        self.Bind(wx.EVT_CHECKLISTBOX, self.OnCheckListBox)
        panel.SetSizer(hsizer)
        self.Layout()
        self.Centre()

    def OnListBox(self, evt):
        pub.sendMessage('Emulator_ListSelect', obj=self, evt=evt)

    def OnCheckListBox(self, evt):
        pub.sendMessage('Emulator_CheckSelect', obj=self, evt=evt)

    def RefreshFig(self):
        self.fig.canvas.draw()#_idle()

class EmulatorController:

    def __init__(self, store):
        prior = store['PriorAndConfig']
        self.exp_Y = store['Exp_Y']
        self.exp_Yerr = store['Exp_YErr']
        self.model_Y = store['Model_Y']
        self.model_X = store['Model_X']

        config = store.get_storer('PriorAndConfig').attrs.my_attribute
        emulator  = eval(config['repr'])
        emulator.Fit(self.model_X.values, self.model_Y.values)  

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
        self.viewer.clb.SetCheckedItems(np.arange(0, len(self.items)))

        pub.subscribe(self.CheckObj, 'Slider_Value', func=self.OnSlider)
        pub.subscribe(self.CheckObj, 'Emulator_ListSelect', func=self.OnListSelect)
        pub.subscribe(self.CheckObj, 'Emulator_CheckSelect', func=self.OnCheckboxSelect)

    def CheckObj(self, func, obj, evt):
        orig_obj = obj
        while True:
            if obj is self.model or obj is self.viewer:
                func(orig_obj, evt)
                break
            obj = obj.GetParent()
            if obj is None:
                break

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
        print(checked)
        print(self.model_Y)
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
    store = pd.HDFStore('result/newhist.h5')

    app = wx.App()
    controller = EmulatorController(store)
    ex = controller.viewer
    ex.Show()
    app.MainLoop()
