import numpy as np
import wx

########################################################################
class EmulatorFrame(wx.Dialog):

    #----------------------------------------------------------------------
    def __init__(self):
         """Constructor"""
         wx.Dialog.__init__(self, None, title="Bayesian Settings")
         panel = wx.Panel(self)
         box = wx.BoxSizer(wx.VERTICAL)

         list_textctrl = {'CSV Output name': '', 'Number of Cores': '7', 'Number of steps': '10000', 'Number of Nodes': '1'}
         self.output = {}
         
         for name, default_value in list_textctrl.iteritems():
             box_new = wx.BoxSizer(wx.HORIZONTAL)
             text = wx.StaticText(panel, -1, name)
             box_new.Add(text, 1, wx.EXPAND|wx.ALIGN_LEFT|wx.ALL,5) 
             textbox = wx.TextCtrl(panel, -1, default_value)
             box_new.Add(textbox, 1, wx.EXPAND|wx.ALIGN_LEFT|wx.ALL,5) 
             self.output[name] = textbox
             box.Add(box_new)

         """
         Checkbox for whether to clear previous results or not
         """
         box_clear = wx.BoxSizer(wx.HORIZONTAL)
         self.clear = wx.CheckBox(panel, label='Clear Previous Trace')
         #self.Bind(wx.EVT_CHECKBOX, self.onChecked)
         box_clear.Add(self.clear, 0, wx.ALIGN_CENTER)
         box.Add(box_clear)

         box_submit = wx.BoxSizer(wx.HORIZONTAL)
         self.submit = wx.Button(panel, wx.ID_OK, "Submit")
         box_submit.Add(self.submit, 0, wx.ALIGN_CENTER)
         box.Add(box_submit)

         self.Bind(wx.EVT_CLOSE, self.OnQuit)
         panel.SetSizer(box)

         panel_sizer = wx.BoxSizer(wx.HORIZONTAL)
         panel_sizer.Add(panel)
         self.SetSizer(panel_sizer)

         self.Fit()

    def AdditionalData(self,args):
        args['Output_name'] = self.output['CSV Output name'].GetValue()
        args['cores'] = int(self.output['Number of Cores'].GetValue())
        args['steps'] = int(self.output['Number of steps'].GetValue())
        args['nodes'] = int(self.output['Number of Nodes'].GetValue())
        if not self.clear.GetValue():
            args['concat'] = True
        
    def OnQuit(self, event):
        self.Destroy()
        return wx.ID_CANCEL
