import numpy as np
import wx

########################################################################
class TrainingFrame(wx.Dialog):

    #----------------------------------------------------------------------
    def __init__(self):
         """Constructor"""
         wx.Dialog.__init__(self, None, title="Second Frame")
         panel = wx.Panel(self)
         box = wx.BoxSizer(wx.VERTICAL)

         box1 = wx.BoxSizer(wx.HORIZONTAL)
         cblbl = wx.StaticText(panel, -1, label="Covariance function", style = wx.ALIGN_CENTRE)
         self.Cov_func = ['ARD', 'RBF']
         self.combo_cov_func = wx.Choice(panel, choices=self.Cov_func)
         self.combo_cov_func.SetSelection(0)
         box1.Add(cblbl, 1, wx.EXPAND|wx.ALIGN_LEFT|wx.ALL,5) 
         box1.Add(self.combo_cov_func, 1, wx.EXPAND|wx.ALIGN_LEFT|wx.ALL,5) 
         box.Add(box1)

         list_textctrl = {'Number of PCA Components': '3', 'PCA Fraction': '0.99', 'Initial scale': '0.5', 'Initial nugget': '1', 'Scale learning rate': '0.00003', 'Nugget learning rate': '0.00003', 'Maximum iterations': '1000'}
         self.output = {}
         
         for name, default_value in list_textctrl.iteritems():
             box_new = wx.BoxSizer(wx.HORIZONTAL)
             text = wx.StaticText(panel, -1, name)
             box_new.Add(text, 1, wx.EXPAND|wx.ALIGN_LEFT|wx.ALL,5) 
             textbox = wx.TextCtrl(panel, -1, default_value)
             box_new.Add(textbox, 1, wx.EXPAND|wx.ALIGN_LEFT|wx.ALL,5) 
             self.output[name] = textbox
             box.Add(box_new)

         box_submit = wx.BoxSizer(wx.HORIZONTAL)
         self.submit = wx.Button(panel, wx.ID_OK, "Submit")
         box_submit.Add(self.submit, 0, wx.ALIGN_CENTER)
         box.Add(box_submit)

         panel.SetSizer(box)

    def AdditionalData(self,args):
        args['covariancefunc'] = self.Cov_func[self.combo_cov_func.GetSelection()]
        args['principalcomp'] = int(self.output['Number of PCA Components'].GetValue())
        args['fraction'] = float(self.output['PCA Fraction'].GetValue())
        args['initialscale'] = np.fromstring(self.output['Initial scale'].GetValue(), dtype=np.float, sep=',')
        args['initialnugget'] = float(self.output['Initial nugget'].GetValue())
        args['scalerate'] = float(self.output['Scale learning rate'].GetValue())
        args['nuggetrate'] = float(self.output['Nugget learning rate'].GetValue())
        args['maxsteps'] = int(self.output['Maximum iterations'].GetValue())
        
