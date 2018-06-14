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

         box2 = wx.BoxSizer(wx.HORIZONTAL)
         l1 = wx.StaticText(panel, -1, "Number of PCA Components")
         box2.Add(l1, 1, wx.EXPAND|wx.ALIGN_LEFT|wx.ALL,5) 
         self.pca_comp = wx.TextCtrl(panel, -1, '3')
         box2.Add(self.pca_comp, 1, wx.EXPAND|wx.ALIGN_LEFT|wx.ALL,5) 
         box.Add(box2)

         box3 = wx.BoxSizer(wx.HORIZONTAL)
         l2 = wx.StaticText(panel, -1, "Initial scale")
         box3.Add(l2, 1, wx.EXPAND|wx.ALIGN_LEFT|wx.ALL,5)
         self.init_scale = wx.TextCtrl(panel, -1, '0.5')
         box3.Add(self.init_scale, 1, wx.EXPAND|wx.ALIGN_LEFT|wx.ALL,5)
         box.Add(box3)

         box4 = wx.BoxSizer(wx.HORIZONTAL)
         l3 = wx.StaticText(panel, -1, "Initial nugget")
         box4.Add(l3, 1, wx.EXPAND|wx.ALIGN_LEFT|wx.ALL,5)
         self.init_nugget = wx.TextCtrl(panel, -1, '1')
         box4.Add(self.init_nugget, 1, wx.EXPAND|wx.ALIGN_LEFT|wx.ALL,5)
         box.Add(box4)

         box5 = wx.BoxSizer(wx.HORIZONTAL)
         l4 = wx.StaticText(panel, -1, "Scale learning rate")
         box5.Add(l4, 1, wx.EXPAND|wx.ALIGN_LEFT|wx.ALL,5)
         self.scalerate = wx.TextCtrl(panel, -1, '0.003')
         box5.Add(self.scalerate, 1, wx.EXPAND|wx.ALIGN_LEFT|wx.ALL,5)
         box.Add(box5)

         box6 = wx.BoxSizer(wx.HORIZONTAL)
         l5 = wx.StaticText(panel, -1, "Nugget learning rate")
         box6.Add(l5, 1, wx.EXPAND|wx.ALIGN_LEFT|wx.ALL,5)
         self.nuggetrate = wx.TextCtrl(panel, -1, '0.003')
         box6.Add(self.nuggetrate, 1, wx.EXPAND|wx.ALIGN_LEFT|wx.ALL,5)
         box.Add(box6)

         box7 = wx.BoxSizer(wx.HORIZONTAL)
         l6 = wx.StaticText(panel, -1, "Maximum iterations")
         box7.Add(l6, 1, wx.EXPAND|wx.ALIGN_LEFT|wx.ALL,5)
         self.maxsteps = wx.TextCtrl(panel, -1, '1000')
         box7.Add(self.maxsteps, 1, wx.EXPAND|wx.ALIGN_LEFT|wx.ALL,5)
         box.Add(box7)

         box8 = wx.BoxSizer(wx.HORIZONTAL)
         self.submit = wx.Button(panel, wx.ID_OK, "Submit")
         box8.Add(self.submit, 0, wx.ALIGN_CENTER)
         box.Add(box8)


         panel.SetSizer(box)

    def AdditionalData(self,args):
        args['covariancefunc'] = self.Cov_func[self.combo_cov_func.GetSelection()]
        args['principalcomp'] = int(self.pca_comp.GetValue())
        args['initialscale'] = np.fromstring(self.init_scale.GetValue(), dtype=np.float, sep=',')
        args['initialnugget'] = float(self.init_nugget.GetValue())
        args['scalerate'] = float(self.scalerate.GetValue())
        args['nuggetrate'] = float(self.nuggetrate.GetValue())
        args['maxsteps'] = int(self.maxsteps.GetValue())

