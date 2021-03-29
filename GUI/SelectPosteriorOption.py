import sys

import matplotlib.pyplot as plt
import wx


# combobox with auto complete
class PromptingComboBox(wx.ComboBox):
    def __init__(self, parent, value, choices=[], style=0, **par):
        wx.ComboBox.__init__(
            self,
            parent,
            wx.ID_ANY,
            value,
            style=style | wx.CB_DROPDOWN,
            choices=choices,
            **par
        )
        self.choices = choices
        self.default_value = value
        self.Bind(wx.EVT_TEXT, self.EvtText)
        self.Bind(wx.EVT_CHAR, self.EvtChar)
        self.Bind(wx.EVT_COMBOBOX, self.EvtCombobox)
        self.ignoreEvtText = False

    def GetValue(self):
        # override parentes GetValue
        # if value is not found in choice, return default instead of empty
        val = super().GetValue()
        if val in self.choices:
            return val
        else:
            return self.default_value

    def EvtCombobox(self, event):
        self.ignoreEvtText = True
        event.Skip()

    def EvtChar(self, event):
        if event.GetKeyCode() == 8:
            self.ignoreEvtText = True
        event.Skip()

    def EvtText(self, event):
        if self.ignoreEvtText:
            self.ignoreEvtText = False
            event.Skip()
            return
        insert_pt = self.GetInsertionPoint()
        currentText = event.GetString()
        # insert_pt may be zero if the entry is selected from the list
        # make the entire current text appear if insert_pt == 0
        if insert_pt > 0:
            currentText = currentText[: insert_pt + 1]
        found = False
        for n, choice in enumerate(self.choices):
            if choice.startswith(currentText):
                self.ignoreEvtText = True
                self.SetValue(choice)
                self.SetInsertionPoint(insert_pt)
                self.SetTextSelection(insert_pt, len(choice))
                found = True
                break
        if not found:
            # audio warning if things are not found
            wx.Bell()
            self.ignoreEvtText = True
            self.SetValue(currentText[:-1])
            self.SetInsertionPoint(insert_pt)
        event.Skip()


class SelectPosteriorOption(wx.Dialog):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        content = {
            "nbins": ("Number of bins", 100),
            "sigma": ("Smooth result", 0),
            "nlevels": ("Number of countours", 10),
        }
        self.values = {}
        sizer = wx.GridBagSizer(
            len(content) + 4, 5
        )  # add two for color maps combobox and plot button

        for i, (key, (title, default)) in enumerate(content.items()):
            title_text = wx.StaticText(self, -1, title)
            value_ctrl = wx.TextCtrl(self, value=str(default))
            sizer.Add(
                title_text,
                pos=(i, 0),
                flag=wx.EXPAND | wx.ALIGN_LEFT | wx.ALL,
                border=5,
            )
            sizer.Add(
                value_ctrl,
                pos=(i, 1),
                span=(1, 4),
                flag=wx.EXPAND | wx.ALIGN_LEFT | wx.ALL,
            )
            self.values[key] = value_ctrl

        # color map selection
        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        title_text = wx.StaticText(self, -1, "Color maps")
        value_combo = PromptingComboBox(
            self, value="Blues", choices=plt.colormaps())
        sizer.Add(
            title_text,
            pos=(
                len(content),
                0),
            flag=wx.ALIGN_LEFT | wx.ALL,
            border=5)
        sizer.Add(
            value_combo, pos=(
                len(content), 1), span=(
                1, 4), flag=wx.ALIGN_LEFT)
        self.values["cmap"] = value_combo

        self.show_confidence = wx.CheckBox(
            self, label='Show confidence interval')
        sizer.Add(
            self.show_confidence,
            pos=(
                len(content) + 1,
                0),
            span=(
                1,
                5),
            flag=wx.ALIGN_LEFT)

        self.show_ref_pt = wx.CheckBox(
            self, label='Overlay reference points in "Ask Emulator"')
        sizer.Add(
            self.show_ref_pt,
            pos=(
                len(content) + 2,
                0),
            span=(
                1,
                5),
            flag=wx.ALIGN_LEFT)

        self.fix_range = wx.CheckBox(
            self, label='Fix parameters ranges')
        sizer.Add(
            self.fix_range,
            pos=(
                len(content) + 3,
                0),
            span=(
                1,
                5),
            flag=wx.ALIGN_LEFT)


        self.button = wx.Button(self, wx.ID_OK, "Plot")
        sizer.Add(
            self.button,
            pos=(
                len(content) + 4,
                0),
            span=(
                1,
                5),
            flag=wx.ALIGN_LEFT)

        self.SetSizer(sizer)
        self.Fit()

        self.Bind(wx.EVT_CLOSE, self.OnClose)

    def OnClose(self, evt):
        self.Destroy()
        return wx.ID_CANCEL

    def GetValue(self):
        try:
            result = {
                "bins": int(self.values["nbins"].GetValue()),
                "sigma": float(self.values["sigma"].GetValue()),
                "nlevels": int(self.values["nlevels"].GetValue()),
                "cmap": self.values["cmap"].GetValue(),
                "overlay_pt": self.show_ref_pt.GetValue(),
                "show_confidence": self.show_confidence.GetValue(),
                "auto_range": not self.fix_range.GetValue()
            }
        except Exception as e:
            print("Cannot cast result to desired type")
            print(e, flush=True)
        return result
