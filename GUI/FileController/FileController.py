import os
import filecmp
from pubsub import pub
from wx.lib.mixins.listctrl import ListCtrlAutoWidthMixin
import wx
import pandas as pd


class FileModel:
    def __init__(self, trace_filename, emulator_filename, list_filenames=None):
        if list_filenames is None:
            # remove all None elements
            self.list_filenames = list(
                set([trace_filename, emulator_filename]) - set([None]))
        else:
            self.list_filenames = list_filenames
        self.trace_filename = trace_filename
        self.emulator_filename = emulator_filename

    @property
    def emulator_filename(self):
        return self.__emulator_filename

    @emulator_filename.setter
    def emulator_filename(self, emulator_filename):
        if emulator_filename is not None:
            assert emulator_filename in self.list_filenames, 'Emulator filename is not found in the given list of files'
        self.__emulator_filename = emulator_filename
        pub.sendMessage('emulatorFileChanged')

    @property
    def trace_filename(self):
        return self.__trace_filename

    @trace_filename.setter
    def trace_filename(self, trace_filename):
        if trace_filename is not None:
            assert trace_filename in self.list_filenames, 'Trace filename is not found in the given list of files'
        self.__trace_filename = trace_filename
        pub.sendMessage('traceFileChanged')

    @property
    def list_filenames(self):
        return self.__list_filenames

    @list_filenames.setter
    def list_filenames(self, list_filenames):
        self.__list_filenames = list_filenames
        pub.sendMessage('listFileChanged')

    def get_list_filenames_with_trace_first(self):
        list_return = self.list_filenames.copy()
        assert self.trace_filename is not None, 'Trace file is not selected'
        list_return.insert(
            0, list_return.pop(
                list_return.index(
                    self.trace_filename)))
        return list_return

    def remove_file(self, filename):
        if filename == self.trace_filename:
            self.trace_filename = None
        if filename == self.emulator_filename:
            self.emulator_filename = None
        try:
            # highlight the element in the same place
            index = self.list_filenames.index(filename)
            self.list_filenames.remove(filename)
            pub.sendMessage('listFileChanged')
            if index < len(self.list_filenames):
                return index
            else:
                return -1
        except Exception as e:
            print(e)

    def add_file(self, filename, exist_ok=False):
        if not exist_ok:
            if filename in self.list_filenames:
                raise FileNotfoundError(
                    'File to be added already exist. Suppress this exception by setting exist_ok=True in add_file')
        if filename not in self.list_filenames:
            self.list_filenames = self.list_filenames + [filename]

# custom listCtrl that will resize with width


class ListCtrlAutoWidth(wx.ListCtrl, ListCtrlAutoWidthMixin):
    def __init__(self, parent, ID=-1, pos=wx.DefaultPosition,
                 size=wx.DefaultSize, style=0):
        wx.ListCtrl.__init__(self, parent, ID, pos, size, style)
        ListCtrlAutoWidthMixin.__init__(self)
        self.setResizeColumn(0)


class FileViewer(wx.Panel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.lst = ListCtrlAutoWidth(
            self, size=(-1, 100), style=wx.LC_REPORT | wx.LC_SINGLE_SEL)
        self.lst.InsertColumn(0, 'File name', format=wx.LIST_FORMAT_LEFT)
        self.lst.SetColumnWidth(0, wx.LIST_AUTOSIZE)
        self.lst.Bind(wx.EVT_LIST_ITEM_SELECTED, self.on_update)
        self.lst.Bind(wx.EVT_LIST_ITEM_ACTIVATED, self.on_select)

        self.add_btn = wx.Button(self, -1, 'Add file', size=(50, 20))
        self.add_btn.Bind(wx.EVT_BUTTON, self.on_add)
        self.remove_btn = wx.Button(self, -1, 'Remove', size=(50, 20))
        self.remove_btn.Bind(wx.EVT_BUTTON, self.on_remove)
        self.add_chain_btn = wx.Button(self, -1, 'Load chained', size=(50, 20))
        self.add_chain_btn.Bind(wx.EVT_BUTTON, self.on_load_chain)
        self.add_chain_btn.Disable()

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.lst, 1., wx.EXPAND)

        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        btn_sizer.Add(self.add_btn, 0.4)
        btn_sizer.Add(self.remove_btn, 0.4)
        btn_sizer.Add(self.add_chain_btn, 0.4)
        sizer.Add(btn_sizer, 0)

        self.selected_item = None
        self.SetSizerAndFit(sizer)

    def on_load_chain(self, evt):
        pub.sendMessage('LoadChain')

    def on_update(self, evt):
        pub.sendMessage(
            "EmulatorSelected",
            filename=self.lst.GetItem(
                self.lst.GetFirstSelected()).GetText())

    def on_select(self, evt):
        pub.sendMessage(
            "TraceSelected",
            filename=self.lst.GetItem(
                self.lst.GetFirstSelected()).GetText())

    def on_remove(self, evt):
        if self.lst.GetFirstSelected() is wx.NOT_FOUND:
            filename = None
        else:
            filename = self.lst.GetItem(self.lst.GetFirstSelected()).GetText()
        pub.sendMessage("FileRemove", filename=filename)

    def on_add(self, evt):
        pub.sendMessage("FileAdd")

    def SetList(self, list_):
        self.lst.DeleteAllItems()
        index = 0
        for val in list_:
            if val is not None:
                self.lst.InsertItem(index, val)
                index += 1

    def Select(self, element):
        i = self.lst.FindItem(-1, element)
        assert i != wx.NOT_FOUND, 'Element ' + element + ' is not found in listbox'
        self.lst.Select(i)
        # function do not emit EVT_LISTBOX_SELECT. Have to do it manually
        self.on_update(None)

    def highlight_only(self, element=None):
        # re-highlight current element if it is None
        # usefule when model list is updated
        if element is None:
            if self.selected_item is None:
                return  # do nothing if nothing needs to be highlighted
            element = self.selected_item
        i = self.lst.FindItem(-1, element)
        # un-highlight previously selected item
        if self.selected_item is not None:
            item = self.lst.FindItem(-1, self.selected_item)
            if item != wx.NOT_FOUND:
                self.lst.SetItemBackgroundColour(
                    item, wx.Colour(255, 255, 255, 255))
        self.selected_item = element
        if i != wx.NOT_FOUND:
            self.lst.SetItemBackgroundColour(i, wx.Colour(255, 0, 0, 255))


class FileDisplay(wx.Panel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # , style=wx.ALIGN_LEFT)#, size=(-1,30))
        self.trace_file_display = wx.StaticText(self, -1)
        # , style=wx.ALIGN_LEFT)#, size=(-1,30))
        self.emulator_file_display = wx.StaticText(self, -1)

        font = wx.Font(pointSize=10, family=wx.DEFAULT,
                       style=wx.NORMAL, weight=wx.NORMAL)
        self.trace_file_display.SetFont(font)
        self.emulator_file_display.SetFont(font)

        sizer = wx.BoxSizer(wx.VERTICAL)
        #sizer.Add(wx.StaticText(self, -1, label='Trace file:', style=wx.ALIGN_LEFT, size=(-1,15)), 0., wx.EXPAND)
        sizer.Add(self.trace_file_display, 1, wx.EXPAND)
        #sizer.Add(wx.StaticText(self, -1, label='Emulator file:', style=wx.ALIGN_LEFT, size=(-1,15)), 0., wx.EXPAND)
        sizer.Add(self.emulator_file_display, 1, wx.EXPAND)

        self.SetSizer(sizer)

    def display_file(self, trace_file=None, emulator_file=None):
        if trace_file is None:
            trace_file = ''
        self.trace_file_display.SetLabel('Trace file: ' + trace_file)
        # 1self.trace_file_display.Wrap(self.Size[0])
        # self.trace_file_display.SetAutoLayout(True)
        if emulator_file is None:
            emulator_file = ''
        self.emulator_file_display.SetLabel('Emulator file: ' + emulator_file)
        # self.emulator_file_display.Wrap(self.Size[0])
        if self.GetSizer() is not None:
            self.GetSizer().Layout()


class FileController:

    def __init__(
            self,
            trace_filename=None,
            emulator_filename=None,
            all_filenames=None,
            file_viewer_kwargs={},
            display_kwargs={}):
        self.file_view = FileViewer(**file_viewer_kwargs)
        self.display_view = FileDisplay(**display_kwargs)
        # trace_filename, emulator_filename, all_filenames)
        self.model = FileModel(None, None)

        # sync method must comes first.
        pub.subscribe(self.update_emulator, 'EmulatorSelected')
        pub.subscribe(self.update_trace, 'TraceSelected')
        pub.subscribe(self.remove_file_highlight_inplace, 'FileRemove')
        pub.subscribe(self.add_file, 'FileAdd')

        pub.subscribe(self._LoadChain, 'LoadChain')
        pub.subscribe(self._SyncListContent, 'listFileChanged')
        pub.subscribe(self._SyncDisplayData, 'emulatorFileChanged')
        pub.subscribe(self._CheckIfChained, 'emulatorFileChanged')
        pub.subscribe(self._SyncDisplayData, 'traceFileChanged')

        # convert none in all_filenames to empty list
        # because add_file with None prompts FileDialog
        # supress that in constructor
        if all_filenames is None:
            all_filenames = []
        self.add_file(all_filenames)
        # self.update_trace(trace_filename)
        # self.update_emulator(emulator_filename)

        self._SyncDisplayData()
        self._SyncListContent()

    def update_emulator(self, filename):
        self.model.emulator_filename = filename
        # if trace_file is empty, it will also be assigned as emulator
        if self.model.trace_filename is None:
            self.update_trace(filename)

    def update_trace(self, filename):
        self.model.trace_filename = filename
        self.file_view.highlight_only(filename)

    def remove_file_highlight_inplace(self, filename):
        if filename is not None:
            idx = self.model.remove_file(filename)
        if idx < len(
                self.model.list_filenames) and idx >= 0 and idx is not None:
            self.model.emulator_file = self.model.list_filenames[idx]
            self.file_view.Select(self.model.emulator_file)

    def add_file(self, filelist=None, exist_ok=False):
        if filelist is None:
            # if there are emulator file, get the directory to it as default directory
            default_dir = ''
            if self.model.emulator_filename is not None:
                default_dir = os.path.dirname(self.model.emulator_filename)
            with wx.FileDialog(self.file_view, "Open file", defaultDir=default_dir, style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST | wx.FD_MULTIPLE) as fileDialog:
                if fileDialog.ShowModal() == wx.ID_CANCEL:
                    return
                filelist = fileDialog.GetPaths()
        else:
            if not isinstance(filelist, list):
                filelist = [filelist]
            # check if the file exist
            for filename in filelist:
                if not os.path.isfile(filename):
                    wx.MessageBox(
                        'File %s does not exist' %
                        filename, 'Error', wx.OK | wx.ICON_ERROR)
                    return

        # check if files that you want to add already exist. If so we raise the
        # warning
        non_repeat_filelist = {}  # a set to avoid duplicated entry
        repeated_filelist = []
        for new_file in filelist:
            for old_file in self.model.list_filenames:
                if filecmp.cmp(new_file, old_file):
                    repeated_filelist.append(new_file)
        non_repeat_filelist = set(filelist) - set(repeated_filelist)
        # raise error if they exist
        if len(repeated_filelist) > 0 and not exist_ok:
            wx.MessageBox(
                '\n'.join(
                    ['The following files are not added since they are already included:'] +
                    repeated_filelist),
                'Warning',
                wx.OK | wx.ICON_WARNING)
        for i, filename in enumerate(non_repeat_filelist):
            self.model.add_file(filename, exist_ok)
            if i == 0:
                self.file_view.Select(filename)

    def _LoadChain(self):
        with pd.HDFStore(self.model.emulator_filename, 'r') as store:
            self.add_file(store.get_storer('trace').attrs.chained_files)

    def _CheckIfChained(self):
        # really needs to open the file to inspecf
        should_enable = False
        try:
            if self.model.emulator_filename is not None:
                with pd.HDFStore(self.model.emulator_filename, 'r') as store:
                    if 'trace' in store:
                        if 'chained_files' in store.get_storer('trace').attrs:
                            should_enable = True
        except Exception as e:
            pass
        finally:
            if should_enable:
                self.file_view.add_chain_btn.Enable()
            else:
                self.file_view.add_chain_btn.Disable()

    def _SyncListContent(self):
        self.file_view.SetList(self.model.list_filenames)
        self.file_view.highlight_only()

    def _SyncDisplayData(self):
        self.display_view.display_file(
            self.model.trace_filename,
            self.model.emulator_filename)
        self.file_view.highlight_only(self.model.trace_filename)


if __name__ == "__main__":
    app = wx.App()
    frame = wx.Frame(None, size=(200, 500))
    # controller = FileController(file_viewer_kwargs={'parent': frame}, display_kwargs={'parent': frame, 'size': (-1, 100)}, \
    # trace_filename='file1', all_filenames=['file1', 'file2', 'file3'])
    controller = FileController(file_viewer_kwargs={'parent': frame}, display_kwargs={
                                'parent': frame, 'size': (-1, 100)}, trace_filename=None, all_filenames=None)

    sizer = wx.BoxSizer(wx.VERTICAL)
    sizer.Add(controller.file_view, 1, wx.EXPAND)
    sizer.Add(controller.display_view, 0, wx.EXPAND)
    sizer.Layout()
    frame.SetSizer(sizer)
    frame.Show()
    app.MainLoop()
