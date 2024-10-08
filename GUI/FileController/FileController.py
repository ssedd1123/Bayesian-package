import os
import filecmp
from pubsub import pub
from wx.lib.mixins.listctrl import ListCtrlAutoWidthMixin
import wx
import pandas as pd
import copy

class NodeData: 
    def __init__(self, text_is_data=True, data=None, is_file=False):
        self.text_is_data = text_is_data
        self.data = data
        self.is_file = is_file

class TreePanel(wx.Panel):

    def __init__(self, parent):
        wx.Panel.__init__(self, parent)
        
        self.tree = wx.TreeCtrl(self, wx.ID_ANY, style=wx.TR_MULTIPLE | wx.TR_HAS_BUTTONS)    
        
        self.root = self.tree.AddRoot('Available files', data=NodeData())
        #self.tree.SetItemData(self.root, ('key', 'value'))
        #os = self.tree.AppendItem(self.root, 'Operating Systems')
        self.tree.Expand(self.root)
        self.tree.Bind(wx.EVT_TREE_ITEM_ACTIVATED, self.select_trace)
        self.tree.Bind(wx.EVT_TREE_SEL_CHANGED, self.select_emulator)

        self.add_btn = wx.Button(self, -1, 'Add file', size=(50, 20))
        self.add_btn.Bind(wx.EVT_BUTTON, self.on_add)
        self.remove_btn = wx.Button(self, -1, 'Remove', size=(50, 20))
        self.remove_btn.Bind(wx.EVT_BUTTON, lambda evt: self.remove_file(files=None, item=self.tree.GetFocusedItem()))
        self.add_dir_btn = wx.Button(self, -1, 'Add Dir', size=(50, 20))
        self.add_dir_btn.Bind(wx.EVT_BUTTON, self.on_add_dir)

        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        btn_sizer.Add(self.add_btn, 0.4)
        btn_sizer.Add(self.remove_btn, 0.4)
        btn_sizer.Add(self.add_dir_btn, 0.4)

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.tree, 1, wx.EXPAND)
        sizer.Add(btn_sizer, 0)
        self.SetSizerAndFit(sizer)

        self.__emulator_item = self.root
        self.__trace_item = self.root
        # image for metadata
        il = wx.ImageList(16,16)
        self.fileidx = il.Add(wx.ArtProvider.GetBitmap(wx.ART_NORMAL_FILE, wx.ART_OTHER, (16,16)))
        self.folderCloseIdx = il.Add(wx.ArtProvider.GetBitmap(wx.ART_FOLDER, wx.ART_OTHER, (16,16)))
        self.folderOpenIdx = il.Add(wx.ArtProvider.GetBitmap(wx.ART_NEW_DIR, wx.ART_OTHER, (16,16)))
        self.tree.AssignImageList(il)

        self.tree.SetIndent(5)

    def on_load_chain(self, evt):
        pub.sendMessage('LoadChain')

    def select(self, filename):
        item = self.find_file(filename)
        evt = wx.TreeEvent(wx.wxEVT_TREE_ITEM_ACTIVATED, self.tree, item)
        self.select_trace(evt)

    def collapse(self, dirname):
        item = self.find_file(dirname)
        if not self._IsNodeAFile(item):
            self.tree.Collapse(item)

    def collapseAll(self, filename):
        # collapse all directory AND sub directory beneath them
        node = self.find_file(filename)
        def _collapseAll(self, node):
            if self.tree.GetChildrenCount(node) > 0:
                item, cookie = self.tree.GetFirstChild(node)
                while item.IsOk():
                    _collapseAll(self, item)
                    self.tree.Collapse(item)
                    item, cookie = self.tree.GetNextChild(node, cookie)
        _collapseAll(self, node)


    def select_trace(self, evt):
        item = evt.GetItem()
        while not self.tree.GetItemData(item).text_is_data:
            item = self.tree.GetItemParent(item)
        if self._IsNodeAFile(item):
            # change the icon of the previously opened file
            if self._OnCompareItems(self.__trace_item, self.root) != 0:
                self.tree.SetItemImage(self.__trace_item, self.folderCloseIdx, wx.TreeItemIcon_Normal)
            self.__trace_item = item
            self.tree.SetItemImage(self.__trace_item, self.folderOpenIdx, wx.TreeItemIcon_Normal)
            pub.sendMessage('traceFileChanged')
            self.select_emulator(evt)


    def select_emulator(self, evt):
        item = evt.GetItem()
        if item.IsOk() and self.__trace_item.IsOk() and self.tree.GetItemParent(item).IsOk(): 
            # last condition for the above if is set-up for windows user
            # when tree.Delete(node) is called, it sometimes returns an empty item but passes IsOk
            # they are not okay and the only way to check is to check if they have a valid parent
            # the only real valid node without valid parent is the root, but root can't store any file
            # so no matter what if the parent is invalid the operation is invalid
            if not self.tree.GetItemData(item).text_is_data:
                # append metadata only if it comes from trace
                if self._OnCompareItems(self.tree.GetItemParent(item), self.__trace_item) == 0:
                    self.__emulator_item = item
                    pub.sendMessage('emulatorFileChanged')
            # if non-metadata is selected, accept only if it equal trace
            elif self._OnCompareItems(item, self.__trace_item) == 0:
                self.__emulator_item = item
                pub.sendMessage('emulatorFileChanged')

    def get_list_filenames_with_trace_first(self):
        trace = self.trace_filename
        list_filenames = self.list_filenames
        try:
            list_filenames.remove(trace)
        except ValueError:
            pass
        return [trace] + list_filenames

    def _OnCompareItems(self, item1, item2):
        # apparently self.tree.OnCompareItems only compare the item text
        # if two leafs with the same name from different branches are compared, it will also return 0
        # this custom compare function compare all parents to make sure they are really equal
        # return 0 if identical, 1 if not
        if item1.IsOk() and item2.IsOk():
            if self.tree.GetItemText(item1) != self.tree.GetItemText(item2):
                return 1
            item1 = self.tree.GetItemParent(item1)
            item2 = self.tree.GetItemParent(item2)
            return self._OnCompareItems(item1, item2)
        elif not item1.IsOk() and not item2.IsOk():
            return 0
        else:
            return 1

    @property
    def emulator_filename(self):
        if not self.__emulator_item.IsOk() or self._OnCompareItems(self.__emulator_item, self.root) == 0:
            return None
        if not self.tree.GetItemText(self.__emulator_item): # needed it for windows. Sometimes when nodes are removed, it returns a valid item with empty content and no children or parents
            return None
        else:
            return self._GetFilename(self.__emulator_item)

    @emulator_filename.setter
    def emulator_filename(self, emulator_filename):
        if emulator_filename is not None:
            item = self.find_file(emulator_filename)
            assert item is not None, 'Emulator filename is not found in the given list of files'
        else:
            item = self.root
        self.__emulator_item = item
        pub.sendMessage('emulatorFileChanged')

    @property
    def trace_filename(self):
        if not self.__trace_item.IsOk() or self._OnCompareItems(self.__trace_item, self.root) == 0:
            return None
        elif not self.tree.GetItemText(self.__trace_item):
            return None
        else:
            return self._GetFilename(self.__trace_item)

    @trace_filename.setter
    def trace_filename(self, trace_filename):
        if trace_filename is not None:
            item = self.find_file(trace_filename)
            assert item is not None, 'Trace filename ' + trace_filename + ' is not found in the given list of files'
        else:
            item = self.root
        self.__trace_item = item
        pub.sendMessage('traceFileChanged')

    @property
    def list_filenames(self):
        files = []
        items = self.tree.GetSelections()
        visited = set()
        for item in items:
            if item not in visited:
                files = files + [self._GetFilename(i) for i in self._GetAllChildren(item, visited=visited)]
        files = list(set(files))
        return files

    @property
    def list_dirnames(self):
        items = self.tree.GetSelections()
        dirs = []
        for item in items:
            if not self._IsNodeAFile(item):
                dirs.append(self._GetFilename(item))
        return dirs

    def get_files_in_dir(self, dirname):
        item = self.find_file(dirname)
        visited = set()
        return [self._GetFilename(i) for i in self._GetAllChildren(item, visited=visited)]


    def _GetAllChildren(self, root, no_metadata=True, visited=set()):
        if not root.IsOk():
            return []
        if no_metadata and not self.tree.GetItemData(root).text_is_data:
            return []
        visited.add(root)
        if self.tree.GetChildrenCount(root) == 0:
            return [root]
        if no_metadata and not self.tree.GetItemData(self.tree.GetFirstChild(root)[0]).text_is_data:
            return [root]
        nodes = []
        node, cookie = self.tree.GetFirstChild(root)
        while node.IsOk():
            nodes = nodes + self._GetAllChildren(node, no_metadata, visited)
            node, cookie = self.tree.GetNextChild(root, cookie)
        return nodes 

    def _IsNodeAFile(self, node):
        if node.IsOk():
            data = self.tree.GetItemData(node)
            if data.is_file:
                return True
        return False

    def on_add(self, evt):
        pub.sendMessage("FileAdd")

    def on_add_dir(self, evt):
        pub.sendMessage("DirAdd")

    def add_meta_data(self, filename, metadata):
        node = self.find_file(filename)
        assert node is not None, 'Cannot find file ' + filename + ' to attach metadata onto.'
        def shorten_path(path):
            orig_path = path
            path_component = []
            while path:
                path, folder = os.path.split(path)
                if folder != "":
                    path_component.append(folder)
                elif path != "":
                    path_component.append(path)
                    break
            path_component.reverse()

            if len(path_component) <= 2:
                return orig_path
            else:
                return os.path.join('...', path_component[-2], path_component[-1])
    
        for data in metadata:
            item = self.tree.AppendItem(node, shorten_path(data), data=NodeData(False, data))
            self.tree.SetItemImage(item, self.fileidx, wx.TreeItemIcon_Normal)

    def remove_meta_data(self, filename):
        node = self.find_file(filename)
        assert node is not None, 'Cannot find file ' + filename + ' to remove metadata from.'
        if self.tree.GetChildrenCount(node) > 0:
            metadata = []
            item, cookie = self.tree.GetFirstChild(node)
            while item.IsOk():
                assert not self.tree.GetItemData(item).text_is_data, 'Child of branch ' + filename + ' is not meta data.'
                metadata.append(item)
                item, cookie = self.tree.GetNextChild(node, cookie)
            for item in metadata:
                self.remove_file(None, item=item, force=True)

    def _GetFilename(self, item, discard_meta=False):
        if not discard_meta and not self.tree.GetItemData(item).text_is_data:
            return self.tree.GetItemData(item).data

        path = None 
        while item.IsOk() and self._OnCompareItems(item, self.root) != 0:
            if path is None:
                if self.tree.GetItemData(item).text_is_data:
                    path = self.tree.GetItemText(item)
                elif not discard_meta:
                    path = self.tree.GetItemData(item)
            else:
                path = os.path.join(self.tree.GetItemText(item), path)
            item = self.tree.GetItemParent(item)
        return path

    def _FindNode(self, path, root):
        item, cookie = self.tree.GetFirstChild(root)

        node = None
        while item.IsOk() and self.tree.GetItemData(item).text_is_data:
            if self.tree.GetItemText(item) == path:
                node = item
                break
            item, cookie = self.tree.GetNextChild(root, cookie)
        return node

    def _RemoveNode(self, dir_, filename, item=None, force=False):
        if item is not None:
            if not force and not self.tree.GetItemData(item).text_is_data:
                raise RuntimeError('Node to be deleted ' + self._GetFilename(item) + ' is metadata and cannot be removed')
            self.tree.Delete(item)
        else: 
            nodeList = []
            dirList = dir_ + [filename]
            node = self.root

            while len(dirList) > 0:
                node = self._FindNode(dirList[0], node)
                dirList.pop(0)
                if node is None:
                    raise RuntimeError('Cannot find tree node/element: ' + path)
                if len(dirList) == 0:
                    if not self.tree.GetItemData(node).text_is_data:
                        raise RuntimeError('Node to be deleted ' + dirList[0] + ' is metadata and cannot be removed')
                    self.tree.Delete(node)
                else:
                    nodeList.append(node)

            prev_node = None
            for node in reversed(nodeList):
                if self.tree.GetChildrenCount(node, False) > 1:
                    self.tree.Delete(prev_node)
                    return
                prev_node = node
            self.tree.Delete(prev_node)

    def _AddNode(self, dir_, node, nodeIsCreated=False):
        if len(dir_) == 0:
            if nodeIsCreated:
                self.tree.SetItemImage(node, self.folderCloseIdx, wx.TreeItemIcon_Normal)
                self.tree.SetItemData(node, data=NodeData(True, None, True))
                return node
            else:
                return None
        curr_dir = dir_[0]
        child = self._FindNode(curr_dir, node)
        if child is None:
            nodeIsCreated = True
            child = self.tree.AppendItem(node, curr_dir, data=NodeData())
        return self._AddNode(dir_[1:], child, nodeIsCreated)

    def _SplitDirPath(self, path):
        folders = []
        filename = None
        while True:
            path, folder = os.path.split(path)
            if folder != '':
                if filename is None:
                    filename = folder
                else:
                    folders.append(folder)
            else:
                if len(folders) > 0:
                    folders[-1] = path + folders[-1]
                else:
                    filename = path + filename 
                break

        folders.reverse()
        return folders, filename

    def add_file(self, files):
        if isinstance(files, str):
            files = [files]
        status = []
        for file_ in files:
            dir_, filename = self._SplitDirPath(file_)
            node = self._AddNode(dir_ + [filename], self.root)
            status.append(False if node is None else True)
            while node is not None and node.IsOk():
                self.tree.Expand(node)
                node = self.tree.GetItemParent(node)
        if len(status) == 1:
            return status[0]
        else:
            return status


    def remove_file(self, files, item=None, force=False):
        trace = self.trace_filename
        emulator = self.emulator_filename
        if item is not None:
            if self._OnCompareItems(item, self.root) != 0: # can't remove root
                self._RemoveNode('', '', item=item, force=force)
        else:
            if isinstance(files, str):
                files = [files]
            for file_ in files:
                dir_, filename = self._SplitDirPath(file_)
                self._RemoveNode(dir_, filename, force=force)
            #pub.sendMessage('emulatorFileChanged')
            #pub.sendMessage('traceFileChanged')
        self.__trace_item = self.root
        self.__emulator_item = self.root
        try:
            self.trace_filename = trace
            self.emulator_filename = emulator
        except AssertionError:
            self.trace_filename = None
            self.emulator_filename = None


    def find_file(self, file_):
        dirList, filename = self._SplitDirPath(file_)
        dirList = dirList + [filename]
        node = self.root

        while len(dirList) > 0:
            node = self._FindNode(dirList[0], node)
            dirList.pop(0)
            if node is None:
                return None
            if len(dirList) == 0:
                return node
        return None

    def get_all_files(self):
        ans = []
        #Depth first search
        def dfs(self, node, dir_, ans):
            if self.tree.GetChildrenCount(node) == 0:
                ans.append(dir_)
                return
            item, cookie = self.tree.GetFirstChild(node)
            while item.IsOk() and self.tree.GetItemData(item).text_is_data:
                dfs(self, item, os.path.join(dir_, self.tree.GetItemText(item)), ans)
                item, cookie = self.tree.GetNextChild(node, cookie)
        dfs(self, self.root, '', ans)
        return ans

    def get_trace_metadata(self):
        metadata = []
        node = self.__trace_item
        if self._OnCompareItems(node, self.root) != 0 and self.tree.GetChildrenCount(node) > 0:
            item, cookie = self.tree.GetFirstChild(node)
            while item.IsOk() and not self.tree.GetItemData(item).text_is_data:
                metadata.append(self.tree.GetItemData(item).data)
                item, cookie = self.tree.GetNextChild(node, cookie)
        return metadata

class FileDisplay(wx.Panel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trace_file_display = wx.StaticText(self, -1)
        self.emulator_file_display = wx.StaticText(self, -1)

        font = wx.Font(pointSize=10, family=wx.DEFAULT,
                       style=wx.NORMAL, weight=wx.NORMAL)
        self.trace_file_display.SetFont(font)
        self.emulator_file_display.SetFont(font)

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.trace_file_display, 1, wx.EXPAND)
        sizer.Add(self.emulator_file_display, 1, wx.EXPAND)
        self.SetSizer(sizer)

    def display_file(self, trace_file=None, emulator_file=None):
        self.trace_file_display.SetLabel('Trace file: %s' % ( '' if trace_file is None else trace_file))
        self.emulator_file_display.SetLabel('Emulator file: %s' %('' if emulator_file is None else emulator_file))
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
        self.file_view = TreePanel(**file_viewer_kwargs)
        self.model = self.file_view
        self.display_view = FileDisplay(**display_kwargs)
        #self.display_view.display_file('test', 'test')

        pub.subscribe(self.add_file, 'FileAdd')
        pub.subscribe(self.add_dir, 'DirAdd')
        pub.subscribe(self.add_meta_file, 'LoadChain')
        pub.subscribe(self._SyncDisplayData, 'traceFileChanged')
        pub.subscribe(self._SyncDisplayData, 'emulatorFileChanged')

        self._SyncDisplayData()

    def _SyncDisplayData(self):
        self.display_view.display_file(
            self.model.trace_filename,
            self.model.emulator_filename)

    def add_meta_file(self):
        metadata = self.file_view.get_trace_metadata()
        self.add_file(metadata)

    def add_dir(self):
        self.add_file(None, False, True)

    def add_file(self, filelist=None, exist_ok=False, add_dir=False):
        directory = None
        if filelist is None:
            # if there are emulator file, get the directory to it as default directory
            default_dir = ''
            if self.model.trace_filename is not None:
                default_dir = os.path.dirname(self.model.trace_filename)
            if add_dir:
                with wx.DirDialog(self.file_view, "Open Dir", defaultPath=default_dir, style=wx.DD_DEFAULT_STYLE | wx.DD_DIR_MUST_EXIST) as dirDialog:
                    if dirDialog.ShowModal() == wx.ID_CANCEL:
                        return
                    directory = dirDialog.GetPath()

                    # get all files in all subdirectory
                    filelist = []
                    for path, subdirs, files in os.walk(directory):
                        for name in files:
                            filelist.append(os.path.join(path, name))
            else:
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

        last_filename = None
	# warn people of the repeated files
        repeated_filelist = []
        for filename in filelist:
            # turn everything into absolute path
            filename = os.path.abspath(filename)
            chained_filenames = None
            try:
                with pd.HDFStore(filename) as store:
                    chained_filenames = store.get_storer('trace').attrs.chained_files
            except Exception:
                pass
            if self.model.add_file(filename):
                last_filename = filename
                if chained_filenames is not None:
                    self.model.add_meta_data(filename, chained_filenames)
            else:
                repeated_filelist.append(filename)


        if len(repeated_filelist) > 0 and not exist_ok:
            wx.MessageBox(
                '\n'.join(
                    ['The following files are not added since they are already included:'] +
                    list(repeated_filelist)),
                'Warning',
                wx.OK | wx.ICON_WARNING)

        if last_filename is not None:
            self.model.select(last_filename)

        # if add directory, collapse the added directory
        if directory is not None:
            if len(filelist) > 0:
                self.file_view.collapseAll(directory)
            else:
                raise ValueError('No file is added! Check if the directory is empty!')



    def update_metadata(self, filename):
        chained_filenames = None
        try:
            with pd.HDFStore(filename) as store:
                chained_filenames = store.get_storer('trace').attrs.chained_files
        except Exception:
            pass
        self.model.remove_meta_data(filename)
        if chained_filenames is not None:
            self.model.add_meta_data(filename, chained_filenames)

            

if __name__ == "__main__":
    app = wx.App()
    frame = wx.Frame(None, size=(200, 500))
    controller = FileController(file_viewer_kwargs={'parent': frame}, display_kwargs={'parent': frame, 'size': (-1, 100)})
    # trace_filename='file1', all_filenames=['file1', 'file2', 'file3'])

    panel = controller.file_view
    panel.add_file('/projects/hira/FileController.py')
    panel.add_file('/projects/hira/tsangc/diffusion_tsang7.doc')
    panel.add_file('/projects/hira/tsangc/diffusion_tsang3.doc')
    panel.add_file('/mnt/hira/FileController.py')
    panel.add_file('/macros/data/pbuu_sn108.root')
    panel.add_file('/test.txt')
    panel.add_meta_data('/test.txt', ['file1', 'file2'])

    sizer = wx.BoxSizer(wx.VERTICAL)
    sizer.Add(controller.file_view, 1, wx.EXPAND)
    sizer.Add(controller.display_view, 0., wx.EXPAND)
    sizer.Layout()
    frame.SetSizer(sizer)
    frame.Show()

    #panel.remove_file('/projects/hira/tsangc/diffusion_tsang7.doc')
    #panel.remove_file('/mnt/hira/FileController.py')

    app.MainLoop()
