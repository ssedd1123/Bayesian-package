from io import StringIO
import copy
import pandas as pd
import numpy as np
import string
import itertools
from pubsub import pub
import wx
import wx.grid as gridlib

def _column_name_generator():
    for i in itertools.count(1):
        for p in itertools.product(string.ascii_uppercase, repeat=i):
            yield ''.join(p)

def enum(*sequential, **named):
    enums = dict(zip(sequential, range(len(sequential))), **named)
    return type('Enum', (), enums)

HistoryType = enum('Changed', 'DeleteShift', 'Insert')
Direction = enum('Up', 'Down', 'Left', 'Right')

def OppositeDir(direction):
    if direction == Direction.Up:
        return Direction.Down
    elif direction == Direction.Down:
        return Direction.Up
    elif direction == Direction.Left:
        return Direction.Right
    else:
        return Direction.Left

class GridData(gridlib.GridTableBase):

    def __init__(self, num_row, num_col, df=None, parent=None):
        # fill headers with alphabets
        super().__init__()
        self.parent = parent 
        if df is None:
            self.header = [name for i, name in zip(range(num_col), _column_name_generator())]
            self.data = pd.DataFrame(None, index=range(num_row), columns=self.header)
        else:
            self.data = df

        self.history = []
        self.undo_history = []

    def GetParent(self):
        pass

    def GetData(self, first_row_as_header=True, drop_index=True):  
        df = self.data.replace(r'^\s*$', np.nan, regex=True)
        df = df.dropna(axis=0, how='all').dropna(axis=1, how='all')
        if first_row_as_header:
            df.columns=df.iloc[0]
            df = df.drop(df.index[0])
            if drop_index:
                df = df.reset_index(drop=True)
                df.columns.name = None
        return df
 

    def SetData(self, df, include_header=True, include_index=False):
        if include_index:
            df = df.reset_index()
        if include_header:
            df = df.T.reset_index().T.reset_index(drop=True)
        self.ChangeValues(np.arange(df.shape[0]), np.arange(df.shape[1]), df)

    def ResetUndo(self):
        self.history = []
        self.undo_history = []
        pub.sendMessage('Data_CannotUndo', obj=self)
        pub.sendMessage('Data_CannotRedo', obj=self)

    def Undo(self):
        if len(self.history) > 0:
            change_content = self.history[-1]
            func, change_content = change_content[0], change_content[1:]
            func(*change_content, in_undo=True, delete_redo=False)
            self.history.pop()

            pub.sendMessage('Data_CanRedo', obj=self)
            if len(self.history) == 0:
                pub.sendMessage('Data_CannotUndo', obj=self)
            #rows = change_content[0]
            #cols = change_content[1]
            #pub.sendMessage('Data_Changed', obj=self, evt=[rows, cols])

    def Redo(self):
        if len(self.undo_history) > 0:
            change_content = self.undo_history[-1]
            func, change_content = change_content[0], change_content[1:]
            func(*change_content, in_undo=False, delete_redo=False)
            self.undo_history.pop()

            pub.sendMessage('Data_CanUndo', obj=self)
            if len(self.undo_history) == 0:
                pub.sendMessage('Data_CannotRedo', obj=self)
            #rows = change_content[0]
            #cols = change_content[1]
            #pub.sendMessage('Data_Changed', obj=self, evt=[rows, cols])


    def ChangeValues(self, rows, cols, values, delete_redo=True, send_changed=True, in_undo=False):
        rows = np.atleast_1d(np.asarray(rows))
        cols = np.atleast_1d(np.asarray(cols))
        values = np.atleast_2d(np.asarray(values))
        nrows = rows.shape[0] 
        ncols = cols.shape[0]
        if values.shape[0] == 1 and values.shape[1] == 1:
            values = np.full((nrows, ncols), values[0, 0])
        if values.shape[0] == 1:
            if ncols == 1:
                values = values.reshape(-1, 1)
            elif nrows != 1:
                raise RuntimeError('Input 1D values cannot be made to agree with the shape of rows and cols')
        assert nrows == values.shape[0] and ncols == values.shape[1],\
            'Change requested cannot be be done because supplied rows, colums and value sizes are inconsistent.'
        if in_undo:
            self.undo_history.append((self.ChangeValues, rows, cols, self.data.iloc[rows, cols].values))
        else:
            pub.sendMessage('Data_CanUndo', obj=self)
            self.history.append((self.ChangeValues, rows, cols, self.data.iloc[rows, cols].values))
        self.data.iloc[rows, cols] = values

        if delete_redo:
            pub.sendMessage('Data_CannotRedo', obj=self)
            self.undo_history.clear()
        if send_changed:
            pub.sendMessage('Data_Changed', obj=self, evt=[rows, cols])

    def DeleteShift(self, rows, cols, direction=Direction.Up, delete_redo=True, in_undo=False):
        rows = np.atleast_1d(np.asarray(rows))
        cols = np.atleast_1d(np.asarray(cols))
        if in_undo:
            self.undo_history.append((self.Insert, rows, cols, self.data.iloc[rows, cols].values, OppositeDir(direction)))
        else:
            self.history.append((self.Insert, rows, cols, self.data.iloc[rows, cols].values, OppositeDir(direction)))
            pub.sendMessage('Data_CanUndo', obj=self)


        if direction == Direction.Up:
            self.data.iloc[rows[0]:, cols[0]:cols[-1]+1] = self.data.iloc[rows[0]:, cols[0]:cols[-1]+1].shift(-rows.shape[0])
        elif direction == Direction.Down:
            self.data.iloc[:rows[-1]+1, cols[0]:cols[-1]+1] = self.data.iloc[:rows[-1]+1, cols[0]:cols[-1]+1].shift(rows.shape[0])
        elif direction == Direction.Left:
            self.data.iloc[rows[0]:rows[-1]+1, cols[0]:] = self.data.iloc[rows[0]:rows[-1]+1, cols[0]:].T.shift(-cols.shape[0]).T
        else:
            self.data.iloc[rows[0]:rows[-1]+1, :cols[-1]+1] = self.data.iloc[rows[0]:rows[-1]+1, :cols[-1]+1].T.shift(cols.shape[0]).T

        if delete_redo:
            pub.sendMessage('Data_CannotRedo', obj=self)
            self.undo_history.clear()
        pub.sendMessage('Data_Changed', obj=self, evt=[rows, cols])


    def Insert(self, rows, cols, values, direction=Direction.Up, delete_redo=True, in_undo=False):
        rows = np.atleast_1d(np.asarray(rows))
        cols = np.atleast_1d(np.asarray(cols))
        values = np.atleast_2d(np.asarray(values))

        assert rows.shape[0] == values.shape[0] and cols.shape[0] == values.shape[1],\
            'Insert requested cannot be be done because supplied rows, colums and value sizes are inconsistent.'
        if in_undo:
            self.undo_history.append((self.DeleteShift, rows, cols, OppositeDir(direction)))
        else:
            self.history.append((self.DeleteShift, rows, cols, OppositeDir(direction)))
            pub.sendMessage('Data_CanUndo', obj=self)
        
        if direction == Direction.Up:
            self.data.iloc[:rows[-1]+1, cols[0]:cols[-1]+1] = self.data.iloc[:rows[-1]+1, cols[0]:cols[-1]+1].shift(-rows.shape[0])
        elif direction == Direction.Down:
            self.data.iloc[rows[0]:, cols[0]:cols[-1]+1] = self.data.iloc[rows[0]:, cols[0]:cols[-1]+1].shift(rows.shape[0])
        elif direction == Direction.Left:
            self.data.iloc[rows[0]:rows[-1]+1, :cols[-1]+1] = self.data.iloc[rows[0]:rows[-1]+1, :cols[-1]+1].T.shift(-cols.shape[0]).T
        else:
            self.data.iloc[rows[0]:rows[-1]+1, cols[0]:] = self.data.iloc[rows[0]:rows[-1]+1, cols[0]:].T.shift(cols.shape[0]).T
        self.data.iloc[rows[0]:rows[-1]+1, cols[0]:cols[-1]+1] = values

        if delete_redo:
            pub.sendMessage('Data_CannotRedo', obj=self)
            self.undo_history.clear()
        pub.sendMessage('Data_Changed', obj=self, evt=[rows, cols])


    def GetValue(self, row, col):
        value = self.data.iloc[row, col]
        return value if not pd.isna(value) else None

    def SetValue(self, row, col, value):
        self.ChangeValues(row, col, value)

    def GetNumberRows(self):
        return self.data.shape[0]

    def GetNumberCols(self):
        return self.data.shape[1]

    def GetColLabelValue(self, col):
        return str(self.data.columns[col])

    def GetRowLabelValue(self, row):
        return str(self.data.index[row])

    def GetTypeName(self, row, col):
        return gridlib.GRID_VALUE_STRING

    def IsEmptyCell(self, row, col):
        return pd.isna(self.data.iloc[row, col])


