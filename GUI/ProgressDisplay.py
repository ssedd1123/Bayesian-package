import sys, wx

from multiprocessing import Process, Queue, cpu_count, current_process, freeze_support
from Queue import Empty
from StatParalleldevelopment import StatParallel, TraceCreator, MergeTrace

class QueuePipe:
    """
    Redirect all stdout output to queue
    tasknum corresponds to the id of the thread
    """

    def __init__(self, queue, tasknum):
        self.queue = queue
        self.tasknum = tasknum

    def write(self, output):
        if output.rstrip():
            self.queue.put((self.tasknum, output))

    def flush(self):
        pass


class MyFrame(wx.Frame):
    def __init__(self, parent, id, title, args):
        self.args = args         
        self.trace = TraceCreator(self.args)

        wx.Frame.__init__(self, parent, id, title)

        self.panel = wx.Panel(self, wx.ID_ANY)

        #widgets
        #self.start_bt = wx.Button(self.panel, wx.ID_ANY, "Start")
        #self.Bind(wx.EVT_BUTTON, self.OnButton, self.start_bt)

        self.output_tc = []
        for i in range(self.args['cores']):
            self.output_tc.append(wx.TextCtrl(self.panel, wx.ID_ANY, style=wx.TE_READONLY))

        # sizer
        self.sizer = wx.GridBagSizer(1, 1)
        #self.sizer.Add(self.start_bt, (0, 0), flag=wx.ALIGN_CENTER|wx.LEFT|wx.TOP|wx.RIGHT, border=5)

        for i, output_tc in enumerate(self.output_tc):
            self.sizer.Add(output_tc, (i, 0), flag=wx.EXPAND|wx.LEFT|wx.RIGHT|wx.BOTTOM, border=5, span=(1,4))
            #self.sizer.AddGrowableRow(i)

        self.sizer.AddGrowableCol(0)
        self.panel.SetSizer(self.sizer)

        self.trace_queue = []

        # Set some program flags
        self.keepgoing = True
        self.i = 0
        self.j = 0

        frame_sizer = wx.BoxSizer(wx.VERTICAL)
        frame_sizer.Add(self.panel, 1, wx.EXPAND)
        self.SetSizer(frame_sizer)
        self.SetMinSize((600, -1))
        self.Fit()


    def OnCalculate(self):
        self.Show(True)
        self.Center()
        #self.start_bt.Enable(False)

        self.numtasks = self.args['cores']
        self.numproc = self.args['cores']

        # Create the queues
        self.outputQueue = Queue()
        process_list = []

        # The worker processes...
        for n in range(self.numproc):
            try:
                trace_queue = Queue()
                process = Process(target=self.worker, args=(self.trace, n, self.outputQueue, trace_queue))
                process.daemon = True
                process_list.append(process)
                process.start()
            except Exception:
                print('Cannot create process %d' % n)
                self.numproc = n
                break
            self.trace_queue.append(trace_queue)


        # Start processing tasks
        self.processTasks(self.update)

        #if (self.keepgoing):
        #    self.start_bt.Enable(True)

        self.Close()
        return MergeTrace([trace.get() for trace in self.trace_queue], self.args, self.trace.data), self.trace.training_data.par_name, self.trace.prior


    def processTasks(self, resfunc=None):
        self.keepgoing = True
        numprocstart = min(self.numproc, self.numtasks)
        # Submit first set of tasks
        self.j = -1 # done queue index
        self.i = numprocstart - 1 # task queue index
        while (self.j < self.i):
            # Get and print results
            self.j += 1
            output = None
            while output != 'STOP!':
                try:
                    output = self.outputQueue.get()
                    if output != 'STOP!': 
                        resfunc(output)
                except Empty:
                    break
        return
            

    def update(self, output):
        self.output_tc[output[0]].SetValue('Process-%d:  ' % output[0] + output[1])#AppendText(output[1])
        wx.YieldIfNeeded()

    def worker(self, trace_creator, tasknum, outputq, traceq):
        #while True:
        try:
            sys.stdout = QueuePipe(outputq, tasknum)
            traceq.put(trace_creator.CreateTrace(tasknum))
            outputq.put('STOP!') 
        except Empty:
            pass
            #break

    # The worker must not require any existing object for execution!
    worker = classmethod(worker)

class MyApp(wx.App):
    def OnInit(self):
        self.frame = MyFrame(None, -1, 'stdout to GUI using multiprocessing', {'Training_file': 'training/test', 'Output_name':'para', 'cores':5, 'steps':10000})
        self.frame.OnCalculate()#Show(True)
        #self.frame.Center()
        return True

if __name__ == '__main__':
    #freeze_support()
    app = MyApp(0)
    app.MainLoop()
