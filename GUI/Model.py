import math
import re
import sys
import time
import traceback

import matplotlib
import numpy as np
import wx
import wx.lib.agw.speedmeter as SM
from mpi4py import MPI

from Utilities.MasterSlave import MasterSlave, tags

matplotlib.use("WXAgg")
import os
import pickle
import shutil
import tempfile

from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg
from matplotlib.figure import Figure

from MCMCTrace import MCMCParallel, Merging


class RunningAve(object):
    def __init__(self, num=5):
        self.prev_speed = [0] * num

    def GetAve(self, val):
        if val is not None:
            self.prev_speed.pop(0)
            self.prev_speed.append(val)
        return np.mean(self.prev_speed)


class InfoBar(wx.StaticText):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # font = wx.Font(int(200/wx.Font().GetPointSize()[0]), wx.DEFAULT, wx.NORMAL, wx.NORMAL)
        # self.SetFont(font)

    def PrintInfo(self, message):
        self.SetForegroundColour((0, 0, 0))
        self.SetLabel(message)

    def PrintError(self, message):
        self.SetForegroundColour((255, 0, 0))
        self.SetLabel(message)


class EvtSpeedMeter(SM.SpeedMeter):
    def __init__(self, num_ranks, max_speed_per_cpu, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_ranks = num_ranks

        # Set The Region Of Existence Of SpeedMeter (Always In Radians!!!!)
        self.SetAngleRange(-math.pi / 6, 7 * math.pi / 6)

        # Create The Intervals That Will Divide Our SpeedMeter In Sectors
        self.max_speed = num_ranks * max_speed_per_cpu
        intervals = range(0, self.max_speed + 1, int(self.max_speed / 10))
        self.SetIntervals(intervals)
        # Assign The Same Colours To All Sectors (We Simulate A Car Control For Speed)
        # Usually This Is Black
        colours = [wx.WHITE] * 10
        self.SetIntervalColours(colours)

        # Assign The Ticks: Here They Are Simply The String Equivalent Of The Intervals
        ticks = [str(interval) for interval in intervals]
        self.SetTicks(ticks)
        # Set The Ticks/Tick Markers Colour
        self.SetTicksColour(wx.BLACK)
        # We Want To Draw 5 Secondary Ticks Between The Principal Ticks
        self.SetNumberOfSecondaryTicks(5)

        # Set The Font For The Ticks Markers
        # self.SetTicksFont(wx.Font(int(200/wx.Font().GetPointSize()[0]), wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))

        # Set The Text In The Center Of SpeedMeter
        self.SetMiddleText("Evts/s")
        # Assign The Colour To The Center Text
        self.SetMiddleTextColour(wx.BLACK)
        # Assign A Font To The Center Text
        # self.SetMiddleTextFont(wx.Font(int(300/wx.Font().GetPointSize()[0]), wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))

        # Set The Colour For The Hand Indicator
        self.SetHandColour(wx.Colour(255, 50, 0))
        self.SetArcColour(wx.WHITE)
        self.SetSpeedBackground(wx.WHITE)

        # Smooth the speedmeter
        self.SpeedCalculator = RunningAve(2 * num_ranks)
        # self.speed_list = [0]*num_ranks
        self.prev_speed = 0

    def UpdateSpeed(self, int_speed):
        speed = self.SpeedCalculator.GetAve(int_speed)  # np.sum(self.speed_list))
        if speed > self.max_speed:
            speed = self.max_speed
        self.SetSpeedValue(speed)

    def ReturnZero(self, refresh_rate=0.03):
        # graduately reduce the speed to 0
        while True:
            time.sleep(refresh_rate)
            speed = self.SpeedCalculator.GetAve(0)
            if speed <= 0:
                break
            elif speed >= self.max_speed:
                speed = self.max_speed
            self.SetSpeedValue(speed)


class ProgressBar(FigureCanvasWxAgg):
    def __init__(self, tot_num, width, height, parent):
        self.tot_num = tot_num
        self.fig = Figure((width, height), dpi=wx.ScreenDC().GetPPI()[0])
        self.fig.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0)
        super().__init__(parent, -1, self.fig)

        self.progress = self.fig.add_subplot(111)
        data = [0.0, 1.0]
        self.wedges, _ = self.progress.pie(
            data,
            colors=["r", "lightgrey"],
            radius=1.1,
            wedgeprops=dict(width=0.2),
            startangle=-90,
            counterclock=False,
        )
        self.text = self.progress.text(
            0.0,
            0,
            "Progress",
            horizontalalignment="center",
            verticalalignment="bottom",
            fontsize=10,
        )
        self.title = self.progress.text(
            0.0,
            -0.1,
            "Percentage",
            horizontalalignment="center",
            verticalalignment="top",
            fontsize=20,
        )
        self.fig.canvas.draw_idle()

    def UpdateProgress(self, tot_completed):
        if tot_completed > self.tot_num:
            tot_completed = self.tot_num
        theta = -tot_completed * 360 / self.tot_num - 90
        self.wedges[0].set_theta1(theta)
        self.wedges[1].set_theta2(theta)
        self.text.set_text("Finished %d out of %d evts" % (tot_completed, self.tot_num))
        self.title.set_text("%.1f%%" % (100.0 * tot_completed / self.tot_num))
        self.fig.canvas.draw_idle()


class CalculationFrame(wx.Frame):
    def __init__(self, parent, id, title, enviro, tot_per_rank=50000, burnin=1000):
        self.enviro = enviro
        self.orig_stdout = sys.stdout

        """
    All parameters that need adjustment
    """
        self.max_speed_per_cpu = 400
        self.pixel_width = 700
        self.pixel_height = 400
        self.spacer_prop = 0.05

        self.tot_per_rank = tot_per_rank
        self.tot = self.tot_per_rank * enviro.nworkers
        """
    if enviro.nworkers < 10:
      self.refresh_rate = 0.02
    elif enviro.nworkers < 40:
      self.refresh_rate = 0.2/enviro.nworkers
    else:
      self.refresh_rate = 0
    """
        self.refresh_rate = 0.3
        self.refresh_interval = enviro.size * self.refresh_rate
        enviro.RefreshRate(self.refresh_interval)
        enviro.RefreshSmear(self.refresh_interval)

        self.vspacer = self.pixel_height * self.spacer_prop
        self.hspacer = self.pixel_width * self.spacer_prop

        wx.Frame.__init__(
            self,
            parent,
            id,
            title,
            size=(self.pixel_width, self.pixel_height),
            style=wx.DEFAULT_FRAME_STYLE ^ wx.RESIZE_BORDER,
        )
        self.panel = wx.Panel(
            self, wx.ID_ANY, size=(self.pixel_width, self.pixel_height)
        )
        self.panel.SetBackgroundColour("white")

        self.Bind(wx.EVT_CLOSE, self.OnClose)

        """
    Draw Speedometer
    """
        self.speedmeter = EvtSpeedMeter(
            enviro.nworkers,
            self.max_speed_per_cpu,
            parent=self,
            agwStyle=SM.SM_DRAW_HAND
            | SM.SM_DRAW_SECTORS
            | SM.SM_DRAW_MIDDLE_TEXT
            | SM.SM_DRAW_SECONDARY_TICKS,
        )

        """
    Draw Progress bar
    """
        width = 0.5 * (self.pixel_width - 3 * self.hspacer) / wx.ScreenDC().GetPPI()[0]
        height = 0.9 * self.pixel_height / wx.ScreenDC().GetPPI()[0]
        self.progress_bar = ProgressBar(self.tot, width, height, self)

        MatPlotSizer = wx.BoxSizer(wx.HORIZONTAL)
        MatPlotSizer.Add(self.progress_bar, 1, wx.EXPAND)

        """
    Arrange the 2 meters side by side 
    ini the same sizer
    """
        top_sizer = wx.BoxSizer(wx.HORIZONTAL)
        top_sizer.AddSpacer(self.hspacer)
        top_sizer.Add(MatPlotSizer, 1, wx.EXPAND)
        top_sizer.AddSpacer(self.hspacer)
        top_sizer.Add(self.speedmeter, 1, wx.EXPAND)
        top_sizer.AddSpacer(self.hspacer)

        """
    Infobar and its sizer
    """
        info_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.info_bar = InfoBar(self.panel, label="Progress report")
        info_sizer.AddSpacer(self.hspacer)
        info_sizer.Add(self.info_bar)
        info_sizer.AddSpacer(self.hspacer)

        """
    Put info bar below thew  meters
    """
        panel_sizer = wx.BoxSizer(wx.VERTICAL)
        panel_sizer.Add(top_sizer, 0)
        panel_sizer.Add(info_sizer, 0, wx.EXPAND)
        self.panel.SetSizer(panel_sizer)
        frame_sizer = wx.BoxSizer(wx.VERTICAL)
        frame_sizer.Add(self.panel, 0)

        self.SetBackgroundColour("white")
        self.SetSizer(frame_sizer)
        self.SetAutoLayout(True)
        self.Layout()
        self.Fit()

    def OnCalculate(self, args):
        self.Center()

        # create temporary file for which each rank must write to
        with tempfile.TemporaryDirectory(dir=os.path.dirname(os.path.realpath(__file__))) as dirpath:
            self.enviro.Submit(
                MCMCParallel,
                config_file=args["config_file"],
                dirpath=dirpath,
                nevents=args["nsteps"],
                burnin=args["burnin"],
            )
            self.info_bar.PrintInfo("%d workers are working" % self.enviro.nworking)

            # check for jobs completions and collect results
            # avarage speed for each rank
            start_time = time.time()
            num_list = [0] * self.enviro.nworkers
            num_prev = 0
            time_prev = start_time
            last_update = start_time

            try:
                while self.enviro.IsRunning(self.refresh_rate / 10):  # self.refresh_rate):
                    source, result, tag = self.enviro.stdout
                    idx = source - 1
                    new_time = time.time()
                    if tag == tags.NOTHING:
                        speed = None
                    else:
                        if tag == tags.END:
                            speed = 0
                            self.info_bar.PrintInfo(
                                "%d workers are still working. Worker %d completed"
                                % (self.enviro.nworking, idx)
                            )
                        elif tag == tags.ERROR:
                            speed = 0
                            self.info_bar.PrintInfo('Error from worker %d' % idx)
                        else:
                            num = re.findall(r"\d+\.?\d*", result)
                            last_num = int(num[1])
                            num_list[idx] = last_num

                            if len(num) > 1:

                                new_tot = np.sum(num_list)
                                dn = new_tot  # new_tot - num_prev
                                dt = new_time - start_time  # new_time-time_prev

                                time_prev = new_time
                                num_prev = new_tot
                                speed = dn / dt

                                self.speedmeter.UpdateSpeed(speed)
                                self.progress_bar.UpdateProgress(np.sum(num_prev))
                                last_update = new_time
                                wx.YieldIfNeeded()

                self.info_bar.PrintInfo("All calculations completed. Merging...")
                self.progress_bar.UpdateProgress(np.sum(num_prev))
                self.speedmeter.ReturnZero()
                wx.YieldIfNeeded()
            except Exception as e:
                raise e
            else:
                try:
                    result = Merging(
                        args["config_file"], self.enviro.results, args["clear_trace"]
                    )
                except Exception as e:
                    raise e
                    #traceback.print_exc()
                    #print("Error merging files.")
                    #sys.stdout.flush()
                else:
                    self.info_bar.PrintInfo("Merging completed.")
                    wx.YieldIfNeeded()
            finally:
                self.Destroy()

    def OnClose(self, event):
        # self.enviro.Close()
        self.Destroy()


class MyApp(wx.App):  # , wx.lib.mixins.inspection.InspectionMixin):
    def __init__(self, size, enviro, args):
        self.size = size
        self.enviro = enviro
        self.args = args
        wx.App.__init__(self, False)

    def OnInit(self):
        # self.Init()
        self.frame = CalculationFrame(
            None, -1, "Progress", self.enviro, self.args["nsteps"]
        )
        self.frame.Show()
        self.SetTopWindow(self.frame)
        self.frame.OnCalculate(self.args)
        return True


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
status = MPI.Status()
root = 0

if __name__ == "__main__":
    kargs = {
        "config_file": "/projects/hira/tsangc/GaussianEmulator/result/test.h5",
        "nsteps": 10000,
    }

    work_environment = MasterSlave(comm)

    app = MyApp(size=size, enviro=work_environment, args=kargs)
    app.MainLoop()
