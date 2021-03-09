import sys
import wx

from Utilities.MasterSlaveMP import MasterSlave
from GUI.GUIController.GUIController import GUIController

def main():
    size = 10
    work_environment = MasterSlave(None, ncores=size)

    # gd.UseDefaultOutput()
    app = wx.App(0)
    controller = GUIController(None, app=app, workenv=work_environment)
    controller.view.Show()

    if len(sys.argv) >= 2:
        controller.LoadFile(sys.argv[1:])

    app.MainLoop()
    work_environment.Close()


if __name__ == "__main__":
    main()
