import sys
import wx

from Utilities.ControllerDeviceMP import ControllerDevice
from GUI.GUIController.GUIController import GUIController

def main():
    size = int(sys.argv[1])
    work_environment = ControllerDevice(None, ncores=size)

    # gd.UseDefaultOutput()
    app = wx.App(0)
    controller = GUIController(None, app=app, workenv=work_environment)
    controller.view.Show()

    if len(sys.argv) >= 3:
        controller.LoadFile(sys.argv[2:])

    app.MainLoop()
    work_environment.Close()


if __name__ == "__main__":
    main()
