import autograd.numpy as np
import sys
from autograd import grad

class GradientDescentForEmulator:


    def __init__(self, step_size_scales, step_size_nuggets):
        self.step_scales_size = step_size_scales
        self.step_nuggets_size = step_size_nuggets
        self.func = None
        self.grad_scales_exp = None
        self.grad_nuggets_exp = None
        self.nuggets_log = None
        self.scales_log = None

    def SetFunc(self, func):
        self.func = func

        def scales_exp_cal(t_scales_log):
            return func(scales=np.exp(t_scales_log), nuggets=np.exp(self.nuggets_log))
 
        def nuggets_exp_cal(t_nuggets_log):
            return func(scales=np.exp(self.scales_log), nuggets=np.exp(t_nuggets_log))

        self.grad_scales_exp = grad(scales_exp_cal)
        self.grad_nuggets_exp = grad(nuggets_exp_cal)

    def StepDescent(self, scales, nuggets):
        self.scales_log = np.log(scales)
        self.nuggets_log = np.log(nuggets)
        gradient_scales = self.grad_scales_exp(self.scales_log)
        gradient_nuggets = self.grad_nuggets_exp(self.nuggets_log)

        scale_temp = self.scales_log + gradient_scales*self.step_scales_size
        nuggets_temp = self.nuggets_log + gradient_nuggets*self.step_nuggets_size

        return np.exp(scale_temp), np.exp(nuggets_temp), gradient_scales, gradient_nuggets
        
    def Descent(self, scales, nuggets, nsteps=10, tolerance=1e-3):
        history_scales = []
        history_nuggets = []
        scales = np.array(scales)
        nuggets = np.array(nuggets)
        for i in range(nsteps):
        
            hist_scales, hist_nuggets, grad_scales, grad_nuggets = self.StepDescent(scales, nuggets)
            history_scales.append(hist_scales)
            history_nuggets.append(hist_nuggets)
            (scales, nuggets) = history_scales[-1], history_nuggets[-1]
            mag = np.linalg.norm(grad_scales*self.step_scales_size + grad_nuggets*self.step_nuggets_size)
            sys.stdout.write("\rProcessing %i iteration, gradient magnitude = %f, nuggets = %f, scales = %s" % (i, mag, nuggets, np.array2string(scales)))
            #sys.stdout.flush()
            #if mag < tolerance:
                #break

        return np.array(history_scales), np.array(history_nuggets)
        
