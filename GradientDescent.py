import autograd.numpy as np
import sys
from autograd import grad

class GradientDescentForEmulator:


    def __init__(self, step_size_scales, step_size_nuggets):
        self.step_size = np.array([step_size_scales, step_size_nuggets])
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
        self.par_log = np.array([self.scales_log, self.nuggets_log])
        gradient_ = np.array([self.grad_scales_exp(self.scales_log), self.grad_nuggets_exp(self.nuggets_log)])

        par_temp = self.par_log + gradient_*self.step_size

        return np.exp(par_temp), gradient_
        
    def Descent(self, scales, nuggets, nsteps=10, tolerance=1e-3):
        history = []
        scales = np.array(scales)
        nuggets = np.array(nuggets)
        for i in range(nsteps):
        
            hist, grad = self.StepDescent(scales, nuggets)
            history.append(hist)
            (scales, nuggets) = history[-1]
            mag = np.linalg.norm(grad*self.step_size)
            sys.stdout.write("\rProcessing %i iteration, gradient magnitude = %f, scales = %f, nuggets = %f" % (i, mag, scales, nuggets))
            sys.stdout.flush()
            #if mag < tolerance:
                #break

        return np.array(history)
        
