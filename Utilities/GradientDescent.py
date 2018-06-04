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
        
    def Descent(self, scales, nuggets, nsteps=10, tolerance=1e-5):
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
            sys.stdout.flush()
            if mag < tolerance:
                break

        print('')
        return np.array(history_scales), np.array(history_nuggets)
        
class MomentumDescentForEmulator(GradientDescentForEmulator):

     
    def __init__(self, step_size_scales, step_size_nuggets, momentum=0.95):
        GradientDescentForEmulator.__init__(self, step_size_scales, step_size_nuggets)
        self.momentum = momentum
        self.momentum_vector_scale = None
        self.momentum_vector_nugget = None

    def StepDescent(self, scales, nuggets):
        self.scales_log = np.log(scales)
        self.nuggets_log = np.log(nuggets)
        if self.momentum_vector_scale is None:
            self.momentum_vector_scale = np.zeros_like(self.scales_log)
        if self.momentum_vector_nugget is None:
            self.momentum_vector_nugget = np.zeros_like(self.nuggets_log)

        gradient_scales = self.grad_scales_exp(self.scales_log)
        gradient_nuggets = self.grad_nuggets_exp(self.nuggets_log)
            
        self.momentum_vector_scale = self.momentum*self.momentum_vector_scale \
                                     - self.step_scales_size*gradient_scales
        self.momentum_vector_nugget = self.momentum*self.momentum_vector_nugget \
                                      - self.step_nuggets_size*gradient_nuggets

        scales_temp = self.scales_log - self.momentum_vector_scale
        nuggets_temp = self.nuggets_log - self.momentum_vector_nugget

        return np.exp(scales_temp), np.exp(nuggets_temp), gradient_scales, gradient_nuggets       

class RMSPropForEmulator(GradientDescentForEmulator):

     
    def __init__(self, step_size_scales, step_size_nuggets, momentum=0.9):
        GradientDescentForEmulator.__init__(self, step_size_scales, step_size_nuggets)
        self.momentum = momentum
        self.momentum_vector_scale = None
        self.momentum_vector_nugget = None

    def StepDescent(self, scales, nuggets):
        self.scales_log = np.log(scales)
        self.nuggets_log = np.log(nuggets)
        gradient_scales = self.grad_scales_exp(self.scales_log)
        gradient_nuggets = self.grad_nuggets_exp(self.nuggets_log)
        if self.momentum_vector_scale is None:
            self.momentum_vector_scale = np.zeros_like(self.scales_log)
        if self.momentum_vector_nugget is None:
            self.momentum_vector_nugget = np.zeros_like(self.nuggets_log)
            
        self.momentum_vector_scale = self.momentum*self.momentum_vector_scale \
                                     + (1 - self.momentum)*gradient_scales*gradient_scales
        self.momentum_vector_nugget = self.momentum*self.momentum_vector_nugget \
                                      + (1 - self.momentum)*gradient_nuggets*gradient_nuggets

        scales_temp = self.scales_log + self.step_scales_size*gradient_scales/np.sqrt(self.momentum_vector_scale + 1e-10)
        nuggets_temp = self.nuggets_log + self.step_nuggets_size*gradient_nuggets/np.sqrt(self.momentum_vector_nugget + 1e-10)

        return np.exp(scales_temp), np.exp(nuggets_temp), gradient_scales, gradient_nuggets       

class AdamForEmulator(GradientDescentForEmulator):

     
    def __init__(self, step_size_scales, step_size_nuggets, beta1=0.9, beta2=0.999):
        GradientDescentForEmulator.__init__(self, step_size_scales, step_size_nuggets)
        self.beta1 = beta1
        self.beta2 = beta2
        self.m_scales = None
        self.m_nuggets = None
        self.s_scales = None
        self.s_nuggets = None
        self.iteration = 1

    def StepDescent(self, scales, nuggets):
        self.scales_log = np.log(scales)
        self.nuggets_log = np.log(nuggets)
        gradient_scales = self.grad_scales_exp(self.scales_log)
        gradient_nuggets = self.grad_nuggets_exp(self.nuggets_log)
        if self.m_scales is None:
            self.m_scales = np.zeros_like(self.scales_log)
            self.s_scales = np.zeros_like(self.scales_log)

            self.m_nuggets = np.zeros_like(self.nuggets_log)
            self.s_nuggets = np.zeros_like(self.nuggets_log)
            
        self.m_scales = self.beta1*self.m_scales - (1 - self.beta1)*gradient_scales
        self.m_nuggets = self.beta1*self.m_nuggets - (1 - self.beta1)*gradient_nuggets

        self.s_scales = self.beta2*self.s_scales + (1 - self.beta2)*gradient_scales*gradient_scales
        self.s_nuggets = self.beta2*self.s_nuggets + (1 - self.beta2)*gradient_nuggets*gradient_nuggets

        """
        self.m_scales = self.m_scales/(1 - self.beta1**self.iteration)
        self.m_nuggets = self.m_nuggets/(1 - self.beta1**self.iteration)

        self.s_scales = self.s_scales/(1 - self.beta2**self.iteration)
        self.s_nuggets = self.s_nuggets/(1 - self.beta2**self.iteration)

        self.iteration = self.iteration + 1
        """
        scales_temp = self.scales_log - self.step_scales_size*self.m_scales/np.sqrt(self.s_scales + 1e-10)
        nuggets_temp = self.nuggets_log - self.step_nuggets_size*self.m_nuggets/np.sqrt(self.s_nuggets + 1e-10)

        return np.exp(scales_temp), np.exp(nuggets_temp), gradient_scales, gradient_nuggets       

