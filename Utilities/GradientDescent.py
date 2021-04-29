import sys

import autograd.numpy as np
from autograd import grad
from pubsub import pub


def DefaultOutput(step, progress, mag, nuggets, scales):
    sys.stdout.write(
        "\rProcessing %i iteration, progress = %.1f%% gradient magnitude = %2.6f, nuggets = %2.3f, scales = %s      " %
        (step,
         progress,
         mag,
         nuggets,
         np.array2string(
             scales,
             formatter={
                 "float_kind": lambda x: "%02.3f" %
                 x}),
         ))
    if progress == 100:
        sys.stdout.write("\n")
    sys.stdout.flush()


def NewLine():
    pass
    # print('')


def UseDefaultOutput():
    pub.subscribe(DefaultOutput, "GradientProgress")
    pub.subscribe(NewLine, "GradientEnd")


def GetOptimizer(name, *args, **kwargs):
    if name == "GradientDescent":
        return GradientDescentForEmulator(*args, **kwargs)
    elif name == "MomentumDescent":
        return MomentumDescentForEmulator(*args, **kwargs)
    elif name == "RMSProp":
        return RMSPropForEmulator(*args, **kwargs)
    elif name == "Adam":
        return AdamForEmulator(*args, **kwargs)


class GradientDescentForEmulator:
    class ProgressCalculator:
        def __init__(self):
            self.startMag = None
            self.tolMag = None
            self.magSlope = None
            self.lastProgress = None
            self.scale_max = 5000  # larger than this number and covariant matrix will not pos define due to numerical errors

        def Get(self, nsteps, step, mag, tolerance):
            stepProgress = step / nsteps * 100
            if self.startMag is None:
                # progress is calculated with log scale, with progress = 0 at
                # first step and 1 when mag = tolerance
                self.startMag = np.log(mag)
                self.totMag = np.log(tolerance)
                self.magSlope = self.totMag - self.startMag
                self.lastProgress = 0
            magProgress = (np.log(mag) - self.startMag) * 100 / self.magSlope
            progress = magProgress if magProgress > stepProgress else stepProgress
            # progress could only move forward
            # prevent it from going back
            if progress > self.lastProgress:
                self.lastProgress = progress
            else:
                progress = self.lastProgress
            return progress if progress < 100 else 100

    def __init__(self, step_size_scales, step_size_nuggets):
        self.step_scales_size = step_size_scales
        self.step_nuggets_size = step_size_nuggets
        self.func = None
        self.grad_scales_exp = None
        self.grad_nuggets_exp = None
        self.nuggets_log = None
        self.scales_log = None
        self.scaleMax = 1000

    def SetFunc(self, func):
        self.func = func

        def exp_cal(t_para_log):
            return func(np.exp(t_para_log))

        self.grad_exp = grad(exp_cal)

    def StepDescent(self, para):
        self.para_log = np.log(para)
        gradient = self.grad_exp(self.para_log)

        para = self.para_log + gradient * self.step_size

        return np.exp(para), gradient

    def Descent(
            self,
            scales,
            nuggets,
            nsteps=10,
            tolerance=1e-5,
            progress=DefaultOutput):
        history_para = []
        scales = np.atleast_1d(scales)
        nuggets = np.atleast_1d(nuggets)
        para = np.concatenate([nuggets, scales])
        self.step_size = np.full((1,), self.step_nuggets_size)
        self.step_size = np.concatenate(
            [self.step_size, np.full(scales.shape, self.step_scales_size)]
        )

        pCalculator = GradientDescentForEmulator.ProgressCalculator()
        progress = 0
        for i in range(nsteps):
            new_para, grad = self.StepDescent(para)

            # stop updating parameters that reaches max values
            idCap = new_para > self.scaleMax
            new_para[idCap] = self.scaleMax
            grad[idCap] = 0

            para = new_para
            history_para.append(new_para)
            (scales, nuggets) = new_para[1:], new_para[0]

            mag = np.linalg.norm(grad * self.step_size)

            progress = pCalculator.Get(nsteps, i, mag, tolerance)
            pub.sendMessage(
                "GradientProgress",
                step=i,
                progress=progress,
                mag=mag,
                nuggets=nuggets,
                scales=scales,
            )
            # or mag < 0.5*(self.step_scales_size + self.step_nuggets_size):
            if (mag < tolerance):
                break
        if progress < 100:
            pub.sendMessage(
                "GradientProgress",
                step=i,
                progress=100,
                mag=mag,
                nuggets=nuggets,
                scales=scales,
            )
        pub.sendMessage("GradientEnd")
        return np.array(history_para)


class MomentumDescentForEmulator(GradientDescentForEmulator):
    def __init__(self, step_size_scales, step_size_nuggets, momentum=0.95):
        GradientDescentForEmulator.__init__(
            self, step_size_scales, step_size_nuggets)
        self.momentum = momentum
        self.momentum_vector = None

    def StepDescent(self, para):
        self.para_log = np.log(para)
        if self.momentum_vector is None:
            self.momentum_vector = np.zeros_like(self.para_log)

        gradient = self.grad_exp(self.para_log)

        self.momentum_vector = (
            self.momentum * self.momentum_vector - self.step_size * gradient
        )

        para_temp = self.para_log - self.momentum_vector
        return np.exp(para_temp), gradient


class RMSPropForEmulator(GradientDescentForEmulator):
    def __init__(self, step_size_scales, step_size_nuggets, momentum=0.9):
        GradientDescentForEmulator.__init__(
            self, step_size_scales, step_size_nuggets)
        self.momentum = momentum
        self.momentum_vector = None

    def StepDescent(self, para):
        self.para_log = np.log(para)
        gradient = self.grad_exp(self.para_log)
        if self.momentum_vector is None:
            self.momentum_vector = np.zeros_like(self.para_log)

        self.momentum_vector = (
            self.momentum * self.momentum_vector
            + (1 - self.momentum) * gradient * gradient
        )

        para_temp = self.para_log + self.step_size * gradient / np.sqrt(
            self.momentum_vector + 1e-10
        )

        return np.exp(para_temp), gradient


class AdamForEmulator(GradientDescentForEmulator):
    def __init__(
            self,
            step_size_scales,
            step_size_nuggets,
            beta1=0.9,
            beta2=0.99):
        GradientDescentForEmulator.__init__(
            self, step_size_scales, step_size_nuggets)
        self.beta1 = beta1
        self.beta2 = beta2
        self.m_para = None
        self.s_para = None
        self.iteration = 1

    def StepDescent(self, parameters):
        self.para_log = np.log(parameters)
        gradient = self.grad_exp(self.para_log)
        if self.m_para is None:
            self.m_para = np.zeros_like(self.para_log)
            self.s_para = np.zeros_like(self.para_log)

        self.m_para = self.beta1 * self.m_para - (1 - self.beta1) * gradient
        self.s_para = self.beta2 * self.s_para + \
            (1 - self.beta2) * gradient * gradient

        para_temp = self.para_log - self.step_size * self.m_para / np.sqrt(
            self.s_para + 1e-10
        )
        return np.exp(para_temp), gradient
