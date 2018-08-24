import autograd.numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import math

from autograd import grad
from functools import partial
from autograd.scipy.linalg import solve_triangular
from scipy.spatial import distance
from autograd.scipy.stats import multivariate_normal as mvn

from Utilities.GradientDescent import *# GradientDescentForEmulator

def RBF(xp, xq, scales):
    y2 = np.expand_dims(xp.T, axis=1)
    y3 = np.expand_dims(xq.T, axis=2)
    return np.exp(-((y3 - y2)*(y3 - y2)/(scales*scales)).sum(axis=0))

def squared_exponential(xp, xq, scales):
    scale = scales[..., None, None]
    y2 = np.expand_dims(xp.T, axis=1)
    y3 = np.expand_dims(xq.T, axis=2)
    return np.exp(-((y3 - y2)*(y3 - y2)/(scale*scale)).sum(axis=0))


class EmulatorMultiOutput:


    def __init__(self, input_, target):
        self.num_output = target.shape[1]
        self.num_input = input_.shape[1]
        self.emulator_list = []
        for i in range(0, self.num_output):
            emulator = Emulator(input_, target[:, i])
            self.emulator_list.append(emulator)

    def SetCovariance(self, covariance):
        for emulator in self.emulator_list:
            emulator.SetCovariance(covariance)

    def Train(self, initial_scales=0.5, initial_nuggets=1, 
              scales_rate=0.1, nuggets_rate=0.05, max_step = 300):
        scales = []
        nuggets = []
        for emulator in self.emulator_list:
            gd = GetOptimizer('Adam', scales_rate, nuggets_rate)#
            # trainning with marginallikelihood instead of LOOCV 
            #gd.SetFunc(emulator.LOOCrossValidation)
            gd.SetFunc(emulator.MarginalLikelihood)
            history_scale, history_nuggets = gd.Descent(initial_scales, initial_nuggets, max_step)
            
            # use the last trained scales and nuggets
            scales.append(history_scale[-1])
            nuggets.append(history_nuggets[-1])
            emulator.SetScales(history_scale[-1])
            emulator.SetNuggets(history_nuggets[-1])
            emulator.StartUp()
        return scales, nuggets

    def GetScales(self):
        return [emulator.scales for emulator in self.emulator_list]

    def GetNuggets(self):
        return [emulator.nuggets for emulator in self.emulator_list]

    def SetScales(self, scales):
        for emulator, value in zip(self.emulator_list, scales):
            emulator.SetScales(value)

    def SetNuggets(self, nuggets):
        for emulator, value in zip(self.emulator_list, nuggets):
            emulator.SetNuggets(value)

    def StartUp(self):
        for emulator in self.emulator_list:
            emulator.StartUp()

    def Emulate(self, input_):
        mean_list = []
        var_list = []
        for i in range(0, self.num_output):
            (mean, var) = self.emulator_list[i].Emulate(input_)
            mean_list.append(mean)
            var_list.append(var[0][0])
        return np.array(mean_list), np.array(var_list)

        

class Emulator:


    def __init__(self, input_, target):
        self.input_ = input_
        self.target = target
        self.covariance = squared_exponential
        self.scales = None
        self.nuggets = None
        self.cholesky = None
        self.alpha = None
        self.cov_matrix = None 

        assert self.input_.shape[0] == self.target.shape[0], \
               "Number of rows of input is %d, target is %d, and they should be the same" \
               % (self.input_.shape[0], self.target.shape[0])

    def SetCovariance(self, covariance):
        self.covariance = covariance

    def SetScales(self, scales):
        self.scales = scales

    def SetNuggets(self, nuggets):
        self.nuggets = nuggets


    def StartUp(self):
        assert self.covariance, "You must SetCovariance first"
        assert self.scales is not None, "You must SetScales first"
        assert self.nuggets is not None, "You must SetNuggets first"

        K = self.covariance(self.input_, self.input_, self.scales)
        K = K + self.nuggets*self.nuggets*np.eye(K.shape[0])
        L = np.linalg.cholesky(K)
        self.alpha = solve_triangular(L.transpose(), solve_triangular(L, self.target, lower=True))
        self.cholesky = L
        self.cov_matrix = K

    def MarginalLikelihood(self, scales=None, nuggets=None):
        if(scales is not None and nuggets is not None):
            self.scales = scales
            self.nuggets = nuggets
            self.StartUp()

        assert self.cholesky is not None and self.alpha is not None, \
               "You must StartUp the emulator first before using it"

        log_marginal_likelihood = - 0.5*np.dot(self.target.transpose(), self.alpha) \
                                  - np.log(np.diag(self.cholesky)).sum() \
                                  - self.cholesky.shape[0]*np.log(2*np.pi)
        return log_marginal_likelihood.sum()

    def LOOCrossValidation(self, scales=None, nuggets=None):
        if(scales is not None and nuggets is not None):
            self.scales = scales
            self.nuggets = nuggets
            self.StartUp()


        Kinv_diag = np.diag(np.linalg.inv(self.cov_matrix))
        LOO_mean_minus_target = self.alpha/Kinv_diag
        LOO_sigma = np.reciprocal(Kinv_diag)
        log_cv = - 0.5*(np.log(LOO_sigma) + np.square(LOO_mean_minus_target)*Kinv_diag + math.log(2*math.pi))
        return log_cv.sum()

    def Emulate(self, input_):
        assert self.cholesky is not None and self.alpha is not None, \
               "You must StartUp the emulator first before using it"

        kstar = self.covariance(input_, self.input_, self.scales)
        predictive_mean = np.dot(np.transpose(kstar), self.alpha)
        v = solve_triangular(self.cholesky, kstar, lower=True)
        predictive_variance = self.covariance(input_, input_, self.scales) - np.dot(np.transpose(v), v)
        return (predictive_mean, predictive_variance)

    def LogLikelihood(self, input_, value):
        (predictive_mean, predictive_variance) = self.Emulate(input_)
        return mvn.logpdf(value, predictive_mean, predictive_variance)
        #return -0.5*(np.log(predictive_variance) + np.square((input_ - value)/predictive_variance) + np.log(np.pi)).sum()
        
class EmulatorMaster(EmulatorMultiOutput):


    def __init__(self, input_, target, input_pipe, output_pipe, fit=True):
        if fit:
            input_pipe.Fit(input_)
            output_pipe.Fit(target)

        input_ = input_pipe.Transform(input_)
        target = output_pipe.Transform(target)

        self.input_pipe = input_pipe
        self.output_pipe = output_pipe
        EmulatorMultiOutput.__init__(self, input_, target)

    def __repr__(self):
        string = "Input Pipe:  "
        string += repr(self.input_pipe)
        string += "\n\nOutput Pipe:  "
        string += repr(self.output_pipe)
        string += "\n\nEmulator covariance function:  "
        string += self.emulator_list[0].covariance.__name__
        return string + "\n"

    def Emulate(self, input_):
        input_ = self.input_pipe.Transform(input_).reshape(1,-1)
        mean, cov = EmulatorMultiOutput.Emulate(self, input_)
        return self.output_pipe.TransformInv(mean.flatten()), self.output_pipe.TransformCovInv(np.diag(cov))

    def ResetData(self, input_, target):
        self = self.__init__(input_, target, self.input_pipe, self.output_pipe, False)
