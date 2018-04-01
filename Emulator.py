import autograd.numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import math

from autograd import grad
from functools import partial
from autograd.scipy.linalg import solve_triangular
from scipy.spatial import distance
from autograd.scipy.stats import multivariate_normal as mvn

from GradientDescent import GradientDescentForEmulator


def squared_exponential(xp, xq, scales):
    """ Accept 1D array for xp and xq
    will reshape them into corresponding 2D matrix inside
    """
    if xp.shape[0] > 1:
        distance_ = distance.cdist(xq, xp, 'euclidean')
        distance2 = np.square(distance_)
    else:
        """
        this is for autograd
        it does not work with cdist
        so if we are only asking for 1 output at a time
        """
        distance_ = xp - xq
        distance2 = distance_*distance_
        distance2 = distance2.sum(axis=1).reshape(-1,1)
    return np.exp( - (0.5 / (scales*scales) * distance2))

class EmulatorMultiOutput:


    def __init__(self, input_, target):
        self.num_output = target.shape[1]
        self.num_input = input_.shape[1]
        self.emulator_list = []
        for i in xrange(0, self.num_output):
            emulator = Emulator(input_, target[:, i])
            self.emulator_list.append(emulator)

    def SetCovariance(self, covariance):
        for emulator in self.emulator_list:
            emulator.SetCovariance(covariance)

    def Train(self, initial_scales=0.5, initial_nuggets=1, 
              scales_rate=0.001, nuggets_rate=0.01, max_step = 300):
        for emulator in self.emulator_list:
            gd = GradientDescentForEmulator(scales_rate, nuggets_rate)
            # trainning with marginallikelihood instead of LOOCV 
            gd.SetFunc(emulator.MarginalLikelihood)
            history = gd.Descent(initial_scales, initial_nuggets, max_step)
            
            # use the last trained scales and nuggets
            emulator.SetScales(history[-1, 0])
            emulator.SetNuggets(history[-1, 1])
            emulator.StartUp()

    def Emulate(self, input_):
        mean_list = []
        var_list = []
        for i in xrange(0, self.num_output):
            print(self.num_output)
            (mean, var) = self.emulator_list[i].Emulate(input_)
            mean_list.append(mean)
            var_list.append(var)
        return np.array(mean_list), np.array(var_list)

    def LogLikelihood(self, input_, value, value_cov):
        log_sum = 0
        assert input_.shape[0] == 1, "LogLikelihood can only take 1 value at a time"
        mean_list = []
        cov_list = []
        for i in xrange(0, self.num_output):
           (mean, var) = self.emulator_list[i].Emulate(input_)
           mean_list.append(mean[0])
           cov_list.append(var[0][0])
        mean = np.array(mean_list)
        cov = np.diag(np.array(cov_list)) + value_cov
        return np.array(mvn.logpdf(value, mean, cov))
        

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
        K = K + self.nuggets*np.eye(K.shape[0])
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
        #print(self.target, self.alpha, self.cholesky)
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
        

