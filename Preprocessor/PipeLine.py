import numpy as nnp
import autograd.numpy as np
import autograd.scipy.linalg as linalg
from functools import partial
from autograd.scipy.linalg import solve_triangular
from scipy.spatial.distance import cdist

from Utilities.GradientDescent import *


def DistSquareMatrix(xp, xq):
    xps2 = np.sum(np.square(xp), 1)
    xqs2 = np.sum(np.square(xq), 1)
    #return np.sum(np.square(xp[:, np.newaxis, :] - xq[np.newaxis, :, :]), axis=-1)
    return xps2.reshape((-1,1)) + xqs2.reshape((1,-1)) - 2*np.matmul(xp, xq.T)


def squared_exponential(xp, xq, scales):
    xp = xp/scales.reshape(1, -1)
    xq = xq/scales.reshape(1, -1)
    return np.exp(-DistSquareMatrix(xq, xp))
    """
    xp = xp/scales.reshape(1, -1)
    xq = xq/scales.reshape(1, -1)
    dist = cdist(xp, xq).T
    return np.exp(-dist)
    """


class Transformer:
    
    def __init__(self):
        pass

    def __repr__(self):
        return("{}()").format(self.__class__.__name__)

    def Fit(self, X, Y):
        pass

    def Score(self, X, Y):
        predict_Y, _ = self.Predict(X)
        u = Y - predict_Y
        v = Y - np.mean(Y, axis=0)
        return 1 - ((u*u).sum()/(v*v).sum())

    def _TransformX(self, X):
        return X

    def _TransformXInv(self, X):
        return X

    def _TransformY(self, Y):
        return Y

    def _TransformYInv(self, Y):
        return Y

    def TransformCov(self, data_cov):
        return data_cov

    def TransformCovInv(self, data_cov):
        return data_cov

    """
    The 2 functions below should not be touched
    """
    def Transform(self, X, Y):
        return self._TransformX(X), self._TransformY(Y)

    def TransformInv(self, X, Y):
        return self._TransformXInv(X), self._TransformYInv(Y)


class PipeLine(Transformer):

    
    def __init__(self, named_steps):
        self.named_steps = named_steps
        self.predictor = named_steps[-1][1]
        self._steps = {}
        for name, step in self.named_steps:
            self._steps[name] = step

    def __getitem__(self, key):
        return self._steps[key]

    def Fit(self, X, Y):
        for name, step in self.named_steps:
            step.Fit(X, Y)
            X, Y = step.Transform(X, Y)

    def __repr__(self):
        string = "%s([" % self.__class__.__name__
        for name, pipe in self.named_steps:
            string += "('%s', %s)," % (name, repr(pipe))
        string +="])"
        return string

    def Predict(self, X):
        X = self._TransformX(X)
        Y, cov = self.predictor.Predict(X)
        return self._TransformYInv(Y), self.TransformCovInv(cov)

    def _TransformX(self, X):
        for name, step in self.named_steps:
            X = step._TransformX(X) 
        return X

    def _TransformY(self, Y):
        for name, step in self.named_steps:
            Y = step._TransformY(Y)
        return Y

    def TransformCov(self, data_cov):
        for name, step in self.named_steps:
            data_cov = step.TransformCov(data_cov) 
        return data_cov

    def _TransformXInv(self, X):
        for name, step in reversed(self.named_steps):
            X = step._TransformXInv(X) 
        return X

    def _TransformYInv(self, Y):
        for name, step in reversed(self.named_steps):
            Y = step._TransformYInv(Y)
        return Y

    def TransformCovInv(self, data_cov):
        for name, step in reversed(self.named_steps):
            data_cov = step.TransformCovInv(data_cov) 
        return data_cov


class Normalize(Transformer):
    
    
    def __init__(self, ignore_X=False):
        self.Xsigma = None
        self.Ysigma = None
        self.Xmean = None
        self.Ymean = None
        self.ignore_X = ignore_X

    def __repr__(self):
        return("{}(ignore_X={!r})").format(self.__class__.__name__, self.ignore_X)

    def Fit(self, X, Y):
        if self.ignore_X:
            self.Xmean = 0
            self.Xsigma = 1
        else:
            self.Xmean = np.mean(X, axis=0)
            self.Xsigma = np.std(X, axis=0)
        self.Ymean = np.mean(Y, axis=0)
        self.Ysigma = np.std(Y, axis=0)

    def _TransformX(self, X):
        return (X - self.Xmean) / self.Xsigma
 
    def _TransformY(self, Y):
        return (Y - self.Ymean) / self.Ysigma

    def TransformCov(self, data_cov):
        transform = np.diag(np.reciprocal(self.Ysigma))
        return np.matmul(np.matmul(transform.T, data_cov), transform)

    def _TransformXInv(self, X):
        return X*self.Xsigma + self.Xmean
 
    def _TransformYInv(self, Y):
        return Y*self.Ysigma + self.Ymean
 
    def TransformCovInv(self, data_cov):
        transform_inv = np.diag(self.Ysigma)
        return np.matmul(np.matmul(transform_inv.T,data_cov), transform_inv)

class PCA(Transformer):


    def __init__(self, component, percentage=None):
        self.cov = None
        self.eigval = None
        self.eigvec = None
        self.component = component
        self.percentage = percentage
        self.reconstruction_error = 0

    def __repr__(self):
        return('{}({}, {})').format(self.__class__.__name__, self.component, self.percentage)

    def Fit(self, X, Y):
        self.cov = np.cov(Y.T)
        if not self.cov.shape:
            # you could be spllied with a 1 feature data set, in which cas self.cov is just a number
            self.eigval = self.cov
            self.eigvec = np.eye(1)
            self.cov = self.cov.reshape(-1,1)
        else:
            self.eigval, self.eigvec = np.linalg.eigh(self.cov)
            idx = self.eigval.argsort()[::-1]
            self.eigval = self.eigval[idx]
            self.eigvec = self.eigvec[:,idx]
            if self.percentage is not None:
                total_val = sum(self.eigval)
                running_fraction = np.cumsum(self.eigval)/total_val
                self.component = np.searchsorted(running_fraction, self.percentage)

            assert self.component <= Y.shape[1], "number of components cannot exceed number of variables"
            self.reconstruction_error = np.mean(self.eigval[self.component:])
            if self.reconstruction_error is None or np.isnan(self.reconstruction_error):
                self.reconstruction_error = 0
            self.eigval = self.eigval[0:self.component]
            self.eigvec = self.eigvec[:, 0:self.component]

    def _TransformY(self, Y):
        return np.matmul(Y, self.eigvec)

    def TransformCov(self, data_cov):
        return np.matmul(np.matmul(self.eigvec.T, data_cov), self.eigvec)

    def _TransformYInv(self, Y):
        return np.matmul(Y, self.eigvec.T);

    def TransformCovInv(self, data_cov):
        return np.matmul(np.matmul(self.eigvec, data_cov), self.eigvec.T) + self.reconstruction_error*np.eye(self.cov.shape[0])

class Emulator(Transformer):

    
    def __init__(self, 
                 covariance=squared_exponential, 
                 initial_scales=0.5,
                 initial_nuggets = 0.5,
                 scales_rate=0.05, 
                 nuggets_rate=0.05,
                 max_steps=1000,
                 scales=None,
                 nuggets=None,
                 save_train_history=False):
        self.covariance = squared_exponential
        self.initial_scales = initial_scales
        self.scales_rate = scales_rate
        self.initial_nuggets = initial_nuggets
        self.nuggets_rate = nuggets_rate
        self.max_steps = max_steps
        self.scales = scales
        self.nuggets = nuggets
        self.save_train_history = save_train_history

    def Fit(self, X, Y):
        assert X.shape[0] == Y.shape[0], \
            "Number of samples in X = %d and Y = %d are not the same." \
            % (X.shape[0], Y.shape[0])
        assert Y.shape[1] == 1, \
            "Individual emulator can only handle one feature while supplied samples has %d features" \
            % (Y.shape[1])

        self.X = X
        self.Y = Y
        history_para = None
        if self.scales is None or self.nuggets is None:
            gd = GetOptimizer('Adam', self.scales_rate, self.nuggets_rate)
            gd.SetFunc(self._LOOCrossValidation)
            history_para = gd.Descent(self.initial_scales, self.initial_nuggets, self.max_steps)

            self.scales = history_para[-1][1:]
            self.nuggets = history_para[-1][0]

        self._CalculateNecessaryMatrices(self.scales, self.nuggets)
        if self.save_train_history:
            self.history_para = history_para

    def _CalculateNecessaryMatrices(self, scales, nuggets):
        K = self.covariance(self.X, self.X, scales)
        K = K + nuggets*nuggets*np.eye(K.shape[0])
        L = np.linalg.cholesky(K)
        self.alpha = solve_triangular(L.transpose(), solve_triangular(L, self.Y, lower=True))
        self.cholesky = L
        self.cov_matrix = K

    def _LOOCrossValidation(self, hyperparameters):#scales, nuggets):
        self._CalculateNecessaryMatrices(scales=hyperparameters[1:], nuggets=hyperparameters[0])
        Kinv_diag = np.diag(np.linalg.inv(self.cov_matrix)).reshape(-1,1)
        LOO_mean_minus_target = self.alpha/Kinv_diag
        LOO_sigma = np.reciprocal(Kinv_diag)
        log_CV = -0.5*(np.log(LOO_sigma) + np.square(LOO_mean_minus_target)*Kinv_diag + np.log(2*np.pi))
        #print(self.alpha.shape, self.cholesky.shape, self.cov_matrix.shape, Kinv_diag.shape)
        return log_CV.sum()

    def __repr__(self):
        return("{}(covariance={}, initial_scales={}, initial_nuggets={}, scales_rate={}, nuggets_rate={}, max_steps={}, scales={!r}, nuggets={!r}, save_train_history={!r})").format(self.__class__.__name__, self.covariance.__name__, self.initial_scales, self.initial_nuggets, self.scales_rate, self.nuggets_rate, self.max_steps, self.scales, self.nuggets, self.save_train_history)

    def Predict(self, X):
        kstar = self.covariance(X, self.X, self.scales)
        predictive_mean = np.matmul(np.transpose(kstar), self.alpha)
        v = solve_triangular(self.cholesky, kstar, lower=True)
        predictive_variance = self.covariance(X, X, self.scales) - np.matmul(np.transpose(v), v)
        return predictive_mean.reshape(-1,1), np.diag(predictive_variance).reshape(-1,1,1)
        
class MultEmulator(Transformer):

    def __init__(self, *args, scales=None, nuggets=None, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.scales = scales
        self.nuggets = nuggets

    def __repr__(self):
        output = ''
        if self.args is not None:
            output = ', '.join(self.args)
        if self.kwargs is not None:
            output += ', '.join(['{}={!r}'.format(k, v) for k, v in self.kwargs.items()])
        output += ', scales={!r}, nuggets={!r}'.format(self.scales, self.nuggets)
        return("{}({})").format(self.__class__.__name__, output)

    def Fit(self, X, Y):
        self.num_features = Y.shape[1]
        self.emulators = []
        Ys = np.hsplit(Y, self.num_features)

        need_training = True
        if self.scales is not None and self.nuggets is not None:
            assert self.scales.shape[0] == self.num_features, 'Number of rows in scales does not agree with number of features'
            assert self.scales.shape[1] == X.shape[1], 'Number of columns in scales does not agree with number of parameters'
            assert self.nuggets.shape[0] == self.num_features, 'Number of rows in nuggets does not agree with number of features'
            need_training = False

        for idx, Ysplit in enumerate(Ys):
            if need_training:
                nuggets = None
                scales = None
            else:
                nuggets = self.nuggets[idx]
                scales = self.scales[idx, :]

            emulator = Emulator(*self.args, **self.kwargs, scales=scales, nuggets=nuggets)
            emulator.Fit(X, Ysplit)
            self.emulators.append(emulator)

        """
        extract nuggets and scales from its list of emulators
        """
        if need_training:
            self.nuggets = np.array([emulator.nuggets for emulator in self.emulators])
            self.scales = np.vstack([emulator.scales.flatten() for emulator in self.emulators])


    def Predict(self, X):
        results = [emulator.Predict(X) for emulator in self.emulators]
        mean = np.concatenate([result[0] for result in results], axis=1)
        SD = np.concatenate([result[1].reshape(-1,1) for result in results], axis=1)
         
        # create multiple diagonal matrices from all samples
        cov = np.zeros((SD.shape[0], SD.shape[1], SD.shape[1]))
        diag = np.arange(SD.shape[1])
        cov[:, diag, diag] = SD

        return mean, cov
        
        
