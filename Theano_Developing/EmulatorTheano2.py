import os
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,floatX=float32,optimizer=None"

import matplotlib.pyplot as plt
import math
import numpy as np
from theano import tensor
import theano.tensor.slinalg as linalg
import theano.tensor.nlinalg as nlinalg
import theano


def square_cov(input_x, input_y, scale):
    dist = tensor.sum((input_x.dimshuffle('x', 0, 1) \
                  -input_y.dimshuffle(0, 'x', 1))**2, axis=2)
    return tensor.exp(-0.5*dist/(scale*scale))

class EmulatorMultiOutputT:


    def __init__(self, input_, target):
        self.num_output = target.shape[1]
        self.num_input = input_.shape[1]
        self.emulator_list = []
        self.x_star = tensor.dmatrix('test_input')
        for i in xrange(0, self.num_output):
            emulator = EmulatorT(input_, target[:, i].reshape(-1,1))
            self.emulator_list.append(emulator)

        """
        for i in xrange(2):
            self.emulator_list[i].SetScales(0.5)
            self.emulator_list[i].SetNuggets(0.5)
            self.emulator_list[i].StartUp()
        """
        value_var = tensor.dmatrix('data var')
        target_val = tensor.dmatrix('compare data')

        emu = self._Emulate(self.x_star)
        var = self._Var(self.x_star)
        self.emu = theano.function([self.x_star], emu)        
        self.var = theano.function([self.x_star], var)

        loglikelihood = self._LogLikelihood(self.x_star, target_val, value_var)
        self.LogLikeLihood = theano.function([self.x_star, target_val, value_var], loglikelihood)

    def Train(self, scales=0.5, nuggets=1, 
              scales_rate=0.001, nuggets_rate=0.01, max_step = 300):
        for emulator in self.emulator_list:
            emulator.Descent(scales, nuggets, nsteps=max_step, scales_rate=0.001, nuggets_rate=0.01)
            emulator.StartUp()

    def _Emulate(self, input_):
        mean_list = []
        for i in xrange(0, self.num_output):
            mean = self.emulator_list[i]._Emulate(input_)
            mean_list.append(mean)
        return tensor.concatenate(mean_list).T

    def _Var(self, input_):
        var_list = []
        for i in xrange(0, self.num_output):
            var = self.emulator_list[i]._Var(input_)
            var_list.append(var[0])
        var = tensor.concatenate(var_list, axis=0)
        return var

    def _LogLikelihood(self, input_, value, value_var):
        emu = self._Emulate(input_)
        var = self._Var(input_)
        
        tot_var = nlinalg.diag(var) + value_var
        diff = emu - value
        return  (-0.5 * (tensor.log(tensor.nlinalg.Det()(tot_var)) \
                + tensor.dot(diff, tensor.dot(tensor.nlinalg.matrix_inverse(tot_var), diff.T)) \
                + self.num_output*tensor.log(2*3.1415))).sum()

    def Emulate(self, input_):
        return self.emu(input_), self.var(input_)

    def LogLikelihood(self, input_, value, value_cov):
        return self.LogLikeLihood(input_, value, value_cov)
 

class EmulatorT:



    def __init__(self, input_, targets):
        self.y = theano.shared(targets)
        self.input_ = theano.shared(input_)
        self.scale = theano.shared(0.1)
        self.nuggets = theano.shared(0.1)
        self.L = theano.shared(np.zeros((targets.shape[0], targets.shape[0])))
        self.alpha = theano.shared(np.zeros(targets.shape))
        self.cov = square_cov
        self.x_star = tensor.dmatrix('test_input')

        # Shared variables calculation
        # Will be called each time StartUp is called
        cov = self.cov(self.input_, self.input_, self.scale)
        L = linalg.cholesky(cov + self.nuggets*tensor.identity_like(cov))
        alpha = linalg.solve_symmetric(L.T, linalg.solve_symmetric(L, self.y)) 
        self.update_L = theano.function([], L, updates=[(self.L, L)])
        self.update_alpha = theano.function([], alpha, updates=[(self.alpha, alpha)])

        # Mean and variance calculation routine
        emu = self._Emulate(self.x_star)
        var = self._Var(self.x_star) 

        # compile all the function
        self.predictive_mean = theano.function([self.x_star], emu)
        self.predictive_var = theano.function([self.x_star], var)

        like = ((- 0.5*tensor.dot(self.y.T, alpha)) \
                - tensor.log(nlinalg.diag(L)).sum() \
                - self.L.shape[0]*tensor.log(2*3.1415926)).sum()#self._MarginalLikelihood()
        self.marginal_likelihood = theano.function([], like)

        # gradient for marginal likelihood
        dLdscales = tensor.grad(like, self.scale)
        dLdnuggets = tensor.grad(like, self.nuggets)
        self.dLdscales = theano.function([], dLdscales)
        self.dLdnuggets = theano.function([], dLdnuggets)

        # LOOCrossValidation
        Kinv_diag = nlinalg.diag(nlinalg.matrix_inverse(cov + self.nuggets*tensor.identity_like(cov))).dimshuffle(0, 'x')
        LOO_mean_minus_target = alpha/Kinv_diag
        LOO_sigma = tensor.inv(Kinv_diag)
        self.log_cv = ( - 0.5*(tensor.log(LOO_sigma) + tensor.sqr(LOO_mean_minus_target)*Kinv_diag + tensor.log(2*3.1315926))).sum()
        self.loo_cross_validation = theano.function([], self.log_cv)

        
        self.dLOOdnuggets = tensor.grad(cost=self.log_cv, wrt=self.nuggets)
        
    def SetScales(self, scales):
        self.scale.set_value(scales)

    def SetNuggets(self, nuggets):
        self.nuggets.set_value(nuggets)

    def StartUp(self):
        self.update_L()
        self.update_alpha()
        
 
    def _Emulate(self, input_):
        k_star = square_cov(input_, self.input_, self.scale)
        f_star = tensor.dot(k_star.T, self.alpha)
        return f_star
 
    def _Var(self, input_):
        k_star = square_cov(input_, self.input_, self.scale)
        v = linalg.solve_symmetric(self.L, k_star)
        var = square_cov(input_, input_, self.scale) - tensor.dot(v.T, v)
        return var

    def Emulate(self, input_):
        return (self.predictive_mean(input_), self.predictive_var(input_))

    def MarginalLikelihood(self, scale, nuggets):
        self.SetScales(scale)
        self.SetNuggets(nuggets)
        return self.marginal_likelihood()

    def LOOCrossValidation(self, scale, nuggets):
        self.SetScales(scale)
        self.SetNuggets(nuggets)
        return self.loo_cross_validation()

    def StepDescent(self, scales, nuggets):
        self.scales_log = np.log(scales)
        self.nuggets_log = np.log(nuggets)
        self.par_log = np.array([self.scales_log, self.nuggets_log])
        gradient_ = np.array([self.grad_scales_exp(self.scales_log), self.grad_nuggets_exp(self.nuggets_log)])

        par_temp = self.par_log + gradient_*self.step_size

        return np.exp(par_temp), gradient_
        
    def Descent(self, scales, nuggets, nsteps=1000, scales_rate=1e-3, nuggets_rate=1e-3):
        history = []
        self.SetScales(scales)
        self.SetNuggets(nuggets)
        log_scale = theano.shared(math.log(self.scale.get_value()))
        log_nuggets = theano.shared(math.log(self.nuggets.get_value()))
        dLOOdscales = tensor.grad(cost=self.log_cv, wrt=self.scale)
        dLOOdnuggets = tensor.grad(cost=self.log_cv, wrt=self.nuggets)
        updates = [(log_scale, log_scale + dLOOdscales*scales_rate*self.scale),
                   (log_nuggets, log_nuggets + dLOOdnuggets*nuggets_rate*self.nuggets),
                   (self.scale, tensor.exp(log_scale)),
                   (self.nuggets, tensor.exp(log_nuggets))]
                   
        train = theano.function(inputs=[],
                                outputs=[self.log_cv],
                                updates=updates)
        for i in range(nsteps):
        
            (log_cv) = train()
            print(log_cv, self.scale.get_value(), self.nuggets.get_value())
            #if mag < tolerance:
                #break

        return np.array(history)

"""
x = np.linspace(0,4,15).reshape(-1,1)
y = (np.exp(-x)*np.cos(2*x)) + 0.5*(np.random.rand(*x.shape) - 0.5)
observed = np.array([0.5]).reshape(-1,1)

emulator = EmulatorMultiOutputT(x, y)
#emulator.Descent(0.5, 0.1, 200, tolerance=1e-3)
emulator.Train(0.5, 1, scales_rate=0.001, nuggets_rate=0.01, max_step=200)


points = np.linspace(0,3.14,num=10).reshape(-1,1)
targets = np.exp(-points)*np.sin(2*points) + 0.*(np.random.rand(*points.shape) - 0.5)

print(points, targets)
        
emu = EmulatorMultiOutputT(points, targets)
#emu2 = EmulatorMultiOutput(points, targets)

for i in xrange(2):
    emu.emulator_list[i].SetScales(2.26909751)
    emu.emulator_list[i].SetNuggets(7.5509196e-6)
    emu.emulator_list[i].StartUp()
    emu2.emulator_list[i].SetScales(2.26909751)
    emu2.emulator_list[i].SetNuggets(7.5509196e-6)
    emu2.emulator_list[i].StartUp()
emu.Train( scales=0.5, nuggets=1, 
           scales_rate=0.001, nuggets_rate=0.01, max_step = 300)

points = np.linspace(0,3.14,num=50).reshape(-1,1)
val_list = []
for val in points:
    val_list.append(emu.Emulate(val.reshape(1,1))[0])
print(points, np.array(val_list))
plt.plot(points.reshape(-1,1), np.array(val_list).reshape(-1,1))
plt.show()
PlotMarginalLikelihood(emulator.emulator_list[0].MarginalLikelihood, nuggets_min=1e-4, nuggets_max=1, nuggets_num=90,  scales_max=10, scales_num=90 )
for i in xrange(0,1000):
    print emu.Emulate([[0.01*i]])
"""
