import pymc3 as pm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import theano
import theano.tensor as tt
from theano.compile.ops import as_op

from EmulatorTheano2 import *
from PipeLine import *
from Convergency_check import PlotMarginalLikelihood

#trainning_x = np.arange(1, 4, 0.3).reshape(-1,1)

"""
testing with model
the model is simulation with
v=0.5*a*x^b + uniform_err(-0.05,0.05)
experimental data is v=0.5*x^2
"""

x = np.linspace(0,4,15).reshape(-1,1)
y = (np.exp(-x)*np.sin(2*x)) + 0.1*(np.random.rand(*x.shape) - 0.5)
observed = np.array([0.3]).reshape(-1,1)

emulator = EmulatorMultiOutputT(x, y)
#emulator.Descent(0.5, 0.1, 200, tolerance=1e-3)
emulator.Train(0.5, 0.1, scales_rate=0.001, nuggets_rate=0.01, max_step=200)

cov = np.array([0]).reshape(-1,1)

emu = []
x = np.linspace(0,4,50).reshape(-1,1)
for val in x:
    emu.append(emulator.Emulate(val.reshape(-1,1))[0])
emu = np.array(emu).reshape(-1,1)
print(x, emu)
plt.plot(x, emu)
plt.show()


class Beta(pm.Continuous):
    def __init__(self, x, *args, **kwargs):
        super(Beta, self).__init__(*args, **kwargs)
        self.x = x

    def logp(self, value):
        x = tt.stack(self.x)
        #value = pipe.Transform(value)
        return emulator._LogLikelihood(x.dimshuffle('x', 0), value.dimshuffle('x', 0), cov)



with pm.Model() as model:
    a = pm.Uniform('a', 0, 3)
    #b = pm.Uniform('b', 0.5, 3)
    

    emulator_result = Beta('emulator', x=a, observed=observed)
    step = pm.Metropolis()
    trace = pm.sample(10000, tune=1000, step=step, init=None, njobs=5)
    #step = pm.Metropolis()

    # plot the result in a nice matrix of histograms
    plt.hist(trace['a'], bins = 100)
    plt.show()
