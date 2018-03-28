import pymc3 as pm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import theano.tensor as tt
from theano.compile.ops import as_op

from Emulator import *
from PipeLine import *
from Convergency_check import PlotMarginalLikelihood

#trainning_x = np.arange(1, 4, 0.3).reshape(-1,1)

"""
testing with model
the model is simulation with
v=0.5*a*x^b + uniform_err(-0.05,0.05)
experimental data is v=0.5*x^2
"""

df = pd.read_csv('ModelData.csv')
sim_data = df[['x2', 'x4', 'x6']].as_matrix()
sim_para = df[['A', 'B']].as_matrix()

A, B = 0.7, 2.3
exp_result = np.array([0.5*A*math.pow(2, B), 0.5*A*math.pow(4,B), 0.5*A*math.pow(5,B)])

# we need to normalized the observed points for better emulation
pipe = PipeLine([('Normalize', Normalize()), ('PCA', PCA(3))])
pipe.Fit(sim_data)

sim_emulate = pipe.Transform(sim_data)

# setting up emulator for training
emulator = EmulatorMultiOutput(sim_para, sim_emulate)
emulator.SetCovariance(squared_exponential)
emulator.Train()

class Beta(pm.Continuous):
    def __init__(self, x, y, *args, **kwargs):
        super(Beta, self).__init__(*args, **kwargs)
        self.x = x
        self.y = y

    def logp(self, value):
        x = self.x
        y = self.y
        return my_logp(x, y, value)


@as_op(itypes=[tt.dscalar, tt.dscalar, tt.dvector], otypes=[tt.dscalar])
def my_logp(x, y, value):
    value = pipe.Transform(value)
    return emulator.LogLikelihood(np.array([x, y]).reshape(1, -1), value)


with pm.Model() as model:
    a = pm.Uniform('a', 0.5, 2)
    b = pm.Uniform('b', 0.5, 3)
    

    emulator_result = Beta('emulator', x=a, y=b, observed=exp_result)
    step = pm.Metropolis()
    trace = pm.sample(5000, step=step, njobs=25)
    print(len(trace['a']))
    plt.hist(trace['a'], bins=500)
    plt.show()

    plt.hist2d(trace['a'], trace['b'], bins=40)
    plt.colorbar()
    plt.show()
