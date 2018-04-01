import pymc3 as pm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import theano
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

# load the prior
prior = pd.read_csv('parameter_priors.csv')

par_name = list(prior)

df = pd.read_csv('e120_model.csv')
df = df[df.columns.drop(list(df.filter(regex='_Error')))]
sim_data = df.drop(par_name, axis=1).as_matrix()
sim_para = df[par_name].as_matrix()

print(sim_para, sim_data, par_name, prior)

"""
A, B = 1.5, 2
err = 0.25
exp_result = np.array([0.5*A*math.pow(2, B), 0.5*A*math.pow(4,B), 0.5*A*math.pow(5,B)]) \
             + (np.random.rand(3) - 0.5)*err
"""
df = pd.read_csv('e120_exp_result.csv')
error = df[list(df.filter(regex='_Error'))].as_matrix().flatten()
exp_result = df[df.columns.drop(list(df.filter(regex='_Error')))].as_matrix().flatten()
# we need to normalized the observed points for better emulation
pipe = PipeLine([('Normalize', Normalize()), ('PCA', PCA(3)), ('Normalized', Normalize())])
pipe.Fit(sim_data)
pipe2 = Normalize()
pipe2.Fit(sim_para)

print(error, exp_result)

sim_emulate = pipe.Transform(sim_data)

# setting up emulator for training
emulator = EmulatorMultiOutput(pipe2.Transform(sim_para), sim_emulate)
emulator.SetCovariance(squared_exponential)

print(sim_para, sim_emulate)

"""
initial_scales=0.5, initial_nuggets=1, 
              scales_rate=0.001, nuggets_rate=0.01, max_step = 300):
"""
#def PlotMarginalLikelihood(emulator_function, scales_min=1e-2, scales_max=2, scales_num=40, nuggets_min=1e-2, nuggets_max=20, nuggets_num=40):
#PlotMarginalLikelihood(emulator.emulator_list[0].MarginalLikelihood, nuggets_min=1e-4, nuggets_max=100, scales_max=50)
emulator.Train(0.5, 0.1, scales_rate=0.001, nuggets_rate=0.001, max_step=1000)

cov = pipe.TransformCov(np.diag(error))
print(cov)


class Beta(pm.Continuous):
    def __init__(self, x, *args, **kwargs):
        super(Beta, self).__init__(*args, **kwargs)
        self.x = x

    def logp(self, value):
        x = tt.stack(self.x)
        return my_logp(x, value)

@as_op(itypes=[tt.dvector, tt.dvector], otypes=[tt.dscalar])
def my_logp(x, value):
    value = pipe.Transform(value)
    return emulator.LogLikelihood(pipe2.Transform(np.array(x)).reshape(1, -1), value, cov)


with pm.Model() as model:
    a = []
    for column in prior:
        a.append(pm.Uniform(column, prior[column][0], prior[column][1]))
    #b = pm.Uniform('b', 0.5, 3)
    

    emulator_result = Beta('emulator', x=a, observed=exp_result)
    step = pm.Metropolis()
    trace = pm.sample(15000, step=step, njobs=20)

    # plot the result in a nice matrix of histograms
    num_par = len(par_name)
    graph_num = 1
    for namex in par_name:
        for namey in par_name:
            plt.subplot(num_par, num_par, graph_num)
            graph_num = graph_num + 1
            if namex == namey:
                plt.hist(trace[namex], bins = 100)
            else:
                plt.hist2d(trace[namey], trace[namex], bins=100)
    plt.show()
