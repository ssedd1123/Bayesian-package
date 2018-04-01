import pymc3 as pm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import theano
import theano.tensor as tt
from theano.compile.ops import as_op

from EmulatorTheano2 import *
from PipeLineTheano import *

#trainning_x = np.arange(1, 4, 0.3).reshape(-1,1)

"""
testing with model
the model is simulation with
v=0.5*a*x^b + uniform_err(-0.05,0.05)
experimental data is v=0.5*x^2
"""

# load the prior
prior = pd.read_csv('Prior.csv')

par_name = list(prior)

df = pd.read_csv('ModelData.csv')
df = df[df.columns.drop(list(df.filter(regex='_Error')))]
sim_data = df.drop(par_name, axis=1).as_matrix()
sim_para = df[par_name].as_matrix()


df = pd.read_csv('ExpData.csv')
error = df[list(df.filter(regex='_Error'))].as_matrix().flatten()
exp_result = df[df.columns.drop(list(df.filter(regex='_Error')))].as_matrix().flatten()

# switch everything to theano array
sim_data = theano.shared(sim_data)
sim_para = theano.shared(sim_para)
exp_result = theano.shared(exp_result)

# we need to normalized the observed points for better emulation
pipe = PipeLineT([('Normalize', NormalizeT()), ('PCA', PCAT(3)), ('Normalized', NormalizeT())])
pipe.Fit(sim_data)
sim_emulate = pipe.Transform(sim_data)

pipe_input = NormalizeT()
pipe_input.Fit(sim_para)
para_emulate = pipe_input.Transform(sim_para)

get_sim_emulate = theano.function([], sim_emulate)
get_para_emulate = theano.function([], para_emulate)
get_mean = theano.function([], pipe_input.mean)
# setting up emulator for training
emulator = EmulatorMultiOutputT(get_para_emulate(), get_sim_emulate())

emulator.Train(0.5, 0.1, scales_rate=0.0001, nuggets_rate=0.0001, max_step=1000)

print('mean', get_mean())


class Beta(pm.Continuous):
    def __init__(self, x, *args, **kwargs):
        super(Beta, self).__init__(*args, **kwargs)
        self.x = x

    def logp(self, value):
        x = pipe_input.Transform(tt.stack(self.x).dimshuffle('x', 0))
        value = pipe.Transform(value.dimshuffle('x', 0))
        return emulator._LogLikelihood(x, value, cov)


with pm.Model() as model:
    pipe = PipeLineT([('Normalize', NormalizeT()), ('PCA', PCAT(3)), ('Normalized', NormalizeT())])
    pipe.Fit(sim_data)
    sim_emulate = pipe.Transform(sim_data)

    cov = pipe.TransformCov(theano.shared(np.diag(error)))
    
    pipe_input = NormalizeT()
    pipe_input.Fit(sim_para)

    a = []
    for column in prior:
        a.append(pm.Uniform(column, prior[column][0], prior[column][1]))
    #b = pm.Uniform('b', 0.5, 3)
    

    emulator_result = Beta('emulator', x=a, observed=exp_result)
    #step = pm.Metropolis()
    trace = pm.sample(15000, init='advi', njobs=2)

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
