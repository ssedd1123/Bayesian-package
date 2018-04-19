import cPickle as pickle
import pymc3 as pm
import autograd.numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import theano
import theano.tensor as tt
import sys
import os
from theano.compile.ops import as_op

from Emulator import *
from PipeLine import *
from Convergency_check import PlotMarginalLikelihood
from Utilities import PlotTrace

if len(sys.argv) != 5:
    print('Use this script by entering: python %s Prior Training_file ExpData Output_name' % (sys.argv[0]))
    sys.exit()


#trainning_x = np.arange(1, 4, 0.3).reshape(-1,1)

"""
Loading prior of each variables
will be used to load the parameter names
such that when model data is read
it can tell which one is input parameter and which one is output 
"""
# load the prior
prior = pd.read_csv(sys.argv[1])
# load the name of the variables in the prior
par_name = list(prior)

"""
Loading experiment output data
"""
# rad the experiment result
df = pd.read_csv(sys.argv[3])
# load the experimental error
error = df[list(df.filter(regex='_Error'))].as_matrix().flatten()
exp_result = df[df.columns.drop(list(df.filter(regex='_Error')))].as_matrix().flatten()
cov = np.square(np.diag(error))

"""
Use trained emulator
"""
with open(sys.argv[2], 'rb') as buff:
    data = pickle.load(buff)

pipe2 = data['input_pipe']
pipe = data['output_pipe']
sim_data = data['input_data']
sim_para = data['input_para']
scales = data['scales']
nuggets = data['nuggets']

"""
we need to normalized the observed points for better emulation
We need to normalize both the output and input space
for output space, PCA is also performed for dimension reduction
"""
pipe.Fit(sim_data)
pipe2.Fit(sim_para)

emulator = EmulatorMultiOutput(pipe2.Transform(sim_para), pipe.Transform(sim_data))
emulator.SetCovariance(squared_exponential)
emulator.SetScales(scales)
emulator.SetNuggets(nuggets)
emulator.StartUp()


model = pm.Model()

with model:

    """
    Interface for theano to talk to our emulator
    pymc3 uses theano for calculation
    this interface is necessary
    """
    class EmulatorLogLikelihood(pm.Continuous):
        def __init__(self, x, *args, **kwargs):
            super(EmulatorLogLikelihood, self).__init__(*args, **kwargs)
            self.x = x
    
        def logp(self, value):
            x = tt.stack(self.x)
            return my_logp(x, value)
    
    @as_op(itypes=[tt.dvector, tt.dvector], otypes=[tt.dscalar])
    def my_logp(x, value):
        mean, var = emulator.Emulate(pipe2.Transform(np.array(x)).reshape(1, -1))
        mean = pipe.TransformInv(mean.flatten())
        var = pipe.TransformCovInv(np.diag(var)) + cov
        return np.array(mvn.logpdf(value, mean, var))

    parameters = []
    # form random variables according to prior 
    for column in prior:
        parameters.append(pm.Uniform(column, prior[column][0], prior[column][1]))
    
    emulator_result = EmulatorLogLikelihood('emulator', x=parameters, observed=theano.shared(exp_result))
    step = pm.Metropolis()
    trace = pm.sample(2000, step=step, njobs=20)

    pm.traceplot(trace)

    # plot the result in a nice matrix of histograms
    num_par = len(par_name)
    graph_num = 1
    fig, axes2d = plt.subplots(num_par, num_par) 
 
    PlotTrace(trace, par_name, prior)
    df = pm.backends.tracetab.trace_to_dataframe(trace)
    df.to_csv('%s.csv' % sys.argv[4], sep='\t')


with open('%s.pkl' % sys.argv[4], 'wb') as buff:
    pickle.dump({'model': emulator, 'trace': df, \
                 'input_pipe': pipe2, 'output_pipe': pipe, \
                 'exp_data': exp_result, 'exp_err': error, \
                 'prior': prior}, buff)
    
