import cPickle as pickle
import pymc3 as pm
import autograd.numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import theano
import theano.tensor as tt
import sys
from theano.compile.ops import as_op

from Emulator import *
from PipeLine import *
from Convergency_check import PlotMarginalLikelihood

if len(sys.argv) != 4:
    print('Use this script by entering: python %s Prior ModelData ExpData' % (sys.argv[0]))
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
Loading model simulation data
"""
# read the model data
df = pd.read_csv(sys.argv[2])
# ignore the Error eolumn for the model
df = df[df.columns.drop(list(df.filter(regex='_Error')))]
# load the model output
sim_data = df.drop(par_name, axis=1).as_matrix()
# load the input space to which we obtain the output
sim_para = df[par_name].as_matrix()

"""
Loading experiment output data
"""
# rad the experiment result
df = pd.read_csv(sys.argv[3])
# load the experimental error
error = df[list(df.filter(regex='_Error'))].as_matrix().flatten()
exp_result = df[df.columns.drop(list(df.filter(regex='_Error')))].as_matrix().flatten()

"""
we need to normalized the observed points for better emulation
We need to normalize both the output and input space
for output space, PCA is also performed for dimension reduction
"""
pipe = PipeLine([('Normalize', Normalize()), ('PCA', PCA(3)), ('Normalized', Normalize())])
pipe.Fit(sim_data)
pipe2 = Normalize()
pipe2.Fit(sim_para)
# form covariance matrix from the experimental error
# will be added to variance from gaussian emulation prediction
cov = pipe.TransformCov(np.diag(error))

# setting up emulator for training
emulator = EmulatorMultiOutput(pipe2.Transform(sim_para), pipe.Transform(sim_data))
emulator.SetCovariance(squared_exponential)
emulator.Train(np.ones(len(par_name)), 0.2, scales_rate=0.01, nuggets_rate=0.01, max_step=500)


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
        value = pipe.Transform(value)
        return emulator.LogLikelihood(pipe2.Transform(np.array(x)).reshape(1, -1), value, cov)

    parameters = []
    # form random variables according to prior 
    for column in prior:
        parameters.append(pm.Uniform(column, prior[column][0], prior[column][1]))
    
    emulator_result = EmulatorLogLikelihood('emulator', x=parameters, observed=exp_result)
    step = pm.Metropolis()
    trace = pm.sample(2000, step=step, njobs=20)

    # plot the result in a nice matrix of histograms
    num_par = len(par_name)
    graph_num = 1
    fig, axes2d = plt.subplots(num_par, num_par) 

    for i, row in enumerate(axes2d):
        for j, cell in enumerate(row):
            namex = par_name[j]
            namey = par_name[i]
            if namex == namey:
                cell.hist(trace[namex], bins = 100)
            else:
                cell.hist2d(trace[namex], trace[namey], bins=100)
            if i == num_par - 1:
                cell.set_xlabel(namex)
            if j == 0:
                cell.set_ylabel(namey)
            
    df = pm.backends.tracetab.trace_to_dataframe(trace)
    df.to_csv('trace.csv', sep='\t')
    
    plt.show()

with open('trace.pkl', 'wb') as buff:
    pickle.dump({'model': emulator, 'trace': df, \
                 'input_pipe': pipe2, 'output_pipe': pipe, \
                 'exp_data': exp_result, 'exp_err': error, \
                 'prior': prior}, buff)
    
