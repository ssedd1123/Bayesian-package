from multiprocessing import Pool
import cPickle as pickle
import pandas as pd
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import math

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

pipe.Fit(sim_data)
pipe2.Fit(sim_para)

emulator = EmulatorMultiOutput(pipe2.Transform(sim_para), pipe.Transform(sim_data))
emulator.SetCovariance(squared_exponential)
emulator.SetScales(scales)
emulator.SetNuggets(nuggets)
emulator.StartUp()

"""
parameters = []
# form random variables according to prior 
for column in prior:
    parameters.append(pymc.Uniform(column, prior[column][0], prior[column][1]))
"""
 
# "model" the observed y values: again, I reiterate that PyMC treats y as
# evidence -- as fixed; it's going to use this as evidence in updating our belief
# about the "unobserved" parameters (b0, b1, and err), which are the
# things we're interested in inferring after all
 
# put everything we've modeled into a PyMC model


def running(prior, id_):
    # form random variables according to prior 
    import pymc
    pymc.numpy.random.seed(id_)
    parameters = []
    for column in prior:
        parameters.append(pymc.Uniform(column, prior[column][0], prior[column][1]))
    
    @pymc.stochastic(observed=True)
    def emulator_result(value=exp_result, x=parameters):
        mean, var = emulator.Emulate(pipe2.Transform(np.array(x)).reshape(1, -1))
        mean = pipe.TransformInv(mean.flatten())
        var = pipe.TransformCovInv(np.diag(var)) + cov
        return np.array(mvn.logpdf(value, mean, var))
    
    parameters.append(emulator_result)

    model = pymc.Model(parameters)
    
    # prepare for MCMC
    mcmc = pymc.MCMC(model)
     
    # sample from our posterior distribution 50,000 times, but
    # throw the first 20,000 samples out to ensure that we're only
    # sampling from our steady-state posterior distribution
    mcmc.sample(50000, 500)
    trace_dict = {}
    for column in prior:
        trace_dict[column] = mcmc.trace(column)[:]
    return pd.DataFrame.from_dict(trace_dict)

processes=14
pool = Pool(processes=processes)

result = []
for i in xrange(processes):
    result.append(pool.apply_async(running, (prior, i)))
trace = [r.get() for r in result]
pool.close()
pool.join()

trace = pd.concat(trace, ignore_index=True)

PlotTrace(trace, par_name, prior)
trace.to_csv('%s.csv' % sys.argv[4], sep='\t')


with open('%s.pkl' % sys.argv[4], 'wb') as buff:
    pickle.dump({'model': emulator, 'trace': trace, \
                 'input_pipe': pipe2, 'output_pipe': pipe, \
                 'exp_data': exp_result, 'exp_err': error, \
                 'prior': prior}, buff)
