import pymc3 as pm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import theano
import theano.tensor as tt
from theano.compile.ops import as_op
import numpy.ma as ma

from Emulator import *
from PipeLine import *
from Convergency_check import PlotMarginalLikelihood

#trainning_x = np.arange(1, 4, 0.3).reshape(-1,1)

"""
Loading prior of each variables
will be used to load the parameter names
such that when model data is read
it can tell which one is input parameter and which one is output 
"""
# load the prior
prior = pd.read_csv('parameter_priors.csv')
par_name = list(prior)

"""
Loading model simulation data
"""
# read the model data
df = pd.read_csv('e120_model.csv')
df = df[df.columns.drop(list(df.filter(regex='_Error')))]
sim_data = df.drop(par_name, axis=1).as_matrix()
sim_para = df[par_name].as_matrix()

"""
we need to normalized the observed points for better emulation
We need to normalize both the output and input space
for output space, PCA is also performed for dimension reduction
"""
pipe = PipeLine([('Normalize', Normalize()), ('PCA', PCA(3)), ('Normalized', Normalize())])
pipe.Fit(sim_data)
pipe2 = Normalize()
pipe2.Fit(sim_para)

# setting up emulator for training
emulator = EmulatorMultiOutput(pipe2.Transform(sim_para), pipe.Transform(sim_data))
emulator.SetCovariance(squared_exponential)
emulator.Train(np.array([0.5, 0.5, 0.5, 0.5]), 0.2, scales_rate=0.01, nuggets_rate=0.01, max_step=500)

scales = emulator.GetScales()
nuggets = emulator.GetNuggets()

print('')
print(scales, nuggets)

num_output = sim_data.shape[0]
output_list = []
for i in xrange(num_output):
    y = np.delete(sim_data, i, 0)
    x = np.delete(sim_para, i, 0)
    
    pipe.Fit(y)
    pipe2.Fit(x)
    emulator = EmulatorMultiOutput(pipe2.Transform(x), pipe.Transform(y))
    emulator.SetScales(scales)
    emulator.SetNuggets(nuggets)
    emulator.StartUp()
    (mean, var) = emulator.Emulate(pipe2.Transform(sim_para[i,:].reshape(1,-1)))
    mean = pipe.TransformInv(mean.T)
    output_list.append(mean[0,:])

print(sim_data)
print(np.array(output_list))
plt.plot(sim_data.flatten(), np.array(output_list).flatten(), 'ro')
plt.show()
