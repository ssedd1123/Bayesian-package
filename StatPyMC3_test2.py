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
from Utilities import *
from EarlyStoppingFunc import *

if len(sys.argv) != 4:
    print('Use this script by entering: python %s Prior ModelData ValidationRuns' % (sys.argv[0]))
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
df_no_error = df[df.columns.drop(list(df.filter(regex='_Error')))]
sim_data = df_no_error.drop(par_name, axis=1).as_matrix()
sim_para = df_no_error[par_name].as_matrix()

"""
Leave data out for validation and set it as "experiment result"
"""
validation_run = np.fromstring(sys.argv[3], dtype=np.int32, sep=',')
error = df[list(df.filter(regex='_Error'))].as_matrix()[validation_run.tolist()]
validation_result = sim_data[validation_run.tolist()]
validation_para = sim_para[validation_run.tolist()]

"""
delete exp data for validation
"""
sim_data = np.delete(sim_data, validation_run, 0)
sim_para = np.delete(sim_para, validation_run, 0)


"""
we need to normalized the observed points for better emulation
We need to normalize both the output and input space
for output space, PCA is also performed for dimension reduction
"""
pipe = PipeLine([('Normalize', Normalize()), ('PCA', PCA(6)), ('Normalized', Normalize())])
pipe2 = Normalize()

# form covariance matrix from the experimental error
# will be added to variance from gaussian emulation prediction
#cov = np.diag([0,0,0,0])

# setting up emulator for training


pipe.Fit(sim_data)
pipe2.Fit(sim_para)
scales = np.load('Scales.npy')
nuggets = np.load('Nuggets.npy')

emulator = EmulatorMultiOutput(pipe2.Transform(sim_para), pipe.Transform(sim_data))
emulator.SetCovariance(squared_exponential)
#emulator.Train(np.full(len(par_name), 0.3), 0.2, scales_rate=0.005, nuggets_rate=0.01, max_step=1000)
emulator.SetScales(scales)
emulator.SetNuggets(nuggets)
emulator.StartUp()

fig = plt.figure(1)
plot = fig.add_subplot(111)

for para, valid_result, valid_err in zip(validation_para, validation_result, error):
    para = pipe2.Transform(para)
    result, cov = emulator.Emulate(para.reshape(1, -1))
    result = pipe.TransformInv(result.flatten())
    cov = pipe.TransformCovInv(np.diag(cov.flatten()))

    index = np.arange(result.shape[0])
    prediction = plot.errorbar(index.flatten(), result.flatten(), yerr=np.sqrt(np.diag(cov)), fmt='--ro', label='Emulator Prediction', color='red')
    exact = plot.errorbar(index.flatten(), valid_result.flatten(), yerr=valid_err, fmt='--ro', label='Exact result', color='blue')
    plot.legend(loc='upper left', handles=[prediction, exact])
    plot.set_xlabel('Single/double ratio', fontsize=40)
    plot.tick_params(axis='both', which='major', labelsize=20)
    
    plt.show()
