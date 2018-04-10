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
    print('Use this script by entering: python %s Prior ModelData EarlyStoppingRuns' % (sys.argv[0]))
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
we need to normalized the observed points for better emulation
We need to normalize both the output and input space
for output space, PCA is also performed for dimension reduction
"""
pipe = PipeLine([('Normalize', Normalize()), ('PCA', PCA(6)), ('Normalized', Normalize())])
pipe2 = Normalize()

validation_run = np.fromstring(sys.argv[3], dtype=np.int32, sep=',')
scales, nuggets = EarlyStopping(input_=sim_para, target=sim_data, validation_runs=validation_run,
                                initial_scales=np.ones(len(par_name)), initial_nuggets=np.array(0.2),
                                input_pipe=pipe2, output_pipe=pipe)

""" 
Write down all scales and nuggets numbers to Config file
"""

np.save('Scales', scales)
np.save('Nuggets', nuggets)
