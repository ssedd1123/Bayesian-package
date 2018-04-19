import os
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

from Emulator.Emulator import EmulatorMaster, RBF
from Preprocessor.PipeLine import PipeLine, Normalize, PCA
from DataReader.DataLoader import DataLoader

if len(sys.argv) != 5:
    print('Use this script by entering: python %s Prior ModelData ExpData Training_name' % (sys.argv[0]))
    sys.exit()


#trainning_x = np.arange(1, 4, 0.3).reshape(-1,1)

data = DataLoader(sys.argv[1], sys.argv[2], sys.argv[3])

"""
we need to normalized the observed points for better emulation
We need to normalize both the output and input space
for output space, PCA is also performed for dimension reduction
"""
output_pipe = PipeLine([('Normalize', Normalize()), ('PCA', PCA(4)), ('Normalized', Normalize())])
input_pipe = Normalize()

#scales, nuggets = EarlyStopping(input_=sim_para, target=sim_data, validation_runs=validation_run,
#                                initial_scales=np.ones(len(par_name)), initial_nuggets=np.array(0.2),
#                                input_pipe=pipe2, output_pipe=pipe)
emulator = EmulatorMaster(data.sim_para, data.sim_data, input_pipe, output_pipe)
#emulator.SetCovariance(RBF)
scales, nuggets = emulator.Train(np.ones(len(data.par_name)), max_step=1000, scales_rate=0.001, nuggets_rate=0.001)
#scales, nuggets = emulator.Train(1., max_step=1000, scales_rate=0.001, nuggets_rate=0.001)

"""
Write all the training result, together with training points and pipe used to a file
"""
with open(os.path.join('training', '%s.pkl' % sys.argv[4]), 'wb') as buff:
    pickle.dump({'emulator': emulator, 'data': data,
                 'scales': scales, 'nuggets': nuggets}, buff)
