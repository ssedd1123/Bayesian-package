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

if len(sys.argv) != 4:
    print('Use this script by entering: python %s Original_training new_prior output_name' % (sys.argv[0]))
    sys.exit()

with open(sys.argv[1], 'rb') as buff:
    data = pickle.load(buff)

emulator = data['emulator']
training_data = data['data']
training_data.ChangePrior(sys.argv[2])
scales = data['scales']
nuggets = data['nuggets']

print('Prior changed to %s' % sys.argv[2])

"""
Write all the training result, together with training points and pipe used to a file
"""
with open(os.path.join('training', '%s.pkl' % sys.argv[3]), 'wb') as buff:
    pickle.dump({'emulator': emulator, 'data': training_data,
                 'scales': scales, 'nuggets': nuggets}, buff)
