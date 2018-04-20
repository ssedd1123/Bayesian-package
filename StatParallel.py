from multiprocessing import Pool
import cPickle as pickle
import pandas as pd
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import math

from Emulator.Emulator import *
from Preprocessor.PipeLine import *
from Utilities.Utilities import PlotTrace, GenerateTrace

if len(sys.argv) != 3:
    print('Use this script by entering: python %s Training_file Output_name' % (sys.argv[0]))
    sys.exit()


"""
Use trained emulator
"""
with open(sys.argv[1], 'rb') as buff:
    data = pickle.load(buff)

emulator = data['emulator']
training_data = data['data']
scales = data['scales']
nuggets = data['nuggets']

emulator.SetCovariance(squared_exponential)
emulator.SetScales(scales)
emulator.SetNuggets(nuggets)
emulator.StartUp()

prior=training_data.prior


processes=14
pool = Pool(processes=processes)

result = []
for i in xrange(processes):
    result.append(pool.apply_async(GenerateTrace, (emulator, training_data.exp_result, training_data.exp_cov, prior, i)))
trace = [r.get() for r in result]
pool.close()
pool.join()

trace = pd.concat(trace, ignore_index=True)


#trace = GenerateTrace(emulator, input_pipe=pipe2, output_pipe=pipe, exp_result=training_data.exp_result, exp_cov=training_data.exp_cov, prior=training_data.prior)

PlotTrace(trace, training_data.par_name, prior)
plt.show()

trace.to_csv('result/%s.csv' % sys.argv[2], sep='\t')


with open('result/%s.pkl' % sys.argv[2], 'wb') as buff:
    pickle.dump({'model': emulator, 'trace': trace, \
                 'data': data, 'prior': prior}, buff)
