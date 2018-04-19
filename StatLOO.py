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
from DataReader.DataLoader import DataLoader
from Utilities.Utilities import *

if len(sys.argv) != 4:
    print('Use this script by entering: python %s Training_file validation_run result_name' % (sys.argv[0]))
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

validation_run = int(sys.argv[2])
sim_data = training_data.sim_data
sim_para = training_data.sim_para

sim_data = np.delete(sim_data, validation_run, 0)
sim_para = np.delete(sim_para, validation_run, 0)

training_data.exp_result = training_data.sim_data[validation_run]
training_data.exp_cov = np.diag(np.square(training_data.sim_error[validation_run].flatten()))
exp_para = training_data.sim_para[validation_run]
training_data.sim_data = sim_data
training_data.sim_para = sim_para


emulator.ResetData(sim_para, sim_data)
#emulator.SetCovariance(squared_exponential)
emulator.SetScales(scales)
emulator.SetNuggets(nuggets)
emulator.StartUp()

prior=training_data.prior

print(training_data.exp_result, training_data.exp_cov)

processes=7
pool = Pool(processes=processes)

result = []
for i in xrange(processes):
    result.append(pool.apply_async(GenerateTrace, (emulator, training_data.exp_result, training_data.exp_cov, prior, 100*i)))
trace = [r.get() for r in result]
pool.close()
pool.join()

trace = pd.concat(trace, ignore_index=True)
trace.to_csv('%s.csv' % sys.argv[3], sep='\t')

#trace = GenerateTrace(emulator, input_pipe=pipe2, output_pipe=pipe, exp_result=training_data.exp_result, exp_cov=training_data.exp_cov, prior=training_data.prior)
par_name = training_data.par_name
fig, axes2d = PlotTrace(trace, par_name, prior)

for i, row in enumerate(axes2d):
    for j, cell in enumerate(row):
        if i == j:
            cell.axvline(exp_para[i], color='r')
        else:
            cell.plot(exp_para[j], exp_para[i], 'white', marker='v', markersize=10)

plt.show()

with open('result/%s.pkl' % sys.argv[3], 'wb') as buff:
    pickle.dump({'model': emulator, 'trace': trace, \
                 'real_para': exp_para, 'prior': prior, \
                 'data': data}, buff)
