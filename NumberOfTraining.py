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

if len(sys.argv) != 3:
    print('Use this script by entering: python %s Training_file validation_run' % (sys.argv[0]))
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

validation_run = np.fromstring(sys.argv[2], dtype=int, sep=',')
sim_data = training_data.sim_data
sim_para = training_data.sim_para

valid_data = training_data.sim_data[validation_run, :]
valid_para = training_data.sim_para[validation_run, :]

sim_data = np.delete(sim_data, validation_run, 0)
sim_para = np.delete(sim_para, validation_run, 0)

# see if the emulator's performance increase with number of training points
log_likelihood = []
percentage_diff_per_point = []
num_points = []
for i in xrange(2, len(sim_data)):
    emulator.ResetData(sim_para[:i, :], sim_data[:i, :])
    emulator.SetScales(scales)
    emulator.SetNuggets(nuggets)
    emulator.StartUp()

    tot_likelihood = 0
    percentage_error = []
    for par, var in zip(valid_para, valid_data):
        mean, cov = emulator.Emulate(par.reshape(1, -1))
        percentage_error += ((mean - var.reshape(1, -1))/mean).tolist()
        try:
            tot_likelihood += mvn.logpdf(var.reshape(1, -1), mean, cov)
        except:
            pass

    num_points.append(i)
    log_likelihood.append(tot_likelihood)
    percentage_diff_per_point.append(np.average(np.absolute(np.array(percentage_error))))

plt.plot(num_points, log_likelihood)
plt.show()
plt.plot(num_points, percentage_diff_per_point)
plt.xlabel('Num training points')
plt.ylabel('Percentage difference per observables')
plt.show()
