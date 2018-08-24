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

def RunValidation(validation_run, emulator, training_data, scales, nuggets):
    sim_data = training_data.sim_data
    sim_para = training_data.sim_para
    
    valid_data = training_data.sim_data[validation_run, :]
    valid_para = training_data.sim_para[validation_run, :]
    
    sim_data = np.delete(sim_data, validation_run, 0)
    sim_para = np.delete(sim_para, validation_run, 0)

    # shuffle before use for the case of regular lattice
    idx = np.arange(len(sim_data))
    np.random.shuffle(idx)
 
    sim_data = sim_data[idx]
    sim_para = sim_para[idx]
    
    # see if the emulator's performance increase with number of training points
    absolute_diff_per_point = []
    num_points = []
    for i in range(5, len(sim_data)):
        num_points.append(i)
        try:
            emulator.ResetData(sim_para[:i, :], sim_data[:i, :])
            emulator.SetScales(scales)
            emulator.SetNuggets(nuggets)
            emulator.StartUp()
        except:
            absolute_diff_per_point.append(np.zeros_like(valid_data))
            continue

        tot_likelihood = 0
        absolute_error = []
        for par, var in zip(valid_para, valid_data):
            mean, cov = emulator.Emulate(par.reshape(1, -1))
            absolute_error += np.square((mean - var.reshape(1, -1))).tolist()
    
        absolute_diff_per_point.append(np.sqrt(np.average(np.array(absolute_error))))

    return num_points, absolute_diff_per_point

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Use this script by entering: python %s Training_file' % (sys.argv[0]))
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
    
    
    # randomly choose 5 points as validation
    # different points will be chosen for averaging
    num_data = len(training_data.sim_data)
    abs_list = []
    for i in range(0, 10):
        # generate non-repeating random numbers
        a = np.arange(num_data)
        np.random.shuffle(a)
        _, absolute_diff_per_point = RunValidation(a[:5], emulator, training_data, scales, nuggets)
        abs_list.append(absolute_diff_per_point)
    rmse = np.average(np.array(abs_list), axis=0)
    print(' '.join(map(str, rmse)))
    plt.plot(rmse)
    plt.xlabel('Num training points')
    plt.ylabel('Absolute difference of observables')
    plt.show()
