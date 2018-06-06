from multiprocessing import Pool
import cPickle as pickle
import pandas as pd
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import math
import argparse

from Emulator.Emulator import *
from Preprocessor.PipeLine import *
from DataReader.DataLoader import DataLoader
from Utilities.Utilities import *

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='This script will choose an optimal set of hyperparameters by minizing loss function')
    parser.add_argument('Training_file', help='Location of the trained parameter (output from Training.py)')
    parser.add_argument('ValidationRun', type=int, help='Runs to be taken out for validation')
    parser.add_argument('Output_name', help='Name of the posterior trace')
    parser.add_argument('-c', '--cores', type=int, default=14, help='Number of cores used. (Default: 14)')
    parser.add_argument('-n', '--steps', type=int, default=10000, help='Number of iterations used in each core. (Default: 10000)')
    parser.add_argument('-p', '--plot', action='store_true', help='Use this if you want to plot posterior immediatly after trace generation')
    args = vars(parser.parse_args())
    
    
    """
    Use trained emulator
    """
    with open(args['Training_file'], 'rb') as buff:
        data = pickle.load(buff)
    
    emulator = data['emulator']
    training_data = data['data']
    scales = data['scales']
    nuggets = data['nuggets']
    prior=training_data.prior
    
    
    """
    Remove the validation set from training data
    """
    validation_run = args['ValidationRun']
    sim_data = training_data.sim_data
    sim_para = training_data.sim_para
    
    sim_data = np.delete(sim_data, validation_run, 0)
    sim_para = np.delete(sim_para, validation_run, 0)
    
    training_data.exp_result = training_data.sim_data[validation_run]
    training_data.exp_cov = np.diag(np.square(training_data.sim_error[validation_run].flatten()))
    exp_para = training_data.sim_para[validation_run]
    training_data.sim_data = sim_data
    training_data.sim_para = sim_para
    
    """
    Put the truncated training data back into the emulator
    """
    emulator.ResetData(sim_para, sim_data)
    #emulator.SetCovariance(squared_exponential)
    emulator.SetScales(scales)
    emulator.SetNuggets(nuggets)
    emulator.StartUp()
    
    
    processes=args['cores']
    pool = Pool(processes=processes)
    result = []
    for i in xrange(processes):
        result.append(pool.apply_async(GenerateTrace, (emulator, training_data.exp_result, training_data.exp_cov, prior, i, args['steps'])))
    trace = [r.get() for r in result]
    pool.close()
    pool.join()
    trace = pd.concat(trace, ignore_index=True)
    
    
    if args['plot']:
        par_name = training_data.par_name
        fig, axes2d = PlotTrace(trace, par_name, prior)
        """
        Plot the actual result on top for comparison
        """   
        for i, row in enumerate(axes2d):
            for j, cell in enumerate(row):
                if i == j:
                    cell.axvline(exp_para[i], color='r')
                else:
                    cell.plot(exp_para[j], exp_para[i], 'white', marker='v', markersize=10)
        
        plt.show()
    
    trace.to_csv('result/%s.csv' % args['Output_name'], sep='\t')
    with open('result/%s.pkl' % args['Output_name'], 'wb') as buff:
        pickle.dump({'model': emulator, 'trace': trace, \
                     'real_para': exp_para, 'prior': prior, \
                     'data': data}, buff)
