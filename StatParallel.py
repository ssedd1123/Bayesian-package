import random
import sys
import os
from multiprocessing import Pool
if sys.version_info > (3, 0):
    import pickle
else:
    import cPickle as pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import argparse

from Emulator.Emulator import *
from Preprocessor.PipeLine import *
from Utilities.Utilities import PlotTrace, GenerateTrace

def StatParallel(args):
    """
    Load the trained emulator
    """
    with open(args['Training_file'], 'rb') as buff:
        data = pickle.load(buff)
    
    emulator = data['emulator']
    training_data = data['data']
    prior=training_data.prior

    
    """
    Trace generation with multiple cores
    """
    processes=args['cores']
    pool = Pool(args['cores'])
    result = []
    for i in xrange(processes):
        result.append(pool.apply_async(GenerateTrace, (emulator, training_data.exp_result, training_data.exp_cov, prior, random.randint(0, 1000) + i, args['steps'])))
    trace = [r.get() for r in result]
    pool.close()
    pool.join()
    trace = pd.concat(trace, ignore_index=True)
    
    """
    If trace is inside the file, we will concatenate the trace
    """
    if 'trace' in data and 'concat' in args:
        trace = pd.concat([trace, data['trace']], ignore_index=True)

    """
    Save everything
    """
    if args['Output_name']:
        trace.to_csv(os.path.join(os.getcwd(), '%s.txt' % args['Output_name']), sep=' ')

    data['trace'] = trace
    with open(args['Training_file'], 'wb') as buff:
        pickle.dump(data, buff)

    return trace, training_data.par_name, prior


if __name__=="__main__":

    parser = argparse.ArgumentParser(description='This script will choose an optimal set of hyperparameters by minizing loss function')
    parser.add_argument('Training_file', help='Location of the trained parameter (output from Training.py)')
    parser.add_argument('Output_name', help='Name of the posterior trace')
    parser.add_argument('-c', '--cores', type=int, default=14, help='Number of cores used. (Default: 14)')
    parser.add_argument('-n', '--steps', type=int, default=10000, help='Number of iterations used in each core. (Default: 10000)')
    parser.add_argument('-p', '--plot', action='store_true', help='Use this if you want to plot posterior immediatly after trace generation')
    parser.add_argument('-co', '--concat', action='store_true', help='Use this if you want to plot posterior immediatly after trace generation')
    args = vars(parser.parse_args())
    
    trace, par_name, prior = StatParallel(args)
    
    if args['plot']:
        PlotTrace(trace, par_name, prior)
        plt.show()
