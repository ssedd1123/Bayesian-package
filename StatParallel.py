import random
import sys
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
from multiprocessing import Pool, Process, Queue
if sys.version_info > (3, 0):
    import pickle
else:
    import cPickle as pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import argparse
import tempfile
import shutil

from Emulator.Emulator import *
from Preprocessor.PipeLine import *
from Utilities.Utilities import PlotTrace, GenerateTrace


class TraceCreator:

    def __init__(self, args):
        with open(args['Training_file'], 'rb') as buff:
            data = pickle.load(buff)
        self.emulator = data['emulator']
        self.training_data = data['data']
        self.prior = self.training_data.prior
        self.processes = args['cores']
        self.steps = args['steps']
        self.data = data

    def CreateTrace(self, id_, dir, queue=None):
        trace = GenerateTrace(self.emulator, self.training_data.exp_result, 
                             self.training_data.exp_cov, self.prior, 
                             id_, self.steps, dir)
        if queue:
            queue.put(trace)
        return trace #self.output_name


def MergeTrace(list_of_traces, args, data):
    #trace = pd.concat(list_of_traces, ignore_index=True)

    """
    Save everything
    """
    if args['Output_name']:
        output_name = args['Output_name']
    else:
        output_name = '%s.h5' % args['Training_file']



    """
    If trace is inside the file, we will concatenate the trace
    """
    if 'concat' not in args:
        os.remove(output_name)
        
    #if 'trace' in data and 'concat' in args:
    #    trace = pd.concat([trace, data['trace']], ignore_index=True)
    store = pd.HDFStore(output_name, complevel=5, comblib='blosc')
    for f in list_of_traces:
        df = pd.read_hdf(f)
        store.append(key='trace', value=df, format='t')

    #with open(args['Training_file'], 'wb') as buff:
    #    pickle.dump(data, buff)
    return output_name

    
def StatParallel(args):
    trace = TraceCreator(args)
    result = []
    process = []
    dirpath = tempfile.mkdtemp()
    for i in range(args['cores']):
        queue = Queue()
        try:
            process.append(Process(target=trace.CreateTrace, args=(i, os.path.join(dirpath, 'temp'), queue)))
            process[-1].start()
        except Exception:
            process.pop()
            print('Cannot create process %d' % i)
            break
        result.append(queue)
        
    result = [r.get() for r in result]
    process = [p.join() for p in process]
    #print(result[0])
    try: 
        result = MergeTrace(result, args, trace.data)
    finally:
        shutil.rmtree(dirpath)
    return result, trace.training_data.par_name, trace.prior


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
        PlotTrace(df.read_hdf(trace), par_name, prior)
        plt.show()
