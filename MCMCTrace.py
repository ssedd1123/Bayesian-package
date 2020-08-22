import tables
from mpi4py import MPI
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
from numpy import array
import pandas as pd
import matplotlib.pyplot as plt
import math
import argparse
import tempfile
import shutil
import pymc
from scipy.stats import multivariate_normal as mvn
from contextlib import contextmanager

from Preprocessor.PipeLine import *
from Utilities.Utilities import PlotTrace, GetTrainedEmulator
from Utilities.MasterSlave import MasterSlave

def GenerateTrace(emulator, exp_Y, exp_Yerr, prior, id_, iter, output_filename, burnin=1000):
    """
    The main function to generate pandas trace file after comparing the emulator with experimental value
    Uses pymc2 as it is found to be faster
    """
    pymc.numpy.random.seed(random.randint(0, 1000) + id_)
    parameters = []
    for name, row in prior.iterrows():
        if row['Type'] == 'Uniform':
            parameters.append(pymc.Uniform(name, float(row['Min']), float(row['Max']), 
                                           value=0.5*(float(row['Min']) + float(row['Max']))))
        else:
            parameters.append(pymc.TruncatedNormal(name, mu=float(row['Mean']), tau=1./float(row['SD'])**2, 
                                                   a=float(row['Min']), b=float(row['Max']), value=float(row['Mean'])))

    exp_cov = np.diag(np.square(exp_Yerr))
    @pymc.stochastic(observed=True)
    def emulator_result(value=exp_Y, x=parameters):
        mean, var = emulator.Predict(np.array(x).reshape(1, -1))
        return np.array(mvn.logpdf(value, np.squeeze(mean), np.squeeze(var)+exp_cov))


    #model = pymc.Model(parameters)
    # prepare for MCMC
    new_output_filename = '%s_%d.h5' % (output_filename, id_)
    mcmc = pymc.MCMC(parameters, dbname=new_output_filename, db='hdf5', dbmode='w')
     
    # sample from our posterior distribution 50,000 times, but
    # throw the first 20,000 samples out to ensure that we're only
    # sampling from our steady-state posterior distribution
    mcmc.sample(iter, burn=burnin)
    mcmc.db.close()
    return new_output_filename#pd.DataFrame.from_dict(trace_dict)
    
 

def MCMCParallel(config_file, dirpath=None, nevents=10000, burnin=1000):
    args = GetTrainedEmulator(config_file)
    clf = args[0]
    prior = args[1]
    exp_Y = args[2]
    exp_Yerr = args[3]

    if dirpath is None:
        dirpath = tempfile.mkdtemp()
    result = GenerateTrace(clf, exp_Y, exp_Yerr, prior, MPI.COMM_WORLD.Get_rank(), nevents, os.path.join(dirpath, 'temp'), burnin)
    return result    

def Merging(config_file, list_of_traces, clear_trace=False):
    with pd.HDFStore(config_file) as store:
        if clear_trace and 'trace' in store:
            store.remove('trace')
        for f in list_of_traces:
            tab = tables.open_file(f)
            df = pd.DataFrame.from_records(tab.root.chain0.PyMCsamples.read())
            store.append(key='trace', value=df.astype(float))
            tab.close()
        prior = store['PriorAndConfig']
    return prior

    
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
status = MPI.Status()
root = 0

if __name__=="__main__":

    work_environment = MasterSlave(comm)
    parser = argparse.ArgumentParser(description='This script will choose an optimal set of hyperparameters by minizing loss function')
    parser.add_argument('config_file', help='Location of the trained parameter (output from Training.py)')
    parser.add_argument('nevents', type=int, help='Number of events')

    #parser.add_argument('-p', '--plot', action='store_true', help='Use this if you want to plot posterior immediatly after trace generation')
    args = vars(parser.parse_args())

    
    #if rank == root:
    work_environment.Submit(MCMCParallel, **args)
    refresh_rate = 0.3
    refresh_interval = refresh_rate*size
    # some stdio must be discarded for MPI network efficiency
    work_environment.RefreshRate(refresh_interval)

    while work_environment.IsRunning():
        out = work_environment.stdout[1]
        if out is not None:
            print(out, flush=True)

    prior = Merging(args['config_file'], work_environment.results)
    #shutil.rmtree(work_environment.results) 
    
    PlotTrace(args['config_file'])
    plt.show()
    #work_environment.Close()
