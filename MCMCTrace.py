import os
import random
import sys

import tables
from mpi4py import MPI

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
from multiprocessing import Pool, Process, Queue
from types import SimpleNamespace

if sys.version_info > (3, 0):
    import pickle
else:
    import cPickle as pickle

import argparse
import math
import shutil
import tempfile
from contextlib import contextmanager

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc
from numpy import array
from scipy.stats import multivariate_normal as mvn

from Preprocessor.PipeLine import *
from Utilities.MasterSlave import MasterSlave, ThreadsException
from Utilities.Utilities import GetTrainedEmulator, PlotTrace


def GenerateTrace(
    emulators, exp_Ys, exp_Yerrs, prior, id_, iter, output_filename, burnin=1000
):
    """
    The main function to generate pandas trace file after comparing the emulator with experimental value
    Uses pymc2 as it is found to be faster
    """
    pymc.numpy.random.seed(random.randint(0, 1000) + id_)
    parameters = []
    for name, row in prior.iterrows():
        if row["Type"] == "Uniform":
            parameters.append(
                pymc.Uniform(
                    name,
                    float(row["Min"]),
                    float(row["Max"]),
                    value=0.5 * (float(row["Min"]) + float(row["Max"])),
                )
            )
        else:
            parameters.append(
                pymc.TruncatedNormal(
                    name,
                    mu=float(row["Mean"]),
                    tau=1.0 / float(row["SD"]) ** 2,
                    a=float(row["Min"]),
                    b=float(row["Max"]),
                    value=float(row["Mean"]),
                )
            )
    
    for emulator, exp_Y, exp_Yerr in zip(emulators, exp_Ys, exp_Yerrs):
        exp_cov = np.diag(np.square(exp_Yerr))

        @pymc.stochastic(observed=True)
        def emulator_result(value=exp_Y, x=parameters, exp_cov=exp_cov, emulator=emulator):
            mean, var = emulator.Predict(np.array(x).reshape(1, -1))
            return np.array(mvn.logpdf(value, np.squeeze(mean), np.squeeze(var) + exp_cov))

    # model = pymc.Model(parameters)
    # prepare for MCMC
    new_output_filename = "%s_%d.h5" % (output_filename, id_)
    mcmc = pymc.MCMC(parameters, dbname=new_output_filename, db="hdf5", dbmode="w")

    # sample from our posterior distribution 50,000 times, but
    # throw the first 20,000 samples out to ensure that we're only
    # sampling from our steady-state posterior distribution
    mcmc.sample(iter, burn=burnin)
    mcmc.db.close()
    return new_output_filename  # pd.DataFrame.from_dict(trace_dict)


def MCMCParallel(config_file, dirpath=None, nevents=10000, burnin=1000):
    if not isinstance(config_file, list):
        config_file = [config_file]
    clf = []
    prior = []
    exp_Y = []
    exp_Yerr = []
    for file_ in config_file:
        args = GetTrainedEmulator(file_)
        clf.append(args[0])
        prior.append(args[1])
        exp_Y.append(args[2])
        exp_Yerr.append(args[3])

    """
    If config_file is a list of strings, then it means we want to chain up multiple emulators
    Check if they have the same prior. Can't work if they don't agree
    """
    #if not all([all(prior[0].columns == prior1.columns) for prior1 in prior]):
    #    raise RuntimeError("The variables list from all files are not consistent.")
    prior = prior[0] # since they all agree, only one prior is needed

    if dirpath is None:
        dirpath = tempfile.mkdtemp()
    result = GenerateTrace(
        clf,
        exp_Y,
        exp_Yerr,
        prior,
        MPI.COMM_WORLD.Get_rank(),
        nevents,
        os.path.join(dirpath, "temp"),
        burnin,
    )
    return result


def Merging(config_file, list_of_traces, clear_trace=False):
    chained_files = None
    if isinstance(config_file, list):
        # store list of config file as meta data
        chained_files = config_file[1:]
        config_file = config_file[0]
    with pd.HDFStore(config_file) as store:
        if clear_trace and "trace" in store:
            store.remove("trace")
        for f in list_of_traces:
            tab = tables.open_file(f)
            df = pd.DataFrame.from_records(tab.root.chain0.PyMCsamples.read())
            store.append(key="trace", value=df.astype(float))
            tab.close()
        prior = store["PriorAndConfig"]
         
        if chained_files is not None:
            #store.get_storer('trace').meta = SimpleNamespace()
            store.get_storer('trace').attrs.chained_files = chained_files
    
    return prior


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
status = MPI.Status()
root = 0

if __name__ == "__main__":

    work_environment = MasterSlave(comm)
    parser = argparse.ArgumentParser(
        description="This script will choose an optimal set of hyperparameters by minizing loss function"
    )
    parser.add_argument(
        "-i",
        "--inputs",
        nargs='+',
        help="Location of the trained parameter (output from Training.py)",
        required=True
    )
    parser.add_argument("-n", type=int, help="Number of events", required=True)

    # parser.add_argument('-p', '--plot', action='store_true', help='Use this if you want to plot posterior immediatly after trace generation')
    args = vars(parser.parse_args())

    # if rank == root:
    work_environment.Submit(MCMCParallel, **{"config_file": args['inputs'], "nevents": args['n']})
    refresh_rate = 0.3
    refresh_interval = refresh_rate * size
    # some stdio must be discarded for MPI network efficiency
    work_environment.RefreshRate(refresh_interval)

    try:
      while work_environment.IsRunning():
          out = work_environment.stdout[1]
          if out is not None:
              print(out, flush=True)
    except ThreadsException as ex:
      print(ex)

    prior = Merging(args["inputs"], work_environment.results, clear_trace=True)
    # shutil.rmtree(work_environment.results)

    PlotTrace(args["inputs"][0])
    plt.show()
    # work_environment.Close()
