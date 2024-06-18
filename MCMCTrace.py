from Utilities.Utilities import GetTrainedEmulator, PlotTrace
from Preprocessor.PipeLine import *
from scipy.stats import multivariate_normal as mvn
from numpy import array
import pymc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from contextlib import contextmanager
import tempfile
import shutil
import math
import argparse
from types import SimpleNamespace
import multiprocessing as mp
import os
import random
import sys

import tables

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

if sys.version_info > (3, 0):
    import pickle
else:
    import cPickle as pickle


def GenerateTrace(
        emulators,
        exp_Ys,
        exp_Yerrs,
        prior,
        id_,
        iter,
        output_filename,
        burnin=1000):
    """
    The main function to generate pandas trace file after comparing the emulator with experimental value
    Uses pymc2 as it is found to be faster
    """
    pymc.numpy.random.seed(random.randint(0, 1000) + id_)
    n_models = len(emulators)
    emulators_list = []
    id_to_model_names = []
    parameters = []
    pRanges = []
    for i, ename in enumerate(sorted(emulators.keys())):
      id_to_model_names.append(ename)
      emulators_list.append(emulators[ename])
      ind_parameters = []
      ind_pRanges = []
      for name, row in prior.iterrows():
          ind_pRanges.append(float(row['Max']) - float(row['Min']))
          if row["Type"] == "Uniform":
              ind_parameters.append(
                  pymc.Uniform(
                      name if i == 0 else '%s_%d' % (name, i),
                      float(row["Min"]),
                      float(row["Max"]),
                      value=0.5 * (float(row["Min"]) + float(row["Max"])),
                  )
              )
          else:
              ind_parameters.append(
                  pymc.TruncatedNormal(
                      name if i == 0 else '%s_%d' % (name, i),
                      mu=float(row["Mean"]),
                      tau=1.0 / float(row["SD"]) ** 2,
                      a=float(row["Min"]),
                      b=float(row["Max"]),
                      value=float(row["Mean"]),
                  )
              )
      parameters.append(ind_parameters)
      pRanges.append(ind_pRanges)

    # transpose emulator_list
    emulators_list = list(map(list, zip(*emulators_list)))

    if n_models == 1:
      model_choice = 0
    else:
      model_choice = pymc.DiscreteUniform('ModelChoice', lower=0, upper=n_models-1)
 
    for emu, exp_Y, exp_Yerr in zip(emulators_list, exp_Ys, exp_Yerrs):
        exp_cov = np.diag(np.square(exp_Yerr))

        @pymc.stochastic(observed=True)
        def emulator_result(
                value=exp_Y,
                x=parameters,
                exp_cov=exp_cov,
                emulator=emu,
                mc=model_choice):
            mean, var = emulator[mc].Predict(np.array(x[mc]).reshape(1, -1))
            return np.array(
                mvn.logpdf(
                    value,
                    np.squeeze(mean),
                    np.squeeze(var) +
                    exp_cov))

    # model = pymc.Model(parameters)
    # prepare for MCMC
    new_output_filename = "%s_%d.h5" % (output_filename, id_)
    mcmc = pymc.MCMC(
        parameters if model_choice == 0 else parameters + [model_choice],
        dbname=new_output_filename,
        db="hdf5",
        dbmode="w")
    
    for prs, ps in zip(pRanges, parameters):
        for pr, p in zip(prs, ps):
            mcmc.use_step_method(pymc.AdaptiveMetropolis, p, shrink_if_necessary=True, cov=np.atleast_2d(np.square(0.5*pr)))
            #mcmc.use_step_method(pymc.Metropolis, p, proposal_sd=0.5*pr)


    # sample from our posterior distribution 50,000 times, but
    # throw the first 20,000 samples out to ensure that we're only
    # sampling from our steady-state posterior distribution
    mcmc.sample(iter, burn=burnin)
    mcmc.db.close()

    return new_output_filename, id_to_model_names  # pd.DataFrame.from_dict(trace_dict)


def MCMCParallel(config_file, dirpath=None, nevents=10000, burnin=1000, model_comp=False):
    if isinstance(config_file, str):
        config_file = [config_file]

    prior_filename = config_file[0]
    if model_comp:
        config_file = GroupConfigFiles(config_file)
    else:
        config_file = {None: config_file}

    clf = {}
    prior = None
    exp_Y = []
    exp_Yerr = []
    first_model = True
    for name, files in config_file.items():
        clf[name] = []
        for file_ in files:
            args = GetTrainedEmulator(file_)
            clf[name].append(args[0])
            if first_model:
                exp_Y.append(args[2])
                exp_Yerr.append(args[3])
            if file_ == prior_filename:
                prior = args[1]
        first_model = False

    """
    If config_file is a list of strings, then it means we want to chain up multiple emulators
    Check if they have the same prior. Can't work if they don't agree
    """
    # if not all([all(prior[0].columns == prior1.columns) for prior1 in prior]):
    #    raise RuntimeError("The variables list from all files are not consistent.")

    if dirpath is None:
        dirpath = tempfile.mkdtemp()
    result = GenerateTrace(
        clf,
        exp_Y,
        exp_Yerr,
        prior,
        mp.current_process().pid,   
        nevents,
        os.path.join(dirpath, "temp"),
        burnin,
    )
    return result


def GroupConfigFiles(config_files):
    files_by_model = {}
    for filename in config_files:
        with pd.HDFStore(filename, 'r') as store:
            ynames = tuple(store['Exp_Y'].index)
            attr = store.get_storer("PriorAndConfig").attrs.my_attribute
            if 'name' not in attr:
                raise RuntimeError('Model name is not specified in file ' + filename)
            model_name = attr['name']
            if model_name in files_by_model:
                files_by_model[model_name].append((ynames, filename))
            else:
                files_by_model[model_name] = [(ynames, filename)]

    obsToId = {}
    first_filenames = []
    first_modelname = None
    first = True
    for name in files_by_model.keys():
        res = []
        if first:
            for idx, (obs_names, filename) in enumerate(files_by_model[name]):
                obsToId[obs_names] = idx
                res.append(filename)
            first_modelname = name
            first_filenames = res
        else:
            res = ['']*len(obsToId)
            for obs_names, filename in files_by_model[name]:
                try:
                    res[obsToId[obs_names]] = filename
                except Exception as e:
                    raise RuntimeError('Files that corresponds to %s of model %s is missing from model %s' % (filename, name, first_modelname))
            if len(files_by_model[name]) != len(res):
                missing_file = first_filenames[res.index('')]
                raise RuntimeError('Files that corresponds to %s of model %s is missing from model %s' % (missing_file, first_modelname, name))
        files_by_model[name] = res
        first = False
    return files_by_model 


def Merging(config_file, list_of_traces, clear_trace=False):
    chained_files = None
    if isinstance(config_file, list):
        # store list of config file as meta data
        chained_files = config_file[1:]
        config_file = config_file[0]
    with pd.HDFStore(config_file) as store:
        id_to_model = []
        if clear_trace and "trace" in store:
            store.remove("trace")
        for f in list_of_traces:
            if isinstance(f, tuple):
                id_to_model = f[1]
                f = f[0]
            tab = tables.open_file(f)
            df = pd.DataFrame.from_records(tab.root.chain0.PyMCsamples.read())
            store.append(key="trace", value=df.astype(float))
            tab.close()
        prior = store["PriorAndConfig"]
 
        if len(id_to_model) > 0:
            store.get_storer('trace').attrs.model_names = id_to_model

        if chained_files is not None:
            #store.get_storer('trace').meta = SimpleNamespace()
            store.get_storer('trace').attrs.chained_files = chained_files

    return prior


if __name__ == "__main__":
    #from mpi4py import MPI
    from Utilities.MasterSlaveMP import MasterSlave, ThreadsException

    #comm = MPI.COMM_WORLD
    #size = comm.Get_size()
    size = 7

    work_environment = MasterSlave(None, ncores=size)#comm)
    parser = argparse.ArgumentParser(
        description="This script will choose an optimal set of hyperparameters by minizing loss function")
    parser.add_argument(
        "-i",
        "--inputs",
        nargs='+',
        help="Location of the trained parameter (output from Training.py)",
        required=True
    )
    parser.add_argument("-n", type=int, help="Number of events", required=True)
    parser.add_argument("-c", action='store_true', help="Enable model comparison")
    parser.add_argument('-p', '--plot', action='store_true', help='Use this if you want to plot posterior immediatly after trace generation')

    args = vars(parser.parse_args())

    #MCMCParallel(**{"config_file": args['inputs'], "nevents": args['n'], "model_comp": args['c']})
    work_environment.Submit(
        MCMCParallel, **{"config_file": args['inputs'], "nevents": args['n'], "model_comp": args['c']})
    refresh_rate = 0.2
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
    else:
        prior = Merging(args["inputs"], work_environment.results, clear_trace=True)
        # shutil.rmtree(work_environment.results)

        if args['plot']:
            PlotTrace(args["inputs"][0])
            plt.show()
        # work_environment.Close()
