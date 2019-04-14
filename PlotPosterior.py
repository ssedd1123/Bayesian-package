import profile
from numpy import array
from multiprocessing import Pool
import glob
import matplotlib as mpl
import scipy
import pickle as pickle
import pandas as pd
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import math
import random
import time

from Preprocessor.PipeLine import *
from Utilities.Utilities import GetTrainedEmulator

one_sigma_confidence = 0.6827
two_sigma_confidence = 0.9545

def PlotOutput(filename, fig):

    """
    Function to plot both the posterior and prior point
    if prior is chosen, it will chose points at random
    otherwise it will load configuration from trace
    """
    clf, prior, exp_Y, exp_Yerr, model_X, model_Y, training_idx, _ = GetTrainedEmulator(filename)
    store = pd.HDFStore(filename, 'r')
    trace = store['trace']
    store.close()

    section_size = 1000
    dfs = np.array_split(trace, trace.shape[0]/section_size, axis=0)
    num_obs = model_Y.shape[1]
    prior_predictions = []
    posterior_predictions = []
    start = time.time()
    for index, df in enumerate(dfs):
        par = np.random.uniform(low=prior['Min'], high=prior['Max'], size=(df.shape[0], prior.shape[0]))
        # transform input by input_pipe and put it in our emulator
        result, _ = clf.Predict(par)
        prior_predictions.append(result)

        par = df[prior.index.tolist()].values
        result, _ = clf.Predict(par)
        posterior_predictions.append(result)
        
        print('time used %f' % (time.time() - start), flush=True)
        start = time.time()
        print('Processing %d' % index, flush=True)
        if index > 10:
            break
    
    # plot the result
    ax = fig.subplots(1, 1)
    prior_predictions = np.vstack(prior_predictions)
    prior_interval = scipy.stats.norm.interval(0.95, loc=np.mean(prior_predictions, axis=0), scale=np.std(prior_predictions, axis=0))

    posterior_predictions = np.vstack(posterior_predictions)
    posterior_interval = scipy.stats.norm.interval(0.95, loc=np.mean(posterior_predictions, axis=0), scale=np.std(posterior_predictions, axis=0))

    X = np.arange(num_obs)
    ax.fill_between(X, prior_interval[0], prior_interval[1], alpha=0.3, color='r', label=r'$Prior 2 \sigma region$')
    ax.fill_between(X, posterior_interval[0], posterior_interval[1], alpha=0.3, color='b', label=r'$Posterior 2 \sigma region$')
    ax.plot(X, np.mean(posterior_predictions, axis=0), label=r'Posterior mean value', linestyle='--', marker='o')
    ax.errorbar(X, exp_Y, yerr=exp_Yerr, label='Experimental results', ecolor='g', color='g')
    par_name = [name[0:15] if len(name) > 14 else name for name in list(model_Y)]
    ax.set_xticks(X)
    ax.set_xticklabels(par_name, rotation=45, ha='right')
    ax.margins(0.2)
    ax.set_xlim([-1, num_obs+1])
    
    ax.legend()

if __name__ == '__main__':
    fig = plt.figure(figsize=(13,10))
    PlotOutput('result/junk.h5', fig)
    plt.show()
