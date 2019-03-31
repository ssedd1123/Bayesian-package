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

from Emulator.Emulator import *
from Preprocessor.PipeLine import *

one_sigma_confidence = 0.6827
two_sigma_confidence = 0.9545

def PlotOutput(filename):

    """
    Function to plot both the posterior and prior point
    if prior is chosen, it will chose points at random
    otherwise it will load configuration from trace
    """

    store = pd.HDFStore('%s.h5' % filename, 'r')
    config = store.get_storer('PriorAndConfig').attrs.my_attribute
    prior = store['PriorAndConfig']
    model_X = store['Model_X'].values
    model_Y = store['Model_Y'].values
    exp_Y = store['Exp_Y'].values
    exp_Yerr = store['Exp_YErr'].values
    trace = store['tracek']

    clf = eval(config['repr'])
    clf.Fit(np.asarray(model_X, dtype=float), np.asarray(model_Y, dtype=float))

    section_size = 4000
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
        
        print('time used %f' % (time.time() - start))
        start = time.time()
        print('Processing %d' % index)
        if index > 4:
            break
    
    # plot the result
    prior_predictions = np.vstack(prior_predictions)
    prior_interval = scipy.stats.norm.interval(0.95, loc=np.mean(prior_predictions, axis=0), scale=np.std(prior_predictions, axis=0))

    posterior_predictions = np.vstack(posterior_predictions)
    posterior_interval = scipy.stats.norm.interval(0.95, loc=np.mean(posterior_predictions, axis=0), scale=np.std(posterior_predictions, axis=0))

    X = np.arange(num_obs)
    plt.fill_between(X, prior_interval[0], prior_interval[1], alpha=0.3, color='r', label=r'$Prior 2 \sigma region$')
    plt.fill_between(X, posterior_interval[0], posterior_interval[1], alpha=0.3, color='b', label=r'$Posterior 2 \sigma region$')
    plt.plot(X, np.mean(posterior_predictions, axis=0), label=r'Posterior mean value', linestyle='--', marker='o')
    plt.errorbar(X, exp_Y, yerr=exp_Yerr, label='Experimental results', ecolor='g', color='g')
    par_name = [name[0:15] if len(name) > 14 else name for name in list(store['Model_Y'])]
    plt.xticks(X, par_name, rotation='vertical')
    plt.margins(0.2)
    plt.xlim([-1, num_obs+1])
    
    plt.legend()
    plt.show()

    # save all results in a dataframe
    data = {'Name': list(store['Model_Y']),
            'PriorMean': np.mean(prior_predictions, axis=0),
            'PriorSD': np.std(prior_predictions, axis=0),
            'PosteriorMean': np.mean(posterior_predictions, axis=0),
            'PosteriorSD': np.std(posterior_predictions, axis=0),
            'Exp_Y': exp_Y,
            'Exp_Yerr': exp_Yerr}
    pd.DataFrame.from_dict(data).to_csv('%s_posterior.csv' % filename)


if __name__ == '__main__':
    PlotOutput('result/newhist')
