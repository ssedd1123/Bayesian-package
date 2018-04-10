import cPickle as pickle
import pymc3 as pm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.pyplot import *
import math
import random
import scipy

from Emulator import *
from PipeLine import *
from Convergency_check import PlotMarginalLikelihood

# load the prior
prior = pd.read_csv('parameter_priors.csv')

with open('e120/e120_result.pkl', 'rb') as buff:
    data = pickle.load(buff)

emulator, trace, pipe2,  = data['model'], data['trace'], data['input_pipe'] 
pipe, exp_data, exp_err = data['output_pipe'], data['exp_data'], data['exp_err']
prior = data['prior']

def PlotOutput(plot_prior=True):

    """
    Function to plot both the posterior and prior point
    if prior is chosen, it will chose points at random
    otherwise it will load configuration from trace
    """

    if plot_prior:
        print('we are ploting prior')
    else:
        print('we are plotting posterior')
    # storing the index of each variable
    index_list = []
    # storing the result
    result_list = []

    for i, row in trace.iterrows():
        par = []
        for par_name in list(prior):
            if plot_prior:
                par.append(random.uniform(prior[par_name][0], prior[par_name][1]))
            else:
                par.append(row[par_name])

        # transform input by input_pipe and put it in our emulator
        par = np.array(par)
        par = pipe2.Transform(par.reshape(1,-1))
        mean, var = emulator.Emulate(par)

        # need to transform back to our output space by output_pipe
        result = pipe.TransformInv(mean.reshape(1,-1)).flatten()
        num_output = result.shape[0]
    
        # interpolate points inbetween index for visualization purpose
        index = np.linspace(0, num_output - 1, 100)
        index_list.append(index)
        result_list.append(np.interp(index, np.arange(num_output), result))
        
        if i > 2000:
            break

    # plot the result
    result = np.stack(result_list)
    interval = scipy.stats.norm.interval(0.95, loc=np.mean(result, axis=0), scale=np.std(result, axis=0))
    index = np.stack(index_list)
    plt.hist2d(index.flatten(), result.flatten(), bins=100, norm=mpl.colors.LogNorm())

    # calculate and plot 95 % confident level
    index = np.arange(num_output)
    plt.errorbar(index, exp_data, yerr=exp_err)
    plt.plot(np.linspace(0, num_output - 1, interval[0].shape[0]), interval[0], linewidth=6, color='black')
    plt.plot(np.linspace(0, num_output - 1, interval[1].shape[0]), interval[1], linewidth=6, color='black')
    plt.show()

PlotOutput(True)
PlotOutput(False)
