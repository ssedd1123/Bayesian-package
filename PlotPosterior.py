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

from Emulator.Emulator import *
from Preprocessor.PipeLine import *

def PlotOutput(plot_prior, filename):

    """
    Function to plot both the posterior and prior point
    if prior is chosen, it will chose points at random
    otherwise it will load configuration from trace
    """

    with open(filename, 'rb') as buff:
        data = pickle.load(buff)

    emulator  = data['emulator']
    trace = pd.read_hdf('%s.h5' % filename)
    dataloader = data['data']
    prior = dataloader.prior
    exp_data, exp_err = dataloader.exp_result, np.diag(np.sqrt(dataloader.exp_cov))

    if plot_prior:
        print('we are ploting prior')
    else:
        print('we are plotting posterior')
    # storing the index of each variable
    index_list = []
    # storing the result
    result_list = []

    for i, row in trace.iloc[::30, :].iterrows():
        par = []
        for name, par_row in prior.iterrows():
            if plot_prior:
                par.append(random.uniform(par_row[1], par_row[2]))
            else:
                par.append(row[name])

        # transform input by input_pipe and put it in our emulator
        result, var = emulator.Emulate(par)

        # need to transform back to our output space by output_pipe
        num_output = result.shape[0]
    
        # interpolate points inbetween index for visualization purpose
        index = np.linspace(0, num_output - 1, 100)
        index_list.append(index)
        result_list.append(np.interp(index, np.arange(num_output), result))
        
        if i > 40000:
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
    #plt.ylim([0.7, 1.6])
    plt.xlim([-1, num_output])
    plt.show()

if len(sys.argv) != 2:
    #print('Use this script by entering: python %s Iutput_name' % (sys.argv[0]))
    #sys.exit()
    list_of_files = glob.glob('training/*')
    filename = max(list_of_files, key=os.path.getctime)
else:
    filename = sys.argv[1]


PlotOutput(True, filename)
PlotOutput(False, filename)
