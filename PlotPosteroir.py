import cPickle as pickle
import pymc3 as pm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import math
import theano
import theano.tensor as tt
import random
from theano.compile.ops import as_op

from Emulator import *
from PipeLine import *
from Convergency_check import PlotMarginalLikelihood

# load the prior
prior = pd.read_csv('parameter_priors.csv')

with open('trace.pkl', 'rb') as buff:
    data = pickle.load(buff)

emulator, trace, pipe2,  = data['model'], data['trace'], data['input_pipe'] 
pipe, exp_data, exp_err = data['output_pipe'], data['exp_data'], data['exp_err']
prior = data['prior']

def PlotOutput(plot_prior=True):

    if plot_prior:
        print('we are ploting prior')
    else:
        print('we are plotting posterior')
    index_list = []
    result_list = []
    for i, row in trace.iterrows():
        par = []
        for par_name in list(prior):
            if plot_prior:
                par.append(random.uniform(prior[par_name][0], prior[par_name][1]))
            else:
                par.append(row[par_name])
        par = np.array(par)
        par = pipe2.Transform(par.reshape(1,-1))
        mean, var = emulator.Emulate(par)
        result = pipe.TransformInv(mean.reshape(1,-1)).flatten()
        num_output = result.shape[0]
    
        index = np.linspace(0, num_output - 1, 100)
        index_list.append(index)
        result_list.append(np.interp(index, np.arange(num_output), result))
        
        if i > 2000:
            break
    plt.hist2d(np.array(index_list).flatten(), np.concatenate(result_list).flatten(), bins=100, norm=mpl.colors.LogNorm())
    plt.errorbar(np.arange(num_output), exp_data, yerr=exp_err)
    plt.show()
