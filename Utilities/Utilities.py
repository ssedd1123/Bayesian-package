from multiprocessing import Pool
import cPickle as pickle
import pandas as pd
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import math

from Emulator.Emulator import *
from Preprocessor.PipeLine import *

# input is the pymc3 trace and list of parameters
def PlotTrace(trace, par_name, prior):
    """
    Arrange trace in a n*n matrix of plots
    where n is the number of variables
    """
    # plot the result in a nice matrix of histograms
    num_par = len(par_name)
    graph_num = 1
    fig, axes2d = plt.subplots(num_par, num_par)

    for i, row in enumerate(axes2d):
        for j, cell in enumerate(row):
            namex = par_name[j]
            namey = par_name[i]
            if namex == namey:
                cell.hist(trace[namex], bins = 50, range=np.array([prior[namex][0], prior[namex][1]]))
                cell.set_xlim([prior[namex][0], prior[namex][1]])
            else:
                im = cell.hist2d(trace[namex], trace[namey], bins=50, range=np.array([(prior[namex][0], prior[namex][1]),(prior[namey][0], prior[namey][1])]))#, norm=colors.LogNorm())
                cell.set_xlim([prior[namex][0], prior[namex][1]])
                cell.set_ylim([prior[namey][0], prior[namey][1]])
                fig.colorbar(im[3], ax=cell)
            if i == num_par - 1:
                cell.set_xlabel(namex, fontsize=30)
            if j == 0:
                cell.set_ylabel(namey, fontsize=30)

    return fig, axes2d


def GenerateTrace(emulator, exp_result, exp_cov, prior, id_):
    """
    The main function to generate pandas trace file after comparing the emulator with experimental value
    Uses pymc2 as it is found to be faster
    """
    # form random variables according to prior 
    import pymc
    #pymc.numpy.random.seed(id_)
    parameters = []
    for column in prior:
        parameters.append(pymc.Uniform(column, prior[column][0], prior[column][1]))
    
    @pymc.stochastic(observed=True)
    def emulator_result(value=exp_result, x=parameters):
        mean, var = emulator.Emulate(np.array(x).reshape(1, -1))
        var = var + exp_cov
        return np.array(mvn.logpdf(value, mean, var))
    
    parameters.append(emulator_result)
    
    model = pymc.Model(parameters)
    
    # prepare for MCMC
    mcmc = pymc.MCMC(model)
     
    # sample from our posterior distribution 50,000 times, but
    # throw the first 20,000 samples out to ensure that we're only
    # sampling from our steady-state posterior distribution
    mcmc.sample(iter=20000, burn=5000)
    trace_dict = {}
    for column in prior:
        trace_dict[column] = mcmc.trace(column)[:]
    return pd.DataFrame.from_dict(trace_dict)
    
    
def PlotMarginalLikelihood(emulator_function, scales_min=1e-2, scales_max=2, scales_num=20, nuggets_min=1e-2, nuggets_max=20, nuggets_num=20):
    """
    This function only works if there is only 1 value of scale 
    It works for old version of emulator which has only 1 scale value
    It will not work for emulator with ARD
    """
    scales = np.logspace(np.log10(scales_min), np.log10(scales_max), num=scales_num, base=10)
    nuggets = np.logspace(np.log10(nuggets_min), np.log10(nuggets_max), num=nuggets_num, base=10)

    # create log scale meshgrid for filling array of log likelihood
    scalesv, nuggetsv = np.meshgrid(scales, nuggets)

    log_likelihood = np.array([emulator_function(scale, nugget) for sv, nv in zip(scalesv, nuggetsv) for scale, nugget in zip(sv, nv)])

    # reshape such that z array matches the shape of the meshtrid
    z = log_likelihood.reshape(scalesv.shape[0], -1)
    z_range = np.amax(z) - np.amin(z)
    threshold = np.amax(z) - 0.01*z_range
    threshold_indices = z < threshold
    z[threshold_indices] = threshold
    plot = plt.contour(scalesv, nuggetsv, z, 40)
    plt.yscale('log')
    plt.xscale('log')
    plt.colorbar(plot)
    plt.show()


def PlotOutput(plot_prior=True):

    """
    Function to plot both the posterior and prior point
    if prior is chosen, it will chose points at random
    otherwise it will load configuration from trace
    """

    with open('result/test.pkl', 'rb') as buff:
        data = pickle.load(buff)

    emulator, trace, pipe2,  = data['model'], data['trace'], data['input_pipe'] 
    pipe, dataloader = data['output_pipe'], data['data']
    prior = data['prior']
    exp_data, exp_err = dataloader.exp_result, np.diag(np.sqrt(dataloader.exp_cov))

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

