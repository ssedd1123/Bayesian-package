from multiprocessing import Pool
import sys
import os
if sys.version_info > (3, 0):
    import pickle
else:
    import cPickle as pickle
import pandas as pd
from pandas.tools.plotting import scatter_matrix
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.gridspec as gridspec
import math
# form random variables according to prior 
import pymc


from Emulator.Emulator import *
from Preprocessor.PipeLine import *

# input is the pymc3 trace and list of parameters
def PlotTrace(trace, par_name, prior, fig=None):
    """
    Arrange trace in a n*n matrix of plots
    where n is the number of variables
    """
    # plot the result in a nice matrix of histograms
    num_par = len(par_name)
    graph_num = 1
    if fig is None:
        fig, axes2d = plt.subplots(num_par, num_par)
    else:
        axes2d = fig.subplots(num_par, num_par)
    if num_par == 1:
        axes2d = [[axes2d]]
    for i, row in enumerate(axes2d):
        for j, cell in enumerate(row):
            namex = par_name[j]
            namey = par_name[i]
            if namex == namey:
                cell.hist(trace[namex], bins = 50, range=np.array([prior[1][namex], prior[2][namex]]))
                cell.set_xlim([prior[1][namex], prior[2][namex]])
            else:
                im = cell.hist2d(trace[namex], trace[namey], bins=50, range=np.array([(prior[1][namex], prior[2][namex]),(prior[1][namey], prior[2][namey])]))#, norm=colors.LogNorm())
                cell.set_xlim([prior[1][namex], prior[2][namex]])
                cell.set_ylim([prior[1][namey], prior[2][namey]])
                #fig.colorbar(im[3], ax=cell)
            # Modify axis labels such that the top and bottom label never show up
            xlist = cell.get_xticks().tolist()
            xlist[0] = ''
            xlist[-1] = ''
            #cell.set_xticklabels(xlist, rotation=45)

            ylist = cell.get_yticks().tolist()
            ylist[0] = ''
            ylist[-1] = ''
            #cell.set_yticklabels(ylist)

            cell.tick_params(axis='both', which='major', labelsize=20)

            if i == num_par - 1:
                cell.set_xlabel(namex, fontsize=30)
            else:
                cell.set_xticklabels([])
            if j == 0:
                cell.set_ylabel(namey, fontsize=30)
            else:
                cell.set_yticklabels([])
            if i == 0 and j == 0:
                cell.set_yticklabels(cell.get_xticks().tolist())
 
            
    plt.subplots_adjust(wspace=0, hspace=0)
    return fig, axes2d


def GenerateTrace(emulator, exp_result, exp_cov, prior, id_, iter):
    """
    The main function to generate pandas trace file after comparing the emulator with experimental value
    Uses pymc2 as it is found to be faster
    """
    pymc.numpy.random.seed(id_)
    parameters = []
    for name, row in prior.iterrows():
        if row[0] == 'Uniform':
            parameters.append(pymc.Uniform(name, row[1], row[2], value=(0.5*row[1] + 0.5*row[2])))
        else:
            sys.stderr.write('%s, %f, %f\n' % (name, row[3], row[4]))
            parameters.append(pymc.TruncatedNormal(name, mu=row[3], tau=1./row[4]**2, a=row[1], b=row[2], value=row[3]))
    
    @pymc.stochastic(observed=True)
    def emulator_result(value=exp_result, x=parameters):
        mean, var = emulator.Emulate(np.array(x).reshape(1, -1))
        var = var + exp_cov
        return np.array(mvn.logpdf(value, mean, var))
        #return np.array(emulator.GetLogP(np.array(x).reshape(1, -1), value, exp_cov))

    parameters.append(emulator_result)
    
    model = pymc.Model(parameters)
    
    # prepare for MCMC
    mcmc = pymc.MCMC(model)
     
    # sample from our posterior distribution 50,000 times, but
    # throw the first 20,000 samples out to ensure that we're only
    # sampling from our steady-state posterior distribution
    mcmc.sample(iter, burn=1000)
    trace_dict = {}
    for name, row in prior.iterrows():
        trace_dict[name] = mcmc.trace(name)[:]
    mcmc.db.close()
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
        for par_name in list(prior.index.values):
            if plot_prior:
                par.append(random.uniform(prior[1][par_name], prior[2][par_name]))
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

