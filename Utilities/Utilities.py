import random
import sys
import os
if sys.version_info > (3, 0):
    import pickle
else:
    import cPickle as pickle
import pandas as pd
from pandas.plotting import scatter_matrix
import numpy as np
from numpy import array
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.gridspec as gridspec
import math
# form random variables according to prior 
import pymc


from Preprocessor.PipeLine import *

def GetTrainedEmulator(store):
    if type(store) is str:
        newstore = pd.HDFStore(store, 'r')
    else:
        newstore = store
    prior = newstore['PriorAndConfig']
    exp_Y = newstore['Exp_Y']
    exp_Yerr = newstore['Exp_YErr']
    model_Y = newstore['Model_Y']
    model_X = newstore['Model_X']
    history_para = newstore['ParaHistory']
   
    emulator = eval(newstore.get_storer('PriorAndConfig').attrs.my_attribute['repr'])

    if 'Training_idx' in newstore:
        training_idx = newstore['Training_idx'].astype('int').values.flatten()
    else:
        training_idx = np.arange(model_X.shape[0])
    emulator.Fit(model_X.values[training_idx], model_Y.values[training_idx])

    if type(store) is str:
        newstore.close()
 
    return emulator, prior, exp_Y, exp_Yerr, model_X, model_Y, training_idx, history_para

# input is the pymc3 trace and list of parameters
def PlotTrace(config_file, fig=None):
    """
    Arrange trace in a n*n matrix of plots
    where n is the number of variables
    """
    # plot the result in a nice matrix of histograms
    store = pd.HDFStore(config_file, 'r')
    trace = store['trace']
    prior = store['PriorAndConfig']
    num_par = prior.shape[0]
    par_name = prior.index

    graph_num = 1
    if fig is None:
        fig, axes2d = plt.subplots(num_par, num_par)
    else:
        axes2d = fig.subplots(num_par, num_par)

    if num_par == 1:
        axes2d = [[axes2d]]
    prior['Min'] = prior['Min'].astype('float')
    prior['Max'] = prior['Max'].astype('float')
    for i, row in enumerate(axes2d):
        for j, cell in enumerate(row):
            namex = par_name[j]
            namey = par_name[i]
            if namex == namey:
                cell.hist(trace[namex], bins = 50, range=np.array([prior['Min'][namex], prior['Max'][namex]]))
                cell.set_xlim([prior['Min'][namex], prior['Max'][namex]])
            else:
                im = cell.hist2d(trace[namex], trace[namey], bins=50, 
                                 range=np.array([(prior['Min'][namex], prior['Max'][namex]),
                                                (prior['Min'][namey], prior['Max'][namey])]))#, norm=colors.LogNorm())
                cell.set_xlim([prior['Min'][namex], prior['Max'][namex]])
                cell.set_ylim([prior['Min'][namey], prior['Max'][namey]])
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
    store.close()
    return fig, axes2d



