import os
import random
import sys

if sys.version_info > (3, 0):
    import pickle
else:
    import cPickle as pickle

import math

import matplotlib.colors as colors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# form random variables according to prior
import pymc
from matplotlib import cm
from numpy import array
from pandas.plotting import scatter_matrix
from scipy.interpolate import interpn
from scipy.ndimage.filters import gaussian_filter, gaussian_filter1d

from Preprocessor.PipeLine import *


def GetTrainedEmulator(store):
    if type(store) is str:
        newstore = pd.HDFStore(store, "r")
    else:
        newstore = store
    prior = newstore["PriorAndConfig"]
    exp_Y = newstore["Exp_Y"]
    exp_Yerr = newstore["Exp_YErr"]
    model_Y = newstore["Model_Y"]
    model_X = newstore["Model_X"]
    history_para = newstore["ParaHistory"]

    emulator = eval(newstore.get_storer("PriorAndConfig").attrs.my_attribute["repr"])

    if "Training_idx" in newstore:
        training_idx = newstore["Training_idx"].astype("int").values.flatten()
    else:
        training_idx = np.arange(model_X.shape[0])
    emulator.Fit(model_X.values[training_idx], model_Y.values[training_idx])

    if type(store) is str:
        newstore.close()

    return (
        emulator,
        prior,
        exp_Y,
        exp_Yerr,
        model_X,
        model_Y,
        training_idx,
        history_para,
    )


def smoothed_histogram2D(x, y, ax, bins=20, sigma=0, nlevels=10, extent=None, **kwargs):
    if extent is not None:
        h, x, y = np.histogram2d(
            x, y, bins=bins, range=[[extent[0], extent[1]], [extent[2], extent[3]]]
        )
    else:
        h, x, y = np.histogram2d(x, y, bins=bins)
    if extent is None:
        extent = [x[0], x[-1], y[0], y[-1]]
    if sigma > 0:
        h = gaussian_filter(h, sigma=sigma)
        im = ax.contourf(h.T, nlevels, origin="lower", extent=extent, **kwargs)
    else:
        im = ax.imshow(h.T, origin="lower", extent=extent, aspect="auto", **kwargs)
    # im = ax.contour(h.T, 10, origin='lower', extent=extent, aspect='auto', **kwargs)
    return ax, im


def smoothed_histogram1D(x, ax, bins=20, sigma=0, range=None, **kwargs):
    h, x = np.histogram(x, bins=bins, range=range)
    if sigma > 0:
        h = gaussian_filter1d(h, sigma=sigma)
        g = ax.plot(0.5 * (x[1:] + x[:-1]), h, **kwargs)
    else:
        g = ax.step(0.5 * (x[1:] + x[:-1]), h, **kwargs)
    ax.set_ylim(bottom=0)
    return ax, g


# input is the pymc3 trace and list of parameters
def PlotTrace(config_file, fig=None, sigma=0, bins=100, cmap="Blues", nlevels=10):
    """
    Arrange trace in a n*n matrix of plots
    where n is the number of variables
    """
    # plot the result in a nice matrix of histograms
    store = pd.HDFStore(config_file, "r")
    trace = store["trace"]
    prior = store["PriorAndConfig"]
    num_par = prior.shape[0]
    par_name = prior.index

    graph_num = 1
    if fig is None:
        fig, axes2d = plt.subplots(num_par, num_par)
    else:
        axes2d = fig.subplots(num_par, num_par)

    if num_par == 1:
        axes2d = [[axes2d]]
    prior["Min"] = prior["Min"].astype("float")
    prior["Max"] = prior["Max"].astype("float")

    for i, row in enumerate(axes2d):
        for j, cell in enumerate(row):
            namex = par_name[j]
            namey = par_name[i]
            if namex == namey:
                # use a dummy twinx plot to 'fake' y-axis for diagonal plots at upper left coner
                cell.set_ylim([prior["Min"][namey], prior["Max"][namey]])
                dummy = cell.twinx()
                # dummy.hist(trace[namex], bins=100, range=[prior['Min'][namex], prior['Max'][namex]])
                smoothed_histogram1D(
                    trace[namex],
                    bins=bins,
                    sigma=sigma,
                    range=[prior["Min"][namex], prior["Max"][namex]],
                    ax=dummy,
                )
                dummy.yaxis.set_ticks([])
                dummy.set_xlim([prior["Min"][namex], prior["Max"][namex]])
            else:
                smoothed_histogram2D(
                    trace[namex].to_numpy(),
                    trace[namey].to_numpy(),
                    bins=bins,
                    sigma=sigma,
                    ax=cell,
                    cmap=cmap,
                    extent=[
                        prior["Min"][namex],
                        prior["Max"][namex],
                        prior["Min"][namey],
                        prior["Max"][namey],
                    ],
                )

                cell.set_xlim([prior["Min"][namex], prior["Max"][namex]])
                cell.set_ylim([prior["Min"][namey], prior["Max"][namey]])

    # handle axis labes for coner graphs
    for i, row in enumerate(axes2d):
        for j, cell in enumerate(row):
            # Modify axis labels such that the top and bottom label never show up
            cell.tick_params(axis="both", which="major", labelsize=20)
            namex = par_name[j]
            namey = par_name[i]

            if i == num_par - 1:
                cell.set_xlabel(namex, fontsize=30)
            else:
                cell.set_xlabel("")
                cell.set_xticklabels([])
            if j == 0:
                cell.set_ylabel(namey, fontsize=30)
            else:
                cell.set_ylabel("")
                cell.set_yticklabels([])

    plt.subplots_adjust(wspace=0, hspace=0)
    store.close()
    return fig, axes2d
