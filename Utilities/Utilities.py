import os
import random
import sys

if sys.version_info > (3, 0):
    import pickle
else:
    import cPickle as pickle

import math
import scipy
from scipy.optimize import bisect
from scipy.stats.mstats import mquantiles

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
    if isinstance(store, str):
        newstore = pd.HDFStore(store, "r")
    else:
        newstore = store
    prior = newstore["PriorAndConfig"]
    exp_Y = newstore["Exp_Y"]
    exp_Yerr = newstore["Exp_YErr"]
    model_Y = newstore["Model_Y"]
    model_X = newstore["Model_X"]
    history_para = newstore["ParaHistory"]

    attrs = newstore.get_storer("PriorAndConfig").attrs.my_attribute
    emulator = eval(attrs["repr"])

    model_name = None
    if "name" in attrs:
        model_name = attrs["name"]

    if "Training_idx" in newstore:
        training_idx = newstore["Training_idx"].astype("int").values.flatten()
    else:
        training_idx = np.arange(model_X.shape[0])
    emulator.Fit(model_X.values[training_idx], model_Y.values[training_idx])

    if isinstance(store, str):
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
        model_name
    )


def smoothed_histogram2D(
        x,
        y,
        ax,
        bins=20,
        sigma=0,
        nlevels=10,
        extent=None,
        show_confidence=False,
        **kwargs):
    if extent is not None:
        h, x, y = np.histogram2d(x, y, bins=bins, range=[
            [extent[0], extent[1]], [extent[2], extent[3]]], density=True)
    else:
        h, x, y = np.histogram2d(x, y, bins=bins, density=True)
    if extent is None:
        extent = [x[0], x[-1], y[0], y[-1]]
    if sigma > 0 or show_confidence:
        h = gaussian_filter(h, sigma=sigma)
        if show_confidence:
            confidence_interval = [0.99, 0.95, 0.68]
            levels = [HeightAtFraction(h, val) for val in confidence_interval] + [np.amax(h)]#, HeightAtFraction(h, 0.99)] 
            im = ax.contourf(h.T, levels=levels, origin="lower", extent=extent, **kwargs)
            cset = ax.contour(h.T, levels=levels, origin='lower', extent=extent, colors='k')
            # Label plot
            fmt = {}
            for lvl, height in zip(levels[:-1], confidence_interval):
                fmt[lvl] = '%g %%' % (100*height)
            ax.clabel(cset, inline=1, fontsize=10, fmt=fmt)
        else:
            im = ax.contourf(h.T, nlevels, origin='lower', extent=extent, **kwargs)
    else:
        im = ax.imshow(
            h.T,
            origin="lower",
            extent=extent,
            aspect="auto",
            **kwargs)
    # im = ax.contour(h.T, 10, origin='lower', extent=extent, aspect='auto', **kwargs)
    return ax, im


def smoothed_histogram1D(x, ax, bins=20, sigma=0, range=None, show_confidence=False, **kwargs):
    h, xax = np.histogram(x, bins=bins, range=range)
    xax = 0.5 * (xax[1:] + xax[:-1])

    if sigma > 0:
        #h = gaussian_filter1d(h, sigma=sigma)
        from KDEpy import FFTKDE
        kde = FFTKDE(bw='silverman').fit(x.to_numpy())
        x2, y = kde.evaluate(bins)
        g = ax.plot(x2, y, **kwargs)
        #g = ax.plot(xax, h, **kwargs)
    else:
        g = ax.step(xax, h, **kwargs)
    ax.set_ylim(bottom=0)

    if show_confidence:
        quant = mquantiles(x, [0.025, 0.5, 0.975])
        for i, val in enumerate(quant):
            ax.axvline(val, linestyle='-' if i == 1 else '--', color='black')
    return ax, g


def HeightAtFraction(h, frac):
    def FractionAboveZ(h, z):
        total = h.sum()
        value_above = h[h > z].sum()
        return value_above / total
    return bisect(
        lambda x: FractionAboveZ(
            h,
            x) - frac,
        0,
        np.max(h),
        xtol=1e-5)


# input is the pymc3 trace and list of parameters
def PlotTrace(
        config_file,
        fig=None,
        sigma=0,
        bins=100,
        cmap="Blues",
        nlevels=10,
        mark_point=None,
        show_confidence=False,
        only_lower=True):
    """
    Arrange trace in a n*n matrix of plots
    where n is the number of variables
    """
    # plot the result in a nice matrix of histograms
    with pd.HDFStore(config_file, "r") as store:
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
            if only_lower and j > i:
                cell.remove()
                continue
            namex = par_name[j]
            namey = par_name[i]
            if namex == namey:
                # use a dummy twinx plot to 'fake' y-axis for diagonal plots at
                # upper left coner
                cell.set_ylim([prior["Min"][namey], prior["Max"][namey]])
                dummy = cell.twinx()
                # dummy.hist(trace[namex], bins=100, range=[prior['Min'][namex], prior['Max'][namex]])
                smoothed_histogram1D(
                    trace[namex],
                    bins=bins,
                    sigma=sigma,
                    range=[prior["Min"][namex], prior["Max"][namex]],
                    ax=dummy,
                    show_confidence=show_confidence
                )
                dummy.yaxis.set_ticks([])
                dummy.set_xlim([prior["Min"][namex], prior["Max"][namex]])
                if mark_point is not None:
                    for point in np.atleast_1d(
                            mark_point[namex]).astype(
                            np.float):
                        dummy.axvline(point, color='r')
            else:
                smoothed_histogram2D(
                    trace[namex].to_numpy(),
                    trace[namey].to_numpy(),
                    bins=bins,
                    sigma=sigma,
                    ax=cell,
                    cmap=cmap,
                    vmin=0,
                    nlevels=nlevels,
                    extent=[
                        prior["Min"][namex],
                        prior["Max"][namex],
                        prior["Min"][namey],
                        prior["Max"][namey],
                    ],
                    show_confidence=show_confidence
                )

                cell.set_xlim([prior["Min"][namex], prior["Max"][namex]])
                cell.set_ylim([prior["Min"][namey], prior["Max"][namey]])

                if mark_point is not None:
                    cell.scatter(
                        np.atleast_1d(
                            mark_point[namex]).astype(
                            np.float),
                        np.atleast_1d(
                            mark_point[namey]).astype(
                            np.float),
                        color='r')
                    cell.axvline(np.atleast_1d(mark_point[namex]).astype(np.float), color='r')
                    cell.axhline(np.atleast_1d(mark_point[namey]).astype(np.float), color='r')

    # handle axis labes for coner graphs
    for i, row in enumerate(axes2d):
        for j, cell in enumerate(row):
            # Modify axis labels such that the top and bottom label never show
            # up
            if only_lower and j > i:
                continue

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
    plt.tight_layout()
    return fig, axes2d
