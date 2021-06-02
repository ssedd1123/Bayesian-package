import glob
import math
import os
import pickle as pickle
import profile
import random
import sys
import time
from multiprocessing import Pool

from scipy.interpolate import InterpolatedUnivariateSpline
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np
import pandas as pd
import scipy
from numpy import array
import pymc as pc

from Preprocessor.PipeLine import *
from Utilities.Utilities import GetTrainedEmulator

one_sigma_confidence = 0.6827
two_sigma_confidence = 0.9545


def PlotOutput(filename, fig, n_samples=20000, trace_filename=None):
    """
    Function to plot both the posterior and prior point
    if prior is chosen, it will chose points at random
    otherwise it will load configuration from trace
    """
    curr_model_name = None
    clf, prior, exp_Y, exp_Yerr, model_X, model_Y, training_idx, _, curr_model_name = GetTrainedEmulator(
        filename)
    # trace filename can be different from emulator
    if trace_filename is None:
        trace_filename = filename

    id_to_model = None
    with pd.HDFStore(trace_filename, "r") as store:
        print('Trace file: ' + trace_filename, flush=True)
        trace = store["trace"]
        if 'ModelChoice' in trace:
            id_to_model = store.get_storer("trace").attrs['model_names']
    id_ = None
    if id_to_model is not None:
        id_ = id_to_model.index(curr_model_name)
            
    n_sample = n_samples if n_samples < trace.shape[0] else trace.shape[0]
    trace = trace.sample(n=n_sample)

    n_progress_divisions = 10
    section_size = n_sample/n_progress_divisions
    prior_trace = np.random.uniform(low=prior["Min"], high=prior["Max"], size=(n_sample, prior.shape[0]))
    prior_trace = pd.DataFrame(prior_trace, columns=prior.index.tolist())

    steps = 0
    nsteps = 2 # two steps process, one for prior and one for posterior

    def PosteriorCalculate(trace, model_Y, clf, confidence=two_sigma_confidence, id_=None):
        # subdivide trace to 20 subdivistions for progress report purposes
        nonlocal steps

        num_obs = model_Y.shape[1]
        posterior_predictions = []
        para_name = prior.index.tolist()
        # load different parameters for model comparison
        if id_ is not None:
            if id_ != 0:
                para_name = ['%s_%d' % (name, id_) for name in para_name]
            trace = trace[trace['ModelChoice'] == id_]

        dfs = np.array_split(trace, n_progress_divisions, axis=0)
        for index, df in enumerate(dfs):
            par = df[para_name].values
            result, _ = clf.Predict(par)
            posterior_predictions.append(result)

            steps = steps + 1
            pub.sendMessage(
                "PosteriorOutputProgress", progress= steps/(nsteps*n_progress_divisions))

        posterior_predictions = np.vstack(posterior_predictions)
        posterior_interval = [[], []]
        for idx in range(posterior_predictions.shape[1]):
            temp = pc.utils.hpd(posterior_predictions[:, idx], 1-confidence)
            posterior_interval[0].append(temp[0])
            posterior_interval[1].append(temp[1])
        
        X_fill = np.arange(num_obs)
        if num_obs == 1:  # expand the x-range so that the band is visible
            X_fill = np.array([-0.5, 0.5])
            posterior_interval = np.repeat(posterior_interval, 2, axis=1)
            posterior_predictions = np.repeat(posterior_predictions, 2, axis=1)

        return X_fill, posterior_interval, np.mean(posterior_predictions, axis=0)

    # plot the result
    ax = fig.subplots(1, 1)

    try:
        pub.sendMessage('Posterior_Drawing')
        X_fill, posterior_interval, posterior_predictions = PosteriorCalculate(trace, model_Y, clf, id_=id_)
        X_fill, prior_interval, _ = PosteriorCalculate(prior_trace, model_Y, clf, confidence=0.9999)
    except Exception as e:
        raise e
    finally:
        pub.sendMessage('Posterior_Drawn')

    x_interpolate = np.linspace(X_fill[0] - 0.1, X_fill[-1] + 0.1, 100)
    num_obs = exp_Y.shape[0]
    X = np.arange(num_obs)
    if num_obs == 1:
        korder = 1
    elif num_obs <= 3:
        korder = 2
    else:
        korder = 3
    prior_area = ax.fill_between(
        x_interpolate,
        InterpolatedUnivariateSpline(X_fill, prior_interval[0], k=korder, ext=0)(x_interpolate),
        InterpolatedUnivariateSpline(X_fill, prior_interval[1], k=korder, ext=0)(x_interpolate),
        alpha=1,
        color="skyblue",
        label=r"Prior region",
        zorder=1
    )
    ax.fill_between(
        x_interpolate,
        InterpolatedUnivariateSpline(X_fill, posterior_interval[0], k=korder, ext=0)(x_interpolate),
        InterpolatedUnivariateSpline(X_fill, posterior_interval[1], k=korder, ext=0)(x_interpolate),
        alpha=0.7,
        color="darkviolet",
        gid='post',
        zorder=2
        #label=r"Posterior $2 \sigma$ region",
    )
    p2 = ax.fill(np.NaN, np.NaN, alpha=0.7, color="darkviolet")
    p1 = ax.plot(
        x_interpolate,
        InterpolatedUnivariateSpline(X_fill, posterior_predictions, k=korder, ext=0)(x_interpolate),
        #label=r"Posterior mean value",
        linestyle="--",
        linewidth=4,
        color="black",
        zorder=3
        # marker="o",
    )
    exp_plot = ax.errorbar(
        X,
        exp_Y,
        yerr=exp_Yerr,
        label="Experimental results",
        ecolor="orange",
        color="orange",
        linewidth=0,
        elinewidth=3,
        markersize=14,
        zorder=100,
        marker="o")
    par_name = [name[0:15] if len(
        name) > 14 else name for name in list(model_Y)]
    ax.set_xticks(X)
    ax.set_xticklabels(par_name, rotation=45, ha="right")
    ax.tick_params(axis='both', labelsize=20)
    ax.set_xlim([-0.5, num_obs-0.5])

    fig.subplots_adjust(bottom=0.2)
    fig.subplots_adjust(left=0.1)
    fig.subplots_adjust(top=0.95)
    fig.subplots_adjust(right=0.95)
    ax.legend([(p2[0], p1[0]), prior_area, exp_plot], [r"Posterior $2 \sigma$ region", r"Prior region", "Experimental results"], fontsize=20)


    #if id_to_model is not None:
    #    axbutton = fig.add_axes([0.81, 0.05, 0.1, 0.05])#plt.axes([0.81, 0.05, 0.1, 0.05])
    #    btn = Button(axbutton, 'Only %s' % curr_model_name)
    #    btn.label.set_fontsize(15)
    #    id_ = id_to_model.index(curr_model_name)
    #    nsteps = 1 # for every update, only posterior needs to be drawn

    #    def onClick(event):
    #        try:
    #            pub.sendMessage('Posterior_Drawing')
    #            nonlocal steps
    #            steps = 0
    #            if btn.label.get_text() == 'All models':
    #                btn.label.set_text('Only %s' % curr_model_name)
    #                X_fill, posterior_interval, posterior_predictions = PosteriorCalculate(trace, model_Y, clf, id_=id_)
    #            else:
    #                btn.label.set_text('All models')
    #                X_fill, posterior_interval, posterior_predictions = PosteriorCalculate(trace[trace['ModelChoice'] == id_], model_Y, clf, id_=id_)
    #            for collection in ax.collections:
    #                if collection.get_gid() == 'post':
    #                    collection.remove()
    #            p1[0].set_ydata(InterpolatedUnivariateSpline(X_fill, posterior_predictions, k=korder, ext=0)(x_interpolate))
    #            ax.fill_between(
    #                x_interpolate,
    #                InterpolatedUnivariateSpline(X_fill, posterior_interval[0], k=korder, ext=0)(x_interpolate),
    #                InterpolatedUnivariateSpline(X_fill, posterior_interval[1], k=korder, ext=0)(x_interpolate),
    #                alpha=0.7,
    #                color="darkviolet",
    #                gid='post',
    #                zorder=2
    #                #label=r"Posterior $2 \sigma$ region",
    #            )
    #            fig.canvas.draw_idle()
    #        except Exception as e:
    #            raise e
    #        finally:
    #            pub.sendMessage('Posterior_Drawn')

    #    btn.on_clicked(onClick)
    #    return btn
    #return None
    


if __name__ == "__main__":
    fig = plt.figure(figsize=(13, 10))
    btn = PlotOutput("result/SineMD", fig)
    plt.show()
