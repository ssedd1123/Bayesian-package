import glob
import math
import os
import pickle as pickle
import profile
import random
import sys
import time
from multiprocessing import Pool

from scipy.interpolate import InterpolatedUnivariateSpline, interp1d
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

def PlotClosureTest(fig, varNames, truths, predictions, prediction_errs): 
    """
    truths, predictions and prediction_errs must have the same size
    col: Different parameter set
    row: Different observable values
    """
    truths = np.atleast_2d(truths)
    predictions = np.atleast_2d(predictions)
    prediction_errs = np.atleast_2d(prediction_errs)
    assert truths.shape == predictions.shape, "Shape of truths and predictions are not identical!"

    nvar = predictions.shape[1]
    assert nvar == len(varNames), 'Number of observable names and predicted observable are not identical!'
    # plot the result
    ncol = min(5, nvar)
    nrow = max(1, int(nvar/ncol) + 1)
    axes = fig.subplots(nrow, ncol)

    for ax, varName, true, pred, err in zip(axes.flat, varNames, truths.T, predictions.T, prediction_errs.T):
        ax.errorbar(true, pred, yerr=err, linewidth=0, 
                     marker='o', elinewidth=2, markersize=10, capsize=5, capthick=2)
        ax.plot([min(true), max(true)], [min(true), max(true)], color='r', linestyle='--')
        ax.set_xlabel('True ' + varName)
        ax.set_ylabel('Predicted ' + varName)
        ax.grid()

    fig.subplots_adjust(bottom=0.2)
    fig.subplots_adjust(left=0.1)
    fig.subplots_adjust(top=0.95)
    fig.subplots_adjust(right=0.95)
    plt.tight_layout()


def PlotClosureTestEmulator(filename, fig, truth_paras, truths):
    assert truths.shape[0] == truth_paras.shape[0], "Number of rows in truth_para and truth are not identical!"

    clfs = GetTrainedEmulator(filename)
    clf = clfs[0]
    para_name = list(clfs[5])

    predictions = []
    prediction_errs = []
    for para in truth_paras:
        result, cov = clf.Predict(para)
        predictions.append(np.atleast_1d(np.squeeze(result)))
        prediction_errs.append(np.sqrt(np.diag(np.atleast_2d(np.squeeze(cov)))))

    PlotClosureTest(fig, para_name, truths, predictions, prediction_errs)
       


if __name__ == "__main__":
    fig = plt.figure(figsize=(13, 10))
    truth_paras = np.atleast_2d([[0.15157219085711038, 0.011806911125960648, 0.09271798338419147, 0.6967770226022433, 0.0242448133700898, 0.04179542835220651],
        [0.1579547785062808, 0.05119859048617932, 0.035342754478854564, 0.8167342997741291, 0.04776475587854532, 0.014474965624455742]])
    truths = np.atleast_2d([[0.996744287, 1.233738449, 0.707882851], [0.979281775, 1.111279748, 1.626370862]])
    PlotClosureTestEmulator('result/THERMUS_DR', fig, truth_paras, truths)
    plt.show()
