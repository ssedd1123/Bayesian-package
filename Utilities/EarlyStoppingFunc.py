import math
import operator
import sys

import autograd.numpy as np
import cPickle as pickle
import matplotlib.pyplot as plt
import pandas as pd
import pymc3 as pm
import theano
import theano.tensor as tt
from Emulator.Emulator import *
from theano.compile.ops import as_op

from Preprocessor.PipeLine import *
from Utilities.Convergency_check import PlotMarginalLikelihood
from Utilities.Utilities import *


def EarlyStopping(
    input_,
    target,
    validation_runs,
    initial_scales,
    initial_nuggets,
    input_pipe,
    output_pipe,
    scales_rate=0.001,
    nuggets_rate=0.003,
    nsteps=1500,
):
    valid_par = input_[validation_runs.tolist()]
    valid_obs = target[validation_runs.tolist()]

    input_ = np.delete(input_, validation_runs, 0)
    target = np.delete(target, validation_runs, 0)

    input_pipe.Fit(input_)
    output_pipe.Fit(target)

    valid_par = input_pipe.Transform(valid_par)
    valid_obs = output_pipe.Transform(valid_obs)

    emulator = EmulatorMultiOutput(
        input_pipe.Transform(input_), output_pipe.Transform(target)
    )
    emulator.SetCovariance(squared_exponential)

    scales_list = []
    nuggets_list = []

    for index, emu in enumerate(emulator.emulator_list):
        gd = GradientDescentForEmulator(scales_rate, nuggets_rate)
        gd.SetFunc(emu.MarginalLikelihood)

        nuggets = initial_nuggets
        scales = initial_scales
        hist_scales = []
        hist_nuggets = []
        partial_likelihood = []
        for i in range(nsteps):

            scales, nuggets, grad_scales, grad_nuggets = gd.StepDescent(scales, nuggets)
            emu.SetScales(scales)
            emu.SetNuggets(nuggets)
            emu.StartUp()

            hist_scales.append(scales)
            hist_nuggets.append(nuggets)

            val = 0
            for par, exp in zip(valid_par, valid_obs):
                val += emu.LogLikelihood(par.reshape(1, -1), exp[index].reshape(1, -1))
            partial_likelihood.append(val)

            sys.stdout.write(
                "\rProcessing %i iteration, likelihood = %f, nuggets = %f, scales = %s"
                % (i, val, nuggets, np.array2string(scales))
            )
            sys.stdout.flush()

        # plt.plot(partial_likelihood)
        # plt.show()
        # get index corresponds to maximum validation log likelihood
        i, value = max(enumerate(partial_likelihood), key=operator.itemgetter(1))
        print("max", i)
        scales_list.append(hist_scales[i])
        nuggets_list.append(hist_nuggets[i])
    return np.array(scales_list), np.array(nuggets_list)
