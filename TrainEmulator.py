import os
import sys

if sys.version_info > (3, 0):
    import pickle
else:
    import cPickle as pickle

import argparse

import autograd.numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure
from numpy import array

import Preprocessor.PipeLine as pl
from Preprocessor.PipeLine import *
from Utilities.GradientDescent import UseDefaultOutput
from Utilities.Utilities import GetTrainedEmulator


def TrainingCurve(fig, config_file):
    args = GetTrainedEmulator(config_file)
    clf = args[0]
    prior = args[1]
    exp_Y = args[2]
    exp_Yerr = args[3]
    model_X = args[4].values
    model_Y = args[5].values
    training_idx = args[6]
    validation_idx = np.setdiff1d(np.arange(model_X.shape[0]), training_idx)
    history_para = args[7].fillna(method="ffill")

    clf.Fit(model_X, model_Y)
    axes = fig.subplots(1, 2)
    NumberOfPts(axes[0], clf, model_X, model_Y, training_idx, validation_idx)
    NumberOfSteps(
        axes[1],
        clf,
        model_X,
        model_Y,
        training_idx,
        validation_idx,
        history_para)


def NumberOfPts(ax, clf, model_X, model_Y, training_idx, validation_idx):
    training_scores = []
    validation_scores = []

    training_X = model_X[training_idx]
    training_Y = model_Y[training_idx]
    validation_X = model_X[validation_idx]
    validation_Y = model_Y[validation_idx]

    valid_ntrain = []
    num_train = len(training_X)
    for ntrain in range(2, num_train):
        pub.sendMessage("NumberOfPtsProgress", progress=ntrain / num_train)
        try:
            clf.Fit(training_X[0:ntrain], training_Y[0:ntrain])
            pred_Y, _ = clf.Predict(training_X[0:ntrain])
            training_scores.append(
                np.sqrt(np.mean((training_Y[0:ntrain] - pred_Y) ** 2))
            )
            pred_Y, _ = clf.Predict(validation_X)
            validation_scores.append(
                np.sqrt(np.mean((validation_Y - pred_Y) ** 2)))
            valid_ntrain.append(ntrain)
        except Exception:
            pass

    ax.plot(valid_ntrain, training_scores, label="training")
    ax.plot(valid_ntrain, validation_scores, label="validation")
    ax.set_xlabel("Number of training points")
    ax.set_ylabel("RMSE per feature")
    ax.legend()


def NumberOfSteps(
    ax, clf, model_X, model_Y, training_idx, validation_idx, history_para
):
    training_scores = []
    validation_scores = []

    training_X = model_X[training_idx]
    training_Y = model_Y[training_idx]
    validation_X = model_X[validation_idx]
    validation_Y = model_Y[validation_idx]
    for idx, para in history_para.iterrows():
        pub.sendMessage(
            "NumberOfStepsProgress",
            progress=idx /
            history_para.shape[0])
        nuggets = []
        scales = []
        for idemu, emulator in enumerate(clf["Emulator"].emulators):
            nuggets.append(para["Nuggets%d" % idemu])
            scales.append(
                para[
                    [
                        "Scales%d_%d" % (idemu, idinput)
                        for idinput in range(model_X.shape[1])
                    ]
                ].values
            )
        clf["Emulator"].scales = np.atleast_2d(scales)
        clf["Emulator"].nuggets = np.atleast_1d(nuggets)

        clf.Fit(training_X, training_Y)
        # training_scores.append(clf.Score(training_X, training_Y))
        # validation_scores.append(clf.Score(validation_X, validation_Y))
        training_scores.append(clf.ChiSq(training_X, training_Y))
        validation_scores.append(clf.ChiSq(validation_X, validation_Y))

    ax.plot(
        range(history_para.shape[0]),
        training_scores,
        label=r"Training $\chi^2$/deg. free",
    )
    ax.plot(
        range(history_para.shape[0]),
        validation_scores,
        label=r"Validation $\chi^2$/deg. free",
    )
    ax.set_xlabel("Number of ephoes")
    ax.set_ylabel(r"$\chi^2$/deg. free")
    ax.legend()


def Training(
    prior,
    model_X,
    model_Y,
    exp,
    training_file,
    principalcomp=None,
    fraction=0.99,
    initialscale=[1],
    initialnugget=0.1,
    scalerate=0.01,
    nuggetrate=0.01,
    maxsteps=1000,
    abs_output=False,
    gradthreshold=1e-5,
    **kwargs
):

    if isinstance(prior, str):
        prior = pd.read_csv(prior, index_col=0)
    if isinstance(exp, str):
        exp = pd.read_csv(exp, index_col=0)
    if isinstance(model_X, str):
        model_X = pd.read_csv(model_X)
    if isinstance(model_Y, str):
        model_Y = pd.read_csv(model_Y)

    parameter_names = list(prior)
    prior = prior.T
    prior[prior.columns.difference(["Type"])] = prior[
        prior.columns.difference(["Type"])
    ].astype("float")

    target_names = list(exp)
    exp_Y = exp.loc["Values"].astype("float")
    exp_Yerr = exp.loc["Errors"].astype("float")

    model_X = model_X[parameter_names].astype("float")
    model_Y = model_Y[target_names].astype("float")

    """
    we need to normalized the observed points for better emulation
    We need to normalize both the output and input space
    for output space, PCA is also performed for dimension reduction
    """
    if len(initialscale) == 1:
        initialscale = np.full(len(parameter_names), initialscale[0])

    clf = pl.PipeLine(
        [
            ("Normalize", pl.Normalize()),
            ("PCA", pl.PCA(principalcomp, fraction)),
            ("NormalizeNew", pl.Normalize(ignore_X=True)),
            (
                "Emulator",
                pl.MultEmulator(
                    initial_scales=initialscale,
                    initial_nuggets=initialnugget,
                    scales_rate=scalerate,
                    nuggets_rate=nuggetrate,
                    max_steps=maxsteps,
                    save_train_history=True,
                ),
            ),
        ]
    )
    X = model_X.values
    Y = model_Y.values
    clf.Fit(X, Y, tolerance=gradthreshold)

    """
    Write all the training result, together with training points and pipe used to a file
    """
    if not abs_output:
        output_name = os.path.join("result", training_file)
    else:
        output_name = training_file
    pd.set_option("io.hdf.default.format", "table")
    store = pd.HDFStore(output_name, mode="w")
    store["PriorAndConfig"] = prior
    store["Model_X"] = model_X
    store["Model_Y"] = model_Y
    store["Exp_Y"] = exp_Y
    store["Exp_YErr"] = exp_Yerr

    emulator = clf["Emulator"]
    config = {"repr": repr(clf)}
    if 'modelname' in kwargs and kwargs['modelname'] is not None:
        config['name'] = kwargs['modelname']
    store.get_storer("PriorAndConfig").attrs.my_attribute = config

    df_para = []
    for idemu, emulator in enumerate(clf["Emulator"].emulators):
        df_para.append(
            pd.DataFrame(
                emulator.history_para,
                columns=["Nuggets%d" % idemu]
                + [
                    "Scales%d_%d" % (idemu, idinput)
                    for idinput in range(model_X.shape[1])
                ],
            )
        )
    df_para = pd.concat(df_para, axis=1)
    store["ParaHistory"] = df_para

    store.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="This script will choose an optimal set of hyperparameters by minizing loss function")
    parser.add_argument("prior", help="Locatioin of parameter priors")
    parser.add_argument("model_X", help="Location of the model simulation files")
    parser.add_argument("model_Y", help="Location of the model simulation files")
    parser.add_argument("exp", help="Location of the experimental result")
    parser.add_argument(
        "training_file",
        help='Output filename of the optimized emulator. It will be stored under folder "training/"',
    )
    parser.add_argument(
        "-pc",
        "--principalcomp",
        default=4,
        type=int,
        help="Number of principal components used (Default: 3)",
    )
    parser.add_argument(
        "-is",
        "--initialscale",
        default=[0.5],
        type=float,
        nargs="+",
        help="Initial Scale. If array is needed, please enter more than 1 number in this argument",
    )
    parser.add_argument(
        "-in",
        "--initialnugget",
        default=0.01,
        type=float,
        help="Initial Scale. Input must be an array of the same size as number of parameters.",
    )
    parser.add_argument(
        "-sr",
        "--scalerate",
        default=0.01,
        type=float,
        help="Rate at which scale will advance in 1 step (Default: 0.003)",
    )
    parser.add_argument(
        "-nr",
        "--nuggetrate",
        default=0.01,
        type=float,
        help="Rate at which nugget will advance in 1 step (Default: 0.003)",
    )
    parser.add_argument(
        "-ms",
        "--maxsteps",
        default=1000,
        type=int,
        help="Maximum training steps allowed (Default: 1000)",
    )
    parser.add_argument(
        "-fr",
        "--fraction",
        default=None,
        type=float,
        help="Fraction of PCA variance used. Once set it will override pc (Default: None)",
    )
    parser.add_argument(
        "-mn",
        "--modelname",
        help="Name of the model. Used for model comparison (Default: None)",
    )

    args = vars(parser.parse_args())
    #args["Model_X"] = args["Model"]
    #args["Model_Y"] = args["Model"]
    #del args["Model"]

    args['abs_output'] = True
    UseDefaultOutput()
    Training(**args)

    #fig = plt.figure(None, (15, 12), 75)
    #TrainingCurve(fig, args["training_file"])
    #plt.show()
