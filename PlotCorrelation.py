import argparse
import glob
import math
import os
import pickle as pickle
import sys
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Preprocessor.PipeLine import *
from Utilities.Utilities import PlotTrace

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Plot Correlation of the Bayesian trace"
    )
    parser.add_argument(
        "-r",
        "--result",
        help="Location of the resultant file (output from StatParallel.py/StatLOO.py). If this is not supplied it will choose the latest created file",
    )
    parser.add_argument(
        "-p",
        "--points",
        action="store_true",
        help="Plot the exact result (if exist). It will be used if your result comes from StatLOO.py",
    )
    parser.add_argument(
        "-er",
        "--expresult",
        type=float,
        nargs="+",
        help="Experimental result to be overlay on top. Use this when you have validation trace but it does not come from StatLOO.py",
    )
    args = vars(parser.parse_args())

    if args["result"] is None:
        list_of_files = glob.glob("training/*")
        filename = max(list_of_files, key=os.path.getctime)
    else:
        filename = args["result"]

    print("Plotting correlation for %s" % filename)

    fig = plt.figure(figsize=(10, 7))#, 75)
    fig.clf()
    PlotTrace(filename, fig, mark_point={'S0': 31, 'L': 60, 'ms': 0.8, 'mv': 0.85, 'etac': 0.1}, sigma=2, bins=50)
    plt.show()
