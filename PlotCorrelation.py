from multiprocessing import Pool
import cPickle as pickle
import pandas as pd
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse

import math
import glob

from Emulator.Emulator import *
from Preprocessor.PipeLine import *
from Utilities.Utilities import PlotTrace, GenerateTrace

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Plot Correlation of the Bayesian trace')
    parser.add_argument('-r', '--result', help='Location of the resultant file (output from StatParallel.py/StatLOO.py). If this is not supplied it will choose the latest created file')
    parser.add_argument('-p', '--points', action='store_true', help='Plot the exact result (if exist). It will be used if your result comes from StatLOO.py')
    parser.add_argument('-er', '--expresult', type=float, nargs='+', help='Experimental result to be overlay on top. Use this when you have validation trace but it does not come from StatLOO.py')
    args = vars(parser.parse_args())
    
    if args['result'] is None:
        list_of_files = glob.glob('result/*')
        filename = max(list_of_files, key=os.path.getctime)
    else:
        filename = args['result']
    
    print('Plotting correlation for %s' % filename)
    with open(filename, 'rb') as buff:
        data = pickle.load(buff)
    
    trace = data['trace']
    par_name = data['data']['data'].par_name
    prior = data['prior']
    
    fig, axes2d = PlotTrace(trace, par_name, prior)
    if args['points'] and ('real_para' not in data):
        print('Warning: experimental data points not found in trace result. You may want to use -er.')
    
    if (args['points'] and ('real_para' in data)) or (args['expresult'] is not None):
        if(args['expresult'] is not None):
            exp_para = args['expresult']
        else:
            exp_para = data['real_para'] 
        
        for i, row in enumerate(axes2d):
            for j, cell in enumerate(row):
                if i == j:
                    cell.axvline(exp_para[i], color='r')
                else:
                    cell.plot(exp_para[j], exp_para[i], 'red', marker='v', markersize=10)
    """
    exp_para = [29,90,0.9,-0.2]
    """
    plt.show()
