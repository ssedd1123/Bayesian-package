from multiprocessing import Pool
import cPickle as pickle
import pandas as pd
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.font_manager as font_manager

font_dirs = ['/projects/hira/tsangc/Polarizability/fonts', ]
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
font_list = font_manager.createFontList(font_files)
font_manager.fontManager.ttflist.extend(font_list)


import matplotlib.pylab as pylab
params = {#'figure.autolayout': True,
          'figure.figsize': [16, 9],
          'legend.fontsize': 25,
          'legend.framealpha': 0,
          'lines.linewidth': 2,
          'font.family': 'serif',
          'font.serif': 'CMU Typewriter Text', 
          #'text.usetex': True,
          #'mathtext.fontset': 'cm',
          'mathtext.rm': 'serif',
          'mathtext.default': 'regular',
          'patch.linewidth': 0,
          'axes.linewidth': 4,
          'axes.labelsize': 40,
          #'axes.titlesize': 5,
          'axes.labelpad': 0,
          'xtick.labelsize':30,
          'ytick.labelsize':30,
          'xtick.major.size': 20,
          'xtick.minor.size': 5,
          'xtick.major.width': 4,
          'xtick.minor.width': 2,
          'xtick.direction': 'in',
          'ytick.major.size': 20,
          'ytick.minor.size': 5,
          'ytick.major.width': 2,
          'ytick.minor.width': 1.5,
          'ytick.direction': 'in',
          'figure.facecolor': 'white'}
pylab.rcParams.update(params)

import math
import glob

from Emulator.Emulator import *
from Preprocessor.PipeLine import *
from Utilities.Utilities import PlotTrace, GenerateTrace

if len(sys.argv) != 2:
    #print('Use this script by entering: python %s Iutput_name' % (sys.argv[0]))
    #sys.exit()
    list_of_files = glob.glob('result/*')
    filename = max(list_of_files, key=os.path.getctime)
else:
    filename = sys.argv[1]

print('Plotting correlation for %s' % filename)
with open(filename, 'rb') as buff:
    data = pickle.load(buff)

trace = data['trace']
par_name = data['data']['data'].par_name
prior = data['prior']

fig, axes2d = PlotTrace(trace, par_name, prior)
if('real_para' in data):
    exp_para = data['real_para']
    
    for i, row in enumerate(axes2d):
        for j, cell in enumerate(row):
            if i == j:
                cell.axvline(exp_para[i], color='r')
            else:
                cell.plot(exp_para[j], exp_para[i], 'red', marker='v', markersize=10)
"""
exp_para = [29,90,0.9,-0.2]
    
for i, row in enumerate(axes2d):
    for j, cell in enumerate(row):
        if i == j:
            cell.axvline(exp_para[i], color='r')
        else:
            cell.plot(exp_para[j], exp_para[i], 'red', marker='v', markersize=10)
"""
plt.show()
