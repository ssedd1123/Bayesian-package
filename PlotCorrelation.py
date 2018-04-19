from multiprocessing import Pool
import cPickle as pickle
import pandas as pd
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import math

from Emulator.Emulator import *
from Preprocessor.PipeLine import *
from Utilities.Utilities import PlotTrace, GenerateTrace

if len(sys.argv) != 2:
    print('Use this script by entering: python %s Iutput_name' % (sys.argv[0]))
    sys.exit()

with open(sys.argv[1], 'rb') as buff:
    data = pickle.load(buff)

trace = data['trace']
par_name = data['data']['data'].par_name
prior = data['prior']

fig, axes2d = PlotTrace(trace, par_name, prior)
exp_para = [32,70,0.75,0.2]

for i, row in enumerate(axes2d):
    for j, cell in enumerate(row):
        if i == j:
            cell.axvline(exp_para[i], color='r')
        else:
            cell.plot(exp_para[j], exp_para[i], 'black', marker='v', markersize=10)

plt.show()
plt.show()