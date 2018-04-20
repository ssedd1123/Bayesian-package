from multiprocessing import Pool
import cPickle as pickle
import pandas as pd
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
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
                cell.plot(exp_para[j], exp_para[i], 'black', marker='v', markersize=10)

plt.show()
