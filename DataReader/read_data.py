import pandas as pd
import os
import re
import sys

if len(sys.argv) != 3:
    print('Use this script by entering: python %s ModelFolder Output_name' % (sys.argv[0]))
    sys.exit()

results_list = []
results_err_list = []
parameters_list = []

for directory in os.listdir(sys.argv[1]):
    if re.match('run[0-9]+', directory):
        print('Collecting data from %s' % directory)
        results = pd.read_csv(os.path.join(sys.argv[1], directory, 'results.dat'), header=None,
                              index_col=0, delim_whitespace=True, usecols=[0, 1]).T
        results_err = pd.read_csv(os.path.join(sys.argv[1], directory, 'results.dat'), header=None,
                              index_col=0, delim_whitespace=True, usecols=[0, 2]).T
        parameters = pd.read_csv(os.path.join(sys.argv[1], directory, 'parameters.dat'), header=None,
                              index_col=0, delim_whitespace=True).T
        results_list.append(results)
        results_err_list.append(results_err)
        parameters_list.append(parameters)
        

results = pd.concat(results_list, ignore_index=True)
results_err = pd.concat(results_err_list, ignore_index=True)
results_err.columns = [str(col) + '_Error' for col in results_err.columns]
parameters = pd.concat(parameters_list, ignore_index=True)

data = pd.concat([parameters, results, results_err], axis=1)        
print(data)
data.to_csv('%s.csv' % sys.argv[2], index=False)
