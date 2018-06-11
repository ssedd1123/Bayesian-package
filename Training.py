import os
import sys
if sys.version_info > (3, 0):
    import pickle
else:
    import cPickle as pickle
import autograd.numpy as np
import pandas as pd
import argparse

from Emulator.Emulator import EmulatorMaster, RBF
from Preprocessor.PipeLine import PipeLine, Normalize, PCA
from DataReader.DataLoader import DataLoader


if __name__=='__main__':

    parser = argparse.ArgumentParser(description='This script will choose an optimal set of hyperparameters by minizing loss function')
    parser.add_argument('Prior', help='Locatioin of parameter priors')
    parser.add_argument('ModelData', help='Location of the model simulation files')
    parser.add_argument('ExpData', help='Location of the experimental result')
    parser.add_argument('Training_name', help='Output filename of the optimized emulator. It will be stored under folder "training/"')
    parser.add_argument('-cf', '--covariancefunc', default='ARD', help='Your choice of different covariance function (Default: ARD)')
    parser.add_argument('-pc', '--principalcomp', default=3, type=int, help='Number of principal components used (Default: 3)')
    parser.add_argument('-is', '--initialscale', default=[0.5], type=float, nargs='+', help='Initial Scale. If array is needed, please enter more than 1 number in this argument')
    parser.add_argument('-in', '--initialnugget', default=1, type=float, help='Initial Scale. Input must be an array of the same size as number of parameters.')
    parser.add_argument('-sr', '--scalerate', default=0.003, type=float, help='Rate at which scale will advance in 1 step (Default: 0.003)')
    parser.add_argument('-nr', '--nuggetrate', default=0.003, type=float, help='Rate at which nugget will advance in 1 step (Default: 0.003)')
    parser.add_argument('-ms', '--maxsteps', default=1000, type=int, help='Maximum training steps allowed (Default: 1000)')
    args = vars(parser.parse_args())
    
    
    data = DataLoader(args['Prior'], args['ModelData'], args['ExpData'])
    
    """
    we need to normalized the observed points for better emulation
    We need to normalize both the output and input space
    for output space, PCA is also performed for dimension reduction
    """
    output_pipe = PipeLine([('Normalize', Normalize()), ('PCA', PCA(args['principalcomp'])), ('Normalized', Normalize())])
    input_pipe = Normalize()
    
    emulator = EmulatorMaster(data.sim_para, data.sim_data, input_pipe, output_pipe)
    if args['covariancefunc'] == 'RBF':
        emulator.SetCovariance(RBF)
    
    if len(args['initialscale']) == 1:
        initial_scale = np.full(len(data.par_name), args['initialscale'][0])
    elif len(args['initialscale']) != len(data.par_name):
        print('Initial scale must have the same dimension as number of parameters. Abort!')
        sys.exit()
    else:
        initial_scale = np.array(args['initialscale'])
    
    scales, nuggets = emulator.Train(initial_scale, 
                                     args['initialnugget'], 
                                     max_step=args['maxsteps'], 
                                     scales_rate=args['scalerate'], 
                                     nuggets_rate=args['nuggetrate'])
    
    """
    Write all the training result, together with training points and pipe used to a file
    """
    with open(os.path.join('training', '%s.pkl' % args['Training_name']), 'wb') as buff:
        pickle.dump({'emulator': emulator, 'data': data,
                     'scales': scales, 'nuggets': nuggets}, buff)
