import os
import sys
if sys.version_info > (3, 0):
    import pickle
else:
    import cPickle as pickle
import autograd.numpy as np
import pandas as pd
import argparse

import Preprocessor.PipeLine as pl

def Training(prior, model_X, model_Y, exp, training_file,
             principalcomp=None, fraction=0.99,
             initialscale=[1], initialnugget=0.1, 
             scalerate=0.01, nuggetrate=0.01, maxsteps=1000, abs_output=False):

    if type(prior) is str:
        prior = pd.read_csv(prior, index_col=0)
    if type(exp) is str:
        exp = pd.read_csv(exp, index_col=0)
    if type(model_X) is str:
        model_X = pd.read_csv(model_X)
    if type(model_Y) is str:
        model_Y = pd.read_csv(model_Y)

    parameter_names = list(prior)   
    prior = prior.T    
    prior[prior.columns.difference(['Type'])] = prior[prior.columns.difference(['Type'])].astype('float')

    target_names = list(exp)
    exp_Y = exp.loc['Values'].astype('float')
    exp_Yerr = exp.loc['Errors'].astype('float')

    model_X = model_X[parameter_names].astype('float')
    model_Y = model_Y[target_names].astype('float')


    """
    we need to normalized the observed points for better emulation
    We need to normalize both the output and input space
    for output space, PCA is also performed for dimension reduction
    """
    if len(initialscale) == 1:
        initialscale = np.full(len(parameter_names), initialscale[0])

    clf = pl.PipeLine([('Normalize', pl.Normalize()), 
                       ('PCA', pl.PCA(principalcomp, fraction)), 
                       ('NormalizeNew', pl.Normalize(ignore_X=True)),
                       ('Emulator', pl.MultEmulator(initial_scales=initialscale,
                                                    initial_nuggets=initialnugget,
                                                    scales_rate=scalerate,
                                                    nuggets_rate=nuggetrate,
                                                    max_steps=maxsteps))])
    
    clf.Fit(model_X.values, model_Y.values) 

    """
    Write all the training result, together with training points and pipe used to a file
    """
    if not abs_output:
        output_name = os.path.join('result', training_file)
    else:
        output_name = training_file
    pd.set_option('io.hdf.default.format', 'table')
    store = pd.HDFStore(output_name, mode='w')
    store['PriorAndConfig'] = prior
    store['Model_X'] = model_X
    store['Model_Y'] = model_Y
    store['Exp_Y'] = exp_Y
    store['Exp_YErr'] = exp_Yerr
  
    emulator = clf.named_steps[-1][1]
    config = {'repr': repr(clf)}
    store.get_storer('PriorAndConfig').attrs.my_attribute = config

    store.close()


if __name__=='__main__':

    parser = argparse.ArgumentParser(description='This script will choose an optimal set of hyperparameters by minizing loss function')
    parser.add_argument('prior', help='Locatioin of parameter priors')
    parser.add_argument('model', help='Location of the model simulation files')
    parser.add_argument('exp', help='Location of the experimental result')
    parser.add_argument('training_file', help='Output filename of the optimized emulator. It will be stored under folder "training/"')
    parser.add_argument('-pc', '--principalcomp', default=4, type=int, help='Number of principal components used (Default: 3)')
    parser.add_argument('-is', '--initialscale', default=[0.5], type=float, nargs='+', help='Initial Scale. If array is needed, please enter more than 1 number in this argument')
    parser.add_argument('-in', '--initialnugget', default=0.01, type=float, help='Initial Scale. Input must be an array of the same size as number of parameters.')
    parser.add_argument('-sr', '--scalerate', default=0.01, type=float, help='Rate at which scale will advance in 1 step (Default: 0.003)')
    parser.add_argument('-nr', '--nuggetrate', default=0.01, type=float, help='Rate at which nugget will advance in 1 step (Default: 0.003)')
    parser.add_argument('-ms', '--maxsteps', default=1000, type=int, help='Maximum training steps allowed (Default: 1000)')
    parser.add_argument('-fr', '--fraction', default=None, type=float, help='Fraction of PCA variance used. Once set it will override pc (Default: None)')
    args = vars(parser.parse_args())
    args['Model_X'] = args['Model']
    args['Model_Y'] = args['Model']
    del args['Model']
    
    Training(**args)
    