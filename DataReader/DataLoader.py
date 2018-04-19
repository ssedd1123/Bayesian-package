import numpy as np
import pandas as pd

class DataLoader:


    def __init__(self, prior_filename, model_filename, exp_filename):
        """
        Loading prior of each variables
        will be used to load the parameter names
        such that when model data is read
        it can tell which one is input parameter and which one is output 
        """
        # load the prior
        self.prior = pd.read_csv(prior_filename)
        # load the name of the variables in the prior
        self.par_name = list(self.prior)

        """
        Loading experiment output data
        """
        df = pd.read_csv(model_filename)
        self.sim_error = df[list(df.filter(regex='_Error'))].as_matrix()
        df_no_error = df[df.columns.drop(list(df.filter(regex='_Error')))]
        self.sim_data = df_no_error.drop(self.par_name, axis=1).as_matrix()
        self.sim_para = df_no_error[self.par_name].as_matrix() 


        """
        Loading model result
        """
        # rad the experiment result
        df = pd.read_csv(exp_filename)
        # load the experimental error
        error = df[list(df.filter(regex='_Error'))].as_matrix().flatten()
        self.exp_result = df[df.columns.drop(list(df.filter(regex='_Error')))].as_matrix().flatten()
        self.exp_cov = np.square(np.diag(error))
