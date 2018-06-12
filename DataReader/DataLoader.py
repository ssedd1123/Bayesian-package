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
        self.sim_error = df[list(df.filter(regex='_Error'))].values
       
        df_no_error = df[df.columns.drop(list(df.filter(regex='_Error')))]
        self.var_name = list(df_no_error.drop(self.par_name, axis=1))
        self.sim_data = df_no_error.drop(self.par_name, axis=1).values
        self.sim_para = df_no_error[self.par_name].values 


        """
        Loading model result
        """
        # rad the experiment result
        df = pd.read_csv(exp_filename)
        # load the experimental error
        error = df[list(df.filter(regex='_Error'))].values.flatten()
        self.exp_result = df[df.columns.drop(list(df.filter(regex='_Error')))].values.flatten()
        self.exp_cov = np.square(np.diag(error))

        """
        Save the filenames so when it is printed, we know which files the data corresponds to
        """
        self.prior_filename = prior_filename
        self.model_filename = model_filename
        self.exp_filename = exp_filename

    def __repr__(self):
        return "Prior: %s\n%s\n\nModel: %s\nExperimental data: %s\n" % (self.prior_filename, repr(self.prior), self.model_filename, self.exp_filename)

    def ChangeExp(self, exp_filename):
        # rad the experiment result
        df = pd.read_csv(exp_filename)
        # load the experimental error
        error = df[list(df.filter(regex='_Error'))].values.flatten()
        self.exp_result = df[df.columns.drop(list(df.filter(regex='_Error')))].values.flatten()
        self.exp_cov = np.square(np.diag(error))

        self.exp_filename = exp_filename

    def ChangePrior(self, prior_filename):
        self.prior = pd.read_csv(prior_filename)
        self.prior_filename = prior_filename
        # check and see if number of variables is changed
        # if so the new prior cannot be accepted. You must start a completely new training
        if(set(self.par_name) != set(list(self.prior))):
            print('Number of parameters in the new prior changes! You must start a completely new training for it. ')

            
