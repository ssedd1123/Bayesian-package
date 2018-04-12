import matplotlib.pyplot as plt
import numpy as np

# input is the pymc3 trace and list of parameters
def PlotTrace(trace, par_name, prior):
    # plot the result in a nice matrix of histograms
    num_par = len(par_name)
    graph_num = 1
    fig, axes2d = plt.subplots(num_par, num_par)

    for i, row in enumerate(axes2d):
        for j, cell in enumerate(row):
            namex = par_name[j]
            namey = par_name[i]
            if namex == namey:
                cell.hist(trace[namex], bins = 50, range=np.array([prior[namex][0], prior[namex][1]]))
                cell.set_xlim([prior[namex][0], prior[namex][1]])
            else:
                im = cell.hist2d(trace[namex], trace[namey], bins=50, range=np.array([(prior[namex][0], prior[namex][1]),(prior[namey][0], prior[namey][1])]))
                cell.set_xlim([prior[namex][0], prior[namex][1]])
                cell.set_ylim([prior[namey][0], prior[namey][1]])
                fig.colorbar(im[3], ax=cell)
            if i == num_par - 1:
                cell.set_xlabel(namex, fontsize=30)
            if j == 0:
                cell.set_ylabel(namey, fontsize=30)

    plt.show()

class DataLoader:


    def __init___(self, prior_filename, model_filename, exp_filename):
        """
        Loading prior of each variables
        will be used to load the parameter names
        such that when model data is read
        it can tell which one is input parameter and which one is output 
        """
        # load the prior
        self.prior = pd.read_csv(prior_filename)
        # load the name of the variables in the prior
        self.par_name = list(prior)

        """
        Loading experiment output data
        """
        df = pd.read_csv(model_filename)
        self.sim_error = df[list(df.filter(regex='_Error'))].as_matrix()
        df_no_error = df[df.columns.drop(list(df.filter(regex='_Error')))]
        self.sim_data = df_no_error.drop(par_name, axis=1).as_matrix()
        self.sim_para = df_no_error[par_name].as_matrix() 


        """
        Loading model result
        """
        # rad the experiment result
        df = pd.read_csv(exp_filename)
        # load the experimental error
        error = df[list(df.filter(regex='_Error'))].as_matrix().flatten()
        self.exp_result = df[df.columns.drop(list(df.filter(regex='_Error')))].as_matrix().flatten()
        self.exp_cov = np.square(np.diag(error))
