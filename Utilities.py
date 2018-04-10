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
