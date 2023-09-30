import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pylab import acorr, mlab
from matplotlib.widgets import Button
from matplotlib.widgets import RadioButtons

def PlotTrace(trace_filename, fig=None):
    # plot the result in a nice matrix of histograms
    id_to_model = None
    with pd.HDFStore(trace_filename, "r") as store:
        trace = store["trace"]
        prior = store["PriorAndConfig"]

    num_par = prior.shape[0]
    par_name = prior.index

    if fig is None:
        fig = plt.figure()

    gs = fig.add_gridspec(2, 2, width_ratios=[5, 1])
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[:, 1])]
        
    threadCP, = np.where(trace.index == 0)
    threadCP = np.append(threadCP, trace.shape[0])
    maxlags = None#threadCP[1]#min(1000, threadCP[1])
    def drawTrace(pname):
        axes[0].clear()
        axes[1].clear()
        axes[0].plot(trace.index[threadCP[0]:threadCP[1]], trace[pname][threadCP[0]:threadCP[1]])
        axes[0].set_ylabel(pname)
        axes[0].set_xlabel('trace')
        axes[1].acorr(trace[pname][threadCP[0]:threadCP[1]],  detrend=mlab.detrend_mean, maxlags=maxlags)
        axes[1].set_xlabel('Lags')
        axes[1].set_ylabel(pname + ' acorr')
        fig.canvas.draw()


    drawTrace(par_name[0])
    radios = RadioButtons(axes[2], tuple(par_name))

    #keep radio button circlar
    rpos = axes[2].get_position().get_points()
    fh = fig.get_figheight()
    fw = fig.get_figwidth()
    rscale = (rpos[:,1].ptp() / rpos[:,0].ptp()) * (fh / fw)
    for circ in radios.circles:
        circ.height /= rscale
    radios.on_clicked(drawTrace)


    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()
    fig.radios = radios

    return fig, axes

if __name__ == '__main__':
    PlotTrace('result/imqmd_09152023_replyToRef/OnlySn132V1')
    plt.show()
