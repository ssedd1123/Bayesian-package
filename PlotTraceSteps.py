import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pylab import acorr, mlab
from matplotlib.widgets import Button
from matplotlib.widgets import RadioButtons

def running_average(data, window_size):
    # Define a simple averaging kernel
    kernel = np.ones(window_size) / window_size

    # Use numpy's convolve function to compute the running average
    running_avg = np.convolve(data, kernel, mode='valid')

    return running_avg

def PlotTraceSteps(trace_filename, fig=None, trace_running_ave_size=0):
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
        tr = trace[pname][threadCP[0]:threadCP[1]]
        id = trace.index[threadCP[0]:threadCP[1]]
        axes[0].plot(id, tr)
        window_size = trace_running_ave_size
        if window_size > 1:
            window_size = min(window_size, len(tr))
            axes[0].plot(np.arange(window_size-1, len(tr)), running_average(tr, window_size))
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
