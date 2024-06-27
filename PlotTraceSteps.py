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

    gs = fig.add_gridspec(2, 3, width_ratios=[5, 1, 1])
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[:, 1]), fig.add_subplot(gs[:, 2])]
        
    threadCP, = np.where(trace.index == 0)
    threadCP = np.append(threadCP, trace.shape[0])
    maxlags = None#threadCP[1]
    threadID = 0
    def drawTrace(pname, threadID=0):
        axes[0].clear()
        axes[1].clear()
        tr = trace[pname][threadCP[threadID]:threadCP[threadID+1]]
        id = trace.index[threadCP[threadID]:threadCP[threadID+1]]
        axes[0].plot(id, tr)
        window_size = trace_running_ave_size
        if window_size > 1:
            window_size = min(window_size, len(tr))
            axes[0].plot(np.arange(window_size-1, len(tr)), running_average(tr, window_size))
        axes[0].set_ylabel(pname)
        axes[0].set_xlabel('trace')
        axes[1].acorr(trace[pname][threadCP[threadID]:threadCP[threadID+1]],  detrend=mlab.detrend_mean, maxlags=maxlags)
        axes[1].set_xlabel('Lags')
        axes[1].set_ylabel(pname + ' acorr')
        fig.canvas.draw()

    drawTrace(par_name[0])
    radios = RadioButtons(axes[2], tuple(par_name))
    radios_thread = RadioButtons(axes[3], tuple(range(len(threadCP)-1)))

    #keep radio button circlar
    for ax, radio in zip(axes[2:4], [radios, radios_thread]):
        rpos = ax.get_position().get_points()
        fh = fig.get_figheight()
        fw = fig.get_figwidth()
        rscale = (rpos[:,1].ptp() / rpos[:,0].ptp()) * (fh / fw)
        for circ in radio.circles:
            circ.height /= rscale

    def drawTraceObs(pname_clicked):
        drawTrace(pname_clicked, int(radios_thread.value_selected))

    def drawTraceThread(threadID_clicked):
        drawTrace(radios.value_selected, int(threadID_clicked))

    radios.on_clicked(drawTraceObs)
    radios_thread.on_clicked(drawTraceThread)


    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()
    fig.radios = radios
    fig.radios_thread = radios_thread

    return fig, axes

if __name__ == '__main__':
    PlotTrace('result/imqmd_09152023_replyToRef/OnlySn132V1')
    plt.show()
