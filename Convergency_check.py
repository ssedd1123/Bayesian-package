import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from Emulator import *

def PlotMarginalLikelihood(emulator_function, scales_min=1e-2, scales_max=2, scales_num=40, nuggets_min=1e-2, nuggets_max=20, nuggets_num=40):
    # log scale for plotting everything
    scales = np.logspace(np.log10(scales_min), np.log10(scales_max), num=scales_num, base=10)
    nuggets = np.logspace(np.log10(nuggets_min), np.log10(nuggets_max), num=nuggets_num, base=10)

    # create log scale meshgrid for filling array of log likelihood
    scalesv, nuggetsv = np.meshgrid(scales, nuggets)

    log_likelihood = np.array([emulator_function(scale, nugget) for sv, nv in zip(scalesv, nuggetsv) for scale, nugget in zip(sv, nv)])

    # reshape such that z array matches the shape of the meshtrid
    z = log_likelihood.reshape(scalesv.shape[0], -1)
    z_range = np.amax(z) - np.amin(z)
    threshold = np.amax(z) - 0.01*z_range
    threshold_indices = z < threshold
    z[threshold_indices] = threshold
    plot = plt.contour(scalesv, nuggetsv, z, 10)
    plt.yscale('log')
    plt.xscale('log')
    plt.colorbar(plot)
    plt.show()
    
