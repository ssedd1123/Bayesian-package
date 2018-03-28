import pymc
import numpy as np
import matplotlib.pyplot as plt
import math

from Emulator import *
from PipeLine import *
from Convergency_check import PlotMarginalLikelihood

#trainning_x = np.arange(1, 4, 0.3).reshape(-1,1)

x = np.arange(-3.9, 4, 0.5)
y = np.arange(-3.9, 4, 0.5)
xv, yv = np.meshgrid(x, y)
xy = np.column_stack((xv.reshape(-1, 1), yv.reshape(-1, 1)))
r = np.sqrt(xv*xv + yv*yv)
z = np.sin(2*r)/r
"""
xy = np.arange(0.5, 4, 0.2).reshape(-1,1)
z = np.sinh(xy) * np.sin(3*xy)
"""
#z = xv*yv
z = z + (np.random.rand(*z.shape) - 0.5)
#plt.plot(xy, z, 'ro')
plt.pcolor(xv, yv, z)
plt.show()

# we need to normalized the observed points for better emulation
pipe = PipeLine([('Normalize', Normalize())])
pipe.Fit(z.reshape(-1,1))
z_emulate = pipe.Transform(z.reshape(-1,1))

# setting up emulator for training
emulator = Emulator(xy, z_emulate)
emulator.SetCovariance(squared_exponential)

# visual confirmation on the distribution of marginal likelihood
PlotMarginalLikelihood(emulator.MarginalLikelihood, scales_min=1e-1, scales_max=10, nuggets_min=1e-3, nuggets_max=1)

# training in progress with gradient descent
gd = GradientDescentForEmulator(0.001, 0.01)
# trainning with marginallikelihood instead of LOOCV 
gd.SetFunc(emulator.MarginalLikelihood)
history = gd.Descent(0.5, 1, 100)
print(history)

# use the last trained scales and nuggets
emulator.SetScales(history[-1, 0])
emulator.SetNuggets(history[-1, 1])
emulator.StartUp()

# NOTE: the linear regression model we're trying to solve for is
# given by:
# y = b0 + b1(x) + error
# where b0 is the intercept term, b1 is the slope, and error is
# the error
 
# model the intercept/slope terms of our model as
# normal random variables with comically large variances
b0 = pymc.Uniform('b0', -4, 4)
b1 = pymc.Uniform('b1', -4, 4)
 
# "model" the observed x values as a normal random variable
# in reality, because x is observed, it doesn't actually matter
# how we choose to model x -- PyMC isn't going to change x's values
 
# this is the heart of our model: given our b0, b1 and our x observations, we want
# to predict y

@pymc.stochastic(observed=True)
def emulator_result(value=0.5, x=b0, y=b1):
    value = pipe.Transform(value)
    (mean, var) = emulator.Emulate(np.array([x, y]).reshape(1,-1))
    return - (value - mean)**2 / (2*var) - np.log(var) - 0.5*np.log(2*np.pi)
 
# "model" the observed y values: again, I reiterate that PyMC treats y as
# evidence -- as fixed; it's going to use this as evidence in updating our belief
# about the "unobserved" parameters (b0, b1, and err), which are the
# things we're interested in inferring after all
 
# put everything we've modeled into a PyMC model
y = emulator_result
model = pymc.Model([y, b0, b1])

"""
xy = np.arange(0.5, 4, 0.01).reshape(-1,1)
(z, z_err) = emulator.Emulate(xy)
z_err = np.array([pipe.TransformCovInv(z_)[0, 0] for z_ in np.diag(z_err)]).reshape(-1,1)
z = pipe.TransformInv(z)
plt.plot(xy, z)
plt.plot(xy, (z - z_err), color='red')
plt.plot(xy, (z + z_err), color='red')
plt.show()
"""
x = np.arange(-4, 4, 0.1)
y = np.arange(-4, 4, 0.1)
xv, yv = np.meshgrid(x, y)
xy = np.column_stack((xv.reshape(-1, 1), yv.reshape(-1, 1)))
(z, z_err) = emulator.Emulate(xy)
plt.pcolor(xv, yv, z.reshape(xv.shape[0], -1))
plt.show()

# prepare for MCMC
mcmc = pymc.MCMC(model)
 
# sample from our posterior distribution 50,000 times, but
# throw the first 20,000 samples out to ensure that we're only
# sampling from our steady-state posterior distribution
mcmc.sample(100000, 20000)
plt.hist(mcmc.trace('b0')[:], bins=500)
plt.show()

plt.hist2d(mcmc.trace('b0')[:], mcmc.trace('b1')[:], bins=40)
plt.colorbar()
plt.show()
