import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pymc3 as pm


model = pm.Model()
with model:
    mu1 = pm.Normal("mu1", mu=0, sd=1, shape=10)
    step = pm.NUTS()
    trace = pm.sample(2000, tune=1000, init='advi', step=step, cores=2)
