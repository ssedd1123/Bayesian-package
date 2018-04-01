import theano
from PipeLineTheano import *
from PipeLine import *
import theano.tensor as tt
import numpy as np

data = np.array([[1,2],[2,3],[3,4],[6,-9]])
cov = np.array([[2.,-1.],[-1.,2.]])
normal = PCA(2)
normal.Fit(data)
tra = normal.Transform(data)
print(tra)
print(normal.TransformCovInv(normal.TransformCov(cov)))
print(normal.TransformInv(tra))

data = theano.shared(data)
cov = theano.shared(cov)
normalT = PCAT(2)
normalT.Fit(data)
tra = normalT.Transform(data)
trainv = normalT.TransformInv(tra)
covtran = normalT.TransformCovInv(normalT.TransformCov(cov))
func = theano.function([], [tra, covtran, trainv])
print(func())
