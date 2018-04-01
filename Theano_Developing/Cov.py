import theano
import theano.tensor as T
import numpy as np

def Cov(y1):
    y1_mean = T.mean(y1, axis=1)
    y1_centered = y1 - y1_mean.dimshuffle(0, 'x')
    return T.sum(y1_centered.dimshuffle(1, 0, 'x')*y1_centered.dimshuffle(1, 'x', 0), axis=0)/(y1.shape[1] - 1)

x = theano.shared(np.array([[0, 2], [1, 1], [2, 0], [9,12]]))
y = theano.function([], Cov(x))
y1_mean = T.mean(x, axis=1)
y1_centered = x - y1_mean.dimshuffle(0, 'x')
y2 = y1_centered.dimshuffle(1, 0, 'x')*y1_centered.dimshuffle(1, 'x', 0)
y1 = theano.function([], [y1_centered, y2])
print(y())
print(y1())
print('cov', np.cov(np.array([[0, 2], [1, 1], [2, 0], [9,12]])))
