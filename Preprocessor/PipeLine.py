import autograd.numpy as np
import autograd.scipy.linalg as linalg

class PipeLine:

    
    def __init__(self, pipe_dict):
        self.pipe_dict = pipe_dict

    def Fit(self, data):
        for pipe_name, pipe in self.pipe_dict:
            pipe.Fit(data)
            data = pipe.Transform(data)

    def __repr__(self):
        string = "["
        for pipe_name, pipe in self.pipe_dict:
            string += " %s;" % repr(pipe)
        string +="]"
        return string

    def Transform(self, data):
        for pipe_name, pipe in self.pipe_dict:
            data = pipe.Transform(data) 
        return data

    def TransformCov(self, data_cov):
        for pipe_name, pipe in self.pipe_dict:
            data_cov = pipe.TransformCov(data_cov) 
        return data_cov

    def TransformInv(self, data):
        for pipe_name, pipe in reversed(self.pipe_dict):
            data = pipe.TransformInv(data) 
        return data

    def TransformCovInv(self, data_cov):
        for pipe_name, pipe in reversed(self.pipe_dict):
            data_cov = pipe.TransformCovInv(data_cov) 
        return data_cov

class Identity:
    
    
    def __init__(self):
        pass

    def __repr__(self):
        return "Identity"

    def Fit(self, data):
        pass

    def Transform(self, data):
        return data

    def TransformCov(self, data_cov):
        return data_cov

    def TransformInv(self, data):
        return data
 
    def TransformCovInv(self, data_cov):
        return data_cov


class Normalize:
    
    
    def __init__(self):
        self.sigma = None
        self.mean = None

    def __repr__(self):
        return "Normalize"

    def Fit(self, data):
        self.mean = np.mean(data, axis=0)
        self.sigma = np.std(data, axis=0)

    def Transform(self, data):
        return (data - self.mean) / self.sigma

    def TransformCov(self, data_cov):
        transform = np.diag(np.reciprocal(self.sigma))
        return transform.T.dot(data_cov).dot(transform)

    def TransformInv(self, data):
        return data*self.sigma + self.mean
 
    def TransformCovInv(self, data_cov):
        transform_inv = np.diag(self.sigma)
        return transform_inv.T.dot(data_cov).dot(transform_inv)

class PCA:


    def __init__(self, component):
        self.cov = None
        self.eigval = None
        self.eigvec = None
        self.component = component
        self.reconstruction_error = None

    def __repr__(self):
        return "PCA(%d)" % self.component

    def Fit(self, data):
        self.cov = np.cov(data.T)
        if not self.cov.shape:
            self.eigval = self.cov
            self.eigvec = np.eye(1)
        else:
            self.eigval, self.eigvec = np.linalg.eigh(self.cov)
            idx = self.eigval.argsort()[::-1]
            self.eigval = self.eigval[idx]
            self.eigvec = self.eigvec[:,idx]
            assert self.component <= data.shape[1], "number of components cannot exceed number of variables"
            self.reconstruction_error = np.mean(self.eigval[self.component:])
            self.eigval = self.eigval[0:self.component]
            self.eigvec = self.eigvec[:, 0:self.component]

    def Transform(self, data):
        return np.dot(data, self.eigvec)

    def TransformCov(self, data_cov):
        return self.eigvec.T.dot(data_cov).dot(self.eigvec)

    def TransformInv(self, data):
        return np.dot(data, self.eigvec.T);

    def TransformCovInv(self, data_cov):
        return self.eigvec.dot(data_cov).dot(self.eigvec.T) + self.reconstruction_error*np.eye(self.cov.shape[0])
