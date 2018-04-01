import theano
import theano.tensor as tt

def Cov(y1):
    y1_mean = tt.mean(y1, axis=1)
    y1_centered = y1 - y1_mean.dimshuffle(0, 'x')
    return tt.sum(y1_centered.dimshuffle(1, 0, 'x')*y1_centered.dimshuffle(1, 'x', 0), axis=0)/(y1.shape[1] - 1)

class PipeLineT:

    
    def __init__(self, pipe_dict):
        self.pipe_dict = pipe_dict

    def Fit(self, data):
        for pipe_name, pipe in self.pipe_dict:
            pipe.Fit(data)
            data = pipe.Transform(data)

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



class NormalizeT:
    
    
    def __init__(self):
        self.sigma = None
        self.mean = None

    def Fit(self, data):
        self.mean = tt.mean(data, axis=0)
        self.sigma = tt.std(data, axis=0)

    def Transform(self, data):
        return (data - self.mean) / self.sigma

    def TransformCov(self, data_cov):
        transform = tt.nlinalg.diag(tt.inv(self.sigma))
        return tt.dot(transform.T, tt.dot(data_cov, transform))

    def TransformInv(self, data):
        return data*self.sigma + self.mean
 
    def TransformCovInv(self, data_cov):
        transform_inv = tt.nlinalg.diag(self.sigma)
        return tt.dot(transform_inv.T, tt.dot(data_cov, transform_inv))

class PCAT:


    def __init__(self, component):
        self.cov = None
        self.eigval = None
        self.eigvec = None
        self.component = component

    def Fit(self, data):
        self.cov = Cov(data.T)
        self.eigval, self.eigvec = tt.nlinalg.eig(self.cov)
        self.eigval = self.eigval[-self.component:]
        self.eigvec = self.eigvec[:, -self.component:]

    def Transform(self, data):
        return tt.dot(data, self.eigvec)

    def TransformCov(self, data_cov):
        return tt.dot(self.eigvec.T, tt.dot(data_cov, self.eigvec))

    def TransformInv(self, data):
        return tt.dot(data, self.eigvec.T);

    def TransformCovInv(self, data_cov):
        return tt.dot(self.eigvec, tt.dot(data_cov, self.eigvec.T))
