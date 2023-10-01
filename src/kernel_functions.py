import numpy as np
from sklearn.kernel_approximation import RBFSampler

def class_kernel_mean_embedding(fm, y, divide_by_L2_norm=False):
    
        if len(y.shape) > 1:
            y = y.reshape(-1)
    
        y_unique = np.unique(y)
        w_s = np.zeros((len(y_unique), fm.shape[1]))
    
        for i, c in enumerate(y_unique):
            fm_c = fm[y == c, :]
            if divide_by_L2_norm:
                s = np.sum(fm_c, axis=0)
                m = np.linalg.norm(s)
                w_s[i] = s / m
            else:
                w_s[i] = np.mean(fm_c, axis=0)
        
        return w_s

class RFF:

    def __init__(self, d, sigma):
        self.sigma = sigma
        self.d = d
    
    def forward(self, x):
        return self.F.transform(x)
    
    def fit(self, x):
        self.F = RBFSampler(gamma=1.0/(2*self.sigma**2), n_components=self.d)
        self.F.fit(x)