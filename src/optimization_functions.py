import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import pairwise_distances


def HSIC(Kx, Ky):
    """
    Implemented according to Kernel Mean Embedding of Distributions: A Review and Beyond 
    (p58) and grettons work http://www.gatsby.ucl.ac.uk/~gretton/indepTestFiles/indep.htm
    """

    m = Kx.shape[0]

    H = np.eye(m)-1/m*np.ones((m,m));

    HKxH = H.dot(Kx.dot(H)) # equivalent to		HKᵪ = H.dot(Kᵪ)
    # HKyH = H.dot(Ky.dot(H)) # equivalent to		HKᵧ = H.dot(Kᵧ)

    return 1 / m * np.sum(HKxH.T * Ky)

def normalized_HSIC(Kx, Ky):

    """
    Wu, C. T., Masoomi, A., Gretton, A., & Dy, J. (2022, May). 
    Deep Layer-wise Networks Have Closed-Form Weights. 
    In International Conference on Artificial Intelligence and Statistics (pp. 188-225). PMLR.
    """
    HKx = Kx - np.mean(Kx, axis=0) # equivalent to		HKᵪ = H.dot(Kᵪ)
    HKy = Ky - np.mean(Ky, axis=0) # equivalent to		HKᵧ = H.dot(Kᵧ)

    Hxy = np.sum(HKx.T*HKy)

    Hx = np.linalg.norm(HKx)
    Hy = np.linalg.norm(HKy)

    return Hxy / (Hx * Hy + 1e-5)
    

def find_optimal_sigma(X, Ky, hsic=HSIC):
    """Maximize the HSIC with respect to sigma
    """

    dX_2 = - (pairwise_distances(X) **2 )

    def objective(sigma):
        ## faster implementation instead of using rbf_kernel
        Kx = np.exp(dX_2 / (2 * sigma**2))

        return -hsic(Kx, Ky)

    res = minimize(objective, 1.0, method='BFGS', options={'gtol': 1e-5, 'disp': False})
    return res.x[0]