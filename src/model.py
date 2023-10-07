from kernel_functions import RFF, class_kernel_mean_embedding
from optimization_functions import HSIC, find_optimal_sigma
from utils import sort_array_based_on_other

from sklearn.metrics.pairwise import rbf_kernel, pairwise_distances
from sklearn.preprocessing import OneHotEncoder

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

class kchain:
    
    def __init__(self, min_sigma=0.01, RFF_d=200, hsic=HSIC, stop_crit='hsic', delta=0.001, verbose=False):

        if stop_crit not in ['hsic', 'max_layers']:
            raise ValueError('stop_crit must be either "hsic" or "max_layers"')

        self.min_sigma = min_sigma
        self.RFF_d = RFF_d
        self.hsic_func = hsic
        self.stop_crit = stop_crit
        self.verbose = verbose
        self.layers = []
        self.delta = delta

    def forward(self, x, n=0):
        z = x
        
        if n > len(self.layers):
            raise ValueError(f"n must be less than {len(self.layers)}")
        i = 0

        for l in self.layers:

            z = l.forward(z)
            i += 1
            if i == n:
                break
        return z

    def fit(self, X, y, X_test=None, y_test=None, max_layers=10, verbose=True):
        z, y, z_test, y_test = self._init_fit(X, y, X_test, y_test)

        if X_test is not None and y_test is not None:
            validate = True
        else:
            validate = False

        hsic_previous = 0

        for i in range(max_layers):
            l = layer(self.RFF_d)
            hsic_current, z = l.fit_layer(z, y, self.Ky, self.min_sigma, self.hsic_func)

            if verbose:
                print(f"HSIC for layer {i}: {hsic_current:.3f} - Previous: {hsic_previous:.3f}")

            if hsic_previous > (hsic_current - self.delta) and self.stop_crit == 'hsic':
                break
            
            if validate:
                l.validate(z_test, y_test, self.Ky_test, self.hsic_func)
                z_test = l.forward(z_test)

            self.layers.append(l)
            hsic_previous = hsic_current
    

    def _init_fit(self, X, y, X_test, y_test):
        self.ohe = OneHotEncoder()
        y_ohe = self.ohe.fit_transform(y.reshape(-1, 1)).toarray()
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        self.Ky = y_ohe.dot(y_ohe.T)
        # self.Ky = y.dot(y.T)

        if y_test is not None:
            if len(y_test.shape) == 1:
                y_test = y_test.reshape(-1, 1)
            y_test_ohe = self.ohe.transform(y_test.reshape(-1, 1)).toarray()
            self.Ky_test = y_test_ohe.dot(y_test_ohe.T)
            # self.Ky_test = y_test.dot(y_test.T)
        
        return X, y, X_test, y_test
    
        
class layer:

    def __init__(self, d):
        self.W = None
        self.sigma = None
        self.d = d
        self.metrics = {}
        pass

    def forward(self, x):
        return self.RFF.forward(x).dot(self.W.T)
    
    def fit_layer(self, x, y, Ky, min_sigma, hsic_func):

        sigma = find_optimal_sigma(x, Ky, hsic_func)

        if sigma < min_sigma:
            sigma = min_sigma

        self.sigma = sigma

        self.RFF = RFF(self.d, sigma)
        self.RFF.fit(x)

        r = self.RFF.forward(x)

        self.W = class_kernel_mean_embedding(r, y)

        Kx = rbf_kernel(x, gamma=1/(2*sigma**2))
        hsic = hsic_func(Kx, Ky)

        z = r.dot(self.W.T)

        self.knn = KNeighborsClassifier(n_neighbors=1).fit(z, y.ravel())
        self.gnb = GaussianNB().fit(z, y.ravel())

        self.metrics = {**self.metrics, **{
            'hsic_train': hsic,
            'knn_train': self.knn.score(z, y.ravel()),
            'gnb_train': self.gnb.score(z, y.ravel())
        }}
        
        return hsic, z
    
    def validate(self, x_test, y_test, Ky, hsic_func):
        z_test = self.forward(x_test)
 
        Kx_test = rbf_kernel(x_test, gamma=1/(2*self.sigma**2))
        hsic = hsic_func(Kx_test, Ky)

        knn_acc = self.knn.score(z_test, y_test.ravel())
        gnb_acc = self.gnb.score(z_test, y_test.ravel())

        self.metrics = {**self.metrics, **{
            'hsic_test': hsic,
            'knn_test': knn_acc,
            'gnb_test': gnb_acc
        }}
    
    def get_sorted_kernel(self, x, y):
        x = sort_array_based_on_other(y, x)
        return rbf_kernel(x, gamma=1/(2*self.sigma**2))
