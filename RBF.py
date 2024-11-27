import numpy as np

xk = np.linspace(0, 1, 5)
x = np.linspace(0, 1, 100)
def true_fn(x): return x ** 2 - x - np.cos(np.pi * x)
def eculidean_distance(x, xk):
    return np.sqrt(((x.reshape(-1, 1)) - xk.reshape(1, -1)) ** 2)
print(eculidean_distance(xk, xk))

def gauss_rbf(radius, eps): return np.exp(-(eps*radius)**2)
print(gauss_rbf(eculidean_distance(xk, xk), 2))

class RBF(object):
    def __init__(self, eps):
        self.eps = eps

    def fit(self, xk, yk):
        self.xk = xk
        tr = gauss_rbf(eculidean_distance(xk, xk), self.eps)
        self.w_ = np.linalg.solve(tr, yk)

    def __call__(self, xn):
        tr = gauss_rbf(eculidean_distance(xk, xk), self.eps)
