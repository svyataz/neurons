import numpy as np


class neuron:
    def __init__(self, n, index, a = 0.01):
        self.w = np.random.rand(n)
        self.w[index] = 0
        self.t = 0
        self.distances = None
        self.output = np.zeros(n)
        self.n = n
        self.output = None

    def net(self, x):
        output = np.dot(x, self.w) * (self.t - 1)
        return output

    def sigmoid(self, output):
        self.output =  1 / (1 + np.exp(-output))
        return self.output
