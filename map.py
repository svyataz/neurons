import numpy as np


class maps:
    def __init__(self, n, k, lr=0.01, epochs=1):
        self.n = n
        self.lr = lr
        self.epochs = epochs
        self.w = np.random.rand(n, k)
        self.b = np.random.randn(1)

    def winner(self, X):
        X = np.array(X)
        print(self.w)
        self.distances = np.sqrt(np.sum((X - self.w[:, np.newaxis]) ** 2, axis=2))
        print("---------")
        print(self.distances)
        closest = np.min(self.distances)
        return np.where(self.distances == closest)[0]

    def g(self, i, j):
        R = np.sqrt(np.sum((self.w[i] - self.w[j]) ** 2))
        g = np.exp(R ** 2 / (2 * 1 ** 2))
        return g

    def g2(self, i, j):
        if abs(i - j) <= 5:
            return 1
        return 0

    def train(self, X):
        for _ in range(self.epochs):
            for k in range(len(X)):
                j = self.winner(X[k])
                for i in range(len(self.w)):
                    self.w[i] += self.lr * self.g(i,j) * self.distances[i]

X = np.random.rand(5, 3)
print(X)
inst = maps(5, 3)

print(*inst.winner(X[0]))
inst.train(X)
print(X[0])
print("---------")
print(*inst.winner(X[0]))