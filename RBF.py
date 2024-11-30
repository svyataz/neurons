import numpy as np


class Network:
    def __init__(self, k, lr=0.01, epochs=100):
        self.k = k
        self.lr = lr
        self.epochs = epochs
        self.w = np.random.randn(k)
        self.b = np.random.randn(1)

    def kmeans(self, X, k):
        clusters = np.random.choice(np.squeeze(X), size=k)
        prevClusters = clusters.copy()
        converged = False
        while not converged:
            distances = np.squeeze(np.abs(X[:, np.newaxis] - clusters[np.newaxis, :]))
            closestCluster = np.argmin(distances, axis=1)
            for i in range(k):
                pointsForCluster = X[closestCluster == i]
                if len(pointsForCluster) > 0:
                    clusters[i] = np.mean(pointsForCluster, axis=0)
            converged = np.linalg.norm(clusters - prevClusters) < 1e-6
            prevClusters = clusters.copy()
        return clusters

    def predict(self, X):
            a = np.array([self.rbf(X, c, s) for c, s, in zip(self.centers, self.stds)])
            F = np.dot(a, self.w) + self.b
            return F

    def fit(self, X, y):
        self.centers = self.kmeans(X, self.k)
        dMax = max([np.abs(c1 - c2) for c1 in self.centers for c2 in self.centers])
        self.stds = np.repeat(dMax / np.sqrt(2 * self.k), self.k)
        for epoch in range(self.epochs):
            for i in range(X.shape[0]):
                a = np.array([self.rbf(X[i], c, s) for c, s, in zip(self.centers, self.stds)])
                F = np.dot(a, self.w) + self.b
                if F != y[i]:
                    error = (y[i] - F).flatten()
                    self.w += self.lr * a * error
                    self.b += self.lr * error


    def rbf(self, x, c, s):
        return np.exp(-1 / (2 * s ** 2) * (x - c) ** 2)

X = np.random.uniform(0., 1., 30)
y = (1 / (1 + np.exp(-X)))
print(y)
rbfnet = Network(k=2)
rbfnet.fit(X, y)
for i in X:
    print(rbfnet.predict(i), end=' ')