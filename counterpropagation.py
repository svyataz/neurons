import numpy as np

#кахонен
class som_layer:
    def __init__(self, n, k, a = 0.01):
        self.w = np.full((n, k), 1 / np.sqrt(k))
        self.distances = None
        self.output = np.zeros(n)
        self.a = a
        self.n = n
        self.k = k

    #победитель по сколярному произвдению (вывод наибольшего)
    def winner(self, x):
        self.output = np.zeros(self.n)
        x = self.preparation(x)
        self.distances = [np.dot(x, self.w[i]) for i in range(self.n)]
        closest = np.max(self.distances)
        self.output[np.nonzero(self.distances == closest)[0][0]] = closest
        return self.output, np.nonzero(self.distances == closest)[0][0]

    #подготовка данных
    def preparation(self, x):
        out = np.zeros(self.k)
        for i in range(self.k):
            out[i] = x[i] / np.sqrt(np.sum(x ** 2))
            out[i] = self.a * out[i] +  (1 - self.a) / np.sqrt(self.n)
        return out

#гроcсберг
class Grossberg_layer:
    def __init__(self, n, k, b = 0.1):
        self.w = np.random.rand(n, k)
        self.n = n
        self.k = k
        self.output = None
        self.b = b

    #выход вектора сколяров
    def out(self, x):
        self.output = np.zeros(self.n)
        for i in range(self.n):
            self.output[i] = np.dot(self.w[i], x)
        return self.output

#сеть
class Network:
    def __init__(self, n, k, epochs=100, lr = 0.7):
        self.som = som_layer(n, k)
        self.output_layer = Grossberg_layer(k, n)
        self.epochs = epochs
        self.lr = lr
        self.n = n
        self.k = k
    #просчёт сети

    def out(self, x):
        return self.output_layer.out(self.som.winner(x)[0])

    #тренировка
    def train(self, x, y):
        #запись превых Lr
        prev_lr = self.lr
        prev_a = self.som.a
        prev_b = self.output_layer.b
        #повторы
        for _ in range(self.epochs):
            #по векторам входов
            for k in range(self.k):
                som_v = self.som.winner(x[k])
                output = self.output_layer.out(som_v[0])
                #по нейронам кохонена веса меняю
                for i in range(self.n):
                    for j in range(self.k):
                        self.som.w[i][j] += self.lr * (x[k][j] - self.som.w[i][j])
                #по нейронам гроcсберга веса меняю
                for i in range(self.k):
                    if np.linalg.norm(y[i] - output) > 1e-5:
                        for j in range(self.n):
                            self.output_layer.w[i][j] += self.output_layer.b * (y[k][i] - self.output_layer.w[i][j]) * output[i]
            #изменение Lr
            self.lr -= prev_lr / self.epochs / 2
            self.som.a += (1 - prev_a) / self.epochs
            self.output_layer.b -= prev_b / self.epochs / 2



X = np.random.rand(15, 6)
Y = np.random.rand(15, 6)
print('XXXXXXXXXXXXXXXXXXXXXXXX')
print(X)
print('YYYYYYYYYYYYYYYYYYYYYYYY')
print(Y)
inst = Network(4, 6)
print('111111111111111111111111')
for i in range(15):
    print(inst.out(X[i]))
inst.train(X, Y)
print('222222222222222222222222')
for i in range(15):
    print(inst.out(X[i]))
print('YYYYYYYYYYYYYYYYYYYYYYYY')
print(Y)
