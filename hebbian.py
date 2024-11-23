import numpy as np

class Neuron:
    def __init__(self, input_size, lr = 1):
        self.W = np.random.rand(input_size)
        self.LR = lr
        self.T = 0

    def predict(self, x_image):
        s = np.dot(x_image, self.W) - self.T
        if s <= 0:
            return -1
        else:
            return 1

    def learning(self, inputs, outputs, loops = 1):
        for i in range(loops - 1):
            for i in range(len(inputs)):
                if self.predict(inputs[i]) != outputs[i]:
                    for j in range(len(self.W)):
                        self.W[j] += inputs[i][j] * outputs[i] * self.LR
                    self.T -= outputs[i]

X = ([-1, -1], [-1, 1], [1, -1], [1, 1])
Y = [-1, -1, -1, 1]
inst = Neuron(2)

print(inst.W, inst.T)
for i in range(4):
    print(inst.predict(X[i]))

inst.learning(X, Y, 3)

print(inst.W, inst.T)
for i in range(4):
    print(inst.predict(X[i]))