import numpy as np


class Neuron:
    def __init__(self, input_size):
        self.weights = np.random.randn(input_size + 1) * 0.1
        self.output = None

    def activate(self, inputs):
        z = np.dot(inputs, self.weights[:-1]) + self.weights[-1]
        self.output = self.sigmoid(z)
        return self.output

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

class Layer:
    def __init__(self, num_neurons, input_size):
        self.neurons = [Neuron(input_size)
                      for i in range(num_neurons)]
        self.output = None
        self.error = None
        self.delta = None

    def forward(self, inputs):
        self.output = np.array([neuron.activate(inputs)
                         for neuron in self.neurons])
        return self.output

class Network:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, X, y, learning_rate):
        output = self.forward(X)

        error = y - output
        self.layers[-1].error = error
        self.layers[-1].delta = \
            (error * self.layers[-1].neurons[-1].sigmoid_derivative(output))

        for i in reversed(range(len(self.layers) - 1)):
            layer = self.layers[i]
            next_layer = self.layers[i + 1]
            layer.error = np.dot(next_layer.delta,
                                 np.array([neuron.weights[:-1]
                                           for neuron in next_layer.neurons]))
            layer.delta = layer.error * np.array(
                [neuron.sigmoid_derivative(neuron.output)
                 for neuron in layer.neurons])

        for i in range(len(self.layers)):
            layer = self.layers[i]
            inputs = X if i == 0 else self.layers[i - 1].output
            for j, neuron in enumerate(layer.neurons):
                for k in range(len(neuron.weights) - 1):
                    neuron.weights[k] += learning_rate * layer.delta[j] * inputs[k]
                neuron.weights[-1] += learning_rate * layer.delta[j]

    def train(self, X, y, learning_rate, epochs):
        for epoch in range(epochs):
           for xi, yi in zip(X, y):
                self.backward(xi, yi, learning_rate)

input_size = 3
hidden_size = 5
output_size = 1

layer1 = Layer(hidden_size, input_size)
layer2 = Layer(output_size, hidden_size)

network = Network([layer1, layer2])

X = np.array([[0.5, 0.1, 0.4], [0.9, 0.7, 0.3], [0.2, 0.8, 0.6]])
y = np.array([[1], [0], [1]])

learning_rate = 0.1
epochs = 10000

for xi in X:
    output = network.forward(xi)
    print("Input:", xi, "Output:", output)

network.train(X, y, learning_rate, epochs)

print("----------------")
for xi in X:
    output = network.forward(xi)
    print("Input:", xi, "Output:", output)

