import numpy as np
import math

from backprogration import epochs


class RNN_TBPTT:
    def __init__(self, lr, epochs, len_of_seq, hidden_n, output_n,
                 bptt_truncate = 5, min_clip_value = -10, max_clip_value = 10):
        self.lr = lr
        self.epochs = epochs
        self.len_of_seq = len_of_seq
        self.hidden_n = hidden_n
        self.output_n = output_n
        self.bptt_truncate = bptt_truncate
        self.min = min_clip_value
        self.max = max_clip_value
        self.input_w = np.random.uniform(0, 1, (hidden_n, len_of_seq))
        self.hidden_w = np.random.uniform(0, 1, (hidden_n, hidden_n))
        self.output_w = np.random.uniform(0, 1, (output_n, hidden_n))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, X, Y):
        for i in range(Y.shape[0]):
            x, y = X[i], Y[i]
            prev_s = np.zeros((self.hidden_n, 1))
            for _ in range(self.len_of_seq):
                input = np.dot(self.input_w, x)
                hidden = np.dot(self.hidden_w, prev_s)
                add = input + hidden
                s = self.sigmoid(add)
                output = np.dot(self.output_w, s)
                prev_s = s
            print(f"ожидаемых выход: {y}, реальный:{output}")

    def train(self, X, Y):
        for epoch in range(self.epochs):
            loss = 0.0
            for i in range(Y.shape[0]):
                x, y = X[i], Y[i]
                prev_s = np.zeros((self.hidden_n, 1))
                for t in range(self.len_of_seq):
                    new_input = np.zeros(x.shape)
                    new_input[t] = x[t]
                    u = np.dot(self.input_w, new_input)
                    w = np.dot(self.hidden_w, prev_s)
                    add = w + u
                    s = self.sigmoid(add)
                    mulv = np.dot(self.output_w, s)
                    prev_s = s

                # calculate error
                loss_per_record = (y - mulv) ** 2 / 2
                loss += loss_per_record
            loss = loss / float(y.shape[0])
            print(f"итерация {epoch + 1} потери = {loss}")
            for i in range(Y.shape[0]):
                x, y = X[i], Y[i]
                layers = []
                prev_s = np.zeros((self.hidden_n, 1))
                dU= np.zeros(self.input_w.shape)
                dW = np.zeros(self.hidden_w.shape)
                dV = np.zeros(self.output_w.shape)

                dU_t = np.zeros(self.input_w.shape)
                dW_t = np.zeros(self.hidden_w.shape)
                dV_t = np.zeros(self.output_w.shape)

                dU_i= np.zeros(self.input_w.shape)
                dW_i = np.zeros(self.hidden_w.shape)

                # forward pass
                for t in range(self.len_of_seq):
                    new_input = np.zeros(x.shape)
                    new_input[t] = x[t]
                    input = np.dot(self.input_w, new_input)
                    hidden = np.dot(self.hidden_w, prev_s)
                    add = input + hidden
                    s = self.sigmoid(add)
                    output = np.dot(self.output_w, s)
                    layers.append({'s': s, 'prev_s': prev_s})
                    prev_s = s
                d_output = output - y

                for t in range(self.len_of_seq):
                    dV_t = np.dot(d_output, np.transpose(layers[t]['s']))
                    dsv = np.dot(np.transpose(self.output_w), d_output)

                    ds = dsv
                    dadd = add * (1 - add) * ds

                    d_hidden = dadd * np.ones_like(hidden)

                    dprev_s = np.dot(np.transpose(self.hidden_w), d_hidden)

                    for i in range(t - 1, max(-1, t - self.bptt_truncate - 1), -1):
                        ds = dsv + dprev_s
                        dadd = add * (1 - add) * ds

                        d_hidden = dadd * np.ones_like(hidden)
                        d_input = dadd * np.ones_like(input)

                        dW_i = np.dot(self.hidden_w, layers[t]['prev_s'])
                        dprev_s = np.dot(np.transpose(self.hidden_w), d_hidden)

                        new_input = np.zeros(x.shape)
                        new_input[t] = x[t]
                        dU_i = np.dot(self.input_w, new_input)
                        dx = np.dot(np.transpose(self.input_w), d_input)

                        dU_t += dU_i
                        dW_t += dW_i

                    dV += dV_t
                    dU += dU_t
                    dW += dW_t

                    if dU.max() > self.max:
                        dU[dU > self.max] = self.max
                    if dV.max() > self.max:
                        dV[dV > self.max] = self.max
                    if dW.max() > self.max:
                        dW[dW > self.max] = self.max

                    if dU.min() < self.min:
                        dU[dU < self.min] = self.min
                    if dV.min() < self.min:
                        dV[dV < self.min] = self.min
                    if dW.min() < self.min:
                        dW[dW < self.min] = self.min

                self.input_w -= self.lr * dU
                self.output_w -= self.lr * dV
                self.hidden_w-= self.lr * dW




#подготовка тренеровачных данных
sin_wave = np.array([math.sin(x) for x in np.arange(200)])
X = []
Y = []
seq_len = 50
num_records = len(sin_wave) - seq_len
for i in range(num_records - 50):
    X.append(sin_wave[i:i+seq_len])
    Y.append(sin_wave[i+seq_len])
X = np.array(X)
X = np.expand_dims(X, axis=2)
Y = np.array(Y)
Y = np.expand_dims(Y, axis=1)
#запуск
inst = RNN_TBPTT(0.0001, 100, 50, 100, 1)
print("до тренеровки")
inst.forward(X,Y)
print("тренеровка в 25 шагов")
inst.train(X, Y)
print("после тренеровки")
inst.forward(X,Y)