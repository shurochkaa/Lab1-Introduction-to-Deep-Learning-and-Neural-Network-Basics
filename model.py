import numpy as np
from activation_functions import get_activation, sigmoid, sigmoid_deriv


class NeuralNetwork:
    def __init__(self, input_dim, hidden_dim, activation_hidden='relu'):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        limit_in = np.sqrt(1.0 / input_dim)
        limit_hidden = np.sqrt(1.0 / hidden_dim)

        self.W1 = np.random.uniform(-limit_in, limit_in, (input_dim, hidden_dim))
        self.b1 = np.zeros((1, hidden_dim))

        self.W2 = np.random.uniform(-limit_hidden, limit_hidden, (hidden_dim, 1))
        self.b2 = np.zeros((1, 1))

        self.act_hidden, self.act_hidden_deriv = get_activation(activation_hidden)
        self.act_out, self.act_out_deriv = sigmoid, sigmoid_deriv

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.act_hidden(self.z1)

        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.act_out(self.z2)

        return self.a2

    def backward(self, X, y_true, y_pred, lr=0.01):
        N = X.shape[0]

        dz2 = (y_pred - y_true)

        dW2 = np.dot(self.a1.T, dz2) / N
        db2 = np.sum(dz2, axis=0, keepdims=True) / N

        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * self.act_hidden_deriv(self.z1)

        dW1 = np.dot(X.T, dz1) / N
        db1 = np.sum(dz1, axis=0, keepdims=True) / N

        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1

    def train(self, X, y, epochs=1000, lr=0.01):
        for e in range(epochs):
            y_pred = self.forward(X)

            eps = 1e-9
            loss = -np.mean(
                y * np.log(y_pred + eps) + (1 - y) * np.log(1 - y_pred + eps)
            )

            self.backward(X, y, y_pred, lr)

    def predict(self, X):
        y_prob = self.forward(X)
        return (y_prob >= 0.5).astype(int)
