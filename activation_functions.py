import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_deriv(z):
    s = sigmoid(z)
    return s * (1 - s)


def tanh(x):
    return np.tanh(x)


def tanh_deriv(z):
    return 1.0 - np.tanh(z) ** 2


def relu(x):
    return np.maximum(0, x)


def relu_deriv(z):
    return (z > 0).astype(float)


def swish(x):
    return x * sigmoid(x)


def swish_deriv(z):
    s = sigmoid(z)
    return s + z * s * (1 - s)


def get_activation(name):
    name = name.lower()
    if name == 'sigmoid':
        return sigmoid, sigmoid_deriv
    elif name == 'tanh':
        return tanh, tanh_deriv
    elif name == 'relu':
        return relu, relu_deriv
    elif name == 'swish':
        return swish, swish_deriv
    else:
        raise ValueError(f"Unknown activation function '{name}'")
