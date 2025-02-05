import numpy as np
from dataset import load_data, load_xor_data, generate_heart_dataset
from model import NeuralNetwork


def train_and_return_data(epochs, hidden_dim, lr, activation, dataset):
    if dataset == "XOR":
        X_train, Y_train = load_xor_data()
        X_test, Y_test = X_train, Y_train
    elif dataset == "Спіраль":
        X_train, X_test, Y_train, Y_test = load_data(500, 2, 3)
    elif dataset == "Серце":
        X_train, X_test, Y_train, Y_test = generate_heart_dataset(1000, 0.1)

    nn = NeuralNetwork(input_dim=2, hidden_dim=hidden_dim, activation_hidden=activation)

    losses = []
    for e in range(epochs):
        y_pred = nn.forward(X_train)
        eps = 1e-9
        loss = -np.mean(Y_train * np.log(y_pred + eps) + (1 - Y_train) * np.log(1 - y_pred + eps))
        losses.append(loss)
        nn.backward(X_train, Y_train, y_pred, lr)

    return nn, X_train, Y_train, losses
