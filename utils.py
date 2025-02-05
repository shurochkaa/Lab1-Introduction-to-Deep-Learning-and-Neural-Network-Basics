import numpy as np
import matplotlib.pyplot as plt


def calculate_accuracy(model, X, Y):
    predictions = model.forward(X)
    predicted_labels = np.argmax(predictions, axis=1) if predictions.shape[1] > 1 else (predictions > 0.5).astype(int)
    accuracy = np.mean(predicted_labels == Y.flatten()) * 100
    return accuracy


def plot_decision_boundary(model, X, Y, ax=None):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200)
    )
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    Z = model.predict(grid_points)
    Z = Z.reshape(xx.shape)

    if ax is None:
        plt.figure()
        plt.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.Spectral)
        plt.scatter(X[:, 0], X[:, 1], c=Y[:, 0], s=40, edgecolors='k', cmap=plt.cm.Spectral)
        plt.title("Decision Boundary")
        plt.show()
    else:
        ax.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.Spectral)
        ax.scatter(X[:, 0], X[:, 1], c=Y[:, 0], s=40, edgecolors='k', cmap=plt.cm.Spectral)
        ax.set_title("Кордони рішень")
