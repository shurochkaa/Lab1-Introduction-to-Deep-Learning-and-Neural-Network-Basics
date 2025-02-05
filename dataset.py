import numpy as np
from sklearn.model_selection import train_test_split


def load_data(points_per_class=100, num_classes=2, spiral_size=2):
    np.random.seed(42)
    X = []
    Y = []
    for j in range(num_classes):
        r = np.linspace(0.0, spiral_size, points_per_class)  # Більший радіус
        t = np.linspace(j * np.pi, (j + 2) * np.pi, points_per_class) + np.random.randn(points_per_class) * 0.2  # Кут
        X.append(np.c_[r * np.sin(t), r * np.cos(t)])
        Y.append(np.full(points_per_class, j))

    X = np.vstack(X)
    Y = np.hstack(Y)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y.reshape(-1, 1),
                                                        test_size=0.2, random_state=42)

    return X_train, X_test, Y_train, Y_test


def load_xor_data():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y = np.array([[0], [1], [1], [0]])
    return X, Y


def generate_heart_dataset(n_samples=1000, noise=0.1):
    np.random.seed(42)
    X = []
    Y = []

    for _ in range(n_samples):
        t = np.random.uniform(0, 2 * np.pi)

        x = 16 * np.sin(t) ** 3
        y = 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)

        x += np.random.normal(0, noise)
        y += np.random.normal(0, noise)

        X.append([x, y])

        Y.append(0 if x < 0 else 1)

    X = np.array(X)
    Y = np.array(Y).reshape(-1, 1)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    return X_train, X_test, Y_train, Y_test