"""
TEMPORARY stand-in model. DELETE once Abdullah's models/ (base.py etc.) exist.

It exists only so the federated code can be run and tested now. It follows the
interface we expect from models/base.py:

    model.train(X, y, epochs, lr, batch_size, seed)   mini-batch SGD, in place
    model.predict(X) -> probability of tumor, shape (N,)
    model.get_weights() -> list of numpy arrays
    model.set_weights(list_of_numpy_arrays)

If Abdullah's real signatures differ, only experiments/simulate_fedavg.py's
make_model() and local_train() need small changes.
"""
import numpy as np


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


class TempLogisticRegression:
    """TEMPORARY: plain logistic regression, binary cross-entropy, mini-batch SGD."""

    def __init__(self, n_features):
        self.w = np.zeros(n_features, dtype=np.float32)
        self.b = np.zeros(1, dtype=np.float32)

    def predict(self, X):
        return _sigmoid(X @ self.w + self.b[0])

    def train(self, X, y, epochs=1, lr=0.01, batch_size=64, seed=0):
        rng = np.random.default_rng(seed)
        n = len(X)
        losses = []
        for _ in range(epochs):
            order = rng.permutation(n)
            total = 0.0
            for start in range(0, n, batch_size):
                idx = order[start:start + batch_size]
                p = self.predict(X[idx])
                err = p - y[idx]                       # d(loss)/d(z) for BCE + sigmoid
                self.w -= lr * (X[idx].T @ err) / len(idx)
                self.b -= lr * err.mean()
                eps = 1e-7
                total += -np.sum(y[idx] * np.log(p + eps) + (1 - y[idx]) * np.log(1 - p + eps))
            losses.append(total / n)
        return losses

    def get_weights(self):
        return [self.w.copy(), self.b.copy()]

    def set_weights(self, weights):
        self.w = np.array(weights[0], dtype=np.float32, copy=True)
        self.b = np.array(weights[1], dtype=np.float32, copy=True)
