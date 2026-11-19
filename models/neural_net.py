"""
Deep neural network (two hidden layers) for tumor vs. not tumor, from scratch in NumPy.

Architecture (shapes for a batch of 64 images):
    z1 = X @ W1 + b1        (64, 128)
    a1 = relu(z1)           (64, 128)
    z2 = a1 @ W2 + b2       (64, 64)
    a2 = relu(z2)           (64, 64)
    z3 = a2 @ W3 + b3       (64, 1)
    p  = sigmoid(z3)        (64, 1)   -> flattened to (64,) in predict()

Follows the same interface as fed/temp_model.py, so the FedAvg script can use it:
    model.train(X, y, epochs, lr, batch_size, seed)  -> list of losses, one per epoch
    model.predict(X)                                 -> tumor probabilities, shape (N,)
    model.get_weights()                              -> [W1, b1, W2, b2, W3, b3] (copies)
    model.set_weights(list_of_arrays)

Status: forward pass, loss, backprop, training, and interface done.
Next steps: gradient check, Phase 1 learning-rate check, hook into simulate_fedavg.py.
"""
import numpy as np


def relu(z):
    return np.maximum(0, z)


def sigmoid(z):
    # clip so np.exp never overflows for very large or very small z
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


class NeuralNetwork:
    def __init__(self, n_features, hidden_sizes=(128, 64), seed=0):
        rng = np.random.default_rng(seed)
        h1, h2 = hidden_sizes

        # He initialization: random weights scaled by sqrt(2 / inputs to the layer),
        # which suits ReLU. Weights can't start at zero, or every unit would learn
        # the same thing. Biases can start at zero.
        self.W1 = (rng.standard_normal((n_features, h1)) * np.sqrt(2.0 / n_features)).astype(np.float32)
        self.b1 = np.zeros(h1, dtype=np.float32)
        self.W2 = (rng.standard_normal((h1, h2)) * np.sqrt(2.0 / h1)).astype(np.float32)
        self.b2 = np.zeros(h2, dtype=np.float32)
        self.W3 = (rng.standard_normal((h2, 1)) * np.sqrt(2.0 / h2)).astype(np.float32)
        self.b3 = np.zeros(1, dtype=np.float32)

    # ------------------------------------------------------------------ forward
    def _forward(self, X):
        """Runs the network and keeps every intermediate value, since backprop needs them."""
        z1 = X @ self.W1 + self.b1
        a1 = relu(z1)
        z2 = a1 @ self.W2 + self.b2
        a2 = relu(z2)
        z3 = a2 @ self.W3 + self.b3
        p = sigmoid(z3)                      # (N, 1)
        cache = {"X": X, "z1": z1, "a1": a1, "z2": z2, "a2": a2, "p": p}
        return p, cache

    def predict(self, X):
        p, _ = self._forward(X)
        return p.ravel()                     # (N, 1) -> (N,), so it lines up with the labels y

    # --------------------------------------------------------------------- loss
    @staticmethod
    def loss(p, y):
        """Binary cross-entropy. p: (N, 1) probabilities, y: (N,) labels of 0/1."""
        y = y.reshape(-1, 1)                 # (N,) -> (N, 1), avoids the (N, N) broadcast bug
        eps = 1e-7                           # keeps log() away from log(0)
        return float(-np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)))

    # ----------------------------------------------------------------- backward
    def _backward(self, cache, y):
        """Returns the gradients [dW1, db1, dW2, db2, dW3, db3], in the same order as
        get_weights(). Every layer follows the same pattern:
            dW = (layer input).T @ dz      weight gradient
            db = dz.sum(axis=0)            bias gradient (summed over the batch)
            da = dz @ W.T                  pass the error back to the previous layer
            dz = da * (z > 0)              back through ReLU (blocks units that were off)
        """
        N = len(y)
        y = y.reshape(-1, 1).astype(np.float32)   # (N,) -> (N, 1), avoids the broadcast bug
        X, z1, a1, z2, a2, p = (cache[k] for k in ("X", "z1", "a1", "z2", "a2", "p"))

        # Output layer: sigmoid + binary cross-entropy simplifies to (p - y) / N
        dz3 = (p - y) / N                    # (N, 1)
        dW3 = a2.T @ dz3                     # (64, N) @ (N, 1)   -> (64, 1)
        db3 = dz3.sum(axis=0)                # (1,)

        # Hidden layer 2
        da2 = dz3 @ self.W3.T                # (N, 1) @ (1, 64)   -> (N, 64)
        dz2 = da2 * (z2 > 0)                 # element-wise: ReLU gradient is 1 if z > 0, else 0
        dW2 = a1.T @ dz2                     # (128, N) @ (N, 64) -> (128, 64)
        db2 = dz2.sum(axis=0)                # (64,)

        # Hidden layer 1 (no need to pass the error back past X)
        da1 = dz2 @ self.W2.T                # (N, 64) @ (64, 128) -> (N, 128)
        dz1 = da1 * (z1 > 0)
        dW1 = X.T @ dz1                      # (2352, N) @ (N, 128) -> (2352, 128)
        db1 = dz1.sum(axis=0)                # (128,)

        return [dW1, db1, dW2, db2, dW3, db3]

    # -------------------------------------------------------------------- train
    def train(self, X, y, epochs=1, lr=0.01, batch_size=64, seed=0):
        """Mini-batch gradient descent, updates the weights in place.
        Returns the average loss for each epoch."""
        rng = np.random.default_rng(seed)
        n = len(X)
        losses = []
        for _ in range(epochs):
            order = rng.permutation(n)           # new shuffle every epoch
            total = 0.0
            for start in range(0, n, batch_size):
                idx = order[start:start + batch_size]
                p, cache = self._forward(X[idx])
                total += self.loss(p, y[idx]) * len(idx)
                grads = self._backward(cache, y[idx])
                # gradient descent step: move every weight a little downhill
                params = [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]
                for param, grad in zip(params, grads):
                    param -= lr * grad           # in place, so self.W1 etc. are updated
            losses.append(total / n)
        return losses

    # ---------------------------------------------------- weights (for FedAvg)
    def get_weights(self):
        # copies, so later training doesn't change weights the server already saved
        return [self.W1.copy(), self.b1.copy(), self.W2.copy(),
                self.b2.copy(), self.W3.copy(), self.b3.copy()]

    def set_weights(self, weights):
        self.W1, self.b1, self.W2, self.b2, self.W3, self.b3 = (
            np.array(w, dtype=np.float32, copy=True) for w in weights)