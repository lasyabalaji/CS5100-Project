"""
Logistic regression from scratch in NumPy.

    z = w . x + b
    p = sigmoid(z) = 1 / (1 + e^(-z))      probability that the image is tumor

Labels are 0 (not tumor) and 1 (tumor), same as data/load_data.py.

Loss is binary cross-entropy (log loss), with an optional weight on tumor images:
    L = -mean( pos_weight * y * log(p) + (1 - y) * log(1 - p) )

pos_weight = 1 is plain log loss. Only ~14% of PathMNIST images are tumor, so
with pos_weight = 1 the model learns to always say "not tumor" (F1 = 0).
Setting pos_weight = (#not tumor / #tumor) makes both classes count equally.

AI Disclousre: I psuedo coded this and made AI fill it in. I used AI to implement the weight on tumors
after results without it were unhelpful
"""
import numpy as np


def sigmoid(z):
    # clip so np.exp does not overflow for very large |z|
    z = np.clip(z, -30, 30)
    return 1.0 / (1.0 + np.exp(-z))


class LogisticRegression:

    def __init__(self, n_features, l2=0.0, pos_weight=1.0):
        """n_features: 2352 for PathMNIST (28 * 28 * 3).
        l2: strength of L2 regularization (0 means none).
        pos_weight: how much more a tumor image counts in the loss (1 means equal)."""
        self.n_features = n_features
        self.l2 = l2
        self.pos_weight = pos_weight
        # logistic regression loss is convex, so starting at zero is fine
        self.w = np.zeros(n_features, dtype=np.float32)
        self.b = np.zeros(1, dtype=np.float32)

    # forward pass
    def predict(self, X):
        """Probability of tumor for each row of X."""
        z = X @ self.w + self.b[0]
        return sigmoid(z)

    @staticmethod
    def balanced_pos_weight(y):
        """pos_weight that makes tumor and non-tumor count equally: #neg / #pos."""
        pos = float(np.sum(y))
        return (len(y) - pos) / max(pos, 1.0)

    def predict_labels(self, X, threshold=0.5):
        """0/1 predictions. Tumor if probability >= threshold."""
        return (self.predict(X) >= threshold).astype(int)

    # loss and gradient
    def loss(self, X, y):
        """Average binary cross-entropy on (X, y), plus the L2 term."""
        p = self.predict(X)
        eps = 1e-7  # avoid log(0)
        bce = -np.mean(self.pos_weight * y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
        return float(bce + 0.5 * self.l2 * np.sum(self.w ** 2))

    def gradients(self, X, y):
        """Returns (dw, db) for one batch."""
        n = len(y)
        p = self.predict(X)
        error = p * (1 + (self.pos_weight - 1) * y) - self.pos_weight * y   # shape (n,)
        dw = X.T @ error / n + self.l2 * self.w
        db = np.sum(error) / n
        return dw, db

    # training
    def train(self, X, y, epochs=1, lr=0.01, batch_size=64, seed=0):
        """Mini-batch gradient descent. Updates the weights in place.
        Returns the average training loss for each epoch."""
        rng = np.random.default_rng(seed)
        y = y.astype(np.float32)
        n = len(y)
        losses = []

        for epoch in range(epochs):
            order = rng.permutation(n)      # shuffle every epoch
            total_loss = 0.0

            for start in range(0, n, batch_size):
                idx = order[start:start + batch_size]
                X_batch = X[idx]
                y_batch = y[idx]

                dw, db = self.gradients(X_batch, y_batch)
                self.w -= (lr * dw).astype(np.float32)
                self.b -= np.float32(lr * db)

                total_loss += self.loss(X_batch, y_batch) * len(idx)

            losses.append(total_loss / n)

        return losses

    # weights
    def get_weights(self):
        return [self.w.copy(), self.b.copy()]

    def set_weights(self, weights):
        self.w = np.array(weights[0], dtype=np.float32, copy=True)
        self.b = np.array(weights[1], dtype=np.float32, copy=True)

    def num_weights(self):
        """Number of numbers sent to the server each round (w and b)."""
        return self.w.size + self.b.size

    def weights_size_bytes(self):
        """Size of the weights sent each round, in bytes."""
        return self.w.nbytes + self.b.nbytes
