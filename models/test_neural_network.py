"""
Gradient check for the neural network.

Backprop *calculates* each gradient. Here we also *measure* it: nudge one weight up
and down by a tiny amount, see how the loss changes, and use the slope formula
    numerical gradient = (loss(w + EPS) - loss(w - EPS)) / (2 * EPS)
If backprop is correct, the two agree. We compare them with the relative error
    |numerical - backprop| / (|numerical| + |backprop|)
so the comparison means the same thing whether a gradient is large or small.

Run:  python -m models.test_neural_net
(also works with pytest if it is installed: pytest models/)
"""
import numpy as np

from models.neural_net import NeuralNetwork

EPS = 1e-5        # size of the nudge
TOLERANCE = 1e-6  # max allowed relative error. Correct backprop gives about 1e-10 to 1e-7
                  # here; a real bug gives about 1e-3 or much larger.

NAMES = ["W1", "b1", "W2", "b2", "W3", "b3"]   # same order as get_weights() and _backward()


def _float64_model(n_features, hidden_sizes, seed=0):
    """The network stores float32 weights, which aren't precise enough to measure tiny
    changes in the loss. For the check only, switch every weight to float64."""
    model = NeuralNetwork(n_features, hidden_sizes=hidden_sizes, seed=seed)
    for name in NAMES:
        setattr(model, name, getattr(model, name).astype(np.float64))
    return model


def _fake_batch(n_images, n_features, seed=0):
    """Random image-like data: pixel values in [0, 1], some tumor and some not."""
    rng = np.random.default_rng(seed)
    X = rng.random((n_images, n_features))
    y = (rng.random(n_images) < 0.3).astype(np.int64)
    y[0], y[1] = 1, 0                     # make sure both labels appear
    return X, y


def _exact_loss(p, y):
    """Binary cross-entropy WITHOUT the tiny 1e-7 safety term that model.loss() adds
    inside the log. Backprop's starting formula dz3 = (p - y) / N is the exact gradient
    of this loss, so this is the right loss to measure against. (The safety term only
    matters when p is exactly 0 or 1, which never happens in this check.)"""
    y = y.reshape(-1, 1)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _relative_error(numerical, backprop):
    # max(..., 1e-12): when both gradients are exactly 0 (a ReLU unit that was off
    # for every image), this gives 0 instead of dividing 0 by 0
    return abs(numerical - backprop) / max(abs(numerical) + abs(backprop), 1e-12)


def gradient_check(model, X, y, n_per_array=None, seed=0):
    """Compares backprop with measured gradients.

    n_per_array: how many weights to check in each array (W1, b1, ...).
                 None checks every weight; a number checks that many random ones.
    Returns the worst relative error for each array, e.g. {"W1": 2e-7, ...}.
    """
    p, cache = model._forward(X)
    grads = model._backward(cache, y)             # what backprop says
    rng = np.random.default_rng(seed)

    worst = {}
    for name, grad in zip(NAMES, grads):
        param = getattr(model, name)
        positions = list(np.ndindex(param.shape))
        if n_per_array is not None and n_per_array < len(positions):
            picks = rng.choice(len(positions), n_per_array, replace=False)
            positions = [positions[i] for i in picks]

        worst[name] = 0.0
        for pos in positions:
            original = param[pos]
            param[pos] = original + EPS                       # nudge up
            loss_plus = _exact_loss(model._forward(X)[0], y)
            param[pos] = original - EPS                       # nudge down
            loss_minus = _exact_loss(model._forward(X)[0], y)
            param[pos] = original                             # put it back

            numerical = (loss_plus - loss_minus) / (2 * EPS)  # slope formula
            worst[name] = max(worst[name], _relative_error(numerical, grad[pos]))
    return worst


# ----------------------------------------------------------------------- tests
def test_tiny_network_every_weight():
    """Small network (5 inputs, hidden layers of 4 and 3), so every weight is checked.
    Same two-hidden-layer code as the real network, just fewer units."""
    model = _float64_model(5, (4, 3))
    X, y = _fake_batch(8, 5)
    worst = gradient_check(model, X, y)
    assert all(err < TOLERANCE for err in worst.values()), worst
    return worst


def test_real_size_network_sample():
    """The actual network size (2352 inputs, 128 and 64 hidden units), checking 10
    random weights in each array instead of all 309,505."""
    model = _float64_model(2352, (128, 64))
    X, y = _fake_batch(16, 2352)
    worst = gradient_check(model, X, y, n_per_array=10)
    assert all(err < TOLERANCE for err in worst.values()), worst
    return worst


def test_check_catches_a_wrong_gradient():
    """Proves the check isn't fooled: make backprop 1% wrong for W1 on purpose and
    confirm the check fails."""
    model = _float64_model(5, (4, 3))
    X, y = _fake_batch(8, 5)
    correct_backward = model._backward

    def broken_backward(cache, y):
        grads = correct_backward(cache, y)
        grads[0] = grads[0] * 1.01                        # dW1 is now 1% off
        return grads

    model._backward = broken_backward
    worst = gradient_check(model, X, y)
    assert worst["W1"] > TOLERANCE, worst
    return worst


def test_predict_shape():
    """predict() must return shape (N,), not (N, 1), or F1 is silently wrong."""
    model = NeuralNetwork(2352)
    X, _ = _fake_batch(7, 2352)
    assert model.predict(X.astype(np.float32)).shape == (7,)


def _show(title, worst):
    print(title)
    for name in NAMES:
        print(f"  {name:3s} worst relative error {worst[name]:.1e}")


if __name__ == "__main__":
    _show("tiny network, every weight:", test_tiny_network_every_weight())
    _show("real-size network, 10 random weights per array:", test_real_size_network_sample())
    w = test_check_catches_a_wrong_gradient()
    print(f"planted 1% bug in dW1 -> W1 relative error {w['W1']:.1e} (check fails, as it should)")
    test_predict_shape()
    print("predict() returns shape (N,)")
    print(f"\nall neural network tests passed (tolerance {TOLERANCE:.0e})")