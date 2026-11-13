"""
Quick checks for fedavg.

Run:  python -m fed.test_aggregators
(also works with pytest if it is installed: pytest fed/)
"""
import numpy as np

from fed.aggregators import fedavg


def _rand_weights(rng):
    # looks like a small model: a weight matrix and a bias vector
    return [rng.normal(size=(5, 3)), rng.normal(size=(3,))]


def test_equal_sizes_is_plain_average():
    rng = np.random.default_rng(0)
    hospitals = [_rand_weights(rng) for _ in range(4)]
    out = fedavg(hospitals, sizes=[100, 100, 100, 100])
    for layer in range(2):
        plain = np.mean([h[layer] for h in hospitals], axis=0)
        assert np.allclose(out[layer], plain)


def test_one_hospital_returns_its_own_weights():
    rng = np.random.default_rng(1)
    only = _rand_weights(rng)
    out = fedavg([only], sizes=[123])
    for a, b in zip(out, only):
        assert np.allclose(a, b)


def test_bigger_hospital_counts_more():
    # hospital A (size 3) has all 1s, hospital B (size 1) has all 0s -> 0.75
    a = [np.ones((2, 2))]
    b = [np.zeros((2, 2))]
    out = fedavg([a, b], sizes=[3, 1])
    assert np.allclose(out[0], 0.75)


if __name__ == "__main__":
    test_equal_sizes_is_plain_average()
    test_one_hospital_returns_its_own_weights()
    test_bigger_hospital_counts_more()
    print("all fedavg tests passed")
