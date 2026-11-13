"""
Ways for the server to combine the hospitals' weights into one global model.

FedAvg (McMahan et al., 2017, https://arxiv.org/abs/1602.05629):
    global = sum_k (n_k / n) * w_k
where n_k is the number of training images at hospital k and n = sum_k n_k.
So a hospital with more images counts more.
"""
import numpy as np


def fedavg(weight_lists, sizes):
    """Weighted average of the hospitals' weights.

    weight_lists: one entry per hospital. Each entry is a list of numpy arrays
                  (what model.get_weights() returns), in the same order and
                  shapes for every hospital.
    sizes:        number of training images at each hospital.

    Returns a new list of numpy arrays, with the same shapes as one hospital's list.
    """
    if len(weight_lists) == 0:
        raise ValueError("need at least one hospital")
    if len(weight_lists) != len(sizes):
        raise ValueError("need one size per hospital")

    sizes = np.asarray(sizes, dtype=np.float64)
    if np.any(sizes < 0) or sizes.sum() <= 0:
        raise ValueError("sizes must be non-negative and not all zero")
    shares = sizes / sizes.sum()  # n_k / n, adds up to 1

    n_layers = len(weight_lists[0])
    for w in weight_lists:
        if len(w) != n_layers:
            raise ValueError("every hospital must send the same number of arrays")

    averaged = []
    for layer in range(n_layers):
        shape = np.shape(weight_lists[0][layer])
        total = np.zeros(shape, dtype=np.float64)
        for share, w in zip(shares, weight_lists):
            if np.shape(w[layer]) != shape:
                raise ValueError(f"array {layer} has different shapes across hospitals")
            total += share * np.asarray(w[layer], dtype=np.float64)
        averaged.append(total.astype(np.asarray(weight_lists[0][layer]).dtype))
    return averaged
