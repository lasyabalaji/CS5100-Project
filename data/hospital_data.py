"""
Read the per-hospital .npz shards back off disk.

AI Disclosure: This file is almost entirely AI generated

Layout written by `python -m data.partition --mode <mode> --n 4`:
    data/hospitals/<mode>/hospital_<i>.npz   training data, one file per hospital
    data/hospitals/val.npz                   shared validation set
    data/hospitals/test.npz                  shared test set
"""
import os

import numpy as np

HOSPITAL_DIR = "data/hospitals"
N_HOSPITALS = 4


def load_npz(path):
    """Returns (X, y) from one .npz shard."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run: python -m data.partition --mode <mode> --n 4")
    d = np.load(path)
    return d["X"], d["y"]


def hospital_path(mode, i, root=HOSPITAL_DIR):
    return os.path.join(root, mode, f"hospital_{i}.npz")


def load_hospital(mode, i, root=HOSPITAL_DIR):
    """One hospital's training data."""
    return load_npz(hospital_path(mode, i, root))


def load_hospitals(mode, n_hospitals=N_HOSPITALS, root=HOSPITAL_DIR):
    """All hospitals' training data, as a list of (X, y)."""
    return [load_hospital(mode, i, root) for i in range(n_hospitals)]


def load_pooled(mode, n_hospitals=N_HOSPITALS, root=HOSPITAL_DIR):
    """Every hospital's training data concatenated back into one set.

    Any partition mode covers the same images, so this is the centralized
    (non-federated) training set no matter which mode was used.
    """
    parts = load_hospitals(mode, n_hospitals, root)
    X = np.concatenate([X_k for X_k, _ in parts])
    y = np.concatenate([y_k for _, y_k in parts])
    return X, y


def load_split(split, root=HOSPITAL_DIR):
    """The shared "val" or "test" set. No hospital trains on these."""
    if split not in ("val", "test"):
        raise ValueError(f'split must be "val" or "test", got {split!r}')
    return load_npz(os.path.join(root, f"{split}.npz"))
