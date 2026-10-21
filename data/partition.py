"""

Modes:
  iid         random, equal split
  label_skew  each hospital gets a different share of tumor cases
  size_skew   one big hospital, several small ones

Run:  python -m data.partition --mode label_skew --n 4
"""
import argparse
import os
import numpy as np


def partition(y, n_hospitals, mode="iid", seed=0, pos_ratios=None, sizes=None):
    """Returns a list of index arrays, one per hospital."""
    rng = np.random.default_rng(seed)
    n = len(y)

    if mode == "iid":
        return np.array_split(rng.permutation(n), n_hospitals)

    if mode == "label_skew":
        if pos_ratios is None:  # e.g. cancer center ... small clinic
            pos_ratios = np.linspace(0.6, 0.05, n_hospitals)
        pos = rng.permutation(np.where(y == 1)[0])
        neg = rng.permutation(np.where(y == 0)[0])
        per = n // n_hospitals
        want = np.array(pos_ratios) * per
        if want.sum() > len(pos):          # not enough tumor cases: scale down
            want = want * len(pos) / want.sum()
        splits = []
        for r_count in want:
            n_pos = min(int(r_count), len(pos))
            n_neg = min(per - n_pos, len(neg))
            splits.append(np.concatenate([pos[:n_pos], neg[:n_neg]]))
            pos, neg = pos[n_pos:], neg[n_neg:]
        return [rng.permutation(s) for s in splits]

    if mode == "size_skew":
        if sizes is None:  # one big hospital, the rest small
            sizes = np.array([0.5] + [0.5 / (n_hospitals - 1)] * (n_hospitals - 1))
        cuts = (np.cumsum(sizes)[:-1] * n).astype(int)
        return np.split(rng.permutation(n), cuts)

    raise ValueError(f"unknown mode: {mode}")


def save_hospitals(X, y, splits, out_dir):
    """One file per hospital, so each container only loads its own data."""
    os.makedirs(out_dir, exist_ok=True)
    for i, idx in enumerate(splits):
        np.savez_compressed(os.path.join(out_dir, f"hospital_{i}.npz"), X=X[idx], y=y[idx])


def describe(y, splits):
    for i, idx in enumerate(splits):
        print(f"hospital {i}: {len(idx):6d} images, tumor share {y[idx].mean():.2f}")


def plot_mix(y, splits, path):
    import matplotlib.pyplot as plt
    pos = [y[s].sum() for s in splits]
    neg = [len(s) - p for s, p in zip(splits, pos)]
    names = [f"H{i}" for i in range(len(splits))]
    plt.figure(figsize=(6, 4))
    plt.bar(names, neg, label="not tumor")
    plt.bar(names, pos, bottom=neg, label="tumor")
    plt.ylabel("images"); plt.legend(); plt.title("Data per hospital")
    plt.tight_layout(); plt.savefig(path, dpi=150)


if __name__ == "__main__":
    from data.load_data import load_pathmnist

    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="iid", choices=["iid", "label_skew", "size_skew"])
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    d = load_pathmnist(binary=True)
    splits = partition(d["y_train"], a.n, a.mode, a.seed)
    describe(d["y_train"], splits)

    out = f"data/hospitals/{a.mode}"
    save_hospitals(d["X_train"], d["y_train"], splits, out)
    # shared test set: no hospital trains on it
    np.savez_compressed("data/hospitals/test.npz", X=d["X_test"], y=d["y_test"])
    os.makedirs("results", exist_ok=True)
    plot_mix(d["y_train"], splits, f"results/hospitals_{a.mode}.png")
    print(f"saved to {out}/")
