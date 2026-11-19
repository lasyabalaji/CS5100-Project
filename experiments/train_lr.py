"""
Train logistic regression without federation. This supports single hospital or centralized (all hospitals)

AI Disclosure: This code is mostly AI Generated

Run:
  python -m experiments.train_lr --setting centralized --epochs 30
  python -m experiments.train_lr --setting hospital --mode label_skew --hospital 3

AI Disclousre: I psuedo coded this and made AI fill it in. AI had to fix my psuedo code
"""
import argparse
import json
import os
import time

import numpy as np

from models.logistic_regression import LogisticRegression
from data.hospital_data import load_hospital, load_pooled, load_split



# Metrics

def f1(y_true, y_pred):
    """F1 for the tumor class (1). y_pred are 0/1 labels."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return float(2 * precision * recall / (precision + recall))


def auc(y_true, scores):
    """ROC AUC = chance a random tumor image scores higher than a random
    non-tumor image (Mann-Whitney U), with ties counted as half."""
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=np.float64)
    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    # average rank for tied scores
    _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)                 # rank of the last item in each tie group
    avg_rank = ends - (counts - 1) / 2.0     # 1-based average rank per group
    ranks = avg_rank[inverse]
    u = ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def load_train(setting, mode, hospital):
    if setting == "hospital":
        return load_hospital(mode, hospital)
    # centralized: pool every hospital back together (any split has the same images)
    return load_pooled(mode)


def evaluate(model, X, y):
    probs = model.predict(X)
    preds = (probs >= 0.5).astype(int)
    return f1(y, preds), auc(y, probs)


def run(args):
    X_train, y_train = load_train(args.setting, args.mode, args.hospital)
    X_val, y_val = load_split("val")
    X_test, y_test = load_split("test")

    print(f"setting={args.setting} mode={args.mode} hospital={args.hospital} "
          f"epochs={args.epochs} lr={args.lr} batch={args.batch_size} seed={args.seed}")
    print(f"train size {len(y_train)}, tumor share {y_train.mean():.3f}")

    pos_weight = (LogisticRegression.balanced_pos_weight(y_train)
                  if args.pos_weight == "balanced" else float(args.pos_weight))
    print(f"pos_weight {pos_weight:.2f}")
    model = LogisticRegression(X_train.shape[1], l2=args.l2, pos_weight=pos_weight)
    print(f"weights: {model.num_weights()} numbers, {model.weights_size_bytes()} bytes")

    history = []
    best = {"epoch": 0, "val_f1": -1.0, "weights": model.get_weights()}
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        # one epoch at a time, with a different shuffle each epoch
        loss = model.train(X_train, y_train, epochs=1, lr=args.lr,
                           batch_size=args.batch_size, seed=args.seed * 1000 + epoch)[0]
        secs = time.time() - t0
        val_f1, val_auc = evaluate(model, X_val, y_val)
        history.append({"epoch": epoch, "train_loss": loss, "val_f1": val_f1,
                        "val_auc": val_auc, "seconds": secs})
        mark = ""
        if val_f1 > best["val_f1"]:
            best = {"epoch": epoch, "val_f1": val_f1, "weights": model.get_weights()}
            mark = "  <- best so far"
        print(f"epoch {epoch:3d}  loss {loss:.4f}  val F1 {val_f1:.4f}  "
              f"val AUC {val_auc:.4f}  ({secs:.1f}s){mark}")

    model.set_weights(best["weights"])
    test_f1, test_auc = evaluate(model, X_test, y_test)
    print(f"\nbest epoch {best['epoch']} (val F1 {best['val_f1']:.4f})")
    print(f"TEST  F1 {test_f1:.4f}  AUC {test_auc:.4f}")

    name = args.setting if args.setting == "centralized" else f"hospital{args.hospital}_{args.mode}"
    out = os.path.join("results", f"lr_{name}_E{args.epochs}_lr{args.lr}_s{args.seed}")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump({"best_epoch": best["epoch"], "best_val_f1": best["val_f1"],
                   "test_f1": test_f1, "test_auc": test_auc,
                   "config": vars(args)}, f, indent=2)
    np.savez(os.path.join(out, "best_weights.npz"), *best["weights"])
    print(f"saved results to {out}/")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train logistic regression (centralized or one hospital)")
    p.add_argument("--setting", default="centralized", choices=["centralized", "hospital"])
    p.add_argument("--mode", default="iid", choices=["iid", "label_skew", "size_skew"])
    p.add_argument("--hospital", type=int, default=0, help="which hospital, for --setting hospital")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--l2", type=float, default=0.0)
    p.add_argument("--pos_weight", default="balanced",
                   help='weight on tumor images in the loss: a number, or "balanced" (#neg/#pos)')
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
