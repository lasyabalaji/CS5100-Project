"""
Federated training with FedAvg, simulated on one machine (no networking).

Each round:
  1. the server sends the current global weights to all 4 hospitals
  2. each hospital trains E local epochs on its own data only
  3. each hospital sends back its new weights and its number of images
  4. the server combines them with FedAvg (bigger hospitals count more)
  5. the server checks validation F1 of the new global model; the best round is kept
At the end, the best round's weights are evaluated once on the test set (F1 and AUC).

Run:  python -m experiments.simulate_fedavg --mode iid --model lr --E 1 --rounds 3
"""
import argparse
import csv
import json
import os
import time

import numpy as np

from fed.aggregators import fedavg

HOSPITAL_DIR = "data/hospitals"
N_HOSPITALS = 4

# ---------------------------------------------------------------------------
# Metrics: use Abdullah's models/metrics.py if it exists, else the temporary ones.
# ---------------------------------------------------------------------------
try:
    from models.metrics import auc, f1  # expected: f1(y_true, y_pred), auc(y_true, scores)
    METRICS_SOURCE = "models.metrics"
except ImportError:
    METRICS_SOURCE = "TEMPORARY (simulate_fedavg.py)"

    # ===== TEMPORARY: replace with models/metrics.py when Abdullah adds it =====
    def f1(y_true, y_pred):
        """TEMPORARY. F1 for the tumor class (1). y_pred are 0/1 labels."""
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
        """TEMPORARY. ROC AUC = chance a random tumor image scores higher than a
        random non-tumor image (Mann-Whitney U), with ties counted as half."""
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
    # ===== end TEMPORARY =======================================================


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
def make_model(name, n_features):
    """TEMPORARY: uses the stand-in model in fed/temp_model.py.
    Swap these lines for Abdullah's classes once models/ exists."""
    if name == "lr":
        from fed.temp_model import TempLogisticRegression
        return TempLogisticRegression(n_features)
    if name == "nn":
        raise NotImplementedError(
            "The neural network is not in the repo yet (models/ is missing). "
            "Hook it up in make_model() once Abdullah adds it.")
    raise ValueError(f"unknown model: {name}")


def local_train(model, X, y, epochs, lr, batch_size, seed):
    """One hospital trains on its own data. Kept in one place so it is easy to
    match Abdullah's real train() signature later."""
    return model.train(X, y, epochs=epochs, lr=lr, batch_size=batch_size, seed=seed)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_npz(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run: python -m data.partition --mode <mode> --n 4")
    d = np.load(path)
    return d["X"], d["y"]


def load_hospitals(mode):
    return [load_npz(os.path.join(HOSPITAL_DIR, mode, f"hospital_{i}.npz")) for i in range(N_HOSPITALS)]


def evaluate(model, X, y):
    probs = model.predict(X)
    preds = (probs >= 0.5).astype(int)   # tumor if probability >= 0.5
    return f1(y, preds), auc(y, probs)


# ---------------------------------------------------------------------------
# Federated training
# ---------------------------------------------------------------------------
def run(args):
    hospitals = load_hospitals(args.mode)
    X_val, y_val = load_npz(os.path.join(HOSPITAL_DIR, "val.npz"))
    X_test, y_test = load_npz(os.path.join(HOSPITAL_DIR, "test.npz"))
    sizes = [len(y) for _, y in hospitals]
    n_features = hospitals[0][0].shape[1]

    print(f"mode={args.mode} model={args.model} E={args.E} rounds={args.rounds} "
          f"lr={args.lr} batch={args.batch_size} seed={args.seed}")
    print(f"hospital sizes: {sizes}   metrics: {METRICS_SOURCE}")

    np.random.seed(args.seed)
    global_model = make_model(args.model, n_features)   # the server's model
    local_model = make_model(args.model, n_features)    # reused by each hospital in turn

    history = []
    best = {"round": 0, "val_f1": -1.0, "weights": global_model.get_weights()}

    for r in range(1, args.rounds + 1):
        t0 = time.time()
        global_weights = global_model.get_weights()
        new_weights, local_losses = [], []

        for k, (X_k, y_k) in enumerate(hospitals):
            local_model.set_weights(global_weights)                   # 1. server -> hospital
            seed_k = args.seed * 100_000 + r * 100 + k                # different shuffle per round/hospital
            losses = local_train(local_model, X_k, y_k, args.E,       # 2. local training
                                 args.lr, args.batch_size, seed_k)
            new_weights.append(local_model.get_weights())             # 3. hospital -> server
            local_losses.append(float(losses[-1]) if losses else float("nan"))

        global_model.set_weights(fedavg(new_weights, sizes))          # 4. FedAvg

        val_f1, val_auc = evaluate(global_model, X_val, y_val)        # 5. validation
        secs = time.time() - t0
        history.append({"round": r, "val_f1": val_f1, "val_auc": val_auc,
                        "mean_local_loss": float(np.mean(local_losses)), "seconds": secs})
        mark = ""
        if val_f1 > best["val_f1"]:
            best = {"round": r, "val_f1": val_f1, "weights": global_model.get_weights()}
            mark = "  <- best so far"
        print(f"round {r:3d}  local loss {np.mean(local_losses):.4f}  "
              f"val F1 {val_f1:.4f}  val AUC {val_auc:.4f}  ({secs:.1f}s){mark}")

    # final test evaluation, once, with the best round's weights
    global_model.set_weights(best["weights"])
    test_f1, test_auc = evaluate(global_model, X_test, y_test)
    print(f"\nbest round {best['round']} (val F1 {best['val_f1']:.4f})")
    print(f"TEST  F1 {test_f1:.4f}  AUC {test_auc:.4f}")

    save_results(args, sizes, history, best, test_f1, test_auc)


def save_results(args, sizes, history, best, test_f1, test_auc):
    run_name = args.run_name or (f"fedavg_{args.mode}_{args.model}_E{args.E}_R{args.rounds}"
                                 f"_lr{args.lr}_s{args.seed}")
    out = os.path.join("results", run_name)
    os.makedirs(out, exist_ok=True)

    config = {k: v for k, v in vars(args).items()}
    config.update({"run_name": run_name, "hospital_sizes": sizes, "metrics": METRICS_SOURCE})
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    with open(os.path.join(out, "rounds.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        w.writeheader()
        w.writerows(history)

    summary = {"best_round": best["round"], "best_val_f1": best["val_f1"],
               "test_f1": test_f1, "test_auc": test_auc}
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    np.savez(os.path.join(out, "best_weights.npz"), *best["weights"])
    print(f"saved results to {out}/")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Simulated federated training with FedAvg")
    p.add_argument("--mode", default="iid", choices=["iid", "label_skew", "size_skew"])
    p.add_argument("--model", default="lr", choices=["lr", "nn"])
    p.add_argument("--E", type=int, default=1, help="local epochs per round")
    p.add_argument("--rounds", type=int, default=50)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--run_name", default=None, help="folder name under results/ (auto if not given)")
    a = p.parse_args(argv)
    if a.E < 1 or a.rounds < 1:
        p.error("--E and --rounds must be at least 1")
    return a


if __name__ == "__main__":
    run(parse_args())
