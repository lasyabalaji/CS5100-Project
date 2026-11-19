# Preliminary results (check-in 2)

Federated training with FedAvg, simulated on one machine with 4 hospitals.
These are **early, short runs**, not the final experiments in `experimental_design.pdf`.

## Setup
| Setting | Value |
|---|---|
| Algorithm | FedAvg (hospitals with more images count more) |
| Hospitals | 4 |
| Local epochs per round (E) | 1 |
| Rounds | 10 (the final plan uses E x R = 50) |
| Learning rate | 0.01 |
| Batch size | 64 |
| Seed | 0 (the final plan uses seeds 0, 1, 2) |
| Inputs | pixels standardized per colour channel (see below) |
| Model kept | the round with the best validation F1 |
| Metrics | F1 for the tumor class (probability >= 0.5), and ROC AUC |

Command (one per split and model):
`python -m experiments.simulate_fedavg --mode <split> --model <lr|nn> --E 1 --rounds 10`

## Results

### Neural network (two hidden layers, 128 and 64 units)
| Split | Hospital sizes | Best round | Val F1 | Test F1 | Test AUC |
|---|---|---|---|---|---|
| iid | 22,499 x 4 | 10 | 0.513 | 0.641 | 0.929 |
| label_skew | 22,499 x 3, 22,496 | 9 | 0.466 | 0.598 | 0.927 |
| size_skew | 44,998 / 14,999 / 14,999 / 15,000 | 10 | 0.536 | 0.648 | 0.931 |

### Logistic regression (TEMPORARY stand-in, `fed/temp_model.py`)
| Split | Best round | Val F1 | Test F1 | Test AUC |
|---|---|---|---|---|
| iid | 1 | 0.319 | 0.436 | 0.722 |
| label_skew | 1 | 0.203 | 0.359 | 0.763 |
| size_skew | 1 | 0.271 | 0.408 | 0.730 |

## What we see so far
- **The federated NN learns on every split.** Validation F1 was still rising at round 10
  on all three splits, so longer runs (50 local epochs) should do better.
- **label_skew is the hardest split** for the NN (val F1 0.466 vs 0.513 for iid), as
  expected when hospitals see very different tumor rates. size_skew did not hurt.
- **The NN beats the temporary LR by a wide margin** (test AUC about 0.93 vs about 0.73).
- **The temporary LR is unstable at learning rate 0.01**: its best round is always
  round 1, and its validation F1 then falls to about 0 (it predicts "not tumor" for
  almost every image). The real logistic regression should be tested with a smaller
  learning rate (0.001) in Phase 1.
- Test F1 is higher than validation F1 for every run. The test set has more tumor
  images (17.2% vs 14.3%) and comes from different slides (see `data_summary.md`).

## Important finding: inputs must be standardized
On raw pixels (values in [0, 1], all positive, mean about 0.66), the NN predicted
"not tumor" for every image: **val F1 = 0.000** after 3 federated rounds, compared with
0.374 with standardization (iid, E=1, same seed). We now subtract each colour channel's
mean and divide by its std, using the training-set values in `data_summary.md`
(R, G, B mean 0.741 / 0.533 / 0.706, std 0.124 / 0.177 / 0.124). These are 6 fixed
numbers applied the same way at every hospital, so no hospital's data is shared.
**Centralized and single-hospital runs must use the same preprocessing**, or the
comparison will not be fair.

## Caveats
- One seed, 10 rounds only, E = 1 only. No centralized or single-hospital baselines yet.
- LR results use a temporary model; the real one is not pushed yet.
- F1 and AUC come from temporary functions in `experiments/simulate_fedavg.py` until
  `models/metrics.py` exists.

## Next
- Phase 1: learning-rate check for both models (centralized, about 10 epochs).
- Phase 2: E = 1, 5, 10 for each split (50 local epochs per hospital).
- Phase 3: centralized and single-hospital baselines, seeds 0, 1, 2.
- Networked server, hospital clients, and Docker.
