# Federated Learning for Tumor Detection

CS 5100 Final Project · Abdullah Ismail, Nour Shoreibah, Sree Lasya Balaji

## Overview

Hospitals want to train machine learning models together, but they cannot share patient
images because of privacy rules. Federated learning solves this: each hospital trains a
model on its own data and sends only the model weights to a central server, which
combines them into one global model. Patient data never leaves the hospital.

We simulate this with 4 mock hospitals. The task is to predict whether a colon tissue
image is **tumor** or **not tumor**. All models are written from scratch in NumPy.

## Dataset

We use **PathMNIST** from [MedMNIST](https://medmnist.com/): 28×28 color images of colon
tissue in 9 classes.

- About 90,000 training, 10,000 validation, and 7,200 test images
- Class 8 (colorectal adenocarcinoma epithelium) is labeled **tumor (1)**; all other
  classes are **not tumor (0)**
- About 14% of training images are tumor

The dataset is not stored in this repo. It downloads automatically the first time you
run the data loading script.

## Setup

```bash
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # Mac/Linux
pip install -r requirements.txt
```

## How to run

Run all commands from the repo's main folder.

**1. Load the data** (downloads PathMNIST on the first run)
```bash
python -m data.load_data
```

**2. Explore the data** (saves charts to `results/`)
```bash
python -m data.explore
```

**3. Make the mock hospitals**
```bash
python -m data.partition --mode iid --n 4
python -m data.partition --mode label_skew --n 4
python -m data.partition --mode size_skew --n 4
```

| Mode | What it does |
|---|---|
| `iid` | Random, equal split. Every hospital has about 14% tumor. |
| `label_skew` | Equal sizes, different tumor shares (about 26%, 18%, 10%, 2%). |
| `size_skew` | Same tumor share, but one hospital holds half the data. |

Each hospital's data is saved to its own file in `data/hospitals/<mode>/`. Two shared sets
that no hospital trains on are saved next to them: `data/hospitals/val.npz`, used to pick
the best epoch or round, and `data/hospitals/test.npz`, used once at the very end. The
split uses a fixed random seed, so everyone gets identical hospitals.

**4. Train one model on its own** (no federation, for the baselines)
```bash
python -m experiments.train_lr --setting centralized --epochs 30
python -m experiments.train_lr --setting hospital --mode label_skew --hospital 0
```

| Setting | What it trains on |
|---|---|
| `centralized` | Every hospital's data pooled back together. The best case. |
| `hospital` | One hospital's data only, chosen with `--hospital`. The worst case. |

Only ~14% of images are tumor, so by default the loss counts a tumor image
`#not tumor / #tumor` times more. Pass `--pos_weight 1` to turn that off.

**5. Federated training** (FedAvg, all 4 hospitals, simulated on one machine)
```bash
python -m experiments.simulate_fedavg --mode iid --model lr --E 1 --rounds 50
```

`--E` is how many epochs each hospital trains before sending its weights back.
`--normalize channel`, the default, standardizes the pixels per colour channel using six
fixed numbers; without it both models predict "not tumor" for every image. `--model nn`
uses `models/neural_net.py`. `--model lr` still builds the stand-in in `fed/temp_model.py`
and trains on raw pixels in step 4, so the two are not comparable yet; both are one small
change in `make_model()`. Early numbers are in `results/preliminary_results.md`.

**6. Run the tests**
```bash
python -m pytest
```

Every run in steps 4 and 5 writes its metrics and best weights to its own folder under
`results/`.

## Code organization

```
data/
  load_data.py      loads PathMNIST and creates tumor / not tumor labels
  explore.py        class balance and sample image charts
  partition.py      splits training data into mock hospitals
  hospital_data.py  reads the hospital files back off disk
models/
  logistic_regression.py  logistic regression, with an optional weight on tumor images
  neural_net.py           two hidden layers, ReLU, written out by hand
fed/
  aggregators.py    FedAvg: averages hospital weights, bigger hospitals count more
  temp_model.py     TEMPORARY stand-in, delete once make_model() uses models/
experiments/
  train_lr.py          one model on its own: centralized or a single hospital
  simulate_fedavg.py   the federated loop, simulated on one machine
results/            charts, metrics, and saved weights from every run
requirements.txt    Python packages
```

## Project status

- [x] Dataset chosen, loaded, and explored
- [x] Mock hospitals (iid, label skew, size skew)
- [x] NumPy models: logistic regression and neural network
- [ ] Federated server and hospital clients (Docker)
- [ ] Experiments: federated vs. single hospital vs. centralized
