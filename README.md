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

Each hospital's data is saved to its own file in `data/hospitals/<mode>/`. A shared test
set that no hospital trains on is saved to `data/hospitals/test.npz`. The split uses a
fixed random seed, so everyone gets identical hospitals.

## Code organization

```
data/
  load_data.py      loads PathMNIST and creates tumor / not tumor labels
  explore.py        class balance and sample image charts
  partition.py      splits training data into mock hospitals
models/             NumPy models written from scratch (in progress)
experiments/        tests and experiments (in progress)
requirements.txt    Python packages
```

## Project status

- [x] Dataset chosen, loaded, and explored
- [x] Mock hospitals (iid, label skew, size skew)
- [ ] NumPy models: logistic regression and neural network
- [ ] Federated server and hospital clients (Docker)
- [ ] Experiments: federated vs. single hospital vs. centralized
