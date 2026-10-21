"""

Run:  python -m data.explore
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from data.load_data import load_pathmnist, CLASS_NAMES

os.makedirs("results", exist_ok=True)
d = load_pathmnist(binary=False)
X, y = d["X_train"], d["y_train"]

# 1. Basic sizes
for split in ["train", "val", "test"]:
    print(f"{split}: {len(d['y_' + split])} images")

# 2. Class balance
counts = np.bincount(y, minlength=9)
names = [CLASS_NAMES[i] for i in range(9)]
plt.figure(figsize=(10, 4))
plt.bar(names, counts)
plt.xticks(rotation=45, ha="right")
plt.ylabel("number of images")
plt.title("PathMNIST training set: images per class")
plt.tight_layout()
plt.savefig("results/class_balance.png", dpi=150)
print("tumor share:", counts[8] / counts.sum())

# 3. Sample images (5 per class)
imgs = X.reshape(-1, 28, 28, 3)
fig, axes = plt.subplots(9, 5, figsize=(6, 11))
rng = np.random.default_rng(0)
for c in range(9):
    picks = rng.choice(np.where(y == c)[0], 5, replace=False)
    for j, i in enumerate(picks):
        axes[c, j].imshow(imgs[i])
        axes[c, j].axis("off")
    axes[c, 0].set_title(CLASS_NAMES[c], fontsize=7, loc="left")
plt.tight_layout()
plt.savefig("results/samples.png", dpi=150)

# 4. Pixel stats (useful for normalization later)
print("pixel mean:", X.mean(), "pixel std:", X.std())
print("Saved results/class_balance.png and results/samples.png")
