"""
Load PathMNIST as plain numpy arrays.

We read the official pathmnist.npz file directly with numpy instead of using
medmnist's PathMNIST class, because that class requires PyTorch.

Run:  python -m data.load_data
"""
import hashlib
import os
import urllib.request
import numpy as np

# Official file + checksum (from medmnist.info.INFO["pathmnist"])
URL = "https://zenodo.org/records/10519652/files/pathmnist.npz?download=1"
MD5 = "a8b06965200029087d5bd730944a56c1"
CACHE = os.path.join(os.path.expanduser("~"), ".medmnist", "pathmnist.npz")

# PathMNIST class names (checked against medmnist.info.INFO["pathmnist"]["label"])
CLASS_NAMES = {
    0: "adipose", 1: "background", 2: "debris", 3: "lymphocytes",
    4: "mucus", 5: "smooth muscle", 6: "normal mucosa",
    7: "cancer stroma", 8: "tumor epithelium",
}
TUMOR_CLASS = 8  # used for the binary "tumor vs not tumor" version


def download(path=CACHE):
    """Download pathmnist.npz once (about 200 MB) and verify its checksum."""
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"downloading PathMNIST to {path} ...")
    urllib.request.urlretrieve(URL, path)
    with open(path, "rb") as f:
        if hashlib.md5(f.read()).hexdigest() != MD5:
            os.remove(path)
            raise IOError("download corrupted (MD5 mismatch), please run again")
    return path


def load_pathmnist(binary=True):
    """Returns dict with X_train, y_train, X_val, y_val, X_test, y_test.

    X: float32 in [0, 1], shape (N, 28*28*3)  -- flattened for numpy models
    y: int labels. If binary=True, 1 = tumor, 0 = everything else.
    """
    raw = np.load(download())

    out = {}
    for split in ["train", "val", "test"]:
        X = raw[f"{split}_images"].astype(np.float32) / 255.0  # (N, 28, 28, 3)
        X = X.reshape(len(X), -1)                              # (N, 2352)
        y = raw[f"{split}_labels"].ravel().astype(np.int64)
        if binary:
            y = (y == TUMOR_CLASS).astype(np.int64)
        out[f"X_{split}"], out[f"y_{split}"] = X, y
    return out


if __name__ == "__main__":
    d = load_pathmnist(binary=False)
    for split in ["train", "val", "test"]:
        X, y = d[f"X_{split}"], d[f"y_{split}"]
        print(f"{split:5s}  X={X.shape}  counts per class={np.bincount(y)}")
