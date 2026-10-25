"""Loading Fashion-MNIST from the official IDX files and building the fair split.

Protocol (fair comparison):
  * official 60,000 / 10,000 train / test split is kept untouched;
  * the validation set (20 %) is drawn ONLY from the training set, stratified;
  * the test set is used exactly once, for the final report.
"""
import gzip
from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import train_test_split

from .config import RAW_DIR, SEED, VAL_SIZE, IMG_SIZE


def _read_images(path):
    with gzip.open(path, "rb") as f:
        buf = f.read()
    # IDX3 header: magic(4) n(4) rows(4) cols(4) -> 16 bytes. Row-major, upright images.
    return np.frombuffer(buf, dtype=np.uint8, offset=16).reshape(-1, IMG_SIZE, IMG_SIZE)


def _read_labels(path):
    with gzip.open(path, "rb") as f:
        buf = f.read()
    return np.frombuffer(buf, dtype=np.uint8, offset=8)


def load_raw():
    """Return uint8 images (N,28,28) and int labels for the official train/test split."""
    X_train = _read_images(RAW_DIR / "train-images-idx3-ubyte.gz")
    y_train = _read_labels(RAW_DIR / "train-labels-idx1-ubyte.gz").astype(np.int64)
    X_test = _read_images(RAW_DIR / "t10k-images-idx3-ubyte.gz")
    y_test = _read_labels(RAW_DIR / "t10k-labels-idx1-ubyte.gz").astype(np.int64)
    return X_train, y_train, X_test, y_test


def to_unit(X_uint8):
    """Scale pixels from [0,255] to [0,1] (float32). Labels are never touched."""
    return X_uint8.astype(np.float32) / 255.0


@dataclass
class Splits:
    X_tr: np.ndarray
    y_tr: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray

    @property
    def X_trainval(self):
        return np.concatenate([self.X_tr, self.X_val])

    @property
    def y_trainval(self):
        return np.concatenate([self.y_tr, self.y_val])


def make_splits(val_size=VAL_SIZE, seed=SEED):
    """Official split + stratified validation set carved out of the training data only."""
    X_train, y_train, X_test, y_test = load_raw()
    X_train, X_test = to_unit(X_train), to_unit(X_test)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=val_size, stratify=y_train, random_state=seed
    )
    return Splits(X_tr, y_tr, X_val, y_val, X_test, y_test)


def check_splits(S, expected_classes=10):
    """Sanity-check the splits and return a small report (raises AssertionError on a problem).

    Checks: expected sizes, pixel range [0, 1], all classes present everywhere,
    class proportions of train/validation match (stratified), and that no image is
    shared between train, validation and test (no data leakage).
    """
    report = {
        "train": len(S.X_tr), "val": len(S.X_val), "test": len(S.X_test),
        "pixel_min": float(min(S.X_tr.min(), S.X_val.min(), S.X_test.min())),
        "pixel_max": float(max(S.X_tr.max(), S.X_val.max(), S.X_test.max())),
    }
    assert report["train"] + report["val"] == 60000, "train + val must equal the 60,000 official training images"
    assert report["test"] == 10000, "test must be the official 10,000 images"
    assert 0.0 <= report["pixel_min"] and report["pixel_max"] <= 1.0, "pixels must be scaled to [0, 1]"

    for name, y in (("train", S.y_tr), ("val", S.y_val), ("test", S.y_test)):
        assert len(np.unique(y)) == expected_classes, f"{name}: missing classes"

    # stratification: each class share in train and validation should be (almost) identical
    share_tr = np.bincount(S.y_tr, minlength=expected_classes) / len(S.y_tr)
    share_val = np.bincount(S.y_val, minlength=expected_classes) / len(S.y_val)
    report["max_class_share_gap"] = float(np.abs(share_tr - share_val).max())
    assert report["max_class_share_gap"] < 0.002, "validation is not stratified like train"

    # leakage: hash every image and make sure the three sets are disjoint
    def hashes(X):
        return {hash(im.tobytes()) for im in X}
    h_tr, h_val, h_te = hashes(S.X_tr), hashes(S.X_val), hashes(S.X_test)
    report["overlap_train_test"] = len(h_tr & h_te)
    report["overlap_val_test"] = len(h_val & h_te)
    assert report["overlap_val_test"] == 0, "validation images leak into the test set"
    return report