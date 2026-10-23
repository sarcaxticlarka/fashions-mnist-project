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
