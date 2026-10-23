"""Feature engineering: flattened pixels, HOG descriptors and hand-crafted shape statistics.

All functions take images of shape (N, 28, 28) scaled to [0, 1] and return a 2-D
float32 matrix (N, d).  Images are already upright (row 0 = top of the garment), so HOG
is computed directly on the (rows, cols) layout without any rotation/transpose.
"""
import numpy as np
from joblib import Parallel, delayed
from scipy import ndimage as ndi
from skimage.feature import hog

from .config import HOG_PARAMS


# --------------------------------------------------------------------------- pixels
def flatten_pixels(X):
    """Baseline representation: raw intensities in row-major order (784-d)."""
    return X.reshape(len(X), -1).astype(np.float32)


# --------------------------------------------------------------------------- HOG
def _hog_one(img, orientations, pixels_per_cell, cells_per_block, block_norm):
    return hog(
        img, orientations=orientations, pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block, block_norm=block_norm, feature_vector=True,
    )


def _hog_chunk(chunk, **kw):
    return np.stack([_hog_one(im, **kw) for im in chunk]).astype(np.float32)


def compute_hog(X, orientations=None, pixels_per_cell=None, cells_per_block=None,
                block_norm="L2-Hys", n_jobs=-1):
    """Histogram of Oriented Gradients for every image (parallelised over chunks)."""
    p = dict(HOG_PARAMS)
    kw = dict(
        orientations=orientations or p["orientations"],
        pixels_per_cell=pixels_per_cell or p["pixels_per_cell"],
        cells_per_block=cells_per_block or p["cells_per_block"],
        block_norm=block_norm,
    )
    chunks = np.array_split(X, max(1, min(64, len(X) // 500)))
    parts = Parallel(n_jobs=n_jobs)(delayed(_hog_chunk)(c, **kw) for c in chunks)
    return np.concatenate(parts)


def hog_with_image(img, **overrides):
    """HOG vector plus the visualisation image (for figures only)."""
    p = dict(HOG_PARAMS)
    p.update(overrides)
    return hog(img, orientations=p["orientations"], pixels_per_cell=p["pixels_per_cell"],
               cells_per_block=p["cells_per_block"], block_norm="L2-Hys", visualize=True)


# --------------------------------------------------------------------------- shape stats
STAT_NAMES = [
    "mean_intensity", "std_intensity", "ink_fraction", "bbox_height", "bbox_width",
    "aspect_ratio", "centroid_row", "centroid_col", "lr_symmetry", "tb_symmetry",
    "edge_density", "top_half_mass", "bottom_half_mass", "width_top", "width_bottom",
    "holes_between_legs",
]


def shape_stats(X, ink_thr=0.1):
    """16 interpretable global descriptors per image (see STAT_NAMES)."""
    n = len(X)
    out = np.zeros((n, len(STAT_NAMES)), dtype=np.float32)
    rows = np.arange(28, dtype=np.float32)
    for i in range(n):
        im = X[i]
        mask = im > ink_thr
        ink = mask.sum()
        total = im.sum() + 1e-8
        r_any, c_any = mask.any(1), mask.any(0)
        if ink:
            r_idx, c_idx = np.where(r_any)[0], np.where(c_any)[0]
            h, w = r_idx[-1] - r_idx[0] + 1, c_idx[-1] - c_idx[0] + 1
        else:
            h = w = 0
        cy = (im.sum(1) * rows).sum() / total
        cx = (im.sum(0) * rows).sum() / total
        lr = 1 - np.abs(im - im[:, ::-1]).sum() / (2 * total)
        tb = 1 - np.abs(im - im[::-1, :]).sum() / (2 * total)
        gx, gy = ndi.sobel(im, axis=1), ndi.sobel(im, axis=0)
        edge = (np.hypot(gx, gy) > 1.0).mean()
        top, bot = im[:14].sum() / total, im[14:].sum() / total
        wt = mask[:10].any(0).sum() if ink else 0  # horizontal extent of the top rows
        wb = mask[-10:].any(0).sum() if ink else 0
        # number of ink-free columns inside the bbox in the lower half (gap between trouser legs)
        low = mask[14:]
        if ink and low.any():
            cols = np.where(low.any(0))[0]
            gap = (~low.any(0))[cols[0]:cols[-1] + 1].sum()
        else:
            gap = 0
        out[i] = [im.mean(), im.std(), ink / 784, h, w, h / max(w, 1), cy, cx, lr, tb,
                  edge, top, bot, wt, wb, gap]
    return out


# Fixed (data-independent) divisors that bring every statistic to roughly [0, 1], so that the
# statistics do not dominate the HOG block in distance/kernel-based models.
STAT_SCALE = np.array([1, 1, 1, 28, 28, 3, 28, 28, 1, 1, 1, 1, 1, 28, 28, 28], dtype=np.float32)


# --------------------------------------------------------------------------- registry
FEATURE_SETS = ("pixels", "hog", "hog_stats")


def build_features(X, name):
    """Build a named feature matrix: 'pixels', 'hog' or 'hog_stats' (HOG + shape stats)."""
    if name == "pixels":
        return flatten_pixels(X)
    if name == "hog":
        return compute_hog(X)
    if name == "hog_stats":
        return np.hstack([compute_hog(X), shape_stats(X) / STAT_SCALE])
    raise ValueError(name)
