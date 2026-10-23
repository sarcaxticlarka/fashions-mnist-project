"""Feature-engineering experiments: HOG ablation, feature-set comparison, PCA, importance."""
import time
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import CLASS_NAMES, TABLE_DIR, SEED
from .features import compute_hog, flatten_pixels, shape_stats, STAT_NAMES, hog_with_image
from .plotting import save, COLORS


def _fast_model():
    """Probe classifier used for ablations (fast, deterministic, same for all inputs)."""
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=200, random_state=SEED))


def hog_ablation(S, n_train=12000):
    """Grid over orientations x cell size x block size; scored on the validation set."""
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(S.X_tr), n_train, replace=False)
    Xa, ya = S.X_tr[idx], S.y_tr[idx]
    rows = []
    grid = itertools.product([6, 9, 12], [(2, 2), (4, 4), (7, 7)], [(1, 1), (2, 2)])
    for o, ppc, cpb in grid:
        t0 = time.time()
        Fa = compute_hog(Xa, o, ppc, cpb)
        Fv = compute_hog(S.X_val, o, ppc, cpb)
        t_feat = time.time() - t0
        m = _fast_model().fit(Fa, ya)
        rows.append(dict(orientations=o, pixels_per_cell=f"{ppc[0]}x{ppc[1]}", cells_per_block=f"{cpb[0]}x{cpb[1]}",
                         dim=Fa.shape[1], val_acc=m.score(Fv, S.y_val), feat_time_s=round(t_feat, 1)))
        print(rows[-1], flush=True)
    df = pd.DataFrame(rows).sort_values("val_acc", ascending=False).reset_index(drop=True)
    df.to_csv(TABLE_DIR / "fe_hog_ablation.csv", index=False)
    return df


def plot_hog_ablation(df):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    d = df.copy(); d["cells"] = d["pixels_per_cell"] + " / blk " + d["cells_per_block"]
    sns.barplot(data=d, x="orientations", y="val_acc", ax=axes[0], color="#DD8452", errorbar=None); axes[0].set_title("Effect of orientation bins")
    sns.barplot(data=d, x="pixels_per_cell", y="val_acc", ax=axes[1], color="#DD8452", errorbar=None); axes[1].set_title("Effect of cell size")
    sns.barplot(data=d, x="cells_per_block", y="val_acc", ax=axes[2], color="#DD8452", errorbar=None); axes[2].set_title("Effect of block size")
    lo = d.val_acc.min() - 0.01
    for ax in axes: ax.set_ylim(lo, d.val_acc.max() + 0.01); ax.set_ylabel("validation accuracy (mean over grid)")
    save(fig, "fe_01_hog_ablation")


def hog_visual(X, y):
    """Original image next to its HOG visualisation, one per class."""
    from skimage import exposure
    fig, axes = plt.subplots(2, 10, figsize=(15, 3.6))
    for c in range(10):
        i = np.where(y == c)[0][0]
        _, h = hog_with_image(X[i])
        axes[0, c].imshow(X[i], cmap="gray"); axes[0, c].set_title(CLASS_NAMES[c], fontsize=8)
        axes[1, c].imshow(exposure.rescale_intensity(h, in_range=(0, h.max() + 1e-9)), cmap="gray")
        axes[0, c].axis("off"); axes[1, c].axis("off")
    fig.suptitle("Image (top) and its HOG gradient-orientation map (bottom)", fontweight="bold")
    save(fig, "fe_02_hog_visual")


def shape_stats_analysis(X, y):
    """Distribution of the hand-crafted statistics per class + RF importance."""
    F = shape_stats(X[:20000]); yy = y[:20000]
    df = pd.DataFrame(F, columns=STAT_NAMES); df["class"] = [CLASS_NAMES[i] for i in yy]
    z = (df[STAT_NAMES] - df[STAT_NAMES].mean()) / df[STAT_NAMES].std()
    z["class"] = df["class"]
    prof = z.groupby("class").mean().loc[CLASS_NAMES]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    sns.heatmap(prof, cmap="vlag", center=0, annot=True, fmt=".1f", annot_kws={"size": 7}, ax=ax)
    ax.set_title("Hand-crafted shape statistics: class profile (z-score)")
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
    save(fig, "fe_03_shape_stats_profile")
    rf = RandomForestClassifier(200, n_jobs=-1, random_state=SEED).fit(F, yy)
    imp = pd.Series(rf.feature_importances_, index=STAT_NAMES).sort_values()
    fig, ax = plt.subplots(figsize=(6, 4.5)); imp.plot.barh(ax=ax, color="#55A868")
    ax.set_title("Random-Forest importance of shape statistics"); ax.set_xlabel("importance")
    save(fig, "fe_04_shape_stats_importance")
    imp.sort_values(ascending=False).to_csv(TABLE_DIR / "fe_shape_stats_importance.csv", header=["importance"])
    return imp


def pca_dimension_sweep(S, ks=(10, 20, 30, 50, 100, 200)):
    """Is a compact linear projection of pixels enough?  Probe with logistic regression."""
    rows = []
    flat_tr, flat_val = flatten_pixels(S.X_tr), flatten_pixels(S.X_val)
    for k in ks:
        pca = PCA(k, random_state=SEED).fit(flat_tr)
        m = _fast_model().fit(pca.transform(flat_tr), S.y_tr)
        rows.append(dict(n_components=k, val_acc=m.score(pca.transform(flat_val), S.y_val)))
        print(rows[-1], flush=True)
    m = _fast_model().fit(flat_tr, S.y_tr)
    rows.append(dict(n_components=784, val_acc=m.score(flat_val, S.y_val)))
    df = pd.DataFrame(rows); df.to_csv(TABLE_DIR / "fe_pca_sweep.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 3.8)); ax.plot(df.n_components, df.val_acc, "o-")
    ax.set_xscale("log"); ax.set_xlabel("PCA components (pixels)"); ax.set_ylabel("validation accuracy (LR probe)")
    ax.set_title("PCA dimensionality sweep"); save(fig, "fe_05_pca_sweep")
    return df


def feature_set_comparison(S, F):
    """F: dict name -> (F_tr, F_val).  Same probe model for all feature sets."""
    rows = []
    for name, (a, b) in F.items():
        t0 = time.time(); m = _fast_model().fit(a, S.y_tr)
        pred = m.predict(b)
        rows.append(dict(feature_set=name, dim=a.shape[1], val_acc=(pred == S.y_val).mean(), fit_s=round(time.time() - t0, 1)))
        print(rows[-1], flush=True)
    df = pd.DataFrame(rows); df.to_csv(TABLE_DIR / "fe_feature_set_probe.csv", index=False)
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.bar(df.feature_set, df.val_acc, color=[COLORS[n] for n in df.feature_set])
    ax.set_ylim(df.val_acc.min() - 0.03, df.val_acc.max() + 0.01); ax.set_ylabel("validation accuracy")
    for i, v in enumerate(df.val_acc): ax.text(i, v + 0.001, f"{v:.3f}", ha="center")
    ax.set_title("Feature sets with the same LR probe"); save(fig, "fe_06_feature_set_probe")
    return df
