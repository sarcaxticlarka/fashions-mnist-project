"""Exploratory data analysis: every function saves a figure and/or returns a tidy table."""
import hashlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.feature_selection import f_classif
from sklearn.manifold import TSNE

from .config import CLASS_NAMES, TABLE_DIR, SEED
from .plotting import save, PALETTE


# ------------------------------------------------------------------ 1. data quality
def quality_report(X_train, y_train, X_test, y_test):
    """Shape, range, missing values, blank images, duplicates (within and across splits)."""
    def row_hashes(X):
        return [hashlib.md5(x.tobytes()).hexdigest() for x in X]

    h_tr, h_te = row_hashes(X_train), row_hashes(X_test)
    s_tr, s_te = set(h_tr), set(h_te)
    # duplicates with *different* labels would be label noise
    seen, conflicts = {}, 0
    for h, y in zip(h_tr, y_train):
        if h in seen and seen[h] != y:
            conflicts += 1
        seen.setdefault(h, y)
    rep = {
        "train_shape": str(X_train.shape), "test_shape": str(X_test.shape),
        "dtype": str(X_train.dtype), "pixel_min": int(X_train.min()), "pixel_max": int(X_train.max()),
        "missing_values": int(np.isnan(X_train.astype(float)).sum()),
        "n_classes": len(np.unique(y_train)),
        "blank_images_train": int((X_train.reshape(len(X_train), -1).max(1) == 0).sum()),
        "exact_duplicates_in_train": len(h_tr) - len(s_tr),
        "train_test_overlap": len(s_tr & s_te),
        "duplicate_label_conflicts": conflicts,
        "mean_pixel_train": round(float(X_train.mean()) / 255, 4),
        "std_pixel_train": round(float(X_train.std()) / 255, 4),
        "sparsity_zero_pixels_%": round(100 * float((X_train == 0).mean()), 2),
    }
    df = pd.DataFrame(rep.items(), columns=["check", "value"])
    df.to_csv(TABLE_DIR / "eda_quality_report.csv", index=False)
    return df


# ------------------------------------------------------------------ 2. class balance
def class_distribution(y_train, y_test, y_val=None):
    tr = np.bincount(y_train, minlength=10)
    te = np.bincount(y_test, minlength=10)
    df = pd.DataFrame({"class": CLASS_NAMES, "train": tr, "test": te})
    df["train_%"] = (100 * tr / tr.sum()).round(2)
    df["test_%"] = (100 * te / te.sum()).round(2)
    df.to_csv(TABLE_DIR / "eda_class_distribution.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 3.8))
    x = np.arange(10)
    ax.bar(x - 0.2, tr / tr.sum() * 100, 0.4, label="train (60k)", color="#4C72B0")
    ax.bar(x + 0.2, te / te.sum() * 100, 0.4, label="test (10k)", color="#DD8452")
    ax.axhline(10, ls="--", c="grey", lw=1)
    ax.set_xticks(x, CLASS_NAMES, rotation=35, ha="right")
    ax.set_ylabel("share of split (%)")
    ax.set_title("Class distribution: perfectly balanced (10 % each)")
    ax.legend()
    save(fig, "eda_01_class_distribution")
    return df


# ------------------------------------------------------------------ 3. samples
def sample_grid(X, y, n=10, seed=SEED):
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(10, n, figsize=(n * 0.9, 10 * 0.95))
    for c in range(10):
        idx = rng.choice(np.where(y == c)[0], n, replace=False)
        for j, i in enumerate(idx):
            ax = axes[c, j]
            ax.imshow(X[i], cmap="gray")
            ax.axis("off")
        axes[c, 0].set_title(CLASS_NAMES[c], loc="left", fontsize=8, pad=2)
    fig.suptitle("Random samples per class", y=0.995, fontweight="bold")
    save(fig, "eda_02_sample_grid")


def mean_std_images(X, y):
    fig, axes = plt.subplots(2, 10, figsize=(14, 3.4))
    for c in range(10):
        Xc = X[y == c]
        axes[0, c].imshow(Xc.mean(0), cmap="gray")
        axes[1, c].imshow(Xc.std(0), cmap="magma")
        axes[0, c].set_title(CLASS_NAMES[c], fontsize=8)
        axes[0, c].axis("off"); axes[1, c].axis("off")
    axes[0, 0].text(-6, 14, "mean", rotation=90, va="center", fontsize=9)
    axes[1, 0].text(-6, 14, "std", rotation=90, va="center", fontsize=9)
    fig.suptitle("Per-class mean image (top) and pixel standard deviation (bottom)", fontweight="bold")
    save(fig, "eda_03_mean_std_images")


# ------------------------------------------------------------------ 4. intensity & shape
def per_image_stats(X, y):
    """Brightness, ink coverage and bounding box per image -> DataFrame."""
    flat = X.reshape(len(X), -1)
    mask = X > 0.1
    rows, cols = mask.any(2), mask.any(1)
    h = rows.sum(1)
    w = cols.sum(1)
    df = pd.DataFrame({
        "label": y, "class": [CLASS_NAMES[i] for i in y],
        "mean_intensity": flat.mean(1), "ink_fraction": mask.reshape(len(X), -1).mean(1),
        "bbox_height": h, "bbox_width": w,
    })
    return df


def intensity_and_shape_plots(df):
    order = CLASS_NAMES
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, col, t in zip(axes, ["mean_intensity", "ink_fraction", "bbox_width"],
                          ["Mean pixel intensity", "Ink coverage (pixels > 0.1)", "Bounding-box width (px)"]):
        sns.boxplot(data=df, y="class", x=col, order=order, ax=ax, fliersize=1, palette="tab10", hue="class", legend=False)
        ax.set_title(t); ax.set_ylabel("")
    save(fig, "eda_04_intensity_shape_boxplots")
    summ = df.groupby("class")[["mean_intensity", "ink_fraction", "bbox_height", "bbox_width"]].agg(["mean", "std"]).round(3)
    summ.columns = ["_".join(c) for c in summ.columns]
    summ = summ.loc[order]
    summ.to_csv(TABLE_DIR / "eda_class_summary_stats.csv")
    return summ


def pixel_histogram(X, y):
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    v = X.ravel()
    axes[0].hist(v, bins=64, color="#4C72B0", log=True)
    axes[0].set_title("Pixel intensity histogram (log-count)"); axes[0].set_xlabel("intensity [0,1]")
    for c in [0, 1, 5, 8]:
        axes[1].hist(X[y == c].ravel(), bins=50, histtype="step", lw=1.5, density=True, label=CLASS_NAMES[c])
    axes[1].set_yscale("log"); axes[1].legend(); axes[1].set_title("Per-class intensity density (selected)")
    axes[1].set_xlabel("intensity [0,1]")
    save(fig, "eda_05_pixel_histogram")


# ------------------------------------------------------------------ 5. pixel relevance
def pixel_variance_and_fscore(X, y):
    flat = X.reshape(len(X), -1)
    var = flat.var(0).reshape(28, 28)
    F, _ = f_classif(flat, y)
    F = np.nan_to_num(F).reshape(28, 28)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, m, t in zip(axes, [flat.mean(0).reshape(28, 28), var, F],
                        ["Mean pixel value", "Pixel variance", "ANOVA F-score (class separability)"]):
        im = ax.imshow(m, cmap="viridis"); ax.set_title(t, fontsize=10); ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046)
    save(fig, "eda_06_pixel_variance_fscore")
    low_var = int((flat.var(0) < 1e-3).sum())
    return {"near_constant_pixels(var<1e-3)": low_var}


def spatial_correlation(X, seed=SEED):
    """Neighbouring pixels are strongly correlated -> structure that flattening ignores."""
    rng = np.random.default_rng(seed)
    flat = X.reshape(len(X), -1)
    keep = flat.std(0) > 0.01
    corr = np.corrcoef(flat[:, keep].T)
    idx = np.where(keep)[0]
    coords = np.stack([idx // 28, idx % 28], 1)
    d = np.sqrt(((coords[:, None] - coords[None]) ** 2).sum(-1))
    dists = np.arange(1, 15)
    mean_corr = [corr[(d > k - 0.5) & (d <= k + 0.5)].mean() for k in dists]
    # null: same pixels, shuffled coordinates
    perm = rng.permutation(len(idx))
    d_null = d[perm][:, perm]
    null = [corr[(d_null > k - 0.5) & (d_null <= k + 0.5)].mean() for k in dists]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(dists, mean_corr, "o-", label="actual image grid")
    ax.plot(dists, null, "s--", label="shuffled pixel positions")
    ax.set_xlabel("distance between two pixels (px)"); ax.set_ylabel("mean Pearson correlation")
    ax.set_title("Spatial correlation: neighbourhood structure exists"); ax.legend()
    save(fig, "eda_07_spatial_correlation")
    return pd.DataFrame({"distance": dists, "corr_actual": mean_corr, "corr_shuffled": null})


# ------------------------------------------------------------------ 6. class similarity
def class_similarity(X, y):
    """Distance between class-mean images: predicts which classes will be confused."""
    means = np.stack([X[y == c].reshape(-1, 784).mean(0) for c in range(10)])
    d = np.sqrt(((means[:, None] - means[None]) ** 2).sum(-1))
    fig, ax = plt.subplots(figsize=(7.2, 6))
    sns.heatmap(d, annot=True, fmt=".1f", cmap="mako_r", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax, cbar_kws={"label": "Euclidean distance"})
    ax.set_title("Distance between class-mean images (small = visually similar)")
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
    save(fig, "eda_08_class_mean_distance")
    pairs = [(CLASS_NAMES[i], CLASS_NAMES[j], d[i, j]) for i in range(10) for j in range(i + 1, 10)]
    df = pd.DataFrame(pairs, columns=["class_a", "class_b", "mean_image_distance"]).sort_values("mean_image_distance")
    df.to_csv(TABLE_DIR / "eda_class_pair_distance.csv", index=False)
    return df


# ------------------------------------------------------------------ 7. dimensionality
def pca_analysis(X, y, n_comp=300):
    flat = X.reshape(len(X), -1)
    pca = PCA(n_components=n_comp, random_state=SEED).fit(flat)
    cum = np.cumsum(pca.explained_variance_ratio_)
    k = {t: int(np.searchsorted(cum, t) + 1) for t in (0.8, 0.9, 0.95, 0.99) if cum[-1] >= t}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(np.arange(1, n_comp + 1), cum, lw=2)
    axes[0].set_xlim(0, n_comp)
    for t, kk in k.items():
        axes[0].axhline(t, ls=":", c="grey"); axes[0].annotate(f"{int(t*100)}% -> {kk} comps", (kk, t), textcoords="offset points", xytext=(6, -12), fontsize=8)
    axes[0].set_xlabel("number of principal components"); axes[0].set_ylabel("cumulative explained variance")
    axes[0].set_title("PCA on flattened pixels")
    Z = pca.transform(flat[:8000])
    sc = axes[1].scatter(Z[:, 0], Z[:, 1], c=y[:8000], cmap="tab10", s=3, alpha=0.7)
    axes[1].set_xlabel("PC 1"); axes[1].set_ylabel("PC 2"); axes[1].set_title("First two components (8,000 images)")
    axes[1].legend(*sc.legend_elements(), title="class id", fontsize=7, loc="best", ncol=2)
    save(fig, "eda_09_pca")
    pd.DataFrame({"variance_target": list(k.keys()), "n_components": list(k.values())}).to_csv(TABLE_DIR / "eda_pca_components.csv", index=False)
    return k, pca


def tsne_embedding(X, y, n=5000, seed=SEED):
    idx = np.random.default_rng(seed).choice(len(X), n, replace=False)
    flat = X[idx].reshape(n, -1)
    Z50 = PCA(50, random_state=seed).fit_transform(flat)
    Z = TSNE(2, perplexity=30, init="pca", random_state=seed, n_jobs=-1).fit_transform(Z50)
    np.save(TABLE_DIR / "tsne_coords.npy", np.column_stack([Z, y[idx]]))
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    for c in range(10):
        m = y[idx] == c
        ax.scatter(Z[m, 0], Z[m, 1], s=5, color=PALETTE[c], label=CLASS_NAMES[c], alpha=0.8)
    ax.legend(markerscale=3, fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.set_title("t-SNE of 5,000 training images (after PCA-50)"); ax.set_xticks([]); ax.set_yticks([])
    save(fig, "eda_10_tsne")


# ------------------------------------------------------------------ 8. distribution shift
def train_test_shift(X_train, X_test):
    a, b = X_train.reshape(len(X_train), -1).mean(1), X_test.reshape(len(X_test), -1).mean(1)
    ks = stats.ks_2samp(a, b)
    diff = X_train.mean(0) - X_test.mean(0)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].hist(a, 60, density=True, alpha=.6, label="train"); axes[0].hist(b, 60, density=True, alpha=.6, label="test")
    axes[0].set_title(f"Per-image mean intensity (KS p = {ks.pvalue:.2f})"); axes[0].legend()
    im = axes[1].imshow(diff, cmap="coolwarm", vmin=-0.02, vmax=0.02); axes[1].set_title("mean(train) - mean(test) per pixel"); axes[1].axis("off")
    fig.colorbar(im, ax=axes[1], fraction=0.046)
    save(fig, "eda_11_train_test_shift")
    return {"ks_stat": float(ks.statistic), "ks_pvalue": float(ks.pvalue), "max_abs_mean_pixel_diff": float(np.abs(diff).max())}


# ------------------------------------------------------------------ 9. hard examples
def similar_class_examples(X, y, classes=("T-shirt/top", "Pullover", "Coat", "Shirt"), n=8, seed=SEED):
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(len(classes), n, figsize=(n * 1.0, len(classes) * 1.1))
    for r, name in enumerate(classes):
        c = CLASS_NAMES.index(name)
        for j, i in enumerate(rng.choice(np.where(y == c)[0], n, replace=False)):
            axes[r, j].imshow(X[i], cmap="gray"); axes[r, j].axis("off")
        axes[r, 0].set_title(name, loc="left", fontsize=9)
    fig.suptitle("The visually ambiguous upper-body group", fontweight="bold")
    save(fig, "eda_12_ambiguous_group")


def orientation_check(X, y):
    """Garments are upright: verify via top-vs-bottom ink mass per class."""
    top = X[:, :14].reshape(len(X), -1).sum(1); bot = X[:, 14:].reshape(len(X), -1).sum(1)
    df = pd.DataFrame({"class": [CLASS_NAMES[i] for i in y], "top_minus_bottom_share": (top - bot) / (top + bot)})
    s = df.groupby("class").mean().loc[CLASS_NAMES].round(3)
    s.to_csv(TABLE_DIR / "eda_orientation_check.csv")
    return s
