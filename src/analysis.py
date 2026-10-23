"""Post-hoc analysis of the finished runs: tables, confusion matrices, error analysis, statistics."""
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import confusion_matrix

from .config import CLASS_NAMES, RUN_DIR, TABLE_DIR, UPPER_BODY
from .models import MODEL_NAMES
from .features import FEATURE_SETS
from .plotting import save, COLORS


def load_runs():
    runs = {}
    for f in RUN_DIR.glob("*__*.json"):
        r = json.loads(f.read_text())
        r["pred"] = np.load(RUN_DIR / f"{r['model']}__{r['features']}__pred.npy")
        runs[(r["model"], r["features"])] = r
    return runs


def results_table(runs):
    df = pd.read_csv(TABLE_DIR / "model_results.csv")
    return df


def bar_overview(df):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax, col, t in zip(axes, ["accuracy", "macro_f1"], ["Test accuracy", "Test macro-F1"]):
        sns.barplot(data=df, x="model", y=col, hue="features", order=MODEL_NAMES, hue_order=FEATURE_SETS,
                    palette=COLORS, ax=ax)
        ax.set_ylim(df[col].min() - 0.03, df[col].max() + 0.015); ax.set_title(t); ax.set_xlabel("")
        for c in ax.containers: ax.bar_label(c, fmt="%.3f", fontsize=7, padding=2)
    save(fig, "res_01_accuracy_f1_overview")


def per_class_recall_heatmap(runs, feats=("pixels", "hog")):
    rows = {}
    for (m, f), r in runs.items():
        if f in feats:
            rows[f"{m} | {f}"] = r["per_class_recall"]
    df = pd.DataFrame(rows).T[CLASS_NAMES]
    df = df.loc[sorted(df.index, key=lambda s: (s.split(" | ")[1], s))]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    sns.heatmap(df, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0.5, vmax=1.0, ax=ax, annot_kws={"size": 8})
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    ax.set_title("Per-class recall on the test set")
    save(fig, "res_02_per_class_recall")
    df.to_csv(TABLE_DIR / "per_class_recall_all.csv")
    full = pd.DataFrame({f"{m}|{f}": r["per_class_recall"] for (m, f), r in runs.items()}).T[CLASS_NAMES]
    full.to_csv(TABLE_DIR / "per_class_recall_all.csv")
    return full


def best_of(runs, feat):
    cands = [(k, r) for k, r in runs.items() if k[1] == feat]
    return max(cands, key=lambda kr: kr[1]["accuracy"])


def confusion_pair(runs, y_test, feat_a="pixels", feat_b="hog"):
    (ka, ra), (kb, rb) = best_of(runs, feat_a), best_of(runs, feat_b)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2))
    cms = {}
    for ax, (k, r) in zip(axes, [(ka, ra), (kb, rb)]):
        cm = confusion_matrix(y_test, r["pred"])
        cms[k] = cm
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax, cbar=False, annot_kws={"size": 7})
        ax.set_title(f"{k[0]} on {k[1]}  (acc {r['accuracy']:.3f})"); ax.set_xlabel("predicted"); ax.set_ylabel("true")
        plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
    save(fig, "res_03_confusion_best_pixels_vs_best_hog")
    return ka, kb, cms


def difference_confusion(cms, ka, kb):
    d = cms[kb] - cms[ka]
    fig, ax = plt.subplots(figsize=(7.5, 6.3))
    sns.heatmap(d, annot=True, fmt="d", cmap="PuOr", center=0, xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax, cbar=False, annot_kws={"size": 7})
    ax.set_title(f"Confusion change: {kb[0]}|{kb[1]} minus {ka[0]}|{ka[1]}\n(off-diagonal negative = fewer errors)")
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right"); ax.set_xlabel("predicted"); ax.set_ylabel("true")
    save(fig, "res_04_confusion_difference")


def top_confusions(runs, y_test, n=10):
    rows = []
    for (m, f), r in runs.items():
        cm = confusion_matrix(y_test, r["pred"]); np.fill_diagonal(cm, 0)
        for i, j in zip(*np.unravel_index(np.argsort(cm, axis=None)[::-1][:n], cm.shape)):
            rows.append(dict(model=m, features=f, true=CLASS_NAMES[i], predicted=CLASS_NAMES[j], count=int(cm[i, j])))
    df = pd.DataFrame(rows); df.to_csv(TABLE_DIR / "top_confusions_all.csv", index=False)
    return df


def upper_body_error_share(runs, y_test):
    """Share of all errors that fall inside the 4-class upper-body group."""
    idx = [CLASS_NAMES.index(c) for c in UPPER_BODY]
    rows = []
    for (m, f), r in runs.items():
        err = r["pred"] != y_test
        within = err & np.isin(y_test, idx) & np.isin(r["pred"], idx)
        rows.append(dict(model=m, features=f, errors=int(err.sum()), upper_body_confusions=int(within.sum()), share_pct=round(100 * within.sum() / max(err.sum(), 1), 1)))
    df = pd.DataFrame(rows).sort_values(["features", "model"]); df.to_csv(TABLE_DIR / "upper_body_error_share.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    sns.barplot(data=df, x="model", y="share_pct", hue="features", hue_order=FEATURE_SETS, palette=COLORS, ax=ax, order=MODEL_NAMES)
    ax.set_ylabel("% of all errors"); ax.set_xlabel(""); ax.set_title("Errors that stay inside {T-shirt, Pullover, Coat, Shirt}")
    save(fig, "res_05_upper_body_error_share")
    return df


def mcnemar(y_true, p1, p2):
    c1, c2 = p1 == y_true, p2 == y_true
    b, c = int((c1 & ~c2).sum()), int((~c1 & c2).sum())
    stat = (abs(b - c) - 1) ** 2 / max(b + c, 1)
    return dict(only_first_correct=b, only_second_correct=c, chi2=stat, p_value=float(1 - stats.chi2.cdf(stat, 1)))


def significance_table(runs, y_test):
    (ka, ra), (kb, rb) = best_of(runs, "pixels"), best_of(runs, "hog")
    rows = []
    for m in MODEL_NAMES:
        p, h = runs[(m, "pixels")], runs[(m, "hog")]
        rows.append(dict(model=m, acc_pixels=p["accuracy"], acc_hog=h["accuracy"], delta=h["accuracy"] - p["accuracy"], **mcnemar(y_test, p["pred"], h["pred"])))
    df = pd.DataFrame(rows); df.to_csv(TABLE_DIR / "mcnemar_pixels_vs_hog.csv", index=False)
    return df


def bootstrap_ci(runs, y_test, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed); n = len(y_test); rows = []
    idx = rng.integers(0, n, (n_boot, n))
    for (m, f), r in runs.items():
        correct = (r["pred"] == y_test)
        accs = correct[idx].mean(1)
        rows.append(dict(model=m, features=f, accuracy=correct.mean(), ci_low=np.percentile(accs, 2.5), ci_high=np.percentile(accs, 97.5)))
    df = pd.DataFrame(rows).sort_values("accuracy", ascending=False); df.to_csv(TABLE_DIR / "bootstrap_ci.csv", index=False)
    return df


def misclassified_gallery(runs, X_test, y_test, key, n=12, name="res_06_misclassified"):
    r = runs[key]; wrong = np.where(r["pred"] != y_test)[0]
    # focus on the most informative: upper-body group first
    idx_u = [CLASS_NAMES.index(c) for c in UPPER_BODY]
    pri = [i for i in wrong if y_test[i] in idx_u and r["pred"][i] in idx_u]
    pick = (pri + [i for i in wrong if i not in set(pri)])[:0] or []
    rng = np.random.default_rng(1); pick = list(rng.choice(pri, min(n, len(pri)), replace=False))
    cols = 6; rows_ = int(np.ceil(len(pick) / cols))
    fig, axes = plt.subplots(rows_, cols, figsize=(cols * 1.7, rows_ * 2.0))
    for ax in np.ravel(axes): ax.axis("off")
    for ax, i in zip(np.ravel(axes), pick):
        ax.imshow(X_test[i], cmap="gray"); ax.set_title(f"true: {CLASS_NAMES[y_test[i]]}\npred: {CLASS_NAMES[r['pred'][i]]}", fontsize=7, color="#b22")
    fig.suptitle(f"Misclassified upper-body garments - {key[0]} on {key[1]}", fontweight="bold")
    save(fig, name)


def pareto_plot(df):
    d = df.copy(); d["lat"] = d.total_ms_per_image
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    markers = dict(zip(MODEL_NAMES, "osD^"))
    for _, r in d.iterrows():
        ax.scatter(r.lat, r.accuracy, s=90, marker=markers[r.model], color=COLORS[r.features], edgecolor="k", lw=.5)
        ax.annotate(r.model, (r.lat, r.accuracy), fontsize=6.5, xytext=(4, 3), textcoords="offset points")
    ax.set_xscale("log"); ax.set_xlabel("inference latency per image incl. feature extraction (ms, log)"); ax.set_ylabel("test accuracy")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=c, label=f) for f, c in COLORS.items()], title="features", loc="lower right")
    ax.set_title("Accuracy vs. inference cost"); save(fig, "res_07_accuracy_vs_latency")


def learning_curve(S, feat_data, sizes=(1000, 2000, 5000, 10000, 20000, 48000)):
    """Accuracy vs. number of training images for LR on pixels vs. HOG (data-efficiency of the descriptor)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    rng = np.random.default_rng(0); rows = []
    for feat in ("pixels", "hog"):
        tr, val = feat_data[feat]["tr"], feat_data[feat]["val"]
        for n in sizes:
            idx = rng.choice(len(tr), n, replace=False)
            m = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=300)).fit(tr[idx], S.y_tr[idx])
            rows.append(dict(features=feat, n_train=n, val_acc=m.score(val, S.y_val)))
            print(rows[-1], flush=True)
    df = pd.DataFrame(rows); df.to_csv(TABLE_DIR / "learning_curve.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 3.8))
    for f, g in df.groupby("features"): ax.plot(g.n_train, g.val_acc, "o-", color=COLORS[f], label=f)
    ax.set_xscale("log"); ax.set_xlabel("training images"); ax.set_ylabel("validation accuracy"); ax.legend(); ax.set_title("Learning curve (Logistic Regression)")
    save(fig, "res_08_learning_curve")
    return df
