"""Phase 2 / Step 8 - Presentation-grade analysis & figures.

Adds the analyses the first Phase 2 pass was missing and redraws every key
result as a chart whose axes say, in plain words, what they measure.

New analyses
  * 95% bootstrap confidence intervals for the CNNs (Phase 1 only had them for
    classical models), so every bar on the leaderboard carries an error bar.
  * Row-normalised confusion matrices for SVM-RBF (HOG) and CNN3Block_GAP, plus
    the 4x4 upper-body sub-matrix and the CNN's top-10 confusions.
  * Generalisation gap (train acc - val acc) per CNN at its best epoch.
  * The 2x2 depth/augmentation ablation, when CNN3Block_GAP_NoAug has been run
    (scripts/07_ablation_gap_noaug.py).

Outputs
  report/figures/p2_*.png                 - figures used by the report
  results/tables/p2_*.csv                 - the numbers behind them
  results/tables/p2_deck_data.json        - data consumed by the slide builder
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from src.config import CLASS_NAMES, RUN_DIR, TABLE_DIR, UPPER_BODY, ROOT
from src.data_loader import load_raw

OUT = ROOT / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Validated palette (dataviz validator, all-pairs, light surface)
CNN = "#6250d6"     # Phase 2 / CNN  (violet)
SVM = "#eb6834"     # best Phase 1 / SVM-RBF (orange)
AQUA = "#1baf7a"    # third series (always direct-labelled)
GREY = "#b4b8c4"    # other Phase 1 models
INK, INK2, GRID = "#1d1b2e", "#5b5a6b", "#e6e6ee"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "axes.titlesize": 14,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.labelsize": 11.5,
    "axes.labelcolor": INK, "axes.edgecolor": "#c9c9d6", "axes.spines.top": False,
    "axes.spines.right": False, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "figure.dpi": 110, "savefig.dpi": 220, "savefig.bbox": "tight",
    "legend.frameon": False,
})
PCT = FuncFormatter(lambda v, _: f"{v:.0f}%")


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", facecolor="white")
    plt.close(fig)
    print("  wrote", name)


def pred(model, feat="pixels"):
    return np.load(RUN_DIR / f"{model}__{feat}__pred.npy")


def bootstrap_ci(y, p, n=2000, seed=42):
    rng = np.random.default_rng(seed)
    correct = (y == p).astype(float)
    idx = rng.integers(0, len(y), size=(n, len(y)))
    accs = correct[idx].mean(axis=1)
    return float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))


def main():
    _, _, _, y = load_raw()
    lb = pd.read_csv(TABLE_DIR / "all_phases_model_results.csv")
    ci_p1 = pd.read_csv(TABLE_DIR / "bootstrap_ci.csv")

    cnn_names = ["CNN2Block", "CNN2Block_Aug", "CNN3Block_GAP"]
    noaug = RUN_DIR / "CNN3Block_GAP_NoAug__pixels.json"
    has_noaug = noaug.exists()
    if has_noaug:
        cnn_names.append("CNN3Block_GAP_NoAug")
        r = json.loads(noaug.read_text())
        if "CNN3Block_GAP_NoAug" not in set(lb.model):
            lb = pd.concat([lb, pd.DataFrame([{
                "phase": "Phase 2 (Deep Learning)", "model": r["model"],
                "features": "Learned CNN filters", "accuracy": r["accuracy"],
                "macro_f1": r["macro_f1"], "fit_seconds": r["fit_seconds"],
                "infer_ms": r["infer_ms_per_image"], "params": r["params"]}])],
                ignore_index=True)

    # ---------- 1. Bootstrap CIs for CNNs --------------------------------
    ci_rows = []
    for m in cnn_names:
        lo, hi = bootstrap_ci(y, pred(m))
        ci_rows.append(dict(model=m, features="pixels (learned CNN)",
                            accuracy=float((pred(m) == y).mean()), ci_low=lo, ci_high=hi))
    ci_cnn = pd.DataFrame(ci_rows)
    ci_cnn.to_csv(TABLE_DIR / "p2_bootstrap_ci_cnn.csv", index=False)

    # Leaderboard rows used in charts (one per model family + all CNNs)
    nice = {"hog": "HOG", "hog_stats": "HOG+Stats", "pixels": "Pixels"}
    board = []
    for _, r in lb.iterrows():
        is_cnn = r["phase"].startswith("Phase 2")
        if is_cnn:
            c = ci_cnn[ci_cnn.model == r["model"]].iloc[0]
            label = r["model"].replace("_", " ")
        else:
            c = ci_p1[(ci_p1.model == r["model"]) & (ci_p1.features == r["features"])].iloc[0]
            label = f"{r['model']} ({nice[r['features']]})"
        board.append(dict(label=label, model=r["model"], features=r["features"],
                          phase=2 if is_cnn else 1, acc=r["accuracy"] * 100,
                          lo=c.ci_low * 100, hi=c.ci_high * 100, f1=r["macro_f1"],
                          ms=r["infer_ms"], params=int(r["params"])))
    board = pd.DataFrame(board).sort_values("acc").reset_index(drop=True)
    board.to_csv(TABLE_DIR / "p2_leaderboard_with_ci.csv", index=False)

    # ---------- Fig 1: leaderboard ---------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6.4))
    colors = [CNN if p == 2 else (SVM if (m == "SVM-RBF" and f == "hog") else GREY)
              for p, m, f in zip(board.phase, board.model, board.features)]
    ax.barh(board.label, board.acc, color=colors, height=0.62, zorder=2)
    ax.errorbar(board.acc, board.label, xerr=[board.acc - board.lo, board.hi - board.acc],
                fmt="none", ecolor=INK, elinewidth=1.1, capsize=3, zorder=3)
    for i, a in enumerate(board.acc):
        ax.text(board.hi[i] + 0.12, i, f"{a:.2f}%", va="center", fontsize=10, color=INK)
    ax.set_xlim(83, 95)
    ax.xaxis.set_major_formatter(PCT)
    ax.set_xlabel("Test accuracy on 10,000 unseen images (%)  —  whisker = 95% bootstrap CI")
    ax.set_ylabel("Model (feature set)")
    ax.grid(axis="y", visible=False)
    ax.set_title("Every CNN beats every classical model; CNN3Block GAP leads")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=CNN, label="Phase 2 — CNN"),
                       Patch(color=SVM, label="Phase 1 best — SVM-RBF (HOG)"),
                       Patch(color=GREY, label="Other Phase 1 models")],
              loc="lower right")
    save(fig, "p2_01_leaderboard_ci")

    # ---------- Fig 2: validation curves ---------------------------------
    hists = {m: pd.read_csv(TABLE_DIR / f"{m}__history.csv") for m in cnn_names}
    style = {"CNN2Block": (GREY, "-"), "CNN2Block_Aug": (AQUA, "-"),
             "CNN3Block_GAP": (CNN, "-"), "CNN3Block_GAP_NoAug": (CNN, "--")}
    fig, ax = plt.subplots(figsize=(10, 5.6))
    for m, h in hists.items():
        c, ls = style[m]
        ax.plot(h.epoch, h.val_acc * 100, color=c, ls=ls, lw=2.2, marker="o", ms=4)
        ax.text(h.epoch.iloc[-1] + 0.3, h.val_acc.iloc[-1] * 100,
                f"{m.replace('_', ' ')}  {h.val_acc.max()*100:.1f}%", va="center",
                fontsize=10, color=INK)
    ax.axhline(91.30, color=SVM, lw=1.5, ls=":")
    ax.text(26.5, 90.75, "Phase 1 ceiling — SVM-RBF (HOG), 91.30% test acc", color=SVM,
            fontsize=10, ha="right")
    ax.set_xlabel("Training epoch (one full pass over 48,000 training images)")
    ax.set_ylabel("Validation accuracy on 12,000 held-out images (%)")
    ax.yaxis.set_major_formatter(PCT)
    ax.set_xlim(0.5, 27)
    ax.set_ylim(84, 95.5)
    ax.set_title("Validation accuracy per epoch — deeper GAP network keeps climbing")
    save(fig, "p2_02_validation_curves")

    # ---------- Fig 3: overfitting gap (train vs val loss) ---------------
    pair = ["CNN2Block", "CNN3Block_GAP"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, m in zip(axes, pair):
        h = hists[m]
        ax.plot(h.epoch, h.train_loss, color=INK2, lw=2.2, label="Training loss")
        ax.plot(h.epoch, h.val_loss, color=CNN, lw=2.2, label="Validation loss")
        ax.fill_between(h.epoch, h.train_loss, h.val_loss,
                        where=h.val_loss > h.train_loss, color=SVM, alpha=0.15,
                        label="Gap = overfitting")
        ax.set_title(m.replace("_", " "))
        ax.set_xlabel("Training epoch")
    axes[0].set_ylabel("Cross-entropy loss (lower = better)")
    axes[0].legend(loc="upper right")
    fig.suptitle("Train vs validation loss — baseline starts memorising, GAP model does not",
                 x=0.06, ha="left", fontweight="bold", fontsize=14, y=1.03)
    save(fig, "p2_03_overfitting_gap")

    gap = []
    for m, h in hists.items():
        b = h.val_macro_f1.idxmax()
        gap.append(dict(model=m, best_epoch=int(h.epoch[b]), epochs_run=len(h),
                        train_acc=h.train_acc[b], val_acc=h.val_acc[b],
                        gap_pts=(h.train_acc[b] - h.val_acc[b]) * 100,
                        still_improving=bool(b == len(h) - 1)))
    pd.DataFrame(gap).to_csv(TABLE_DIR / "p2_generalisation_gap.csv", index=False)

    # ---------- Fig 4: per-class recall ----------------------------------
    p_svm, p_cnn = pred("SVM-RBF", "hog"), pred("CNN3Block_GAP")
    rec = lambda p: np.array([((p == k) & (y == k)).sum() / (y == k).sum() for k in range(10)])
    r_svm, r_cnn = rec(p_svm) * 100, rec(p_cnn) * 100
    order = np.argsort(r_cnn)
    fig, ax = plt.subplots(figsize=(10, 6))
    yy = np.arange(10)
    ax.barh(yy + 0.2, r_svm[order], height=0.38, color=SVM, label="SVM-RBF (HOG) — Phase 1", zorder=2)
    ax.barh(yy - 0.2, r_cnn[order], height=0.38, color=CNN, label="CNN3Block GAP — Phase 2", zorder=2)
    for i, k in enumerate(order):
        d = r_cnn[k] - r_svm[k]
        ax.text(max(r_cnn[k], r_svm[k]) + 0.4, i, f"{d:+.1f} pts", va="center",
                fontsize=10, color=INK, fontweight="bold" if abs(d) >= 3 else "normal")
    ax.set_yticks(yy, [CLASS_NAMES[k] for k in order])
    ax.set_xlim(70, 103)
    ax.xaxis.set_major_formatter(PCT)
    ax.set_xlabel("Recall = % of that class's 1,000 test images classified correctly")
    ax.set_ylabel("Clothing class")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    ax.set_title("Per-class recall — big gains on Pullover & Coat, Shirt still hardest")
    save(fig, "p2_04_per_class_recall")

    # ---------- Fig 5: where errors live ---------------------------------
    ub = [CLASS_NAMES.index(c) for c in UPPER_BODY]
    def err_split(p):
        wrong = p != y
        upper = wrong & np.isin(y, ub)
        return int(upper.sum()), int((wrong & ~np.isin(y, ub)).sum())
    err_models = [("LogReg (Pixels)\nPhase 1 start", pred("LogReg")),
                  ("SVM-RBF (HOG)\nPhase 1 best", p_svm),
                  ("CNN2Block\nPhase 2", pred("CNN2Block")),
                  ("CNN3Block GAP\nPhase 2 best", p_cnn)]
    errs = [(n, *err_split(p)) for n, p in err_models]
    fig, ax = plt.subplots(figsize=(10, 5.6))
    names = [e[0] for e in errs]
    up = np.array([e[1] for e in errs]); other = np.array([e[2] for e in errs])
    ax.bar(names, up, color=CNN, width=0.58, label="Upper-body classes (T-shirt, Pullover, Coat, Shirt)", zorder=2)
    ax.bar(names, other, bottom=up, color=GREY, width=0.58, label="All other 6 classes", zorder=2,
           edgecolor="white", linewidth=2)
    for i in range(len(names)):
        ax.text(i, up[i] / 2, f"{up[i]}", ha="center", va="center", color="white", fontweight="bold")
        ax.text(i, up[i] + other[i] + 25, f"{up[i]+other[i]} errors\n({(up[i])/(up[i]+other[i])*100:.0f}% upper-body)",
                ha="center", fontsize=10, color=INK)
    ax.set_ylabel("Misclassified test images (out of 10,000)")
    ax.set_xlabel("Model")
    ax.set_ylim(0, 1800)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right")
    ax.set_title("Total errors fell 55% — but the upper-body share did not shrink")
    save(fig, "p2_05_error_breakdown")
    pd.DataFrame(errs, columns=["model", "upper_body_errors", "other_errors"]).to_csv(
        TABLE_DIR / "p2_error_breakdown.csv", index=False)

    # ---------- Fig 6: McNemar disagreement ------------------------------
    mc = pd.read_csv(TABLE_DIR / "mcnemar_phase1_vs_phase2.csv")
    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = np.arange(len(mc))
    ax.bar(x - 0.2, mc.svm_only_correct, width=0.38, color=SVM, label="Only SVM-RBF (HOG) correct", zorder=2)
    ax.bar(x + 0.2, mc.cnn_only_correct, width=0.38, color=CNN, label="Only the CNN correct", zorder=2)
    for i, r in mc.iterrows():
        ax.text(i - 0.2, r.svm_only_correct + 8, str(r.svm_only_correct), ha="center", fontsize=10)
        ax.text(i + 0.2, r.cnn_only_correct + 8, str(r.cnn_only_correct), ha="center", fontsize=10)
        sig = "significant" if r.p_value < 0.001 else "not significant at 0.001"
        ax.text(i, 625, f"p = {r.p_value:.2g}\n{sig}", ha="center", va="top", fontsize=10,
                color=CNN if r.p_value < 0.001 else INK2,
                fontweight="bold" if r.p_value < 0.001 else "normal")
    ax.set_xticks(x, [m.replace("_", " ") for m in mc.model_2])
    ax.set_ylim(0, 640)
    ax.set_ylabel("Test images where the two models disagree")
    ax.set_xlabel("CNN compared against SVM-RBF (HOG)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", bbox_to_anchor=(0, 0.84))
    ax.set_title("McNemar test — only CNN3Block GAP wins clearly (445 vs 266)")
    save(fig, "p2_06_mcnemar")

    # ---------- Fig 7: upper-body confusion before/after -----------------
    cms = {}
    for name, p in [("SVM-RBF (HOG)", p_svm), ("CNN3Block GAP", p_cnn)]:
        cm = confusion_matrix(y, p, labels=range(10))
        cms[name] = cm
        pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
            TABLE_DIR / f"p2_confusion_{name.split()[0].replace('-', '')}.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    for ax, (name, cm) in zip(axes, cms.items()):
        sub = cm[np.ix_(ub, ub)] / cm[ub].sum(axis=1, keepdims=True) * 100
        ax.imshow(sub, cmap="Purples", vmin=0, vmax=100)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{sub[i, j]:.1f}%", ha="center", va="center",
                        color="white" if sub[i, j] > 55 else INK, fontsize=11,
                        fontweight="bold" if i == j else "normal")
        ax.set_xticks(range(4), UPPER_BODY, rotation=20)
        ax.set_yticks(range(4), UPPER_BODY)
        ax.set_xlabel("Predicted class")
        ax.grid(False)
        ax.set_title(name)
    axes[0].set_ylabel("True class (row = 100%)")
    fig.suptitle("Upper-body confusion (row %): diagonal = correct, off-diagonal = mix-ups",
                 x=0.06, ha="left", fontweight="bold", fontsize=14, y=1.02)
    save(fig, "p2_07_upper_body_confusion")

    # top confusions for CNN
    cm = cms["CNN3Block GAP"].copy(); np.fill_diagonal(cm, 0)
    top = sorted(((cm[i, j], CLASS_NAMES[i], CLASS_NAMES[j]) for i in range(10) for j in range(10)),
                 reverse=True)[:10]
    pd.DataFrame(top, columns=["count", "true", "predicted"]).to_csv(
        TABLE_DIR / "p2_top_confusions_cnn.csv", index=False)

    # ---------- Fig 8: efficiency ----------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, col, xl in [(axes[0], "ms", "CPU time to classify ONE image (ms, log scale)"),
                        (axes[1], "params", "Numbers the model must store (log scale)")]:
        for _, r in board.sort_values("phase").iloc[::-1].iterrows():
            c = CNN if r.phase == 2 else (SVM if r.model == "SVM-RBF" and r.features == "hog" else GREY)
            ax.scatter(r[col], r.acc, s=90 if c != GREY else 55, color=c,
                       edgecolor="white", linewidth=1.5, zorder=3)
        ax.set_xscale("log")
        ax.set_xlabel(xl)
        ax.yaxis.set_major_formatter(PCT)
        for lab in ["CNN3Block GAP", "SVM-RBF (HOG)", "CNN2Block", "LogReg (Pixels)", "kNN (Pixels)"]:
            r = board[board.label == lab].iloc[0]
            off, ha = ((-8, -16), "right") if lab.startswith("SVM") else ((6, 6), "left")
            ax.annotate(lab, (r[col], r.acc), textcoords="offset points", xytext=off, fontsize=9.5, ha=ha,
                        color=INK, fontweight="bold" if lab in ("CNN3Block GAP", "SVM-RBF (HOG)") else "normal")
    axes[0].set_ylabel("Test accuracy (%)")
    axes[0].set_title("Accuracy vs speed")
    axes[1].set_title("Accuracy vs model size")
    fig.suptitle("Top-left is best: CNN3Block GAP is more accurate, 8x faster & 60x smaller than SVM",
                 x=0.06, ha="left", fontweight="bold", fontsize=14, y=1.03)
    save(fig, "p2_08_efficiency")

    # ---------- Fig 9: 2x2 ablation --------------------------------------
    abl = None
    if has_noaug:
        acc = {m: float((pred(m) == y).mean()) * 100 for m in cnn_names}
        abl = {"rows": ["CNN2Block (Flatten)", "CNN3Block GAP"],
               "cols": ["No augmentation", "Flip + crop augmentation"],
               "acc": [[acc["CNN2Block"], acc["CNN2Block_Aug"]],
                       [acc["CNN3Block_GAP_NoAug"], acc["CNN3Block_GAP"]]]}
        fig, ax = plt.subplots(figsize=(9, 5))
        x = np.arange(2)
        a = np.array(abl["acc"])
        ax.bar(x - 0.2, a[:, 0], width=0.38, color=GREY, label="No augmentation", zorder=2)
        ax.bar(x + 0.2, a[:, 1], width=0.38, color=CNN, label="Flip + crop augmentation", zorder=2)
        for i in range(2):
            for j, dx in enumerate([-0.2, 0.2]):
                ax.text(i + dx, a[i, j] + 0.08, f"{a[i, j]:.2f}%", ha="center", fontsize=10.5)
        ax.axhline(91.30, color=SVM, ls=":", lw=1.5, label="Phase 1 best: SVM-RBF (HOG) 91.30%")
        ax.set_xticks(x, abl["rows"])
        ax.set_ylim(90, 94)
        ax.set_yticks([90, 91, 92, 93, 94])
        ax.yaxis.set_major_formatter(PCT)
        ax.set_ylabel("Test accuracy (%)")
        ax.set_xlabel("Architecture")
        ax.grid(axis="x", visible=False)
        ax.legend(loc="upper left")
        ax.set_title("2x2 ablation — augmentation only pays off in the deeper GAP network")
        save(fig, "p2_09_ablation_2x2")

    # ---------- deck data -------------------------------------------------
    deck = dict(
        leaderboard=board.to_dict(orient="records"),
        histories={m: h[["epoch", "train_loss", "val_loss", "train_acc", "val_acc"]].to_dict(orient="list")
                   for m, h in hists.items()},
        recall=dict(classes=CLASS_NAMES, svm=r_svm.round(1).tolist(), cnn=r_cnn.round(1).tolist()),
        errors=[dict(model=e[0].replace("\n", " · "), upper=e[1], other=e[2]) for e in errs],
        mcnemar=mc.to_dict(orient="records"),
        upper_conf={k: (v[np.ix_(ub, ub)] / v[ub].sum(axis=1, keepdims=True) * 100).round(1).tolist()
                    for k, v in cms.items()},
        upper_names=UPPER_BODY,
        top_conf_cnn=[dict(count=int(c), true=t, pred=p) for c, t, p in top],
        gap=gap, ablation=abl, ci_cnn=ci_rows,
    )
    (TABLE_DIR / "p2_deck_data.json").write_text(json.dumps(deck, indent=1, default=float))
    print("  wrote p2_deck_data.json")


if __name__ == "__main__":
    main()
