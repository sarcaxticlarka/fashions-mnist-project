"""Phase 2 / Step 10 - Final model: multi-seed robustness, ensemble + TTA, calibration,
abstention and the Shirt decision-bias study.

Inputs : results/tables/runs/logits/*.npz written by scripts/09_train_final_models.py
Outputs: results/tables/p2_final_*.csv / .json, report/figures/p2_1*_*.png,
         results/tables/runs/FinalEnsemble__pixels__pred.npy (+ calibrated test probabilities)

Every choice (which ensemble, temperature, Shirt bias, abstain threshold) is made on the
12,000-image VALIDATION set. The test set is only used to report the final numbers.
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
from scipy import stats
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax, softmax
from sklearn.metrics import f1_score, recall_score

from src.config import CLASS_NAMES, RUN_DIR, TABLE_DIR, ROOT
from src.data_loader import make_splits

OUT = ROOT / "report" / "figures"
LOGIT_DIR = RUN_DIR / "logits"
CNN, SVM, AQUA, GREY, INK, INK2 = "#6250d6", "#eb6834", "#1baf7a", "#b4b8c4", "#1d1b2e", "#5b5a6b"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "axes.titlesize": 14, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.labelsize": 11.5, "axes.labelcolor": INK, "axes.edgecolor": "#c9c9d6",
    "axes.spines.top": False, "axes.spines.right": False, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": "#e6e6ee", "savefig.dpi": 220, "savefig.bbox": "tight", "legend.frameon": False,
})
PCT = FuncFormatter(lambda v, _: f"{v:.0f}%")
SHIRT = CLASS_NAMES.index("Shirt")


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", facecolor="white")
    plt.close(fig)
    print("  wrote", name)


def ece(probs, y, bins=15):
    conf, pred = probs.max(1), probs.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    e, rows = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum():
            acc, c = (pred[m] == y[m]).mean(), conf[m].mean()
            e += m.mean() * abs(acc - c)
            rows.append((lo, hi, c, acc, int(m.sum())))
    return e, rows


def nll(logp, y):
    return -logp[np.arange(len(y)), y].mean()


def mcnemar(y, a, b):
    ca, cb = a == y, b == y
    n01, n10 = int((ca & ~cb).sum()), int((~ca & cb).sum())
    chi = (abs(n01 - n10) - 1) ** 2 / max(n01 + n10, 1)
    return n01, n10, chi, float(stats.chi2.sf(chi, 1))


def boot_ci(y, p, n=2000, seed=42):
    rng = np.random.default_rng(seed)
    c = (y == p).astype(float)
    a = c[rng.integers(0, len(y), (n, len(y)))].mean(1)
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def main():
    S = make_splits()
    yv, yt = S.y_val, S.y_test
    runs = {}
    for f in sorted(LOGIT_DIR.glob("*.npz")):
        meta = json.loads((RUN_DIR / f"{f.stem}__pixels.json").read_text())
        z = np.load(f)
        runs[f.stem] = dict(meta=meta, **{k: z[k] for k in z.files})
    if not runs:
        sys.exit("No logits found - run scripts/09_train_final_models.py first.")
    print(f"Loaded {len(runs)} runs: {list(runs)}")

    # ---------------- 1. per-run and per-family (multi-seed) -------------
    rows = []
    for n, r in runs.items():
        tta_t = (softmax(r["test"], 1) + softmax(r["test_flip"], 1)) / 2
        tta_v = (softmax(r["val"], 1) + softmax(r["val_flip"], 1)) / 2
        rows.append(dict(run=n, family=r["meta"]["family"], seed=r["meta"]["seed"],
                         params=r["meta"]["params"], ms=r["meta"]["infer_ms_per_image"],
                         best_epoch=r["meta"]["best_epoch"], epochs_run=r["meta"]["epochs_run"],
                         val_acc=(r["val"].argmax(1) == yv).mean(), test_acc=(r["test"].argmax(1) == yt).mean(),
                         val_acc_tta=(tta_v.argmax(1) == yv).mean(), test_acc_tta=(tta_t.argmax(1) == yt).mean(),
                         shirt_recall=((r["test"].argmax(1) == SHIRT) & (yt == SHIRT)).sum() / (yt == SHIRT).sum()))
    per_run = pd.DataFrame(rows)
    per_run.to_csv(TABLE_DIR / "p2_final_per_run.csv", index=False)
    fam = per_run.groupby("family").agg(n_seeds=("seed", "count"), test_mean=("test_acc", "mean"),
                                        test_std=("test_acc", "std"), tta_mean=("test_acc_tta", "mean"),
                                        val_mean=("val_acc", "mean"), shirt_mean=("shirt_recall", "mean"),
                                        params=("params", "first"), ms=("ms", "mean")).reset_index()
    fam.to_csv(TABLE_DIR / "p2_final_seed_summary.csv", index=False)
    print(fam.to_string())

    # ---------------- 2. candidate ensembles, chosen on validation --------
    def ens(members, split, tta=True):
        ps = []
        for m in members:
            p = softmax(runs[m][split], 1)
            if tta:
                p = (p + softmax(runs[m][split + "_flip"], 1)) / 2
            ps.append(p)
        return np.mean(ps, 0)

    fams = sorted(per_run.family.unique())
    cands = {}
    for fm in fams:
        mem = list(per_run[per_run.family == fm].run)
        cands[f"{fm} x{len(mem)} + TTA"] = mem
    if len(fams) > 1:
        cands[f"All {len(runs)} nets + TTA"] = list(runs)
    ens_rows = []
    for k, mem in cands.items():
        pv, pt = ens(mem, "val"), ens(mem, "test")
        ens_rows.append(dict(ensemble=k, members=len(mem), val_acc=(pv.argmax(1) == yv).mean(),
                             test_acc=(pt.argmax(1) == yt).mean(),
                             params=int(sum(runs[m]["meta"]["params"] for m in mem)),
                             ms=float(2 * sum(runs[m]["meta"]["infer_ms_per_image"] for m in mem))))
    ens_df = pd.DataFrame(ens_rows).sort_values("val_acc", ascending=False)
    ens_df.to_csv(TABLE_DIR / "p2_final_ensembles.csv", index=False)
    print(ens_df.to_string())
    final_name = ens_df.iloc[0].ensemble
    members = cands[final_name]
    pv_raw, pt_raw = ens(members, "val"), ens(members, "test")

    # ---------------- 3. temperature scaling (fit on val) -----------------
    lv, lt = np.log(pv_raw + 1e-12), np.log(pt_raw + 1e-12)
    T = minimize_scalar(lambda t: nll(log_softmax(lv / t, 1), yv), bounds=(0.3, 5), method="bounded").x
    pv, pt = softmax(lv / T, 1), softmax(lt / T, 1)
    ece_before, rel_before = ece(pt_raw, yt)
    ece_after, rel_after = ece(pt, yt)

    # ---------------- 4. Shirt decision bias (fit on val) -----------------
    def apply_bias(p, b):
        lp = np.log(p + 1e-12).copy()
        lp[:, SHIRT] += b
        return lp.argmax(1)
    grid = np.round(np.arange(0, 1.51, 0.05), 2)
    bias_rows = []
    for b in grid:
        pvb, ptb = apply_bias(pv, b), apply_bias(pt, b)
        bias_rows.append(dict(bias=b, val_f1=f1_score(yv, pvb, average="macro"), val_acc=(pvb == yv).mean(),
                              test_acc=(ptb == yt).mean(), test_f1=f1_score(yt, ptb, average="macro"),
                              test_shirt_recall=recall_score(yt, ptb, average=None)[SHIRT],
                              test_shirt_precision=((ptb == SHIRT) & (yt == SHIRT)).sum() / max((ptb == SHIRT).sum(), 1)))
    bias_df = pd.DataFrame(bias_rows)
    bias_df.to_csv(TABLE_DIR / "p2_final_shirt_bias.csv", index=False)
    b_star = float(bias_df.loc[bias_df.val_f1.idxmax(), "bias"])
    pred_final = apply_bias(pt, b_star)
    print(f"final={final_name} T={T:.3f} shirt_bias={b_star}")

    # ---------------- 5. abstention (threshold on val) --------------------
    def coverage_curve(p, y, pred):
        conf = p.max(1)
        order = np.argsort(-conf)
        correct = (pred == y)[order]
        k = np.arange(1, len(y) + 1)
        return k / len(y), np.cumsum(correct) / k, conf[order]
    cov_v, acc_v, conf_v = coverage_curve(pv, yv, apply_bias(pv, b_star))
    target = 0.99
    ok = np.where(acc_v >= target)[0]
    thr = float(conf_v[ok.max()]) if len(ok) else 1.0
    conf_t = pt.max(1)
    keep = conf_t >= thr
    abst = dict(target_val_acc=target, threshold=thr, test_coverage=float(keep.mean()),
                test_acc_on_kept=float((pred_final[keep] == yt[keep]).mean()),
                test_acc_all=float((pred_final == yt).mean()),
                shirt_share_of_abstained=float((yt[~keep] == SHIRT).mean()) if (~keep).any() else 0.0)
    cov_t, acc_t, _ = coverage_curve(pt, yt, pred_final)

    # ---------------- 6. compare against Phase 1 & first Phase 2 model ----
    p_svm = np.load(RUN_DIR / "SVM-RBF__hog__pred.npy")
    p_gap = np.load(RUN_DIR / "CNN3Block_GAP__pixels__pred.npy")
    mc_svm, mc_gap = mcnemar(yt, p_svm, pred_final), mcnemar(yt, p_gap, pred_final)
    lo, hi = boot_ci(yt, pred_final)
    rec_final = recall_score(yt, pred_final, average=None)
    np.save(RUN_DIR / "FinalEnsemble__pixels__pred.npy", pred_final)
    np.save(RUN_DIR / "FinalEnsemble__test_probs.npy", pt.astype(np.float32))

    def fixed_bins(p, y, edges=(0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0)):
        conf, pr = p.max(1), p.argmax(1)
        out = []
        for lo_, hi_ in zip(edges[:-1], edges[1:]):
            m = (conf > lo_) & (conf <= hi_)
            out.append(dict(bin=f"{int(lo_*100)}–{int(hi_*100)}%", n=int(m.sum()),
                            conf=float(conf[m].mean()) if m.any() else None,
                            acc=float((pr[m] == y[m]).mean()) if m.any() else None))
        return out

    summary = dict(
        calib_bins_before=fixed_bins(pt_raw, yt), calib_bins_after=fixed_bins(pt, yt),
        final_model=final_name, members=members, temperature=float(T), shirt_bias=b_star,
        test_acc=float((pred_final == yt).mean()), test_macro_f1=float(f1_score(yt, pred_final, average="macro")),
        test_acc_before_bias=float((pt.argmax(1) == yt).mean()), ci=[lo, hi],
        errors=int((pred_final != yt).sum()), params=int(sum(runs[m]["meta"]["params"] for m in members)),
        ms_per_image=float(2 * sum(runs[m]["meta"]["infer_ms_per_image"] for m in members)),
        per_class_recall=dict(zip(CLASS_NAMES, rec_final.round(4).tolist())),
        ece_before=float(ece_before), ece_after=float(ece_after), abstain=abst,
        mcnemar_vs_svm=dict(zip(["svm_only", "final_only", "chi2", "p"], mc_svm)),
        mcnemar_vs_gap=dict(zip(["gap_only", "final_only", "chi2", "p"], mc_gap)),
        best_single=per_run.sort_values("val_acc", ascending=False).iloc[0][["run", "val_acc", "test_acc", "params", "ms"]].to_dict(),
        reliability_before=rel_before, reliability_after=rel_after,
        coverage_curve=dict(coverage=cov_t[::50].round(4).tolist(), accuracy=acc_t[::50].round(4).tolist()),
    )
    # unified leaderboard: Phase 1 + all Phase 2 models + final ensemble
    lb = pd.read_csv(TABLE_DIR / "all_phases_model_results.csv")
    extra = []
    noaug = RUN_DIR / "CNN3Block_GAP_NoAug__pixels.json"
    if noaug.exists():
        r = json.loads(noaug.read_text())
        extra.append(dict(model="CNN3Block_GAP_NoAug", accuracy=r["accuracy"], macro_f1=r["macro_f1"],
                          fit_seconds=r["fit_seconds"], infer_ms=r["infer_ms_per_image"], params=r["params"]))
    for n, r in runs.items():
        m = r["meta"]
        extra.append(dict(model=n, accuracy=m["accuracy"], macro_f1=m["macro_f1"], fit_seconds=m["fit_seconds"],
                          infer_ms=m["infer_ms_per_image"], params=m["params"]))
    extra.append(dict(model="FinalEnsemble", accuracy=summary["test_acc"], macro_f1=summary["test_macro_f1"],
                      fit_seconds=float(sum(runs[m]["meta"]["fit_seconds"] for m in members)),
                      infer_ms=summary["ms_per_image"], params=summary["params"]))
    ex = pd.DataFrame(extra)
    ex["phase"], ex["features"] = "Phase 2 (Deep Learning)", "Learned CNN filters"
    lb = pd.concat([lb[~lb.model.isin(ex.model)], ex], ignore_index=True).sort_values("accuracy", ascending=False)
    lb.to_csv(TABLE_DIR / "p2_final_leaderboard.csv", index=False)

    (TABLE_DIR / "p2_final_summary.json").write_text(json.dumps(summary, indent=1, default=float))
    print(json.dumps({k: v for k, v in summary.items() if k not in ("reliability_before", "reliability_after", "coverage_curve")}, indent=1, default=float))

    # ================= figures =================
    # seeds
    fig, ax = plt.subplots(figsize=(10, 4.6))
    labels = {"CNN3Block_GAP_E40": "CNN3Block-GAP (40 epochs)", "ResNetSmall": "ResNetSmall"}
    for i, fm in enumerate(fams):
        d = per_run[per_run.family == fm]
        ax.scatter(d.test_acc * 100, [i] * len(d), s=110, color=CNN if fm.startswith("Res") else AQUA,
                   edgecolor="white", linewidth=1.5, zorder=3)
        m, s = d.test_acc.mean() * 100, d.test_acc.std() * 100 if len(d) > 1 else 0
        ax.errorbar(m, i + 0.22, xerr=s, fmt="D", color=INK, capsize=4, ms=6)
        ax.text(m, i + 0.42, f"mean {m:.2f}% ± {s:.2f}", ha="center", fontsize=10, color=INK)
    ax.axvline(93.09, color=CNN, ls="--", lw=1.3)
    ax.text(93.09, -0.55, " first Phase 2 run 93.09%", color=CNN, fontsize=9.5)
    ax.axvline(91.30, color=SVM, ls=":", lw=1.5)
    ax.text(91.30, -0.55, " SVM-RBF 91.30%", color=SVM, fontsize=9.5)
    ax.set_yticks(range(len(fams)), [labels.get(f, f) for f in fams])
    ax.set_ylim(-0.7, len(fams) - 0.3)
    ax.set_xlabel("Test accuracy (%) — each dot is one training run with a different random seed")
    ax.set_ylabel("Model family")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1f}%"))
    ax.set_title("Robustness: results hold across 3 random seeds")
    save(fig, "p2_10_seed_robustness")

    # ensemble ladder (best single -> TTA -> ensemble -> + bias)
    best_single = per_run.sort_values("val_acc", ascending=False).iloc[0]
    ladder = [("Phase 1 best\nSVM-RBF (HOG)", 91.30, SVM), ("First Phase 2\nCNN3Block-GAP", 93.09, GREY),
              (f"Best single net\n{best_single.run.replace('_', ' ')}", best_single.test_acc * 100, AQUA),
              ("+ flip TTA", best_single.test_acc_tta * 100, AQUA),
              (f"Ensemble\n{len(members)} nets + TTA", (pt.argmax(1) == yt).mean() * 100, CNN),
              ("+ Shirt bias\n(final)", summary["test_acc"] * 100, CNN)]
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, (l, v, c) in enumerate(ladder):
        ax.bar(i, v, color=c, width=0.62, zorder=2)
        ax.text(i, v + 0.06, f"{v:.2f}%", ha="center", fontsize=10.5, color=INK, fontweight="bold" if i == 5 else "normal")
    ax.set_xticks(range(len(ladder)), [l for l, _, _ in ladder], fontsize=10)
    lo_y = min(v for _, v, _ in ladder) - 0.8
    ax.set_ylim(lo_y, max(v for _, v, _ in ladder) + 0.6)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1f}%"))
    ax.set_ylabel("Test accuracy on 10,000 images (%)")
    ax.set_xlabel("Each step adds one improvement (all choices made on validation data)")
    ax.grid(axis="x", visible=False)
    ax.set_title("Step-by-step: how the final Phase 2 model was built")
    save(fig, "p2_11_improvement_ladder")

    # reliability diagram
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5), sharey=True)
    for ax, rel, e, t in [(axes[0], rel_before, ece_before, "Before temperature scaling"),
                          (axes[1], rel_after, ece_after, f"After temperature scaling (T = {T:.2f})")]:
        conf = np.array([r[2] for r in rel]) * 100
        acc = np.array([r[3] for r in rel]) * 100
        w = np.array([r[1] - r[0] for r in rel]) * 100
        ax.bar([r[0] * 100 for r in rel], acc, width=w, align="edge", color=CNN, alpha=0.85, edgecolor="white", label="Actual accuracy in bin")
        ax.plot([0, 100], [0, 100], color=SVM, ls="--", lw=1.5, label="Perfect calibration")
        ax.set_xlim(30, 100); ax.set_ylim(30, 101)
        ax.set_title(f"{t}\nECE = {e*100:.2f}%", fontsize=12.5)
        ax.set_xlabel("Model's confidence in its answer (%)")
        ax.xaxis.set_major_formatter(PCT); ax.yaxis.set_major_formatter(PCT)
    axes[0].set_ylabel("How often it was actually right (%)")
    axes[0].legend(loc="upper left")
    fig.suptitle("Calibration: does 90% confidence really mean 90% correct?", x=0.06, ha="left", fontweight="bold", fontsize=14, y=1.04)
    save(fig, "p2_12_calibration")

    # abstention curve
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(cov_t * 100, acc_t * 100, color=CNN, lw=2.5)
    ax.scatter([abst["test_coverage"] * 100], [abst["test_acc_on_kept"] * 100], s=90, color=SVM, zorder=3)
    ax.annotate(f"Abstain below {thr*100:.0f}% confidence:\nanswers {abst['test_coverage']*100:.1f}% of images\nat {abst['test_acc_on_kept']*100:.2f}% accuracy",
                (abst["test_coverage"] * 100, abst["test_acc_on_kept"] * 100), xytext=(-250, -70), textcoords="offset points",
                fontsize=10.5, arrowprops=dict(arrowstyle="->", color=INK2))
    ax.set_xlim(40, 100.5); ax.set_ylim(92, 100.2)
    ax.xaxis.set_major_formatter(PCT); ax.yaxis.set_major_formatter(PCT)
    ax.set_xlabel("Coverage: % of test images the model answers (most confident first)")
    ax.set_ylabel("Accuracy on the answered images (%)")
    ax.set_title("Know when to say \"not sure\": hand the hardest images to a human")
    save(fig, "p2_13_abstain_curve")

    # shirt bias trade-off
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(bias_df.bias, bias_df.test_shirt_recall * 100, color=CNN, lw=2.4, marker="o", ms=4, label="Shirt recall (test)")
    ax.plot(bias_df.bias, bias_df.test_shirt_precision * 100, color=SVM, lw=2.4, marker="o", ms=4, label="Shirt precision (test)")
    ax.plot(bias_df.bias, bias_df.test_acc * 100, color=INK2, lw=2.4, ls="--", label="Overall accuracy (test)")
    ax.axvline(b_star, color=AQUA, lw=1.5)
    ax.text(b_star + 0.02, 70.5, f"chosen on validation: +{b_star:.2f}", color=AQUA, fontsize=10)
    ax.set_xlabel("Extra score added to the 'Shirt' class before picking the answer (log-probability units)")
    ax.set_ylabel("Percentage (%)")
    ax.yaxis.set_major_formatter(PCT)
    ax.set_ylim(70, 100)
    ax.legend(loc="lower right")
    ax.set_title("Shirt trade-off: catch more shirts vs. raise false alarms")
    save(fig, "p2_14_shirt_bias")


if __name__ == "__main__":
    main()
