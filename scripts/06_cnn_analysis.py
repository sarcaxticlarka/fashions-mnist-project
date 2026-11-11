"""Phase 2 / Step 6 - Post-hoc analysis & comparison of Classical ML (Phase 1) vs CNNs (Phase 2).

Generates:
  1. all_phases_model_results.csv: Unified leaderboard of Phase 1 + Phase 2.
  2. mcnemar_phase1_vs_phase2.csv: Statistical significance tests between SVM-RBF(HOG) and CNNs.
  3. cnn_01_training_curves.png: Epoch loss & validation curves for all CNN variants.
  4. res_09_per_class_recall_with_cnn.png: Full per-class recall heatmap comparison.
  5. res_10_upper_body_error_share_with_cnn.png: Upper-body garment error share comparison.
  6. cnn_02_gradcam_gallery.png: Grad-CAM explainability maps for ambiguous upper-body garments.
  7. res_11_pareto_with_cnn.png: Accuracy vs Latency Pareto frontier across all models.
"""
import json
import sys
from pathlib import Path
import warnings

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import torch

from src.config import CLASS_NAMES, FIG_DIR, MODEL_DIR, RUN_DIR, TABLE_DIR, UPPER_BODY
from src.data_loader import make_splits
from src.plotting import save, COLORS
from src.cnn_models import get_model
from src.cnn_explain import GradCAM


def load_all_runs():
    runs = {}
    for f in RUN_DIR.glob("*__*.json"):
        try:
            r = json.loads(f.read_text())
            pred_f = RUN_DIR / f"{r['model']}__{r['features']}__pred.npy"
            if not pred_f.exists():
                pred_f = RUN_DIR / f"{r['model']}__pixels__pred.npy"
            if pred_f.exists():
                r["pred"] = np.load(pred_f)
                runs[(r["model"], r["features"])] = r
        except Exception:
            continue
    return runs


def mcnemar_stat(y_true, p1, p2):
    c1, c2 = (p1 == y_true), (p2 == y_true)
    b, c = int((c1 & ~c2).sum()), int((~c1 & c2).sum())
    stat = (abs(b - c) - 1) ** 2 / max(b + c, 1)
    p_val = float(1 - stats.chi2.cdf(stat, 1))
    return dict(only_model1=b, only_model2=c, chi2=round(stat, 2), p_value=p_val)


def main():
    print("=== Generating Phase 1 vs Phase 2 Comparative Analysis ===")
    S = make_splits()
    runs = load_all_runs()

    # 1. Combined leaderboard
    rows = []
    p1_df = pd.read_csv(TABLE_DIR / "model_results.csv")
    for _, r in p1_df.iterrows():
        rows.append({
            "phase": "Phase 1 (Classical)",
            "model": r["model"],
            "features": r["features"],
            "accuracy": r["accuracy"],
            "macro_f1": r["macro_f1"],
            "fit_seconds": r["fit_seconds"],
            "infer_ms": r["infer_ms_per_image"],
            "params": r["params"],
        })

    cnn_df = pd.read_csv(TABLE_DIR / "cnn_model_results.csv")
    for _, r in cnn_df.iterrows():
        rows.append({
            "phase": "Phase 2 (Deep Learning)",
            "model": r["model"],
            "features": "Learned CNN filters",
            "accuracy": r["accuracy"],
            "macro_f1": r["macro_f1"],
            "fit_seconds": r["fit_seconds"],
            "infer_ms": r["infer_ms_per_image"],
            "params": r["params"],
        })

    all_df = pd.DataFrame(rows).sort_values("accuracy", ascending=False)
    all_df.to_csv(TABLE_DIR / "all_phases_model_results.csv", index=False)
    print("\n--- Unified Model Leaderboard ---")
    print(all_df.to_string(index=False))

    # 2. McNemar Statistical Significance: SVM-RBF(HOG) vs CNNs
    svm_key = ("SVM-RBF", "hog")
    if svm_key in runs:
        svm_pred = runs[svm_key]["pred"]
        mc_rows = []
        for cnn_name in ["CNN2Block", "CNN2Block_Aug", "CNN3Block_GAP"]:
            cnn_key = (cnn_name, "pixels (learned CNN)")
            if cnn_key in runs:
                cnn_pred = runs[cnn_key]["pred"]
                mc = mcnemar_stat(S.y_test, svm_pred, cnn_pred)
                delta = float(runs[cnn_key]["accuracy"] - runs[svm_key]["accuracy"])
                mc_rows.append({
                    "model_1": "SVM-RBF (HOG)",
                    "model_2": cnn_name,
                    "acc_model_1": runs[svm_key]["accuracy"],
                    "acc_model_2": runs[cnn_key]["accuracy"],
                    "delta_acc": round(delta, 4),
                    "svm_only_correct": mc["only_model1"],
                    "cnn_only_correct": mc["only_model2"],
                    "chi2": mc["chi2"],
                    "p_value": mc["p_value"],
                    "significant_p001": mc["p_value"] < 0.001,
                })
        df_mc = pd.DataFrame(mc_rows)
        df_mc.to_csv(TABLE_DIR / "mcnemar_phase1_vs_phase2.csv", index=False)
        print("\n--- McNemar Significance (SVM-RBF vs CNNs) ---")
        print(df_mc.to_string(index=False))

    # 3. Training curves plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    cnn_models = ["CNN2Block", "CNN2Block_Aug", "CNN3Block_GAP"]
    colors_map = {"CNN2Block": "#e74c3c", "CNN2Block_Aug": "#f39c12", "CNN3Block_GAP": "#27ae60"}

    for name in cnn_models:
        hf = TABLE_DIR / f"{name}__history.csv"
        if hf.exists():
            h = pd.read_csv(hf)
            c = colors_map.get(name, "#333")
            axes[0].plot(h["epoch"], h["train_loss"], "--", color=c, alpha=0.5)
            axes[0].plot(h["epoch"], h["val_loss"], "-", color=c, label=f"{name} (val)", lw=1.8)

            axes[1].plot(h["epoch"], h["val_acc"], "-", color=c, label=f"{name} acc", lw=1.8)
            axes[1].plot(h["epoch"], h["val_macro_f1"], ":", color=c, label=f"{name} F1", lw=1.5)

    axes[0].set_title("Training & Validation Loss Across Epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross Entropy Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].set_title("Validation Accuracy & Macro-F1")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)
    plt.tight_layout()
    save(fig, "cnn_01_training_curves")

    # 4. Per-Class Recall Heatmap Comparison
    recall_dict = {}
    # Key classical models
    for k in [("LogReg", "pixels"), ("LogReg", "hog"), ("RandomForest", "pixels"),
              ("SVM-RBF", "pixels"), ("SVM-RBF", "hog")]:
        if k in runs:
            recall_dict[f"{k[0]} ({k[1]})"] = runs[k]["per_class_recall"]
    # CNN models
    for name in cnn_models:
        k = (name, "pixels (learned CNN)")
        if k in runs:
            recall_dict[f"{name} (CNN)"] = runs[k]["per_class_recall"]

    df_rec = pd.DataFrame(recall_dict).T[CLASS_NAMES]
    df_rec.to_csv(TABLE_DIR / "per_class_recall_phase2.csv")

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(df_rec, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0.6, vmax=1.0, ax=ax, annot_kws={"size": 8.5})
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    ax.set_title("Test-Set Per-Class Recall: Classical ML vs. Deep Learning (CNNs)")
    plt.tight_layout()
    save(fig, "res_09_per_class_recall_with_cnn")

    # 5. Upper-body Error Share Comparison
    idx_upper = [CLASS_NAMES.index(c) for c in UPPER_BODY]
    err_rows = []
    for (m, f), r in runs.items():
        if "pred" not in r:
            continue
        err = r["pred"] != S.y_test
        within = err & np.isin(S.y_test, idx_upper) & np.isin(r["pred"], idx_upper)
        phase_lbl = "Phase 2 (CNN)" if "CNN" in m else "Phase 1 (Classical)"
        err_rows.append({
            "model_display": f"{m} ({f.split()[0]})",
            "phase": phase_lbl,
            "errors": int(err.sum()),
            "upper_body_errors": int(within.sum()),
            "share_pct": round(100 * within.sum() / max(err.sum(), 1), 1),
            "accuracy": r["accuracy"],
        })
    df_err = pd.DataFrame(err_rows).sort_values("accuracy")
    df_err.to_csv(TABLE_DIR / "upper_body_error_share_with_cnn.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 4.2))
    pal = {"Phase 1 (Classical)": "#3498db", "Phase 2 (CNN)": "#2ecc71"}
    sns.barplot(data=df_err, x="model_display", y="share_pct", hue="phase", palette=pal, ax=ax)
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("% of all errors inside Upper-Body")
    ax.set_xlabel("")
    ax.set_title("Upper-Body Error Bottleneck: Classical ML vs CNNs")
    for c in ax.containers:
        ax.bar_label(c, fmt="%.1f%%", fontsize=7.5, padding=2)
    plt.tight_layout()
    save(fig, "res_10_upper_body_error_share_with_cnn")

    # 6. Grad-CAM Explainability Gallery
    print("\n--- Generating Grad-CAM Interpretability Visualizations ---")
    best_cnn_name = "CNN3Block_GAP" if (MODEL_DIR / "CNN3Block_GAP.pt").exists() else "CNN2Block"
    model_obj = get_model(best_cnn_name)
    model_obj.load_state_dict(torch.load(MODEL_DIR / f"{best_cnn_name}.pt", map_location="cpu"))
    grad_cam = GradCAM(model_obj, model_obj.get_last_conv_layer())

    # Pick samples from the upper-body classes (T-shirt, Pullover, Coat, Shirt)
    target_classes = [0, 2, 4, 6]  # T-shirt, Pullover, Coat, Shirt
    fig, axes = plt.subplots(4, 4, figsize=(9.5, 9.5))
    
    for row_idx, cls_idx in enumerate(target_classes):
        cls_name = CLASS_NAMES[cls_idx]
        sample_indices = np.where(S.y_test == cls_idx)[0][:2]  # Pick 2 samples per class
        for col_idx, s_idx in enumerate(sample_indices):
            img_np = S.X_test[s_idx]
            t_input = torch.from_numpy(img_np[np.newaxis, np.newaxis, :, :]).float()
            cam = grad_cam.generate(t_input, target_class=cls_idx)

            ax_img = axes[row_idx, col_idx * 2]
            ax_img.imshow(img_np, cmap="gray")
            ax_img.axis("off")
            ax_img.set_title(f"{cls_name}\n(Input Image)", fontsize=8)

            ax_cam = axes[row_idx, col_idx * 2 + 1]
            ax_cam.imshow(img_np, cmap="gray")
            ax_cam.imshow(cam, cmap="jet", alpha=0.55)
            ax_cam.axis("off")
            ax_cam.set_title(f"{cls_name}\nGrad-CAM Heatmap", fontsize=8, color="#1b4f72")

    grad_cam.remove_hooks()
    plt.suptitle(f"Grad-CAM Salience Maps ({best_cnn_name}) on Upper-Body Categories", fontweight="bold", y=0.99)
    plt.tight_layout()
    save(fig, "cnn_02_gradcam_gallery")

    # 7. Pareto plot with CNNs
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    for _, r in all_df.iterrows():
        lat = r["infer_ms"]
        acc = r["accuracy"]
        is_cnn = "Phase 2" in r["phase"]
        col = "#27ae60" if is_cnn else "#2980b9"
        marker = "^" if is_cnn else "o"
        sz = 120 if is_cnn else 80
        ax.scatter(lat, acc, s=sz, marker=marker, color=col, edgecolor="k", lw=0.7)
        ax.annotate(f"{r['model']} ({r['params']//1000}k)", (lat, acc), fontsize=7, xytext=(4, 3), textcoords="offset points")

    ax.set_xscale("log")
    ax.set_xlabel("Single-Image CPU Inference Latency (ms, log scale)")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Accuracy vs. Inference Latency & Parameter Count (Phase 1 Classical vs Phase 2 CNN)")
    ax.grid(True, alpha=0.3)
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2980b9", markersize=9, label="Phase 1 (Classical ML)"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#27ae60", markersize=9, label="Phase 2 (Deep Learning / CNN)"),
    ]
    ax.legend(handles=legend_elems, loc="lower right")
    plt.tight_layout()
    save(fig, "res_11_pareto_with_cnn")

    print("\nAll Phase 2 figures, tables, and comparative analyses generated successfully.")


if __name__ == "__main__":
    main()

