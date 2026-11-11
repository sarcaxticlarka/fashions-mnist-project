"""Phase 2 / Step 5 - Convolutional Neural Networks training & ablation.

Models trained:
  1. CNN2Block: Baseline 2-block ConvNet (no data augmentation)
  2. CNN2Block_Aug: Baseline 2-block ConvNet with spatial data augmentation (flip + crop)
  3. CNN3Block_GAP: Deep 3-block ConvNet with Global Average Pooling and data augmentation

Outputs:
  - Model weights saved to results/models/{model_name}.pt
  - Predictions saved to results/tables/runs/{model_name}__pixels__pred.npy
  - Run metadata saved to results/tables/runs/{model_name}__pixels.json
  - Result tables saved to results/tables/cnn_model_results.csv
"""
import json
import sys
import time
from pathlib import Path
import warnings

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch

from src.config import MODEL_DIR, RUN_DIR, TABLE_DIR, SEED
from src.data_loader import make_splits
from src.cnn_dataset import build_dataloaders
from src.cnn_models import get_model, count_parameters
from src.cnn_trainer import train_model, evaluate_model, measure_inference_latency, set_seed


def main():
    force = "--force" in sys.argv
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Phase 2: CNN Training Pipeline ===")
    print(f"PyTorch Device: {device} | Random Seed: {SEED}\n")

    S = make_splits(seed=SEED)

    experiments = [
        {
            "name": "CNN2Block",
            "model_type": "CNN2Block",
            "augment": False,
            "epochs": 15,
            "lr": 1e-3,
            "desc": "Baseline 2-Block ConvNet (no augmentation)",
        },
        {
            "name": "CNN2Block_Aug",
            "model_type": "CNN2Block",
            "augment": True,
            "epochs": 18,
            "lr": 1e-3,
            "desc": "2-Block ConvNet + Spatial Augmentation",
        },
        {
            "name": "CNN3Block_GAP",
            "model_type": "CNN3Block_GAP",
            "augment": True,
            "epochs": 20,
            "lr": 1e-3,
            "desc": "Deep 3-Block ConvNet + GAP + Augmentation",
        },
    ]

    summary_rows = []

    for exp in experiments:
        name = exp["name"]
        run_file = RUN_DIR / f"{name}__pixels.json"
        pred_file = RUN_DIR / f"{name}__pixels__pred.npy"
        model_file = MODEL_DIR / f"{name}.pt"
        history_file = TABLE_DIR / f"{name}__history.csv"

        if run_file.exists() and pred_file.exists() and model_file.exists() and not force:
            print(f"[{name}] Found cached run. Loading from {run_file}...")
            cached = json.loads(run_file.read_text())
            summary_rows.append(cached)
            continue

        print(f"\n--- Training {name} ({exp['desc']}) ---")
        set_seed(SEED)

        # Build loaders
        loaders = build_dataloaders(S, batch_size=128, augment_train=exp["augment"], seed=SEED)

        # Build model
        model = get_model(exp["model_type"])
        n_params = count_parameters(model)
        print(f"Trainable parameters: {n_params:,}")

        # Train model with early stopping on validation
        trained_model, train_info = train_model(
            model=model,
            train_loader=loaders["tr"],
            val_loader=loaders["val"],
            device=device,
            epochs=exp["epochs"],
            lr=exp["lr"],
            patience=5,
            log_fn=lambda msg: print(msg, flush=True),
        )

        # Save training history
        df_hist = pd.DataFrame(train_info["history"])
        df_hist.to_csv(history_file, index=False)

        # Final evaluation on the unseen official test set (used once)
        test_metrics, test_preds, test_targets = evaluate_model(trained_model, loaders["test"], device)
        np.save(pred_file, test_preds)

        # Measure CPU single-sample inference latency
        infer_latency_ms = measure_inference_latency(trained_model, loaders["test"], device=device)

        # Save PyTorch model state dict
        torch.save(trained_model.state_dict(), model_file)

        res = {
            "model": name,
            "features": "pixels (learned CNN)",
            "dim": 784,
            "val_acc": float(df_hist["val_acc"].max()),
            "val_macro_f1": float(df_hist["val_macro_f1"].max()),
            "accuracy": test_metrics["accuracy"],
            "macro_f1": test_metrics["macro_f1"],
            "macro_precision": test_metrics["macro_precision"],
            "fit_seconds": train_info["fit_seconds"],
            "infer_ms_per_image": round(infer_latency_ms, 4),
            "params": n_params,
            "params_note": "weights + biases (learnable)",
            "per_class_recall": test_metrics["per_class_recall"],
        }

        run_file.write_text(json.dumps(res, indent=2))
        summary_rows.append(res)
        print(
            f"==> [{name}] Test Accuracy: {res['accuracy'] * 100:.2f}% | "
            f"Macro-F1: {res['macro_f1']:.4f} | "
            f"Latency: {res['infer_ms_per_image']:.3f} ms/img | "
            f"Params: {n_params:,}"
        )

    # Save CNN results table
    df_cnn = pd.DataFrame(summary_rows)
    df_cnn_export = df_cnn[[
        "model", "features", "val_acc", "accuracy", "macro_f1", "fit_seconds",
        "infer_ms_per_image", "params"
    ]]
    df_cnn_export.to_csv(TABLE_DIR / "cnn_model_results.csv", index=False)
    print("\n=== Phase 2 Summary Table ===")
    print(df_cnn_export.to_string(index=False))


if __name__ == "__main__":
    main()

