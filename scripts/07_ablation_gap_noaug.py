"""Phase 2 / Step 7 - Controlled ablation: CNN3Block_GAP trained WITHOUT augmentation.

Why this run exists
-------------------
The original Phase 2 grid changed three things at once between CNN2Block and
CNN3Block_GAP (depth, Global Average Pooling AND augmentation), so the +0.93 pt
gain could not be attributed to any single change. This run keeps the
architecture fixed (CNN3Block_GAP) and removes augmentation, which completes a
2 x 2 grid:

                    no augmentation      augmentation
    CNN2Block       CNN2Block            CNN2Block_Aug
    CNN3Block_GAP   CNN3Block_GAP_NoAug  CNN3Block_GAP

Same split (48k/12k/10k, seed 42), optimiser, schedule and epoch budget as the
augmented CNN3Block_GAP run. Outputs follow the existing naming convention and
never overwrite the original runs.
"""
import json
import sys
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

NAME = "CNN3Block_GAP_NoAug"


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    run_file = RUN_DIR / f"{NAME}__pixels.json"
    if run_file.exists() and "--force" not in sys.argv:
        print(f"[{NAME}] cached at {run_file}; pass --force to retrain.")
        return

    device = pick_device()
    print(f"=== Ablation: {NAME} on {device} ===", flush=True)
    S = make_splits(seed=SEED)
    set_seed(SEED)
    loaders = build_dataloaders(S, batch_size=128, augment_train=False, seed=SEED)
    model = get_model("CNN3Block_GAP")
    n_params = count_parameters(model)

    model, info = train_model(model, loaders["tr"], loaders["val"], device,
                              epochs=20, lr=1e-3, patience=5,
                              log_fn=lambda m: print(m, flush=True))
    hist = pd.DataFrame(info["history"])
    hist.to_csv(TABLE_DIR / f"{NAME}__history.csv", index=False)

    test_m, preds, _ = evaluate_model(model, loaders["test"], device)
    np.save(RUN_DIR / f"{NAME}__pixels__pred.npy", preds)
    latency = measure_inference_latency(model, loaders["test"], device=device)
    torch.save(model.state_dict(), MODEL_DIR / f"{NAME}.pt")

    res = {
        "model": NAME,
        "features": "pixels (learned CNN)",
        "dim": 784,
        "val_acc": float(hist["val_acc"].max()),
        "val_macro_f1": float(hist["val_macro_f1"].max()),
        "accuracy": test_m["accuracy"],
        "macro_f1": test_m["macro_f1"],
        "macro_precision": test_m["macro_precision"],
        "fit_seconds": info["fit_seconds"],
        "infer_ms_per_image": round(latency, 4),
        "params": n_params,
        "params_note": "weights + biases (learnable)",
        "per_class_recall": test_m["per_class_recall"],
        "device": str(device),
    }
    run_file.write_text(json.dumps(res, indent=2))
    print(f"==> {NAME}: test acc {res['accuracy']*100:.2f}% | F1 {res['macro_f1']:.4f}", flush=True)


if __name__ == "__main__":
    main()
