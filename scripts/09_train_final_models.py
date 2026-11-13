"""Phase 2 / Step 9 - Final-model training: longer budgets, 3 seeds, residual network.

Addresses the open items from the Phase 2 self-review:
  * The best runs were still improving at their last epoch  -> 40-epoch budget, patience 10.
  * Single seed                                               -> seeds 42, 43, 44 per model.
  * Stronger architecture                                     -> ResNetSmall (residual blocks)
    trained with crop+flip+RandomErasing and label smoothing 0.1.

For every run we also store raw logits on the validation and test sets, with and
without horizontal-flip test-time augmentation (TTA). Those logits feed
scripts/10_ensemble_calibration.py (ensembles, temperature scaling, abstention).

Usage:  python scripts/09_train_final_models.py [--only ResNetSmall] [--force]
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

from src.config import MODEL_DIR, RUN_DIR, TABLE_DIR
from src.data_loader import make_splits
from src.cnn_dataset import build_dataloaders
from src.cnn_models import get_model, count_parameters
from src.cnn_trainer import train_model, evaluate_model, measure_inference_latency

LOGIT_DIR = RUN_DIR / "logits"
LOGIT_DIR.mkdir(parents=True, exist_ok=True)

FAMILIES = {
    "CNN3Block_GAP_E40": dict(arch="CNN3Block_GAP", epochs=40, erasing=0.0, smoothing=0.0),
    "ResNetSmall": dict(arch="ResNetSmall", epochs=40, erasing=0.25, smoothing=0.1),
}
SEEDS = [42, 43, 44]


def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def logits(model, loader, dev, flip=False):
    model.eval()
    out = []
    for x, _ in loader:
        x = x.to(dev)
        if flip:
            x = torch.flip(x, dims=[3])
        out.append(model(x).float().cpu().numpy())
    return np.concatenate(out)


def main():
    force = "--force" in sys.argv
    only = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None
    dev = device()
    S = make_splits()
    print(f"=== Final Phase 2 models on {dev} ===", flush=True)

    # Interleave families so partial runs still cover both architectures.
    plan = [(fam, seed) for seed in SEEDS for fam in FAMILIES if not only or fam == only]
    for fam, seed in plan:
        cfg = FAMILIES[fam]
        name = f"{fam}_s{seed}"
        run_file = RUN_DIR / f"{name}__pixels.json"
        if run_file.exists() and not force:
            print(f"[{name}] cached", flush=True)
            continue
        print(f"\n--- {name} ---", flush=True)
        loaders = build_dataloaders(S, batch_size=128, augment_train=True, seed=seed,
                                    random_erasing=cfg["erasing"])
        torch.manual_seed(seed)
        model = get_model(cfg["arch"])
        n_params = count_parameters(model)
        model, info = train_model(model, loaders["tr"], loaders["val"], dev, epochs=cfg["epochs"],
                                  lr=1e-3, patience=10, seed=seed, label_smoothing=cfg["smoothing"],
                                  log_fn=lambda m: print(m, flush=True))
        hist = pd.DataFrame(info["history"])
        hist.to_csv(TABLE_DIR / f"{name}__history.csv", index=False)

        test_m, preds, _ = evaluate_model(model, loaders["test"], dev)
        np.save(RUN_DIR / f"{name}__pixels__pred.npy", preds)
        np.savez_compressed(LOGIT_DIR / f"{name}.npz",
                            val=logits(model, loaders["val"], dev),
                            val_flip=logits(model, loaders["val"], dev, flip=True),
                            test=logits(model, loaders["test"], dev),
                            test_flip=logits(model, loaders["test"], dev, flip=True))
        latency = measure_inference_latency(model, loaders["test"], device=dev)
        torch.save(model.state_dict(), MODEL_DIR / f"{name}.pt")
        best = int(hist.val_macro_f1.idxmax())
        res = {
            "model": name, "family": fam, "arch": cfg["arch"], "seed": seed,
            "features": "pixels (learned CNN)", "dim": 784,
            "epochs_run": int(len(hist)), "best_epoch": best + 1,
            "val_acc": float(hist.val_acc[best]), "val_macro_f1": float(hist.val_macro_f1[best]),
            "accuracy": test_m["accuracy"], "macro_f1": test_m["macro_f1"],
            "macro_precision": test_m["macro_precision"], "fit_seconds": info["fit_seconds"],
            "infer_ms_per_image": round(latency, 4), "params": n_params,
            "label_smoothing": cfg["smoothing"], "random_erasing": cfg["erasing"],
            "per_class_recall": test_m["per_class_recall"], "device": str(dev),
        }
        run_file.write_text(json.dumps(res, indent=2))
        print(f"==> {name}: test {res['accuracy']*100:.2f}% | F1 {res['macro_f1']:.4f} | "
              f"best epoch {res['best_epoch']}/{res['epochs_run']} | {info['fit_seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
