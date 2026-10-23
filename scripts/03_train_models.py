"""Phase 1 / Step 3 - Model application: 4 classical models x 3 feature sets.

Run:  python scripts/03_train_models.py            (resumable: finished runs are cached)
      python scripts/03_train_models.py --force    (recompute everything)
"""
import json, sys, time, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR, TABLE_DIR, HOG_PARAMS
from src.data_loader import make_splits
from src.experiment import run
from src.features import build_features, FEATURE_SETS
from src.models import MODEL_NAMES

force = "--force" in sys.argv
S = make_splits()
F, feat_ms = {}, {}
for feat in FEATURE_SETS:
    cache = PROCESSED_DIR / f"features_{feat}.npz"
    if cache.exists() and not force:
        z = np.load(cache); F[feat] = {k: z[k] for k in ("tr", "val", "trval", "test")}
    else:
        tr, val, te = (build_features(X, feat) for X in (S.X_tr, S.X_val, S.X_test))
        F[feat] = dict(tr=tr, val=val, trval=np.concatenate([tr, val]), test=te)
        np.savez_compressed(cache, **F[feat])
    t0 = time.time(); build_features(S.X_test[:2000], feat); feat_ms[feat] = 1000 * (time.time() - t0) / 2000
    print(feat, F[feat]["tr"].shape, f"extract {feat_ms[feat]:.3f} ms/img (parallel)", flush=True)

rows = []
for feat in FEATURE_SETS:
    for name in MODEL_NAMES:
        res = run(name, feat, S, F, force=force, log=lambda m: print(m, flush=True))
        res["feat_ms_per_image"] = feat_ms[feat]
        rows.append({k: v for k, v in res.items() if k not in ("search", "per_class_recall", "best_params")} | {"best_params": json.dumps(res["best_params"])})
df = pd.DataFrame(rows)
df["total_ms_per_image"] = df.infer_ms_per_image + df.feat_ms_per_image
df.to_csv(TABLE_DIR / "model_results.csv", index=False)
print(df[["model", "features", "dim", "val_acc", "accuracy", "macro_f1", "fit_seconds", "infer_ms_per_image", "params"]].to_string())
