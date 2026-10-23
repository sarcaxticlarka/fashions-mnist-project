"""Run one (model, feature set) experiment: tune on validation -> refit on train+val -> test."""
import json
import time

import joblib

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix

from .config import RUN_DIR, SEED, CLASS_NAMES, MODEL_DIR
from .features import build_features
from .models import make_model, GRIDS, TUNE_SUBSAMPLE, count_parameters


def metrics(y_true, y_pred):
    return dict(
        accuracy=float(accuracy_score(y_true, y_pred)),
        macro_f1=float(f1_score(y_true, y_pred, average="macro")),
        macro_precision=float(precision_score(y_true, y_pred, average="macro")),
        per_class_recall={c: float(r) for c, r in zip(CLASS_NAMES, recall_score(y_true, y_pred, average=None))},
    )


def run(name, feat, S, F, force=False, log=print):
    """F: dict feature_name -> dict(tr, val, trval, test) matrices (cached upstream)."""
    out = RUN_DIR / f"{name}__{feat}.json"
    if out.exists() and not force:
        return json.loads(out.read_text())
    Ftr, Fval, Ftv, Fte = (F[feat][k] for k in ("tr", "val", "trval", "test"))
    ytr, yval, ytv, yte = S.y_tr, S.y_val, S.y_trainval, S.y_test

    # --- 1. hyper-parameter search on the validation split (never touches test)
    sub = TUNE_SUBSAMPLE.get(name)
    if sub:
        idx = np.random.default_rng(SEED).choice(len(Ftr), sub, replace=False)
        Fa, ya = Ftr[idx], ytr[idx]
    else:
        Fa, ya = Ftr, ytr
    search = []
    for params in GRIDS[name]:
        t0 = time.time()
        m = make_model(name, **params).fit(Fa, ya)
        pred = m.predict(Fval)
        search.append(dict(params=params, val_acc=float(accuracy_score(yval, pred)),
                           val_macro_f1=float(f1_score(yval, pred, average="macro")), sec=round(time.time() - t0, 1)))
        log(f"  [{name}/{feat}] {params} val_acc={search[-1]['val_acc']:.4f} ({search[-1]['sec']}s)")
    best = max(search, key=lambda r: r["val_acc"])

    # --- 2. refit on train+val with the chosen hyper-parameters, evaluate once on test
    t0 = time.time()
    model = make_model(name, **best["params"]).fit(Ftv, ytv)
    fit_s = time.time() - t0
    if hasattr(model, "set_params") and name in ("RandomForest", "kNN"):
        model.set_params(n_jobs=1)  # single-thread inference timing for a fair latency comparison
    t0 = time.time()
    pred = model.predict(Fte)
    infer_s = time.time() - t0
    n_params, note = count_parameters(model, name)
    res = dict(model=name, features=feat, dim=int(Ftr.shape[1]), best_params=best["params"], search=search,
               val_acc=best["val_acc"], fit_seconds=round(fit_s, 1),
               infer_ms_per_image=1000 * infer_s / len(Fte), params=n_params, params_note=note, **metrics(yte, pred))
    np.save(RUN_DIR / f"{name}__{feat}__pred.npy", pred)
    out.write_text(json.dumps(res, indent=2))
    log(f"==> {name}/{feat}: test acc={res['accuracy']:.4f} macroF1={res['macro_f1']:.4f} fit={fit_s:.0f}s infer={res['infer_ms_per_image']:.3f}ms/img")
    if name in ("LogReg", "SVM-RBF"):  # compact enough to ship with the demo app
        joblib.dump(model, MODEL_DIR / f"{name}__{feat}.joblib", compress=3)
    return res
