"""Model factories, hyper-parameter grids and model-size accounting."""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .config import SEED

MODEL_NAMES = ("LogReg", "kNN", "RandomForest", "SVM-RBF")


def make_model(name, **p):
    if name == "LogReg":
        return make_pipeline(StandardScaler(), LogisticRegression(C=p.get("C", 1.0), max_iter=300, random_state=SEED))
    if name == "kNN":
        return KNeighborsClassifier(n_neighbors=p.get("k", 5), weights=p.get("weights", "distance"), n_jobs=-1)
    if name == "RandomForest":
        return RandomForestClassifier(n_estimators=p.get("n_estimators", 300), max_depth=p.get("max_depth"),
                                      min_samples_leaf=p.get("min_samples_leaf", 1), max_features="sqrt",
                                      n_jobs=-1, random_state=SEED)
    if name == "SVM-RBF":
        return SVC(kernel="rbf", C=p.get("C", 10.0), gamma=p.get("gamma", "scale"), random_state=SEED)
    raise ValueError(name)


# Small grids: every combination is scored on the VALIDATION split only.
GRIDS = {
    "LogReg": [dict(C=c) for c in (0.001, 0.01, 0.1, 1.0)],
    "kNN": [dict(k=k, weights=w) for k in (3, 5, 9) for w in ("uniform", "distance")],
    "RandomForest": [dict(n_estimators=300, max_depth=d, min_samples_leaf=l) for d in (20, None) for l in (1, 2)],
    "SVM-RBF": [dict(C=c) for c in (1.0, 5.0, 20.0)],
}
# SVM is O(n^2)-O(n^3): the grid is evaluated on a stratified subsample of the training split.
TUNE_SUBSAMPLE = {"SVM-RBF": 15000}


def count_parameters(model, name):
    """Number of stored numbers needed at inference time (definition depends on model family)."""
    if name == "LogReg":
        lr = model[-1]
        return int(lr.coef_.size + lr.intercept_.size), "weights + biases"
    if name == "SVM-RBF":
        return int(model.support_vectors_.size + model.dual_coef_.size + model.intercept_.size), "support vectors + dual coefficients"
    if name == "RandomForest":
        internal = sum(int((t.tree_.children_left != -1).sum()) for t in model.estimators_)
        return 2 * internal, "(feature, threshold) per internal node"
    if name == "kNN":
        return int(model._fit_X.size), "stored training matrix (non-parametric)"
    raise ValueError(name)
