"""Phase 1 / Step 2 - Feature engineering.  Run:  python scripts/02_feature_engineering.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from src import feature_experiments as fe
from src.config import PROCESSED_DIR
from src.data_loader import make_splits
from src.features import build_features
from src.plotting import set_style

set_style()
S = make_splits()
fe.hog_visual(S.X_tr, S.y_tr)
print(fe.hog_ablation(S).head(10))
fe.plot_hog_ablation(__import__("pandas").read_csv(__import__("src.config", fromlist=["x"]).TABLE_DIR / "fe_hog_ablation.csv"))
print(fe.shape_stats_analysis(S.X_tr, S.y_tr))
print(fe.pca_dimension_sweep(S))
F = {n: (build_features(S.X_tr, n), build_features(S.X_val, n)) for n in ("pixels", "hog", "hog_stats")}
print(fe.feature_set_comparison(S, F))
