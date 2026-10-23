"""Phase 1 / Step 4 - Evaluation, error analysis and statistics.  Run after 03_train_models.py"""
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
import numpy as np

from src import analysis as A
from src.config import PROCESSED_DIR
from src.data_loader import make_splits
from src.plotting import set_style

set_style()
S = make_splits()
runs = A.load_runs()
df = A.results_table(runs)
A.bar_overview(df)
A.per_class_recall_heatmap(runs)
ka, kb, cms = A.confusion_pair(runs, S.y_test)
A.difference_confusion(cms, ka, kb)
print("best pixels:", ka, " best hog:", kb)
print(A.top_confusions(runs, S.y_test).head(12))
print(A.upper_body_error_share(runs, S.y_test))
print(A.significance_table(runs, S.y_test))
print(A.bootstrap_ci(runs, S.y_test).head(12))
A.misclassified_gallery(runs, S.X_test, S.y_test, ka, name="res_06a_misclassified_best_pixels")
A.misclassified_gallery(runs, S.X_test, S.y_test, kb, name="res_06b_misclassified_best_hog")
A.pareto_plot(df)
fd = {f: dict(np.load(PROCESSED_DIR / f"features_{f}.npz")) for f in ("pixels", "hog")}
A.learning_curve(S, fd)
