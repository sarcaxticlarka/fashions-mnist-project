"""Phase 1 / Step 1 - Dataset quality & EDA.  Run:  python scripts/01_eda.py"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import eda
from src.config import TABLE_DIR
from src.data_loader import load_raw, to_unit
from src.plotting import set_style

set_style()
Xtr_u8, ytr, Xte_u8, yte = load_raw()
Xtr, Xte = to_unit(Xtr_u8), to_unit(Xte_u8)

print(eda.quality_report(Xtr_u8, ytr, Xte_u8, yte).to_string(index=False))
print(eda.class_distribution(ytr, yte))
eda.sample_grid(Xtr, ytr)
eda.mean_std_images(Xtr, ytr)
df = eda.per_image_stats(Xtr, ytr)
print(eda.intensity_and_shape_plots(df).to_string())
eda.pixel_histogram(Xtr, ytr)
extra = eda.pixel_variance_and_fscore(Xtr, ytr)
print(eda.spatial_correlation(Xtr))
print(eda.class_similarity(Xtr, ytr).head(8))
k, _ = eda.pca_analysis(Xtr, ytr)
extra["pca_components_for_variance"] = {str(a): b for a, b in k.items()}
eda.tsne_embedding(Xtr, ytr)
extra["train_test_shift"] = eda.train_test_shift(Xtr, Xte)
eda.similar_class_examples(Xtr, ytr)
print(eda.orientation_check(Xtr, ytr))
(TABLE_DIR / "eda_extra.json").write_text(json.dumps(extra, indent=2))
print(extra)
