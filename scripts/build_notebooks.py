"""Generate the three Phase-1 notebooks (thin narrative layer over src/).  Execute with:
   python scripts/build_notebooks.py && jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb"""
from pathlib import Path
import nbformat as nbf

NB = Path(__file__).resolve().parents[1] / "notebooks"
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
SETUP = """import sys, warnings; sys.path.insert(0, '..'); warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from IPython.display import Image, display
from src.config import *
from src.data_loader import load_raw, to_unit, make_splits
from src.plotting import set_style
set_style()
def show(name, w=900): display(Image(filename=str(FIG_DIR / f'{name}.png'), width=w))"""

def write(name, cells):
    nb = nbf.v4.new_notebook(); nb.cells = cells
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nbf.write(nb, NB / name)

# ---------------------------------------------------------------- 01 EDA
write("01_eda.ipynb", [
    md("# 01 · Dataset Quality & Exploratory Data Analysis\nFashion-MNIST (Xiao et al., 2017): 70,000 28×28 grayscale clothing images, 10 classes.\n\nEvery plot is produced by `src/eda.py`; this notebook narrates the findings."),
    code(SETUP),
    md("## 1. Load & audit\nIDX files are read directly (`src/data_loader.py`). We check shape, value range, missing values, blank images, exact duplicates and train/test leakage."),
    code("Xtr_u8, ytr, Xte_u8, yte = load_raw()\nfrom src import eda\neda.quality_report(Xtr_u8, ytr, Xte_u8, yte)"),
    md("**Finding:** the data is clean - no missing/blank/duplicate images and zero train-test overlap. The only preprocessing step is scaling to [0,1]."),
    md("## 2. Class balance"),
    code("show('eda_01_class_distribution', 700); pd.read_csv(TABLE_DIR/'eda_class_distribution.csv')"),
    md("Perfectly balanced (6,000 / 1,000 per class) -> accuracy is a valid headline metric, but we still report macro-F1 and per-class recall."),
    md("## 3. What do the images look like?"),
    code("show('eda_02_sample_grid', 650); show('eda_12_ambiguous_group', 650)"),
    md("The four upper-body classes (T-shirt/top, Pullover, Coat, Shirt) are visually near-identical silhouettes -> expected hard group."),
    md("## 4. Class means, intensity and shape statistics"),
    code("show('eda_03_mean_std_images'); show('eda_04_intensity_shape_boxplots'); pd.read_csv(TABLE_DIR/'eda_class_summary_stats.csv', index_col=0)"),
    code("show('eda_05_pixel_histogram')"),
    md("About 50 % of pixels are exactly zero (background): the data is sparse. Footwear has small height / large width; trousers are narrow; bags have the widest variance in size."),
    md("## 5. Which pixels matter?"),
    code("show('eda_06_pixel_variance_fscore', 1000)"),
    md("The border pixels have ~zero variance and ~zero F-score: they carry no information. Discriminative pixels lie at the centre and along the silhouette edges."),
    md("## 6. Is there spatial structure that flattening destroys?"),
    code("show('eda_07_spatial_correlation', 600); pd.read_csv(TABLE_DIR/'eda_class_pair_distance.csv').head(8)"),
    md("Adjacent pixels correlate at r≈0.75, far above the shuffled null (≈0.16). Flattening treats pixel 5 and pixel 6 like any two unrelated pixels, which is the motivation for neighbourhood-based descriptors (HOG)."),
    code("show('eda_08_class_mean_distance', 650)"),
    md("Mean-image distances predict the confusion structure: **Pullover-Coat-Shirt-T-shirt** are the closest pairs."),
    md("## 7. Intrinsic dimensionality & embeddings"),
    code("show('eda_09_pca'); show('eda_10_tsne', 650); pd.read_csv(TABLE_DIR/'eda_pca_components.csv')"),
    md("24 principal components explain 80 % and 187 explain 95 % of the variance (of 784) -> strong redundancy. t-SNE shows footwear, trousers and bags as clean clusters while the upper-body classes overlap."),
    md("## 8. Train/test shift & orientation"),
    code("show('eda_11_train_test_shift', 800); pd.read_csv(TABLE_DIR/'eda_orientation_check.csv', index_col=0)"),
    md("KS test: no shift between train and test. Images are upright (footwear/bags heavier at the bottom, tops/trousers heavier at the top), so HOG can be computed directly without rotating."),
    md("### EDA conclusions\n1. Clean, balanced data; no cleaning required.\n2. Strong spatial structure + sparsity -> neighbourhood descriptors are justified.\n3. Upper-body group is the predicted difficulty.\n4. Heavy pixel redundancy -> PCA/HOG compress without losing the class signal."),
])

# ---------------------------------------------------------------- 02 FE
write("02_feature_engineering.ipynb", [
    md("# 02 · Feature Engineering\nThree representations are compared under identical protocol:\n\n| name | definition | dim |\n|---|---|---|\n| `pixels` | flattened intensities in [0,1] | 784 |\n| `hog` | Histogram of Oriented Gradients (skimage) | 864 |\n| `hog_stats` | HOG + 16 hand-crafted shape statistics | 880 |"),
    code(SETUP + "\nfrom src.features import *\nS = make_splits()"),
    md("## 1. Orientation check and scaling\nPixels are scaled to [0,1], labels untouched. Images are stored upright, so `skimage.feature.hog` is applied on the (row, col) array directly."),
    md("## 2. How HOG works\n1. Compute horizontal/vertical gradients `gx, gy`.\n2. Magnitude `sqrt(gx²+gy²)` and orientation `atan2(gy, gx)` per pixel.\n3. Split the image into cells (4×4 px); each cell makes a histogram of orientations, weighted by magnitude.\n4. Group 2×2 cells into overlapping blocks and L2-Hys normalise (illumination invariance).\n5. Concatenate all block vectors."),
    code("show('fe_02_hog_visual', 1000)"),
    md("## 3. HOG hyper-parameter ablation\nGrid: orientations {6,9,12} × cell {2,4,7} × block {1,2}; scored on validation with a Logistic-Regression probe (12k training images)."),
    code("pd.read_csv(TABLE_DIR/'fe_hog_ablation.csv').head(12)"),
    code("show('fe_01_hog_ablation', 1000)"),
    md("**Decision:** finer cells improve accuracy but explode dimensionality (4k-8k) and slow kNN/SVM/RF. We choose the best configuration under a 2,000-d budget: **6 orientations, 4×4 cells, 2×2 blocks (864-d)**."),
    md("## 4. Hand-crafted shape statistics\n16 interpretable global descriptors: intensity mean/std, ink fraction, bounding-box height/width/aspect, centroid, left-right and top-bottom symmetry, Sobel edge density, top/bottom mass, width of top/bottom rows and the gap between trouser legs."),
    code("show('fe_03_shape_stats_profile', 900); show('fe_04_shape_stats_importance', 500)"),
    md("## 5. PCA on pixels (reference compression)"),
    code("show('fe_05_pca_sweep', 520); pd.read_csv(TABLE_DIR/'fe_pca_sweep.csv')"),
    md("## 6. Feature-set comparison (same LR probe)"),
    code("show('fe_06_feature_set_probe', 520); pd.read_csv(TABLE_DIR/'fe_feature_set_probe.csv')"),
    md("The probe is deliberately simple; the full 4-model comparison is in notebook 03."),
])

# ---------------------------------------------------------------- 03 models
write("03_models_and_error_analysis.ipynb", [
    md("# 03 · Model Application, Evaluation & Error Analysis\nFour classical classifiers × three feature sets. Hyper-parameters are tuned on the validation split (drawn from training only); each configuration is refit on train+val and scored **once** on the official test set."),
    code(SETUP + "\nfrom src import analysis as A\nS = make_splits(); runs = A.load_runs(); df = pd.read_csv(TABLE_DIR/'model_results.csv')"),
    md("## 1. Leaderboard"),
    code("df.sort_values('accuracy', ascending=False)[['model','features','dim','val_acc','accuracy','macro_f1','fit_seconds','infer_ms_per_image','feat_ms_per_image','params','best_params']]"),
    code("show('res_01_accuracy_f1_overview', 1000)"),
    md("## 2. Cost: accuracy vs inference time and model size"),
    code("show('res_07_accuracy_vs_latency', 650)"),
    md("## 3. Data efficiency"),
    code("show('res_08_learning_curve', 520)"),
    md("## 4. Per-class recall"),
    code("show('res_02_per_class_recall', 950)"),
    md("## 5. Confusion matrices: best pixel model vs best HOG model"),
    code("show('res_03_confusion_best_pixels_vs_best_hog', 1100); show('res_04_confusion_difference', 600)"),
    md("## 6. Upper-body group (Shirt / T-shirt / Pullover / Coat)"),
    code("show('res_05_upper_body_error_share', 600); pd.read_csv(TABLE_DIR/'upper_body_error_share.csv')"),
    code("show('res_06a_misclassified_best_pixels', 800); show('res_06b_misclassified_best_hog', 800)"),
    md("## 7. Statistical validation"),
    code("pd.read_csv(TABLE_DIR/'mcnemar_pixels_vs_hog.csv')"),
    code("pd.read_csv(TABLE_DIR/'bootstrap_ci.csv')"),
    md("## 8. Largest confusions per model"),
    code("pd.read_csv(TABLE_DIR/'top_confusions_all.csv').query('features==\"hog\"').head(12)"),
])
print("notebooks written")
