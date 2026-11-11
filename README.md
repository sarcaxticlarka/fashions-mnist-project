# Automatic Clothing Category Recognition — Phase 1 & Phase 2

**Project 11 · Inventory image tagging via handcrafted features and Convolutional Neural Networks.**
This repository contains the complete deliverables for:
- **Phase 1 (Classical Machine Learning):** Raw pixels vs. HOG descriptors evaluated across Logistic Regression, $k$-NN, Random Forest, and RBF Support Vector Machines.
- **Phase 2 (Deep Learning):** End-to-end learned hierarchical spatial representations evaluated across baseline 2-block ConvNets, spatial data augmentation, and deep 3-block ConvNets with Global Average Pooling (GAP) and Grad-CAM interpretability.

## Headline Results (Test set, 10,000 images)

### Unified Leaderboard (Phase 1 vs. Phase 2)

| Phase | Model | Representation | Test Acc | Macro-F1 | CPU Latency | Parameters |
|---|---|---|---|---|---|---|
| **Phase 2 (Deep Learning)** | **CNN3Block-GAP** | **Learned CNN filters** | **93.09%** | **0.9306** | **0.68 ms** | **304.7k** |
| Phase 2 (Deep Learning) | CNN2Block | Learned CNN filters | 92.16% | 0.9212 | 0.18 ms | 421.7k |
| Phase 2 (Deep Learning) | CNN2Block + Aug | Learned CNN filters | 91.41% | 0.9140 | 0.22 ms | 421.7k |
| Phase 1 (Classical ML) | SVM-RBF | HOG + 16 shape stats | 91.37% | 0.9137 | 5.28 ms | 17.5M |
| Phase 1 (Classical ML) | **SVM-RBF** | **HOG (Phase 1 Best)** | **91.30%** | **0.9130** | **5.36 ms** | **18.1M** |
| Phase 1 (Classical ML) | SVM-RBF | Raw Pixels | 89.88% | 0.8985 | 4.53 ms | 15.0M |
| Phase 1 (Classical ML) | Logistic Regression | HOG + stats | 88.94% | 0.8890 | 0.001 ms | 8.8k |
| Phase 1 (Classical ML) | Logistic Regression | HOG | 88.52% | 0.8848 | 0.001 ms | 8.7k |
| Phase 1 (Classical ML) | Random Forest | HOG + stats | 88.21% | 0.8811 | 0.041 ms | 2.5M |
| Phase 1 (Classical ML) | Random Forest | HOG | 87.81% | 0.8769 | 0.042 ms | 3.2M |
| Phase 1 (Classical ML) | $k$-NN | HOG | 87.74% | 0.8777 | 0.271 ms | 51.8M |
| Phase 1 (Classical ML) | Random Forest | Raw Pixels | 87.60% | 0.8746 | 0.037 ms | 2.5M |
| Phase 1 (Classical ML) | $k$-NN | Raw Pixels | 85.61% | 0.8564 | 0.316 ms | 47.0M |
| Phase 1 (Classical ML) | Logistic Regression | Raw Pixels | 84.56% | 0.8447 | 0.001 ms | 7.9k |

### Key Scientific Findings
1. **Statistical Superiority (McNemar Test)**: `CNN3Block-GAP` significantly outperforms the best classical model (SVM-RBF on HOG): $\Delta \text{acc} = +1.79\%$, $\chi^2 = 44.56, p = 2.46 \times 10^{-11} \ll 0.001$.
2. **Upper-Body Garment Bottleneck Resolution**:
   - **Pullover Recall**: Increased from 85.6% (SVM HOG) $\to$ **90.6%** (CNN3Block-GAP).
   - **Coat Recall**: Increased from 87.5% (SVM HOG) $\to$ **92.4%** (CNN3Block-GAP).
   - **Shirt Recall**: Improved from 75.0% (SVM HOG) $\to$ **76.3%** (CNN3Block-GAP).
   - **Total Misclassifications**: Reduced from 870 test errors (SVM) to only **691 test errors** (CNN) — eliminating 179 errors.
3. **Storage & Speed Pareto Frontier**: `CNN3Block-GAP` achieves a $60\times$ parameter reduction (305k vs. 18.1M) and an $8\times$ CPU inference speedup (0.68 ms vs. 5.36 ms) compared to the kernel SVM.
4. **Grad-CAM Interpretability**: Saliency heatmaps confirm that the CNN focuses on collar curvature, neckline depth, and hemline structure, distinguishing garments that fixed HOG grids fail to separate.

## Presentation Studio & Diagnostic Tools
The project includes a comprehensive, interactive Streamlit diagnostic and presentation studio (`app/streamlit_app.py`):
- **Design System & Typography**: Custom CSS theme with Google Fonts (`Outfit` & `Inter`), glassmorphic KPI cards, responsive layout, and dark/light mode tokens.
- **Interactive Multi-View Feature Studio (Live Demo)**: Real-time 4-viewport synchronized inspection:
  - Raw $28\times28$ Grayscale image and intensity statistics.
  - Sobel edge gradient magnitude map.
  - Live computed HOG orientation vector field.
- **Edge-Case Preset Gallery**: Quick-select buttons for famous ambiguous test images (e.g. *Shirt #14*, *Coat vs. Pullover #42*, *Sandal #7*, *Sneaker #23*).
- **Head-to-Head Model Comparator**: Real-time evaluation of any two models side-by-side: calculates accuracy delta, macro-F1 gain, parameter counts, and inference latency speedup ratios.
- **Interactive Confusion Pair Inspector**: Diagnostic visual tool allowing evaluators to select specific misclassification pairs (e.g. *Shirt $\to$ T-shirt*) to view the exact error instances.

## Dataset
Fashion-MNIST by Zalando Research — 70,000 grayscale 28×28 images across 10 clothing classes
(60k train / 10k test). Source: <https://github.com/zalandoresearch/fashion-mnist>.

## Folder structure

```
fashion-mnist-phase1/
├── data/raw/                 official Fashion-MNIST IDX .gz files (unchanged)
├── data/processed/           cached feature matrices (git-ignored)
├── src/
│   ├── config.py             paths, seeds, class names, final HOG parameters
│   ├── data_loader.py        IDX reader, [0,1] scaling, stratified train/val split
│   ├── eda.py                all EDA functions (quality, balance, PCA, t-SNE, ...)
│   ├── features.py           pixels, HOG, 16 shape statistics
│   ├── feature_experiments.py HOG ablation, PCA sweep, feature-set probe
│   ├── models.py             classical model factories, grids, parameter counting
│   ├── experiment.py         tune on val -> refit train+val -> single test evaluation
│   ├── analysis.py           confusion, McNemar, bootstrap CI, error analysis, plots
│   ├── plotting.py           consistent styling and palettes
│   ├── cnn_models.py         PyTorch CNNs (CNN2Block, CNN3BlockGAP)
│   ├── cnn_dataset.py        PyTorch Dataset & DataLoader with spatial augmentation
│   ├── cnn_trainer.py        AdamW + CosineAnnealing + EarlyStopping + profiling
│   └── cnn_explain.py        Grad-CAM saliency heatmaps for upper-body garments
├── scripts/                  00_check_splits.py · 01_eda.py · 02_feature_engineering.py · 03_train_models.py · 04_analysis.py · 05_train_cnn.py · 06_cnn_analysis.py · run_all.sh
├── notebooks/                01_eda · 02_feature_engineering · 03_models_and_error_analysis · 04_deep_learning_cnn
├── results/figures|tables|models   every figure (PNG), table (CSV/JSON), saved models (.joblib, .pt)
├── app/streamlit_app.py      interactive 8-page presentation studio & Grad-CAM visualizer
├── report/main.tex           comprehensive IEEE-format conference paper (+ figures/)
└── docs/                     presentation guide, viva Q&A, presentation slides
```

## How to run

Requires Python 3.12. Commands assume you are in the repository root.

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/streamlit run app/streamlit_app.py
```

The raw dataset, result tables, figures and demo models are included, so you can
launch the presentation without retraining. The live demo uses the saved pixel
and HOG models in `results/models/`.

To regenerate the experiment outputs and execute the notebooks:

```bash
bash scripts/run_all.sh                       # full pipeline (~25 min)
# Or select another environment (with requirements.txt installed):
PY="/path/to/environment/bin/python" bash scripts/run_all.sh
```

The runner uses `PY` for every step, including notebook conversion. Individual
steps can also be run with `.venv/bin/python scripts/01_eda.py` through
`scripts/04_analysis.py`, in order. Training reuses completed runs by default;
use `.venv/bin/python scripts/03_train_models.py --force` to retrain them.

Run the data-reader regression checks without training or downloading data:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

The IDX reader checks file headers, 28×28 image dimensions and payload lengths.
A `ValueError` identifying a raw file means its decompressed contents do not
match the expected IDX format; restore that file from the official dataset.

Build the report separately with `cd report && tectonic main.tex`, or upload
`report/` to Overleaf (pdflatex).

## Protocol (fair comparison)
1. Official train/test split untouched; 20 % stratified validation drawn **only from train**.
2. Hyper-parameters chosen on validation; refit on train+val; test evaluated once.
3. Same seed (42), scaling policy and grid budget for every feature set.
4. HOG computed on upright images (orientation verified in EDA).

