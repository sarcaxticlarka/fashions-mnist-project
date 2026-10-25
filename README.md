# Automatic Clothing Category Recognition — Phase 1 (Machine Learning)

**Project 11 · Inventory image tagging via handcrafted features and a small CNN.**
This repository is the **Phase 1 (classical ML)** deliverable: raw pixels vs. HOG descriptors with
Logistic Regression, kNN, Random Forest and RBF-SVM on Fashion-MNIST. (Phase 2 = CNN, not included.)

## Headline results (test set, 10,000 images)

| Model | Pixels | HOG | HOG + stats |
|---|---|---|---|
| Logistic Regression | 84.56 | 88.52 | 88.94 |
| kNN | 85.61 | 87.74 | 87.76 |
| Random Forest | 87.60 | 87.81 | 88.21 |
| **SVM (RBF)** | 89.88 | **91.30** | **91.37** |

* HOG gains are significant (McNemar p < 0.001) for LogReg (+4.0), kNN (+2.1), SVM (+1.4); not for Random Forest (+0.2, p = 0.49).
* 58–65 % of all errors of every model are inside {T-shirt, Pullover, Coat, Shirt}; Shirt recall is 0.75 at best.

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
│   ├── models.py             model factories, hyper-parameter grids, parameter counting
│   ├── experiment.py         tune on val -> refit train+val -> single test evaluation
│   ├── analysis.py           confusion, McNemar, bootstrap CI, error analysis, plots
│   └── plotting.py
├── scripts/                  00_check_splits.py · 01_eda.py · 02_feature_engineering.py · 03_train_models.py · 04_analysis.py · run_all.sh
├── notebooks/                01_eda · 02_feature_engineering · 03_models_and_error_analysis (executed)
├── results/figures|tables|models   every figure (PNG), table (CSV/JSON), saved models
├── app/streamlit_app.py      interactive presentation studio & live demo
├── report/main.tex, main.pdf IEEE-format report (+ figures/)
└── docs/                     presentation script, viva Q&A, glossary
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

