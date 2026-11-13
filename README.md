# Automatic Clothing Category Recognition — Phase 1 & Phase 2 (final)

**Project 11 · Inventory image tagging via handcrafted features and Convolutional Neural Networks.**
This repository contains the complete deliverables for:
- **Phase 1 (Classical Machine Learning):** Raw pixels vs. HOG descriptors evaluated across Logistic Regression, $k$-NN, Random Forest, and RBF Support Vector Machines.
- **Phase 2 (Deep Learning, final):** CNNs on raw pixels — a controlled 2×2 ablation (depth/GAP × augmentation), 40-epoch training with 3 seeds, a residual network (ResNetSmall), an ensemble with test-time augmentation, temperature-scaled confidence, an abstain option and a Shirt-aware decision rule.

## Headline Results (test set, 10,000 images)

| Model | Test acc | Macro-F1 | Errors | Shirt recall | Notes |
|---|---|---|---|---|---|
| SVM-RBF on HOG (Phase 1 best) | 91.30% | 0.9130 | 870 | 75.0% | 18.1M stored numbers, 5.36 ms/img |
| CNN3Block-GAP (first Phase 2 run, 20 epochs) | 93.09% | 0.9306 | 691 | 76.3% | 305k weights, 0.68 ms/img |
| CNN3Block-GAP, 40 epochs (3 seeds) | 93.80% ± 0.21 | – | – | 79.1% | under-training fixed |
| ResNetSmall (3 seeds) | 94.62% ± 0.13 | – | – | 82.7% | 696k weights |
| **Final model: 3 × ResNetSmall + flip TTA + calibration** | **95.20%** (CI 94.78–95.62) | **0.9520** | **480** | **85.4%** | ECE 7.6% → 1.3%; 98.56% accurate on the 87.7% of images it answers |

Key findings:
1. **Learned features beat handcrafted ones.** Final model vs SVM: McNemar χ² = 220.6, p = 6.7×10⁻⁵⁰; vs the first CNN: p = 2.9×10⁻²⁷.
2. **Depth + GAP only help together with augmentation** (2×2 ablation: 92.16 / 91.41 / 91.38 / 93.09%).
3. **The first runs were under-trained**: the same GAP network reaches 93.80% on average with 40 epochs.
4. **Shirt remains the hardest class** (85.4%) but gained +10.4 points over Phase 1; low-confidence images (26% of them Shirts) can be sent to a human.

All Phase 2 numbers are regenerated from `results/tables` by `scripts/08`–`11`; the report (`report/phase2_full_report.pdf`), the 16-slide deck (`docs/Fashion-MNIST_Phase2_Presentation.pptx`, rebuilt with `node scripts/build_phase2_deck.js`; `FULL_DECK=1` adds 8 detail slides) and the app read the same files.

## Streamlit studio (`app/streamlit_app.py`)
Nine pages: overview, **live model comparison** (any test image or an uploaded photo; SVM vs first CNN vs final model with calibrated probabilities, abstain flag and Grad-CAM), feature views, Phase 1, Phase 2 models & training curves, experiments (2×2 ablation, seeds, improvement steps, McNemar), errors & reliability (per-class recall, confusion heat-map, Shirt rule, calibration, interactive abstain threshold), design decisions, and leaderboard with downloads of the report and slides. Every chart labels its axes in plain words.

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
└── docs/                     presentation guide (speaker plan + Q&A), Phase 1 & Phase 2 slides
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

