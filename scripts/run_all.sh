#!/usr/bin/env bash
# Regenerate every figure, table, model and notebook of Phase 1 and Phase 2.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
"$PY" scripts/01_eda.py
"$PY" scripts/02_feature_engineering.py
"$PY" scripts/03_train_models.py       # resumable; add --force to retrain
"$PY" scripts/04_analysis.py
"$PY" scripts/05_train_cnn.py          # Phase 2 CNN models; add --force to retrain
"$PY" scripts/06_cnn_analysis.py       # Phase 2 post-hoc analysis & comparison
"$PY" scripts/07_ablation_gap_noaug.py # 2x2 ablation control run
"$PY" scripts/08_phase2_story_figures.py   # CIs, confusion, figures p2_01-p2_09
"$PY" scripts/09_train_final_models.py # 40-epoch, 3-seed CNN3Block-GAP + ResNetSmall (~2 h on Apple MPS)
"$PY" scripts/10_ensemble_calibration.py   # final ensemble, TTA, calibration, abstain, Shirt rule
"$PY" scripts/11_report_numbers.py     # number macros for report/phase2_full_report.tex
"$PY" scripts/build_notebooks.py
"$PY" -m nbconvert --to notebook --execute --inplace notebooks/*.ipynb
echo "Done. Presentation: streamlit run app/streamlit_app.py"
