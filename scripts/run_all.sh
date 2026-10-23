#!/usr/bin/env bash
# Regenerate every figure, table, model and notebook of Phase 1 (about 25 minutes on 8 CPU cores).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
$PY scripts/01_eda.py
$PY scripts/02_feature_engineering.py
$PY scripts/03_train_models.py       # resumable; add --force to retrain
$PY scripts/04_analysis.py
$PY scripts/build_notebooks.py
.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb
echo "Done. Presentation: streamlit run app/streamlit_app.py"
