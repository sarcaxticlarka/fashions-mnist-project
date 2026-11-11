# Presentation guide (≈14 minutes, 8 Streamlit pages)

Start: `.venv/bin/streamlit run app/streamlit_app.py`. Keep the page order below; one speaker per 2 pages.

| min | Page | Say | Show | Speaker |
|---|---|---|---|---|
| 0–1.5 | 1 Problem & Pipeline | Catalogue tagging; gap: flattening loses neighbourhood structure, HOG fixes the representation first. Protocol: val from train only, test used once. | pipeline diagram, KPI cards | Teammate 1 |
| 1.5–4 | 2 EDA | Data is clean (0 missing/duplicates/leakage), balanced, 50 % zeros. Adjacent pixel corr 0.75 vs 0.16 shuffled → structure. Mean-image distances **predict** Pullover/Coat/Shirt/T-shirt confusion. PCA: 187 comps = 95 %. | Quality audit tab, Structure tab, t-SNE | Teammate 1 |
| 4–6 | 3 Feature Engineering | How HOG works (gradient → cell histogram → block normalisation). Ablation: smaller cells better but 8k dims; we chose 864-d under a 2,000-d budget. 16 shape stats. | **HOG explorer sliders**, ablation tab | Teammate 2 |
| 6–8.5 | 4 Models & Results | 12 configs. Best SVM+HOG 91.3 %. HOG helps LogReg +4.0, kNN +2.1, SVM +1.4; RF +0.2 (n.s.). McNemar + bootstrap CI. **Interactive Head-to-Head Comparator**. | leaderboard, latency plot, **Model Comparator** | **Sibtain** |
| 8.5–10.5 | 5 Error Analysis | 60 % of errors in upper-body group; Shirt recall 0.75. HOG fixes 142 errors, 104 in that group. Shirt details are 1-2 px at 28×28. **Interactive Confusion Pair Inspector**. | confusion matrices, **Confusion Inspector gallery** | Teammate 2 |
| 10.5–11.5 | 6 Live Demo & Studio | **Live Prediction & Multi-View Studio**: Test edge-case presets, live Sobel & HOG fields, real-time prediction cards, and probability distributions. | **Preset buttons, 4-view studio, prediction cards** | Teammate 4 |
| 11.5–12 | 7 Conclusions | Phase 1 limitations; transition to Phase 2 deep learning. | Conclusion summary | Teammate 1 |
| 12–14 | 8 Phase 2 (CNN) | **Deep Learning**: CNN3Block+GAP achieves 93.09 % (+1.79 pts over SVM, p < 10⁻¹⁰); 60× smaller (305k vs 18M params); 8× faster CPU latency (0.68 ms); reduces upper-body errors by 179. **Live Grad-CAM studio**. | **Unified leaderboard, Pareto frontier, Grad-CAM visualizer** | All / Team Lead |

## Rubric mapping
| Criterion | Where |
|---|---|
| Literature review | report §II |
| Dataset quality & EDA | notebooks/01, report §III, app page 2 |
| Feature engineering | notebooks/02, report §IV, app page 3 |
| Theoretical rigor | report §V, §VIII (loss functions, kernels, HOG maths, CNN GAP theory, metrics), McNemar/bootstrap |
| Model application (Classical & CNN) | notebooks/03, 04, report §VII, §VIII, `src/models.py`, `src/cnn_models.py`, `src/experiment.py` |
| Explainability & Diagnostics | app pages 6 & 8, report §VIII, `src/cnn_explain.py` (Grad-CAM) |
| GitHub repo & code quality | README, modular `src/`, scripts, fixed seeds |
| Presentation | Streamlit app (8 interactive pages) |
| Viva | docs/viva_qa.md |
| Report (IEEE LaTeX) | report/main.tex |
