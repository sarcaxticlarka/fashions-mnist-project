"""Interactive Phase-1 presentation: Fashion-MNIST handcrafted features vs. classical ML.

Run:  streamlit run app/streamlit_app.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
from skimage import exposure
from sklearn.metrics import confusion_matrix

from src.config import CLASS_NAMES, FIG_DIR, TABLE_DIR, RUN_DIR, MODEL_DIR, UPPER_BODY, HOG_PARAMS
from src.data_loader import make_splits
from src.features import hog_with_image, compute_hog, shape_stats, STAT_NAMES

st.set_page_config(page_title="Fashion-MNIST | Phase 1", page_icon="👕", layout="wide")

st.markdown("""<style>
.block-container {padding-top: 1.5rem; max-width: 1250px;}
.kpi {background:#f4f6fb;border-radius:12px;padding:14px 18px;border-left:5px solid #4C72B0;}
.kpi h2 {margin:0;font-size:1.9rem;color:#1f2d4d;} .kpi p {margin:0;color:#5b6577;font-size:.85rem;}
@media (prefers-color-scheme: dark){.kpi{background:#1e2330;} .kpi h2{color:#e8ecf5;} .kpi p{color:#a5afc4;}}
</style>""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------- cached data
@st.cache_data(show_spinner="Loading dataset ...")
def get_splits():
    S = make_splits()
    return S


@st.cache_data
def get_results():
    df = pd.read_csv(TABLE_DIR / "model_results.csv")
    runs = {}
    for f in RUN_DIR.glob("*__*.json"):
        r = json.loads(f.read_text())
        r["pred"] = np.load(RUN_DIR / f"{r['model']}__{r['features']}__pred.npy")
        runs[(r["model"], r["features"])] = r
    return df, runs


@st.cache_resource
def get_model(name):
    p = MODEL_DIR / f"{name}.joblib"
    return joblib.load(p) if p.exists() else None


def csv(name):
    p = TABLE_DIR / name
    return pd.read_csv(p) if p.exists() else None


def fig_(name, caption=None):
    p = FIG_DIR / f"{name}.png"
    if p.exists():
        st.image(str(p), caption=caption, use_container_width=True)


def kpi(col, value, label):
    col.markdown(f'<div class="kpi"><h2>{value}</h2><p>{label}</p></div>', unsafe_allow_html=True)


# ----------------------------------------------------------------------------- sidebar
PAGES = ["1 · Problem & Pipeline", "2 · Dataset Quality & EDA", "3 · Feature Engineering",
         "4 · Models & Results", "5 · Error Analysis", "6 · Live Demo", "7 · Conclusions"]
page = st.sidebar.radio("Navigate", PAGES)
st.sidebar.markdown("---")
st.sidebar.caption("Project 11 · Automatic Clothing Category Recognition  \nPhase 1: Machine-Learning evaluation  \nHandcrafted features (HOG) vs. raw pixels")

S = get_splits()

# ============================================================================ 1
if page == PAGES[0]:
    st.title("Automatic Clothing Category Recognition")
    st.subheader("Phase 1 — do fixed image descriptors beat raw pixels for classical ML?")
    c = st.columns(4)
    kpi(c[0], "70,000", "28×28 grayscale images")
    kpi(c[1], "10", "balanced clothing classes")
    kpi(c[2], "4 × 3", "models × feature sets")
    kpi(c[3], "48k / 12k / 10k", "train / validation / test")
    st.markdown("### Research question")
    st.info("A small clothing catalogue needs automatic tagging. **Flattened pixels discard neighbourhood structure; "
            "HOG fixes its representation before the model sees any category-specific example.** "
            "How much does each choice cost in accuracy, macro-F1, per-class recall, model size and inference time?")
    st.markdown("### Pipeline")
    st.graphviz_chart("""digraph G { rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#eef2fb", fontname="Helvetica"];
      A [label="Official IDX files\\n60k train / 10k test"]; B [label="Scale to [0,1]\\nstratified 80/20\\ntrain -> train/val"];
      C [label="EDA & data quality"]; D [label="Feature sets\\npixels | HOG | HOG+stats"];
      E [label="LogReg · kNN\\nRandom Forest · SVM-RBF"]; F [label="Tune on VAL only\\nrefit on train+val"]; G [label="Single TEST evaluation\\nacc · macro-F1 · recall\\nparams · latency"]; H [label="Error analysis\\nMcNemar · bootstrap CI"];
      A->B->C->D->E->F->G->H; }""")
    st.markdown("""### Fair-comparison protocol
- Official train/test split is never altered. The validation set comes **only from the training data** (stratified).
- Hyper-parameters are chosen on validation; the test set is evaluated **once** per model.
- The *same* scaling, grids and seed are used for every feature set.
- HOG is computed on upright images (row 0 = top of the garment, as stored in the IDX files).""")

# ============================================================================ 2
elif page == PAGES[1]:
    st.title("Dataset Quality & Exploratory Data Analysis")
    t = st.tabs(["Quality audit", "Classes & samples", "Pixel statistics", "Structure & separability", "Embeddings"])
    with t[0]:
        q = csv("eda_quality_report.csv")
        if q is not None:
            st.dataframe(q, hide_index=True, use_container_width=True)
        st.success("No missing values, no blank images, no duplicates and zero train/test overlap -> no cleaning needed. "
                   "Only preprocessing: scale pixels to [0,1].")
        fig_("eda_11_train_test_shift", "Train and test come from the same distribution (KS test, p > 0.05).")
    with t[1]:
        d = csv("eda_class_distribution.csv")
        a, b = st.columns([1, 1])
        with a:
            fig_("eda_01_class_distribution")
        with b:
            st.dataframe(d, hide_index=True, use_container_width=True)
        st.markdown("#### Browse the data")
        cls = st.selectbox("Class", CLASS_NAMES, index=6)
        seed = st.slider("Random draw", 0, 50, 0)
        ids = np.random.default_rng(seed).choice(np.where(S.y_tr == CLASS_NAMES.index(cls))[0], 12, replace=False)
        cols = st.columns(12)
        for col, i in zip(cols, ids):
            col.image(np.clip(S.X_tr[i], 0, 1), use_container_width=True)
        fig_("eda_12_ambiguous_group")
    with t[2]:
        fig_("eda_03_mean_std_images")
        fig_("eda_04_intensity_shape_boxplots")
        fig_("eda_05_pixel_histogram")
        fig_("eda_06_pixel_variance_fscore")
        st.caption("Half of all pixels are exactly 0 (sparse data) and the image border carries almost no class information.")
    with t[3]:
        a, b = st.columns(2)
        with a:
            fig_("eda_07_spatial_correlation", "Neighbouring pixels are strongly correlated; flattening throws this layout away.")
        with b:
            fig_("eda_08_class_mean_distance", "Smallest distances predict the confusions: Pullover/Coat/Shirt/T-shirt.")
        pairs = csv("eda_class_pair_distance.csv")
        st.markdown("**Most similar class pairs (distance between mean images)**")
        st.dataframe(pairs.head(8), hide_index=True)
    with t[4]:
        a, b = st.columns(2)
        with a:
            fig_("eda_09_pca")
        with b:
            tsne_p = TABLE_DIR / "tsne_coords.npy"
            if tsne_p.exists():
                z = np.load(tsne_p)
                dfz = pd.DataFrame({"x": z[:, 0], "y": z[:, 1], "class": [CLASS_NAMES[int(i)] for i in z[:, 2]]})
                st.plotly_chart(px.scatter(dfz, x="x", y="y", color="class", opacity=.7, title="t-SNE (interactive)", height=480)
                                .update_traces(marker=dict(size=4)), use_container_width=True)
        pc = csv("eda_pca_components.csv")
        if pc is not None:
            st.dataframe(pc, hide_index=True)

# ============================================================================ 3
elif page == PAGES[2]:
    st.title("Feature Engineering")
    t = st.tabs(["HOG explorer", "Ablation", "Hand-crafted statistics", "Feature-set comparison"])
    with t[0]:
        st.markdown("Change the descriptor parameters and watch the gradient-orientation map and dimensionality change.")
        c1, c2, c3, c4 = st.columns(4)
        cls = c1.selectbox("Class", CLASS_NAMES, index=6, key="hogc")
        idx = c2.slider("Image #", 0, 200, 3)
        ori = c3.select_slider("Orientations", [6, 9, 12], value=HOG_PARAMS["orientations"])
        ppc = c4.select_slider("Cell size (px)", [2, 4, 7], value=HOG_PARAMS["pixels_per_cell"][0])
        i = np.where(S.y_tr == CLASS_NAMES.index(cls))[0][idx]
        vec, himg = hog_with_image(S.X_tr[i], orientations=ori, pixels_per_cell=(ppc, ppc))
        a, b, c = st.columns(3)
        a.image(S.X_tr[i], caption="input (28×28)", use_container_width=True, clamp=True)
        b.image(exposure.rescale_intensity(himg, in_range=(0, himg.max() + 1e-9)), caption="HOG orientation map", use_container_width=True, clamp=True)
        with c:
            st.metric("Descriptor length", f"{vec.size:,}")
            st.metric("vs. 784 raw pixels", f"{vec.size / 784:.2f}×")
            st.caption("Block norm L2-Hys, 2×2 cells per block. Cells vote with gradient magnitude into orientation bins; blocks normalise for contrast.")
        fig_("fe_02_hog_visual")
    with t[1]:
        ab = csv("fe_hog_ablation.csv")
        if ab is not None:
            fig_("fe_01_hog_ablation")
            st.dataframe(ab, hide_index=True, use_container_width=True)
            st.info(f"Chosen under a ≤ 2,000-dimension budget: **{HOG_PARAMS['orientations']} orientations, "
                    f"{HOG_PARAMS['pixels_per_cell'][0]}×{HOG_PARAMS['pixels_per_cell'][1]} cells, 2×2 blocks** (864-d). "
                    "Finer cells score higher but give 4k–8k dimensions, which make kNN/SVM/RF much slower.")
    with t[2]:
        a, b = st.columns([3, 2])
        with a:
            fig_("fe_03_shape_stats_profile")
        with b:
            fig_("fe_04_shape_stats_importance")
        st.caption("16 interpretable global descriptors (bounding box, symmetry, centroid, edge density, ...). Combined with HOG they form `hog_stats`.")
    with t[3]:
        a, b = st.columns(2)
        with a:
            fig_("fe_06_feature_set_probe")
        with b:
            fig_("fe_05_pca_sweep")
        st.dataframe(csv("fe_feature_set_probe.csv"), hide_index=True)

# ============================================================================ 4
elif page == PAGES[3]:
    st.title("Model Application & Results")
    df, runs = get_results()
    best = df.sort_values("accuracy", ascending=False).iloc[0]
    c = st.columns(4)
    kpi(c[0], f"{best.accuracy:.2%}", f"best test accuracy ({best.model} + {best.features})")
    kpi(c[1], f"{best.macro_f1:.3f}", "its macro-F1")
    pix, hog = df[df.features == "pixels"].accuracy.max(), df[df.features == "hog"].accuracy.max()
    kpi(c[2], f"{hog - pix:+.2%}", "best HOG vs best pixels")
    kpi(c[3], f"{len(df)}", "trained configurations")
    flt = st.multiselect("Feature sets", ["pixels", "hog", "hog_stats"], ["pixels", "hog", "hog_stats"])
    show = df[df.features.isin(flt)].sort_values("accuracy", ascending=False)
    st.dataframe(show[["model", "features", "dim", "val_acc", "accuracy", "macro_f1", "fit_seconds", "infer_ms_per_image", "feat_ms_per_image", "params", "best_params"]]
                 .style.format({"val_acc": "{:.4f}", "accuracy": "{:.4f}", "macro_f1": "{:.4f}", "infer_ms_per_image": "{:.4f}", "feat_ms_per_image": "{:.3f}", "params": "{:,.0f}"})
                 .background_gradient(subset=["accuracy", "macro_f1"], cmap="Greens"), hide_index=True, use_container_width=True)
    fig_("res_01_accuracy_f1_overview")
    a, b = st.columns(2)
    with a:
        fig_("res_07_accuracy_vs_latency")
    with b:
        fig_("res_08_learning_curve")
    st.markdown("#### Per-class recall")
    fig_("res_02_per_class_recall")
    st.markdown("#### Is the HOG gain statistically real?  (McNemar, same test images)")
    mc = csv("mcnemar_pixels_vs_hog.csv")
    if mc is not None:
        st.dataframe(mc, hide_index=True, use_container_width=True)
    ci = csv("bootstrap_ci.csv")
    if ci is not None:
        with st.expander("95 % bootstrap confidence intervals"):
            st.dataframe(ci, hide_index=True)
    with st.expander("Hyper-parameter search logs (validation set)"):
        key = st.selectbox("Run", [f"{m} | {f}" for (m, f) in sorted(runs)])
        m, f = key.split(" | ")
        st.dataframe(pd.DataFrame(runs[(m, f)]["search"]), hide_index=True)

# ============================================================================ 5
elif page == PAGES[4]:
    st.title("Error Analysis")
    df, runs = get_results()
    keys = sorted(runs)
    a, b = st.columns(2)
    ka = a.selectbox("Model A", [f"{m} | {f}" for m, f in keys], index=[k for k in keys].index(max([k for k in keys if k[1] == "pixels"], key=lambda k: runs[k]["accuracy"])))
    kb = b.selectbox("Model B", [f"{m} | {f}" for m, f in keys], index=[k for k in keys].index(max([k for k in keys if k[1] == "hog"], key=lambda k: runs[k]["accuracy"])))
    ra, rb = runs[tuple(ka.split(" | "))], runs[tuple(kb.split(" | "))]
    norm = st.toggle("Normalise rows (recall)", False)
    cols = st.columns(2)
    for col, r, k in zip(cols, (ra, rb), (ka, kb)):
        cm = confusion_matrix(S.y_test, r["pred"], normalize="true" if norm else None)
        fig = px.imshow(cm, x=CLASS_NAMES, y=CLASS_NAMES, text_auto=".2f" if norm else "d", color_continuous_scale="Blues",
                        title=f"{k}  (acc {r['accuracy']:.3f})", aspect="auto", height=520)
        fig.update_xaxes(title="predicted", tickangle=45); fig.update_yaxes(title="true")
        col.plotly_chart(fig, use_container_width=True)
    st.markdown("#### Where did Model B fix (or break) Model A?")
    fig_("res_04_confusion_difference")
    tc = csv("top_confusions_all.csv")
    st.markdown("#### Largest confusions")
    if tc is not None:
        mm, ff = ka.split(" | ")
        st.dataframe(tc[(tc.model == mm) & (tc.features == ff)].head(8), hide_index=True)
    ub = csv("upper_body_error_share.csv")
    if ub is not None:
        c1, c2 = st.columns([2, 3])
        c1.markdown("**Share of errors inside T-shirt / Pullover / Coat / Shirt**")
        c1.dataframe(ub, hide_index=True)
        with c2:
            fig_("res_05_upper_body_error_share")
    st.markdown("#### Misclassified examples (upper-body garments)")
    sel = st.radio("Model", [ka, kb], horizontal=True)
    r = runs[tuple(sel.split(" | "))]
    wrong = np.where(r["pred"] != S.y_test)[0]
    ub_idx = [CLASS_NAMES.index(c) for c in UPPER_BODY]
    wrong = np.array([i for i in wrong if S.y_test[i] in ub_idx and r["pred"][i] in ub_idx])
    seed = st.slider("Shuffle", 0, 30, 0)
    pick = np.random.default_rng(seed).choice(wrong, min(12, len(wrong)), replace=False)
    cs = st.columns(6)
    for n, i in enumerate(pick):
        cs[n % 6].image(S.X_test[i], caption=f"T: {CLASS_NAMES[S.y_test[i]]}\nP: {CLASS_NAMES[r['pred'][i]]}", use_container_width=True, clamp=True)

# ============================================================================ 6
elif page == PAGES[5]:
    st.title("Live Demo")
    st.caption("Classify a test-set image, or upload your own garment photo (it is converted to a 28×28 grayscale image).")
    models = {n: get_model(n) for n in ("LogReg__hog", "SVM-RBF__hog", "LogReg__pixels", "SVM-RBF__pixels")}
    models = {k: v for k, v in models.items() if v is not None}
    src = st.radio("Source", ["Test-set image", "Upload"], horizontal=True)
    img = truth = None
    if src == "Test-set image":
        i = st.slider("Test image index", 0, 9999, 42)
        img, truth = S.X_test[i], CLASS_NAMES[S.y_test[i]]
    else:
        up = st.file_uploader("Garment image", type=["png", "jpg", "jpeg"])
        inv = st.checkbox("Invert (use if your photo is dark garment on white background)", True)
        if up:
            im = Image.open(up).convert("L").resize((28, 28))
            img = np.asarray(im, dtype=np.float32) / 255
            img = 1 - img if inv else img
    if img is not None:
        a, b = st.columns([1, 3])
        a.image(img, width=200, clamp=True, caption=f"true: {truth}" if truth else "uploaded")
        rows = []
        for name, m in models.items():
            mdl, feat = name.split("__")
            x = img[None]
            f = x.reshape(1, -1) if feat == "pixels" else compute_hog(x, n_jobs=1)
            pred = int(m.predict(f)[0])
            rows.append({"model": mdl, "features": feat, "prediction": CLASS_NAMES[pred], "correct": (CLASS_NAMES[pred] == truth) if truth else None})
        b.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        if hasattr(models.get("LogReg__hog"), "predict_proba"):
            p = models["LogReg__hog"].predict_proba(compute_hog(img[None], n_jobs=1))[0]
            b.plotly_chart(px.bar(x=CLASS_NAMES, y=p, labels={"x": "", "y": "probability"}, title="Logistic Regression + HOG: class probabilities", height=300), use_container_width=True)

# ============================================================================ 7
else:
    st.title("Conclusions & Next Steps")
    df, runs = get_results()
    pix = df[df.features == "pixels"].sort_values("accuracy", ascending=False).iloc[0]
    hog = df[df.features == "hog"].sort_values("accuracy", ascending=False).iloc[0]
    st.markdown(f"""
### What we found
1. **Best raw-pixel model:** {pix.model} — accuracy {pix.accuracy:.3f}, macro-F1 {pix.macro_f1:.3f}.
2. **Best HOG model:** {hog.model} — accuracy {hog.accuracy:.3f}, macro-F1 {hog.macro_f1:.3f}.
3. HOG improves **three of four models, significantly (McNemar p < 0.001)**: Logistic Regression +4.0 pts, kNN +2.1 pts, SVM-RBF +1.4 pts. **Random Forest gains nothing** (+0.2 pts, p = 0.49): trees already pick informative pixel thresholds.
4. The linear model gains most because HOG supplies the non-linear, neighbourhood-aware features it cannot learn; the RBF kernel already recovers part of that from raw pixels.
5. Remaining errors are concentrated in the **upper-body group** (Shirt, T-shirt, Pullover, Coat): ~60-65 % of all errors for every model. Shirt recall is only 0.75 for the best model; Trouser, Bag, Sandal, Sneaker, Ankle boot exceed 0.96.
6. Cost: HOG adds ~0.15 ms/image of extraction; SVM-RBF is ~4,000× slower at inference than Logistic Regression for +2.8 pts.

### Limitations
- 28×28 grayscale loses texture/buttons/collars, which are exactly what distinguishes Shirt vs T-shirt.
- HOG parameters are fixed a priori; they were tuned on a small probe model, not per classifier.
- Single train/validation split (bootstrap CI reported for test, but no repeated CV).

### Phase 2 (Deep Learning)
Two- or three-block CNN with pooling, dropout and early stopping on the **same splits**, compared with the table above using the same metrics, parameter counts and latency.
""")
