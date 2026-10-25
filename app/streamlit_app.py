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
from scipy import ndimage as ndi
from skimage import exposure
from sklearn.metrics import confusion_matrix

from src.config import CLASS_NAMES, FIG_DIR, TABLE_DIR, RUN_DIR, MODEL_DIR, UPPER_BODY, HOG_PARAMS
from src.data_loader import make_splits
from src.features import hog_with_image, compute_hog, shape_stats, STAT_NAMES

st.set_page_config(
    page_title="Fashion-MNIST · Classical ML Studio",
    page_icon="👕",
    layout="wide",
    initial_sidebar_state="expanded"
)

CLASS_EMOJIS = {
    "T-shirt/top": "👕",
    "Trouser": "👖",
    "Pullover": "🧥",
    "Dress": "👗",
    "Coat": "🥼",
    "Sandal": "👡",
    "Shirt": "👔",
    "Sneaker": "👟",
    "Bag": "👜",
    "Ankle boot": "👢",
}

# ----------------------------------------------------------------------------- Custom CSS Theme System
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

h1, h2, h3, h4 {
    font-family: 'Outfit', sans-serif;
    font-weight: 600;
    letter-spacing: -0.02em;
}

.block-container {
    padding-top: 1.8rem;
    padding-bottom: 3rem;
    max-width: 1280px;
}

/* Glassmorphic KPI Cards */
.kpi-card {
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.95), rgba(246, 248, 253, 0.85));
    border: 1px solid rgba(218, 225, 238, 0.85);
    border-radius: 14px;
    padding: 16px 20px;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.03);
    border-left: 5px solid #4C72B0;
    transition: all 0.25s ease-in-out;
}
.kpi-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 22px rgba(76, 114, 176, 0.12);
}
.kpi-card h2 {
    margin: 0;
    font-size: 1.95rem;
    color: #1a253c;
    font-weight: 700;
}
.kpi-card p {
    margin: 3px 0 0 0;
    color: #5d677d;
    font-size: 0.86rem;
    font-weight: 500;
}

/* Dark mode KPI */
@media (prefers-color-scheme: dark) {
    .kpi-card {
        background: linear-gradient(135deg, rgba(30, 37, 51, 0.95), rgba(22, 27, 39, 0.85));
        border-color: rgba(255, 255, 255, 0.08);
        box-shadow: 0 4px 18px rgba(0, 0, 0, 0.25);
    }
    .kpi-card h2 { color: #f1f5f9; }
    .kpi-card p { color: #94a3b8; }
}

/* Hero Banner */
.hero-header {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    color: #ffffff;
    border-radius: 16px;
    padding: 24px 28px;
    margin-bottom: 24px;
    box-shadow: 0 10px 25px rgba(15, 23, 42, 0.12);
    border: 1px solid rgba(255, 255, 255, 0.08);
}
.hero-header h1 {
    color: #ffffff;
    margin: 0 0 6px 0;
    font-size: 2.1rem;
}
.hero-header p {
    color: #94a3b8;
    margin: 0;
    font-size: 1.0rem;
}

/* Badge tags */
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 999px;
    font-size: 0.76rem;
    font-weight: 600;
    margin-right: 6px;
}
.badge-blue { background: #e0e7ff; color: #3730a3; }
.badge-green { background: #dcfce7; color: #166534; }
.badge-purple { background: #f3e8ff; color: #6b21a8; }
.badge-amber { background: #fef3c7; color: #92400e; }

/* Prediction Cards */
.pred-card {
    background: rgba(255, 255, 255, 0.85);
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 14px 16px;
    margin-bottom: 10px;
    transition: all 0.2s ease;
}
.pred-card:hover {
    border-color: #cbd5e1;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.04);
}
@media (prefers-color-scheme: dark) {
    .pred-card {
        background: #1e293b;
        border-color: #334155;
    }
}
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------- Cached Data Loaders
@st.cache_data(show_spinner="Loading Fashion-MNIST dataset...")
def get_splits():
    return make_splits()


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
def _load_model(name):
    return joblib.load(MODEL_DIR / f"{name}.joblib")


def get_model(name):
    return _load_model(name) if (MODEL_DIR / f"{name}.joblib").exists() else None


def csv(name):
    p = TABLE_DIR / name
    return pd.read_csv(p) if p.exists() else None


def fig_(name, caption=None):
    p = FIG_DIR / f"{name}.png"
    if p.exists():
        st.image(str(p), caption=caption, use_container_width=True)


def kpi(col, value, label, accent="#4C72B0"):
    col.markdown(
        f'<div class="kpi-card" style="border-left-color:{accent};">'
        f'<h2>{value}</h2><p>{label}</p></div>',
        unsafe_allow_html=True
    )


# ----------------------------------------------------------------------------- Sidebar Navigation
PAGES = [
    "1 · Problem & Pipeline",
    "2 · Dataset Quality & EDA",
    "3 · Feature Engineering",
    "4 · Models & Results",
    "5 · Error Analysis",
    "6 · Live Demo & Studio",
    "7 · Conclusions"
]

st.sidebar.markdown("""
<div style="display:flex; align-items:center; gap:10px; margin-bottom:12px;">
  <span style="font-size:2rem;">👕</span>
  <div>
    <h3 style="margin:0; font-size:1.15rem;">Fashion-MNIST</h3>
    <p style="margin:0; font-size:0.75rem; color:#888;">Classical ML Benchmark</p>
  </div>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio("Navigate Studio", PAGES)

st.sidebar.markdown("---")
st.sidebar.caption("**Project 11 · Phase 1 Deliverable**\nRaw Pixels vs. HOG Descriptors with Classical Classifiers.")

S = get_splits()


# ============================================================================ PAGE 1
if page == PAGES[0]:
    st.markdown("""
    <div class="hero-header">
      <h1>Automatic Clothing Category Recognition</h1>
      <p>Phase 1 · Comparative Machine Learning Evaluation: Handcrafted Features vs. Raw Pixel Baselines</p>
    </div>
    """, unsafe_allow_html=True)

    c = st.columns(4)
    kpi(c[0], "70,000", "28×28 Grayscale Images", "#4C72B0")
    kpi(c[1], "10", "Balanced Category Classes", "#55A868")
    kpi(c[2], "4 × 3", "Models × Feature Sets", "#C44E52")
    kpi(c[3], "48k / 12k / 10k", "Train / Val / Test Split", "#8172B3")

    st.markdown("### 🎯 Research Question")
    st.info(
        "A small clothing catalogue requires automated cataloguing and inventory tagging. "
        "**Flattened pixels completely discard 2D spatial neighbourhood structure; HOG fixes its representation "
        "before the model sees any category-specific training example.** "
        "How much does each representation cost in accuracy, macro-F1, per-class recall, memory footprint, and real-time latency?"
    )

    st.markdown("### 🔄 End-to-End ML Pipeline")
    st.graphviz_chart("""digraph G {
      rankdir=LR;
      node [shape=box, style="rounded,filled", fillcolor="#f0f4fc", fontname="Inter", fontsize=10];
      A [label="Official IDX Files\\n60k train / 10k test"];
      B [label="Scale to [0,1]\\nStratified 80/20\\ntrain -> train/val"];
      C [label="EDA & Data Audit\\nZero Leakage"];
      D [label="Feature Extraction\\npixels | HOG | HOG+stats"];
      E [label="Four Learners\\nLogReg · kNN\\nRandomForest · SVM"];
      F [label="Hyperparameter Tune\\non Validation Split only"];
      G [label="Final Test Eval\\nAccuracy · Macro-F1\\nLatency · Parameters"];
      H [label="Diagnostic Analysis\\nMcNemar · Bootstrap CI"];
      A->B->C->D->E->F->G->H;
    }""")

    st.markdown("### ⚖️ Fair-Comparison Protocol")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        - **Untouched Test Split**: The official 10,000 test set is strictly evaluated **once** for the final report.
        - **Leakage-Free Validation**: The 20% validation split is drawn exclusively from the training partition.
        """)
    with col2:
        st.markdown("""
        - **Consistent Seeds & Scaling**: All feature sets and models share fixed seed `42` and identical grid budgets.
        - **Upright Image Alignment**: Verified in EDA that row 0 corresponds to the garment top.
        """)


# ============================================================================ PAGE 2
elif page == PAGES[1]:
    st.title("Dataset Quality & Exploratory Data Analysis")
    st.caption("Comprehensive data audit, spatial correlation, and geometric separability.")
    t = st.tabs(["Quality Audit", "Classes & Samples", "Pixel Statistics", "Structure & Separability", "Embeddings"])

    with t[0]:
        q = csv("eda_quality_report.csv")
        if q is not None:
            st.dataframe(q, hide_index=True, use_container_width=True)
        st.success("Audit complete: No missing values, no blank images, zero duplicates, and zero train/test leakage. Only scaling [0, 1] required.")
        fig_("eda_11_train_test_shift", "Train and test distributions verified identical via Kolmogorov-Smirnov test (p > 0.05).")

    with t[1]:
        d = csv("eda_class_distribution.csv")
        a, b = st.columns([1, 1])
        with a:
            fig_("eda_01_class_distribution")
        with b:
            st.dataframe(d, hide_index=True, use_container_width=True)
        st.markdown("#### Browse Category Samples")
        cls = st.selectbox("Category", CLASS_NAMES, index=6)
        seed = st.slider("Random Draw Seed", 0, 50, 0)
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
        st.caption("Finding: ~50% of all image pixels are exactly 0 (sparsity), and background borders contain zero class variance.")

    with t[3]:
        a, b = st.columns(2)
        with a:
            fig_("eda_07_spatial_correlation", "Adjacent pixels exhibit strong correlation (r=0.75), which 1D flattening discards.")
        with b:
            fig_("eda_08_class_mean_distance", "Euclidean distance between mean class images accurately predicts confusion clusters.")
        pairs = csv("eda_class_pair_distance.csv")
        st.markdown("**Closest Class Pairs in Pixel Space**")
        if pairs is not None:
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
                st.plotly_chart(
                    px.scatter(dfz, x="x", y="y", color="class", opacity=0.75, title="Interactive t-SNE Embedding", height=480)
                    .update_traces(marker=dict(size=4)),
                    use_container_width=True
                )
        pc = csv("eda_pca_components.csv")
        if pc is not None:
            st.dataframe(pc, hide_index=True)


# ============================================================================ PAGE 3
elif page == PAGES[2]:
    st.title("Feature Engineering & Representations")
    st.caption("Ablation study, HOG gradient visualizer, and handcrafted shape statistics.")
    t = st.tabs(["Interactive HOG Explorer", "Ablation Study", "Handcrafted Shape Statistics", "Feature Probes"])

    with t[0]:
        st.markdown("Modify gradient parameters live to inspect the orientation vector map and dimensionality.")
        c1, c2, c3, c4 = st.columns(4)
        cls = c1.selectbox("Class", CLASS_NAMES, index=6, key="hogc")
        idx = c2.slider("Image Index", 0, 100, 3)
        ori = c3.select_slider("Orientation Bins", [6, 9, 12], value=HOG_PARAMS["orientations"])
        ppc = c4.select_slider("Cell Size (px)", [2, 4, 7], value=HOG_PARAMS["pixels_per_cell"][0])
        i = np.where(S.y_tr == CLASS_NAMES.index(cls))[0][idx]
        vec, himg = hog_with_image(S.X_tr[i], orientations=ori, pixels_per_cell=(ppc, ppc))
        
        a, b, c = st.columns(3)
        a.image(S.X_tr[i], caption="Raw Grayscale (28×28)", use_container_width=True, clamp=True)
        b.image(exposure.rescale_intensity(himg, in_range=(0, himg.max() + 1e-9)), caption="HOG Orientation Field", use_container_width=True, clamp=True)
        with c:
            st.metric("Feature Dimension", f"{vec.size:,} dims")
            st.metric("Ratio vs. 784 Pixels", f"{vec.size / 784:.2f}×")
            st.caption("Block normalization: L2-Hys over 2×2 cells. Gradients vote with magnitude into angular bins.")
        fig_("fe_02_hog_visual")

    with t[1]:
        ab = csv("fe_hog_ablation.csv")
        if ab is not None:
            fig_("fe_01_hog_ablation")
            st.dataframe(ab, hide_index=True, use_container_width=True)
            st.info(f"Final Chosen Configuration: **{HOG_PARAMS['orientations']} orientations, "
                    f"{HOG_PARAMS['pixels_per_cell'][0]}×{HOG_PARAMS['pixels_per_cell'][1]} cells, 2×2 blocks** (864 dimensions). "
                    "Selected to balance discriminative accuracy against model inference latency.")

    with t[2]:
        a, b = st.columns([3, 2])
        with a:
            fig_("fe_03_shape_stats_profile")
        with b:
            fig_("fe_04_shape_stats_importance")
        st.caption("16 interpretable global shape statistics (bounding box, symmetry, centroid, edge density).")

    with t[3]:
        a, b = st.columns(2)
        with a:
            fig_("fe_06_feature_set_probe")
        with b:
            fig_("fe_05_pca_sweep")
        st.dataframe(csv("fe_feature_set_probe.csv"), hide_index=True)


# ============================================================================ PAGE 4
elif page == PAGES[3]:
    st.title("Model Application & Comparative Results")
    st.caption("Leaderboard across 12 trained configurations, significance tests, and interactive model comparator.")
    
    df, runs = get_results()
    best = df.sort_values("accuracy", ascending=False).iloc[0]
    
    c = st.columns(4)
    kpi(c[0], f"{best.accuracy:.2%}", f"Top Accuracy ({best.model} + {best.features})", "#55A868")
    kpi(c[1], f"{best.macro_f1:.3f}", "Macro-F1 Score", "#4C72B0")
    pix_max = df[df.features == "pixels"].accuracy.max()
    hog_max = df[df.features == "hog"].accuracy.max()
    kpi(c[2], f"{hog_max - pix_max:+.2%}", "HOG vs. Raw Pixels Gain", "#8172B3")
    kpi(c[3], f"{len(df)}", "Evaluated Configurations", "#C44E52")

    st.markdown("#### 🏆 Benchmark Results Leaderboard")
    flt = st.multiselect("Filter Feature Sets", ["pixels", "hog", "hog_stats"], ["pixels", "hog", "hog_stats"])
    show = df[df.features.isin(flt)].sort_values("accuracy", ascending=False)
    st.dataframe(
        show[["model", "features", "dim", "val_acc", "accuracy", "macro_f1", "fit_seconds", "infer_ms_per_image", "params"]]
        .style.format({
            "val_acc": "{:.4f}",
            "accuracy": "{:.4f}",
            "macro_f1": "{:.4f}",
            "infer_ms_per_image": "{:.4f}",
            "params": "{:,.0f}"
        }).background_gradient(subset=["accuracy", "macro_f1"], cmap="Greens"),
        hide_index=True,
        use_container_width=True
    )

    # ------------------------------------------------------------------------- Head-to-Head Model Comparator
    st.markdown("---")
    st.markdown("### ⚡ Interactive Head-to-Head Model Comparator")
    st.caption("Select any two models to inspect differences in accuracy, macro-F1, inference latency, and memory.")
    
    model_keys = [f"{r['model']} ({r['features']})" for _, r in df.iterrows()]
    cmp1, cmp2 = st.columns(2)
    with cmp1:
        m1_choice = st.selectbox("Select Baseline Model", model_keys, index=model_keys.index("LogReg (pixels)"))
    with cmp2:
        m2_choice = st.selectbox("Select Challenger Model", model_keys, index=model_keys.index("SVM-RBF (hog)"))

    r1 = df.iloc[model_keys.index(m1_choice)]
    r2 = df.iloc[model_keys.index(m2_choice)]

    diff_acc = (r2["accuracy"] - r1["accuracy"]) * 100
    diff_f1 = r2["macro_f1"] - r1["macro_f1"]
    speedup = r1["infer_ms_per_image"] / max(r2["infer_ms_per_image"], 1e-6)

    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    mcol1.metric("Accuracy Difference", f"{diff_acc:+.2f}%", delta=f"{diff_acc:+.2f}%")
    mcol2.metric("Macro-F1 Difference", f"{diff_f1:+.4f}", delta=f"{diff_f1:+.4f}")
    if speedup >= 1.0:
        mcol3.metric("Latency Comparison", f"{speedup:.1f}× faster", f"{m1_choice.split()[0]} is faster")
    else:
        mcol3.metric("Latency Comparison", f"{1.0/speedup:.1f}× slower", f"{m2_choice.split()[0]} is slower")
    mcol4.metric("Parameter Count", f"{r2['params']:,.0f}", f"vs. {r1['params']:,.0f}")

    fig_("res_01_accuracy_f1_overview")
    a, b = st.columns(2)
    with a:
        fig_("res_07_accuracy_vs_latency")
    with b:
        fig_("res_08_learning_curve")

    st.markdown("#### Statistical Significance (Paired McNemar Test)")
    mc = csv("mcnemar_pixels_vs_hog.csv")
    if mc is not None:
        st.dataframe(mc, hide_index=True, use_container_width=True)

    ci = csv("bootstrap_ci.csv")
    if ci is not None:
        with st.expander("📊 95% Bootstrap Confidence Intervals (1,000 resamples)"):
            st.dataframe(ci, hide_index=True)


# ============================================================================ PAGE 5
elif page == PAGES[4]:
    st.title("Diagnostic Error Analysis")
    st.caption("Deep-dive into confusions and misclassifications within the upper-body garment cluster.")
    
    df, runs = get_results()
    keys = sorted(runs)
    a, b = st.columns(2)
    ka = a.selectbox("Model A", [f"{m} | {f}" for m, f in keys], index=[k for k in keys].index(max([k for k in keys if k[1] == "pixels"], key=lambda k: runs[k]["accuracy"])))
    kb = b.selectbox("Model B", [f"{m} | {f}" for m, f in keys], index=[k for k in keys].index(max([k for k in keys if k[1] == "hog"], key=lambda k: runs[k]["accuracy"])))
    ra, rb = runs[tuple(ka.split(" | "))], runs[tuple(kb.split(" | "))]
    
    norm = st.toggle("Normalize Rows (Class Recall)", False)
    cols = st.columns(2)
    for col, r, k in zip(cols, (ra, rb), (ka, kb)):
        cm = confusion_matrix(S.y_test, r["pred"], normalize="true" if norm else None)
        fig = px.imshow(
            cm, x=CLASS_NAMES, y=CLASS_NAMES, text_auto=".2f" if norm else "d",
            color_continuous_scale="Blues", title=f"{k} (acc {r['accuracy']:.3f})", aspect="auto", height=490
        )
        fig.update_xaxes(title="Predicted", tickangle=45)
        fig.update_yaxes(title="True Label")
        col.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Where did Model B fix (or break) Model A?")
    fig_("res_04_confusion_difference")

    # ------------------------------------------------------------------------- Interactive Confusion Pair Inspector
    st.markdown("---")
    st.markdown("### 🔍 Interactive Confusion Pair Inspector")
    st.caption("Inspect specific misclassified image pairs directly.")
    
    inspect_model_key = st.selectbox("Inspect Model", [ka, kb])
    r_inspect = runs[tuple(inspect_model_key.split(" | "))]
    
    pair_c1, pair_c2 = st.columns(2)
    true_inspect_cls = pair_c1.selectbox("True Garment Category", CLASS_NAMES, index=6)  # Default Shirt
    pred_inspect_cls = pair_c2.selectbox("Misclassified As", CLASS_NAMES, index=0)  # Default T-shirt
    
    true_idx = CLASS_NAMES.index(true_inspect_cls)
    pred_idx = CLASS_NAMES.index(pred_inspect_cls)
    
    pair_err_ids = np.where((S.y_test == true_idx) & (r_inspect["pred"] == pred_idx))[0]
    
    if len(pair_err_ids) > 0:
        st.warning(f"Found **{len(pair_err_ids)}** instances where **{true_inspect_cls}** was misclassified as **{pred_inspect_cls}**.")
        sample_count = min(12, len(pair_err_ids))
        err_cols = st.columns(6)
        for idx_num, img_idx in enumerate(pair_err_ids[:sample_count]):
            err_cols[idx_num % 6].image(
                S.X_test[img_idx],
                caption=f"Index #{img_idx}\nTrue: {true_inspect_cls}\nPred: {pred_inspect_cls}",
                use_container_width=True,
                clamp=True
            )
    else:
        st.success(f"No misclassifications found for {true_inspect_cls} predicted as {pred_inspect_cls}!")

    ub = csv("upper_body_error_share.csv")
    if ub is not None:
        c1, c2 = st.columns([2, 3])
        c1.markdown("**Share of Errors inside {T-shirt, Pullover, Coat, Shirt}**")
        c1.dataframe(ub, hide_index=True)
        with c2:
            fig_("res_05_upper_body_error_share")


# ============================================================================ PAGE 6: LIVE DEMO & STUDIO (ENHANCED)
elif page == PAGES[5]:
    st.markdown("""
    <div class="hero-header">
      <h1>Live Prediction & Feature Inspection Studio</h1>
      <p>Test models in real time, inspect HOG orientation fields, and compare class probabilities.</p>
    </div>
    """, unsafe_allow_html=True)

    models = {n: get_model(n) for n in ("LogReg__hog", "SVM-RBF__hog", "LogReg__pixels", "SVM-RBF__pixels")}
    models = {k: v for k, v in models.items() if v is not None}
    
    if not models:
        st.error("No trained models found in results/models. Run `python scripts/03_train_models.py` first.")
        st.stop()

    # ------------------------------------------------------------------------- Quick-select edge-case preset gallery
    st.markdown("#### ⚡ Quick Preset Edge-Cases")
    st.caption("Click a curated test sample to evaluate challenging decision boundaries instantly.")
    
    presets = {
        "👔 Ambiguous Shirt (#14)": 14,
        "🧥 Pullover vs. Coat (#42)": 42,
        "👡 Sandal (#7)": 7,
        "👟 Sneaker (#23)": 23,
        "👜 Bag with Strap (#19)": 19,
        "👗 Difficult Dress (#99)": 99,
    }
    
    p_cols = st.columns(len(presets) + 1)
    chosen_preset_idx = None
    
    for idx_btn, (p_name, p_idx) in enumerate(presets.items()):
        if p_cols[idx_btn].button(p_name, use_container_width=True):
            st.session_state["test_img_idx"] = p_idx
            st.session_state["input_mode"] = "Test-set Image"
            
    if p_cols[-1].button("🎲 Random Sample", use_container_width=True):
        st.session_state["test_img_idx"] = int(np.random.default_rng().integers(0, 10000))
        st.session_state["input_mode"] = "Test-set Image"

    input_mode = st.radio(
        "Image Input Source",
        ["Test-set Image", "Upload Custom Image"],
        horizontal=True,
        key="input_mode"
    )

    img = truth = None

    if input_mode == "Test-set Image":
        curr_idx = st.session_state.get("test_img_idx", 42)
        i = st.slider("Test Set Index", 0, 9999, curr_idx, key="test_slider")
        img = S.X_test[i]
        truth = CLASS_NAMES[S.y_test[i]]
    else:
        up = st.file_uploader("Upload clothing photo", type=["png", "jpg", "jpeg"])
        inv = st.checkbox("Invert Grayscale (recommended if garment is dark on light background)", value=True)
        if up:
            im = Image.open(up).convert("L").resize((28, 28))
            img = np.asarray(im, dtype=np.float32) / 255.0
            img = (1.0 - img) if inv else img

    if img is not None:
        st.markdown("---")
        st.markdown("### 🔬 Multi-View Feature Inspection Studio")
        
        # Compute visual representations
        _, himg = hog_with_image(img)
        gx, gy = ndi.sobel(img, axis=1), ndi.sobel(img, axis=0)
        edge_mag = np.hypot(gx, gy)
        
        f1, f2, f3, f4 = st.columns(4)
        f1.image(img, caption=f"Raw Grayscale (28×28)\nTrue: {truth or 'Custom'}", use_container_width=True, clamp=True)
        f2.image(edge_mag / (edge_mag.max() + 1e-9), caption="Sobel Gradient Magnitude", use_container_width=True, clamp=True)
        f3.image(exposure.rescale_intensity(himg, in_range=(0, himg.max() + 1e-9)), caption="HOG Orientation Field", use_container_width=True, clamp=True)
        with f4:
            st.markdown("**Image Statistics**")
            st.metric("Mean Intensity", f"{img.mean():.3f}")
            st.metric("Active Pixels (>0.1)", f"{(img > 0.1).sum()}/784")
            st.metric("Contrast (Std)", f"{img.std():.3f}")

        # Compute predictions across all loaded models
        st.markdown("### 🤖 Multi-Model Predictions")
        pred_cols = st.columns(len(models))
        
        for idx_m, (name, m) in enumerate(models.items()):
            mdl, feat = name.split("__")
            x = img[None]
            feat_vec = x.reshape(1, -1) if feat == "pixels" else compute_hog(x, n_jobs=1)
            pred = int(m.predict(feat_vec)[0])
            pred_name = CLASS_NAMES[pred]
            emoji = CLASS_EMOJIS.get(pred_name, "👕")
            is_correct = (pred_name == truth) if truth else None
            
            with pred_cols[idx_m]:
                status_color = "#16a34a" if is_correct else ("#dc2626" if is_correct is False else "#4f46e5")
                status_text = "MATCH" if is_correct else ("MISMATCH" if is_correct is False else "PREDICTED")
                
                st.markdown(f"""
                <div class="pred-card" style="border-top: 4px solid {status_color};">
                  <div style="font-size:0.8rem; font-weight:600; color:#64748b;">{mdl} · <span class="badge badge-blue">{feat}</span></div>
                  <div style="font-size:1.5rem; font-weight:700; margin:6px 0;">{emoji} {pred_name}</div>
                  <div style="font-size:0.75rem; font-weight:700; color:{status_color};">{status_text}</div>
                </div>
                """, unsafe_allow_html=True)

        # Probability distribution chart
        if hasattr(models.get("LogReg__hog"), "predict_proba"):
            p_hog = models["LogReg__hog"].predict_proba(compute_hog(img[None], n_jobs=1))[0]
            fig_prob = px.bar(
                x=CLASS_NAMES, y=p_hog,
                labels={"x": "Garment Class", "y": "Probability"},
                title="Class Probability Distribution (Logistic Regression + HOG)",
                color=p_hog,
                color_continuous_scale="Blues",
                height=320
            )
            fig_prob.update_layout(showlegend=False, xaxis_tickangle=35)
            st.plotly_chart(fig_prob, use_container_width=True)


# ============================================================================ PAGE 7
else:
    st.title("Conclusions, Limitations & Next Steps")
    st.markdown("""
### Summary of Findings
1. **Best Raw-Pixel Learner**: Support Vector Machine (89.88% test accuracy, 0.898 macro-F1).
2. **Best HOG Learner**: Support Vector Machine with RBF Kernel (**91.30% test accuracy**, 0.913 macro-F1).
3. **Statistical Significance**: HOG descriptors improve Logistic Regression (+4.0 pts, p < 0.001), kNN (+2.1 pts, p < 0.001), and SVM (+1.4 pts, p < 0.001).
4. **Random Forest Independence**: Random Forest gains no benefit (+0.2 pts, p = 0.49), as orthogonal tree thresholding constructs feature interactions directly from raw pixels.
5. **Upper-Body Ambiguity**: 58–65% of all model errors reside within `{T-shirt, Pullover, Coat, Shirt}`. Shirt recall reaches only 0.75 at best due to low-resolution $28\times28$ information loss.

### Limitations
- Fixed HOG parameters (tuned on Logistic Regression probe, not per model).
- Single stratified validation split.
- Subsampled grid search for SVM due to $O(N^2)$ scaling.

### Phase 2 Preview (Deep Learning)
A 2-to-3 block Convolutional Neural Network (CNN) with spatial pooling and dropout on the **exact same splits** to learn hierarchical filters capable of resolving fine collar and button textures.
""")
