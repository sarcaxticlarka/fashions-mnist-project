"""Fashion-MNIST Studio - Phase 1 (classical ML) and Phase 2 (deep learning, final).

Every number on the Phase 2 pages is read from results/tables, so re-running the
scripts updates the app automatically.
Run: streamlit run app/streamlit_app.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch
from PIL import Image
from scipy import ndimage as ndi
from skimage import exposure

from src.config import CLASS_NAMES, FIG_DIR, TABLE_DIR, RUN_DIR, MODEL_DIR, UPPER_BODY
from src.data_loader import make_splits
from src.features import hog_with_image, compute_hog
from src.cnn_models import get_model as get_cnn
from src.cnn_explain import GradCAM

st.set_page_config(page_title="Fashion-MNIST Studio", page_icon="👕", layout="wide", initial_sidebar_state="expanded")

# =============================================================================
# 2. CLEAN ENGINEERING DESIGN SYSTEM (HIGH CONTRAST, ZERO AI CLUTTER)
# =============================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    color: #f0f6fc;
}

.block-container {
    padding-top: 1.2rem;
    padding-bottom: 3.5rem;
    max-width: 1340px;
}

/* Header Container */
.project-header {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 24px 28px;
    margin-bottom: 20px;
}
.project-header h1 {
    font-size: 1.85rem;
    font-weight: 700;
    margin: 6px 0 6px 0;
    color: #f0f6fc;
    letter-spacing: -0.02em;
}
.project-header p {
    color: #8b949e;
    margin: 0;
    font-size: 0.98rem;
    line-height: 1.55;
    max-width: 980px;
}

/* Status Badges */
.badge {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 3px 8px;
    border-radius: 6px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
.badge-blue {
    background: rgba(139, 124, 246, 0.15);
    border: 1px solid rgba(139, 124, 246, 0.4);
    color: #8b7cf6;
}
.badge-green {
    background: rgba(47, 194, 140, 0.15);
    border: 1px solid rgba(47, 194, 140, 0.4);
    color: #2fc28c;
}
.badge-gray {
    background: rgba(110, 118, 129, 0.15);
    border: 1px solid rgba(110, 118, 129, 0.4);
    color: #8b949e;
}
.badge-red {
    background: rgba(248, 81, 73, 0.15);
    border: 1px solid rgba(248, 81, 73, 0.4);
    color: #f85149;
}

/* Metric KPI Tile */
.metric-box {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 16px 18px;
    height: 100%;
}
.metric-box .metric-label {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #8b949e;
    margin: 0;
}
.metric-box .metric-val {
    font-size: 1.85rem;
    font-weight: 700;
    color: #f0f6fc;
    margin: 4px 0 2px 0;
    line-height: 1.15;
}
.metric-box .metric-sub {
    font-size: 0.8rem;
    font-weight: 500;
    color: #8b949e;
}

/* Content Section Card */
.section-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 20px 22px;
    margin-bottom: 16px;
}
.section-card h3 {
    margin: 0 0 8px 0;
    font-size: 1.15rem;
    font-weight: 600;
    color: #f0f6fc;
}
.section-card p, .section-card li {
    color: #8b949e;
    line-height: 1.6;
    font-size: 0.92rem;
}

/* Diagnostic Box */
.diagnostic-box {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 18px;
    height: 100%;
}
.diagnostic-box h4 {
    margin: 4px 0 8px 0;
    color: #f0f6fc;
    font-size: 1.15rem;
}

/* Callout Alerts */
.callout-info {
    background: rgba(139, 124, 246, 0.08);
    border: 1px solid rgba(139, 124, 246, 0.3);
    border-left: 4px solid #8b7cf6;
    border-radius: 6px;
    padding: 12px 16px;
    margin: 14px 0;
    color: #c9d1d9;
    font-size: 0.9rem;
    line-height: 1.5;
}

.callout-success {
    background: rgba(47, 194, 140, 0.08);
    border: 1px solid rgba(47, 194, 140, 0.3);
    border-left: 4px solid #2fc28c;
    border-radius: 6px;
    padding: 12px 16px;
    margin: 14px 0;
    color: #c9d1d9;
    font-size: 0.9rem;
    line-height: 1.5;
}

.callout-warning {
    background: rgba(248, 81, 73, 0.08);
    border: 1px solid rgba(248, 81, 73, 0.3);
    border-left: 4px solid #f85149;
    border-radius: 6px;
    padding: 12px 16px;
    margin: 14px 0;
    color: #c9d1d9;
    font-size: 0.9rem;
    line-height: 1.5;
}

/* High Contrast Data Table */
.benchmark-table-wrapper {
    border: 1px solid #30363d;
    border-radius: 8px;
    overflow: hidden;
    margin: 14px 0;
}
table.benchmark-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.88rem;
    background: #0d1117;
}
table.benchmark-table th {
    background: #161b22;
    color: #8b949e;
    font-weight: 600;
    text-align: left;
    padding: 10px 14px;
    border-bottom: 1px solid #30363d;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
table.benchmark-table td {
    padding: 10px 14px;
    border-bottom: 1px solid #21262d;
    color: #c9d1d9;
}
table.benchmark-table tr:hover {
    background: #161b22;
}
table.benchmark-table tr.top-row {
    background: rgba(47, 194, 140, 0.08);
}
table.benchmark-table tr.top-row td {
    color: #ffffff;
    font-weight: 600;
}

/* Buttons */
.stButton > button {
    border-radius: 6px;
    border: 1px solid #30363d;
    background: #21262d;
    color: #c9d1d9;
    font-weight: 500;
    font-size: 0.85rem;
    padding: 6px 14px;
    transition: all 0.15s ease;
}
.stButton > button:hover {
    border-color: #8b7cf6;
    color: #ffffff;
    background: #30363d;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    border-bottom: 1px solid #30363d;
    padding-bottom: 2px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 6px 6px 0 0;
    padding: 8px 16px;
    font-weight: 600;
    font-size: 0.88rem;
    color: #8b949e;
}
.stTabs [aria-selected="true"] {
    color: #ffffff !important;
    border-bottom: 2px solid #8b7cf6 !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0d1117;
    border-right: 1px solid #30363d;
}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# 3. DATA LOADERS & HELPERS
# =============================================================================
REPORT_FIG = ROOT / "report" / "figures"
VIOLET, ORANGE, AQUA, GREY = "#8b7cf6", "#f0834f", "#2fc28c", "#8b949e"
BG, PANEL, GRIDC, TXT = "#0d1117", "#161b22", "#21262d", "#c9d1d9"
NICE = {
    "CNN3Block_GAP": "CNN3Block-GAP (first run)", "CNN2Block": "CNN2Block", "CNN2Block_Aug": "CNN2Block-Aug",
    "CNN3Block_GAP_NoAug": "CNN3Block-GAP, no aug", "FinalEnsemble": "★ Final ensemble",
}


@st.cache_data(show_spinner="Loading Fashion-MNIST dataset splits...")
def get_splits():
    return make_splits()


@st.cache_resource
def load_classical_model(name="SVM-RBF__hog"):
    p = MODEL_DIR / f"{name}.joblib"
    return joblib.load(p) if p.exists() else None


def _mtime(p):
    return p.stat().st_mtime if p.exists() else 0.0


@st.cache_resource
def _load_cnn(name, mtime):
    p = MODEL_DIR / f"{name}.pt"
    if not p.exists():
        return None
    net = get_cnn(name)
    net.load_state_dict(torch.load(p, map_location="cpu"))
    net.eval()
    return net


def load_deep_cnn(name="CNN3Block_GAP"):
    return _load_cnn(name, _mtime(MODEL_DIR / f"{name}.pt"))


# Caches are keyed on file modification time, so re-running the scripts refreshes the app.
@st.cache_data
def _csv(path, mtime):
    return pd.read_csv(path) if path.exists() else None


@st.cache_data
def _json(path, mtime):
    return json.loads(path.read_text()) if path.exists() else None


@st.cache_data
def _npy(path, mtime):
    return np.load(path) if path.exists() else None


def get_table(name):
    p = TABLE_DIR / name
    return _csv(p, _mtime(p))


def get_json(name):
    p = TABLE_DIR / name
    return _json(p, _mtime(p))


def get_history(name):
    p = TABLE_DIR / f"{name}__history.csv"
    return _csv(p, _mtime(p))


def load_pred(name):
    for f in (RUN_DIR / f"{name}__pixels__pred.npy", RUN_DIR / f"{name}__hog__pred.npy"):
        if f.exists():
            return _npy(f, _mtime(f))
    return None


def render_fig(name, caption=None):
    for d in (FIG_DIR, REPORT_FIG):
        p = d / f"{name}.png"
        if p.exists():
            st.image(str(p), caption=caption, width="stretch")
            return


def kpi_card(col, value, label, subtext, sub_color="#2fc28c"):
    col.markdown(f"""
    <div class="metric-box">
      <div class="metric-label">{label}</div>
      <div class="metric-val">{value}</div>
      <div class="metric-sub" style="color: {sub_color};">{subtext}</div>
    </div>
    """, unsafe_allow_html=True)


def header(badges, title, text):
    b = "".join(f'<span class="badge {c}">{t}</span>' for t, c in badges)
    st.markdown(f"""
    <div class="project-header">
      <div style="display:flex; gap:8px; margin-bottom:10px; flex-wrap:wrap;">{b}</div>
      <h1>{title}</h1>
      <p>{text}</p>
    </div>""", unsafe_allow_html=True)


def callout(kind, html):
    st.markdown(f'<div class="callout-{kind}">{html}</div>', unsafe_allow_html=True)


def style(fig, x_title, y_title, height=420, legend=True):
    """Shared plotly look: every chart states what its axes measure."""
    fig.update_layout(
        paper_bgcolor=PANEL, plot_bgcolor=PANEL, font=dict(color=TXT, size=13), height=height,
        margin=dict(l=10, r=10, t=40, b=10), showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, title_text=""),
        hoverlabel=dict(bgcolor="#21262d"),
    )
    fig.update_xaxes(title_text=x_title, gridcolor=GRIDC, zeroline=False, title_font=dict(size=13, color="#f0f6fc"))
    fig.update_yaxes(title_text=y_title, gridcolor=GRIDC, zeroline=False, title_font=dict(size=13, color="#f0f6fc"))
    return fig


def axis_note(text):
    st.caption(f"📐 How to read: {text}")


def nice_name(m):
    if m in NICE:
        return NICE[m]
    return m.replace("CNN3Block_GAP_E40", "CNN3Block-GAP 40ep").replace("_s", " · seed ").replace("_", " ")


# ---- final-model inference (same recipe as scripts/10_ensemble_calibration.py) ----
def load_final_members():
    fs = get_json("p2_final_summary.json")
    if not fs:
        return None, None
    nets = [load_deep_cnn(m) for m in fs["members"]]
    if any(n is None for n in nets):
        return None, fs
    return nets, fs


def final_predict(img28, nets, fs):
    """Ensemble + flip TTA + temperature scaling + Shirt bias. Returns (probs, pred)."""
    x = torch.from_numpy(img28[None, None].astype(np.float32))
    with torch.no_grad():
        ps = []
        for n in nets:
            ps.append((torch.softmax(n(x), 1) + torch.softmax(n(torch.flip(x, dims=[3])), 1)) / 2)
        p = torch.stack(ps).mean(0).squeeze().numpy()
    logp = np.log(p + 1e-12) / fs["temperature"]
    probs = np.exp(logp - logp.max()); probs /= probs.sum()
    adj = np.log(probs + 1e-12); adj[CLASS_NAMES.index("Shirt")] += fs["shirt_bias"]
    return probs, int(adj.argmax())


def gradcam_overlay(net, img28, cls):
    t = torch.from_numpy(img28[None, None].astype(np.float32))
    cam_tool = GradCAM(net, net.get_last_conv_layer())
    cam = cam_tool.generate(t, target_class=cls)
    cam_tool.remove_hooks()
    fig, ax = plt.subplots(figsize=(3, 3))
    fig.patch.set_facecolor(PANEL)
    ax.imshow(img28, cmap="gray"); ax.imshow(cam, cmap="jet", alpha=0.5); ax.axis("off")
    return fig


def preprocess_upload(file):
    """Turn a user photo into a Fashion-MNIST-like 28x28 image (light garment on black)."""
    im = Image.open(file).convert("L")
    a = np.asarray(im, dtype=np.float32) / 255.0
    if np.median(np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])) > 0.5:
        a = 1.0 - a  # white studio background -> invert to black background
    h, w = a.shape
    s = max(h, w)
    canvas = np.zeros((s, s), np.float32)
    canvas[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = a
    im = Image.fromarray((canvas * 255).astype(np.uint8)).resize((28, 28), Image.LANCZOS)
    return np.asarray(im, dtype=np.float32) / 255.0


# =============================================================================
# 4. SIDEBAR
# =============================================================================
st.sidebar.markdown("""
<div style="margin-bottom:16px; padding-bottom:12px; border-bottom:1px solid #30363d;">
  <h3 style="margin:0; font-size:1.15rem; font-weight:700; color:#f0f6fc; line-height:1.2;">Fashion-MNIST</h3>
  <p style="margin:2px 0 0 0; font-size:0.75rem; color:#8b949e; text-transform:uppercase; letter-spacing:0.05em;">Phase 1 + Phase 2 · Final</p>
</div>
""", unsafe_allow_html=True)

PAGES = [
    "1 · Overview & Final Results",
    "2 · Try the Models (Live)",
    "3 · Features & Grad-CAM",
    "4 · Phase 1: Classical ML",
    "5 · Phase 2: Models & Training",
    "6 · Phase 2: Experiments",
    "7 · Phase 2: Errors & Reliability",
    "8 · Design Decisions",
    "9 · Leaderboard & Deliverables",
]
page = st.sidebar.radio("Navigation", PAGES)
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="font-size:0.78rem; color:#8b949e; line-height:1.55;">
  <span style="color:#c9d1d9; font-weight:600;">Protocol (both phases)</span><br>
  • 48,000 train / 12,000 val / 10,000 test<br>
  • Exactly 1,000 test images per class<br>
  • All choices made on validation only<br>
  • Test set used once for final numbers
</div>
""", unsafe_allow_html=True)

S = get_splits()
FS = get_json("p2_final_summary.json")
DECK = get_json("p2_deck_data.json")
cnn_gap = load_deep_cnn("CNN3Block_GAP")
svm_hog = load_classical_model("SVM-RBF__hog")
final_nets, _ = load_final_members()
best_single_name = FS["best_single"]["run"] if FS else "CNN3Block_GAP"
cam_net = load_deep_cnn(best_single_name) or cnn_gap

if FS:
    F_ACC, F_ERR = FS["test_acc"] * 100, FS["errors"]
    F_SHIRT = FS["per_class_recall"]["Shirt"] * 100
    F_PARAMS, F_MS = FS["params"], FS["ms_per_image"]
else:
    F_ACC, F_ERR, F_SHIRT, F_PARAMS, F_MS = 93.09, 691, 76.3, 304682, 0.68


# =============================================================================
# PAGE 1: OVERVIEW
# =============================================================================
if page == PAGES[0]:
    header([("Phase 1 complete", "badge-green"), ("Phase 2 complete · final", "badge-green"), ("Official 70k benchmark", "badge-blue")],
           "Fashion-MNIST: from Handcrafted Features to a Reliable CNN",
           "Phase 1 benchmarked classical machine learning on raw pixels and handcrafted HOG features. "
           "Phase 2 replaced handcrafted features with convolutional networks, explained where the gain comes from, "
           "and turned the best networks into a calibrated, well-tested final model.")

    k1, k2, k3, k4 = st.columns(4)
    kpi_card(k1, f"{F_ACC:.2f}%", "Final test accuracy", f"+{F_ACC - 91.30:.2f} pts vs Phase 1 (91.30%)")
    kpi_card(k2, f"{F_ERR}", "Test errors (of 10,000)", f"{870 - F_ERR} fewer than the SVM's 870")
    kpi_card(k3, f"{F_SHIRT:.1f}%", "Shirt recall (hardest class)", "was 75.0% in Phase 1", ORANGE)
    if FS:
        kpi_card(k4, f"{FS['abstain']['test_acc_on_kept']*100:.2f}%", "Accuracy when confident",
                 f"on {FS['abstain']['test_coverage']*100:.0f}% of images · rest to a human", VIOLET)
    else:
        kpi_card(k4, "0.68 ms", "CPU time per image", "8× faster than SVM", VIOLET)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🧭 The project in four steps")
    c = st.columns(4)
    steps = [
        ("Phase 1", "Classical ML + HOG", "Best: SVM-RBF 91.30%. 74% of its errors are T-shirt / Pullover / Coat / Shirt.", GREY),
        ("Phase 2 · step 1", "First CNNs", "CNN3Block-GAP reaches 93.09% (p = 2.5×10⁻¹¹ vs SVM), 60× smaller, 8× faster.", VIOLET),
        ("Phase 2 · step 2", "Why it works", "2×2 ablation: depth + GAP only help together with augmentation. Longer training + 3 seeds.", VIOLET),
        ("Phase 2 · final", "Reliable model", f"ResNet + ensemble + TTA + calibration: {F_ACC:.2f}%, honest confidence, Shirt {F_SHIRT:.1f}%.", AQUA),
    ]
    for col, (k, h, b, colr) in zip(c, steps):
        col.markdown(f"""<div class="section-card" style="min-height:190px;">
          <span class="badge badge-gray" style="color:{colr}; border-color:{colr};">{k}</span>
          <h3 style="margin-top:8px;">{h}</h3><p>{b}</p></div>""", unsafe_allow_html=True)

    st.markdown("### 📈 Accuracy journey")
    journey = [("LogReg (pixels)", 84.56, GREY), ("SVM-RBF (HOG)\nPhase 1 best", 91.30, ORANGE),
               ("CNN2Block", 92.16, VIOLET), ("CNN3Block-GAP\nfirst run", 93.09, VIOLET)]
    if FS:
        journey += [("Best single net", FS["best_single"]["test_acc"] * 100, VIOLET), ("Final model", F_ACC, AQUA)]
    fig = go.Figure(go.Bar(x=[j[0].replace("\n", "<br>") for j in journey], y=[j[1] for j in journey],
                           marker_color=[j[2] for j in journey], text=[f"{j[1]:.2f}%" for j in journey], textposition="outside"))
    fig.update_yaxes(range=[83, 96], ticksuffix="%")
    st.plotly_chart(style(fig, "Model (left = first Phase 1 model, right = final Phase 2 model)", "Test accuracy on 10,000 unseen images (%)", 400, False), width="stretch")
    axis_note("each bar is one model; taller = more of the 10,000 test images classified correctly.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""<div class="section-card"><h3>What Phase 2 proved</h3><ul>
          <li>Learned convolutional filters beat every handcrafted-feature model, with a statistically definitive margin.</li>
          <li>The gain needs <b>both</b> a deeper GAP/residual network <b>and</b> augmentation (2×2 ablation).</li>
          <li>Results are stable across 3 random seeds; the best runs needed longer training.</li></ul></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="section-card"><h3>What makes the final model trustworthy</h3><ul>
          <li><b>Calibrated confidence</b>: temperature scaling cuts the gap between stated confidence and real accuracy (ECE) by about 6×.</li>
          <li><b>Abstain option</b>: low-confidence images can be routed to a human.</li>
          <li><b>Shirt-aware decision rule</b> tuned on validation data to catch more shirts.</li></ul></div>""", unsafe_allow_html=True)


# =============================================================================
# PAGE 2: LIVE DIAGNOSTICS
# =============================================================================
elif page == PAGES[1]:
    header([("Interactive", "badge-blue"), ("Live inference", "badge-gray")], "Try the Models",
           "Pick a test image (or upload your own clothing photo) and compare Phase 1's SVM, the first CNN and the final Phase 2 model side by side.")

    src = st.radio("Image source", ["Test set image", "Upload my own photo"], horizontal=True)
    true_label = None
    if src == "Test set image":
        p1, p2, p3, p4 = st.columns(4)
        for col, (idx, lab) in zip([p1, p2, p3, p4], [(26, "#26 Shirt"), (49, "#49 Pullover"), (51, "#51 Coat"), (98, "#98 Coat")]):
            if col.button(f"Hard case {lab}", width="stretch"):
                st.session_state["diag_idx"] = idx
        f1c, f2c = st.columns([1.2, 2.8])
        with f1c:
            cat = st.selectbox("Filter by true class", ["All 10 classes"] + CLASS_NAMES)
        with f2c:
            cur = st.session_state.get("diag_idx", 26)
            if cat != "All 10 classes":
                idxs = np.where(S.y_test == CLASS_NAMES.index(cat))[0]
                cur = cur if cur in idxs else int(idxs[0])
                test_idx = st.select_slider(f"Browse {len(idxs)} '{cat}' test images", options=idxs, value=cur)
            else:
                test_idx = st.slider("Test image index (0–9,999)", 0, 9999, cur)
            st.session_state["diag_idx"] = int(test_idx)
        img = S.X_test[test_idx]
        true_label = CLASS_NAMES[S.y_test[test_idx]]
    else:
        up = st.file_uploader("Upload a photo of a single garment (JPG/PNG)", type=["png", "jpg", "jpeg"])
        st.caption("Works best with one centred item on a plain background. The photo is converted to 28×28 grayscale like Fashion-MNIST.")
        if up is None:
            st.stop()
        img = preprocess_upload(up)

    # --- predictions
    svm_pred = CLASS_NAMES[int(svm_hog.predict(compute_hog(img[None], n_jobs=1))[0])] if svm_hog is not None else "n/a"
    with torch.no_grad():
        gap_probs = torch.softmax(cnn_gap(torch.from_numpy(img[None, None].astype(np.float32))), 1).squeeze().numpy()
    gap_pred = CLASS_NAMES[int(gap_probs.argmax())]
    if final_nets:
        f_probs, f_idx = final_predict(img, final_nets, FS)
        f_pred, f_conf = CLASS_NAMES[f_idx], f_probs[f_idx] * 100
        abstain = f_probs.max() < FS["abstain"]["threshold"]
    else:
        f_probs, f_pred, f_conf, abstain = gap_probs, gap_pred, gap_probs.max() * 100, False

    st.markdown("---")
    c0, c1, c2, c3 = st.columns([1, 1, 1, 1.25])
    with c0:
        st.markdown(f"""<div class="diagnostic-box"><span class="badge badge-gray">Input</span>
          <h4>{true_label or 'Your photo'}</h4><div style="font-size:0.82rem;color:#8b949e;">28×28 grayscale</div></div>""", unsafe_allow_html=True)
        st.image(img, width="stretch", clamp=True)

    def verdict_box(col, name, pred, extra, badge):
        ok = (true_label is None) or (pred == true_label)
        colr = "#2fc28c" if ok else "#f85149"
        status = ("Correct" if pred == true_label else "Wrong") if true_label else "Prediction"
        col.markdown(f"""<div class="diagnostic-box" style="border-top:3px solid {colr if true_label else VIOLET};">
          <div style="display:flex;justify-content:space-between;"><span class="badge {badge}">{name}</span>
          <span class="badge {'badge-green' if ok else 'badge-red'}">{status}</span></div>
          <h4>{pred}</h4><div style="font-size:0.82rem;color:#8b949e;line-height:1.5;">{extra}</div></div>""", unsafe_allow_html=True)

    verdict_box(c1, "Phase 1 · SVM-RBF", svm_pred, "HOG features · 18.1M stored numbers", "badge-gray")
    verdict_box(c2, "Phase 2 · first CNN", gap_pred, f"CNN3Block-GAP · {gap_probs.max()*100:.1f}% (uncalibrated)", "badge-blue")
    with c3:
        verdict_box(st, "Phase 2 · final model", f_pred,
                    f"Calibrated confidence <b>{f_conf:.1f}%</b><br>" +
                    ("⚠️ <b>Below the abstain threshold</b> — would be sent to a human" if abstain else "✅ Confident enough to answer automatically"),
                    "badge-green")

    st.markdown("#### Final model: probability for every class (calibrated)")
    order = np.argsort(f_probs)
    fig = go.Figure(go.Bar(x=f_probs[order] * 100, y=[CLASS_NAMES[i] for i in order], orientation="h",
                           marker_color=[AQUA if CLASS_NAMES[i] == f_pred else VIOLET for i in order],
                           text=[f"{v*100:.1f}%" for v in f_probs[order]], textposition="outside"))
    fig.update_xaxes(range=[0, 110], ticksuffix="%")
    st.plotly_chart(style(fig, "Probability that the image belongs to this class (%)", "Clothing class", 360, False), width="stretch")
    axis_note("bar length = how likely the final model thinks each class is; the bars add up to 100%.")

    g1, g2 = st.columns([1, 2])
    with g1:
        st.markdown(f"#### Grad-CAM · {nice_name(best_single_name)}")
        cls = CLASS_NAMES.index(f_pred)
        fig_c = gradcam_overlay(cam_net, img, cls)
        st.pyplot(fig_c); plt.close(fig_c)
    with g2:
        st.markdown("<br>", unsafe_allow_html=True)
        callout("info", "<b>Reading the heat-map:</b> red regions pushed the network towards its answer. "
                        "For upper-body garments look for attention on the collar, sleeves, cuffs and hem.")
        if true_label and svm_pred != true_label and f_pred == true_label:
            callout("success", f"<b>Phase 2 fixes this case:</b> the SVM said <b>{svm_pred}</b>, the final model correctly says <b>{true_label}</b>.")
        elif true_label and f_pred != true_label:
            callout("warning", f"<b>Still hard:</b> the final model says <b>{f_pred}</b> but the label is <b>{true_label}</b>.")


# =============================================================================
# PAGE 3: FEATURES & GRAD-CAM
# =============================================================================
elif page == PAGES[2]:
    header([("Representations", "badge-blue"), ("Saliency", "badge-gray")], "What Each Model Sees",
           "Raw pixels, edge maps, handcrafted HOG (Phase 1) and learned Grad-CAM attention (Phase 2) for the same image.")
    s1, s2 = st.columns([1.2, 2.8])
    with s1:
        f_cat = st.selectbox("Garment class", ["All classes"] + CLASS_NAMES, index=7)
    with s2:
        if f_cat != "All classes":
            m_idx = np.where(S.y_test == CLASS_NAMES.index(f_cat))[0]
            s_idx = st.select_slider(f"'{f_cat}' test image", options=m_idx, value=m_idx[0])
        else:
            s_idx = st.slider("Test image index", 0, 9999, 42)
    s_img = S.X_test[s_idx]
    _, h_vis = hog_with_image(s_img)
    edge = np.hypot(ndi.sobel(s_img, axis=1), ndi.sobel(s_img, axis=0))
    v = st.columns(5)
    items = [("Raw pixels", s_img, "What every model receives."),
             ("Edge strength (Sobel)", edge / (edge.max() + 1e-9), "Where brightness changes fast — outlines."),
             ("HOG (Phase 1)", exposure.rescale_intensity(h_vis, in_range=(0, h_vis.max() + 1e-9)), "Fixed 4×4-pixel cells of edge directions.")]
    for col, (t, im, cap) in zip(v[:3], items):
        col.markdown(f"**{t}**"); col.image(im, width="stretch", clamp=True); col.caption(cap)
    for col, (net, t) in zip(v[3:], [(cnn_gap, "Grad-CAM · first CNN"), (cam_net, f"Grad-CAM · {nice_name(best_single_name)}")]):
        with torch.no_grad():
            pc = int(net(torch.from_numpy(s_img[None, None].astype(np.float32))).argmax())
        col.markdown(f"**{t}**")
        fig_c = gradcam_overlay(net, s_img, pc); col.pyplot(fig_c); plt.close(fig_c)
        col.caption(f"Predicted: {CLASS_NAMES[pc]}")
    callout("info", "<b>Honest reading:</b> Grad-CAM focuses on the garment, never the background, and often on cuffs, hems and collars. "
                    "Some maps (patterned shirts, printed T-shirts) are diffuse — consistent with Shirt remaining the hardest class.")
    render_fig("cnn_02_gradcam_gallery", "Grad-CAM gallery for the four upper-body classes (first CNN)")


# =============================================================================
# PAGE 4: PHASE 1
# =============================================================================
elif page == PAGES[3]:
    header([("Phase 1", "badge-gray"), ("Classical ML", "badge-blue")], "Phase 1: Classical Machine Learning",
           "Data audit, spatial correlation, HOG feature engineering and 12 classical model configurations.")
    t1, t2, t3, t4 = st.tabs(["Dataset audit", "HOG feature engineering", "12 classical models", "Upper-body error analysis"])
    with t1:
        st.markdown("- No missing, blank or duplicate images · exactly 6,000 train / 1,000 test per class · zero train–test leakage (hash audit).")
        a, b = st.columns(2)
        with a:
            render_fig("eda_01_class_distribution", "Class distribution (balanced)")
            render_fig("eda_07_spatial_correlation", "Neighbouring pixels are correlated (r ≈ 0.75) — images have 2D structure")
        with b:
            render_fig("eda_03_mean_std_images", "Average and variability image per class")
            render_fig("eda_08_class_mean_distance", "How far apart the class averages are")
    with t2:
        st.markdown("Chosen HOG setting: `orientations=6`, `pixels_per_cell=(4,4)`, `cells_per_block=(2,2)` → **864 features**.")
        render_fig("fe_01_hog_ablation", "HOG settings vs validation accuracy")
        a, b = st.columns(2)
        with a: render_fig("fe_02_hog_visual", "HOG visualised on garments")
        with b: render_fig("fe_03_shape_stats_profile", "16 shape statistics per class")
    with t3:
        df = get_table("model_results.csv")
        if df is not None:
            d = df[["model", "features", "dim", "val_acc", "accuracy", "macro_f1", "infer_ms_per_image", "params"]].copy()
            d["val_acc"] = (d.val_acc * 100).round(2); d["accuracy"] = (d.accuracy * 100).round(2)
            d.columns = ["Model", "Features", "Dim", "Val acc (%)", "Test acc (%)", "Macro-F1", "ms / image", "Stored numbers"]
            st.dataframe(d, width="stretch", hide_index=True)
        render_fig("res_01_accuracy_f1_overview", "Accuracy and macro-F1 by model and feature set")
    with t4:
        st.markdown("58–65% of all classical errors sit in `{T-shirt/top, Pullover, Coat, Shirt}`; Shirt recall stalls at 75%.")
        a, b = st.columns(2)
        with a:
            render_fig("res_03_confusion_best_pixels_vs_best_hog", "Confusion: pixels vs HOG")
            render_fig("res_05_upper_body_error_share", "Share of errors in upper-body classes")
        with b:
            render_fig("res_04_confusion_difference", "Confusion change (HOG − pixels)")
            render_fig("res_06b_misclassified_best_hog", "Misclassified upper-body examples (SVM)")


# =============================================================================
# PAGE 5: PHASE 2 MODELS & TRAINING
# =============================================================================
elif page == PAGES[4]:
    header([("Phase 2", "badge-green"), ("Architectures", "badge-blue")], "Phase 2: Models & Training",
           "Five network designs, trained on raw pixels with AdamW, cosine learning-rate decay and early stopping on validation macro-F1.")
    t1, t2, t3 = st.tabs(["Architectures", "Training curves", "Over-fitting check"])
    with t1:
        st.markdown("#### CNN3Block-GAP — the first strong model")
        st.markdown("`Input 1×28×28` → **Block 1** [Conv 32 → Conv 32 → pool] → **Block 2** [Conv 64 → Conv 64 → pool] → "
                    "**Block 3** [Conv 128 → Conv 128] → **Global average pooling** (128 numbers) → Dense 128 + Dropout 0.3 → **10 classes** · 304,682 weights")
        st.markdown("#### ResNetSmall — the final model family")
        st.markdown("`Input` → Conv 32 → **3 stages × 2 residual blocks** (32 → 64 → 128 channels, 28 → 14 → 7 px) → Global average pooling → Dropout 0.2 → **10 classes** · 696,042 weights. "
                    "Residual shortcuts let 13 conv layers train without vanishing gradients. Trained with crop + flip + **Random Erasing** and **label smoothing 0.1**.")
        arch = pd.DataFrame([
            ["CNN2Block", "2 conv blocks + flatten + dense", "421,738", "no", "15"],
            ["CNN2Block-Aug", "same", "421,738", "crop + flip", "18"],
            ["CNN3Block-GAP (no aug)", "3 double-conv blocks + GAP", "304,682", "no", "20 (stopped at 10)"],
            ["CNN3Block-GAP", "3 double-conv blocks + GAP", "304,682", "crop + flip", "20"],
            ["CNN3Block-GAP 40ep × 3 seeds", "same, longer training", "304,682", "crop + flip", "40"],
            ["ResNetSmall × 3 seeds", "residual network + GAP", "696,042", "crop + flip + erasing", "40"],
        ], columns=["Model", "Design", "Weights", "Augmentation", "Epoch budget"])
        st.dataframe(arch, width="stretch", hide_index=True)
        callout("info", "<b>Why global average pooling?</b> Flattening a 128×7×7 map gives 6,272 inputs to a dense layer (800k+ weights). "
                        "GAP averages each channel to one number, so the head needs only 128 inputs — fewer weights, less over-fitting.")
    with t2:
        cands = ["CNN2Block", "CNN2Block_Aug", "CNN3Block_GAP_NoAug", "CNN3Block_GAP",
                 "CNN3Block_GAP_E40_s42", "ResNetSmall_s42"]
        avail = [c for c in cands if get_history(c) is not None]
        sel = st.multiselect("Models to show", avail, default=[a for a in avail if a in ("CNN3Block_GAP", "CNN3Block_GAP_E40_s42", "ResNetSmall_s42", "CNN3Block_GAP_NoAug")],
                             format_func=nice_name)
        metric = st.radio("Metric", ["Validation accuracy", "Validation loss"], horizontal=True)
        fig = go.Figure()
        pal = [VIOLET, AQUA, ORANGE, "#e87ba4", "#58a6ff", GREY]
        for i, m in enumerate(sel):
            h = get_history(m)
            yv = h.val_acc * 100 if metric == "Validation accuracy" else h.val_loss
            fig.add_trace(go.Scatter(x=h.epoch, y=yv, mode="lines+markers", name=nice_name(m), line=dict(width=2.5, color=pal[i % len(pal)])))
        if metric == "Validation accuracy":
            fig.add_hline(y=91.30, line_dash="dot", line_color=ORANGE, annotation_text="Phase 1 best (SVM, test) 91.30%")
            fig.update_yaxes(range=[84, 95.5], ticksuffix="%")
        st.plotly_chart(style(fig, "Training epoch (1 epoch = one pass over 48,000 training images)",
                              "Validation accuracy (%)" if metric == "Validation accuracy" else "Validation loss (lower = better)", 460), width="stretch")
        axis_note("x = how long the model has trained; y = performance on 12,000 validation images it never trains on.")
        gap = get_table("p2_generalisation_gap.csv")
        if gap is not None:
            callout("info", "<b>Finding:</b> the first CNN3Block-GAP and CNN2Block-Aug were still improving at their last epoch, so they were "
                            "<b>under-trained</b>. That is why the final models get a 40-epoch budget with patience 10.")
    with t3:
        m = st.selectbox("Model", [a for a in ["CNN2Block", "CNN3Block_GAP", "CNN3Block_GAP_NoAug", "CNN3Block_GAP_E40_s42", "ResNetSmall_s42"] if get_history(a) is not None], format_func=nice_name)
        h = get_history(m)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=h.epoch, y=h.train_loss, name="Training loss", line=dict(color=GREY, width=2.5)))
        fig.add_trace(go.Scatter(x=h.epoch, y=h.val_loss, name="Validation loss", line=dict(color=VIOLET, width=2.5), fill="tonexty", fillcolor="rgba(240,131,79,0.15)"))
        st.plotly_chart(style(fig, "Training epoch", "Cross-entropy loss (lower = better)", 420), width="stretch")
        axis_note("the shaded gap between the curves shows memorisation: a big gap means the model fits training images much better than new ones.")
        st.caption("Note: ResNetSmall uses label smoothing, so its loss values sit higher by design and are not directly comparable.")


# =============================================================================
# PAGE 6: PHASE 2 EXPERIMENTS
# =============================================================================
elif page == PAGES[5]:
    header([("Phase 2", "badge-green"), ("Controlled experiments", "badge-blue")], "Phase 2: Experiments",
           "Which ingredients matter, are the results stable, and how much does each improvement add?")
    t1, t2, t3, t4 = st.tabs(["2×2 ablation", "3-seed robustness", "Building the final model", "Significance (McNemar)"])
    with t1:
        if DECK and DECK.get("ablation"):
            a = np.array(DECK["ablation"]["acc"])
            fig = go.Figure([
                go.Bar(name="No augmentation", x=["CNN2Block (flatten)", "CNN3Block-GAP"], y=a[:, 0], marker_color=GREY, text=[f"{v:.2f}%" for v in a[:, 0]], textposition="outside"),
                go.Bar(name="Crop + flip augmentation", x=["CNN2Block (flatten)", "CNN3Block-GAP"], y=a[:, 1], marker_color=VIOLET, text=[f"{v:.2f}%" for v in a[:, 1]], textposition="outside"),
            ])
            fig.add_hline(y=91.30, line_dash="dot", line_color=ORANGE, annotation_text="SVM-RBF 91.30%")
            fig.update_yaxes(range=[90, 94], ticksuffix="%")
            st.plotly_chart(style(fig, "Architecture", "Test accuracy (%)", 420), width="stretch")
            axis_note("grey vs violet = same network trained without / with augmentation.")
            callout("success", "<b>Depth + GAP only pays off with augmentation.</b> Either alone gives ≈ 91.4%; together 93.09%. "
                               "Without augmentation the deeper network over-fits after 5 epochs (McNemar p = 1.0×10⁻¹² between the two GAP runs).")
    with t2:
        pr = get_table("p2_final_per_run.csv")
        if pr is None:
            st.info("Run scripts/09_train_final_models.py and scripts/10_ensemble_calibration.py to populate this tab.")
        else:
            fig = go.Figure()
            for fam, colr in [("CNN3Block_GAP_E40", AQUA), ("ResNetSmall", VIOLET)]:
                d = pr[pr.family == fam]
                fig.add_trace(go.Scatter(x=d.test_acc * 100, y=[nice_name(fam)] * len(d), mode="markers", name=nice_name(fam),
                                         marker=dict(size=16, color=colr, line=dict(color="white", width=1.5)),
                                         text=[f"seed {s}" for s in d.seed], hovertemplate="%{text}: %{x:.2f}%<extra></extra>"))
            fig.add_vline(x=93.09, line_dash="dash", line_color=GREY, annotation_text="first run 93.09%")
            fig.add_vline(x=91.30, line_dash="dot", line_color=ORANGE, annotation_text="SVM 91.30%")
            fig.update_xaxes(ticksuffix="%")
            st.plotly_chart(style(fig, "Test accuracy (%) — each dot is one run with a different random seed", "Model family", 320, False), width="stretch")
            axis_note("dots close together = the result does not depend on luck of initialisation.")
            fs = get_table("p2_final_seed_summary.csv")
            d = fs.copy()
            d["Test acc (mean ± std)"] = d.apply(lambda r: f"{r.test_mean*100:.2f}% ± {0 if pd.isna(r.test_std) else r.test_std*100:.2f}", axis=1)
            d["With flip TTA"] = (d.tta_mean * 100).round(2).astype(str) + "%"
            d["Shirt recall"] = (d.shirt_mean * 100).round(1).astype(str) + "%"
            d["family"] = d.family.map(nice_name)
            st.dataframe(d[["family", "n_seeds", "Test acc (mean ± std)", "With flip TTA", "Shirt recall", "params", "ms"]]
                         .rename(columns={"family": "Model", "n_seeds": "Seeds", "params": "Weights", "ms": "ms / image"}), width="stretch", hide_index=True)
    with t3:
        if FS:
            pr = get_table("p2_final_per_run.csv")
            bs = pr.sort_values("val_acc", ascending=False).iloc[0]
            ladder = [("SVM-RBF (HOG)<br>Phase 1 best", 91.30, ORANGE), ("CNN3Block-GAP<br>first run", 93.09, GREY),
                      ("Best single net", bs.test_acc * 100, AQUA), ("+ flip TTA", bs.test_acc_tta * 100, AQUA),
                      (f"Ensemble<br>{len(FS['members'])} nets + TTA", FS["test_acc_before_bias"] * 100, VIOLET), ("+ Shirt bias<br>= final", F_ACC, VIOLET)]
            fig = go.Figure(go.Bar(x=[l[0] for l in ladder], y=[l[1] for l in ladder], marker_color=[l[2] for l in ladder],
                                   text=[f"{l[1]:.2f}%" for l in ladder], textposition="outside"))
            fig.update_yaxes(range=[90.5, max(l[1] for l in ladder) + 0.7], ticksuffix="%")
            st.plotly_chart(style(fig, "Each step adds one improvement (all choices made on validation data)", "Test accuracy on 10,000 images (%)", 430, False), width="stretch")
            ens = get_table("p2_final_ensembles.csv")
            if ens is not None:
                e = ens.copy(); e["val_acc"] = (e.val_acc * 100).round(2); e["test_acc"] = (e.test_acc * 100).round(2)
                e.columns = ["Candidate", "Nets", "Val acc (%)", "Test acc (%)", "Weights", "ms / image (with TTA)"]
                st.markdown("**Ensemble candidates** — the one with the best *validation* accuracy was chosen:")
                st.dataframe(e, width="stretch", hide_index=True)
    with t4:
        rows = []
        mc = get_table("mcnemar_phase1_vs_phase2.csv")
        if mc is not None:
            for _, r in mc.iterrows():
                rows.append([r.model_2, r.svm_only_correct, r.cnn_only_correct, r.chi2, r.p_value])
        if FS:
            m = FS["mcnemar_vs_svm"]
            rows.append(["Final model", m["svm_only"], m["final_only"], round(m["chi2"], 2), m["p"]])
        d = pd.DataFrame(rows, columns=["Phase 2 model", "Only SVM right", "Only CNN right", "χ²", "p"])
        fig = go.Figure([go.Bar(name="Only SVM-RBF (HOG) right", x=d["Phase 2 model"].map(nice_name), y=d["Only SVM right"], marker_color=ORANGE, text=d["Only SVM right"], textposition="outside"),
                         go.Bar(name="Only the Phase 2 model right", x=d["Phase 2 model"].map(nice_name), y=d["Only CNN right"], marker_color=VIOLET, text=d["Only CNN right"], textposition="outside")])
        st.plotly_chart(style(fig, "Phase 2 model compared with the Phase 1 SVM", "Test images where exactly one model is right", 420), width="stretch")
        axis_note("if both models were equally good, each pair of bars would be about the same height.")
        d["p"] = d["p"].map(lambda v: f"{v:.2e}")
        st.dataframe(d, width="stretch", hide_index=True)


# =============================================================================
# PAGE 7: ERRORS & RELIABILITY
# =============================================================================
elif page == PAGES[6]:
    header([("Phase 2", "badge-green"), ("Error analysis", "badge-red"), ("Calibration", "badge-blue")], "Phase 2: Errors & Reliability",
           "Where the model still fails, and whether its confidence can be trusted.")
    t1, t2, t3, t4 = st.tabs(["Per-class recall", "Confusion matrix", "Shirt decision rule", "Calibration & abstain"])
    pred_svm = load_pred("SVM-RBF")
    pred_fin = load_pred("FinalEnsemble") if FS else load_pred("CNN3Block_GAP")
    pred_gap = load_pred("CNN3Block_GAP")
    y = S.y_test
    rec = lambda p: np.array([((p == k) & (y == k)).sum() / 1000 for k in range(10)]) * 100
    with t1:
        r_s, r_g, r_f = rec(pred_svm), rec(pred_gap), rec(pred_fin)
        order = np.argsort(r_f)
        fig = go.Figure([go.Bar(name="SVM-RBF (Phase 1)", y=[CLASS_NAMES[i] for i in order], x=r_s[order], orientation="h", marker_color=ORANGE),
                         go.Bar(name="CNN3Block-GAP (first run)", y=[CLASS_NAMES[i] for i in order], x=r_g[order], orientation="h", marker_color=GREY),
                         go.Bar(name="Final model", y=[CLASS_NAMES[i] for i in order], x=r_f[order], orientation="h", marker_color=VIOLET,
                                text=[f"{v:.1f}%" for v in r_f[order]], textposition="outside")])
        fig.update_xaxes(range=[65, 104], ticksuffix="%")
        st.plotly_chart(style(fig, "Recall: % of the class's 1,000 test images labelled correctly", "Clothing class (hardest at the bottom)", 560), width="stretch")
        axis_note("longer bar = more of that class recognised correctly.")
    with t2:
        which = st.radio("Model", ["Final model", "CNN3Block-GAP (first run)", "SVM-RBF (Phase 1)"], horizontal=True)
        p = {"Final model": pred_fin, "CNN3Block-GAP (first run)": pred_gap, "SVM-RBF (Phase 1)": pred_svm}[which]
        cm = np.zeros((10, 10)); np.add.at(cm, (y, p), 1)
        cmp = cm / cm.sum(1, keepdims=True) * 100
        upper_only = st.checkbox("Show only the 4 upper-body classes", value=True)
        idx = [CLASS_NAMES.index(c) for c in UPPER_BODY] if upper_only else list(range(10))
        sub = cmp[np.ix_(idx, idx)]
        fig = go.Figure(go.Heatmap(z=sub, x=[CLASS_NAMES[i] for i in idx], y=[CLASS_NAMES[i] for i in idx], colorscale="Purples",
                                   text=[[f"{v:.1f}%" for v in row] for row in sub], texttemplate="%{text}", zmin=0, zmax=100,
                                   colorbar=dict(title="% of row")))
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(style(fig, "Predicted class", "True class (each row adds up to 100%)", 520, False), width="stretch")
        axis_note("diagonal = correct; any other cell = how often a true class (row) is mistaken for another (column).")
    with t3:
        sb = get_table("p2_final_shirt_bias.csv")
        if sb is None or not FS:
            st.info("Run scripts/10_ensemble_calibration.py to populate this tab.")
        else:
            fig = go.Figure([go.Scatter(x=sb.bias, y=sb.test_shirt_recall * 100, name="Shirt recall", line=dict(color=VIOLET, width=3)),
                             go.Scatter(x=sb.bias, y=sb.test_shirt_precision * 100, name="Shirt precision", line=dict(color=ORANGE, width=3)),
                             go.Scatter(x=sb.bias, y=sb.test_acc * 100, name="Overall accuracy", line=dict(color=GREY, width=3, dash="dash"))])
            fig.add_vline(x=FS["shirt_bias"], line_color=AQUA, annotation_text=f"chosen on validation: +{FS['shirt_bias']:.2f}")
            fig.update_yaxes(ticksuffix="%")
            st.plotly_chart(style(fig, "Extra score added to 'Shirt' before picking the answer (log-probability units)", "Percentage (%)", 440), width="stretch")
            axis_note("moving right makes the model say 'Shirt' more often: it catches more real shirts (recall ↑) but raises more false alarms (precision ↓).")
            callout("info", "<b>Why this is fair:</b> the bias was picked to maximise macro-F1 on the <b>validation</b> set; the test curve is shown only to explain the trade-off.")
    with t4:
        if not FS:
            st.info("Run scripts/10_ensemble_calibration.py to populate this tab.")
        else:
            c1, c2 = st.columns(2)
            for col, key, ttl in [(c1, "reliability_before", "Before calibration"), (c2, "reliability_after", f"After temperature scaling (T = {FS['temperature']:.2f})")]:
                rel = FS[key]
                fig = go.Figure([go.Bar(x=[(r[0] + r[1]) / 2 * 100 for r in rel], y=[r[3] * 100 for r in rel], width=[(r[1] - r[0]) * 100 for r in rel],
                                        marker_color=VIOLET, name="Actual accuracy", customdata=[r[4] for r in rel],
                                        hovertemplate="confidence ≈ %{x:.0f}%<br>actual accuracy %{y:.1f}%<br>%{customdata} images<extra></extra>"),
                                 go.Scatter(x=[0, 100], y=[0, 100], mode="lines", name="Perfect calibration", line=dict(color=ORANGE, dash="dash"))])
                fig.update_xaxes(range=[30, 100], ticksuffix="%"); fig.update_yaxes(range=[30, 101], ticksuffix="%")
                ece_v = FS["ece_before"] if key == "reliability_before" else FS["ece_after"]
                fig.update_layout(title=f"{ttl} · ECE = {ece_v*100:.2f}%")
                col.plotly_chart(style(fig, "Model's confidence in its answer (%)", "How often it was actually right (%)", 400), width="stretch")
            axis_note("bars on the dashed line = honest confidence. ECE = average gap between confidence and actual accuracy (lower is better).")

            st.markdown("#### Abstain when unsure — try a confidence threshold")
            probs = _npy(RUN_DIR / "FinalEnsemble__test_probs.npy", _mtime(RUN_DIR / "FinalEnsemble__test_probs.npy"))
            conf = probs.max(1)
            thr = st.slider("Only answer automatically when calibrated confidence is at least…", 0.30, 0.99,
                            float(round(FS["abstain"]["threshold"], 2)), 0.01, format="%.2f")
            keep = conf >= thr
            k1, k2, k3, k4 = st.columns(4)
            kpi_card(k1, f"{keep.mean()*100:.1f}%", "Images answered", "automatically")
            kpi_card(k2, f"{(pred_fin[keep] == y[keep]).mean()*100:.2f}%", "Accuracy on answered", f"vs {(pred_fin == y).mean()*100:.2f}% on all")
            kpi_card(k3, f"{(~keep).sum():,}", "Sent to a human", "lowest-confidence images", ORANGE)
            kpi_card(k4, f"{(y[~keep] == CLASS_NAMES.index('Shirt')).mean()*100:.0f}%" if (~keep).any() else "–", "of those are Shirts", "the hardest class", ORANGE)
            o = np.argsort(-conf); corr = (pred_fin == y)[o]; kk = np.arange(1, len(y) + 1)
            cov, acc = kk / len(y) * 100, np.cumsum(corr) / kk * 100
            fig = go.Figure(go.Scatter(x=cov[::20], y=acc[::20], line=dict(color=VIOLET, width=3), name="Accuracy vs coverage"))
            fig.add_trace(go.Scatter(x=[keep.mean() * 100], y=[(pred_fin[keep] == y[keep]).mean() * 100], mode="markers",
                                     marker=dict(size=14, color=ORANGE), name="Your threshold"))
            fig.update_xaxes(range=[40, 100.5], ticksuffix="%"); fig.update_yaxes(range=[92, 100.2], ticksuffix="%")
            st.plotly_chart(style(fig, "Coverage: % of test images the model answers (most confident first)", "Accuracy on the answered images (%)", 420), width="stretch")
            axis_note("moving left = the model answers fewer images but is right more often.")


# =============================================================================
# PAGE 8: DESIGN DECISIONS
# =============================================================================
elif page == PAGES[7]:
    header([("Architecture audit", "badge-blue")], "Design Decisions",
           "Why these models, and why popular alternatives were not used as the main model.")
    m = pd.DataFrame({
        "Architecture": ["Final ensemble (ResNetSmall / CNN3Block-GAP)", "ResNetSmall", "CNN3Block-GAP", "CNN2Block", "Pre-trained ResNet-18/50",
                         "Vision Transformer (ViT-Tiny)", "Deep MLP", "Very deep plain CNN"],
        "Status": ["✅ Final model", "✅ Evaluated (3 seeds)", "✅ Evaluated (ablation + 3 seeds)", "✅ Evaluated (baseline)", "❌ Not used", "❌ Not used", "❌ Not used", "❌ Not used"],
        "Main reason": ["Best validation accuracy; calibrated; abstain option", "Residual blocks allow depth on 28×28", "Compact, 60× smaller than SVM",
                        "Reference point for the ablation", "Built for 224×224 colour photos; 11–25M weights", "Needs far more data; no locality bias",
                        "Flattening throws away 2D structure (r ≈ 0.75)", "Vanishing gradients without shortcuts"],
    })
    st.dataframe(m, width="stretch", hide_index=True)
    st.caption("Costs for models we did not train are not measured here and are therefore not shown.")
    a, b = st.columns(2)
    with a:
        st.markdown("""<div class="section-card"><h3>Pre-trained ImageNet models</h3><ul>
          <li>Expect 224×224 RGB photos; upsampling 28×28 by 8× adds blur, not information.</li>
          <li>11–25 million weights for 48,000 tiny images.</li>
          <li>ImageNet features (fur, sky, texture) do not match centred grey garments.</li></ul></div>""", unsafe_allow_html=True)
        st.markdown("""<div class="section-card"><h3>Deep MLPs</h3><ul>
          <li>Flattening ignores that neighbouring pixels belong together.</li>
          <li>A 1-pixel shift looks like a completely different input.</li></ul></div>""", unsafe_allow_html=True)
    with b:
        st.markdown("""<div class="section-card"><h3>Vision Transformers</h3><ul>
          <li>No built-in notion of locality; must learn it from data.</li>
          <li>Data-hungry; overfit on 48,000 images without huge pre-training.</li></ul></div>""", unsafe_allow_html=True)
        st.markdown("""<div class="section-card"><h3>Why ResNetSmall made the cut</h3><ul>
          <li>Residual shortcuts fix the vanishing-gradient problem of very deep plain CNNs.</li>
          <li>Stays small (≈0.7M weights) and keeps 28×28 resolution in the first stage.</li></ul></div>""", unsafe_allow_html=True)


# =============================================================================
# PAGE 9: LEADERBOARD & DELIVERABLES
# =============================================================================
elif page == PAGES[8]:
    header([("All models", "badge-green"), ("Deliverables", "badge-blue")], "Leaderboard & Deliverables",
           "Every model from both phases, plus the report, slides and code to reproduce them.")
    t1, t2, t3 = st.tabs(["Leaderboard", "Accuracy vs cost", "Downloads"])
    lb = get_table("p2_final_leaderboard.csv")
    if lb is None:
        lb = get_table("all_phases_model_results.csv")
    with t1:
        d = lb.copy()
        d["Model"] = d.model.map(nice_name)
        d["Test accuracy"] = (d.accuracy * 100).map(lambda v: f"{v:.2f}%")
        d["Macro-F1"] = d.macro_f1.map(lambda v: f"{v:.4f}")
        d["ms / image"] = d.infer_ms.map(lambda v: f"{v:.3f}")
        d["Stored numbers"] = d.params.map(lambda v: f"{v:,.0f}" if v < 1e6 else f"{v/1e6:.1f}M")
        st.dataframe(d[["phase", "Model", "features", "Test accuracy", "Macro-F1", "ms / image", "Stored numbers"]]
                     .rename(columns={"phase": "Phase", "features": "Features"}), width="stretch", hide_index=True, height=560)
        st.download_button("📥 Download leaderboard (CSV)", lb.to_csv(index=False), "fashion_mnist_leaderboard.csv", "text/csv")
    with t2:
        d = lb.copy()
        d["Phase"] = d.phase.str.contains("2").map({True: "Phase 2 (CNN)", False: "Phase 1 (classical)"})
        d["label"] = d.model.map(nice_name) + " · " + d.features.astype(str)
        fig = px.scatter(d, x="infer_ms", y=d.accuracy * 100, color="Phase", log_x=True, hover_name="label",
                         color_discrete_map={"Phase 2 (CNN)": VIOLET, "Phase 1 (classical)": ORANGE})
        fig.update_traces(marker=dict(size=12, line=dict(color="white", width=1)))
        fig.update_yaxes(ticksuffix="%")
        st.plotly_chart(style(fig, "CPU time to classify one image (ms, log scale — each grid line is 10×)", "Test accuracy (%)", 480), width="stretch")
        axis_note("top-left is best: accurate and fast. Hover a dot to see the model.")
    with t3:
        files = [("Phase 2 presentation (PowerPoint)", ROOT / "docs" / "Fashion-MNIST_Phase2_Presentation.pptx",
                  "application/vnd.openxmlformats-officedocument.presentationml.presentation"),
                 ("Phase 2 presentation (PDF)", ROOT / "docs" / "Fashion-MNIST_Phase2_Presentation.pdf", "application/pdf"),
                 ("Phase 2 full report (PDF)", ROOT / "report" / "phase2_full_report.pdf", "application/pdf"),
                 ("Phase 1 presentation (PowerPoint)", ROOT / "docs" / "Fashion-MNIST_Phase1_Presentation.pptx",
                  "application/vnd.openxmlformats-officedocument.presentationml.presentation"),
                 ("Phase 1 report (PDF)", ROOT / "report" / "main.pdf", "application/pdf")]
        cols = st.columns(2)
        for i, (lab, p, mime) in enumerate(files):
            if p.exists():
                cols[i % 2].download_button(f"📄 {lab}", p.read_bytes(), p.name, mime, width="stretch")
        st.markdown("""<div class="section-card" style="margin-top:12px;"><h3>Reproduce everything</h3><ul>
          <li><code>scripts/05_train_cnn.py</code> — first three CNNs</li>
          <li><code>scripts/07_ablation_gap_noaug.py</code> — the 2×2 ablation control run</li>
          <li><code>scripts/08_phase2_story_figures.py</code> — CIs, confusion, report figures</li>
          <li><code>scripts/09_train_final_models.py</code> — 40-epoch, 3-seed CNN3Block-GAP and ResNetSmall</li>
          <li><code>scripts/10_ensemble_calibration.py</code> — ensemble, TTA, calibration, abstain, Shirt rule</li>
          <li><code>scripts/build_phase2_deck.js</code> — rebuilds the slides from the result files</li></ul></div>""", unsafe_allow_html=True)
