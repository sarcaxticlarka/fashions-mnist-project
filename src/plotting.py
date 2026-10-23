"""Shared matplotlib style + helper to save figures."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from .config import FIG_DIR

PALETTE = sns.color_palette("tab10", 10)
COLORS = {"pixels": "#4C72B0", "hog": "#DD8452", "hog_stats": "#55A868"}


def set_style():
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 200, "axes.titleweight": "bold",
        "axes.titlesize": 12, "axes.labelsize": 10, "font.family": "sans-serif",
        "savefig.bbox": "tight",
    })


def save(fig, name, show=False):
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path)
    if not show:
        plt.close(fig)
    return path
