"""Central configuration: paths, class names, seeds and shared constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
TABLE_DIR = RESULTS_DIR / "tables"
MODEL_DIR = RESULTS_DIR / "models"
RUN_DIR = TABLE_DIR / "runs"

for _d in (PROCESSED_DIR, FIG_DIR, TABLE_DIR, MODEL_DIR, RUN_DIR):
    _d.mkdir(parents=True, exist_ok=True)

SEED = 42
VAL_SIZE = 0.2  # fraction of the *training* set held out for validation

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]
NUM_CLASSES = len(CLASS_NAMES)
# The four upper-body garments that are expected to be confused with each other.
UPPER_BODY = ["T-shirt/top", "Pullover", "Coat", "Shirt"]

IMG_SIZE = 28  # Fashion-MNIST images are 28x28 grayscale

# Final HOG configuration (chosen by the ablation in 02_feature_engineering).
HOG_PARAMS = dict(orientations=6, pixels_per_cell=(4, 4), cells_per_block=(2, 2))
