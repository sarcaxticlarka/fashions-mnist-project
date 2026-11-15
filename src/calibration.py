"""Post-hoc probability calibration and reliability evaluation routines.

Provides:
- compute_ece: Expected Calibration Error (ECE) with reliability diagram binning.
- compute_brier_score: Multi-class Brier probability score.
- compute_nll: Multi-class negative log-likelihood.
- TemperatureScaler: Fits optimal temperature T via NLL minimization on validation logits.
"""
from typing import Tuple, List, Optional
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax, softmax


def compute_nll(log_probs: np.ndarray, targets: np.ndarray) -> float:
    """Negative log likelihood given log probabilities and class indices."""
    return float(-log_probs[np.arange(len(targets)), targets].mean())


def compute_ece(
    probs: np.ndarray,
    targets: np.ndarray,
    n_bins: int = 15,
) -> Tuple[float, List[Tuple[float, float, float, float, int]]]:
    """Compute Expected Calibration Error (ECE) and reliability bin statistics.

    Returns:
        ece_value: float Expected Calibration Error in [0, 1].
        bin_stats: List of tuples (bin_lo, bin_hi, avg_confidence, avg_accuracy, count).
    """
    conf = np.max(probs, axis=1)
    preds = np.argmax(probs, axis=1)
    edges = np.linspace(0.0, 1.0, n_bins + 1)

    ece_val = 0.0
    bin_stats = []
    n = len(targets)
    if n == 0:
        return 0.0, []

    for lo, hi in zip(edges[:-1], edges[1:]):
        if lo == 0.0:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf > lo) & (conf <= hi)

        count = int(mask.sum())
        if count > 0:
            acc = float((preds[mask] == targets[mask]).mean())
            c = float(conf[mask].mean())
            ece_val += (count / n) * abs(acc - c)
            bin_stats.append((float(lo), float(hi), c, acc, count))

    return float(ece_val), bin_stats


def compute_brier_score(probs: np.ndarray, targets: np.ndarray) -> float:
    """Compute multi-class Brier probability verification score.

    Brier = mean sum_c (p_ic - y_ic)^2
    Lower is better (0.0 = perfect prediction with 100% confidence).
    """
    n, _ = probs.shape
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(n), targets] = 1.0
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


class TemperatureScaler:
    """Learns a single scalar temperature T > 0 on validation logits to calibrate probabilities.

    Calibrated probabilities: p_i = softmax(logits_i / T)
    - T > 1 softens overconfident predictions.
    - T < 1 sharpens underconfident predictions.
    - T = 1 preserves original softmax probabilities.
    """

    def __init__(self, temperature: float = 1.0):
        self.temperature = float(temperature)

    def fit(
        self,
        val_logits: np.ndarray,
        val_targets: np.ndarray,
        bounds: Tuple[float, float] = (0.05, 5.0),
    ) -> "TemperatureScaler":
        """Find optimal temperature T minimizing NLL on validation logits."""
        def objective(t: float) -> float:
            lp = log_softmax(val_logits / t, axis=1)
            return compute_nll(lp, val_targets)

        res = minimize_scalar(objective, bounds=bounds, method="bounded")
        self.temperature = float(res.x)
        return self

    def transform(self, logits: np.ndarray) -> np.ndarray:
        """Apply learned temperature scaling and return calibrated probabilities."""
        return softmax(logits / self.temperature, axis=1)

    def fit_transform(self, val_logits: np.ndarray, val_targets: np.ndarray) -> np.ndarray:
        """Fit temperature and return calibrated probabilities on the input logits."""
        self.fit(val_logits, val_targets)
        return self.transform(val_logits)
