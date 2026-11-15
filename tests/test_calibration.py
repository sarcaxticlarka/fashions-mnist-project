"""Unit tests for the model calibration and reliability module."""
import unittest
import numpy as np
from scipy.special import softmax

from src.calibration import compute_ece, compute_brier_score, compute_nll, TemperatureScaler


class TestCalibration(unittest.TestCase):
    """Test suite for probability calibration metrics and TemperatureScaler."""

    def test_perfect_calibration_ece_is_zero(self):
        """A classifier whose confidence exactly matches accuracy has near-zero ECE."""
        rng = np.random.default_rng(42)
        n = 1000
        # 10 classes, uniform 0.1 confidence for all classes -> top confidence = 0.1
        # Accuracy also ~0.1
        targets = rng.integers(0, 10, size=n)
        probs = np.full((n, 10), 0.1)

        ece_val, bin_stats = compute_ece(probs, targets, n_bins=10)
        self.assertLess(ece_val, 0.05)
        self.assertTrue(len(bin_stats) > 0)

    def test_overconfident_ece_is_high(self):
        """Overconfident wrong predictions should produce high ECE."""
        n = 500
        # Predictions claim 99% confidence on class 0, but true labels are evenly split between 0 and 1
        targets = np.array([0] * 250 + [1] * 250)
        probs = np.full((n, 10), 0.001)
        probs[:, 0] = 0.991  # 50% accuracy but 99.1% confidence

        ece_val, bin_stats = compute_ece(probs, targets, n_bins=10)
        # Expected gap |0.50 - 0.991| ~ 0.491
        self.assertGreater(ece_val, 0.45)

    def test_brier_score_properties(self):
        """Verify Brier score boundaries and exact values."""
        targets = np.array([0, 1, 2])

        # Perfect predictions
        perfect_probs = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        self.assertAlmostEqual(compute_brier_score(perfect_probs, targets), 0.0)

        # Worst possible predictions
        worst_probs = np.array([
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ])
        self.assertAlmostEqual(compute_brier_score(worst_probs, targets), 2.0)

        # Uniform predictions on 10 classes
        n = 100
        uni_targets = np.zeros(n, dtype=int)
        uni_probs = np.full((n, 10), 0.1)
        # Brier = (1 - 0.1)^2 + 9 * (0.1)^2 = 0.81 + 0.09 = 0.90
        self.assertAlmostEqual(compute_brier_score(uni_probs, uni_targets), 0.90)

    def test_temperature_scaler_optimizes_logits(self):
        """TemperatureScaler should optimize temperature T to minimize NLL and reduce ECE."""
        rng = np.random.default_rng(42)
        n = 2000
        num_classes = 10
        targets = rng.integers(0, num_classes, size=n)

        # Create artificially overconfident logits (scaled by 3.0)
        true_logits = np.zeros((n, num_classes))
        true_logits[np.arange(n), targets] = 1.5
        true_logits += rng.normal(0, 1.0, size=(n, num_classes))
        overconfident_logits = true_logits * 3.0

        uncalibrated_probs = softmax(overconfident_logits, axis=1)
        ece_before, _ = compute_ece(uncalibrated_probs, targets)

        scaler = TemperatureScaler()
        self.assertEqual(scaler.temperature, 1.0)

        scaler.fit(overconfident_logits, targets)
        # Overconfident logits should yield T > 1.0 to soften them
        self.assertGreater(scaler.temperature, 1.5)

        calibrated_probs = scaler.transform(overconfident_logits)
        self.assertEqual(calibrated_probs.shape, (n, num_classes))
        np.testing.assert_allclose(calibrated_probs.sum(axis=1), np.ones(n), atol=1e-5)

        ece_after, _ = compute_ece(calibrated_probs, targets)
        self.assertLess(ece_after, ece_before)

    def test_temperature_scaler_fit_transform(self):
        """fit_transform convenience method should match fit followed by transform."""
        logits = np.array([[2.0, 0.5, 0.1], [0.1, 2.5, 0.2]])
        targets = np.array([0, 1])

        scaler1 = TemperatureScaler()
        out1 = scaler1.fit_transform(logits, targets)

        scaler2 = TemperatureScaler()
        scaler2.fit(logits, targets)
        out2 = scaler2.transform(logits)

        self.assertAlmostEqual(scaler1.temperature, scaler2.temperature)
        np.testing.assert_allclose(out1, out2)


if __name__ == "__main__":
    unittest.main()
