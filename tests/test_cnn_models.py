"""Unit tests for Phase 2 deep learning models and Grad-CAM explainability."""
import unittest
import torch
import torch.nn as nn
import numpy as np

from src.cnn_models import CNN2Block, CNN3BlockGAP, ResNetSmall, get_model, count_parameters
from src.cnn_explain import GradCAM


class TestCNNArchitectures(unittest.TestCase):
    """Test suite for Phase 2 CNN architectures."""

    def test_model_instantiation_and_parameter_counts(self):
        """Verify model instantiation and parameter counts against documented budgets."""
        m_2block = get_model("CNN2Block")
        m_gap = get_model("CNN3Block_GAP")
        m_resnet = get_model("ResNetSmall")

        self.assertIsInstance(m_2block, CNN2Block)
        self.assertIsInstance(m_gap, CNN3BlockGAP)
        self.assertIsInstance(m_resnet, ResNetSmall)

        # Check parameter counts match the documented figures
        self.assertEqual(count_parameters(m_2block), 421738)
        self.assertEqual(count_parameters(m_gap), 304682)
        self.assertEqual(count_parameters(m_resnet), 696042)

        # Check alias in get_model
        m_2block_aug = get_model("CNN2Block_Aug")
        self.assertIsInstance(m_2block_aug, CNN2Block)

        # Check unknown model raises ValueError
        with self.assertRaises(ValueError):
            get_model("NonExistentModel")

    def test_forward_pass_output_shapes(self):
        """Verify forward pass output dimensions across single-item and mini-batch inputs."""
        models = [
            get_model("CNN2Block"),
            get_model("CNN3Block_GAP"),
            get_model("ResNetSmall"),
        ]

        batch_sizes = [1, 4]
        for model in models:
            model.eval()
            for b in batch_sizes:
                x = torch.randn(b, 1, 28, 28)
                with torch.no_grad():
                    out = model(x)
                self.assertEqual(out.shape, (b, 10), f"Failed for {type(model).__name__} with batch {b}")
                self.assertTrue(torch.isfinite(out).all(), f"Non-finite output in {type(model).__name__}")

    def test_get_last_conv_layer(self):
        """Ensure get_last_conv_layer returns a Conv2d module for each model family."""
        models = [
            get_model("CNN2Block"),
            get_model("CNN3Block_GAP"),
            get_model("ResNetSmall"),
        ]
        for model in models:
            conv = model.get_last_conv_layer()
            self.assertIsInstance(conv, nn.Conv2d, f"Last conv in {type(model).__name__} is not Conv2d")


class TestGradCAM(unittest.TestCase):
    """Test suite for Grad-CAM explainability hooks and saliency map generation."""

    def setUp(self):
        self.model = get_model("CNN3Block_GAP")
        self.target_layer = self.model.get_last_conv_layer()
        self.grad_cam = GradCAM(self.model, self.target_layer)

    def tearDown(self):
        self.grad_cam.remove_hooks()

    def test_generate_heatmap_shape_and_range(self):
        """Grad-CAM heatmap should be 28x28 with values scaled to [0, 1]."""
        x = torch.rand(1, 1, 28, 28, requires_grad=True)
        cam = self.grad_cam.generate(x)

        self.assertIsInstance(cam, np.ndarray)
        self.assertEqual(cam.shape, (28, 28))
        self.assertGreaterEqual(float(cam.min()), 0.0)
        self.assertLessEqual(float(cam.max()), 1.0)

    def test_generate_explicit_target_class(self):
        """Grad-CAM should run for a specified class index."""
        x = torch.rand(1, 1, 28, 28, requires_grad=True)
        # Class 6 is 'Shirt'
        cam = self.grad_cam.generate(x, target_class=6)
        self.assertEqual(cam.shape, (28, 28))
        self.assertGreaterEqual(float(cam.min()), 0.0)
        self.assertLessEqual(float(cam.max()), 1.0)

    def test_hook_removal(self):
        """remove_hooks should cleanly unregister PyTorch forward and backward hooks."""
        cam_temp = GradCAM(self.model, self.target_layer)
        self.assertIsNotNone(cam_temp._hook_forward)
        self.assertIsNotNone(cam_temp._hook_backward)

        cam_temp.remove_hooks()
        # Verify handles are removed by checking hook handle dicts
        self.assertEqual(len(cam_temp.target_layer._forward_hooks), 1)  # self.grad_cam remains


if __name__ == "__main__":
    unittest.main()
