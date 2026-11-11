"""Grad-CAM (Gradient-weighted Class Activation Mapping) for CNN explainability on Fashion-MNIST."""
import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    """Computes Grad-CAM heatmaps for a CNN model and a specified target layer."""
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        self._hook_forward = self.target_layer.register_forward_hook(self._forward_hook)
        self._hook_backward = self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def remove_hooks(self):
        self._hook_forward.remove()
        self._hook_backward.remove()

    def generate(self, input_tensor: torch.Tensor, target_class: int = None) -> np.ndarray:
        """Generate Grad-CAM heatmap for a single input tensor of shape (1, 1, 28, 28)."""
        self.model.eval()
        self.model.zero_grad()

        output = self.model(input_tensor)
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        score = output[0, target_class]
        score.backward()

        # gradients: (1, C, H, W), activations: (1, C, H, W)
        grads = self.gradients[0].detach()
        acts = self.activations[0].detach()

        # Global average pooling of gradients per channel
        weights = grads.mean(dim=(1, 2), keepdim=True)
        cam = (weights * acts).sum(dim=0)
        cam = F.relu(cam)

        cam_np = cam.cpu().numpy()
        cam_min, cam_max = cam_np.min(), cam_np.max()
        if cam_max > cam_min:
            cam_np = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_np = np.zeros_like(cam_np)

        # Resize to input resolution (28, 28)
        cam_tensor = torch.from_numpy(cam_np).unsqueeze(0).unsqueeze(0)
        resized_cam = F.interpolate(cam_tensor, size=(28, 28), mode="bilinear", align_corners=False)
        return resized_cam.squeeze().numpy()

