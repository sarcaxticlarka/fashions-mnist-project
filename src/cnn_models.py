"""Convolutional Neural Network architectures for Fashion-MNIST (Phase 2).

Architectures:
1. CNN2Block: Standard 2-stage ConvNet (Conv-BN-ReLU-Pool x 2 + Dense + Dropout).
   Replaces fixed HOG features with end-to-end learned spatial filters.
2. CNN3BlockGAP: Deep 3-stage ConvNet with Global Average Pooling (GAP) and double-conv
   feature extractors. Replaces large fully connected layers with GAP to prevent overfitting
   and maintain compact parameter count while expanding receptive field.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN2Block(nn.Module):
    """2-Block Baseline CNN.
    
    Structure:
      Input: (B, 1, 28, 28)
      Block 1: Conv2d(1, 32, 3, pad=1) -> BatchNorm -> ReLU -> MaxPool2d(2) -> (B, 32, 14, 14)
      Block 2: Conv2d(32, 64, 3, pad=1) -> BatchNorm -> ReLU -> MaxPool2d(2) -> (B, 64, 7, 7)
      Classifier: Flatten -> Linear(64*7*7, 128) -> ReLU -> Dropout(0.4) -> Linear(128, 10)
    """
    def __init__(self, num_classes=10, dropout=0.4):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.classifier(x)
        return x

    def get_last_conv_layer(self):
        """Returns the last convolutional layer for Grad-CAM."""
        return self.block2[0]


class CNN3BlockGAP(nn.Module):
    """Deep 3-Block CNN with Global Average Pooling (GAP).
    
    Structure:
      Input: (B, 1, 28, 28)
      Block 1: [Conv(1->32, 3, pad=1) -> BN -> ReLU -> Conv(32->32, 3, pad=1) -> BN -> ReLU] -> Pool -> (B, 32, 14, 14)
      Block 2: [Conv(32->64, 3, pad=1) -> BN -> ReLU -> Conv(64->64, 3, pad=1) -> BN -> ReLU] -> Pool -> (B, 64, 7, 7)
      Block 3: [Conv(64->128, 3, pad=1) -> BN -> ReLU -> Conv(128->128, 3, pad=1) -> BN -> ReLU] -> (B, 128, 7, 7)
      GAP: AdaptiveAvgPool2d((1, 1)) -> (B, 128, 1, 1)
      Classifier: Flatten -> Linear(128, 128) -> ReLU -> Dropout(0.3) -> Linear(128, 10)
    """
    def __init__(self, num_classes=10, dropout=0.3):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x

    def get_last_conv_layer(self):
        """Returns the last convolutional layer for Grad-CAM."""
        return self.block3[3]


class BasicBlock(nn.Module):
    """ResNet basic block: two 3x3 convs with an identity (or 1x1 projection) shortcut."""
    def __init__(self, c_in, c_out, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.short = nn.Identity()
        if stride != 1 or c_in != c_out:
            self.short = nn.Sequential(nn.Conv2d(c_in, c_out, 1, stride=stride, bias=False), nn.BatchNorm2d(c_out))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.short(x))


class ResNetSmall(nn.Module):
    """Compact residual network for 28x28 inputs (final Phase 2 model family).

    Stem: Conv(1->32) -> BN -> ReLU                      (32x28x28)
    Stage 1: 2 x BasicBlock(32)                          (32x28x28)
    Stage 2: 2 x BasicBlock(64),  first with stride 2   (64x14x14)
    Stage 3: 2 x BasicBlock(128), first with stride 2   (128x7x7)
    GAP -> Dropout(0.2) -> Linear(128, 10)
    Residual shortcuts let the network go deeper (13 conv layers) without vanishing gradients.
    """
    def __init__(self, num_classes=10, dropout=0.2, width=32):
        super().__init__()
        w = width
        self.stem = nn.Sequential(nn.Conv2d(1, w, 3, padding=1, bias=False), nn.BatchNorm2d(w), nn.ReLU(inplace=True))
        self.stage1 = nn.Sequential(BasicBlock(w, w), BasicBlock(w, w))
        self.stage2 = nn.Sequential(BasicBlock(w, 2 * w, stride=2), BasicBlock(2 * w, 2 * w))
        self.stage3 = nn.Sequential(BasicBlock(2 * w, 4 * w, stride=2), BasicBlock(4 * w, 4 * w))
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(4 * w, num_classes))

    def forward(self, x):
        x = self.stage3(self.stage2(self.stage1(self.stem(x))))
        return self.classifier(self.gap(x))

    def get_last_conv_layer(self):
        return self.stage3[-1].conv2


def get_model(name: str, num_classes: int = 10, **kwargs):
    """Model factory."""
    if name.startswith("ResNetSmall"):
        return ResNetSmall(num_classes=num_classes, **kwargs)
    if name in ("CNN2Block", "CNN2Block_Aug"):
        return CNN2Block(num_classes=num_classes, **kwargs)
    if name in ("CNN3Block_GAP", "CNN3BlockGAP") or name.startswith("CNN3Block_GAP"):
        return CNN3BlockGAP(num_classes=num_classes, **kwargs)
    raise ValueError(f"Unknown CNN model name: {name}")


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

