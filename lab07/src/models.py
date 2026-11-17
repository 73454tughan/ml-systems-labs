"""
Model Architectures for Model Compression & Distillation
Author: Tughan Bulut Kurtulus (Student ID: 73454)
Course: Machine Learning Systems (Lab 07)
Institution: Vistula University
Curriculum Reference: Vijay Janapa Reddi, Machine Learning Systems (Chapters 9 & 10)
"""

import torch
import torch.nn as nn
from typing import Dict, Any


class OversizedBaselineCNN(nn.Module):
    """
    Oversized Teacher CNN for Fashion-MNIST classification.
    Target Parameter Count: ~1.0M parameters.
    Architecture:
      Conv1 (1 -> 32, 3x3) -> BatchNorm -> ReLU -> MaxPool (14x14)
      Conv2 (32 -> 64, 3x3) -> BatchNorm -> ReLU -> MaxPool (7x7)
      Conv3 (64 -> 96, 3x3) -> BatchNorm -> ReLU (7x7)
      Classifier: Flatten (4704) -> Linear(4704 -> 200) -> ReLU -> Dropout(0.2) -> Linear(200 -> 10)
    """
    def __init__(self, num_classes: int = 10, dropout_rate: float = 0.2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 28x28 -> 14x14

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 14x14 -> 7x7

            nn.Conv2d(64, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),  # 7x7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96 * 7 * 7, 200),  # 4704 * 200 = 940,800 + 200 = 941,000 params
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(200, num_classes)  # 200 * 10 + 10 = 2,010 params
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.features(x)
        logits = self.classifier(feats)
        return logits


class StudentCompactCNN(nn.Module):
    """
    Compact Student CNN for Knowledge Distillation.
    Target Parameter Count: ~100k parameters (~10x compression factor).
    Architecture:
      Conv1 (1 -> 16, 3x3) -> BatchNorm -> ReLU -> MaxPool (14x14)
      Conv2 (16 -> 32, 3x3) -> BatchNorm -> ReLU -> MaxPool (7x7)
      Conv3 (32 -> 32, 3x3) -> BatchNorm -> ReLU (7x7)
      Classifier: Flatten (1568) -> Linear(1568 -> 60) -> ReLU -> Linear(60 -> 10)
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 14x14

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 7x7

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),  # 7x7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 60),  # 1568 * 60 + 60 = 94,140 params
            nn.ReLU(inplace=True),
            nn.Linear(60, num_classes)   # 60 * 10 + 10 = 610 params
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.features(x)
        logits = self.classifier(feats)
        return logits


def count_parameters(model: nn.Module) -> Dict[str, Any]:
    """Returns total, trainable, and layer-by-layer parameter statistics."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    layer_breakdown = {name: p.numel() for name, p in model.named_parameters()}
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'layer_breakdown': layer_breakdown
    }


if __name__ == '__main__':
    teacher = OversizedBaselineCNN()
    student = StudentCompactCNN()
    t_stats = count_parameters(teacher)
    s_stats = count_parameters(student)
    print(f"Teacher Parameters: {t_stats['total_parameters']:,} (~{t_stats['total_parameters']/1e6:.2f}M)")
    print(f"Student Parameters: {s_stats['total_parameters']:,} (~{s_stats['total_parameters']/1e3:.1f}k)")
    print(f"Compression Ratio:  {t_stats['total_parameters'] / s_stats['total_parameters']:.2f}x")
