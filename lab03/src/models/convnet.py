import torch
import torch.nn as nn

class TughanConvNet(nn.Module):
    """
    Hierarchical 2D Convolutional Vision Classifier
    Input: (B, 1, 28, 28)
    Stage 1: Conv2d(1, 16, k=3, p=1) -> ReLU -> MaxPool2d(2, 2)
    Stage 2: Conv2d(16, 32, k=3, p=1) -> ReLU -> MaxPool2d(2, 2)
    Classifier: Linear(32 * 7 * 7, 10)
    """
    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten()
        )
        self.classifier = nn.Linear(32 * 7 * 7, num_classes)

    def forward(self, x):
        return self.classifier(self.features(x))
