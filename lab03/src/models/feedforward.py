import torch
import torch.nn as nn

class TughanDensePerceptron(nn.Module):
    """
    3-Layer Multilayer Perceptron with Dropout Regularization
    784 -> Linear(256) -> ReLU -> Dropout(0.2) -> Linear(128) -> ReLU -> Linear(10)
    """
    def __init__(self, in_features=784, h1=256, h2=128, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, h1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(h1, h2),
            nn.ReLU(inplace=True),
            nn.Linear(h2, num_classes)
        )

    def forward(self, x):
        return self.net(x)
