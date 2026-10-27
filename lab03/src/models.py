import torch
import torch.nn as nn

class FashionMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    def forward(self, x):
        return self.net(x)

class FashionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten()
        )
        self.classifier = nn.Linear(32 * 7 * 7, 10)
    def forward(self, x):
        return self.classifier(self.features(x))

class FashionAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_proj = nn.Linear(49, 64)
        encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=128, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(64, 10)
    def forward(self, x):
        B = x.shape[0]
        patches = x.unfold(2, 7, 7).unfold(3, 7, 7)
        patches = patches.contiguous().view(B, 16, 49)
        tokens = self.patch_proj(patches)
        enc = self.transformer(tokens)
        pooled = enc.mean(dim=1)
        return self.fc(pooled)
