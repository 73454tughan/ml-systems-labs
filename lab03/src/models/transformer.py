import torch
import torch.nn as nn

class TughanTransformerClassifier(nn.Module):
    """
    Patch-Based Vision Transformer with Sequence Self-Attention
    Splits 28x28 images into 16 visual patches of 7x7
    """
    def __init__(self, in_channels=1, patch_size=7, embed_dim=64, num_heads=4, num_layers=2, num_classes=10):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (28 // patch_size) ** 2  # 16
        patch_dim = in_channels * patch_size * patch_size  # 49
        
        self.patch_proj = nn.Linear(patch_dim, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=128,
            activation="relu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.shape[0]
        p = self.patch_size
        patches = x.unfold(2, p, p).unfold(3, p, p)
        patches = patches.contiguous().view(B, self.num_patches, -1)
        tokens = self.patch_proj(patches) + self.pos_embed
        enc = self.transformer(tokens)
        pooled = enc.mean(dim=1)
        return self.fc(pooled)
