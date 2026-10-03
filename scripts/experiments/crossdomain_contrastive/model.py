"""Causal TCN encoder + projection/classification heads."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    """Conv1d with causal (left) padding."""
    def __init__(self, in_ch, out_ch, kernel_size, dilation=1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation)

    def forward(self, x):
        x = F.pad(x, (self.pad, 0))
        return self.conv(x)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention."""
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (B, C, T)
        w = x.mean(dim=2)  # (B, C)
        w = self.fc(w).unsqueeze(2)  # (B, C, 1)
        return x * w


class CausalTCNEncoder(nn.Module):
    """Lightweight causal TCN with channel attention."""
    def __init__(self, in_channels=26, hidden=64, out_dim=128):
        super().__init__()
        self.stage1 = nn.Sequential(
            CausalConv1d(in_channels, hidden, kernel_size=7),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
        )
        self.stage2 = nn.Sequential(
            CausalConv1d(hidden, hidden, kernel_size=5, dilation=2),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
        )
        self.stage3 = nn.Sequential(
            CausalConv1d(hidden, out_dim, kernel_size=3, dilation=4),
            nn.BatchNorm1d(out_dim),
            nn.GELU(),
        )
        self.se = SEBlock(out_dim)

    def forward(self, x):
        # x: (B, T, C) -> transpose to (B, C, T)
        x = x.transpose(1, 2)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.se(x)
        # Global average pooling over time
        h = x.mean(dim=2)  # (B, out_dim)
        return h


class ProjectionHead(nn.Module):
    """MLP projection head for contrastive learning."""
    def __init__(self, in_dim=128, hidden=64, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, h):
        z = self.net(h)
        return F.normalize(z, dim=1)


class ClassificationHead(nn.Module):
    """Simple linear classifier."""
    def __init__(self, in_dim=128, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.GELU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, h):
        return self.net(h)


class CrossDomainContrastiveModel(nn.Module):
    """Full model: encoder + projection + classification."""
    def __init__(self, in_channels=26, num_classes=3):
        super().__init__()
        self.encoder = CausalTCNEncoder(in_channels=in_channels)
        self.projector = ProjectionHead()
        self.classifier = ClassificationHead(num_classes=num_classes)

    def forward(self, x, return_proj=False):
        h = self.encoder(x)
        logits = self.classifier(h)
        if return_proj:
            z = self.projector(h)
            return logits, z
        return logits
