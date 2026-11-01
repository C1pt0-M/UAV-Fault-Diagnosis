"""Independent M1: untouched26-channel MSTCA plus lightweight residual adapter."""
from __future__ import annotations
import torch
from torch import nn
from .lstm import OnlineMSTCA


class OnlineMSTCAKinematicM1(nn.Module):
    """N1..N4 aligned to the same150 timestamps as the26-channel base.

    No raw30-channel concatenation, old relation encoder or KG gate. A3168
    parameter temporal adapter adds once to the128-dimensional pooled base.
    Zero-initialized output projection preserves the exact initial M0 function.
    """
    def __init__(self, hidden_size=64, dropout=.2):
        super().__init__()
        if hidden_size != 64:
            raise ValueError('M1v1 uses frozen hidden64 only')
        base = OnlineMSTCA(in_channels=26, hidden_size=hidden_size, num_layers=1, num_classes=2, dropout=dropout)
        self.encoder = base.encoder
        self.head = base.head
        self.residual_adapter = nn.Sequential(
            nn.Conv1d(4, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(16, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(16, self.encoder.out_dim))
        nn.init.zeros_(self.residual_adapter[-1].weight)
        nn.init.zeros_(self.residual_adapter[-1].bias)
        self.model_version = 'mstca_kinematic_continuous200ms_m1_v1'

    def load_common_initial(self, state):
        target = self.state_dict()
        common = {k: v for k, v in target.items() if not k.startswith('residual_adapter.')}
        if set(state) != set(common):
            raise ValueError('Exact pure M0 untrained common state required')
        for key, value in state.items():
            if value.shape != common[key].shape or value.dtype != common[key].dtype or not torch.isfinite(value).all():
                raise ValueError('Common tensor layout/finite mismatch:' + key)
            target[key] = value.detach().clone()
        self.load_state_dict(target, strict=True)
        if not all(torch.equal(self.state_dict()[k].cpu(), v.cpu()) for k, v in state.items()):
            raise ValueError('Common initialization not loaded exactly')

    def forward(self, x, residual):
        if x.ndim != 3 or residual.ndim != 3 or x.shape[1:] != (26, 150) or residual.shape[1:] != (4, 150) or len(x) != len(residual):
            raise ValueError('Aligned26x150 and4x150 inputs required')
        if not torch.isfinite(x).all() or not torch.isfinite(residual).all():
            raise ValueError('Invalid input must be filtered before M1/BatchNorm')
        pooled = self.encoder(x)
        addition = self.residual_adapter(residual)
        return self.head(pooled + addition)
