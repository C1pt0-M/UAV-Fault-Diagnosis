"""
BiLSTM with normal prototype calibration support.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class BiLSTMNormalProtoClassifier(nn.Module):
    def __init__(
        self,
        in_channels: int = 31,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        feature_dim = hidden_size * 2
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_classes),
        )
        self.normal_center = nn.Parameter(torch.zeros(feature_dim))
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)  # (N, T, C)
        x = x[:, ::4, :]
        out, _ = self.lstm(x)
        attn_weights = self.attention(out)
        attn_weights = torch.softmax(attn_weights, dim=1)
        return (out * attn_weights).sum(dim=1)

    def classify_features(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(x)
        return self.classify_features(features)
