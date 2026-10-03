"""
CNN-LSTM v2 with normal prototype calibration support.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .cnn_lstm_v2 import CNNLSTMv2Classifier


class CNNLSTMv2NormalProtoClassifier(nn.Module):
    def __init__(
        self,
        in_channels: int = 26,
        num_classes: int = 3,
        target_steps: int = 512,
        raw_proj_dim: int = 32,
        branch_channels: int = 32,
        cnn_out_channels: int = 128,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.backbone = CNNLSTMv2Classifier(
            in_channels=in_channels,
            num_classes=num_classes,
            target_steps=target_steps,
            raw_proj_dim=raw_proj_dim,
            branch_channels=branch_channels,
            cnn_out_channels=cnn_out_channels,
            lstm_hidden=lstm_hidden,
            lstm_layers=lstm_layers,
            dropout=dropout,
        )
        feature_dim = lstm_hidden * 2
        self.normal_center = nn.Parameter(torch.zeros(feature_dim))

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone.forward_features(x)

    def classify_features(self, features: torch.Tensor) -> torch.Tensor:
        return self.backbone.classify_features(features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(x)
        return self.classify_features(features)
