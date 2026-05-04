"""
CNN-LSTM v2 with domain-adversarial training support.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.autograd import Function

from .cnn_lstm_v2 import CNNLSTMv2Classifier


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return grad_output.neg() * ctx.alpha, None


class CNNLSTMv2DGClassifier(nn.Module):
    def __init__(
        self,
        in_channels: int = 31,
        num_classes: int = 3,
        num_domains: int = 3,
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
        self.domain_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feature_dim // 2, num_domains),
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone.forward_features(x)

    def forward(
        self,
        x: torch.Tensor,
        alpha: float = 0.0,
        return_domain: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        features = self.forward_features(x)
        class_logits = self.backbone.classify_features(features)
        if not return_domain:
            return class_logits
        rev_features = ReverseLayerF.apply(features, alpha)
        domain_logits = self.domain_head(rev_features)
        return class_logits, domain_logits
