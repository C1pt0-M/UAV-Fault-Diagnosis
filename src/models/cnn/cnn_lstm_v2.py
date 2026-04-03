"""
CNN-LSTM v2 for transfer3.

Multi-scale CNN front-end + raw skip path + BiLSTM + attention pooling.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_norm(channels: int, preferred_groups: int = 8) -> nn.GroupNorm:
    groups = min(preferred_groups, channels)
    while groups > 1 and channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class _ConvNormAct1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                bias=False,
            ),
            _group_norm(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _ResidualConvBlock1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = _group_norm(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = _group_norm(out_channels)
        self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.shortcut_norm = _group_norm(out_channels)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut_norm(self.shortcut(x))
        residual = self.pool(residual)

        out = self.act(self.norm1(self.conv1(x)))
        out = self.pool(out)
        out = self.norm2(self.conv2(out))
        out = self.act(out + residual)
        return out


class CNNLSTMv2Classifier(nn.Module):
    """
    Multi-scale residual CNN-LSTM with raw skip path.

    Input:
      x: (N, C, T)
    Output:
      logits: (N, num_classes)
    """

    def __init__(
        self,
        in_channels: int = 31,
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
        self.target_steps = target_steps

        self.raw_projection = nn.Linear(in_channels, raw_proj_dim)

        self.branch_k3 = _ConvNormAct1d(in_channels, branch_channels, kernel_size=3)
        self.branch_k7 = _ConvNormAct1d(in_channels, branch_channels, kernel_size=7)
        self.branch_k15 = _ConvNormAct1d(in_channels, branch_channels, kernel_size=15)

        cnn_in_channels = branch_channels * 3  # 96
        self.fusion_norm = nn.GroupNorm(8, cnn_in_channels)
        self.residual_block = _ResidualConvBlock1d(cnn_in_channels, cnn_out_channels)
        self.post_pool = nn.MaxPool1d(kernel_size=2, stride=2)

        fusion_dim = raw_proj_dim + cnn_out_channels  # 160
        self.lstm = nn.LSTM(
            input_size=fusion_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        lstm_out_dim = lstm_hidden * 2
        self.attention = nn.Sequential(
            nn.Linear(lstm_out_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(lstm_out_dim),
            nn.Dropout(dropout),
            nn.Linear(lstm_out_dim, num_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _raw_skip_path(self, x: torch.Tensor) -> torch.Tensor:
        raw = F.interpolate(x, size=self.target_steps, mode="linear", align_corners=False)
        raw = raw.transpose(1, 2)  # (N, 512, C)
        return self.raw_projection(raw)  # (N, 512, 32)

    def _cnn_path(self, x: torch.Tensor) -> torch.Tensor:
        multi_scale = torch.cat(
            [
                self.branch_k3(x),
                self.branch_k7(x),
                self.branch_k15(x),
            ],
            dim=1,
        )
        multi_scale = self.fusion_norm(multi_scale)
        features = self.residual_block(multi_scale)
        features = self.post_pool(features)
        if features.size(-1) != self.target_steps:
            features = F.interpolate(
                features, size=self.target_steps, mode="linear", align_corners=False
            )
        return features.transpose(1, 2)  # (N, 512, 128)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        raw_skip = self._raw_skip_path(x)
        cnn_features = self._cnn_path(x)
        fused = torch.cat([raw_skip, cnn_features], dim=2)  # (N, 512, 160)

        lstm_out, _ = self.lstm(fused)
        attn_weights = self.attention(lstm_out)
        attn_weights = torch.softmax(attn_weights, dim=1)
        return (lstm_out * attn_weights).sum(dim=1)

    def classify_features(self, context: torch.Tensor) -> torch.Tensor:
        return self.head(context)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        context = self.forward_features(x)
        return self.classify_features(context)
