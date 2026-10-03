"""
Online fault detection model family.

This module keeps the online architectures used by both training and runtime
in one place so the detector can load exactly the same model definitions.

Channel schemas -- the online and offline paths are NOT on the same feature set:

  RFLY_SEQ_COLUMNS_V1 (this module, 31 channels)
      The ONLINE runtime schema, fed by the RflySim/PX4 live bridge. It still
      carries baro_temp, baro_pressure, actuator_mean, actuator_std,
      actuator_effort and absolute baro_alt, i.e. exactly the fields the offline
      feature contract v1 dropped. Online models are therefore 31-channel until
      the runtime schema itself migrates.

  feature contract v1 (D:\\UAV-Data\\feature_contract_v1.json, 26 channels)
      The OFFLINE schema used by the window manifest and norm stats. It drops
      the five environment-dependent channels above and renames baro_alt to
      baro_alt_relative.

So: online architectures default to 31, offline architectures default to 26.
"""

from __future__ import annotations

import torch
import torch.nn as nn

RFLY_SEQ_COLUMNS_V1 = [
    "actuator_ctrl_1",
    "actuator_ctrl_2",
    "actuator_ctrl_3",
    "actuator_ctrl_4",
    "actuator_out_1",
    "actuator_out_2",
    "actuator_out_3",
    "actuator_out_4",
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "accel_x",
    "accel_y",
    "accel_z",
    "baro_alt",
    "baro_temp",
    "baro_pressure",
    "att_q_w",
    "att_q_x",
    "att_q_y",
    "att_q_z",
    "vel_x",
    "vel_y",
    "vel_z",
    "mag_x",
    "mag_y",
    "mag_z",
    "actuator_mean",
    "actuator_std",
    "acc_norm",
    "actuator_effort",
]
RFLY_SEQ_INDEX = {name: idx for idx, name in enumerate(RFLY_SEQ_COLUMNS_V1)}


class BiLSTMClassifier(nn.Module):
    def __init__(
        self,
        in_channels: int = 26,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 5,
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
        lstm_out_dim = hidden_size * 2
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = x[:, ::4, :]
        out, _ = self.lstm(x)
        attn_weights = torch.softmax(self.attention(out), dim=1)
        context = (out * attn_weights).sum(dim=1)
        return self.head(context)


class OnlineBiLSTM(nn.Module):
    def __init__(
        self,
        in_channels: int = 31,
        hidden_size: int = 64,
        num_layers: int = 1,
        num_classes: int = 2,
        dropout: float = 0.2,
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
        lstm_out_dim = hidden_size * 2
        self.attention = nn.Sequential(
            nn.Linear(lstm_out_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(lstm_out_dim),
            nn.Dropout(dropout),
            nn.Linear(lstm_out_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        out, _ = self.lstm(x)
        attn_weights = torch.softmax(self.attention(out), dim=1)
        context = (out * attn_weights).sum(dim=1)
        return self.head(context)


class OnlineCNN1D(nn.Module):
    def __init__(
        self,
        in_channels: int = 31,
        hidden_size: int = 64,
        num_layers: int = 1,
        num_classes: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        base_ch = hidden_size
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, base_ch, kernel_size=7, padding=3),
            nn.BatchNorm1d(base_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(base_ch, base_ch * 2, kernel_size=5, padding=2),
            nn.BatchNorm1d(base_ch * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(base_ch * 2, base_ch * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(base_ch * 2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(base_ch * 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.squeeze(-1)
        return self.head(x)


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(channels // reduction, 1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = x.mean(dim=2)
        weights = self.fc(weights)
        return x * weights.unsqueeze(2)


class _MSTCAFeatureEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_size: int):
        super().__init__()
        base_ch = hidden_size
        ch_s = base_ch * 3 // 8
        ch_m = base_ch * 3 // 8
        ch_l = base_ch - ch_s - ch_m

        self.branch_s = nn.Sequential(
            nn.Conv1d(in_channels, ch_s, kernel_size=3, padding=1),
            nn.BatchNorm1d(ch_s),
            nn.ReLU(inplace=True),
        )
        self.branch_m = nn.Sequential(
            nn.Conv1d(in_channels, ch_m, kernel_size=7, padding=3),
            nn.BatchNorm1d(ch_m),
            nn.ReLU(inplace=True),
        )
        self.branch_l = nn.Sequential(
            nn.Conv1d(in_channels, ch_l, kernel_size=15, padding=7),
            nn.BatchNorm1d(ch_l),
            nn.ReLU(inplace=True),
        )
        self.se = SEBlock(base_ch, reduction=4)
        self.pool1 = nn.MaxPool1d(2)
        self.block2 = nn.Sequential(
            nn.Conv1d(base_ch, base_ch * 2, kernel_size=5, padding=2),
            nn.BatchNorm1d(base_ch * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
        )
        self.block3 = nn.Sequential(
            nn.Conv1d(base_ch * 2, base_ch * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(base_ch * 2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.out_dim = base_ch * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.cat([self.branch_s(x), self.branch_m(x), self.branch_l(x)], dim=1)
        x = self.se(x)
        x = self.pool1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x.squeeze(-1)

    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            merged = torch.cat([self.branch_s(x), self.branch_m(x), self.branch_l(x)], dim=1)
            weights = merged.mean(dim=2)
            return self.se.fc(weights)


class OnlineMSTCA(nn.Module):
    def __init__(
        self,
        in_channels: int = 31,
        hidden_size: int = 64,
        num_layers: int = 1,
        num_classes: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.encoder = _MSTCAFeatureEncoder(in_channels=in_channels, hidden_size=hidden_size)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.encoder.out_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))

    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder.get_attention_weights(x)


class OnlineKGMSTCA(nn.Module):
    def __init__(
        self,
        in_channels: int = 31,
        hidden_size: int = 64,
        num_layers: int = 1,
        num_classes: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        if in_channels < len(RFLY_SEQ_COLUMNS_V1):
            raise ValueError(
                f"OnlineKGMSTCA expects at least {len(RFLY_SEQ_COLUMNS_V1)} channels, got {in_channels}"
            )

        relation_channels = 9
        relation_hidden = max(hidden_size // 2, 8)

        self.raw_encoder = _MSTCAFeatureEncoder(
            in_channels=in_channels,
            hidden_size=hidden_size,
        )
        self.relation_encoder = _MSTCAFeatureEncoder(
            in_channels=relation_channels,
            hidden_size=relation_hidden,
        )
        self.relation_proj = nn.Linear(
            self.relation_encoder.out_dim,
            self.raw_encoder.out_dim,
        )
        gate_hidden = max(self.raw_encoder.out_dim // 2, 16)
        self.gate = nn.Sequential(
            nn.Linear(self.raw_encoder.out_dim * 2, gate_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(gate_hidden, 1),
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.raw_encoder.out_dim, num_classes),
        )

    def _feature(self, x: torch.Tensor, name: str) -> torch.Tensor:
        return x[:, RFLY_SEQ_INDEX[name], :]

    def _relation_features(self, x: torch.Tensor) -> torch.Tensor:
        actuator_ctrl = x[
            :,
            RFLY_SEQ_INDEX["actuator_ctrl_1"] : RFLY_SEQ_INDEX["actuator_ctrl_4"] + 1,
            :,
        ]
        actuator_out = x[
            :,
            RFLY_SEQ_INDEX["actuator_out_1"] : RFLY_SEQ_INDEX["actuator_out_4"] + 1,
            :,
        ]
        gyro = x[:, RFLY_SEQ_INDEX["gyro_x"] : RFLY_SEQ_INDEX["gyro_z"] + 1, :]
        quat_xyz = x[:, RFLY_SEQ_INDEX["att_q_x"] : RFLY_SEQ_INDEX["att_q_z"] + 1, :]

        actuator_out_cross_std = actuator_out.std(dim=1, unbiased=False)
        actuator_out_cross_range = actuator_out.max(dim=1).values - actuator_out.min(dim=1).values
        ctrl_out_gap = torch.abs(actuator_ctrl - actuator_out)
        ctrl_out_gap_mean = ctrl_out_gap.mean(dim=1)
        ctrl_out_gap_std = ctrl_out_gap.std(dim=1, unbiased=False)

        actuator_effort = self._feature(x, "actuator_effort")
        acc_norm_deviation = torch.abs(self._feature(x, "acc_norm"))
        gyro_norm = torch.linalg.vector_norm(gyro, dim=1)

        baro_alt = self._feature(x, "baro_alt")
        vel_z = self._feature(x, "vel_z")
        baro_alt_rate = torch.diff(baro_alt, dim=1, prepend=baro_alt[:, :1])
        vertical_consistency = torch.abs(baro_alt_rate - vel_z)

        attitude_energy = torch.linalg.vector_norm(quat_xyz, dim=1)

        return torch.stack(
            [
                actuator_out_cross_std,
                actuator_out_cross_range,
                ctrl_out_gap_mean,
                ctrl_out_gap_std,
                actuator_effort,
                acc_norm_deviation,
                gyro_norm,
                vertical_consistency,
                attitude_energy,
            ],
            dim=1,
        )

    def forward_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw_features = self.raw_encoder(x)
        relation_features = self.relation_encoder(self._relation_features(x))
        relation_features = self.relation_proj(relation_features)
        alpha = torch.sigmoid(
            self.gate(torch.cat([raw_features, relation_features], dim=1))
        )
        fused = alpha * raw_features + (1.0 - alpha) * relation_features
        return fused, alpha.squeeze(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fused, _ = self.forward_features(x)
        return self.head(fused)


class OnlineTransformer(nn.Module):
    def __init__(
        self,
        in_channels: int = 31,
        hidden_size: int = 64,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        d_model = hidden_size
        nhead = 4
        dim_ff = d_model * 2

        self.input_proj = nn.Linear(in_channels, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, 151, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        n_batch, time_steps, _ = x.shape
        x = self.input_proj(x)
        cls = self.cls_token.expand(n_batch, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed[:, : time_steps + 1, :]
        x = self.encoder(x)
        return self.head(x[:, 0, :])
