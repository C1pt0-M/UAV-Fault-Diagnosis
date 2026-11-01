"""
Online fault detection model family.

This module keeps the online architectures used by both training and runtime
in one place so the detector can load exactly the same model definitions.

Channel schemas -- two coexist while the online path migrates to the offline set:

  rfly_seq_columns_v1 (31 channels, RFLY_SEQ_COLUMNS_V1)
      The legacy ONLINE runtime schema fed by the RflySim/PX4 live bridge. It still
      carries baro_alt (absolute), baro_temp, baro_pressure, actuator_mean,
      actuator_std and actuator_effort -- exactly the fields the offline feature
      contract v1 dropped. The three deployed online models are trained on it.

  rfly_seq_columns_v2 (26 channels, RFLY_SEQ_COLUMNS_V2)
      Aligned with the offline feature contract v1 (D:\\UAV-Data\\feature_contract_v1.json):
      the five environment-dependent channels are gone and baro_alt becomes
      baro_alt_relative (window-start subtracted).

Online and offline models only use the same feature set once the online models are
retrained on v2; the architectures here accept either schema.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from uav_fault.model_contract import (
    RFLY_SEQ_COLUMNS_V1,
    RFLY_SEQ_COLUMNS_V2,
    RFLY_SEQ_SCHEMAS,
)

RFLY_SEQ_INDEX = {name: idx for idx, name in enumerate(RFLY_SEQ_COLUMNS_V1)}


def resolve_seq_index(columns: list[str] | None, in_channels: int) -> dict[str, int]:
    """Map channel name -> position, inferring the schema from the channel count."""
    if columns is not None:
        names = list(columns)
        if len(names) != in_channels:
            raise ValueError(
                f"columns has {len(names)} entries but in_channels={in_channels}"
            )
        return {name: idx for idx, name in enumerate(names)}
    for name, names in RFLY_SEQ_SCHEMAS.items():
        if len(names) == in_channels:
            return {column: idx for idx, column in enumerate(names)}
    raise ValueError(
        f"cannot infer a feature schema for in_channels={in_channels}; "
        f"known schemas are {[(n, len(c)) for n, c in RFLY_SEQ_SCHEMAS.items()]}"
    )


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
    """MSTCA plus a hand-made relation branch, fused by a gate.

    The relation branch reads named channels, so it works on either schema. Two
    columns of the legacy 31-channel schema no longer exist in v2, and both are
    recoverable from channels that do:

      actuator_effort  v2 has no stored column. Rebuilt as
                       mean(ctrl + out) / max(acc_norm, 0.1), the same formula the
                       adapter used. NOTE it is computed on the already-normalised
                       input, so it is a proxy rather than the v1 column. A v2 model
                       is retrained anyway, so the branch learns this version.
      baro_alt         v2 ships baro_alt_relative. Only the derivative is used, and
                       subtracting the window start does not change a difference.
    """

    def __init__(
        self,
        in_channels: int = 31,
        hidden_size: int = 64,
        num_layers: int = 1,
        num_classes: int = 2,
        dropout: float = 0.2,
        columns: list[str] | None = None,
    ):
        super().__init__()
        index = resolve_seq_index(columns, in_channels)

        required = (
            [f"actuator_ctrl_{i}" for i in range(1, 5)]
            + [f"actuator_out_{i}" for i in range(1, 5)]
            + ["gyro_x", "gyro_y", "gyro_z"]
            + ["att_q_x", "att_q_y", "att_q_z"]
            + ["acc_norm", "vel_z"]
            + ["baro_alt_relative"]
        )
        missing = [name for name in required if name not in index]
        altitude_column = "baro_alt_relative"
        if "baro_alt_relative" in missing and "baro_alt" in index:
            # Legacy schema: absolute baro_alt, same derivative semantics.
            altitude_column = "baro_alt"
            missing = [name for name in missing if name != "baro_alt_relative"]
        if missing:
            raise ValueError(
                f"OnlineKGMSTCA needs channels {missing}; "
                f"got a schema of length {in_channels}"
            )

        self.in_channels = in_channels
        self.altitude_column = altitude_column
        self._ctrl_idx = [index[f"actuator_ctrl_{i}"] for i in range(1, 5)]
        self._out_idx = [index[f"actuator_out_{i}"] for i in range(1, 5)]
        self._gyro_idx = [index[name] for name in ("gyro_x", "gyro_y", "gyro_z")]
        self._quat_idx = [index[name] for name in ("att_q_x", "att_q_y", "att_q_z")]
        self._acc_norm_idx = index["acc_norm"]
        self._vel_z_idx = index["vel_z"]
        self._alt_idx = index[altitude_column]
        self._effort_idx = index.get("actuator_effort")

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

    def _relation_features(self, x: torch.Tensor) -> torch.Tensor:
        actuator_ctrl = x[:, self._ctrl_idx, :]
        actuator_out = x[:, self._out_idx, :]
        gyro = x[:, self._gyro_idx, :]
        quat_xyz = x[:, self._quat_idx, :]
        acc_norm = x[:, self._acc_norm_idx, :]

        actuator_out_cross_std = actuator_out.std(dim=1, unbiased=False)
        actuator_out_cross_range = actuator_out.max(dim=1).values - actuator_out.min(dim=1).values
        ctrl_out_gap = torch.abs(actuator_ctrl - actuator_out)
        ctrl_out_gap_mean = ctrl_out_gap.mean(dim=1)
        ctrl_out_gap_std = ctrl_out_gap.std(dim=1, unbiased=False)

        if self._effort_idx is not None:
            actuator_effort = x[:, self._effort_idx, :]
        else:
            actuator_mean = torch.cat([actuator_ctrl, actuator_out], dim=1).mean(dim=1)
            actuator_effort = actuator_mean / acc_norm.clamp_min(0.1)
        acc_norm_deviation = torch.abs(acc_norm)
        gyro_norm = torch.linalg.vector_norm(gyro, dim=1)

        baro_alt = x[:, self._alt_idx, :]
        vel_z = x[:, self._vel_z_idx, :]
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
