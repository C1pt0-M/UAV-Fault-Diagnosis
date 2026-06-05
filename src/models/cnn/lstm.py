"""
在线故障检测模型集合：BiLSTM / 1D-CNN / Transformer。
"""
import math
import torch
import torch.nn as nn


class BiLSTMClassifier(nn.Module):
    """
    双向 LSTM + Attention Pooling。

    Args:
        in_channels: 输入通道数
        hidden_size: LSTM 隐层维度
        num_layers: LSTM 层数
        num_classes: 分类数
        dropout: dropout 率
    """

    def __init__(self, in_channels=23, hidden_size=128, num_layers=2,
                 num_classes=5, dropout=0.3):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        lstm_out_dim = hidden_size * 2  # bidirectional

        # Attention pooling
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

    def forward(self, x):
        """x: (N, C, T) -> transpose to (N, T, C) for LSTM."""
        x = x.transpose(1, 2)  # (N, T, C)

        # 下采样：取每 4 个点一个，减少序列长度 2048 -> 512
        x = x[:, ::4, :]

        out, _ = self.lstm(x)  # (N, T', hidden*2)

        # Attention pooling
        attn_weights = self.attention(out)  # (N, T', 1)
        attn_weights = torch.softmax(attn_weights, dim=1)
        context = (out * attn_weights).sum(dim=1)  # (N, hidden*2)

        return self.head(context)


class OnlineBiLSTM(nn.Module):
    """
    轻量版 BiLSTM，用于在线实时故障检测。

    与 BiLSTMClassifier 的区别：
    - 无 ::4 下采样（输入已是短窗口 150 步）
    - 更小的 hidden_size 和 num_layers（推理速度优先）

    Args:
        in_channels: 输入通道数（在线 20 通道）
        hidden_size: LSTM 隐层维度
        num_layers: LSTM 层数
        num_classes: 分类数（二分类 = 2）
        dropout: dropout 率
    """

    def __init__(self, in_channels=20, hidden_size=64, num_layers=1,
                 num_classes=2, dropout=0.2):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        lstm_out_dim = hidden_size * 2  # bidirectional

        # Attention pooling
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

    def forward(self, x):
        """x: (N, C, T) -> transpose to (N, T, C) for LSTM."""
        x = x.transpose(1, 2)  # (N, T, C)

        # 不做下采样：输入已是短窗口（150 步）
        out, _ = self.lstm(x)  # (N, T, hidden*2)

        # Attention pooling
        attn_weights = self.attention(out)  # (N, T, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)
        context = (out * attn_weights).sum(dim=1)  # (N, hidden*2)

        return self.head(context)


class OnlineCNN1D(nn.Module):
    """
    轻量 1D-CNN，用于在线实时故障检测。

    结构: Conv1d×3 + GlobalAvgPool + FC
    最轻量，推理最快，适合部署。

    Args:
        in_channels: 输入通道数
        num_classes: 分类数（二分类 = 2）
        dropout: dropout 率
    """

    def __init__(self, in_channels=33, hidden_size=64, num_layers=1,
                 num_classes=2, dropout=0.2):
        super().__init__()
        # hidden_size 和 num_layers 保留接口兼容，实际用 base_ch 控制
        base_ch = hidden_size

        self.features = nn.Sequential(
            # Block 1: (C, 150) -> (base_ch, 75)
            nn.Conv1d(in_channels, base_ch, kernel_size=7, padding=3),
            nn.BatchNorm1d(base_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            # Block 2: (base_ch, 75) -> (base_ch*2, 37)
            nn.Conv1d(base_ch, base_ch * 2, kernel_size=5, padding=2),
            nn.BatchNorm1d(base_ch * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            # Block 3: (base_ch*2, 37) -> (base_ch*2, 18)
            nn.Conv1d(base_ch * 2, base_ch * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(base_ch * 2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),  # (base_ch*2, 1)
        )

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(base_ch * 2, num_classes),
        )

    def forward(self, x):
        """x: (N, C, T)"""
        x = self.features(x)  # (N, base_ch*2, 1)
        x = x.squeeze(-1)     # (N, base_ch*2)
        return self.head(x)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation 通道注意力。"""
    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = max(channels // reduction, 1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (N, C, T)
        w = x.mean(dim=2)        # (N, C) — squeeze
        w = self.fc(w)            # (N, C) — excitation
        return x * w.unsqueeze(2) # (N, C, T) — scale


class OnlineMSTCA(nn.Module):
    """
    多尺度时序通道注意力 CNN (Multi-Scale Temporal Channel-Attention CNN)。

    改进点（相比 OnlineCNN1D）:
    1. 多尺度并行卷积: k=3/7/15 三分支捕获不同时间尺度的故障模式
    2. SE 通道注意力: 自适应加权传感器通道，提供可解释性
    """
    def __init__(self, in_channels=33, hidden_size=64, num_layers=1,
                 num_classes=2, dropout=0.2):
        super().__init__()
        base_ch = hidden_size
        # 三分支通道分配: 保证总和 = base_ch
        ch_s = base_ch * 3 // 8   # 24 (k=3, 高频)
        ch_m = base_ch * 3 // 8   # 24 (k=7, 中频)
        ch_l = base_ch - ch_s - ch_m  # 16 (k=15, 低频)

        # 多尺度分支
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

        # 通道注意力
        self.se = SEBlock(base_ch, reduction=4)

        # 后续层（与 OnlineCNN1D Block 2/3 相同）
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
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(base_ch * 2, num_classes),
        )

    def forward(self, x):
        """x: (N, C, T)"""
        # 多尺度融合
        x = torch.cat([self.branch_s(x), self.branch_m(x), self.branch_l(x)], dim=1)
        # 通道注意力
        x = self.se(x)
        # 后续特征提取
        x = self.pool1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.squeeze(-1)
        return self.head(x)

    def get_attention_weights(self, x):
        """提取 SE 注意力权重，用于可解释性分析。返回 (N, base_ch)。"""
        with torch.no_grad():
            ms = torch.cat([self.branch_s(x), self.branch_m(x), self.branch_l(x)], dim=1)
            w = ms.mean(dim=2)
            return self.se.fc(w)


class OnlineTransformer(nn.Module):
    """
    轻量 Transformer Encoder，用于在线实时故障检测。

    结构: 线性投影 + PositionalEncoding + TransformerEncoder + CLS token

    Args:
        in_channels: 输入通道数
        hidden_size: d_model 维度
        num_layers: Transformer encoder 层数
        num_classes: 分类数（二分类 = 2）
        dropout: dropout 率
    """

    def __init__(self, in_channels=33, hidden_size=64, num_layers=2,
                 num_classes=2, dropout=0.2):
        super().__init__()
        d_model = hidden_size
        nhead = 4
        dim_ff = d_model * 2

        self.input_proj = nn.Linear(in_channels, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, 151, d_model) * 0.02)  # 150 + 1 CLS

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, activation='gelu',
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x):
        """x: (N, C, T) -> transpose to (N, T, C)"""
        x = x.transpose(1, 2)  # (N, T, C)
        N, T, C = x.shape

        x = self.input_proj(x)  # (N, T, d_model)

        # 拼接 CLS token
        cls = self.cls_token.expand(N, -1, -1)  # (N, 1, d_model)
        x = torch.cat([cls, x], dim=1)  # (N, T+1, d_model)

        # 位置编码（截断或扩展）
        x = x + self.pos_embed[:, :T + 1, :]

        x = self.encoder(x)  # (N, T+1, d_model)

        # 取 CLS token 输出
        cls_out = x[:, 0, :]  # (N, d_model)
        return self.head(cls_out)
