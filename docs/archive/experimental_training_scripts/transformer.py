"""
轻量 Transformer 时序分类模型。
"""
import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """正弦位置编码。"""

    def __init__(self, d_model, max_len=2048, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        """x: (N, T, d_model)"""
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerClassifier(nn.Module):
    """
    轻量 Transformer 时序分类器。

    先用 1D Conv 做 patch embedding（降低序列长度），
    再用 Transformer Encoder 提取特征。

    Args:
        in_channels: 输入通道数
        d_model: Transformer 维度
        nhead: 注意力头数
        num_layers: Encoder 层数
        num_classes: 分类数
        patch_size: patch 大小（Conv1d stride）
        dropout: dropout 率
    """

    def __init__(self, in_channels=23, d_model=128, nhead=4, num_layers=4,
                 num_classes=5, patch_size=8, dropout=0.2):
        super().__init__()

        # Patch embedding: Conv1d 将 (N, C, T) -> (N, d_model, T//patch_size)
        self.patch_embed = nn.Sequential(
            nn.Conv1d(in_channels, d_model, kernel_size=patch_size, stride=patch_size, bias=False),
            nn.LayerNorm([d_model]),  # 会在 forward 里手动处理
        )
        # 手动实现 patch embedding 以避免 LayerNorm 维度问题
        self.embed_conv = nn.Conv1d(in_channels, d_model, kernel_size=patch_size,
                                     stride=patch_size, bias=False)
        self.embed_norm = nn.LayerNorm(d_model)

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Positional encoding
        max_patches = 2048 // patch_size + 1  # +1 for CLS
        self.pos_enc = PositionalEncoding(d_model, max_len=max_patches, dropout=dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification head
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')

    def forward(self, x):
        """x: (N, C, T)"""
        N = x.size(0)

        # Patch embedding
        x = self.embed_conv(x)  # (N, d_model, n_patches)
        x = x.transpose(1, 2)   # (N, n_patches, d_model)
        x = self.embed_norm(x)

        # Prepend CLS token
        cls = self.cls_token.expand(N, -1, -1)  # (N, 1, d_model)
        x = torch.cat([cls, x], dim=1)  # (N, n_patches+1, d_model)

        # Positional encoding
        x = self.pos_enc(x)

        # Transformer encoder
        x = self.encoder(x)

        # CLS token output
        cls_out = x[:, 0, :]  # (N, d_model)

        return self.head(cls_out)
