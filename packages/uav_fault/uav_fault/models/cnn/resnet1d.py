"""
小型 1D ResNet for 时序分类。
GroupNorm 替代 BatchNorm，适合小样本训练。
GAP + GMP 双池化捕捉平均和峰值特征。
"""
import torch
import torch.nn as nn


def _norm_layer(channels, num_groups=8):
    """GroupNorm，对小 batch 更稳定。"""
    # 确保 num_groups 不超过 channels
    g = min(num_groups, channels)
    return nn.GroupNorm(g, channels)


class ResBlock1d(nn.Module):
    """残差块：两层 Conv1d + GN + ReLU，带可选下采样。"""

    def __init__(self, in_ch, out_ch, kernel_size=5, stride=1):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=False)
        self.gn1 = _norm_layer(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, stride=1, padding=padding, bias=False)
        self.gn2 = _norm_layer(out_ch)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                _norm_layer(out_ch),
            )

    def forward(self, x):
        out = self.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        out = self.relu(out + self.shortcut(x))
        return out


class ResNet1d(nn.Module):
    """
    轻量 ResNet1D，GAP+GMP 双池化。

    Args:
        in_channels: 输入通道数
        num_classes: 分类数
        base_channels: stem 输出通道数
        dropout: 分类头的 dropout 率
    """

    def __init__(self, in_channels=20, num_classes=5, base_channels=48, dropout=0.25):
        super().__init__()

        # Stem: k=7, s=2, 不用 MaxPool 保留更多时序信息
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, 7, stride=2, padding=3, bias=False),
            _norm_layer(base_channels),
            nn.ReLU(inplace=True),
        )

        # 4 stages: 48 → 96 → 192 → 192
        self.stage1 = self._make_stage(base_channels, base_channels, 2, stride=1)
        self.stage2 = self._make_stage(base_channels, base_channels * 2, 2, stride=2)
        self.stage3 = self._make_stage(base_channels * 2, base_channels * 4, 2, stride=2)
        self.stage4 = self._make_stage(base_channels * 4, base_channels * 4, 2, stride=2)

        feat_dim = base_channels * 4
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.gmp = nn.AdaptiveMaxPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, num_classes),
        )

        self._init_weights()

    def _make_stage(self, in_ch, out_ch, n_blocks, stride):
        layers = [ResBlock1d(in_ch, out_ch, stride=stride)]
        for _ in range(1, n_blocks):
            layers.append(ResBlock1d(out_ch, out_ch))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """x: (N, C, T)"""
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x_avg = self.gap(x).squeeze(-1)  # (N, feat_dim)
        x_max = self.gmp(x).squeeze(-1)  # (N, feat_dim)
        x = torch.cat([x_avg, x_max], dim=1)  # (N, feat_dim*2)
        return self.head(x)
