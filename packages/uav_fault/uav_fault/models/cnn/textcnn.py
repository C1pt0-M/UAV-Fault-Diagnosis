"""
TextCNN 时序分类模型。
多尺度并行卷积核 + 全局最大池化，捕捉不同时间尺度的模式。
"""
import torch
import torch.nn as nn


class TextCNN1d(nn.Module):
    """
    TextCNN 时序分类器。

    并行使用多个不同 kernel_size 的 Conv1d 分支，
    每个分支做全局最大池化，拼接后分类。

    Args:
        in_channels: 输入通道数
        num_classes: 分类数
        num_filters: 每个 kernel size 分支的输出通道数
        kernel_sizes: 多尺度卷积核大小列表
        dropout: 分类头的 dropout 率
    """

    def __init__(self, in_channels=43, num_classes=5, num_filters=128,
                 kernel_sizes=(3, 5, 7, 11, 21), dropout=0.3):
        super().__init__()

        self.branches = nn.ModuleList()
        for ks in kernel_sizes:
            branch = nn.Sequential(
                nn.Conv1d(in_channels, num_filters, kernel_size=ks,
                          padding=ks // 2, bias=False),
                nn.BatchNorm1d(num_filters),
                nn.ReLU(inplace=True),
            )
            self.branches.append(branch)

        total_filters = num_filters * len(kernel_sizes)

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(total_filters, total_filters // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(total_filters // 2, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """x: (N, C, T)"""
        branch_outs = []
        for branch in self.branches:
            out = branch(x)                           # (N, num_filters, T)
            out = torch.max(out, dim=2).values        # (N, num_filters) 全局最大池化
            branch_outs.append(out)

        combined = torch.cat(branch_outs, dim=1)      # (N, num_filters * n_branches)
        return self.head(combined)
