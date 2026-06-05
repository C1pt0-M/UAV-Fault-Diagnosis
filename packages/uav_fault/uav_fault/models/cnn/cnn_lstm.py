"""
CNN-LSTM 混合时序分类模型。
CNN 提取局部多尺度特征 → BiLSTM 捕捉时序依赖 → Attention Pooling → 分类。
"""
import torch
import torch.nn as nn


class CNNLSTMClassifier(nn.Module):
    """
    CNN-BiLSTM 混合分类器。

    前端用多层 Conv1d 提取局部特征并降采样，
    后端用 BiLSTM 建模时序依赖，Attention 聚合后分类。

    Args:
        in_channels: 输入通道数
        cnn_channels: CNN 各层输出通道数
        lstm_hidden: LSTM 隐层维度
        lstm_layers: LSTM 层数
        num_classes: 分类数
        dropout: dropout 率
    """

    def __init__(self, in_channels=43, cnn_channels=(64, 128, 128),
                 lstm_hidden=128, lstm_layers=2, num_classes=5, dropout=0.3):
        super().__init__()

        # CNN 前端：逐层卷积 + 池化，降采样并提取局部特征
        cnn_blocks = []
        ch_in = in_channels
        for ch_out in cnn_channels:
            cnn_blocks.append(nn.Sequential(
                nn.Conv1d(ch_in, ch_out, kernel_size=5, padding=2, bias=False),
                nn.BatchNorm1d(ch_out),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2, stride=2),  # 每层降 2x
            ))
            ch_in = ch_out
        self.cnn = nn.Sequential(*cnn_blocks)
        # 2048 -> 1024 -> 512 -> 256 (经过3层池化)

        # BiLSTM 后端
        self.lstm = nn.LSTM(
            input_size=cnn_channels[-1],
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        lstm_out_dim = lstm_hidden * 2  # bidirectional

        # Attention pooling
        self.attention = nn.Sequential(
            nn.Linear(lstm_out_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        # 分类头
        self.head = nn.Sequential(
            nn.LayerNorm(lstm_out_dim),
            nn.Dropout(dropout),
            nn.Linear(lstm_out_dim, num_classes),
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
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """x: (N, C, T)"""
        # CNN: (N, C, T) -> (N, cnn_channels[-1], T//8)
        x = self.cnn(x)

        # 转置给 LSTM: (N, T//8, cnn_channels[-1])
        x = x.transpose(1, 2)

        # BiLSTM: (N, T//8, lstm_hidden*2)
        out, _ = self.lstm(x)

        # Attention pooling
        attn_weights = self.attention(out)           # (N, T//8, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)
        context = (out * attn_weights).sum(dim=1)    # (N, lstm_hidden*2)

        return self.head(context)
