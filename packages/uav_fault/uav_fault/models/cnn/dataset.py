"""
PyTorch Dataset: 从 seq_cache 加载预处理好的时序 tensor。
支持训练时数据增强。
"""
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path


class SequenceDataset(Dataset):
    """加载 .pt 缓存文件，返回归一化后的 (x, y)。"""

    def __init__(self, cache_dir: str, norm_stats_path: str, augment: bool = False):
        self.cache_dir = Path(cache_dir)
        self.files = sorted(self.cache_dir.glob('*.pt'))
        self.augment = augment

        with open(norm_stats_path) as f:
            stats = json.load(f)
        self.mean = torch.tensor(stats['mean'], dtype=torch.float32).unsqueeze(1)  # (C, 1)
        self.std = torch.tensor(stats['std'], dtype=torch.float32).unsqueeze(1)    # (C, 1)

        # 缓存标签，方便外部计算 class weights / sampler
        self.labels = []
        for f in self.files:
            sample = torch.load(f, weights_only=True)
            self.labels.append(sample['y'])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        sample = torch.load(self.files[idx], weights_only=True)
        x = (sample['x'] - self.mean) / self.std  # (C, T) z-score
        y = sample['y']

        if self.augment:
            x = self._augment(x.clone(), y)

        return x, y

    def _shift_no_wrap(self, x, shift):
        """非循环平移，用边缘值填充，避免 roll 把故障段卷到序列开头。"""
        if shift == 0:
            return x
        out = torch.zeros_like(x)
        if shift > 0:
            out[:, shift:] = x[:, :-shift]
            out[:, :shift] = x[:, :1]
        else:
            k = -shift
            out[:, :-k] = x[:, k:]
            out[:, -k:] = x[:, -1:]
        return out

    def _augment(self, x, y):
        """轻量数据增强，保持物理关系。"""
        C, T = x.shape

        # 1. 时间平移 (±2%，非循环，用边缘填充)
        if np.random.random() < 0.35:
            shift = int(np.random.uniform(-0.02, 0.02) * T)
            x = self._shift_no_wrap(x, shift)

        # 2. 幅值抖动 (±3%)
        if np.random.random() < 0.35:
            scale = 1.0 + np.random.uniform(-0.03, 0.03)
            x = x * scale

        # 3. 加性高斯噪声 (std=0.01)
        if np.random.random() < 0.35:
            noise = torch.randn_like(x) * 0.01
            x = x + noise

        # 4. 时间遮挡：减轻强度，对 thrust_loss 更保守（避免遮盖渐变信号）
        mask_prob = 0.15
        if np.random.random() < mask_prob:
            mask_len = int(np.random.uniform(0.02, 0.05) * T)
            start = np.random.randint(0, max(1, T - mask_len + 1))
            x[:, start:start + mask_len] = 0.0

        return x
