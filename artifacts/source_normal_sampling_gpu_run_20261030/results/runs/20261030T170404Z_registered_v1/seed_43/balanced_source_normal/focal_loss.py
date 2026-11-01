"""
多分类 Focal Loss。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss for multi-class classification.

    Args:
        alpha: 类别权重向量 (C,)，None 则均等
        gamma: focusing parameter，越大越关注难样本
        reduction: 'mean' | 'sum' | 'none'
    """

    def __init__(self, alpha=None, gamma=2.0, reduction='mean', eps=1e-8):
        super().__init__()
        if alpha is not None:
            alpha_t = torch.tensor(alpha, dtype=torch.float32)
            if alpha_t.ndim != 1:
                raise ValueError('alpha must be 1D class weight vector')
            self.register_buffer('alpha', alpha_t)
        else:
            self.alpha = None
        self.gamma = gamma
        self.reduction = reduction
        self.eps = eps

    def forward(self, logits, targets):
        """
        logits: (N, C)
        targets: (N,) long
        """
        log_p = F.log_softmax(logits, dim=1)  # (N, C)
        p = log_p.exp()

        # gather class probabilities
        log_p_t = log_p.gather(1, targets.unsqueeze(1)).squeeze(1)  # (N,)
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)          # (N,)
        p_t = p_t.clamp(min=self.eps, max=1.0 - self.eps)

        focal_weight = (1 - p_t) ** self.gamma

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_weight = alpha_t * focal_weight

        loss = -focal_weight * log_p_t

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss
