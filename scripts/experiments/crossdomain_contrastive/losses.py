"""Supervised Cross-Domain Contrastive Loss."""
import torch
import torch.nn.functional as F


def supervised_contrastive_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """SupCon loss (Khosla et al. 2020) — works across domains by label.

    Args:
        features: L2-normalized projections, shape (N, D)
        labels: integer class labels, shape (N,)
        temperature: scaling temperature

    Returns:
        Scalar loss
    """
    device = features.device
    N = features.shape[0]
    if N < 2:
        return torch.tensor(0.0, device=device)

    # Similarity matrix
    sim = torch.matmul(features, features.T) / temperature  # (N, N)

    # Mask: same label = positive (excluding self)
    labels_col = labels.unsqueeze(1)  # (N, 1)
    pos_mask = (labels_col == labels_col.T).float()  # (N, N)
    pos_mask.fill_diagonal_(0.0)

    # For numerical stability
    sim_max, _ = sim.max(dim=1, keepdim=True)
    sim = sim - sim_max.detach()

    # Denominator: all except self
    self_mask = torch.eye(N, device=device)
    denom = torch.exp(sim) * (1 - self_mask)
    denom = denom.sum(dim=1, keepdim=True)  # (N, 1)

    # Log probability of positives
    log_prob = sim - torch.log(denom + 1e-8)  # (N, N)

    # Mean over positives for each anchor
    pos_count = pos_mask.sum(dim=1)  # (N,)
    # Avoid division by zero for samples with no positives in batch
    valid = pos_count > 0
    if not valid.any():
        return torch.tensor(0.0, device=device)

    mean_log_prob = (log_prob * pos_mask).sum(dim=1) / (pos_count + 1e-8)
    loss = -mean_log_prob[valid].mean()
    return loss
