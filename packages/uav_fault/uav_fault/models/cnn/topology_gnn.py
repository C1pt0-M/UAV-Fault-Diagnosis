"""Topology-guided multi-task graph network for channel-level UAV fault diagnosis.

Motivation (all three points are measured, see scripts/experiments):

1. Fault onset changes the dependency structure between channels.
   Motor faults change it globally (Cohen's d = 1.27); sensor faults leave the
   global structure alone (d = -0.05) and show up on a few local edges instead
   (accel_x <-> att_q_w moves 145x).

2. The couplings that move under fault are the ones that are QUIET during normal
   hover, not the loud ones. Measured change relative to normal jitter:
     hover_observable edges: motor 1.08x, sensor 0.40x
     hover_silent edges:     motor 1.91x, sensor 0.97x
   So the graph must not be weighted by normal-flight coupling strength alone,
   and the model must be able to read individual edges.

3. A global pooled prediction therefore cannot work for sensor faults. The model
   emits graph-level, node-level and edge-level scores.

Architecture
------------
    x (B, C, T)
      |
      +-- ChannelTemporalEncoder --> node features (B, C, d)     [independent weights per channel]
      |        |
      |        +-- GAT stack over the channel graph --> (B, C, d)
      |        |        +-- attention pool --> graph feature (B, d)
      |        |        +-- NodeHead --------> node logits (B, C)
      |        |
      |        +-- window correlation --> z-score vs normal reference
      |                 +-- EdgeHead ------> edge logits (B, P)
      |                 +-- pool ----------> edge feature (B, d)
      |
      +-- fusion --> graph logits (B, num_classes)

`use_graph=False` replaces message passing with a per-node MLP (ablation).
`use_edge_branch=False` drops the topological branch (ablation).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def window_correlation(x: torch.Tensor) -> torch.Tensor:
    """Differentiable Pearson correlation per window.

    x: (B, C, T) -> (B, C, C). Sigma is a constant here, not a random variable,
    which matches the numpy reference in build_topology_graph.correlation_matrix.
    """
    centered = x - x.mean(dim=2, keepdim=True)
    std = centered.std(dim=2, keepdim=True, unbiased=False).clamp_min(1e-6)
    z = centered / std
    corr = torch.einsum("bct,bdt->bcd", z, z) / x.shape[2]
    return corr.clamp(-1.0, 1.0)


def _make_norm(kind: str, channels: int) -> nn.Module:
    if kind == "batch":
        return nn.BatchNorm1d(channels)
    if kind == "group":
        # Per-channel instance normalisation: no batch statistics, so it does not
        # carry domain-specific scale information across SIL/HIL/Real.
        return nn.GroupNorm(num_groups=channels, num_channels=channels)
    raise ValueError(f"unknown norm kind: {kind}")


class ChannelTemporalEncoder(nn.Module):
    """Encode every channel's own time series into a node feature.

    Grouped convolutions keep independent weights for each channel. Cross-channel
    information is combined by the subsequent graph layers.
    """

    def __init__(
        self,
        in_channels: int,
        node_dim: int,
        kernel_size: int = 5,
        dropout: float = 0.1,
        norm: str = "batch",
    ):
        super().__init__()
        pad = kernel_size // 2
        self.depthwise = nn.Sequential(
            nn.Conv1d(
                in_channels,
                in_channels,
                kernel_size,
                padding=pad,
                groups=in_channels,
            ),
            _make_norm(norm, in_channels),
            nn.GELU(),
            nn.Conv1d(
                in_channels,
                in_channels,
                kernel_size,
                padding=pad,
                groups=in_channels,
            ),
            _make_norm(norm, in_channels),
            nn.GELU(),
        )
        # expand group g to node_dim features for channel g
        self.expand = nn.Conv1d(
            in_channels, in_channels * node_dim, kernel_size=1, groups=in_channels
        )
        self.temporal_attention = nn.Conv1d(
            in_channels, in_channels, kernel_size=1, groups=in_channels
        )
        self.in_channels = in_channels
        self.node_dim = node_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T) -> node features (B, C, node_dim)."""
        batch, channels, steps = x.shape
        h = self.depthwise(x)  # (B, C, T)
        expanded = self.expand(h)  # (B, C*node_dim, T)
        expanded = expanded.view(batch, channels, self.node_dim, steps)
        weights = torch.softmax(self.temporal_attention(h), dim=-1)  # (B, C, T)
        node = torch.einsum("bcdt,bct->bcd", expanded, weights)
        return self.dropout(node)


class DenseGATLayer(nn.Module):
    """Multi-head graph attention over a dense, small channel graph."""

    def __init__(self, dim: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"node_dim {dim} must be divisible by heads {heads}")
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, adjacency: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """h: (B, C, d), adjacency: (C, C) weighted -> updated h and attention (B, H, C, C)."""
        batch, channels, _ = h.shape
        residual = h

        def split(t: torch.Tensor) -> torch.Tensor:
            return t.view(batch, channels, self.heads, self.head_dim).permute(0, 2, 1, 3)

        q = split(self.q(h))  # (B, H, C, hd)
        k = split(self.k(h))
        v = split(self.v(h))

        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        # Self-loops keep every node attendable even if the prior leaves it isolated.
        mask = (adjacency > 0) | torch.eye(
            channels, dtype=torch.bool, device=adjacency.device
        )
        # A multiplicative prior in attention space: softmax(scores + log(weight)).
        # Without this term a dense fused graph would behave like a full graph,
        # discarding all prior strength information.
        weights = adjacency.clone()
        weights.diagonal().clamp_(min=1.0)
        scores = scores + weights.clamp_min(1e-12).log()
        scores = scores.masked_fill(~mask, float("-inf"))
        attention = torch.softmax(scores, dim=-1)
        attention = self.dropout(attention)

        updated = torch.matmul(attention, v)  # (B, H, C, hd)
        updated = updated.permute(0, 2, 1, 3).reshape(batch, channels, self.dim)
        return self.norm(residual + self.out(updated)), attention


class NodeMixingLayer(nn.Module):
    """Ablation replacement for message passing: per-node MLP, no cross-channel flow."""

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, h: torch.Tensor, adjacency: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, channels, _ = h.shape
        attention = torch.zeros(
            batch, 1, channels, channels, device=h.device, dtype=h.dtype
        )
        return self.norm(h + self.mlp(h)), attention


@dataclass
class TopologyGNNOutput:
    logits: torch.Tensor
    node_logits: torch.Tensor
    edge_logits: torch.Tensor
    edge_deviation: torch.Tensor
    attention: torch.Tensor


class TopologyGNN(nn.Module):
    """Graph-level + node-level + edge-level fault diagnosis over the channel graph."""

    def __init__(
        self,
        in_channels: int = 26,
        num_classes: int = 3,
        node_dim: int = 64,
        graph_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.3,
        use_graph: bool = True,
        use_edge_branch: bool = True,
        adjacency: np.ndarray | None = None,
        reference_mean: np.ndarray | None = None,
        reference_std: np.ndarray | None = None,
        norm: str = "batch",
    ):
        super().__init__()
        for name, matrix in (
            ("adjacency", adjacency),
            ("reference_mean", reference_mean),
            ("reference_std", reference_std),
        ):
            if matrix is None:
                continue
            array = np.asarray(matrix)
            if array.shape != (in_channels, in_channels) or not np.isfinite(array).all():
                raise ValueError(f"{name} must be a finite {in_channels}x{in_channels} matrix")
            if not np.allclose(array, array.T, atol=1e-6):
                raise ValueError(f"{name} must be symmetric")
            if name in {"adjacency", "reference_std"} and (array < 0).any():
                raise ValueError(f"{name} must be nonnegative")
            if name == "reference_mean" and (np.abs(array) > 1.00001).any():
                raise ValueError("reference_mean must contain correlations in [-1, 1]")
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.node_dim = node_dim
        self.use_graph = use_graph
        self.use_edge_branch = use_edge_branch

        self.encoder = ChannelTemporalEncoder(
            in_channels=in_channels,
            node_dim=node_dim,
            dropout=min(dropout, 0.2),
            norm=norm,
        )

        if use_graph:
            self.graph_layers = nn.ModuleList(
                [DenseGATLayer(node_dim, heads=heads, dropout=dropout * 0.5) for _ in range(graph_layers)]
            )
        else:
            self.graph_layers = nn.ModuleList(
                [NodeMixingLayer(node_dim, dropout=dropout * 0.5) for _ in range(graph_layers)]
            )

        self.node_head = nn.Sequential(
            nn.Linear(node_dim, node_dim // 2),
            nn.GELU(),
            nn.Linear(node_dim // 2, 1),
        )
        self.node_pool = nn.Linear(node_dim, node_dim)

        pair_i, pair_j = np.triu_indices(in_channels, k=1)
        self.register_buffer("pair_i", torch.as_tensor(pair_i, dtype=torch.long), persistent=False)
        self.register_buffer("pair_j", torch.as_tensor(pair_j, dtype=torch.long), persistent=False)
        self.num_pairs = len(pair_i)

        if adjacency is None:
            adjacency = np.zeros((in_channels, in_channels), dtype=np.float32)
        self.register_buffer(
            "adjacency",
            torch.as_tensor(adjacency, dtype=torch.float32),
        )
        if reference_mean is None:
            reference_mean = np.zeros((in_channels, in_channels), dtype=np.float32)
        if reference_std is None:
            reference_std = np.ones((in_channels, in_channels), dtype=np.float32)
        self.register_buffer("ref_mean", torch.as_tensor(reference_mean, dtype=torch.float32))
        self.register_buffer("ref_std", torch.as_tensor(reference_std, dtype=torch.float32))

        if use_edge_branch:
            hidden = max(node_dim // 2, 16)
            self.edge_head = nn.Sequential(
                nn.Linear(1, hidden),
                nn.GELU(),
                nn.Linear(hidden, 1),
            )
            self.edge_pool = nn.Sequential(
                nn.Linear(self.num_pairs, node_dim),
                nn.GELU(),
                nn.Linear(node_dim, node_dim),
            )
            fusion_dim = node_dim * 2
        else:
            self.edge_head = None
            self.edge_pool = None
            fusion_dim = node_dim

        self.graph_head = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, node_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(node_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> TopologyGNNOutput:
        """x: (B, C, T) normalised -> predictions at three levels."""
        node = self.encoder(x)  # (B, C, d)
        attention = torch.zeros(
            x.shape[0], 1, self.in_channels, self.in_channels, device=x.device
        )
        for layer in self.graph_layers:
            node, attention = layer(node, self.adjacency)

        node_logits = self.node_head(node).squeeze(-1)  # (B, C)
        node_feature = self.node_pool(node.mean(dim=1))  # (B, d)

        parts = [node_feature]
        if self.use_edge_branch:
            corr = window_correlation(x)  # (B, C, C)
            deviation = (corr - self.ref_mean) / (self.ref_std + 1e-3)
            edge_deviation = deviation[:, self.pair_i, self.pair_j]  # (B, P)
            edge_logits = self.edge_head(edge_deviation.unsqueeze(-1)).squeeze(-1)
            parts.append(self.edge_pool(edge_deviation))
        else:
            edge_deviation = torch.zeros(
                x.shape[0], self.num_pairs, device=x.device, dtype=x.dtype
            )
            edge_logits = torch.zeros(x.shape[0], self.num_pairs, device=x.device, dtype=x.dtype)

        graph_logits = self.graph_head(torch.cat(parts, dim=1))
        return TopologyGNNOutput(
            logits=graph_logits,
            node_logits=node_logits,
            edge_logits=edge_logits,
            edge_deviation=edge_deviation,
            attention=attention,
        )

    def auxiliary_scores(self, output: TopologyGNNOutput) -> tuple[torch.Tensor, torch.Tensor]:
        """Aggregate node/edge evidence into one 'a fault exists' logit each.

        Node and edge heads have no per-channel supervision in RflyMAD (a motor fault
        is not attributed to one specific rotor), so they are trained through this
        aggregate: at least one node/edge must fire when a fault is present.
        """
        node_score = torch.logsumexp(output.node_logits, dim=1)
        edge_score = torch.logsumexp(output.edge_logits, dim=1)
        return node_score, edge_score
