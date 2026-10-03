"""Smoke and contract checks for the topology-guided graph network."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "packages" / "uav_fault"))

from uav_fault.models.cnn.topology_gnn import (  # noqa: E402
    TopologyGNN,
    TopologyGNNOutput,
    window_correlation,
)

CHANNELS = 26
FRAMES = 150
PAIRS = CHANNELS * (CHANNELS - 1) // 2


def _batch(batch: int = 4) -> torch.Tensor:
    generator = torch.Generator().manual_seed(0)
    return torch.randn(batch, CHANNELS, FRAMES, generator=generator)


def _adjacency() -> np.ndarray:
    adjacency = np.zeros((CHANNELS, CHANNELS), dtype=np.float32)
    for i in range(CHANNELS - 1):
        adjacency[i, i + 1] = 1.0
        adjacency[i + 1, i] = 1.0
    return adjacency


def _model(**kwargs) -> TopologyGNN:
    defaults = dict(
        in_channels=CHANNELS,
        num_classes=3,
        node_dim=32,
        graph_layers=2,
        heads=4,
        dropout=0.1,
        adjacency=_adjacency(),
        reference_mean=np.zeros((CHANNELS, CHANNELS), dtype=np.float32),
        reference_std=np.ones((CHANNELS, CHANNELS), dtype=np.float32),
    )
    defaults.update(kwargs)
    return TopologyGNN(**defaults)


def test_forward_shapes():
    model = _model().eval()
    output = model(_batch())
    assert isinstance(output, TopologyGNNOutput)
    assert output.logits.shape == (4, 3)
    assert output.node_logits.shape == (4, CHANNELS)
    assert output.edge_logits.shape == (4, PAIRS)
    assert output.edge_deviation.shape == (4, PAIRS)
    assert output.attention.shape == (4, 4, CHANNELS, CHANNELS)


def test_forward_without_graph_still_runs():
    model = _model(use_graph=False).eval()
    output = model(_batch())
    assert output.logits.shape == (4, 3)
    assert output.node_logits.shape == (4, CHANNELS)


def test_forward_without_edge_branch_still_runs():
    model = _model(use_edge_branch=False).eval()
    output = model(_batch())
    assert output.logits.shape == (4, 3)
    assert output.edge_logits.shape == (4, PAIRS)


def test_backward_reaches_every_parameter():
    model = _model()
    output = model(_batch())
    loss = (
        torch.nn.functional.cross_entropy(output.logits, torch.tensor([0, 1, 2, 1]))
        + output.node_logits.mean()
        + output.edge_logits.mean()
    )
    loss.backward()
    missing = [
        name for name, param in model.named_parameters() if param.grad is None and param.requires_grad
    ]
    assert missing == [], f"parameters without gradient: {missing}"


def test_window_correlation_matches_numpy_reference():
    """The tensor copy must agree with the numpy one used to build the graph."""

    def correlation_matrix(window: np.ndarray) -> np.ndarray:
        std = window.std(axis=0, keepdims=True)
        std = np.where(std < 1e-10, 1.0, std)
        centered = (window - window.mean(axis=0, keepdims=True)) / std
        corr = (centered.T @ centered) / window.shape[0]
        np.fill_diagonal(corr, 1.0)
        return np.clip(corr, -1.0, 1.0)

    x = _batch(batch=2)
    raw = x.detach().numpy()
    got = window_correlation(x).detach().numpy()
    for batch in range(x.shape[0]):
        # The numpy reference works on (frames, channels); the tensor holds (C, T).
        assert np.allclose(got[batch], correlation_matrix(raw[batch].T), atol=1e-5)


def test_attention_is_a_distribution_over_the_graph():
    model = _model().eval()
    with torch.no_grad():
        output = model(_batch(batch=1))
    attention = output.attention[0, 0]
    assert torch.allclose(attention.sum(dim=-1), torch.ones(CHANNELS), atol=1e-5)
    adjacency = torch.as_tensor(_adjacency())
    mask = (adjacency > 0) | torch.eye(CHANNELS, dtype=torch.bool)
    assert float(attention[~mask].abs().sum()) == 0.0


def test_isolated_channel_does_not_produce_nan():
    model = _model(adjacency=np.zeros((CHANNELS, CHANNELS), dtype=np.float32)).eval()
    with torch.no_grad():
        output = model(_batch())
    assert torch.isfinite(output.logits).all()
    assert torch.isfinite(output.node_logits).all()


def test_auxiliary_scores_are_finite_and_responsive():
    model = _model().eval()
    with torch.no_grad():
        output = model(_batch(batch=6))
        node_score, edge_score = model.auxiliary_scores(output)
    assert node_score.shape == (6,)
    assert edge_score.shape == (6,)
    assert torch.isfinite(node_score).all()
    assert torch.isfinite(edge_score).all()
    assert float(node_score.std()) > 0.0


def test_wrong_node_dim_for_heads_is_rejected():
    with pytest.raises(ValueError):
        _model(node_dim=30, heads=4)


def test_missing_norm_kind_is_rejected():
    with pytest.raises(ValueError):
        _model(norm="mystery")
