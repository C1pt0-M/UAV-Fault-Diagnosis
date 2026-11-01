"""Minimal runnable checks for the topology GNN training entrypoint."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from uav_fault.models.cnn.topology_gnn import TopologyGNN
from uav_fault.models.train_topology_gnn import (
    LABEL_NAMES,
    TopologyWindowDataset,
    build_model,
    evaluate,
    load_graph,
    make_loader,
    run_epoch,
)


def _stats() -> dict:
    return {
        "columns": [f"channel_{i}" for i in range(26)],
        "mean": [0.0] * 26,
        "std": [1.0] * 26,
        "label_names": LABEL_NAMES,
        "feature_schema": "rfly_seq_columns_v2",
    }


def _payload(path: Path) -> None:
    torch.save(
        {
            "x": torch.randn(12, 26, 150),
            "y": torch.tensor([0, 1, 2] * 4),
            "domains": ["SIL"] * 6 + ["HIL"] * 6,
            "case_ids": [f"case-{i}" for i in range(12)],
        },
        path,
    )


def test_topology_training_smoke(tmp_path: Path):
    path = tmp_path / "train.pt"
    _payload(path)
    stats = _stats()
    dataset = TopologyWindowDataset(path, stats)
    loader = make_loader(dataset, batch_size=4, shuffle=False, num_workers=0)

    adjacency = np.ones((26, 26), dtype=np.float32)
    np.fill_diagonal(adjacency, 0.0)
    model = TopologyGNN(
        in_channels=26,
        num_classes=3,
        node_dim=8,
        graph_layers=1,
        heads=2,
        dropout=0.0,
        adjacency=adjacency,
        reference_mean=np.zeros((26, 26), dtype=np.float32),
        reference_std=np.ones((26, 26), dtype=np.float32),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss = run_epoch(
        model,
        loader,
        optimizer,
        torch.device("cpu"),
        torch.ones(3),
        lambda_aux=0.1,
        train=True,
    )
    report = evaluate(model, loader, torch.device("cpu"))

    assert np.isfinite(loss)
    assert report["samples"] == 12
    assert 0.0 <= report["macro_f1"] <= 1.0


def test_load_graph_rejects_channel_order_mismatch(tmp_path: Path):
    graph_dir = tmp_path / "graph"
    graph_dir.mkdir()
    columns = _stats()["columns"]
    (graph_dir / "feature_names.json").write_text('[]', encoding="utf-8")
    for name, value in (
        ("adjacency_fused.npy", np.eye(26, dtype=np.float32)),
        ("adjacency_normal_mean.npy", np.zeros((26, 26), dtype=np.float32)),
        ("adjacency_normal_std.npy", np.ones((26, 26), dtype=np.float32)),
    ):
        np.save(graph_dir / name, value)
    try:
        load_graph(graph_dir, columns)
    except ValueError as exc:
        assert "feature_names" in str(exc)
    else:
        raise AssertionError("channel order mismatch was accepted")


def test_build_model_keeps_graph_artifacts_in_state():
    adjacency = np.eye(26, dtype=np.float32)
    reference_mean = np.zeros((26, 26), dtype=np.float32)
    reference_std = np.ones((26, 26), dtype=np.float32)
    args = SimpleNamespace(
        node_dim=8,
        graph_layers=1,
        heads=2,
        dropout=0.0,
        no_graph=False,
        no_edge_branch=False,
        norm="group",
    )
    model = build_model(args, (adjacency, reference_mean, reference_std))
    state = model.state_dict()

    assert "adjacency" in state
    assert "ref_mean" in state
    assert "ref_std" in state
    assert torch.equal(state["adjacency"], torch.from_numpy(adjacency))
