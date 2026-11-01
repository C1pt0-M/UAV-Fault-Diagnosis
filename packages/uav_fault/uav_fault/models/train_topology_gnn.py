"""Train and evaluate the offline three-class topology GNN.

The cache is produced by scripts/data/build_topology_cache.py and contains raw
26-channel windows plus case/domain provenance. Normalisation is fitted upstream on
the official train partition and is applied here without recomputing it.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from uav_fault.models.cnn.topology_gnn import TopologyGNN

LABEL_NAMES = ["normal", "motor", "sensor"]
SPLIT_FILES = {
    "train": "train.pt",
    "val": "val.pt",
    "test": "test.pt",
    "external_test": "real_holdout_test.pt",
}


class TopologyWindowDataset(Dataset):
    def __init__(self, path: Path, stats: dict[str, Any], max_samples: int = 0):
        # The train cache is about 2.6 GB. Map it so small smoke runs only read
        # their selected windows instead of allocating the complete raw tensor.
        payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        x = payload["x"].float()
        y = payload["y"].long()
        if x.ndim != 3 or x.shape[1] != len(stats["columns"]):
            raise ValueError(f"invalid x shape in {path}: {tuple(x.shape)}")
        if y.ndim != 1 or len(y) != len(x):
            raise ValueError(f"invalid y shape in {path}")
        if ((y < 0) | (y >= len(LABEL_NAMES))).any():
            raise ValueError(f"labels outside {LABEL_NAMES} in {path}")
        for key in ("domains", "case_ids"):
            if key in payload and len(payload[key]) != len(y):
                raise ValueError(f"{key} metadata length mismatch in {path}")
        if max_samples and max_samples < len(y):
            generator = torch.Generator().manual_seed(42)
            indices = torch.randperm(len(y), generator=generator)[:max_samples].sort().values
            x, y = x[indices], y[indices]
            self.domains = [payload["domains"][int(i)] for i in indices.tolist()]
            self.case_ids = [payload["case_ids"][int(i)] for i in indices.tolist()]
        else:
            self.domains = list(payload.get("domains", [""] * len(y)))
            self.case_ids = list(payload.get("case_ids", [""] * len(y)))
        mean = torch.tensor(stats["mean"], dtype=torch.float32).view(1, -1, 1)
        std = torch.tensor(stats["std"], dtype=torch.float32).view(1, -1, 1)
        self.x = (x - mean) / std.clamp_min(1e-8)
        self.y = y

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[index], self.y[index]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_stats(cache_dir: Path) -> dict[str, Any]:
    stats = json.loads((cache_dir / "norm_stats.json").read_text(encoding="utf-8"))
    if stats.get("label_names") != LABEL_NAMES:
        raise ValueError(f"unexpected label_names: {stats.get('label_names')}")
    if stats.get("feature_schema") != "rfly_seq_columns_v2":
        raise ValueError("topology cache must use rfly_seq_columns_v2")
    if len(stats.get("columns", [])) != 26:
        raise ValueError("topology cache must contain 26 channels")
    mean = np.asarray(stats.get("mean", []), dtype=np.float32)
    std = np.asarray(stats.get("std", []), dtype=np.float32)
    if mean.shape != (26,) or std.shape != (26,) or not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ValueError("normalization statistics must contain 26 finite values")
    if (std <= 0).any():
        raise ValueError("normalization std must be positive")
    return stats


def load_graph(graph_dir: Path, columns: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    channels = len(columns)
    names_path = graph_dir / "feature_names.json"
    if not names_path.exists():
        raise FileNotFoundError(f"missing topology feature contract: {names_path}")
    graph_columns = json.loads(names_path.read_text(encoding="utf-8"))
    if graph_columns != columns:
        raise ValueError("topology graph feature_names.json does not match cache columns")
    adjacency = np.load(graph_dir / "adjacency_fused.npy").astype(np.float32)
    reference_mean = np.load(graph_dir / "adjacency_normal_mean.npy").astype(np.float32)
    reference_std = np.load(graph_dir / "adjacency_normal_std.npy").astype(np.float32)
    for name, value in (("adjacency_fused", adjacency), ("adjacency_normal_mean", reference_mean),
                        ("adjacency_normal_std", reference_std)):
        if value.shape != (channels, channels) or not np.isfinite(value).all():
            raise ValueError(f"invalid {name} shape or values")
    return adjacency, reference_mean, reference_std


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int,
                balanced: bool = False) -> DataLoader:
    sampler = None
    if balanced:
        labels = dataset.y.numpy()
        counts = np.bincount(labels, minlength=len(LABEL_NAMES)).astype(np.float64)
        weights = 1.0 / np.maximum(counts, 1.0)
        sample_weights = torch.as_tensor(weights[labels], dtype=torch.double)
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
        shuffle = False
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=torch.cuda.is_available(),
    )


def metrics(labels: np.ndarray, preds: np.ndarray) -> dict[str, Any]:
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, labels=[0, 1, 2], average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(labels, preds, labels=[0, 1, 2], average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(labels, preds, labels=[0, 1, 2]).tolist(),
        "classification_report": classification_report(
            labels, preds, labels=[0, 1, 2], target_names=LABEL_NAMES,
            output_dict=True, zero_division=0,
        ),
        "samples": int(len(labels)),
    }


def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    labels, preds = [], []
    with torch.no_grad():
        for x, y in loader:
            output = model(x.to(device))
            labels.append(y.numpy())
            preds.append(output.logits.argmax(dim=1).cpu().numpy())
    if not labels:
        raise ValueError("empty evaluation loader")
    return np.concatenate(labels), np.concatenate(preds)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    labels, preds = predict(model, loader, device)
    return metrics(labels, preds)


def evaluate_by_domain(model: nn.Module, dataset: TopologyWindowDataset,
                       loader: DataLoader, device: torch.device) -> dict[str, dict[str, Any]]:
    labels, preds = predict(model, loader, device)
    domains = np.asarray(dataset.domains)
    if len(domains) != len(labels):
        raise ValueError("domain metadata length does not match evaluation samples")
    reports = {}
    for domain in sorted(set(domains.tolist())):
        mask = domains == domain
        if mask.any():
            reports[str(domain)] = metrics(labels[mask], preds[mask])
    return reports


def run_epoch(model: TopologyGNN, loader: DataLoader, optimizer: torch.optim.Optimizer,
              device: torch.device, class_weights: torch.Tensor, lambda_aux: float,
              train: bool) -> float:
    model.train(train)
    total_loss = 0.0
    total_samples = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.set_grad_enabled(train):
            output = model(x)
            loss = nn.functional.cross_entropy(output.logits, y, weight=class_weights)
            is_fault = (y != 0).float()
            node_score, edge_score = model.auxiliary_scores(output)
            loss = loss + lambda_aux * (
                nn.functional.binary_cross_entropy_with_logits(node_score, is_fault)
                + nn.functional.binary_cross_entropy_with_logits(edge_score, is_fault)
            )
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
        count = len(y)
        total_loss += float(loss.detach()) * count
        total_samples += count
    return total_loss / max(total_samples, 1)


def build_model(args: argparse.Namespace, graph: tuple[np.ndarray, np.ndarray, np.ndarray]) -> TopologyGNN:
    adjacency, reference_mean, reference_std = graph
    return TopologyGNN(
        in_channels=26,
        num_classes=3,
        node_dim=args.node_dim,
        graph_layers=args.graph_layers,
        heads=args.heads,
        dropout=args.dropout,
        use_graph=not args.no_graph,
        use_edge_branch=not args.no_edge_branch,
        adjacency=adjacency,
        reference_mean=reference_mean,
        reference_std=reference_std,
        norm=args.norm,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min-epochs", type=int, default=30)
    parser.add_argument("--min-delta", type=float, default=0.002)
    parser.add_argument("--lambda-aux", type=float, default=0.1)
    parser.add_argument("--node-dim", type=int, default=64)
    parser.add_argument("--graph-layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--norm", choices=["batch", "group"], default="batch")
    parser.add_argument("--no-graph", action="store_true")
    parser.add_argument("--no-edge-branch", action="store_true")
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-eval-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0:
        parser.error("epochs and batch-size must be positive")
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_dir, graph_dir, output_dir = args.cache_dir, args.graph_dir, args.output_dir
    stats = load_stats(cache_dir)
    graph = load_graph(graph_dir, stats["columns"])
    train_ds = TopologyWindowDataset(cache_dir / "train.pt", stats, args.max_train_samples)
    val_ds = TopologyWindowDataset(cache_dir / "val.pt", stats, args.max_eval_samples)
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise ValueError("train and val must be non-empty")
    # The weighted CE below already handles the class imbalance; do not combine it
    # with a weighted sampler, which would count the correction twice.
    train_loader = make_loader(train_ds, args.batch_size, True, args.num_workers, balanced=False)
    val_loader = make_loader(val_ds, args.batch_size, False, args.num_workers)
    class_counts = torch.bincount(train_ds.y, minlength=3).float()
    class_weights = (class_counts.sum() / class_counts.clamp_min(1.0)).to(device)
    class_weights = class_weights / class_weights.mean()
    model = build_model(args, graph).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"device: {device}", flush=True)
    print(f"model=topology_gnn params={sum(p.numel() for p in model.parameters()):,}", flush=True)
    print(f"train={len(train_ds)} val={len(val_ds)} class_counts={class_counts.tolist()}", flush=True)

    best_f1 = -math.inf
    best_state = None
    best_epoch = 0
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        loss = run_epoch(model, train_loader, optimizer, device, class_weights, args.lambda_aux, True)
        val = evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "loss": loss, **{f"val_{k}": v for k, v in val.items() if k in {"macro_f1", "accuracy"}}})
        if epoch == 1 or epoch % 5 == 0:
            print(f"epoch {epoch:3d}/{args.epochs} loss={loss:.4f} val_f1={val['macro_f1']:.4f}", flush=True)
        if val["macro_f1"] > best_f1 + args.min_delta:
            best_f1 = val["macro_f1"]
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        if epoch >= args.min_epochs and stale >= args.patience:
            print(f"early stopping at epoch {epoch} (best val_f1={best_f1:.4f})", flush=True)
            break
    if best_state is None:
        raise RuntimeError("no best model was recorded")
    model.load_state_dict(best_state)
    config = {
        "model": "topology_gnn", "label_names": LABEL_NAMES, "in_channels": 26,
        "node_dim": args.node_dim, "graph_layers": args.graph_layers, "heads": args.heads,
        "dropout": args.dropout, "norm": args.norm, "use_graph": not args.no_graph,
        "use_edge_branch": not args.no_edge_branch, "lambda_aux": args.lambda_aux,
        "best_epoch": best_epoch, "best_val_f1": best_f1, "seed": args.seed,
        "cache_dir": str(cache_dir), "graph_dir": str(graph_dir),
    }
    torch.save({"model_state": model.state_dict(), "config": config}, output_dir / "best.pt")
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    (output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    loaded_datasets = {"train": train_ds, "val": val_ds}
    for split, filename in SPLIT_FILES.items():
        path = cache_dir / filename
        if not path.exists():
            continue
        ds = loaded_datasets.get(split)
        if ds is None:
            ds = TopologyWindowDataset(path, stats, args.max_eval_samples)
        loader = make_loader(ds, args.batch_size, False, args.num_workers)
        report = evaluate(model, loader, device)
        report["by_domain"] = evaluate_by_domain(model, ds, loader, device)
        (output_dir / f"metrics_{split}.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"{split}: macro_f1={report['macro_f1']:.4f} accuracy={report['accuracy']:.4f}", flush=True)
        for domain, domain_report in report["by_domain"].items():
            print(f"  {domain}: macro_f1={domain_report['macro_f1']:.4f} samples={domain_report['samples']}", flush=True)
    (output_dir / "graph_artifacts.json").write_text(json.dumps({
        "adjacency": str(graph_dir / "adjacency_fused.npy"),
        "reference_mean": str(graph_dir / "adjacency_normal_mean.npy"),
        "reference_std": str(graph_dir / "adjacency_normal_std.npy"),
    }, indent=2) + "\n", encoding="utf-8")
    print(f"saved outputs to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
