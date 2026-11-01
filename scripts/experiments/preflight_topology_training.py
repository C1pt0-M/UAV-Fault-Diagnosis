"""Verify existing offline three-class caches and exercise a real training batch.

Reads the caches through memory maps; does not rebuild or modify input data.
Run this locally or on AutoDL before the full TopologyGNN training run.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "packages" / "uav_fault"))

from uav_fault.models.cnn.topology_gnn import TopologyGNN
from uav_fault.models.train_topology_gnn import LABEL_NAMES, SPLIT_FILES, load_graph, load_stats


def digest(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def verify(cache_dir: Path, graph_dir: Path) -> dict:
    stats = load_stats(cache_dir)
    if stats.get("fit_partition") != "train":
        raise ValueError("normalization must be fitted on train only")
    summary = json.loads((cache_dir / "build_summary.json").read_text(encoding="utf-8"))
    if summary.get("label_map") != dict(zip(LABEL_NAMES, range(3))):
        raise ValueError("cache summary has a different label mapping")
    if summary.get("real_role") != "external_test_only" or summary.get("live_normals") is not False:
        raise ValueError("expected official-only training with Real held out")
    for summary_key, stats_key in (("contract_sha256", "feature_contract_sha256"),
                                   ("case_split_sha256", "split_manifest_sha256")):
        if not summary.get(summary_key) or summary[summary_key] != stats.get(stats_key):
            raise ValueError(f"cache/normalization provenance mismatch: {summary_key}")
    graph = load_graph(graph_dir, stats["columns"])
    graph_summary = json.loads((graph_dir / "summary.json").read_text(encoding="utf-8"))
    if graph_summary.get("split") != "train" or graph_summary.get("label_filter") != "level2_label == normal":
        raise ValueError("normal reference graph must be built from normal train windows")
    report = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "cache_dir": str(cache_dir.resolve()), "graph_dir": str(graph_dir.resolve()),
        "python": sys.version.split()[0], "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(), "label_names": LABEL_NAMES,
        "feature_columns": stats["columns"], "splits": {},
    }
    case_partitions, pair_partitions = {}, {}
    train_batch = None
    mean = torch.tensor(stats["mean"]).view(1, 26, 1)
    std = torch.tensor(stats["std"]).view(1, 26, 1)
    for split, filename in SPLIT_FILES.items():
        print(f"checking {split}...", flush=True)
        payload = torch.load(cache_dir / filename, map_location="cpu", weights_only=True, mmap=True)
        x, y = payload["x"], payload["y"]
        if x.ndim != 3 or tuple(x.shape[1:]) != (26, 150) or x.dtype != torch.float32:
            raise ValueError(f"invalid input tensor in {split}: {x.shape}, {x.dtype}")
        if y.shape != (len(x),) or y.dtype != torch.long or not len(y):
            raise ValueError(f"invalid label tensor in {split}")
        if ((y < 0) | (y > 2)).any():
            raise ValueError(f"invalid three-class labels in {split}")
        for key in ("case_ids", "domains", "pair_keys", "window_start_rows"):
            if key not in payload or len(payload[key]) != len(y):
                raise ValueError(f"missing or inconsistent {key} in {split}")
        allowed = {"REAL"} if split == "external_test" else {"SIL", "HIL"}
        if not set(payload["domains"]).issubset(allowed):
            raise ValueError(f"unexpected domains in {split}")
        for key, partitions in (("case_ids", case_partitions), ("pair_keys", pair_partitions)):
            for value in set(payload[key]):
                if key == "pair_keys" and value in ("", "nan", "None"):
                    continue
                if value in partitions and partitions[value] != split:
                    raise ValueError(f"{key} leakage between {partitions[value]} and {split}: {value}")
                partitions[value] = split
        for start in range(0, len(x), 1024):
            batch = x[start:start + 1024]
            if not torch.isfinite(batch).all() or not torch.isfinite((batch - mean) / std).all():
                raise ValueError(f"nonfinite raw/normalized values in {split}, window {start}")
        counts = torch.bincount(y, minlength=3).tolist()
        actual = {"windows": len(x), "cases": len(set(payload["case_ids"])),
                  "labels": dict(zip(LABEL_NAMES, counts)),
                  "domains": dict(Counter(payload["domains"]))}
        expected = summary["splits"][split]
        for key in ("windows", "cases", "labels"):
            if actual[key] != expected[key]:
                raise ValueError(f"{split} {key} differs from build summary")
        report["splits"][split] = actual
        if split == "train":
            indices = torch.cat([torch.where(y == label)[0][:4] for label in range(3)])
            if len(indices) != 12:
                raise ValueError("training data must contain all three classes")
            train_batch = ((x[indices].clone() - mean) / std, y[indices].clone())
        print(f"  {actual}", flush=True)
        del payload, x, y
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TopologyGNN(adjacency=graph[0], reference_mean=graph[1], reference_std=graph[2]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    x, y = (tensor.to(device) for tensor in train_batch)
    output = model(x)
    node_score, edge_score = model.auxiliary_scores(output)
    fault = (y != 0).float()
    loss = torch.nn.functional.cross_entropy(output.logits, y) + 0.1 * (
        torch.nn.functional.binary_cross_entropy_with_logits(node_score, fault)
        + torch.nn.functional.binary_cross_entropy_with_logits(edge_score, fault))
    if not torch.isfinite(loss):
        raise ValueError("real-data training batch produced a nonfinite loss")
    loss.backward()
    for name, parameter in model.named_parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            raise ValueError(f"nonfinite gradients: {name}")
    optimizer.step()
    report["real_batch_training"] = {"device": str(device), "samples": len(y),
                                     "logit_shape": list(output.logits.shape), "loss": loss.item(),
                                     "parameters": sum(p.numel() for p in model.parameters())}
    report["metadata_sha256"] = {
        str(path.relative_to(root)): digest(path)
        for root, files in ((cache_dir, ["build_summary.json", "norm_stats.json"]),
                            (graph_dir, ["feature_names.json", "adjacency_fused.npy",
                                         "adjacency_normal_mean.npy", "adjacency_normal_std.npy", "summary.json"]))
        for path in (root / name for name in files)
    }
    report["status"] = "passed"
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads <= 0:
        parser.error("threads must be positive")
    torch.set_num_threads(args.threads)
    report = verify(args.cache_dir, args.graph_dir)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"preflight passed; report: {args.report}", flush=True)


if __name__ == "__main__":
    main()
