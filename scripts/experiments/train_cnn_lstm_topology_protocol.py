"""Train the CNN-LSTM baseline on the official topology three-class protocol."""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset

from uav_fault.models.cnn.cnn_lstm_v2_normalproto import CNNLSTMv2NormalProtoClassifier
from uav_fault.models.cnn.focal_loss import FocalLoss

LABEL_NAMES = ["normal", "motor", "sensor"]
SPLITS = {"train": "train.pt", "val": "val.pt", "test": "test.pt", "external_test": "real_holdout_test.pt"}


class WindowDataset(Dataset):
    def __init__(self, path: Path, stats: dict[str, Any]):
        payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        x, y = payload["x"].float(), payload["y"].long()
        if x.ndim != 3 or x.shape[1] != len(stats["columns"]):
            raise ValueError(f"invalid x shape in {path}: {tuple(x.shape)}")
        mean = torch.tensor(stats["mean"], dtype=torch.float32).view(1, -1, 1)
        std = torch.tensor(stats["std"], dtype=torch.float32).view(1, -1, 1)
        self.x = (x - mean) / std.clamp_min(1e-8)
        self.y = y
        self.domains = list(payload.get("domains", [""] * len(y)))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def report(labels: np.ndarray, preds: np.ndarray) -> dict[str, Any]:
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, labels=[0, 1, 2], average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(labels, preds, labels=[0, 1, 2], average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(labels, preds, labels=[0, 1, 2]).tolist(),
        "classification_report": classification_report(labels, preds, labels=[0, 1, 2], target_names=LABEL_NAMES, output_dict=True, zero_division=0),
        "samples": int(len(labels)),
    }


def evaluate(model, loader, device, dataset=None):
    model.eval()
    labels, preds = [], []
    with torch.no_grad():
        for x, y in loader:
            labels.append(y.numpy())
            preds.append(model(x.to(device)).argmax(dim=1).cpu().numpy())
    y, p = np.concatenate(labels), np.concatenate(preds)
    out = report(y, p)
    if dataset is not None:
        domains = np.asarray(dataset.domains)
        out["by_domain"] = {str(d): report(y[domains == d], p[domains == d]) for d in sorted(set(domains.tolist())) if d}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--min-epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stats = json.loads((args.cache_dir / "norm_stats.json").read_text(encoding="utf-8"))
    datasets = {k: WindowDataset(args.cache_dir / f, stats) for k, f in SPLITS.items()}
    loaders = {
        "train": DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, persistent_workers=args.num_workers > 0, pin_memory=torch.cuda.is_available()),
        **{k: DataLoader(datasets[k], batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, persistent_workers=args.num_workers > 0, pin_memory=torch.cuda.is_available()) for k in ("val", "test", "external_test")},
    }
    model = CNNLSTMv2NormalProtoClassifier(in_channels=len(stats["columns"]), num_classes=3, target_steps=150, raw_proj_dim=32, branch_channels=32, cnn_out_channels=128, lstm_hidden=128, lstm_layers=2, dropout=0.3).to(device)
    criterion = FocalLoss(gamma=2.0).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    best_state, best_f1, best_epoch, stale = None, -math.inf, 0, 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for x, y in loaders["train"]:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            features = model.forward_features(x)
            logits = model.classify_features(features)
            normal = y == 0
            pull = ((features[normal] - model.normal_center) ** 2).mean() if normal.any() else torch.tensor(0.0, device=device)
            fault = ~normal
            distances = torch.linalg.vector_norm(features[fault] - model.normal_center, dim=1) if fault.any() else torch.empty(0, device=device)
            push = F.relu(1.0 - distances).square().mean() if fault.any() else torch.tensor(0.0, device=device)
            loss = criterion(logits, y) + 0.1 * (pull + push)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach()) * len(y)
        val = evaluate(model, loaders["val"], device)
        history.append({"epoch": epoch, "loss": total / len(datasets["train"]), "val_macro_f1": val["macro_f1"], "val_accuracy": val["accuracy"]})
        if epoch == 1 or epoch % 5 == 0:
            print(f"epoch {epoch:3d}/{args.epochs} loss={history[-1]['loss']:.4f} val_f1={val['macro_f1']:.4f}", flush=True)
        if val["macro_f1"] > best_f1:
            best_f1, best_epoch, best_state, stale = val["macro_f1"], epoch, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
        if epoch >= args.min_epochs and stale >= args.patience:
            print(f"early stopping at epoch {epoch} (best val_f1={best_f1:.4f})", flush=True)
            break
    model.load_state_dict(best_state)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test", "external_test"):
        result = evaluate(model, loaders[split], device, datasets[split])
        (args.output_dir / f"metrics_{split}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"{split}: macro_f1={result['macro_f1']:.4f} accuracy={result['accuracy']:.4f}", flush=True)
    torch.save({"model_state": model.state_dict(), "config": {"model": "cnn_lstm_v2_normalproto", "target_steps": 150, "best_epoch": best_epoch, "best_val_f1": best_f1, "seed": args.seed}}, args.output_dir / "best.pt")
    (args.output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"saved outputs to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
