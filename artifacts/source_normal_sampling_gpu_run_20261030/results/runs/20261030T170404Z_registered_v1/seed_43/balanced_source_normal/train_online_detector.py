"""Train online binary fault detectors on stacked window caches.

Requires a new, explicit output directory. Each completed epoch saves history and
configuration; validation improvements atomically save a raw best.pt state_dict.
Input/source hashes are recorded before training. These are recovery artifacts,
not a full optimizer/RNG checkpoint for exact training resumption.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import shutil
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from uav_fault.models.cnn.focal_loss import FocalLoss
from uav_fault.models.cnn.lstm import (
    RFLY_SEQ_SCHEMAS,
    OnlineBiLSTM,
    OnlineCNN1D,
    OnlineKGMSTCA,
    OnlineMSTCA,
    OnlineTransformer,
)

LABEL_NAMES = ["normal", "fault"]

MODEL_REGISTRY = {
    "bilstm": OnlineBiLSTM,
    "cnn1d": OnlineCNN1D,
    "transformer": OnlineTransformer,
    "mstca": OnlineMSTCA,
    "kg_mstca": OnlineKGMSTCA,
}


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _confusion_matrix_binary(labels: np.ndarray, preds: np.ndarray) -> np.ndarray:
    cm = np.zeros((2, 2), dtype=np.int64)
    for true_label, pred_label in zip(labels.tolist(), preds.tolist()):
        cm[int(true_label), int(pred_label)] += 1
    return cm


def _classification_report_binary(labels: np.ndarray, preds: np.ndarray) -> dict:
    cm = _confusion_matrix_binary(labels, preds)
    support = cm.sum(axis=1)
    pred_support = cm.sum(axis=0)

    per_class = {}
    f1s = []
    weighted_f1_sum = 0.0
    total = int(support.sum())
    for class_id, class_name in enumerate(LABEL_NAMES):
        tp = float(cm[class_id, class_id])
        fp = float(pred_support[class_id] - cm[class_id, class_id])
        fn = float(support[class_id] - cm[class_id, class_id])
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2.0 * precision * recall, precision + recall)
        cls_support = float(support[class_id])
        per_class[class_name] = {
            "precision": precision,
            "recall": recall,
            "f1-score": f1,
            "support": cls_support,
        }
        f1s.append(f1)
        weighted_f1_sum += f1 * cls_support

    accuracy = _safe_div(float(np.trace(cm)), float(total))
    macro_precision = float(np.mean([per_class[name]["precision"] for name in LABEL_NAMES]))
    macro_recall = float(np.mean([per_class[name]["recall"] for name in LABEL_NAMES]))
    macro_f1 = float(np.mean(f1s))
    weighted_f1 = _safe_div(weighted_f1_sum, float(total))

    per_class["accuracy"] = accuracy
    per_class["macro avg"] = {
        "precision": macro_precision,
        "recall": macro_recall,
        "f1-score": macro_f1,
        "support": float(total),
    }
    per_class["weighted avg"] = {
        "precision": _safe_div(
            sum(per_class[name]["precision"] * per_class[name]["support"] for name in LABEL_NAMES),
            float(total),
        ),
        "recall": _safe_div(
            sum(per_class[name]["recall"] * per_class[name]["support"] for name in LABEL_NAMES),
            float(total),
        ),
        "f1-score": weighted_f1,
        "support": float(total),
    }
    return per_class


def validate_cache_contract(cache_dir: Path) -> dict:
    norm_stats_path = cache_dir / "norm_stats.json"
    with open(norm_stats_path, encoding="utf-8") as f:
        stats = json.load(f)

    columns = stats.get("columns")
    mean = stats.get("mean")
    std = stats.get("std")
    if not isinstance(columns, list) or not isinstance(mean, list) or not isinstance(std, list):
        raise ValueError("norm_stats.json missing columns/mean/std lists")
    if not (len(columns) == len(mean) == len(std)):
        raise ValueError("norm_stats.json columns/mean/std lengths do not match")

    expected_cfg = {
        "target_hz": 50.0,
        "window_size": 150,
    }
    for key, expected in expected_cfg.items():
        actual = stats.get(key)
        if actual != expected:
            raise ValueError(
                f"norm_stats.json contract mismatch: {key}={actual!r}, expected {expected!r}"
            )

    # stride is sampling provenance, not a data invariant: no consumer reads it, and a
    # cache derived from a differently-strided source is perfectly valid. Require that
    # it is declared and sane rather than pinning one magic value.
    stride = stats.get("stride")
    if not isinstance(stride, int) or isinstance(stride, bool) or stride <= 0:
        raise ValueError(f"norm_stats.json invalid stride={stride!r}, expected a positive int")

    # The declared feature_schema names the exact channel list, and the columns must
    # reproduce it in order. This replaces a plain length check, so a cache built on
    # the wrong schema fails here instead of silently mis-indexing channels at runtime.
    schema_name = stats.get("feature_schema")
    schema_columns = RFLY_SEQ_SCHEMAS.get(schema_name)
    if schema_columns is None:
        raise ValueError(
            f"norm_stats.json unknown feature_schema {schema_name!r}; "
            f"expected one of {sorted(RFLY_SEQ_SCHEMAS)}"
        )
    if list(columns) != list(schema_columns):
        raise ValueError(
            f"norm_stats.json columns do not match feature_schema {schema_name!r} "
            f"({len(columns)} columns given, expected {len(schema_columns)})"
        )

    for split_name in ("train", "val", "test"):
        pt_path = cache_dir / f"{split_name}.pt"
        data = torch.load(pt_path, weights_only=True)
        if "x" not in data or "y" not in data:
            raise ValueError(f"{pt_path} missing x/y")
        x = data["x"]
        y = data["y"]
        if x.ndim != 3:
            raise ValueError(f"{pt_path} x ndim={x.ndim}, expected 3")
        if y.ndim != 1:
            raise ValueError(f"{pt_path} y ndim={y.ndim}, expected 1")
        if x.shape[0] != y.shape[0]:
            raise ValueError(f"{pt_path} x/y sample counts do not match")
        if x.shape[1] != len(columns):
            raise ValueError(f"{pt_path} channels={x.shape[1]}, expected {len(columns)}")
        labels = set(torch.unique(y).cpu().tolist())
        if not labels.issubset({0, 1}):
            raise ValueError(f"{pt_path} has invalid labels: {sorted(labels)}")

    return stats


class OnlineWindowDataset(Dataset):
    def __init__(
        self,
        pt_path: str,
        norm_stats_path: str,
        augment: bool = False,
        max_samples: int = 0,
    ):
        data = torch.load(pt_path, weights_only=True)
        x = data["x"].float()
        y = data["y"].long()

        if max_samples > 0 and max_samples < len(y):
            rng = np.random.RandomState(42)
            idx0 = np.where(y.numpy() == 0)[0]
            idx1 = np.where(y.numpy() == 1)[0]
            fault_ratio = len(idx1) / len(y)
            n1 = int(max_samples * fault_ratio)
            n0 = max_samples - n1
            sel0 = rng.choice(idx0, size=min(n0, len(idx0)), replace=False)
            sel1 = rng.choice(idx1, size=min(n1, len(idx1)), replace=False)
            sel = np.sort(np.concatenate([sel0, sel1]))
            x = x[sel]
            y = y[sel]

        with open(norm_stats_path, encoding="utf-8") as f:
            stats = json.load(f)
        mean = torch.tensor(stats["mean"], dtype=torch.float32).view(1, -1, 1)
        std = torch.tensor(stats["std"], dtype=torch.float32).view(1, -1, 1)
        x.sub_(mean).div_(std)

        self.x = x
        self.y = y
        self.augment = augment
        self.labels = self.y.tolist()

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        x = self.x[idx]
        y = self.y[idx]
        if self.augment:
            x = self._augment(x.clone())
        return x, y

    @staticmethod
    def _augment(x: torch.Tensor) -> torch.Tensor:
        if np.random.random() < 0.35:
            x = x * (1.0 + np.random.uniform(-0.03, 0.03))
        if np.random.random() < 0.35:
            x = x + torch.randn_like(x) * 0.01
        return x


def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(y.numpy().tolist())

    preds_np = np.array(all_preds, dtype=np.int64)
    labels_np = np.array(all_labels, dtype=np.int64)
    report = _classification_report_binary(labels_np, preds_np)
    cm = _confusion_matrix_binary(labels_np, preds_np)
    return {
        "accuracy": report["accuracy"],
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "precision": report["fault"]["precision"],
        "recall": report["fault"]["recall"],
        "f1_fault": report["fault"]["f1-score"],
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }


def print_metrics(metrics: dict, split_name: str) -> None:
    print(f"\n{'=' * 50}")
    print(f"  {split_name}")
    print(f"{'=' * 50}")
    print(f"  Accuracy:     {metrics['accuracy']:.4f}")
    print(f"  Macro F1:     {metrics['macro_f1']:.4f}")
    print(f"  Precision:    {metrics['precision']:.4f}")
    print(f"  Recall:       {metrics['recall']:.4f}")
    print(f"  F1 (fault):   {metrics['f1_fault']:.4f}")


def get_warmup_cosine_lr(
    epoch: int,
    warmup_epochs: int,
    total_epochs: int,
    base_lr: float,
    min_lr: float = 1e-6,
) -> float:
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / warmup_epochs
    progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


def plot_results(output_dir: Path, metrics_test: dict, train_losses: list[float], val_f1s: list[float]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    cm = np.array(metrics_test["confusion_matrix"])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(train_losses)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss")

    axes[1].plot(val_f1s)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Macro F1")
    axes[1].set_title("Validation Macro F1")

    axes[2].imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    axes[2].set_title("Test Confusion Matrix")
    axes[2].set_xticks(range(len(LABEL_NAMES)))
    axes[2].set_yticks(range(len(LABEL_NAMES)))
    axes[2].set_xticklabels(LABEL_NAMES)
    axes[2].set_yticklabels(LABEL_NAMES)
    for i in range(len(LABEL_NAMES)):
        for j in range(len(LABEL_NAMES)):
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            axes[2].text(
                j,
                i,
                f"{cm[i, j]}\n({cm_norm[i, j]:.1%})",
                ha="center",
                va="center",
                color=color,
                fontsize=9,
            )

    plt.tight_layout()
    plt.savefig(output_dir / "results.png", dpi=150)
    plt.close(fig)


def _sha256(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def _write_json(path: Path, value) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def train(args) -> None:
    if not args.output_dir or not str(args.output_dir).strip():
        raise ValueError("output_dir must explicitly name a new experiment directory")
    output_dir = Path(args.output_dir)
    # Exclusive creation also prevents two processes from claiming the same run.
    output_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    cache_dir = Path(args.cache_dir)

    norm_stats_path = cache_dir / "norm_stats.json"
    stats = validate_cache_contract(cache_dir)
    print("loading datasets...")

    train_ds = OnlineWindowDataset(
        str(cache_dir / "train.pt"),
        str(norm_stats_path),
        augment=True,
        max_samples=args.max_train_samples,
    )
    val_ds = OnlineWindowDataset(str(cache_dir / "val.pt"), str(norm_stats_path), augment=False)
    test_ds = OnlineWindowDataset(str(cache_dir / "test.pt"), str(norm_stats_path), augment=False)
    external_test_path = None
    for candidate in ("external_test.pt", "real_holdout_test.pt"):
        candidate_path = cache_dir / candidate
        if candidate_path.exists():
            external_test_path = candidate_path
            break
    external_test_ds = (
        OnlineWindowDataset(str(external_test_path), str(norm_stats_path), augment=False)
        if external_test_path is not None
        else None
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    external_test_loader = (
        DataLoader(
            external_test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        if external_test_ds is not None
        else None
    )

    in_channels = len(stats["columns"])
    model_cls = MODEL_REGISTRY[args.model]
    model = model_cls(
        in_channels=in_channels,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_classes=2,
        dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"model={args.model} params={n_params:,}")

    criterion = FocalLoss(gamma=args.gamma).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    config = {
        "model": args.model,
        "mode": getattr(args, "mode", ""),
        "cache_dir": str(cache_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "task": "online_binary",
        "protocol_name": stats.get("protocol_name"),
        "label_space": stats.get("label_space"),
        "source_fault_classes": stats.get("source_fault_classes", []),
        "feature_schema": stats.get("feature_schema", "unknown"),
        "window_size": int(stats.get("window_size") or train_ds.x.shape[-1]),
        "feature_count": in_channels,
        "in_channels": in_channels,
        "num_classes": 2,
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "label_names": LABEL_NAMES,
        "epochs": args.epochs,
        "epochs_trained": 0,
        "best_epoch": None,
        "best_val_f1": None,
        "gamma": args.gamma,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "weight_decay": args.weight_decay,
        "warmup": args.warmup,
        "patience": args.patience,
        "min_epochs": args.min_epochs,
        "min_delta": args.min_delta,
        "max_train_samples": args.max_train_samples,
        "subsample_seed": 42,
        "num_workers": args.num_workers,
        "n_params": n_params,
    }
    cache_files = [norm_stats_path, *(cache_dir / f"{s}.pt" for s in ("train", "val", "test"))]
    if external_test_path is not None:
        cache_files.append(external_test_path)
    if (cache_dir / "build_summary.json").is_file():
        cache_files.append(cache_dir / "build_summary.json")
    print("hashing input files for experiment provenance...", flush=True)
    _write_json(output_dir / "provenance.json", {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "process_argv": list(sys.argv),
        "working_directory": str(Path.cwd()),
        "python": sys.version,
        "torch": str(torch.__version__),
        "numpy": np.__version__,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "cuda_version": torch.version.cuda,
        "cache_files": {
            path.name: {"sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in cache_files
        },
        "source_sha256": {
            "trainer": _sha256(Path(__file__)),
            "model": _sha256(Path(sys.modules[model_cls.__module__].__file__)),
            "loss": _sha256(Path(sys.modules[FocalLoss.__module__].__file__)),
        },
    })
    shutil.copy2(norm_stats_path, output_dir / "norm_stats.json")
    _write_json(output_dir / "config.json", config)
    best_val_f1 = -math.inf
    best_state = None
    patience_counter = 0
    train_losses: list[float] = []
    val_f1s: list[float] = []
    history: list[dict] = []

    for epoch in range(args.epochs):
        lr = get_warmup_cosine_lr(epoch, args.warmup, args.epochs, args.lr)
        for group in optimizer.param_groups:
            group["lr"] = lr

        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += float(loss.item())
            n_batches += 1

        avg_loss = epoch_loss / max(1, n_batches)
        train_losses.append(avg_loss)

        val_metrics = evaluate(model, val_loader, device)
        val_f1 = float(val_metrics["macro_f1"])
        val_f1s.append(val_f1)
        ep = epoch + 1
        if ep % 5 == 0 or ep <= 5:
            print(
                f"epoch {ep:3d}/{args.epochs} "
                f"loss={avg_loss:.4f} val_f1={val_f1:.4f} "
                f"val_recall={val_metrics['recall']:.4f} lr={lr:.2e}"
            )

        if not math.isfinite(avg_loss) or not math.isfinite(val_f1):
            raise FloatingPointError(f"nonfinite training loss or validation F1 at epoch {ep}")
        if val_f1 > best_val_f1 + args.min_delta:
            best_val_f1 = val_f1
            best_state = copy.deepcopy(model.state_dict())
            # Keep the existing raw state_dict format for inference consumers.
            temporary = output_dir / "best.pt.tmp"
            torch.save(best_state, temporary)
            temporary.replace(output_dir / "best.pt")
            config["best_epoch"] = ep
            config["best_val_f1"] = best_val_f1
            patience_counter = 0
        elif ep >= args.min_epochs:
            patience_counter += 1
        history.append({
            "epoch": ep, "train_loss": avg_loss, "val_macro_f1": val_f1,
            "val_recall": float(val_metrics["recall"]), "lr": lr,
        })
        config["epochs_trained"] = ep
        _write_json(output_dir / "history.json", history)
        _write_json(output_dir / "config.json", config)
        if ep >= args.min_epochs and patience_counter >= args.patience:
            print(f"early stopping at epoch {ep} (best val_f1={best_val_f1:.4f})")
            break

    if best_state is None:
        raise RuntimeError("no best model was recorded")
    model.load_state_dict(best_state)

    val_metrics = evaluate(model, val_loader, device)
    test_metrics = evaluate(model, test_loader, device)
    external_test_metrics = (
        evaluate(model, external_test_loader, device)
        if external_test_loader is not None
        else None
    )
    print_metrics(val_metrics, "Validation (best)")
    print_metrics(test_metrics, "Test")
    if external_test_metrics is not None:
        print_metrics(external_test_metrics, "External Test")

    _write_json(output_dir / "metrics_val.json", val_metrics)
    _write_json(output_dir / "metrics_test.json", test_metrics)
    if external_test_metrics is not None:
        _write_json(output_dir / "metrics_external_test.json", external_test_metrics)
    plot_results(output_dir, test_metrics, train_losses, val_f1s)
    print(f"saved outputs to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train online binary fault detector")
    parser.add_argument(
        "--model",
        default="bilstm",
        choices=sorted(MODEL_REGISTRY.keys()),
        help="Model architecture name.",
    )
    parser.add_argument("--cache-dir", default="data/processed/online_cache_rfly")
    parser.add_argument(
        "--mode",
        default="",
        help="Optional protocol tag such as baseline/fewshot for experiment bookkeeping.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="New experiment directory; existing paths are rejected to protect previous runs.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min-epochs", type=int, default=30)
    parser.add_argument("--min-delta", type=float, default=0.002)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=0,
        help="Optional stratified cap on train windows. 0 means no cap.",
    )
    cli_args = parser.parse_args()

    torch.manual_seed(cli_args.seed)
    np.random.seed(cli_args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cli_args.seed)

    train(cli_args)
