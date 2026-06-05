import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from uav_fault.models.cnn.cnn_lstm import CNNLSTMClassifier
from uav_fault.models.cnn.textcnn import TextCNN1d


LABEL_NAMES = ["normal", "motor", "sensor"]


class SequenceMetaDataset(Dataset):
    def __init__(self, cache_dir: str, norm_stats_path: str):
        self.cache_dir = Path(cache_dir)
        self.files = sorted(self.cache_dir.glob("*.pt"))
        with open(norm_stats_path, encoding="utf-8") as f:
            stats = json.load(f)
        self.mean = torch.tensor(stats["mean"], dtype=torch.float32).unsqueeze(1)
        self.std = torch.tensor(stats["std"], dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        sample = torch.load(self.files[idx], weights_only=True)
        x = (sample["x"] - self.mean) / self.std
        y = int(sample["y"])
        return x, y


def build_model(model_name: str, in_channels: int, num_classes: int):
    if model_name == "textcnn":
        return TextCNN1d(
            in_channels=in_channels,
            num_classes=num_classes,
            num_filters=128,
            kernel_sizes=(3, 5, 7, 11, 21),
            dropout=0.3,
        )
    if model_name == "cnn_lstm":
        return CNNLSTMClassifier(
            in_channels=in_channels,
            cnn_channels=(64, 128, 128),
            lstm_hidden=128,
            lstm_layers=2,
            num_classes=num_classes,
            dropout=0.3,
        )
    raise ValueError("AdaBN 目前只支持 textcnn / cnn_lstm")


def make_loader(cache_root: Path, split_names: list[str], batch_size: int, num_workers: int, shuffle: bool = False):
    norm_stats_path = cache_root / "norm_stats.json"
    datasets = []
    for split_name in split_names:
        split_dir = cache_root / split_name
        if split_dir.exists():
            datasets.append(SequenceMetaDataset(str(split_dir), str(norm_stats_path)))
    if not datasets:
        raise RuntimeError(f"未找到 split: {split_names}")
    dataset = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=torch.cuda.is_available(),
    )


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            preds = model(x).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    labels = sorted(set(all_labels.tolist()) | set(all_preds.tolist()))
    names = [LABEL_NAMES[i] for i in labels]
    return {
        "accuracy": accuracy_score(all_labels, all_preds),
        "balanced_accuracy": balanced_accuracy_score(all_labels, all_preds),
        "macro_f1": f1_score(all_labels, all_preds, average="macro", zero_division=0),
        "weighted_f1": f1_score(all_labels, all_preds, average="weighted", zero_division=0),
        "labels": labels,
        "label_names": names,
        "confusion_matrix": confusion_matrix(all_labels, all_preds, labels=labels).tolist(),
        "classification_report": classification_report(
            all_labels,
            all_preds,
            labels=labels,
            target_names=names,
            output_dict=True,
            zero_division=0,
        ),
    }


def collect_bn_modules(model: nn.Module):
    return [m for m in model.modules() if isinstance(m, nn.BatchNorm1d)]


def adapt_bn_stats(model, loader, device, max_batches: int | None = None):
    bn_modules = collect_bn_modules(model)
    if not bn_modules:
        raise RuntimeError("当前模型没有 BatchNorm1d，不能做 AdaBN")

    model.train()
    for m in model.modules():
        if isinstance(m, nn.BatchNorm1d):
            m.reset_running_stats()
            m.momentum = None

    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(loader):
            x = x.to(device)
            _ = model(x)
            if max_batches is not None and batch_idx + 1 >= max_batches:
                break


def print_metrics(metrics, split_name):
    print(f"\n{'=' * 52}")
    print(f"  {split_name}")
    print(f"{'=' * 52}")
    print(f"  Accuracy:          {metrics['accuracy']:.4f}")
    print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"  Macro F1:          {metrics['macro_f1']:.4f}")
    print(f"  Weighted F1:       {metrics['weighted_f1']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="transfer3 AdaBN 适配与评估")
    parser.add_argument("--model", choices=["textcnn", "cnn_lstm"], required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--adapt-splits", nargs="+", default=["real_adapt_train", "real_adapt_val"])
    parser.add_argument("--eval-split", nargs="+", default=["real_holdout_test"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-adapt-batches", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_root = Path(args.cache_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(cache_root / "norm_stats.json", encoding="utf-8") as f:
        stats = json.load(f)
    in_channels = len(stats["columns"])

    model = build_model(args.model, in_channels, len(LABEL_NAMES)).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state)

    adapt_loader = make_loader(cache_root, args.adapt_splits, args.batch_size, args.num_workers, shuffle=False)
    eval_loader = make_loader(cache_root, args.eval_split, args.batch_size, args.num_workers, shuffle=False)

    before_metrics = evaluate(model, eval_loader, device)
    print_metrics(before_metrics, "Before AdaBN")

    adapt_bn_stats(
        model,
        adapt_loader,
        device,
        max_batches=args.max_adapt_batches if args.max_adapt_batches > 0 else None,
    )

    after_metrics = evaluate(model, eval_loader, device)
    print_metrics(after_metrics, "After AdaBN")

    torch.save(model.state_dict(), output_dir / "best_adabn.pt")
    (output_dir / "metrics_before_adabn.json").write_text(
        json.dumps(before_metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "metrics_after_adabn.json").write_text(
        json.dumps(after_metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "config.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "cache_dir": str(cache_root),
                "checkpoint": args.checkpoint,
                "adapt_splits": args.adapt_splits,
                "eval_split": args.eval_split,
                "batch_size": args.batch_size,
                "max_adapt_batches": args.max_adapt_batches,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
